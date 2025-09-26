from __future__ import annotations

from typing import TYPE_CHECKING, List, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from catalax.model.ode import ODE
from catalax.model.reaction import Reaction
from catalax.model.stoich import derive_stoich_matrix

from .symbolicmodule import SymbolicModule

if TYPE_CHECKING:
    from catalax.tools.simulation import EQUATION_TYPES, SimulationInput


class BaseStack(eqx.Module):
    """
    A base stack for evaluating systems of equations.

    This abstract base class provides common functionality for stacks that evaluate
    collections of symbolic equations. It handles the assembly of SymbolicModule
    instances and provides methods for batch evaluation of multiple equations.

    Attributes:
        modules: Sequence of SymbolicModule instances, one for each equation
    """

    modules: List[eqx.Module]

    def _assemble_modules(
        self,
        sim_input: SimulationInput,
        type: EQUATION_TYPES,
    ):
        """
        Assemble SymbolicModule instances from equations.

        Creates a SymbolicModule for each equation with the specified positional
        arguments mapping. This method is used internally during initialization
        to convert symbolic equations into callable modules.

        Args:
            sim_input: Simulation input containing the equations and variable definitions
            type: Type of equations to assemble ("ode" or "reaction")
        """
        positionals = {
            "parameters": sim_input.parameters,
            "constants": sim_input.constants,
            "states": sim_input.states,
        }

        self.modules = [
            SymbolicModule(equation.equation, positionals=positionals)
            for equation in sim_input.get_equations(type)
        ]

    def _evaluate_modules(
        self,
        t: jax.Array,
        y: jax.Array,
        parameters: jax.Array,
        constants: jax.Array,
    ):
        """
        Evaluate all modules in the stack.

        Calls each SymbolicModule with the provided arguments and stacks the
        results into a single array. This method provides efficient batch
        evaluation of multiple equations.

        Args:
            t: Current time value
            y: Current state vector
            parameters: Parameter values array
            constants: Constant values array

        Returns:
            jax.Array: Stacked results from all module evaluations
        """
        return jnp.stack(
            [
                module(  # type: ignore
                    t=t,
                    parameters=parameters,
                    states=y,
                    constants=constants,
                )
                for module in self.modules
            ],
            axis=-1,
        )

    def __call__(self, t, y, args):
        """
        Abstract method for evaluating the stack.

        This method must be implemented by subclasses to define how the stack
        should be evaluated for specific use cases (ODEs, reactions, etc.).

        Args:
            t: Current time
            y: Current state vector
            args: Additional arguments (typically parameters and constants)

        Raises:
            NotImplementedError: Always, as this is an abstract base class
        """
        raise NotImplementedError(
            "This is the base stack, please use a subclass (ODEStack, ReactionStack, MixedStack)"
        )


class ODEStack(BaseStack):
    """
    A stack of symbolic modules for evaluating systems of ODEs.

    This class creates a collection of SymbolicModule instances from a set of ODE
    equations and provides a unified interface for evaluating all equations
    simultaneously. Used primarily for ODE system evaluation in diffrax solvers.

    The stack directly returns the evaluated rates from each ODE equation,
    making it suitable for systems where each equation directly specifies
    the rate of change for a corresponding state variable.

    Attributes:
        modules: Sequence of SymbolicModule instances, one for each ODE equation
    """

    def __init__(
        self,
        sim_input: SimulationInput,
    ):
        """
        Initialize the ODEStack with ODE equations and variable definitions.

        Args:
            sim_input: Simulation input containing ODE equations, parameters, states, and constants
        """

        # Assemble the modules
        self._assemble_modules(
            sim_input=sim_input,
            type="ode",
        )

    def __call__(self, t, y, args):
        """
        Evaluate all ODE equations in the stack at given time and state.

        Args:
            t: Current time
            y: Current state vector
            args: Tuple containing (parameters, constants)

        Returns:
            jax.Array: Stack of evaluated rates for all ODE equations
        """
        parameters, constants = args
        return self._evaluate_modules(t, y, parameters, constants)

    def fill_zero_modules(self, used_states: List[str], all_states: List[str]) -> Self:
        """
        Fill the modules with zero rates for states that are not used.

        This method creates a new ODEStack where unused states have zero rate modules,
        ensuring the output vector has the correct dimensionality for all states.

        Args:
            used_states: List of state names that have corresponding ODE equations
            all_states: Complete list of all state names in the system

        Returns:
            Self: New ODEStack instance with zero modules for unused states
        """
        replacement = []
        for state in all_states:
            if state not in used_states:
                replacement.append(Zero())
            else:
                replacement.append(self.modules[used_states.index(state)])

        return eqx.tree_at(
            where=lambda tree: tree.modules,
            pytree=self,
            replace=replacement,
        )


class ReactionStack(BaseStack):
    """
    A stack of symbolic modules for evaluating systems of reactions.

    This class handles reaction-based systems where individual reaction rates
    are computed and then combined using a stoichiometry matrix to determine
    the net rate of change for each state. The stoichiometry matrix encodes
    how each reaction affects each state in the system.

    Attributes:
        modules: Sequence of SymbolicModule instances, one for each reaction
        stoich_mat: Stoichiometry matrix relating reactions to state changes
    """

    stoich_mat: jax.Array

    def __init__(self, sim_input: SimulationInput):
        """
        Initialize the ReactionStack with reactions and variable definitions.

        Args:
            sim_input: Simulation input containing reaction equations, parameters, states, and constants
        """
        self._assemble_modules(
            sim_input=sim_input,
            type="reaction",
        )
        self.stoich_mat = derive_stoich_matrix(sim_input.reactions, sim_input.states)

    def __call__(self, t, y, args):
        """
        Evaluate reaction rates and apply stoichiometry matrix.

        Computes individual reaction rates and multiplies by the stoichiometry
        matrix to get the net rate of change for each state.

        Args:
            t: Current time
            y: Current state vector
            args: Tuple containing (parameters, constants)

        Returns:
            jax.Array: Net rates of change for each state
        """
        parameters, constants = args
        rates = self._evaluate_modules(t, y, parameters, constants)
        return self.stoich_mat @ rates

    def fill_zero_modules(self, used_states: List[str], all_states: List[str]) -> Self:
        """
        Adjust stoichiometry matrix to include unused states with zero contributions.

        This method creates a new ReactionStack where the stoichiometry matrix is expanded
        to include all states in the system, with zero contributions for unused states.

        Args:
            used_states: List of state names that participate in reactions
            all_states: Complete list of all state names in the system

        Returns:
            Self: New ReactionStack instance with expanded stoichiometry matrix
        """

        # Iterate through the rows of the stoich matrix
        _, n_reactions = self.stoich_mat.shape
        new_stoich_mat = jnp.zeros((len(all_states), n_reactions))

        for i, used_state in enumerate(used_states):
            new_stoich_index = all_states.index(used_state)
            new_stoich_mat = new_stoich_mat.at[new_stoich_index, :].set(
                self.stoich_mat[i, :]
            )

        return eqx.tree_at(
            where=lambda tree: tree.stoich_mat,
            pytree=self,
            replace=new_stoich_mat,
        )


class MixedStack(eqx.Module):
    """
    A mixed stack combining ODE and reaction rate equations.

    This class handles systems that contain both direct ODE equations and
    reaction-based equations. It evaluates both types and combines them
    to produce the final rate vector. This is useful for models where some
    states have direct rate expressions while others are governed by
    reaction kinetics.

    The final rate for each state is the sum of:
    1. Direct ODE contributions
    2. Net reaction contributions from the stoichiometry matrix

    Attributes:
        ode_stack: Stack containing direct ODE equations
        reac_stack: Stack containing reaction rate equations
    """

    ode_stack: ODEStack
    reac_stack: ReactionStack

    def __init__(self, sim_input: SimulationInput):
        """
        Initialize the MixedStack with ODE and reaction components.

        Creates aligned ODE and reaction stacks where unused states are filled
        with zero contributions to ensure consistent dimensionality.

        Args:
            sim_input: Simulation input containing both ODE and reaction equations,
                      along with parameters, states, and constants
        """

        # We need to align the states of the ODE and reaction stacks
        reaction_states = self._get_reaction_states(sim_input.reactions)
        ode_states = self._get_ode_states(sim_input.odes)

        self.ode_stack = ODEStack(sim_input=sim_input).fill_zero_modules(
            used_states=ode_states,
            all_states=sim_input.states,
        )

        self.reac_stack = ReactionStack(sim_input=sim_input).fill_zero_modules(
            used_states=reaction_states,
            all_states=sim_input.states,
        )

    def __call__(self, t, y, args):
        """
        Evaluate the mixed system combining ODE and reaction rates.

        Computes both direct ODE rates and reaction-based rates, then
        sums them to get the total rate of change for each state.

        Args:
            t: Current time
            y: Current state vector
            args: Tuple containing (parameters, constants)

        Returns:
            jax.Array: Combined rate vector from ODE and reaction contributions
        """
        ode_rates = self.ode_stack(t, y, args)
        reaction_rates = self.reac_stack(t, y, args)
        return reaction_rates + ode_rates

    @staticmethod
    def _get_reaction_states(reactions: List[Reaction]) -> List[str]:
        """
        Extract unique state names from reaction definitions.

        Args:
            reactions: List of Reaction objects

        Returns:
            List[str]: Sorted list of unique state names involved in reactions
        """
        states = set()
        for reaction in reactions:
            for reactant in reaction.reactants:
                states.add(reactant.state)
            for product in reaction.products:
                states.add(product.state)
        return sorted(list(states))

    @staticmethod
    def _get_ode_states(odes: List[ODE]) -> List[str]:
        """
        Extract state names from ODE definitions.

        Args:
            odes: List of ODE objects

        Returns:
            List[str]: Sorted list of state names with direct ODE equations
        """
        return sorted([str(ode.state.symbol) for ode in odes])


class Zero(eqx.Module):
    """
    A module that returns zero for any input.

    Used as a placeholder for states that don't have corresponding equations,
    ensuring consistent dimensionality in stack evaluations.
    """

    def __call__(self, **kwargs):
        """
        Return zero regardless of input arguments.

        Args:
            **kwargs: Any keyword arguments (ignored)

        Returns:
            int: Always returns 0
        """
        return 0
