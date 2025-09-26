from __future__ import annotations

import copy
import json
import os
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import arviz as az
import jax
import jax.numpy as jnp
import pyenzyme as pe
import rich
from deprecated import deprecated
from dotted_dict import DottedDict
from jax import Array
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from pyenzyme import EnzymeMLDocument
from sympy import Expr, Matrix, Symbol, symbols, sympify

from ..dataset.dataset import Dataset
from ..mcmc import priors
from ..predictor import Predictor
from ..surrogate import Surrogate
from ..tools.simulation import Simulation, SimulationInput
from ..tools.stack import BaseStack, MixedStack
from .assignment import Assignment, analyze_and_resolve_dependencies
from .base import CatalaxBase
from .constant import Constant
from .enzymeml import from_enzymeml
from .inaxes import InAxes
from .ode import ODE
from .parameter import HDI, Parameter
from .reaction import Reaction, ReactionElement
from .simconfig import SimulationConfig
from .state import State
from .utils import LOCALS, PrettyDict, check_symbol, eqprint, odeprint

Y0_INDEX = 0
PARAMETERS_INDEX = 1
CONSTANTS_INDEX = 2
TIME_INDEX = 3

HDIOptions = Literal["lower", "upper", "lower_50", "upper_50"]


class Model(CatalaxBase, Predictor, Surrogate):
    """
    Model class for storing ODEs, states, and parameters, which is used
    to describe a biological system. Model classes can be passed to analysis
    and optimisation methods, which will utilize the defined model. The workflow
    of using the Model class is as follows:

        (1) Instantiate a Model object
        (2) Add states to the model
        (3) Add ODEs to the model
        (4) Further modify parameters as needed (values, equations, etc.)

    Example usage:

        model = Model(name="model")
        model.add_state("s1, s2, e1, c1") # see 'add_state' method for more details

        # Add ODEs
        model.add_ode("s1", "k1 * s1")
        model.add_ode("s2", "k2 * s2")

        # Add custom equations (equilibrium constants etc.)
        model.add_parameter(name="E_tot", equation="c1 + e1")
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    name: str
    odes: Dict[str, ODE] = Field(default_factory=DottedDict)
    states: Dict[str, State] = Field(default_factory=PrettyDict)
    parameters: Dict[str, Parameter] = Field(default_factory=PrettyDict)
    reactions: Dict[str, Reaction] = Field(default_factory=PrettyDict)
    constants: Dict[str, Constant] = Field(default_factory=PrettyDict)
    assignments: Dict[str, Assignment] = Field(default_factory=PrettyDict)

    _sim_func: Optional[
        Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    ] = PrivateAttr(default=None)
    _stack: Optional[BaseStack | MixedStack] = PrivateAttr(default=None)
    _in_axes: Optional[Tuple] = PrivateAttr(default=None)
    _dt0: Optional[Tuple] = PrivateAttr(default=None)
    _jacobian_parameters: Optional[Callable] = PrivateAttr(default=None)
    _jacobian_states: Optional[Callable] = PrivateAttr(default=None)
    _sensitivity: Optional[InAxes] = PrivateAttr(default=None)

    @property
    def stack(self) -> BaseStack | MixedStack:
        """Get the stack of the model."""
        return self.sim_input.stack

    @property
    def sim_input(self) -> SimulationInput:
        """Get the simulation input of the model."""
        odes = [
            self.odes[state] for state in self.get_state_order() if state in self.odes
        ]
        reactions = [self.reactions[reaction] for reaction in self.get_reaction_order()]
        return SimulationInput(
            odes=odes,
            reactions=reactions,
            states=self.get_state_order(),
            parameters=self.get_parameter_order(),
            constants=self.get_constants_order(),
        )

    @property
    @deprecated("This property is deprecated. Use states instead.")
    def species(self) -> Dict[str, State]:
        """Get the states of the model."""
        return self.states

    def __call__(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Call the model to evaluate the system dynamics.

        This method makes the model callable, allowing it to be used directly
        as a function in ODE solvers. It evaluates the rates of change for
        all states at a given time point.

        Args:
            t: Time point at which to evaluate the dynamics
            y: Current state values as a JAX array
            constants: Optional constant values as a JAX array

        Returns:
            JAX array containing the rates of change for all states
        """
        return self.rates(t, y, constants)

    def add_ode(
        self,
        state: str,
        equation: str,  # type: ignore
        observable: bool = True,
    ):
        """Adds an ODE to the model and converts the equation to a SymPy expression.

        This method will add the new ODE to the model and the model's ODEs attribute,
        which can be accessed by object dot-notation. For example, if the ODE is set
        for state 's1' and the model is named 'model', the ODE can be accessed by:

        model = Model(name="model")
        model.add_ode(state="s1", equation="k1*s1")
        model.odes.s1 -> ODE(equation=k1*s1, state=s1)

        Parameters will be inferred from the equation and added to the model automatically
        by comparing free symbols in the equation to the model's states. If a symbol is not
        present as a state, it will be added as a parameter. If the symbol is already present
        as a parameter the already defined one will be referenced instead of creating a new one.

        If there already exists an ODE for the given state, a ValueError will be raised.
        Due to the theoretical nature of dynamic systems, there can only be one ODE per
        state. If a state hasn't been added to the model, it will be added automatically.

        Args:
            state (str): The state to be modelled within this ODE.
            equation (str): The equation that describes the dynamics of the state.

        Raises:
            ValueError: _description_
        """

        if any(str(ode.state.name) == state for ode in self.odes.values()):
            raise ValueError(
                f"There already exists an ODE for state '{state}'. Please edit the existing ODE instead."
            )

        if isinstance(equation, str):
            equation: Expr = sympify(equation, locals=LOCALS)

        if state not in self.states:
            self.add_state(state)

        self.odes[state] = ODE(
            equation=equation,  # type: ignore
            state=self.states[state],
            observable=observable,
        )

        self.odes[state]._model = self

    def add_odes(self, **odes):
        """Adds multiple ODEs to the model.

        This method will add multiple new ODEs to the model and the model's odes attribute,
        which can be accessed by object dot-notation. For example, if the ODEs are named 's1' and 's2'
        and the model is named 'model', the ODEs can be accessed by:

        model = Model(name="model")
        model.add_odes(s1="k1*s1", s2="k2*s2")
        """
        for state, equation in odes.items():
            self.add_ode(state, equation)

    def add_reaction(
        self,
        scheme: Optional[str] = None,
        *,
        symbol: str,
        reactants: List[Tuple[float, str]] = [],
        products: List[Tuple[float, str]] = [],
        equation: str | Expr,
        reversible: bool = False,
    ):
        """Adds a reaction to the model and converts the equation to a SymPy expression.

        This method will add a new reaction to the model and the model's reactions attribute,
        which can be accessed by object dot-notation. For example, if the reaction is named 'r1'
        and the model is named 'model', the reaction can be accessed by:

        model = Model(name="model")
        model.add_reaction(symbol="r1", reactants=[(1.0, "A")], products=[(1.0, "B")],
                          equation="k1*A", reversible=False)
        model.reactions.r1 -> Reaction(name=r1, reactants=[...], products=[...])

        Parameters will be inferred from the equation and added to the model automatically
        by comparing free symbols in the equation to the model's states. If a symbol is not
        present as a state, it will be added as a parameter.

        All states referenced in the reactants and products must already exist in the model.
        If any state is not found, a ValueError will be raised.

        Args:
            symbol (str): The unique identifier for this reaction.
            reactants (List[(float, str)]): List of tuples containing (stoichiometry, state_name)
                                          for reactants.
            products (List[(float, str)]): List of tuples containing (stoichiometry, state_name)
                                         for products.
            equation (str): The rate equation that describes the reaction kinetics.
            reversible (bool): Whether the reaction is reversible or not.

        Raises:
            ValueError: If the reaction has no reactants or products, or if any referenced
                       state is not found in the model.
        """

        if scheme and (reactants or products):
            raise ValueError("Cannot specify both scheme and reactants/products")

        if scheme is not None:
            self.reactions[symbol] = Reaction.from_scheme(
                symbol=symbol,
                schema=scheme,
                equation=equation,
                reversible=reversible,
                states=self.get_state_order(modeled=False),
            )
            self.reactions[symbol]._model = self
            return

        if len(reactants) == 0 or len(products) == 0:
            raise ValueError("Reaction must have at least one reactant and one product")

        # Check if all the stoichiometries are positive and not zero
        if any(stoichiometry <= 0 for stoichiometry, _ in reactants + products):
            raise ValueError("Stoichiometries must be positive and not zero")

        undefined_states = []
        for _, state in reactants + products:
            if state not in self.states:
                undefined_states.append(state)
        if undefined_states:
            raise ValueError(f"States {undefined_states} not found in model")

        self.reactions[symbol] = Reaction(
            symbol=symbol,
            reactants=list(map(ReactionElement.from_tuple, reactants)),
            products=list(map(ReactionElement.from_tuple, products)),
            equation=equation,  # type: ignore[arg-type]
            reversible=reversible,
        )
        self.reactions[symbol]._model = self

    def add_assignment(self, symbol: str, equation: str):
        """Adds an assignment to the model and converts the equation to a SymPy expression."""

        if symbol in self.assignments:
            raise ValueError(f"Assignment for symbol '{symbol}' already exists")

        self.assignments[symbol] = Assignment(symbol=symbol, equation=equation)  # type: ignore
        self.assignments[symbol]._model = self

        # Remove all parameters and constants from the model
        if symbol in self.parameters:
            del self.parameters[symbol]
        if symbol in self.constants:
            del self.constants[symbol]
        if symbol in self.states:
            del self.states[symbol]

    def add_assignments(self, **assignments):
        """Adds multiple assignments to the model.

        This method will add multiple new assignments to the model and the model's assignments attribute,
        which can be accessed by object dot-notation. For example, if the assignments are named 'a1' and 'a2'
        and the model is named 'model', the assignments can be accessed by:

        model = Model(name="model")
        model.add_assignments(a1="k1*a1", a2="k2*a2")
        """
        for symbol, equation in assignments.items():
            self.add_assignment(symbol=symbol, equation=equation)

    def add_state(self, state_string: str = "", **state_map) -> List[State]:
        """Adds a single or multiple states to the model, which can later be used in ODEs.

        This method will add new states to the model and the model's states dictionary,
        which can be accessed by object dot-notation. For example, if the state is named 's1'
        and the model is named 'model', the state can be accessed by:

            model = Model(name="model")
            model.add_state("s1")
            model.states.s1 -> State(name="s1", symbol=s1)

        States can be added in three ways. The first is by passing a string of state names that
        are separated by a comma. Please note, symbols will be also be used as names. The second is
        by passing a dictionary of state symbols (key) and names (values) and unpacking them as
        keyword arguments. The third is by passing each as a keyword argument.

        The following are all valid ways to add states:

            (1) model.add_state("s1, s2, s3")
            (2) model.add_state(**{"s1": "state1", "s2": "state2", "s3": "state3"})
            (3) model.add_state(s1="state1", s2="state2", s3="state3")

        If a state already exists, a ValueError will be raised.

        Args:
            state_string (str, optional): String of comma-separated state symbols. Defaults to "".
            **state_map (Dict[str, str]): Dictionary of state symbols (key) and names (values).
        """

        if not all(isinstance(value, str) for value in state_map.values()):
            raise TypeError("State names must be of type str")

        if state_string:
            state_map.update(
                {
                    str(state): str(state)
                    for state in self._split_state_string(state_string)
                }  # type: ignore
            )

        added_states = []

        for symbol, name in state_map.items():
            if not isinstance(name, str):
                raise TypeError("State names must be of type str")

            # Make sure the symbol is valid
            check_symbol(symbol)

            self.states[symbol] = State(name=name, symbol=Symbol(symbol))
            added_states.append(self.states[symbol])

        return added_states

    def add_states(self, **states):
        """Adds multiple states to the model.

        This method will add multiple new states to the model and the model's states attribute,
        which can be accessed by object dot-notation. For example, if the states are named 's1' and 's2'
        and the model is named 'model', the states can be accessed by:

        model = Model(name="model")
        model.add_states(s1="state1", s2="state2")
        """
        for symbol, name in states.items():
            self.add_state(**{symbol: name})

    @deprecated("This method is deprecated. Use add_state instead.")
    def add_species(self, species_string: str = "", **species_map):
        """DEPRECATED: Adds a single or multiple states to the model, which can later be used in ODEs and reactions."""
        self.add_state(species_string, **species_map)

    @staticmethod
    def _split_state_string(state_string: str) -> List[Symbol]:
        """Helper method to split a string of states into a list of states"""

        if len(state_string.split(",")) == 1:
            return [Symbol(state_string)]

        return symbols(state_string)

    def _add_single_state(
        self,
        name: str,
        state_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Helper method to add a single state to a model"""

        if state_map is None:
            self.states[name] = State(name=name, symbol=symbols(name))
            return

        if self._is_symbol(name, state_map):
            symbol = name
            name = state_map[str(symbol)]
        else:
            symbol = self._gather_symbol_from_state_map(name, state_map)

        self.states[str(symbol)] = State(name=name, symbol=Symbol(symbol))

    @staticmethod
    def _is_symbol(name: str, state_map: Dict[str, str]) -> bool:
        """Checks whether the given name is an identifer (key of state_map)"""

        inverse_dict = {v: k for k, v in state_map.items()}

        if str(name) not in state_map and name not in inverse_dict:
            raise ValueError(f"State {name} not found in state map")

        return name in state_map

    @staticmethod
    def _gather_symbol_from_state_map(name: str, state_map: Dict[str, str]) -> str:
        """Converts a name to a sympy symbol"""

        inverse_dict = {v: k for k, v in state_map.items()}

        if name not in inverse_dict:
            raise ValueError(f"State {name} not found in state map")

        return str(inverse_dict[name])

    def add_constant(self, constant_string: str = "", **constants_map):
        """
        Adds a constant or multiple constants to the model.

        This method will add new constants to the model and the model's constants dictionary,
        which can be accessed by object dot-notation. For example, if the constant is named 'c1'
        and the model is named 'model', the constant can be accessed by:

            model = Model(name="model")
            model.add_constant("c1")
            model.constants.c1 -> Constant(name="c1", symbol=c1)

        Constants can be added in two ways. The first is by passing a string of constant names that
        are separated by a comma. The second is by passing a dictionary of constant symbols (key)
        and names (values) and unpacking them as keyword arguments.

        The following are both valid ways to add constants:

            (1) model.add_constant("c1, c2, c3")
            (2) model.add_constant(**{"c1": "constant1", "c2": "constant2", "c3": "constant3"})

        If a constant already exists, a ValueError will be raised.

        Args:
            constant_string (str, optional): String of comma-separated constant symbols. Defaults to "".
            **constants_map (Dict[str, str]): Dictionary of constant symbols (key) and names (values).

        Raises:
            TypeError: If the constant names are not of type str.
        """
        if constant_string:
            constants_map.update(
                {
                    str(constant): str(constant)
                    for constant in self._split_state_string(constant_string)
                }
            )

        for symbol, name in constants_map.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Constants names must be of type str. Got {type(name)}"
                )

            # Make sure the symbol is valid
            check_symbol(symbol)

            self.constants[symbol] = Constant(name=name, symbol=Symbol(symbol))

    def add_constants(self, **constants):
        """Adds multiple constants to the model."""
        for symbol, name in constants.items():
            self.add_constant(**{symbol: name})

    # ! Simulation methods
    def simulate(
        self,
        dataset: Dataset,
        config: SimulationConfig,
        saveat: Union[jax.Array, None] = None,
        parameters: Optional[jax.Array] = None,
        in_axes: InAxes = InAxes.Y0,
        return_array: bool = False,
        use_hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
        hdi_prob: float = 0.95,
    ) -> Dataset:
        """
        Simulate the model with given dataset and parameters.

        Args:
            dataset: Dataset containing initial conditions and simulation data
            config: Optional simulation configuration parameters
            saveat: Optional array of time points to save results at
            parameters: Optional array of model parameters
            in_axes: Axes specification for vectorized operations
            return_array: Whether to return raw arrays instead of Dataset
            use_hdi: Whether to return the lower or upper bound of the HDI. Internally used for MCMC.

        Returns:
            Either a tuple of (time points, results) arrays or a Dataset object
        """

        if config.nsteps is not None and isinstance(saveat, jax.Array):
            raise ValueError("Cannot specify both nsteps and saveat")

        # Initialize simulation data from dataset
        y0, constants = self._initialize_simulation_data(dataset=dataset)
        saveat = self._setup_saveat(config=config, saveat=saveat)
        parameters = self._get_parameters(
            parameters=parameters,
            use_hdi=use_hdi,
            hdi_prob=hdi_prob,
        )

        if saveat.ndim != 1:
            inaxes: Tuple = in_axes + InAxes.TIME  # type: ignore
        else:
            inaxes: Tuple = in_axes.value  # type: ignore

        # Setup and run simulation
        if self._model_changed(inaxes, config.dt0) or self._sim_func is None:
            self._setup_simulation(config, inaxes)

        result = self._run_simulation(y0, parameters, constants, saveat)

        return self._format_simulation_results(
            result,
            saveat,
            return_array,
            y0,
            constants,
        )  # type: ignore

    def predict(
        self,
        dataset: Dataset,
        config: SimulationConfig | None = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
    ) -> Dataset:
        """Predict model behavior using the given dataset.

        This is a convenience wrapper around the `simulate` method that automatically
        creates a simulation configuration if one is not provided.

        Args:
            dataset: Dataset containing initial conditions for prediction
            config: Optional simulation configuration parameters. If None,
                   a configuration will be created from the dataset.
            nsteps: Number of time steps for the simulation. This will override
                   the nsteps value in the provided config if both are specified.
            use_times: Whether to use the time points from the dataset or to simulate at fixed time steps
        Returns:
            A Dataset object containing the prediction results
        """
        if config is None:
            config = dataset.to_config(nsteps=n_steps)
        if use_times:
            config.nsteps = None
            _, times, _ = dataset.to_jax_arrays(self.get_state_order())
        else:
            times = None

        if config.nsteps != n_steps and not use_times:
            config.nsteps = n_steps

        return self.simulate(dataset, config, saveat=times, use_hdi=hdi)

    def rates(
        self,
        t: Array,
        y: Array,
        constants: Optional[Array] = None,
    ) -> Array:
        """Get the rates of the model.

        This method particularly useful to gather rates for quiver plots.

        Args:
            t: Time points
            y: States
            constants: Constants, if any

        Returns:
            Rates
        """
        t, y, constants = self._validate_rate_input(t, y, constants)

        if self._sim_func is None:
            raise RuntimeError("Simulation function not initialized")

        parameters = self._get_parameters()
        return self._sim_func(t, y, parameters, constants)

    def predict_rates(self, dataset: Dataset) -> jax.Array:
        """Predict rates using the given dataset.

        Args:
            dataset: Dataset containing initial conditions for prediction

        Returns:
            A Dataset object containing the prediction results
        """
        data, times, _ = dataset.to_jax_arrays(self.get_state_order())
        constants = dataset.to_y0_matrix(state_order=self.get_constants_order())
        dataset_size, time_size, _ = data.shape
        ins = data.reshape(dataset_size * time_size, -1)
        times = times.ravel()
        return self.rates(times, ins, constants).reshape(dataset_size * time_size, -1)

    def _initialize_simulation_data(
        self,
        dataset: Dataset,
    ) -> Tuple[jax.Array, jax.Array]:
        """Initialize simulation data from dataset."""
        dataset = copy.deepcopy(dataset)
        return (
            dataset.to_y0_matrix(state_order=self.get_state_order()),
            dataset.to_y0_matrix(state_order=self.get_constants_order()),
        )

    def _setup_saveat(
        self,
        config: SimulationConfig,
        saveat: Optional[jax.Array],
    ) -> jax.Array:
        """Setup save points for simulation."""
        if config.nsteps is not None:
            saveat = jnp.linspace(config.t0, config.t1, config.nsteps).T

        if isinstance(saveat, jax.Array):
            return saveat
        else:
            raise ValueError(
                "No saveat or config.nsteps provided. Please provide one of the two."
            )

    def _setup_simulation(
        self,
        config: SimulationConfig,
        in_axes: Tuple,
    ) -> None:
        """Setup the simulation function with given configuration."""
        self._setup_system(
            in_axes=in_axes,
            config=config,
        )

        self._in_axes = in_axes
        self._dt0 = config.dt0  # type: ignore

    def _run_simulation(
        self,
        y0: jax.Array,
        parameters: jax.Array,
        constants: jax.Array,
        saveat: jax.Array,
    ) -> jax.Array:
        """Run the simulation with given parameters."""
        if self._sim_func is None:
            raise RuntimeError("Simulation function not initialized")

        result = self._sim_func(y0, parameters, constants, saveat)
        if len(result.shape) != 3:
            result = jnp.expand_dims(result, axis=0)
        return result

    def _format_simulation_results(
        self,
        result: jax.Array,
        saveat: jax.Array,
        return_array: bool,
        y0: jax.Array,
        constants: jax.Array,
    ) -> Union[tuple[Array, Array], Dataset]:
        """Format simulation results according to return type."""
        if return_array:
            return saveat, result

        dataset = Dataset.from_model(self)
        for y0_i in range(result.shape[0]):
            init_conc = {
                state: float(y0[y0_i, sp_j])
                for sp_j, state in enumerate(self.get_state_order())
            }

            for cons_i, constant in enumerate(self.get_constants_order()):
                init_conc[constant] = float(constants[y0_i, cons_i])

            if saveat.ndim == 1:
                local_saveat = saveat
            else:
                local_saveat = saveat[y0_i, :]

            dataset.add_from_jax_array(
                state_order=self.get_state_order(),
                data=result[y0_i, :, :],
                time=local_saveat,
                initial_condition=init_conc,
            )
        return dataset

    def _model_changed(
        self,
        in_axes: Tuple,
        dt0: float,
    ) -> bool:
        """Check if model configuration has changed."""
        return self._in_axes != in_axes or self._dt0 != dt0

    def _get_stoich_mat(self) -> jax.Array:
        """Creates the stoichiometry matrx based on the given model.

        Models can be defined in both ways - First by using plain ODE models where each state
        has a single assigned equation. Here, the resulting stoichiometry matrx is a unity matrix.
        In the case of a model, that is defined by reactions, the stoichioemtry matrix accordingly
        is constructed per state and reaction.
        """

        if hasattr(self, "reactions"):
            raise NotImplementedError(
                "So far stoichioemtry matrix construction is exclusive to ODE models and not implemented yet."
            )

        stoich_mat = jnp.zeros((len(self.odes), len(self.odes)))
        diag_indices = jnp.diag_indices_from(stoich_mat)

        return stoich_mat.at[diag_indices].set(1)

    def _setup_system(
        self,
        config: SimulationConfig,
        in_axes: Tuple = InAxes.Y0.value,
    ):
        """Converts given SymPy equations into Equinox modules, used for simulation.

        This method will prepare the simulation function, jit it and vmap it across
        the initial condition vector by default.

        Args:
            in_axes: Specifies the axes to map the simulation function across.
                     Defaults to (0, None, None).
        """
        model = self._replace_assignments()
        simulation_setup = Simulation(
            sim_input=model.sim_input,
            config=config,
        )

        self._sim_func, self._stack = simulation_setup._prepare_func(in_axes=in_axes)

    def _replace_assignments(self) -> "Model":
        """Replaces assignment symbols with their values"""
        model = copy.deepcopy(self)

        # First replace assignements in assignments
        resolved_assignments = analyze_and_resolve_dependencies(self)

        for symbol, equation in resolved_assignments.items():
            model.assignments[symbol].equation = equation

        # Then replace assignements in odes
        for ode in model.odes.values():
            for assignment in model.assignments.values():
                # Check if the assignment is used in the ode
                if assignment.symbol in [
                    str(symbol) for symbol in ode.equation.free_symbols
                ]:
                    ode.equation = ode.equation.subs(  # type: ignore
                        assignment.symbol,
                        assignment.equation,  # type: ignore
                    )

        for reaction in model.reactions.values():
            for assignment in model.assignments.values():
                if assignment.symbol in [
                    str(symbol) for symbol in reaction.equation.free_symbols
                ]:
                    reaction.equation = reaction.equation.subs(  # type: ignore
                        assignment.symbol,
                        assignment.equation,  # type: ignore
                    )

        return model

    def _get_parameters(
        self,
        parameters: Optional[jax.Array] = None,
        use_hdi: Optional[HDIOptions] = None,
        hdi_prob: float = 0.95,
    ) -> jax.Array:
        """Gets all the parameters for the model"""

        if parameters is not None:
            return parameters

        missing_parameters = [
            param for param in self.parameters.values() if param.value is None
        ]

        if missing_parameters:
            raise ValueError(
                f"Missing values for parameters: {', '.join([param.name for param in missing_parameters])}"
            )

        # Check if we should use HDI bounds instead of parameter values
        if use_hdi in ["lower", "upper"]:
            # Ensure all parameters have HDI information available
            assert all(
                param.hdi is not None
                and param.hdi.lower is not None
                and param.hdi.upper is not None
                for param in self.parameters.values()
            ), "Lower bound of HDI requested, but no HDI found for parameters"

            # Return either lower or upper HDI bounds for all parameters
            return jnp.array(
                [
                    self.parameters[param].hdi.lower  # type: ignore
                    if use_hdi == "lower"
                    else self.parameters[param].hdi.upper  # type: ignore
                    for param in self.get_parameter_order()
                ]
            )
        elif use_hdi in ["lower_50", "upper_50"]:
            # Return either lower or upper HDI bounds for all parameters
            return jnp.array(
                [
                    self.parameters[param].hdi.lower_50  # type: ignore
                    if use_hdi == "lower_50"
                    else self.parameters[param].hdi.upper_50  # type: ignore
                    for param in self.get_parameter_order()
                ]
            )

        elif use_hdi is not None:
            # Invalid HDI option provided
            raise ValueError(
                "Invalid use_hdi value. Either 'lower' or 'upper' expected."
            )

        return jnp.array(
            [self.parameters[param].value for param in self.get_parameter_order()]
        )  # type: ignore

    def get_parameter_order(self) -> List[str]:
        """Returns the order of the parameters in the model"""
        return sorted(self.parameters.keys())

    @overload
    def get_state_order(
        self,
        modeled: bool = True,
        *,
        as_indices: Literal[True],
    ) -> List[int]:
        """Returns the order of the states in the model as indices"""
        ...

    @overload
    def get_state_order(
        self,
        modeled: bool = True,
        *,
        as_indices: Literal[False] = False,
    ) -> List[str]:
        """Returns the order of the states in the model as strings"""
        ...

    def get_state_order(
        self,
        modeled: bool = True,
        as_indices: bool = False,
    ) -> List[str] | List[int]:
        """Returns the order of the states and constants in the model"""
        if not modeled:
            if as_indices:
                return list(range(len(self.states)))
            else:
                return sorted(list(self.states.keys()))

        ode_species = {str(ode.state.symbol) for ode in self.odes.values()}
        reaction_species = set(self.get_reacting_state_order())
        modeled_species = list(ode_species | reaction_species)
        modeled_species = sorted(modeled_species)

        if as_indices:
            return [i for i, state in enumerate(modeled_species)]
        else:
            return modeled_species

    @deprecated("This method is deprecated. Use get_state_order instead.")
    def get_species_order(self) -> List[str]:
        """Returns the order of the states in the model"""
        return self.get_state_order()

    def get_reaction_order(self) -> List[str]:
        """Returns the order of the reactions in the model"""
        return sorted(list(self.reactions.keys()))

    def get_reacting_state_order(self) -> List[str]:
        """Returns the order of the states that are involved in reactions in the model"""
        return sorted(
            list(
                set(
                    [
                        state.state
                        for reaction in self.reactions.values()
                        for state in reaction.reactants + reaction.products
                    ]
                )
            )
        )

    @overload
    def get_observable_state_order(self) -> List[str]:
        """Returns the order of the observable states in the model as strings"""
        ...

    @overload
    def get_observable_state_order(self, as_indices: Literal[True]) -> List[int]:
        """Returns the order of the observable states in the model as indices"""
        ...

    @overload
    def get_observable_state_order(self, as_indices: Literal[False]) -> List[str]:
        """Returns the order of the observable states in the model as strings"""
        ...

    def get_observable_state_order(self, as_indices: bool = False) -> List[str | int]:  # type: ignore
        """Returns the order of the observable states in the model"""
        observable_states = []

        for i, state in enumerate(self.states.keys()):
            if (
                state not in self.get_reacting_state_order()
                and not self.odes[state].observable
            ):
                continue

            if as_indices:
                observable_states.append(i)
            else:
                observable_states.append(state)

        return sorted(observable_states)

    def get_constants_order(self) -> List[str]:
        """Returns the order of the constants in the model"""
        return sorted(list(self.constants.keys()))

    def _warmup_simulation(
        self,
        y0,
        parameters,
        constants,
        saveat,
        in_axes,
    ):
        """Warms up the simulation to make use of jit compilation"""

        if self._sim_func is None:
            raise ValueError(
                "Simulation function not found. Please run 'simulate' first."
            )

        if in_axes is None:
            self._sim_func(y0, parameters, constants, saveat)
            return

        self._sim_func(
            y0=y0,  # type: ignore
            parameters=parameters,
            constants=constants,
            time=saveat,
        )

    def _setup_rate_function(self, in_axes=None):
        """Prepares a function to evaluate the rates of the vector field.

        This method in particualar is useful to fit rates to the neural ODE and
        thus connect an abstract neural ODE to a physical model. Mainly, this is
        a very efficient approach that mitigates the performance issues of MCMC
        sampling that necessitates full simulations. Thus, by obtaining the rates
        predicted by the neural ODE fitting these to the physical model is faster
        and can be done in parallel.

        Args:
            model (Model): Catalax model to prepare the rate function for.
        """

        assert (
            in_axes is None or len(in_axes) == 4
        ), "Got invalid dimension for 'in_axes' - Needs to have four ints/Nones"

        # Build the stack
        model = self._replace_assignments()

        if in_axes is None:
            return model.stack

        return jax.jit(jax.vmap(model.stack, in_axes=in_axes))

    def reset(self):
        """Resets the model"""

        self._sim_func = None
        self._in_axes = None

    # ! Helper methods
    def reparametrize(self, **replacements) -> "Model":
        """Reparametrizes the model by substituting symbols in all equations with new expressions or values.

        This method performs symbolic substitution across all ODEs and assignments in the model.
        It takes keyword arguments where keys are symbol names and values are the replacement
        expressions, numbers, or None. All substitutions are applied using SymPy's substitution
        mechanism, allowing for both numerical values and symbolic expressions.

        The method handles None values by converting them to 0.0, which is useful for
        eliminating terms from equations. This creates a new model instance with the
        substitutions applied, leaving the original model unchanged.

        Args:
            **replacements: Keyword arguments where keys are symbol names (strings) and
                          values are the replacement expressions, numbers, or None.
                          None values are automatically converted to 0.0.

        Returns:
            Model: A new model instance with the substitutions applied.

        Example:
            >>> model = Model(name="example")
            >>> model.add_state("A, B")
            >>> model.add_ode("A", "k1 * A - k2 * A * B")
            >>> model.add_ode("B", "k2 * A * B - k3 * B")
            >>>
            >>> # Fix parameter k3 to a specific value
            >>> new_model = model.reparametrize(k3=0.5)
            >>>
            >>> # Replace k1 with a more complex expression
            >>> new_model = model.reparametrize(k1="k1_max * A / (K_m + A)")
            >>>
            >>> # Eliminate the k2 term by setting it to zero
            >>> new_model = model.reparametrize(k2=None)
        """

        new_model = copy.deepcopy(self)
        replacements_ = dict()

        for k, v in replacements.items():
            if v is not None:
                replacements_[k] = v
            else:
                replacements_[k] = 0.0

        for equation in list(new_model.odes.values()) + list(
            new_model.assignments.values()
        ):
            equation.reparametrize(**replacements_)  # type: ignore

        # We need to remove parameters that are not used in the model
        # and thus not included in the fit
        new_model._cleanup_unused_parameters()

        return new_model

    def _cleanup_unused_parameters(self):
        """Removes parameters that are not used in the model"""

        all_free_symbols = set()
        for equation in self.odes.values():
            all_free_symbols.update(equation.equation.free_symbols)
        for equation in self.assignments.values():
            all_free_symbols.update(equation.equation.free_symbols)

        all_free_symbols = [str(symbol) for symbol in all_free_symbols]

        unused_params = [
            name
            for name, parameter in self.parameters.items()
            if str(parameter.symbol) not in all_free_symbols
        ]

        for name in unused_params:
            self.parameters.pop(name)

    def has_hdi(self):
        """Checks whether the model has HDI values for any of the parameters"""

        return all(
            parameter.hdi is not None
            and parameter.hdi.lower is not None
            and parameter.hdi.upper is not None
            and parameter.hdi.lower_50 is not None
            and parameter.hdi.upper_50 is not None
            for parameter in self.parameters.values()
        )

    def __str__(self):
        """Prints a summary of the model"""

        print("Model summary")

        eqprint(Symbol("x"), Matrix([state.symbol for state in self.states.values()]).T)

        eqprint(
            Symbol("theta"),
            Matrix([param.name for param in self.parameters.values()]).T,
        )

        for ode in self.odes.values():
            odeprint(y=ode.state.name, expr=ode.equation)

        # Parameter
        self.parameters.__repr__()

        return ""

    @field_validator("states", mode="before")
    def _convert_state_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        symbols_ = []

        for symbol in value:
            if isinstance(symbol, str):
                symbol = symbols(symbol)

            symbols_ += list(symbol)

        return symbols_

    def y0_array_to_dict(self, y0_array: jax.Array) -> List[Dict[str, float]]:
        """
        Converts a 1D or 2D array of initial values to a list of dictionaries.

        Args:
            y0_array (jax.Array): The array of initial values.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, where each dictionary represents
            the initial values for each state in the model.

        Raises:
            AssertionError: If the length of y0 array does not match the number of states in the model.
        """

        if len(y0_array.shape) == 1:
            y0_array = jnp.expand_dims(y0_array, axis=0)

        assert y0_array.shape[-1] == len(
            self.states
        ), "Length of y0 array does not match the number of states in the model."

        return [
            {state: float(value) for state, value in zip(self.get_state_order(), y0)}
            for y0 in y0_array
        ]

    def n_parameters(self) -> int:
        """Get the number of parameters of the model."""
        return len(self.get_parameter_order())

    # ! Exporters
    def save(self, path: str, name: Optional[str] = None, **json_kwargs):
        """Saves a model to a JSON file.

        Args:
            path (str): Path to which the model will be serialized
        """

        if name is None:
            name = self.name.replace(" ", "_")

        fpath = os.path.join(path, f"{name}.json")

        with open(fpath, "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(),
                    default=str,
                    sort_keys=False,
                    indent=2,
                    **json_kwargs,
                )
            )

    def to_enzymeml(self, enzmldoc: Optional[pe.EnzymeMLDocument] = None):
        """Converts the model into an EnzymeML document.

        This method transforms the Catalax model into a PyEnzyme EnzymeMLDocument format,
        preserving all model components including states, parameters, reactions, ODEs,
        and assignments. If an existing document is provided, it will be updated with
        the model's information.

        Args:
            enzmldoc: Optional existing EnzymeML document to update. If None,
                     creates a new document with the model's name.

        Returns:
            pe.EnzymeMLDocument: An EnzymeML document containing all the model's components

        Example:
            >>> model = Model(name="Michaelis-Menten")
            >>> model.add_states(S="Substrate", P="Product", E="Enzyme")
            >>> model.add_reaction("S -> P", symbol="v1", equation="k_cat * E * S / (K_m + S)")
            >>> doc = model.to_enzymeml()
            >>> # Or update existing document
            >>> existing_doc = pe.EnzymeMLDocument(name="existing")
            >>> updated_doc = model.to_enzymeml(existing_doc)
        """
        from catalax.model.enzymeml import to_enzymeml

        return to_enzymeml(self, enzmldoc)

    def to_dict(self):
        """Converts the model into a serializable dictionary."""
        model_dict = DottedDict(self.model_dump(exclude_none=True))
        res = {
            "name": model_dict.name,
            "states": [state.to_dict() for state in model_dict.states.values()],
            "constants": [
                constant.to_dict() for constant in model_dict.constants.values()
            ],
            "odes": [
                {**ode.to_dict(), "state": state}
                for state, ode in model_dict.odes.items()
            ],
            "parameters": [
                parameter.to_dict() for parameter in model_dict.parameters.values()
            ],
            "assignments": [
                assignment.to_dict() for assignment in model_dict.assignments.values()
            ],
        }

        if not res["constants"]:
            res.pop("constants")

        if not res["assignments"]:
            res.pop("assignments")

        return res

    # ! Updaters
    def from_samples(
        self,
        samples: Dict[str, Array],
        hdi_prob: float = 0.95,
        set_bounds: bool = False,
    ) -> "Model":
        """Create a new model from samples drawn from the posterior distribution.

        Args:
            samples (Dict[str, Array]): The samples to update the parameters with.
        """

        new_model = copy.deepcopy(self)
        hdi = az.hdi(samples, hdi_prob=hdi_prob, skipna=True)
        hdi_50 = az.hdi(samples, hdi_prob=0.5, skipna=True)

        for name, parameter in new_model.parameters.items():
            parameter.value = float(jnp.median(samples[name]))
            parameter.initial_value = float(jnp.median(samples[name]))
            parameter.hdi = HDI(
                lower=hdi[name][0],
                upper=hdi[name][1],
                lower_50=hdi_50[name][0],
                upper_50=hdi_50[name][1],
                q=hdi_prob,
            )

            # If any of the HDI boundaries are nan, use the median value and
            # add the opposite bound to the parameter value
            lower_nan = jnp.isnan(parameter.hdi.lower)
            upper_nan = jnp.isnan(parameter.hdi.upper)
            has_nan_hdi = lower_nan and upper_nan

            if has_nan_hdi:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan HDI bounds. Using median value."
                )
            elif lower_nan:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan lower bound. Mirroring from upper bound."
                )
                parameter.hdi.lower = parameter.value - parameter.hdi.upper
            elif upper_nan:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan upper bound. Mirroring from lower bound."
                )
                parameter.hdi.upper = parameter.value + parameter.hdi.lower

            if set_bounds and not has_nan_hdi:
                parameter.upper_bound = parameter.hdi.upper
                parameter.lower_bound = parameter.hdi.lower

        return new_model

    def from_arviz(
        self,
        samples: az.InferenceData,
        hdi_prob: float = 0.95,
        set_bounds: bool = False,
    ) -> "Model":
        """Create a new model from samples drawn from the posterior distribution.

        Args:
            samples (az.InferenceData): The ArviZ InferenceData object containing posterior samples.
            hdi_prob (float): Probability for the highest density interval. Default is 0.95.
            set_bounds (bool): Whether to set parameter bounds based on HDI. Default is False.

        Returns:
            Model: A new model instance with parameters updated from the posterior samples.
        """
        new_model = copy.deepcopy(self)
        hdi = az.hdi(samples, hdi_prob=hdi_prob, skipna=True)
        hdi_50 = az.hdi(samples, hdi_prob=0.5, skipna=True)

        median_posterior = samples.median().posterior  # type: ignore

        for name, parameter in new_model.parameters.items():
            parameter.value = float(median_posterior[name])
            parameter.initial_value = float(median_posterior[name])
            parameter.hdi = HDI(
                lower=hdi[name][0],
                upper=hdi[name][1],
                lower_50=hdi_50[name][0],
                upper_50=hdi_50[name][1],
                q=hdi_prob,
            )

            # If any of the HDI boundaries are nan, use the median value and
            # add the opposite bound to the parameter value
            lower_nan = jnp.isnan(parameter.hdi.lower)
            upper_nan = jnp.isnan(parameter.hdi.upper)
            has_nan_hdi = lower_nan and upper_nan

            if has_nan_hdi:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan HDI bounds. Using median value."
                )
            elif lower_nan:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan lower bound. Mirroring from upper bound."
                )
                parameter.hdi.lower = parameter.value - parameter.hdi.upper
            elif upper_nan:
                rich.print(
                    f"[bold yellow]Warning:[/bold yellow] Parameter {name} has nan upper bound. Mirroring from lower bound."
                )
                parameter.hdi.upper = parameter.value + parameter.hdi.lower

            if set_bounds and not has_nan_hdi:
                parameter.upper_bound = parameter.hdi.upper
                parameter.lower_bound = parameter.hdi.lower

        return new_model

    # ! Importers
    @classmethod
    def load(cls, path: str):
        """Loads a model from a JSON file and initializes a model.

        Args:
            path (str): Path to the model JSON file.

        Raises:
            ValueError: If parameters given in the JSON file are not present in the model.

        Returns:
            Model: Resulting model instance.
        """

        with open(path, "r") as f:
            data = DottedDict(json.load(f))
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, model_dict: Dict):
        """Initializes a model from a dictionary."""

        if not isinstance(model_dict, DottedDict):
            model_dict = DottedDict(model_dict)

        # Initialize the model
        model = cls(name=model_dict.name)

        # Add States
        if "states" in model_dict and model_dict.states:
            model.add_state(**{str(sp.symbol): sp.name for sp in model_dict.states})
        elif "species" in model_dict and model_dict.species:
            model.add_state(**{str(sp.symbol): sp.name for sp in model_dict.species})
        else:
            raise ValueError("No states found in the model")

        if "constants" in model_dict and model_dict.constants:
            model.add_constant(
                **{str(sp.symbol): sp.name for sp in model_dict.constants}
            )

        if "assignments" in model_dict and model_dict.assignments:
            for assignment in model_dict.assignments:
                model.add_assignment(**assignment)

        if "reactions" in model_dict and model_dict.reactions:
            for reaction in model_dict.reactions:
                model.add_reaction(**reaction)

        if "odes" in model_dict and model_dict.odes:
            for ode in model_dict.odes:
                if "species" in ode:
                    ode["state"] = ode.pop("species")
                model.add_ode(**ode)

        # Update given parameters
        for parameter in model_dict.parameters:
            param_name = str(parameter.symbol)
            if param_name not in model.parameters:
                raise ValueError(
                    f"Parameter [symbol: {parameter.symbol}, name: {parameter.name}] not found in the model and thus inconsistent with the given model. Please check the JSON file you are trying to load."
                )

            if "prior" in parameter:
                prior = parameter.pop("prior")
                prior_cls = priors.__dict__[prior.pop("type")]
                model.parameters[param_name].prior = prior_cls(**prior)

            model.parameters[param_name].__dict__.update(parameter)

        return model

    @classmethod
    def from_enzymeml(
        cls,
        enzmldoc: pe.EnzymeMLDocument,
        name: str | None = None,
    ):
        """Initializes a model from an EnzymeML document.

        Args:
            enzmldoc (pe.EnzymeMLDocument): EnzymeML document to initialize the model from.
            name (str | None): Name of the model.
            from_reactions (bool): Whether to initialize the model from reactions.
        """
        return from_enzymeml(cls, enzmldoc, name)

    def update_enzymeml_parameters(self, enzmldoc: EnzymeMLDocument):
        """Updates model parameters of enzymeml document with model parameters.
        Existing parameters will be updated, non-existing parameters will be added.

        Args:
            enzmldoc (EnzymeMLDocument): EnzymeML document to update.
        """

        enzml_param_ids = [param.id for param in enzmldoc.parameters]

        for parameter in self.parameters.values():
            # update existing parameter
            if parameter.name in enzml_param_ids:
                enzymeml_param = next(
                    p for p in enzmldoc.parameters if p.id == parameter.name
                )
                enzymeml_param.value = parameter.value
                enzymeml_param.initial_value = parameter.initial_value
                enzymeml_param.constant = parameter.constant
                enzymeml_param.upper_bound = parameter.upper_bound
                enzymeml_param.lower_bound = parameter.lower_bound

            # add new parameter
            else:
                enzmldoc.add_to_parameters(
                    id=parameter.name,
                    symbol=parameter.name,
                    name=parameter.name,
                    value=parameter.value,
                    initial_value=parameter.initial_value,
                    constant=parameter.constant,
                    upper=parameter.upper_bound,
                    lower=parameter.lower_bound,
                )
