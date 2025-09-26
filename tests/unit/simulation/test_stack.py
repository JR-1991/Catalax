import jax.numpy as jnp

from catalax.model.ode import ODE
from catalax.model.reaction import Reaction
from catalax.model.state import State
from catalax.tools.simulation import SimulationInput
from catalax.tools.stack import MixedStack, ODEStack, ReactionStack


class TestStack:
    """Test suite for stack functionality in simulation framework."""

    def test_mixed_stack(self):
        """Test MixedStack creation and evaluation with both ODEs and reactions.

        Creates a MixedStack with one ODE (s1' = k3) and implicit reactions
        (s1 <-> s2 with rates k1*s1 and k2*s2), then verifies correct evaluation
        of the combined system dynamics.
        """
        sim_input = SimulationInput(
            parameters=["k1", "k2", "k3"],
            states=["s1", "s2"],
            constants=[],
            reactions=[
                Reaction.from_scheme(
                    symbol="r1",
                    schema="s1 -> s2",
                    equation="k1 * s1",
                    reversible=False,
                ),
                Reaction.from_scheme(
                    symbol="r2",
                    schema="s2 -> s1",
                    equation="k2 * s2",
                    reversible=False,
                ),
            ],
            odes=[
                ODE(
                    state=State(symbol="s1", name="s1"),  # type: ignore
                    equation="k3",  # type: ignore
                )
            ],
        )

        stack = sim_input.stack

        assert isinstance(
            stack, MixedStack
        ), "Stack should be MixedStack for mixed ODE/reaction system"
        assert (
            len(stack.ode_stack.modules) == 2
        ), "ODE stack should have two modules when only constant ODEs present"
        assert (
            len(stack.reac_stack.modules) == 2
        ), "Reaction stack should have two modules for the two reactions"

        expected_stoich_mat = jnp.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ]
        )
        actual_stoich_mat = jnp.array(sim_input.stack.reac_stack.stoich_mat.tolist())
        assert jnp.allclose(
            actual_stoich_mat, expected_stoich_mat
        ), "Stoichiometric matrix should match expected bidirectional reaction pattern"

        k1 = 1.0
        k2 = 2.0
        k3 = 3.0
        s1 = 1.0
        s2 = 2.0
        state = [1.0, 2.0]
        expected_result = jnp.array(
            [
                -k1 * s1 + k2 * s2 + k3,  # s1 rate: -1*rate1 + 1*rate2 + k3
                k1 * s1 - k2 * s2,  # s2 rate: 1*rate1 + (-1)*rate2
            ]
        )
        result = sim_input.stack(
            t=1.0,
            y=state,
            args=(jnp.array([k1, k2, k3]), jnp.array([])),
        )
        assert jnp.allclose(
            result, expected_result
        ), "Mixed stack evaluation should correctly combine ODE and reaction contributions"

    def test_ode_stack(self):
        """Test ODEStack creation and evaluation with simple ODE equations.

        Creates an ODEStack with two ODEs (s1' = k1 * s1, s2' = k2 * s2) and
        verifies that it correctly evaluates the derivatives at a given state.
        """
        sim_input = SimulationInput(
            parameters=["k1", "k2"],
            states=["s1", "s2"],
            constants=[],
            odes=[
                ODE(state=State(symbol="s1", name="s1"), equation="k1 * s1"),  # type: ignore
                ODE(state=State(symbol="s2", name="s2"), equation="k2 * s2"),  # type: ignore
            ],
        )
        stack = sim_input.stack

        assert isinstance(
            stack, ODEStack
        ), "Stack should be ODEStack for pure ODE system"
        assert len(stack.modules) == 2, "ODE stack should have two modules for two ODEs"

        k1 = 1.0
        k2 = 2.0
        s1 = 1.0
        s2 = 2.0
        state = [1.0, 2.0]
        expected_result = jnp.array(
            [
                k1 * s1,
                k2 * s2,
            ]
        )
        result = stack(
            t=1.0,
            y=state,
            args=(
                jnp.array([k1, k2]),
                jnp.array([]),
            ),
        )
        assert jnp.allclose(
            result, expected_result
        ), "ODE stack should correctly evaluate derivative expressions"

    def test_reaction_stack(self):
        """Test ReactionStack creation and evaluation with bidirectional reactions.

        Creates a ReactionStack with two opposing reactions (s1 -> s2 and s2 -> s1)
        and verifies correct stoichiometric matrix construction and rate evaluation.
        """
        sim_input = SimulationInput(
            parameters=["k1", "k2"],
            states=["s1", "s2"],
            constants=[],
            reactions=[
                Reaction.from_scheme(
                    symbol="r1",
                    schema="s1 -> s2",
                    equation="k1 * s1",
                    reversible=False,
                ),
                Reaction.from_scheme(
                    symbol="r2",
                    schema="s2 -> s1",
                    equation="k2 * s2",
                    reversible=False,
                ),
            ],
        )
        stack = sim_input.stack

        assert isinstance(
            stack, ReactionStack
        ), "Stack should be ReactionStack for pure reaction system"
        assert (
            len(stack.modules) == 2
        ), "Reaction stack should have two modules for two reactions"
        assert stack.stoich_mat.shape == (
            2,
            2,
        ), "Stoichiometric matrix should be 2x2 for 2 reactions and 2 species"
        assert jnp.allclose(
            stack.stoich_mat,
            jnp.array(
                [
                    [-1.0, 1.0],
                    [1.0, -1.0],
                ]
            ),
        ), "Stoichiometric matrix should correctly represent bidirectional s1 <-> s2 conversion"

        k1 = 1.0
        k2 = 2.0
        s1 = 1.0
        s2 = 2.0
        state = [1.0, 2.0]

        expected_result = jnp.array(
            [
                -k1 * s1 + k2 * s2,  # s1 rate: -1*rate1 + 1*rate2
                k1 * s1 - k2 * s2,  # s2 rate: 1*rate1 + (-1)*rate2
            ]
        )

        result = stack(
            t=1.0,
            y=state,
            args=(jnp.array([k1, k2]), jnp.array([])),
        )

        assert jnp.allclose(
            result, expected_result
        ), "Reaction stack should correctly compute net rates from stoichiometry and kinetics"
