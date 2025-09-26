from typing import List, Optional, Type

import diffrax
import jax

from .neuralbase import NeuralBase


class NeuralODE(NeuralBase):
    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        state_order: List[str],
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        activation=jax.nn.tanh,
        use_final_bias: bool = False,
        final_activation=jax.nn.identity,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            state_order=state_order,
            observable_indices=observable_indices,
            solver=solver,
            activation=activation,
            key=key,
            use_final_bias=use_final_bias,
            final_activation=final_activation,
        )

    def __call__(
        self,
        ts,
        y0,
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
    ):
        stepsize_controller = self._create_controller(solver, rtol, atol)
        solver_instance = self._instantiate_solver(solver)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),  # type: ignore
            solver_instance,  # type: ignore
            t0=ts[0],  # type: ignore
            t1=ts[-1],
            dt0=dt0 or ts[1] - ts[0],
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),  # type: ignore
        )
        return solution.ys

    def get_rates(
        self,
        t: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        """Get the rates of the model.

        This basically evaluates the MLP at the given time points and states, which
        is useful for quiver plots and MCMC surrogates.

        Args:
            t: Time points
            y: States

        Returns:
            Rates
        """
        t, y, _ = self._validate_rate_input(t, y, None)
        res: jax.Array = jax.vmap(self.func, in_axes=(0, 0, None))(t, y, None)
        return res
