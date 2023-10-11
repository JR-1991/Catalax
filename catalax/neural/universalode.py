from typing import List

import jax
import diffrax
import jax.numpy as jnp
import equinox as eqx

from catalax import Model
from catalax.tools import Stack
from .mlp import MLP
from .neuralbase import NeuralBase


class UniversalODE(NeuralBase):
    parameters: jax.Array
    vector_field: Stack

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        use_final_bias: bool = False,
        activation=jax.nn.softplus,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            observable_indices=observable_indices,
            solver=solver,
            model=model,
            activation=activation,
            key=key,
            **kwargs,
        )

        # Save solver and MLP
        self.func = MLP(
            data_size,
            width_size,
            depth,
            key=key,
            use_final_bias=use_final_bias,
            activation=activation,
        )  # type: ignore

        # Get the term of the model and define the closure
        self.parameters = jnp.array(model._get_parameters())
        self.vector_field = eqx.filter_jit(model._setup_rate_function())  # type: ignore

    def _combined_term(self, t, y, args):
        """Merges neural network terms and vector field"""
        return self.vector_field(t, y, self.parameters) + self.func(t, y, args)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self._combined_term),  # type: ignore
            self.solver(),  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),  # type: ignore
            max_steps=64**4,
        )
        return solution.ys
