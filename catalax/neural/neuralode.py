from typing import List

import jax
import diffrax

from catalax import Model
from .mlp import MLP
from .neuralbase import NeuralBase
from .rbf import RBFLayer


class NeuralODE(NeuralBase):
    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        activation=jax.nn.softplus,
        use_final_bias: bool = False,
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
            use_final_bias=use_final_bias,
            **kwargs,
        )

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),  # type: ignore
            self.solver(),  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),  # type: ignore
            saveat=diffrax.SaveAt(ts=ts),  # type: ignore
        )
        return solution.ys
