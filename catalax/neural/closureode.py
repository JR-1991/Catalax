from functools import partial
from typing import Callable, List

import diffrax
import jax.numpy as jnp
import equinox as eqx

from catalax import Model
from .mlp import MLP
from .neuralbase import NeuralBase


class ClosureODE(NeuralBase):
    vector_field: Callable
    func: eqx.Module
    influence: float

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        influence=1e-2,
        *,
        key,
        **kwargs
    ):
        super().__init__(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            observable_indices=observable_indices,
            solver=solver,
            model=model,
            **kwargs,
        )

        # Save solver and MLP
        self.influence = influence
        self.func = MLP(data_size, width_size, depth, key=key)  # type: ignore

        # Get the term of the model and define the closure
        self.vector_field = self._extract_vector_field(model)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self._closure),
            self.solver(),  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

    def _closure(self, t, y, args):
        """Performs closure by adding a neural network term"""
        return self.vector_field(t, y) + self.influence * self.func(t, y, args)

    @staticmethod
    def _extract_vector_field(model: Model) -> Callable:
        """Extracts the vector field from a catalax.Model"""
        vector_field = model._setup_term().vector_field
        args = (
            {species: i for i, species in enumerate(model._get_species_order())},
            {parameter: i for i, parameter in enumerate(model._get_parameter_order())},
            jnp.array(
                [
                    model.parameters[parameter].value
                    for parameter in model._get_parameter_order()
                ]
            ),
            model._get_stoich_mat(),
        )

        return partial(vector_field, args=args)
