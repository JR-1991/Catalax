import json
import os
from typing import Dict, List

import diffrax
import equinox as eqx
import jax.nn as jnn
import jax.random as jrandom

from catalax import Model


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        activation=jnn.softplus,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )  # type: ignore

    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: MLP
    observable_indices: List[int]
    hyperparams: Dict
    solver: diffrax.AbstractSolver = diffrax.Tsit5

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Save solver and MLP
        self.func = MLP(data_size, width_size, depth, key=key)  # type: ignore
        self.solver = solver
        self.observable_indices = observable_indices

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
        }

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            self.solver(),  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

    @classmethod
    def from_model(
        cls, model: Model, width_size: int, depth: int, key, solver=diffrax.Tsit5
    ):
        """Intializes a NeuralODE from a catalax.Model

        Args:
            model (Model): Model to initialize NeuralODE from
        """

        # Get observable indices
        observable_indices = [
            index
            for index, species in enumerate(model._get_species_order())
            if model.odes[species].observable
        ]

        return cls(
            data_size=len(model.species),
            width_size=width_size,
            depth=depth,
            solver=solver,
            observable_indices=observable_indices,
            key=key,
        )

    @classmethod
    def from_eqx(cls, path):
        """Loads a NeuralODE from an eqx file

        Args:
            path (str): Path to the eqx file

        Returns:
            NeuralODE: Trained NeuralODE model.
        """

        with open(path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())["hyperparameters"]
            model = cls(**hyperparams, key=jrandom.PRNGKey(0))
            model = eqx.tree_deserialise_leaves(f, model)

        return model

    def save_to_eqx(self, path: str, name: str):
        """Saves a NeuralODE to an eqx file

        Args:
            path (str): Path to the directory to save the eqx file
            name (str): Name of the eqx file
        """
        
        filename = os.path.join(path, name + ".eqx")
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(self.hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
