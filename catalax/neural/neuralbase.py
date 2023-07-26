import json
import os
from typing import Dict, List
from copy import deepcopy

import diffrax
import equinox as eqx
import jax.random as jrandom

from catalax import Model
from .mlp import MLP


class NeuralBase(eqx.Module):
    func: MLP
    observable_indices: List[int]
    hyperparams: Dict
    solver: diffrax.AbstractSolver

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Save solver and observable indices
        self.solver = solver
        self.observable_indices = observable_indices

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
            "model": model.to_dict(),
        }

    @classmethod
    def from_model(
        cls,
        model: Model,
        width_size: int,
        depth: int,
        key,
        solver=diffrax.Tsit5,
        **kwargs
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
            model=model,
            **kwargs
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

            if "model" not in hyperparams:
                model = Model(name="NO MODEL AVAILABLE")
            else:
                model = Model.from_dict(hyperparams.pop("model"))

            if "observable_indices" not in hyperparams:
                hyperparams["observable_indices"] = [0]

            neuralode = cls(**hyperparams, model=model, key=jrandom.PRNGKey(0))
            neuralode = eqx.tree_deserialise_leaves(f, neuralode)

        return neuralode

    def save_to_eqx(self, path: str, name: str, **kwargs):
        """Saves a NeuralODE to an eqx file

        Args:
            path (str): Path to the directory to save the eqx file
            name (str): Name of the eqx file
        """

        filename = os.path.join(path, name + ".eqx")
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(
                {"hyperparameters": self.hyperparams, **kwargs}, default=str
            )
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
