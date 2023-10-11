import json
import os
from typing import Dict, List, Optional

import diffrax
import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp

from catalax import Model
from catalax.neural.rbf import RBFLayer
from catalax.tools.simulation import Stack

from .mlp import MLP


class NeuralBase(eqx.Module):
    func: MLP
    observable_indices: List[int]
    hyperparams: Dict
    solver: diffrax.AbstractSolver
    vector_field: Optional[Stack]

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        activation=jax.nn.softplus,
        solver=diffrax.Tsit5,
        use_final_bias: bool = False,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Save solver and observable indices
        self.solver = solver
        self.observable_indices = observable_indices
        self.vector_field = None

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
            "model": model.to_dict(),
            "rbf": isinstance(activation, RBFLayer),
            "use_final_bias": use_final_bias,
            "observable_indices": observable_indices,
        }

        # Save solver and MLP
        self.func = MLP(
            data_size,
            width_size,
            depth,
            activation=activation,
            key=key,
            use_final_bias=use_final_bias,
        )  # type: ignore

    def predict(
        self,
        y0s: jax.Array,
        t0: int = 0,
        t1: Optional[int] = None,
        nsteps: int = 1000,
        times: Optional[jax.Array] = None,
    ):
        if times is None:
            assert t1 is not None, "Either times or t1 must be given."

            # Generate time points, if not explicitly given
            times = jnp.linspace(t0, t1, nsteps)

        if y0s.shape[0] > 1 and len(times.shape) == 1:
            # If multiple initial conditions are given, repeat time points
            times = jnp.stack([times] * y0s.shape[0], axis=0)

        # Single simulation case
        if len(times.shape) == 1 and len(y0s.shape) == 1:
            return times, self(times, y0s)  # type: ignore

        # Batch simulation case
        assert (
            len(times.shape) == 2 and len(y0s.shape) == 2
        ), f"Incompatible shapes: time shape = {times.shape}, y0 shape = {y0s.shape}. Both must be 2D."

        assert (
            times.shape[0] == y0s.shape[0]
        ), f"Incompatible shapes: time shape = {times.shape}, y0.shape = {y0s.shape}. First dimension must be equal in case of batches."

        return times, jax.vmap(self, in_axes=(0, 0))(times, y0s)  # type: ignore

    @classmethod
    def from_model(
        cls,
        model: Model,
        width_size: int,
        depth: int,
        seed: int = 0,
        use_final_bias: bool = True,
        solver=diffrax.Tsit5,
        activation=jax.nn.softplus,
        **kwargs,
    ):
        """Intializes a NeuralODE from a catalax.Model

        Args:
            model (Model): Model to initialize NeuralODE from
        """

        key = jrandom.PRNGKey(seed)

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
            use_final_bias=use_final_bias,
            activation=activation,
            **kwargs,
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
            model = Model.from_dict(hyperparams.pop("model"))

            if "observable_indices" not in hyperparams:
                hyperparams["observable_indices"] = [0]
            if hyperparams["rbf"] is True:
                hyperparams["activation"] = RBFLayer(0.4)

            # Remove rbf from hyperparams
            del hyperparams["rbf"]

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
