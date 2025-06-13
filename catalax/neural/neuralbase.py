from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jrandom

from catalax import Model
from catalax.model.model import SimulationConfig
from catalax.neural.rbf import RBFLayer
from catalax.dataset import Dataset
from catalax.predictor import Predictor
from catalax.surrogate import Surrogate
from catalax.tools.simulation import Stack
from .mlp import MLP

if TYPE_CHECKING:
    from catalax.dataset import Dataset


class NeuralBase(eqx.Module, Predictor, Surrogate):
    func: MLP
    observable_indices: List[int]
    hyperparams: Dict
    solver: diffrax.AbstractSolver
    vector_field: Optional[Stack]
    species_order: List[str]

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model | Dataset,
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
        self.solver = solver  # type: ignore
        self.observable_indices = observable_indices
        self.vector_field = None
        self.species_order = model.get_observable_species_order()

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
            "rbf": isinstance(activation, RBFLayer),
            "use_final_bias": use_final_bias,
            "observable_indices": observable_indices,
            "species_order": self.species_order,
        }

        # Save solver and MLP
        self.func = MLP(
            data_size,
            width_size,
            depth,
            activation=activation,
            key=key,
            use_final_bias=use_final_bias,
        )

    def predict(
        self,
        dataset: Dataset,
        config: Optional[SimulationConfig] = None,
        n_steps: int = 100,
    ):
        """Predict model behavior using the given dataset.

        This is a convenience wrapper around the `simulate` method that automatically
        creates a simulation configuration if one is not provided.

        Args:
            dataset: Dataset containing initial conditions for prediction
            config: Optional simulation configuration parameters. If None,
                   a configuration will be created from the dataset.
            n_steps: Number of time steps for the simulation. This will override
                   the nsteps value in the provided config if both are specified.

        Returns:
            A Dataset object containing the prediction results
        """

        if config is None:
            config = dataset.to_config(nsteps=n_steps)
        if config.nsteps != n_steps:
            config.nsteps = n_steps

        _, times, y0s = dataset.to_jax_arrays(
            self.species_order,
            inits_to_array=True,
        )

        if config:
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T

        predictions = jax.vmap(self, in_axes=(0, 0))(times, y0s)  # type: ignore

        return Dataset.from_jax_arrays(
            species_order=self.species_order,
            data=predictions,
            time=times,
            y0s=y0s,  # type: ignore
        )

    def predict_rates(self, dataset: Dataset) -> jax.Array:
        """Predict rates using the given dataset.

        Args:
            dataset: Dataset containing initial conditions for prediction

        Returns:
            A Dataset object containing the prediction results
        """
        data, times, _ = dataset.to_jax_arrays(self.species_order)
        dataset_size, time_size, _ = data.shape
        ins = data.reshape(dataset_size * time_size, -1)
        times = times.ravel()
        return jax.vmap(self.func, in_axes=(0, 0, None))(times, ins, 0.0).reshape(
            dataset_size * time_size, -1
        )

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
        if model.odes:
            observable_indices = [
                index
                for index, species in enumerate(model.get_species_order())
                if model.odes[species].observable
            ]
        else:
            observable_indices = list(range(len(model.get_species_order())))

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
    def from_dataset(
        cls,
        dataset: Dataset,
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
        observable_indices = dataset.get_observable_indices()

        return cls(
            data_size=len(observable_indices),
            width_size=width_size,
            depth=depth,
            solver=solver,
            observable_indices=observable_indices,
            key=key,
            model=dataset,
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

    def save_to_onnx(self):
        return eqxi.to_onnx(self.func)
