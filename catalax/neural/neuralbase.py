from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional


try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtn
import optax

from catalax import Model
from catalax.model.model import SimulationConfig
from catalax.neural.rbf import RBFLayer
from catalax.dataset import Dataset
from catalax.predictor import Predictor
from catalax.surrogate import Surrogate
from catalax.tools.simulation import Stack
from catalax.neural.mlp import MLP

if TYPE_CHECKING:
    from catalax.dataset import Dataset
    from catalax.neural.strategy import Strategy


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
        species_order: List[str],
        observable_indices: List[int],
        activation=jax.nn.softplus,
        solver=diffrax.Tsit5,
        use_final_bias: bool = False,
        final_activation: Optional[Callable] = None,
        out_size: Optional[int] = None,
        *,
        key,
        **kwargs,
    ):
        # Save solver and observable indices
        self.solver = solver  # type: ignore
        self.observable_indices = observable_indices
        self.vector_field = None
        self.species_order = species_order

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
            "out_size": out_size,
            "rbf": isinstance(activation, RBFLayer),
            "use_final_bias": use_final_bias,
            "observable_indices": observable_indices,
            "species_order": self.species_order,
            **kwargs,
        }

        # Save solver and MLP
        self.func = MLP(
            data_size,
            width_size,
            depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
            use_final_bias=use_final_bias,
            out_size=out_size,
        )

    def train(
        self,
        dataset: Dataset,
        strategy: Strategy,
        optimizer=optax.adabelief,
        print_every: int = 10,
        weight_scale: float = 1e-8,
        save_milestones: bool = False,
        milestone_dir: str = "./milestones",
        log: Optional[str] = None,
        seed: int = 420,
    ) -> Self:
        """Train the model on the given dataset.

        This method trains the neural ODE model using the specified training strategy and
        hyperparameters. The training process involves optimizing the neural network parameters
        to minimize the loss function defined in the strategy over multiple epochs and batches.

        The method will:
        - Extract training data from the provided dataset
        - Initialize the optimizer with the specified learning rate schedule
        - Scale model weights according to the weight_scale parameter
        - Execute the training loop following the provided strategy
        - Optionally save model checkpoints at specified milestones
        - Log training progress if a log file is specified

        The training uses JAX for automatic differentiation and JIT compilation to ensure
        efficient computation. The model parameters are updated using the specified optimizer
        (default: AdaBelief) to minimize prediction errors on the training data.

        Returns the trained model with optimized parameters that can be used for making
        predictions on new data or for further analysis.

        Args:
            dataset: Dataset containing initial conditions for training
            strategy: Training strategy to use
            print_every: Print progress every n steps
            weight_scale: Weight scale for the optimizer
            save_milestones: Save model checkpoints
            log: Log file to save progress

        Returns:
            NeuralODE: The trained neural ODE model with updated parameters.
                The model will have learned to approximate the dynamics from the provided
                dataset using the specified training strategy and hyperparameters.
        """

        from catalax.neural.trainer import train_neural_ode

        return train_neural_ode(
            model=self,
            dataset=dataset,
            strategy=strategy,
            optimizer=optimizer,
            print_every=print_every,
            weight_scale=weight_scale,
            save_milestones=save_milestones,
            milestone_dir=milestone_dir,
            log=log,
            seed=seed,
        )

    def predict(
        self,
        dataset: Dataset,
        config: Optional[SimulationConfig] = None,
        n_steps: int = 100,
        use_times: bool = False,
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
            use_times: Whether to use the time points from the dataset or to simulate at fixed time steps

        Returns:
            A Dataset object containing the prediction results
        """

        if config is None and not use_times:
            config = dataset.to_config(nsteps=n_steps)
        if config and config.nsteps != n_steps:
            config.nsteps = n_steps

        if not dataset.has_data():
            assert config is not None, (
                "Dataset consists of only initial conditions, therefore a simulation "
                "configuration is required to generate predictions."
            )
            y0s = dataset.to_y0_matrix(self.species_order)
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore
        else:
            _, times, y0s = dataset.to_jax_arrays(
                self.species_order,
                inits_to_array=True,
            )

        if config:
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore

        predictions = jax.vmap(self, in_axes=(0, 0))(times, y0s)  # type: ignore

        return Dataset.from_jax_arrays(
            species_order=self.species_order,
            data=predictions,
            time=times,
            y0s=y0s,  # type: ignore
        )

    def rates(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Get the rates of the predictor."""
        t, y, _ = self._validate_rate_input(t, y, None)
        return jax.vmap(self.func, in_axes=(0, 0, None))(t, y, None)

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

        rates = self.rates(times, ins, None)
        return rates.reshape(dataset_size * time_size, -1)

    def loss(self, dataset: Dataset, loss: Callable = optax.log_cosh) -> jax.Array:
        """Calculate the loss of the model on the given dataset.

        Args:
            dataset: Dataset to calculate the loss on
            loss: Loss function to use

        Returns:
            Loss value
        """

        y_pred, _, _ = self.predict(dataset, use_times=True).to_jax_arrays(
            self.species_order
        )
        y_true, _, _ = dataset.to_jax_arrays(self.species_order)

        return loss(y_pred, y_true)

    @classmethod
    def from_model(
        cls,
        model: Model,
        width_size: int,
        depth: int,
        seed: int = 0,
        use_final_bias: bool = False,
        final_activation: Optional[Callable] = None,
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
            species_order=model.get_species_order(),
            key=key,
            model=model,
            use_final_bias=use_final_bias,
            activation=activation,
            final_activation=final_activation,
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
    def from_eqx(cls, path) -> Self:
        """Loads a NeuralODE from an eqx file

        Args:
            path (str): Path to the eqx file

        Returns:
            NeuralODE: Trained NeuralODE model.
        """

        with open(path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())["hyperparameters"]

            if "observable_indices" not in hyperparams:
                hyperparams["observable_indices"] = [0]
            if hyperparams["rbf"] is True:
                hyperparams["activation"] = RBFLayer(0.4)

            # Remove rbf from hyperparams
            del hyperparams["rbf"]

            neuralode = cls(**hyperparams, key=jrandom.PRNGKey(0))
            neuralode = eqx.tree_deserialise_leaves(f, neuralode)

        return neuralode

    def save_to_eqx(self, path: str, name: str, **kwargs):
        """Saves a NeuralODE to an eqx file

        Args:
            path (str): Path to the directory to save the eqx file
            name (str): Name of the eqx file
        """

        if name.endswith(".eqx"):
            name = name.rstrip(".eqx")

        filename = os.path.join(path, name + ".eqx")
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(
                {
                    "hyperparameters": self.hyperparams,
                    **kwargs,
                },
                default=str,
            )
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    def save_to_onnx(self):
        return eqxi.to_onnx(self.func)

    def get_weights_and_biases(self) -> List[jax.Array]:
        """Get all weights and biases from the model.

        Returns:
            List of weights and biases
        """
        return [
            layer for layer in jtn.tree_flatten(self)[0] if isinstance(layer, jax.Array)
        ]

    def get_extra_hyperparams(self) -> Dict[str, Any]:
        """Get extra hyperparameters from the model.

        Returns:
            Dict of extra hyperparameters
        """
        return {}

    def get_species_order(self) -> list[str]:
        """Get the species order of the predictor.

        Returns:
            List of species order
        """
        return self.species_order

    def n_parameters(self) -> int:
        """Get the number of parameters of the predictor.

        Returns:
            Number of parameters
        """
        layers = self.get_weights_and_biases()
        n_parameters = 0
        for layer in layers:
            n_parameters += layer.size
        return n_parameters
