from __future__ import annotations

import json
import os
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Type

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self

import diffrax
import diffrax as dfx
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtn
import optax
from deprecated import deprecated

from catalax import Model
from catalax.dataset import Dataset
from catalax.model.model import SimulationConfig
from catalax.neural.mlp import MLP
from catalax.neural.rbf import RBFLayer
from catalax.neural.utils import (
    ACTIVATION_MAP,
    SERIALISATION_WARNING,
    _get_activation_name,
)
from catalax.predictor import Predictor
from catalax.surrogate import Surrogate
from catalax.tools.simulation import ODEStack

if TYPE_CHECKING:
    from catalax.dataset import Dataset
    from catalax.neural.strategy import Strategy

NON_ADAPTIVE_SOLVERS = [
    dfx.Euler,
    dfx.Heun,
]


class NeuralBase(eqx.Module, Predictor, Surrogate):
    func: MLP
    observable_indices: List[int]
    hyperparams: Dict
    solver: Type[diffrax.AbstractSolver]
    vector_field: Optional[ODEStack]
    state_order: List[str]

    @property
    @deprecated("This property is deprecated. Use state_order instead.")
    def species_order(self) -> List[str]:
        """Get the species order of the predictor."""
        return self.state_order

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        state_order: List[str],
        observable_indices: List[int],
        activation=jax.nn.tanh,
        solver=diffrax.Tsit5,
        use_final_bias: bool = False,
        final_activation: Callable = jax.nn.identity,
        out_size: Optional[int] = None,
        *,
        key,
        **kwargs,
    ):
        # Save solver and observable indices
        self.solver = solver
        self.observable_indices = observable_indices
        self.vector_field = None
        self.state_order = state_order

        # Keep hyperparams for serialisation
        self.hyperparams = {
            "data_size": data_size,
            "width_size": width_size,
            "depth": depth,
            "out_size": out_size,
            "rbf": isinstance(activation, RBFLayer),
            "use_final_bias": use_final_bias,
            "observable_indices": observable_indices,
            "state_order": self.state_order,
            **kwargs,
        }

        # Try to get activation names, warn if not found
        try:
            self.hyperparams["activation"] = _get_activation_name(activation)
        except KeyError:
            warnings.warn(SERIALISATION_WARNING)

        try:
            self.hyperparams["final_activation"] = _get_activation_name(
                final_activation
            )
        except KeyError:
            warnings.warn(SERIALISATION_WARNING)

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

    @abstractmethod
    def __call__(
        self,
        ts,
        y0,
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
    ) -> jax.Array:
        raise NotImplementedError("This method is not implemented")

    def train(
        self,
        dataset: Dataset,
        strategy: Strategy,
        validation_dataset: Optional[Dataset] = None,
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
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
            validation_dataset=validation_dataset,
            print_every=print_every,
            weight_scale=weight_scale,
            solver=solver,
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
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
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

        if hdi is not None:
            raise NotImplementedError(
                f"HDI is not available for {self.__class__.__name__}"
            )

        if config is None and not use_times:
            config = dataset.to_config(nsteps=n_steps)
        if config and config.nsteps != n_steps:
            config.nsteps = n_steps

        if not dataset.has_data():
            assert config is not None, (
                "Dataset consists of only initial conditions, therefore a simulation "
                "configuration is required to generate predictions."
            )
            y0s = dataset.to_y0_matrix(self.state_order)
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore
        else:
            _, times, y0s = dataset.to_jax_arrays(
                self.state_order,
                inits_to_array=True,
            )

        if config:
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore

        predictions = jax.vmap(
            lambda ts, y0: self(ts, y0, solver=solver, rtol=rtol, atol=atol, dt0=dt0),
            in_axes=(0, 0),
        )(times, y0s)

        return Dataset.from_jax_arrays(
            state_order=self.state_order,
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
        data, times, _ = dataset.to_jax_arrays(self.state_order)
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
            self.state_order
        )
        y_true, _, _ = dataset.to_jax_arrays(self.state_order)

        return loss(y_pred, y_true)

    @classmethod
    def from_model(
        cls,
        model: Model,
        width_size: int,
        depth: int,
        seed: int = 0,
        use_final_bias: bool = False,
        final_activation: Callable = jax.nn.identity,
        solver=diffrax.Tsit5,
        activation=jax.nn.tanh,
        **kwargs,
    ):
        """Intializes a NeuralODE from a catalax.Model

        Args:
            model (Model): Model to initialize NeuralODE from
        """

        key = jrandom.PRNGKey(seed)

        # Get observable indices
        if len(model.odes) > 0 or len(model.reactions) > 0:
            observable_indices = model.get_state_order(as_indices=True)
            state_order = model.get_state_order()
        else:
            observable_indices = model.get_state_order(as_indices=True, modeled=False)
            state_order = model.get_state_order(modeled=False)

        return cls(
            data_size=len(model.states),
            width_size=width_size,
            depth=depth,
            solver=solver,
            observable_indices=observable_indices,
            state_order=state_order,
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

    def weights_from_eqx(self, path: str) -> Self:
        """Loads the weights and biases from an eqx file.

        Please note that this method requires the same neural network architecture as the one used to train the model. It is recommended to use this path of persistence only for models that use custom activation functions. If you want to persist a model that uses a supported activation function, which you can find in 'https://github.com/JR-1991/Catalax/blob/master/catalax/neural/neuralbase.py'.

        Args:
            path (str): Path to the eqx file
        """
        with open(path, "rb") as f:
            return eqx.tree_deserialise_leaves(f, self)

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

            if "final_activation" in hyperparams:
                hyperparams["final_activation"] = ACTIVATION_MAP[
                    hyperparams["final_activation"]
                ]
            else:
                warnings.warn("No final activation function found. Using identity.")

            if "activation" in hyperparams:
                hyperparams["activation"] = ACTIVATION_MAP[hyperparams["activation"]]
            else:
                warnings.warn("No activation function found. Using tanh.")

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

        if "activation" not in self.hyperparams:
            try:
                self.hyperparams["activation"] = _get_activation_name(
                    self.func.mlp.activation
                )
            except KeyError:
                # Custom activation function - skip storing in hyperparams
                pass
        if "final_activation" not in self.hyperparams:
            try:
                self.hyperparams["final_activation"] = _get_activation_name(
                    self.func.mlp.final_activation
                )
            except KeyError:
                # Custom activation function - skip storing in hyperparams
                pass

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

    @deprecated("This method is deprecated. Use get_state_order instead.")
    def get_species_order(self) -> list[str]:
        """Get the species order of the predictor.

        Returns:
            List of species order
        """
        return self.get_state_order()

    def get_state_order(self) -> list[str]:
        """Get the state order of the predictor.

        Returns:
            List of state order
        """
        return self.state_order

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

    def _create_controller(
        self,
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        """Create the appropriate stepsize controller"""
        if solver is None:
            if self.solver in NON_ADAPTIVE_SOLVERS:
                return diffrax.ConstantStepSize()
            else:
                return diffrax.PIDController(
                    rtol=rtol if rtol is not None else 1e-3,
                    atol=atol if atol is not None else 1e-6,
                )

        if solver in NON_ADAPTIVE_SOLVERS:
            return diffrax.ConstantStepSize()
        else:
            return diffrax.PIDController(
                rtol=rtol if rtol is not None else 1e-3,
                atol=atol if atol is not None else 1e-6,
            )

    def _instantiate_solver(
        self,
        solver: Optional[Type[diffrax.AbstractSolver]],
    ) -> diffrax.AbstractSolver:
        if solver is None:
            return self.solver()
        else:
            return solver()
