import importlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import rich

from catalax.dataset import Dataset
from catalax.neural.neuralbase import NeuralBase
from catalax.predictor import Predictor
from catalax.surrogate import Surrogate

if TYPE_CHECKING:
    from catalax.model.model import SimulationConfig


class NeuralODEEnsemble(eqx.Module, Predictor, Surrogate):
    """Ensemble of neural ODE models for robust predictions and uncertainty quantification.

    This class combines multiple neural ODE models into an ensemble that can provide
    more robust predictions and uncertainty estimates. It implements both Predictor
    and Surrogate interfaces, allowing it to be used for forward predictions and
    as a surrogate model for MCMC sampling.

    The ensemble aggregates predictions from all constituent models, typically by
    taking the mean, but can also provide uncertainty bounds through HDI (Highest
    Density Interval) calculations.

    Attributes:
        models: List of neural ODE models that comprise the ensemble
        state_order: Order of state variables, must be consistent across all models
    """

    models: List[NeuralBase]
    state_order: List[str] = eqx.field(static=True)

    def __init__(self, models: Sequence[NeuralBase]):
        """Initialize the neural ODE ensemble.

        Args:
            models: List of neural ODE models to include in the ensemble

        Raises:
            AssertionError: If models have inconsistent state orders
        """
        self.models = list(models)

        all_state_orders = [model.get_state_order() for model in models]

        assert all(
            set(state_order) == set(all_state_orders[0])
            for state_order in all_state_orders
        ), "All models must have the same state order"

        self.state_order = all_state_orders[0]

    def get_state_order(self) -> list[str]:
        """Get the state order of the predictor.

        Returns:
            List of state variable names in order
        """
        return self.models[0].get_state_order()

    def get_species_order(self) -> list[str]:
        """Get the species order of the predictor.

        Returns:
            List of species names in order
        """
        return self.models[0].get_species_order()

    def n_parameters(self) -> int:
        """Get the total number of parameters across all models in the ensemble.

        Returns:
            Total number of parameters summed across all models
        """
        return sum(model.n_parameters() for model in self.models)

    def has_hdi(self) -> bool:
        """Check if the ensemble has HDI.

        Returns:
            True if the ensemble has HDI, False otherwise
        """
        return True

    @property
    def has_uncertainty(self) -> bool:
        """Check if the ensemble has uncertainty.

        Returns:
            True if the ensemble has uncertainty, False otherwise
        """
        return True

    @overload
    def predict(
        self,
        dataset: Dataset,
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
        solver: Optional[Type] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
        *,
        return_individual: Literal[True],
    ) -> Tuple[jax.Array, jax.Array, jax.Array]: ...

    @overload
    def predict(
        self,
        dataset: Dataset,
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
        solver: Optional[Type] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
        *,
        return_individual: Literal[False] = False,
    ) -> Dataset: ...

    def predict(
        self,
        dataset: Dataset,
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
        solver: Optional[Type] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
        *,
        return_individual: bool = False,
    ) -> Union[Dataset, Tuple[jax.Array, jax.Array, jax.Array]]:
        """Predict model behavior using the ensemble of models.

        This method evaluates predictions from all models in the ensemble and
        aggregates them either by taking the mean or by computing uncertainty bounds.

        Args:
            dataset: Dataset containing initial conditions for prediction
            config: Optional simulation configuration parameters
            n_steps: Number of time steps for the simulation
            use_times: Whether to use the time points from the dataset
            hdi: Optional HDI option for uncertainty quantification:
                - "lower": 2.5th percentile (lower bound of 95% HDI)
                - "upper": 97.5th percentile (upper bound of 95% HDI)
                - "lower_50": 25th percentile (lower bound of 50% HDI)
                - "upper_50": 75th percentile (upper bound of 50% HDI)
                - None: mean predictions
            solver: Optional ODE solver to use for all models
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            dt0: Initial time step for ODE solver
            return_individual: Whether to return individual predictions for each model

        Returns:
            A Dataset object containing the aggregated prediction results
        """
        # Extract arrays from dataset
        times, y0s = self._extract_prediction_arrays(
            dataset, config, n_steps, use_times
        )

        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            model_predictions = self._predict_with_model(
                model, times, y0s, solver, rtol, atol, dt0
            )
            all_predictions.append(model_predictions)

        # Stack predictions into array: (n_models, n_measurements, n_timepoints, n_states)
        all_predictions = jnp.stack(all_predictions, axis=0)
        y0s_array: jax.Array = jnp.asarray(y0s)  # type: ignore

        if return_individual:
            return all_predictions, times, y0s_array
        else:
            predictions = self._aggregate_predictions(all_predictions, hdi)
            return Dataset.from_jax_arrays(
                state_order=self.state_order,
                data=predictions,
                time=times,
                y0s=y0s_array,
            )

    def rates(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
        return_individual: bool = False,
    ) -> jax.Array:
        """Get the rates of the ensemble predictor.

        Aggregates rates from all models in the ensemble by taking the mean.
        This method is used by the Surrogate interface for MCMC sampling.

        Args:
            t: Time points array with shape (...,)
            y: State array with shape (..., n_states)
            constants: Optional constants array (not used by neural models)

        Returns:
            Mean rates across all models with same shape as y
        """
        # Validate input (inherited from Surrogate)
        t, y, _ = self._validate_rate_input(t, y, constants)

        # Get rates from all models
        all_rates = []
        for model in self.models:
            model_rates = model.rates(t, y, constants)
            all_rates.append(model_rates)

        # Stack and return mean rates across models
        all_rates = jnp.stack(all_rates, axis=0)

        if return_individual:
            return all_rates
        else:
            return jnp.mean(all_rates, axis=0)

    def predict_rates(
        self,
        dataset: Dataset,
        return_individual: bool = False,
    ) -> jax.Array:
        """Predict rates using the ensemble of models.

        This method extracts all time-state pairs from the dataset and computes
        the ensemble-averaged rates for each pair. Useful for analyzing model
        behavior across the entire dataset.

        Args:
            dataset: Dataset containing time series data

        Returns:
            Mean rates across all models, reshaped to (dataset_size * time_size, n_states)
        """
        # Extract data following NeuralBase pattern
        data, times, _ = dataset.to_jax_arrays(self.state_order)
        dataset_size, time_size, _ = data.shape
        ins = data.reshape(dataset_size * time_size, -1)
        times_flat = times.ravel()

        # Get rates from ensemble (which aggregates internally)
        if return_individual:
            return self.rates(times_flat, ins, None, return_individual=True)
        else:
            rates = self.rates(times_flat, ins, None)
            return rates.reshape(dataset_size * time_size, -1)

    def rate_uncertainty(self, dataset: Dataset) -> jax.Array:
        """Get the standard deviation of the rates of the ensemble.

        Args:
            dataset: Dataset containing initial conditions for prediction

        Returns:
            Variance of the rates
        """
        rates = self.predict_rates(dataset, return_individual=True)
        return jnp.std(rates, axis=0)

    def rate_sigma(self, dataset: Dataset) -> jax.Array:
        """Get the standard deviation of the rates of the ensemble.

        Args:
            dataset: Dataset containing initial conditions for prediction

        Returns:
            Standard deviation of the rates
        """
        prediction = self.predict(dataset, use_times=True)
        predicted_rates = self.predict_rates(prediction).reshape(
            -1,
            len(self.state_order),
        )
        dataset_rates = self.predict_rates(dataset).reshape(
            -1,
            len(self.state_order),
        )

        return jnp.abs(predicted_rates - dataset_rates).mean(axis=0)

    # =====================
    # Exporters and Importers
    # =====================

    def save_to_eqx(
        self,
        path: Union[Path, str] = ".",
        name: str = "ensemble",
    ) -> None:
        """Save the ensemble to a zip archive containing individual model files and a manifest.

        Each model in the ensemble is serialized as a separate file named {index}.eqx within
        the zip archive. A manifest.json file maps each filepath to its model's class name
        and module path for reloading.

        Args:
            path: Path to the directory to save the zip archive
            name: Name of the zip archive (without .zip extension)
        """
        if isinstance(path, str):
            path = Path(path)

        # Ensure name has .zip extension
        if name.endswith(".zip"):
            name = name.rstrip(".zip")
        elif name.endswith(".eqx"):
            name = name.rstrip(".eqx")

        # Create full path to zip file
        zip_path = os.path.join(path, name + ".zip")

        # Create manifest dictionary
        manifest = {}

        # Create zip archive and serialize each model
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for index, model in enumerate(self.models):
                # Get model class information
                class_name = model.__class__.__name__
                module_path = model.__class__.__module__

                # Create filename for this model
                filename = f"{index}.eqx"

                # Serialize model to BytesIO buffer
                buffer = io.BytesIO()
                model._serialize(buffer)
                buffer.seek(0)

                # Add model file to zip
                zip_file.writestr(filename, buffer.read())

                # Add entry to manifest
                manifest[filename] = {
                    "class_name": class_name,
                    "module": module_path,
                }

            # Add manifest.json to zip
            manifest_json = json.dumps(manifest, indent=2)
            zip_file.writestr("manifest.json", manifest_json)

        rich.print(f"🔗 Ensemble saved to {zip_path}")

    @classmethod
    def from_eqx(cls, path: Union[Path, str]) -> "NeuralODEEnsemble":
        """Load an ensemble from an eqx file.

        Args:
            path: Path to the eqx file
        """
        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f"Path does not exist: {path}"
        assert path.suffix == ".zip", f"Expected .zip file, got: {path.suffix}"

        with zipfile.ZipFile(path, "r") as zip_file:
            # Check that manifest.json exists
            assert "manifest.json" in zip_file.namelist(), (
                "manifest.json not found in zip file"
            )

            manifest = json.load(zip_file.open("manifest.json"))

            # Check that manifest has expected structure
            assert isinstance(manifest, dict), "Manifest must be a dictionary"

            models = []
            for filename, model_info in manifest.items():
                # Skip manifest.json entry
                if filename == "manifest.json":
                    continue

                assert isinstance(model_info, dict), (
                    f"Model info for {filename} must be a dictionary"
                )
                assert "module" in model_info, (
                    f"Missing 'module' key in model info for {filename}"
                )
                assert "class_name" in model_info, (
                    f"Missing 'class_name' key in model info for {filename}"
                )
                assert filename in zip_file.namelist(), (
                    f"Model file {filename} not found in zip archive"
                )

                module_path = model_info["module"]
                class_name = model_info["class_name"]

                assert isinstance(module_path, str), (
                    f"Module path must be string, got {type(module_path)}"
                )
                assert isinstance(class_name, str), (
                    f"Class name must be string, got {type(class_name)}"
                )

                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    raise ImportError(f"Failed to import module {module_path}: {e}")

                assert hasattr(module, class_name), (
                    f"Class {class_name} not found in module {module_path}"
                )
                model_class = getattr(module, class_name)

                assert issubclass(model_class, NeuralBase), (
                    f"Class {class_name} is not a subclass of NeuralBase"
                )

                model = model_class._deserialize(io.BytesIO(zip_file.read(filename)))
                models.append(model)

            assert len(models) > 0, "No models found in ensemble file"
            return cls(models)

    # =====================
    # Private Helper Methods
    # =====================

    def _predict_with_model(
        self,
        model: NeuralBase,
        times: jax.Array,
        y0s: jax.Array,
        solver: Optional[Type] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
    ) -> jax.Array:
        """Predict with a single model, vectorized over measurements.

        This helper method runs predictions for a single model across all
        measurements in the dataset using JAX's vmap for efficiency.

        Args:
            model: The neural model to use for prediction
            times: Time points array with shape (n_measurements, n_timepoints)
            y0s: Initial conditions array with shape (n_measurements, n_states)
            solver: Optional ODE solver to use
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            dt0: Initial time step for ODE solver

        Returns:
            Predictions array with shape (n_measurements, n_timepoints, n_states)
        """
        return jax.vmap(
            lambda ts, y0: model(ts, y0, solver=solver, rtol=rtol, atol=atol, dt0=dt0),
            in_axes=(0, 0),
        )(times, y0s)

    def _extract_prediction_arrays(
        self,
        dataset: Dataset,
        config: Optional["SimulationConfig"],
        n_steps: int,
        use_times: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Extract time and initial condition arrays from dataset.

        This helper method processes the dataset and configuration to extract
        the appropriate time points and initial conditions for prediction.

        Args:
            dataset: Dataset containing initial conditions and possibly time series
            config: Optional simulation configuration
            n_steps: Number of time steps for simulation
            use_times: Whether to use time points from dataset

        Returns:
            Tuple of (times, y0s) arrays where:
            - times: Time points array with shape (n_measurements, n_timepoints) or (n_timepoints,)
            - y0s: Initial conditions array with shape (n_measurements, n_states)
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
            y0s = dataset.to_y0_matrix(self.state_order)
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore
        else:
            _, times, y0s = dataset.to_jax_arrays(
                self.state_order,
                inits_to_array=True,
            )

        if config:
            times = jnp.linspace(config.t0, config.t1, config.nsteps).T  # type: ignore

        return times, y0s  # type: ignore

    def _aggregate_predictions(
        self,
        all_predictions: jax.Array,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
    ) -> jax.Array:
        """Aggregate ensemble predictions.

        This helper method combines predictions from all models in the ensemble
        either by computing the mean or by extracting specific percentiles for
        uncertainty quantification.

        Args:
            all_predictions: Array of predictions from all models
                with shape (n_models, n_measurements, n_timepoints, n_states)
            hdi: Optional HDI option for uncertainty quantification:
                - "lower": 2.5th percentile (lower bound of 95% HDI)
                - "upper": 97.5th percentile (upper bound of 95% HDI)
                - "lower_50": 25th percentile (lower bound of 50% HDI)
                - "upper_50": 75th percentile (upper bound of 50% HDI)
                - None: mean predictions

        Returns:
            Aggregated predictions array with shape (n_measurements, n_timepoints, n_states)

        Raises:
            ValueError: If an unknown HDI option is provided
        """
        if hdi is not None:
            # Compute percentiles for HDI
            if hdi == "lower":
                return jnp.percentile(all_predictions, 2.5, axis=0)
            elif hdi == "upper":
                return jnp.percentile(all_predictions, 97.5, axis=0)
            elif hdi == "lower_50":
                return jnp.percentile(all_predictions, 25.0, axis=0)
            elif hdi == "upper_50":
                return jnp.percentile(all_predictions, 75.0, axis=0)
            else:
                raise ValueError(f"Unknown HDI option: {hdi}")
        else:
            # Return mean predictions
            return jnp.mean(all_predictions, axis=0)
