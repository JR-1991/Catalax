from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from catalax.dataset import Dataset


class Surrogate(abc.ABC):
    """
    Abstract base class for predictors in the catalax framework.

    All predictors must implement the predict method which takes a dataset
    and returns predictions based on the model.
    """

    @abc.abstractmethod
    def rates(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Get the rates of the predictor.
        """
        pass

    @abc.abstractmethod
    def predict_rates(
        self,
        dataset: Dataset,
    ) -> jax.Array:
        """
        Make predictions using the predictor.

        Args:
            dataset: Dataset containing initial conditions
            config: Optional simulation configuration parameters
            parameters: Optional array of model parameters
            **kwargs: Additional keyword arguments specific to the predictor implementation

        Returns:
            Dataset containing the predictions
        """
        pass

    def _validate_rate_input(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Ensure arrays are 2D
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Validate shapes
        assert (
            t.shape[0] == y.shape[0]
        ), "Time and state must have the same number of rows"

        if constants is not None:
            constants = constants.reshape(-1, 1) if constants.ndim == 1 else constants
            assert (
                constants.shape[0] == y.shape[0]
            ), "Constants must have the same number of rows as time and state"
        else:
            constants = jnp.array([])

        return t, y, constants
