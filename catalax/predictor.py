from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from catalax.model.model import SimulationConfig
    from catalax.dataset import Dataset


class Predictor(abc.ABC):
    """
    Abstract base class for predictors in the catalax framework.

    All predictors must implement the predict method which takes a dataset
    and returns predictions based on the model.
    """

    @abc.abstractmethod
    def predict(
        self,
        dataset: Dataset,
        config: Optional[SimulationConfig] = None,
        n_steps: int = 100,
    ) -> Dataset:
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
