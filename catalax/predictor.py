from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal, Optional

from deprecated import deprecated

from catalax.dataset.measurement import Measurement
from catalax.widget import create_interactive_prediction_widget

if TYPE_CHECKING:
    from catalax.dataset import Dataset
    from catalax.model.model import Model, SimulationConfig


class Predictor(abc.ABC):
    """
    Abstract base class for predictors in the catalax framework.

    All predictors must implement the predict method which takes a dataset
    and returns predictions based on the model.
    """

    def widget(
        self,
        model: Model,
        t0: float = 0.0,
        t1: float = 100.0,
        ref_measurement: Optional[Measurement] = None,
    ):
        """
        Create an interactive widget for exploring Neural ODE predictions with adjustable initial conditions.

        This function creates a comprehensive interactive interface that allows users to:
        - Adjust initial conditions for each species in the model
        - Modify the end time for predictions
        - View real-time updates of the prediction plot
        - Reset all values to defaults with a single click

        Args:
            model: The model containing species information and structure
            predictor: The predictor object for generating predictions (e.g., trained Neural ODE)
            t0: Start time for predictions (default: 0.0)
            t1: End time for predictions (default: 100.0)
            ref_measurement: Reference measurement containing initial conditions to use as defaults.
                            If None, uses default values of 10.0 for all species
            plot_style: Dictionary of matplotlib styling options. Merged with defaults:
                       - figsize: (12, 6)
                       - linewidth: 2.5
                       - alpha: 0.8
                       - grid: True
                       - grid_alpha: 0.3

        Returns:
            Interactive widget with text inputs and plot output. The widget automatically
            displays upon creation and updates plots in real-time as inputs change.

        Raises:
            ValueError: If the model or predictor are invalid
            RuntimeError: If widget creation fails

        Note:
            This function is designed for use in Jupyter notebooks and requires
            ipywidgets and matplotlib for proper functionality.
        """
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            raise ImportError(
                "ipywidgets is required to create an interactive widget. Please install it using `pip install ipywidgets`."
            )

        create_interactive_prediction_widget(
            model=model,
            predictor=self,
            t0=t0,
            t1=t1,
            ref_measurement=ref_measurement,
        )

    @deprecated("This method is deprecated. Use get_state_order instead.")
    @abc.abstractmethod
    def get_species_order(self) -> list[str]:
        """
        Get the species order of the predictor.
        """
        pass

    @abc.abstractmethod
    def get_state_order(self) -> list[str]:
        """
        Get the state order of the predictor.
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        dataset: Dataset,
        config: Optional[SimulationConfig] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[Literal["lower", "upper", "lower_50", "upper_50"]] = None,
    ) -> Dataset:
        """
        Make predictions using the predictor.

        Args:
            dataset: Dataset containing initial conditions
            config: Optional simulation configuration parameters
            n_steps: Number of time steps to simulate
            use_times: Whether to use the time points from the dataset or to simulate at fixed time steps
            **kwargs: Additional keyword arguments specific to the predictor implementation

        Returns:
            Dataset containing the predictions
        """

    @abc.abstractmethod
    def n_parameters(self) -> int:
        """
        Get the number of parameters of the predictor.
        """
        pass
