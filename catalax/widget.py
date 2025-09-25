from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional

import ipywidgets as widgets
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from catalax.dataset.measurement import Measurement

if TYPE_CHECKING:
    from catalax.dataset import Dataset
    from catalax.model.model import Model
    from catalax.predictor import Predictor

# Constants for widget styling and configuration
DEFAULT_PLOT_STYLE = {
    "figsize": (12, 6),
    "linewidth": 2.5,
    "alpha": 0.8,
    "grid": True,
    "grid_alpha": 0.3,
}

DEFAULT_INITIAL_VALUE = 10.0
DEFAULT_TIME_POINTS = 100
MIN_TIME_POINTS = 50
MAX_TIME_POINTS = 1000
CONTROLS_PER_ROW = 4  # Number of controls per row

WIDGET_LAYOUTS = {
    "text_input": widgets.Layout(width="100px"),
    "label": widgets.Layout(width="100px"),
    "control_item": widgets.Layout(
        justify_content="flex-start",
        align_items="center",
        margin="5px 1px",
        width="150px",
    ),
    "control_row": widgets.Layout(
        justify_content="flex-start", align_items="center", margin="5px 0px"
    ),
    "controls_area": widgets.Layout(
        border="2px solid #dee2e6",
        border_radius="10px",
        padding="20px",
        margin="10px 0px",
        background_color="#ffffff",
    ),
    "main_widget": widgets.Layout(
        border="1px solid #ced4da",
        border_radius="15px",
        padding="15px",
        background_color="#f8f9fa",
    ),
}

PLOT_AESTHETICS = {
    "xlabel_fontsize": 12,
    "ylabel_fontsize": 12,
    "legend_fontsize": 10,
    "background_color": "#f8f9fa",
}


class PredictionWidget:
    """
    A class to encapsulate the interactive prediction widget functionality.

    This class separates concerns and makes the widget more maintainable
    by organizing related functionality into methods.
    """

    def __init__(
        self,
        model: Model,
        predictor: Predictor,
        t0: float = 0.0,
        t1: float = 100.0,
        ref_measurement: Optional[Measurement] = None,
    ):
        """
        Initialize the interactive prediction widget.

        Args:
            model: The model containing state information
            predictor: The predictor for generating predictions
            t0: Start time for predictions
            t1: End time for predictions
            ref_measurement: Reference measurement for initial conditions
        """
        self.model = model
        self.predictor = predictor
        self.t0 = t0
        self.t1 = t1
        self.ref_measurement = ref_measurement
        self.plot_style = DEFAULT_PLOT_STYLE

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)

        # Initialize widget components
        self._setup_state_info()
        self._setup_default_values()
        self._create_controls()
        self._setup_output()

    def _setup_state_info(self) -> None:
        """Extract state information from the model."""
        self.state_order = self.predictor.get_state_order()
        self.state_names = self._get_state_display_names()

    def _get_state_display_names(self) -> List[str]:
        """
        Get display names for state, preferring name over symbol.

        Returns:
            List of state display names
        """
        names = []
        for state_symbol in self.state_order:
            state_obj = self.model.states[state_symbol]
            name = state_obj.name if state_obj.name is not None else state_symbol
            names.append(name)
        return names

    def _setup_default_values(self) -> None:
        """Setup default values for initial conditions."""
        if self.ref_measurement is not None:
            self.default_values = self.ref_measurement
        else:
            self.default_values = {sp: DEFAULT_INITIAL_VALUE for sp in self.state_order}

    def _get_initial_value(self, state: str) -> float:
        """
        Get initial value for a state from default values.

        Args:
            state: State symbol

        Returns:
            Initial concentration value
        """
        if isinstance(self.default_values, Measurement):
            return self.default_values.initial_conditions.get(
                state, DEFAULT_INITIAL_VALUE
            )
        else:
            return self.default_values.get(state, DEFAULT_INITIAL_VALUE)

    def _create_controls(self) -> None:
        """Create all widget controls."""
        self.controls = {}

        # Create state controls
        for state_symbol, name in zip(self.state_order, self.state_names):
            initial_val = self._get_initial_value(state_symbol)

            text_input = widgets.FloatText(
                value=initial_val,
                description="",
                style={"description_width": "0px"},
                layout=WIDGET_LAYOUTS["text_input"],
            )

            label = widgets.HTML(
                value=f"<b>{name}:</b>", layout=WIDGET_LAYOUTS["label"]
            )

            self.controls[state_symbol] = {"text": text_input, "label": label}

        # Create time control
        self.t1_input = widgets.FloatText(
            value=self.t1,
            description="",
            style={"description_width": "0px"},
            layout=WIDGET_LAYOUTS["text_input"],
        )

        self.t1_label = widgets.HTML(
            value="<b>End Time (t1):</b>", layout=WIDGET_LAYOUTS["label"]
        )

    def _setup_output(self) -> None:
        """Setup the output widget for plots."""
        self.output = widgets.Output()

    def _create_prediction_dataset(self) -> Dataset:
        """
        Create a dataset with current input values for prediction.

        Returns:
            Dataset configured for prediction

        Raises:
            ValueError: If input values are invalid
        """

        from catalax.dataset.dataset import Dataset

        try:
            prediction_dataset = Dataset.from_model(self.model)

            # Build initial conditions from text input values
            initial_conditions = {
                state: self.controls[state]["text"].value for state in self.state_order
            }

            # Validate initial conditions
            for state, value in initial_conditions.items():
                if value < 0:
                    raise ValueError(
                        f"Initial condition for {state} cannot be negative: {value}"
                    )

            prediction_dataset.add_initial(**initial_conditions)

            # Setup time points
            current_t1 = max(self.t1_input.value, self.t0 + 0.1)  # Ensure t1 > t0
            time_points = min(
                max(DEFAULT_TIME_POINTS, MIN_TIME_POINTS), MAX_TIME_POINTS
            )
            prediction_dataset.measurements[0].time = jnp.linspace(
                self.t0, current_t1, time_points
            )

            return prediction_dataset

        except Exception as e:
            raise ValueError(f"Failed to create prediction dataset: {str(e)}")

    def _generate_predictions(self, dataset: Dataset) -> Dataset:
        """
        Generate predictions using the predictor.

        Args:
            dataset: Input dataset for prediction

        Returns:
            Dataset with predictions

        Raises:
            RuntimeError: If prediction fails
        """
        try:
            return self.predictor.predict(dataset)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _create_plot(self, predictions: Dataset) -> None:
        """
        Create and display the prediction plot.

        Args:
            predictions: Dataset containing prediction results
        """
        fig, ax = plt.subplots(figsize=self.plot_style["figsize"])

        # Plot predictions with state names as labels
        for i, state_symbol in enumerate(self.state_order):
            state_name = self.state_names[i]
            time_data = predictions.measurements[0].time
            concentration_data = predictions.measurements[0].data[state_symbol]

            ax.plot(
                time_data,  # type: ignore
                concentration_data,  # type: ignore
                label=state_name,
                linewidth=self.plot_style["linewidth"],
                alpha=self.plot_style["alpha"],
            )

        # Enhance plot aesthetics
        ax.set_xlabel(
            "Time", fontsize=PLOT_AESTHETICS["xlabel_fontsize"], fontweight="bold"
        )
        ax.set_ylabel(
            "Concentration",
            fontsize=PLOT_AESTHETICS["ylabel_fontsize"],
            fontweight="bold",
        )

        if self.plot_style["grid"]:
            ax.grid(True, alpha=self.plot_style["grid_alpha"], linestyle="--")

        # Place legend outside the plot area
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=PLOT_AESTHETICS["legend_fontsize"],
        )

        # Set background color
        ax.set_facecolor(PLOT_AESTHETICS["background_color"])

        plt.tight_layout()
        plt.show()

    def _update_plot(self, change=None) -> None:
        """
        Update the prediction plot based on current input values.

        Args:
            change: Widget change event (unused)
        """
        with self.output:
            clear_output(wait=True)

            try:
                # Create dataset and generate predictions
                prediction_dataset = self._create_prediction_dataset()
                predictions = self._generate_predictions(prediction_dataset)

                # Create and display plot
                self._create_plot(predictions)

            except (ValueError, RuntimeError) as e:
                # Display error message instead of crashing
                print(f"Error: {str(e)}")
                print("Please check your input values and try again.")

    def _create_controls_area(self) -> widgets.VBox:
        """
        Create the controls area containing all controls arranged in rows.

        Returns:
            Controls area widget containing all controls in horizontal rows
        """
        # Create control items for state
        control_items = []
        for state in self.state_order:
            item = widgets.VBox(
                [self.controls[state]["label"], self.controls[state]["text"]],
                layout=WIDGET_LAYOUTS["control_item"],
            )
            control_items.append(item)

        # Add time control item
        time_item = widgets.VBox(
            [self.t1_label, self.t1_input],
            layout=WIDGET_LAYOUTS["control_item"],
        )
        control_items.append(time_item)

        # Create rows with specified number of controls per row
        control_rows = []
        for i in range(0, len(control_items), CONTROLS_PER_ROW):
            row_items = control_items[i : i + CONTROLS_PER_ROW]
            row = widgets.HBox(
                row_items,
                layout=WIDGET_LAYOUTS["control_row"],
            )
            control_rows.append(row)

        # Create controls area
        controls_area = widgets.VBox(
            control_rows,
            layout=WIDGET_LAYOUTS["controls_area"],
        )

        return controls_area

    def _setup_observers(self) -> None:
        """Setup observers for all input widgets."""
        # Set up state input observers
        for state in self.state_order:
            self.controls[state]["text"].observe(self._update_plot, names="value")

        # Set up time input observer
        self.t1_input.observe(self._update_plot, names="value")

    def create_widget(self) -> widgets.Widget:
        """
        Create and return the complete interactive widget.

        Returns:
            Complete interactive widget ready for display
        """
        # Create controls area
        controls_area = self._create_controls_area()

        # Setup observers
        self._setup_observers()

        # Create main layout
        main_content = widgets.VBox(
            [controls_area, self.output],
            layout=widgets.Layout(justify_content="flex-start", align_items="stretch"),
        )

        # Main widget container
        main_widget = widgets.VBox(
            [main_content],
            layout=WIDGET_LAYOUTS["main_widget"],
        )

        # Display the widget and generate initial plot
        display(main_widget)
        self._update_plot()

        return main_widget


def create_interactive_prediction_widget(
    model: Model,
    predictor: Predictor,
    t0: float = 0.0,
    t1: float = 100.0,
    ref_measurement: Optional[Measurement] = None,
) -> widgets.Widget:
    """
    Create an interactive widget for exploring Neural ODE predictions with adjustable initial conditions.

    This function creates a comprehensive interactive interface that allows users to:
    - Adjust initial conditions for each state in the model
    - Modify the end time for predictions
    - View real-time updates of the prediction plot

    Args:
        model: The model containing state information and structure
        predictor: The predictor object for generating predictions (e.g., trained Neural ODE)
        t0: Start time for predictions (default: 0.0)
        t1: End time for predictions (default: 100.0)
        ref_measurement: Reference measurement containing initial conditions to use as defaults.
                        If None, uses default values of 10.0 for all states
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
        widget_creator = PredictionWidget(
            model=model,
            predictor=predictor,
            t0=t0,
            t1=t1,
            ref_measurement=ref_measurement,
        )
        return widget_creator.create_widget()

    except Exception as e:
        raise RuntimeError(f"Failed to create interactive prediction widget: {str(e)}")
