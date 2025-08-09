from __future__ import annotations

import json
from types import NoneType
import warnings
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyenzyme as pe
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.config import ConfigDict
from pydantic.fields import computed_field

# HDI visualization constants
HDI_95_ALPHA = 0.1
HDI_50_ALPHA = 0.35


class Measurement(BaseModel):
    """A class representing a measurement dataset with species concentrations over time.

    The Measurement class stores time series data for chemical species, including their
    initial concentrations and concentration values at each time point. It provides
    methods for data manipulation, conversion, visualization, and integration with
    other data formats.

    Key capabilities:
    - Store and manage time series concentration data
    - Convert data between different formats (JAX arrays, pandas, dictionaries)
    - Visualize measurement data with optional model comparison
    - Data augmentation with noise for simulation purposes
    - Import/export from common scientific data formats

    Example:
        >>> from catalax import Measurement
        >>> data = {
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6],
        ... }
        >>> initial_conditions = {
        ...     "A": 1.0,
        ...     "B": 4.0,
        ... }
        >>> time = [0, 1, 2]
        >>> measurement = Measurement(
        ...     initial_conditions=initial_conditions,
        ...     data=data,
        ...     time=time,
        ... )

        # Adding more data to the measurement
        >>> measurement.add_data("C", jnp.array([7, 8, 9]), 7.0)

        # Creating from a pandas DataFrame
        >>> df = pd.DataFrame({
        ...     "time": [0, 1, 2],
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6],
        ...     "C": [7, 8, 9],
        ... })
        >>> initial_conditions = {
        ...     "A": 1.0,
        ...     "B": 4.0,
        ...     "C": 7.0,
        ... }
        >>> measurement = Measurement.from_dataframe(
        ...     df=df,
        ...     initial_conditions=initial_conditions,
        ... )

    Attributes:
        id (str): Unique identifier for this measurement.
        name (Optional[str]): Optional name for the measurement.
        description (Optional[str]): Optional description of the measurement.
        initial_conditions (Dict[str, float]): Initial concentrations keyed by species name.
        data (Dict[str, jax.Array]): Time series data keyed by species name.
        time (jax.Array): Time points for the measurement data.
        species (List[str]): List of species names present in the data (computed field).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier of this measurement.",
    )

    name: Optional[str] = Field(
        default=None,
        description="Name of the measurement",
    )

    description: Optional[str] = Field(
        default=None,
        description="Description of the measurement",
    )

    initial_conditions: Dict[str, float] = Field(
        ...,
        description="Initial concentrations mapping species names to their initial values.",
    )

    data: Dict[str, Union[List[float], jax.Array]] = Field(
        default_factory=dict,
        description="Time series data mapping species names to their concentration arrays.",
    )

    time: Optional[Union[List[float], jax.Array]] = Field(
        default=None,
        description="Time points for the measurement data.",
    )

    # Computed fields
    @computed_field(return_type=List[str])
    @property
    def species(self) -> List[str]:
        """Returns the list of species names present in the data."""
        return list(self.data.keys())

    # Validators
    @field_validator("time")
    @classmethod
    def _convert_time_into_jax_array(cls, time):
        """Convert time data into JAX array format if needed."""
        if isinstance(time, (np.ndarray, list)):
            return jnp.array(time)
        return time

    @field_validator("data")
    @classmethod
    def _convert_data_into_jax_array(cls, data):
        """Convert species data into JAX array format if needed."""
        for species, array in data.items():
            if isinstance(array, (np.ndarray, list)):
                data[species] = jnp.array(array)
        return data

    @model_validator(mode="after")
    def _check_data_time_length_match(self):
        """Validate that all data arrays match the time array length."""
        if isinstance(self.time, NoneType) or isinstance(self.data, NoneType):
            return self

        assert all(len(data) == len(self.time) for data in self.data.values()), (
            "The data and time arrays must have the same length."
        )
        return self

    @model_validator(mode="after")
    def _check_species_consistency(self):
        """Validate that all species in data also have initial conditions."""
        species_diff = [sp for sp in self.data if sp not in self.initial_conditions]
        if species_diff:
            raise ValueError(
                f"Data and initial concentration species are inconsistent: {species_diff} not found in initial_conditions"
            )
        return self

    # Serializers
    @field_serializer("time")
    def serialize_time(self, value):
        """Convert JAX array time data to Python list for serialization."""
        return value.tolist() if value is not None else None

    @field_serializer("data")
    def serialize_data(self, value):
        """Convert JAX array species data to Python lists for serialization."""
        return {key: val.tolist() for key, val in value.items()}

    # Data Management Methods
    def add_data(
        self,
        species: str,
        data: Union[jax.Array, np.ndarray, List[float]],
        initial_concentration: float,
    ) -> None:
        """Add or update data for a species in the measurement.

        Args:
            species (str): The species name to add or update.
            data (Union[jax.Array, np.ndarray, List[float]]): Concentration values over time.
            initial_concentration (float): The initial concentration of the species.

        Raises:
            AssertionError: If the provided data length doesn't match the existing time points.
        """
        if isinstance(data, (np.ndarray, list)):
            data = jnp.array(data)

        if self.time is not None:
            assert data.shape[0] == self.time.shape[0], (  # type: ignore
                "The data and time arrays must have the same length."
            )

        self.data[species] = data
        self.initial_conditions[species] = initial_concentration

    def add_initial(self, species: str, initial_concentration: float) -> None:
        """Add or update an initial condition for a species in the measurement.

        Args:
            species (str): The species name to add or update.
            initial_concentration (float): The initial concentration of the species.
        """
        self.initial_conditions[species] = initial_concentration

    def has_data(self) -> bool:
        """Check if the measurement contains any non-empty data.

        Returns:
            bool: True if any species has data points, False otherwise.
        """
        return any(len(data) > 0 for data in self.data.values())

    # Data Export Methods
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the measurement to a pandas DataFrame.

        The resulting DataFrame contains columns for each species, time points,
        and a measurementId column.

        Returns:
            pd.DataFrame: DataFrame with time series data and metadata.
        """
        data = self.data.copy()
        data["time"] = self.time  # type: ignore
        data["measurementId"] = [self.id] * len(self.time)  # type: ignore

        return pd.DataFrame(data)

    def to_dict(self) -> Dict[str, Union[List[float], jax.Array, np.ndarray]]:
        """Convert the measurement to a dictionary with time and species data.

        Returns:
            Dict[str, Union[List[float], jax.Array, np.ndarray]]: Dictionary with time and species data.
        """
        return {
            "time": self.time,  # type: ignore
            **self.data,
        }

    def to_jsonl(self) -> str:
        """Convert the measurement to JSON Lines format.

        Each line contains a JSON object with time and all species values for that time point.

        Returns:
            str: JSON Lines formatted string.
        """
        lines = []
        for index in range(len(self.time)):  # type: ignore
            data = {sp: float(self.data[sp][index]) for sp in self.species}
            data["time"] = float(self.time[index])  # type: ignore
            lines.append(json.dumps(data))

        return "\n".join(lines)

    def to_jax_arrays(
        self,
        species_order: List[str],
        inits_to_array: bool = False,
    ) -> Tuple[jax.Array, jax.Array, Union[jax.Array, Dict[str, float]]]:
        """Convert measurement data to JAX arrays in the specified species order.

        Arranges the data following the provided species order for use in models.

        Args:
            species_order (List[str]): Order of species to arrange data.
            inits_to_array (bool): If True, convert initial conditions to a JAX array.
                                  If False, return initial conditions as a dictionary.

        Returns:
            Tuple containing:
                - data array (shape: n_time_points × n_species)
                - time array (shape: n_time_points)
                - initial conditions (as array or dict)

        Raises:
            ValueError: If any species in species_order is missing from the measurement data.
        """
        unused_species = [sp for sp in self.data.keys() if sp not in species_order]
        missing_species = [sp for sp in species_order if sp not in self.data.keys()]

        if unused_species:
            warnings.warn(f"Species {unused_species} are not used in the model.")

        if missing_species:
            raise ValueError(
                f"The measurement species are inconsistent with the dataset species. "
                f"Missing {missing_species} in measurement {self.id}"
            )

        data = jnp.array([self.data[sp] for sp in species_order]).swapaxes(0, 1)
        time = jnp.array(self.time)

        if inits_to_array:
            inits = jnp.array([self.initial_conditions[sp] for sp in species_order])
        else:
            inits = self.initial_conditions

        return data, time, inits

    def to_y0_array(self, species_order: List[str]) -> jax.Array:
        """Convert initial conditions to a JAX array in the specified species order.

        Args:
            species_order (List[str]): Order of species to arrange initial conditions.

        Returns:
            jax.Array: Initial conditions array.
        """
        return jnp.array([self.initial_conditions[sp] for sp in species_order])

    # Data Import Methods
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        initial_conditions: Dict[str, float],
        **kwargs,
    ) -> "Measurement":
        """Create a Measurement object from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing time series data.
                              Must include a 'time' column.
            initial_conditions (Dict[str, float]): Mapping of species names to initial concentrations.
            **kwargs: Additional arguments to pass to the Measurement constructor.

        Returns:
            Measurement: New Measurement object with data from the DataFrame.

        Raises:
            AssertionError: If the DataFrame doesn't have a 'time' column.
            ValueError: If species in DataFrame don't match those in initial_conditions.
        """
        assert "time" in df.columns, "The DataFrame must have a 'time' column."

        species_diff = [
            sp
            for sp in df.columns
            if sp not in initial_conditions.keys()
            and sp != "time"
            and sp != "measurementId"
        ]

        if species_diff:
            raise ValueError(
                f"Initial concentrations are inconsistent with the given DataFrame. "
                f"Missing {species_diff} in initial_conditions"
            )

        time = jnp.array(df["time"].values)
        data = {
            str(col): jnp.array(df[col].values)
            for col in df.columns
            if col != "time" and col != "measurementId"
        }

        return cls(
            data=data,  # type: ignore
            time=time,
            initial_conditions=initial_conditions,
            **kwargs,
        )

    @classmethod
    def from_enzymeml(cls, measurement: pe.Measurement) -> "Measurement":
        """Create a Measurement object from a pyenzyme Measurement object.

        Args:
            measurement (pe.Measurement): PyEnzyme measurement object.

        Returns:
            Measurement: New Measurement object with data from the PyEnzyme measurement.
        """
        initials = {
            species.species_id: species.initial
            for species in measurement.species_data
            if species.initial is not None
        }

        data = {
            species.species_id: jnp.array(species.data)
            for species in measurement.species_data
            if species.data is not None and len(species.data) > 0
        }

        time = next(
            iter(
                [
                    jnp.array(data.time)
                    for data in measurement.species_data
                    if data.time is not None and len(data.time) > 0
                ]
            )
        )

        if measurement.id is None:
            measurement.id = str(uuid4())

        return cls(
            initial_conditions=initials,
            data=data,  # type: ignore
            time=time,
            id=measurement.id,
        )

    # Visualization Methods
    def _format_title_with_initial_conditions(
        self, initial_conditions: Dict[str, float]
    ) -> Tuple[str, int]:
        """Format initial conditions into a multi-line title with appropriate font size.

        This function creates a formatted title string from initial conditions, automatically
        adding line breaks and adjusting font size based on the number of entries to ensure
        readability and proper display.

        Args:
            initial_conditions (Dict[str, float]): Dictionary mapping species names to
                                                  their initial concentration values.

        Returns:
            Tuple[str, int]: A tuple containing:
                - Formatted title string with appropriate line breaks
                - Recommended font size for the title

        Layout strategy:
            - ≤3 entries: Single line, font size 12
            - 4-6 entries: Two lines, font size 10
            - >6 entries: Three lines, font size 9
        """
        init_conditions_list = [
            f"{key}: ${value:.2f}$" for key, value in initial_conditions.items()
        ]
        num_entries = len(init_conditions_list)

        if num_entries <= 3:
            # Few entries: single line
            title = " ".join(init_conditions_list)
            fontsize = 12
        elif num_entries <= 6:
            # Medium entries: break into 2 lines
            mid_point = num_entries // 2
            line1 = " ".join(init_conditions_list[:mid_point])
            line2 = " ".join(init_conditions_list[mid_point:])
            title = f"{line1}\n{line2}"
            fontsize = 10
        else:
            # Many entries: break into 3 lines
            third = num_entries // 3
            line1 = " ".join(init_conditions_list[:third])
            line2 = " ".join(init_conditions_list[third : 2 * third])
            line3 = " ".join(init_conditions_list[2 * third :])
            title = f"{line1}\n{line2}\n{line3}"
            fontsize = 9

        return title, fontsize

    def plot(
        self,
        show: bool = True,
        ax=None,
        model_data: Optional[Measurement] = None,
        xlim: Optional[Tuple[float, float | None]] = (0, None),
        _lower_95: Optional[Measurement] = None,
        _upper_95: Optional[Measurement] = None,
        _lower_50: Optional[Measurement] = None,
        _upper_50: Optional[Measurement] = None,
        **kwargs,
    ) -> None:
        """Plot the measurement data with optional model fit comparison.

        Args:
            show (bool): Whether to show the plot immediately.
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            model_data (Optional[Measurement]): Model prediction data to overlay on the plot.
            xlim: Optional[Tuple[float, float]]: Limits of the x-axis.
            _lower_95: Optional[Measurement]: Lower 95% HDI data. Internally used for MCMC.
            _upper_95: Optional[Measurement]: Upper 95% HDI data. Internally used for MCMC.
            _lower_50: Optional[Measurement]: Lower 50% HDI data. Internally used for MCMC.
            _upper_50: Optional[Measurement]: Upper 50% HDI data. Internally used for MCMC.
            **kwargs: Additional arguments to pass to plot functions.
        """
        is_subplot = ax is not None
        color_iter = iter(mcolors.TABLEAU_COLORS)

        # Create formatted title with initial conditions
        init_title, title_fontsize = self._format_title_with_initial_conditions(
            self.initial_conditions
        )

        if ax is None:
            _, ax = plt.subplots()

        if model_data:
            sim_meas = model_data.to_dict()

        has_95_hdi = _lower_95 and _upper_95
        has_50_hdi = _lower_50 and _upper_50

        if has_95_hdi:
            assert _lower_95 and _upper_95, "Lower and upper 95% HDI must be provided"
            lower_95_meas = _lower_95.to_dict()
            upper_95_meas = _upper_95.to_dict()

        if has_50_hdi:
            assert _lower_50 and _upper_50, "Lower and upper 50% HDI must be provided"
            lower_50_meas = _lower_50.to_dict()
            upper_50_meas = _upper_50.to_dict()

        for species in self.initial_conditions.keys():
            if self.data.get(species) is None or len(self.data[species]) == 0:
                continue

            color = next(color_iter)
            plt_kwargs = {
                "marker": "o",
                "linestyle": "None",
                "label": f"{species}",
                "c": color,
                **kwargs,
            }

            ax.plot(
                self.time,  # type: ignore
                self.data[species],
                **plt_kwargs,
            )

            if model_data:
                ax.plot(
                    sim_meas["time"],
                    sim_meas[species],
                    "-",
                    label=f"{species} fit",
                    c=color,
                    **kwargs,
                )

            if has_95_hdi:
                ax.fill_between(
                    lower_95_meas["time"],
                    lower_95_meas[species],
                    upper_95_meas[species],
                    color=color,
                    alpha=HDI_95_ALPHA,
                    edgecolor="none",
                )

            if has_50_hdi:
                ax.fill_between(
                    lower_50_meas["time"],
                    lower_50_meas[species],
                    upper_50_meas[species],
                    color=color,
                    alpha=HDI_50_ALPHA,
                    edgecolor="none",
                )

        ax.grid(True, which="both", linestyle="--")
        ax.grid(True, which="minor", alpha=0.3)
        ax.minorticks_on()

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Concentration", fontsize=12)
        ax.set_title(init_title, fontsize=title_fontsize)

        if xlim:
            ax.set_xlim(xlim[0], xlim[1])

        if self.data:
            ymax = max(max(series) for series in self.data.values() if len(series) > 0)
            ax.set_ylim(0, ymax * 1.1)

        # Create custom legend with HDI handles
        handles, labels = ax.get_legend_handles_labels()

        # Add HDI patches to legend if they exist
        if has_95_hdi:
            handles.append(Patch(facecolor="gray", alpha=HDI_95_ALPHA))
            labels.append("95% HDI")
        if has_50_hdi:
            handles.append(Patch(facecolor="gray", alpha=HDI_50_ALPHA))
            labels.append("50% HDI")

        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        if show and not is_subplot:
            plt.show()

    # Data Augmentation Methods
    def augment(
        self,
        sigma: float,
        seed: int,
        multiplicative: bool = False,
    ) -> "Measurement":
        """Create a new Measurement with noise-augmented data.

        Args:
            sigma (float): Standard deviation of the Gaussian noise.
            seed (int): Random seed for reproducibility.
            multiplicative (bool): If True, noise is multiplicative rather than additive.

        Returns:
            Measurement: New Measurement object with noise-augmented data.
        """
        data = {
            sp: self._jitter_data(
                x=self.data[sp],  # type: ignore
                sigma=sigma,
                seed=seed,
                multiplicative=multiplicative,
            )
            for sp in self.data.keys()
        }

        return Measurement(
            data=data,  # type: ignore
            time=self.time,
            initial_conditions=self.initial_conditions,
            id=str(uuid4()),
        )

    @staticmethod
    def _jitter_data(
        x: jax.Array,
        sigma: float,
        seed: int,
        multiplicative: bool = False,
    ) -> jax.Array:
        """Add random noise to data.

        Args:
            x (jax.Array): Original data array.
            sigma (float): Standard deviation of the noise.
            seed (int): Random seed.
            multiplicative (bool): If True, apply multiplicative noise (1 + noise).
                                  If False, apply additive noise.

        Returns:
            jax.Array: Data with added noise.
        """
        key = jax.random.PRNGKey(seed)

        if multiplicative:
            noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype) * sigma + 1.0
            return x * noise

        noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype) * sigma
        return x + noise
