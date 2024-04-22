import jax
import jax.numpy as jnp
import numpy as np
import warnings
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Dict, Optional, List, Tuple, Union
from uuid import uuid4
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_serializer,
    field_validator,
)

from pydantic.config import ConfigDict
from pydantic.fields import computed_field


class Measurement(BaseModel):
    """A class to represent a measurement.

    The class is meant to contain a specific set of data points and time points.
    Once passed to Catalax functions, the data and time points are converted
    to jax arrays for further processing.

    Example:

        # Create a measurement object

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

        # Adding data to the measurement
        # The data can be added to the measurement using the add_data method.

        >>> measurement.add_data("C", jnp.array([7, 8, 9]))

        # You can also use pandas dataframes to create measurements

        >>> df = pd.DataFrame({
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
        id (str): The unique identifier of the measurement.
        data (Dict[str, jax.Array]): The data of the measurement.
        time (jax.Array): The time points of the measurement.

    Methods:
        add_data(species, data): Add data to the measurement.
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
        description="""
        The initial concentrations of the measurement.
        The keys are the species names and the values are the concentrations.
        """,
    )

    data: Dict[str, Union[List[float], jax.Array, np.ndarray]] = Field(
        default_factory=dict,
        description="""
        The data of the measurement.
        The keys are the species names and the values are the data points.""",
    )

    time: Union[List[float], jax.Array, np.ndarray] = Field(
        ..., description="The time points of the measurement."
    )

    # ! Computed fields
    @computed_field(return_type=List[str])
    @property
    def species(self):
        return [sp for sp in self.data.keys()]

    # ! Validators
    @field_validator("time")
    @classmethod
    def _convert_time_into_jax_array(cls, time):
        """Converts either list of floats or numpy arrays into JAX arrays."""

        if isinstance(time, np.ndarray) or isinstance(time, list):
            return jnp.array(time)
        else:
            return time

    @field_validator("data")
    @classmethod
    def _convert_data_into_jax_array(cls, data):
        """Converts either list of floats or numpy arrays into JAX arrays."""

        for species, array in data.items():
            if isinstance(array, np.ndarray) or isinstance(array, list):
                data[species] = jnp.array(array)

        return data

    @model_validator(mode="after")
    def _check_data_time_length_match(self):
        assert all(
            len(data) == len(self.time) for data in self.data.values()
        ), "The data and time arrays must have the same length."

        return self

    @model_validator(mode="after")
    def _check_species_consistency(self):
        """Checks whether the data species passed upon initialization are consistent with inits"""

        species_diff = [sp for sp in self.data if sp not in self.initial_conditions]

        if species_diff:
            raise ValueError(
                f"Data and initial concentration species are inconsistent: {species_diff} is not used"
            )

        return self

    # ! Serializers

    @field_serializer("time")
    def serialize_time(self, value):
        return value.tolist()

    @field_serializer("data")
    def serialize_data(self, value):
        return {key: value.tolist() for key, value in value.items()}

    # ! Adders

    def add_data(
        self,
        species: str,
        data: Union[jax.Array, np.ndarray, List[float]],
        initial_concentration: float,
    ) -> None:
        """Add data to the measurement.

        Args:
            species (str): The species name.
            data (jax.Array): The data points.
            initial_concentration (float): The initial concentration.
        """

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = jnp.array(data)

        assert data.shape[0] == self.time.shape[0], "The data and time arrays must have the same length."  # type: ignore
        self.data[species] = data
        self.initial_conditions[species] = initial_concentration

    # ! Exporters

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the measurement to a pandas DataFrame."""

        data = self.data.copy()
        data["time"] = self.time
        data["measurementId"] = [self.id] * self.time.shape[0]  # type: ignore

        return pd.DataFrame(data)

    # ! Importers
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        initial_conditions: Dict[str, float],
        **kwargs,
    ):
        """Create a Measurement object from a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to extract the data from.
            initial_conditions (Dict[str, float]): Mapping of the initial concentrations (species: value)
            measurementId (Optional[str]): ID of the measurement, if given. Defaults to None and thus UUID4
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
                f"Initial concentrations are incosistent with the given DataFrame. Missing {species_diff}"
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

    def to_jax_arrays(
        self,
        species_order: List[str],
        inits_to_array: bool = False,
    ) -> Tuple[jax.Array, jax.Array, Union[jax.Array, Dict[str, float]]]:
        """Convert the measurement data to a JAX array.

        Arranges the data in the order of the species_order list.

        - If inits_to_array is True, the initial conditions are also converted to a JAX array.
        - If inits_to_array is False, the initial conditions are returned as a dictionary.
        - Shape of the data array: (n_time_points, n_species)
        - Shape of the time array: (n_time_points)
        - Shape of the initial conditions array: (n_species)

        Args:
            species_order (List[str]): The order of the species in the array.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: The data, time, and initial conditions.

        """

        unused_species = [sp for sp in self.data.keys() if sp not in species_order]
        missing_species = [sp for sp in species_order if sp not in self.data.keys()]

        if unused_species:
            warnings.warn(f"Species {unused_species} are not used in the model.")

        if missing_species:
            raise ValueError(
                f"The measurement species are inconsistent with the dataset species. Missing {missing_species} in measurement {self.id}"
            )

        data = jnp.array([self.data[sp] for sp in species_order]).swapaxes(0, 1)
        time = jnp.array(self.time)

        if inits_to_array:
            inits = jnp.array([self.initial_conditions[sp] for sp in species_order])
        else:
            inits = self.initial_conditions

        return data, time, inits

    def to_jsonl(self):
        """Converts this measurement object to the JSON Lines format"""

        indices = range(len(self.time))
        lines = []

        for index in indices:
            data = {sp: float(self.data[sp][index]) for sp in self.species}

            data["time"] = float(self.time[index])

            lines.append(json.dumps(data))

        return "\n".join(lines)

    def plot(self, show=True, ax=None):
        """Plot the measurement data.

        Args:
            show (bool): Whether to show the plot. Defaults to True.

        """

        is_subplot = ax is not None
        color_iter = iter(mcolors.TABLEAU_COLORS)
        init_title = " ".join(
            [f"{key}: ${value:.2f}$" for key, value in self.initial_conditions.items()]
        )

        if ax is None:
            fig, ax = plt.subplots()

        for species in self.initial_conditions.keys():
            ax.plot(
                self.time,
                self.data[species],
                "o",
                label=f"{species}",
            )

        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Concentration", fontsize=12)
        ax.set_title(init_title, fontsize=12)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if show and not is_subplot:
            plt.show()

    # ! Data Augmentation
    def augment(
        self,
        sigma: float,
        seed: int,
    ) -> "Measurement":
        """Augment the data by adding Gaussian noise.

        Args:
            sigma (float): The standard deviation of the Gaussian noise.
            seed (int): The seed for the random number generator.

        Returns:
            Measurement: The augmented measurement.
        """

        data = {
            sp: self._jitter_data(self.data[sp], sigma, seed)  # type: ignore
            for sp in self.data.keys()
        }

        return Measurement(
            data=data,  # type: ignore
            time=self.time,
            initial_conditions=self.initial_conditions,
            id=self.id,
        )

    @staticmethod
    def _jitter_data(x: jax.Array, sigma: float, seed: int) -> jax.Array:
        """Jitters the data by adding Gaussian noise.

        Args:
            x (jax.Array): The data to be jittered.
            sigma (float): The standard deviation of the Gaussian noise.
            seed (int): The seed for the random number generator.

        Returns:
            jax.Array: The jittered data.
        """

        rng = np.random.default_rng(seed)

        return jnp.array(x + jnp.array(rng.normal(loc=0.0, scale=sigma, size=x.shape)))
