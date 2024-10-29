import json
import warnings
from typing import Dict, Optional, List, Tuple, Union
from uuid import uuid4

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyenzyme as pe
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

    time: Optional[Union[List[float], jax.Array, np.ndarray]] = Field(
        None,
        description="The time points of the measurement.",
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

    def to_y0_array(self, species_order: List[str]) -> jax.Array:
        """Convert the initial conditions to a JAX array.

        Args:
            species_order (List[str]): The order of the species in the array.

        Returns:
            jax.Array: The initial conditions.
        """

        return jnp.array([self.initial_conditions[sp] for sp in species_order])

    def to_jsonl(self):
        """Converts this measurement object to the JSON Lines format"""

        indices = range(len(self.time))
        lines = []

        for index in indices:
            data = {sp: float(self.data[sp][index]) for sp in self.species}

            data["time"] = float(self.time[index])

            lines.append(json.dumps(data))

        return "\n".join(lines)

    def plot(
        self,
        show=True,
        ax=None,
        model: Optional["Model"] = None,
    ):
        """Plot the measurement data.

        Args:
            show (bool): Whether to show the plot. Defaults to True.
            ax (Optional[plt.Axes]): The axes to plot the data on. Defaults to None.
            model (Model): The model to plot the fit of. Defaults to None.
        """

        is_subplot = ax is not None
        color_iter = iter(mcolors.TABLEAU_COLORS)
        init_title = " ".join(
            [f"{key}: ${value:.2f}$" for key, value in self.initial_conditions.items()]
        )

        if ax is None:
            fig, ax = plt.subplots()

        model_data = self._gather_fit(model)

        for species in self.initial_conditions.keys():
            if self.data.get(species) is None or len(self.data[species]) == 0:
                continue

            color = next(color_iter)

            ax.plot(
                self.time,
                self.data[species],
                "o",
                label=f"{species}",
                c=color,
            )

            if model_data:
                ax.plot(
                    model_data["time"],
                    model_data[species],
                    "-",
                    label=f"{species} fit",
                    c=color,
                )

        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Concentration", fontsize=12)
        ax.set_title(init_title, fontsize=12)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if show and not is_subplot:
            plt.show()

    def _gather_fit(
        self,
        model: Optional["Model"],
    ) -> Dict[str, jax.Array] | None:
        if model:
            times, states = model.simulate(
                initial_conditions=[self.initial_conditions],
                t0=self.time.min(),
                t1=self.time.max(),
                nsteps=len(self.time) * 2,
                return_array=True,
                in_axes=None,
            )

            # Organize results in a dictionary
            model_data = {
                species: states[0, :, index]
                for index, species in enumerate(model.get_species_order())
            }

            model_data["time"] = times
        else:
            model_data = None

        return model_data

    # ! Data Augmentation
    def augment(
        self,
        sigma: float,
        seed: int,
        multiplicative: bool = False,
    ) -> "Measurement":
        """Augment the data by adding Gaussian noise.

        Args:
            sigma (float): The standard deviation of the Gaussian noise.
            seed (int): The seed for the random number generator.
            multiplicative (bool): Whether to multiply the signal by the noise.

        Returns:
            Measurement: The augmented measurement.
        """

        data = {
            sp: self._jitter_data(
                x=self.data[sp],
                sigma=sigma,
                seed=seed,
                multiplicative=multiplicative,
            )  # type: ignore
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
        """Jitters the data by adding Gaussian noise.

        Args:
            x (jax.Array): The data to be jittered.
            sigma (float): The standard deviation of the Gaussian noise.
            seed (int): The seed for the random number generator.
            multiplicative (bool): Whether to multiply the signal by the noise

        Returns:
            jax.Array: The jittered data.
        """

        key = jax.random.PRNGKey(seed)

        if multiplicative:
            noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype) * sigma + 1.0
            return x * noise

        noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype) * sigma
        return x + noise

    @classmethod
    def from_enzymeml(cls, measurement: pe.Measurement):
        """Create a Measurement object from a pyenzyme Measurement object."""

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

        return cls(
            initial_conditions=initials,
            data=data,
            time=time,
        )

    def has_data(self):
        """Check if the measurement has data"""
        return any(len(data) > 0 for data in self.data.values())
