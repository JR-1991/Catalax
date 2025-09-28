from __future__ import annotations

import os
import random
import tempfile
import uuid
import warnings
import zipfile
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlcroissant as mlc
import numpy as np
import pandas as pd
import pyenzyme as pe
from deprecated import deprecated
from jax import Array
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from catalax.dataset.metrics import FitMetrics
from catalax.objectives import l1_loss
from catalax.predictor import Predictor

if TYPE_CHECKING:
    from catalax.model.model import HDIOptions, Model
    from catalax.model.simconfig import SimulationConfig

from .croissant import extract_record_set, json_lines_to_dict
from .measurement import Measurement


class DatasetType(Enum):
    """Enumeration of dataset types in the Catalax framework."""

    MEASUREMENT = "measurement"
    SIMULATION = "simulation"
    PREDICTION = "prediction"


class Dataset(BaseModel):
    """A class representing a collection of experimental measurements.

    This class hosts multiple measurements from an experiment and enables
    integration with other Catalax functionalities, including data processing,
    visualization, and model fitting.

    Attributes:
        id: Unique identifier of the dataset
        states: List of chemical states within this dataset
        name: Human-readable name or identifier
        description: Longer text description of the dataset
        measurements: Collection of experimental measurements
        type: Classification of the dataset (measurement, simulation, or prediction)
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier of the dataset.",
    )

    states: List[str] = Field(
        ...,
        description="List of states within this dataset.",
    )

    name: Optional[str] = Field(
        default=None,
        description="Name or ID of the dataset.",
    )

    description: Optional[str] = Field(
        default=None, description="Additional textual information of the dataset."
    )

    measurements: List[Measurement] = Field(
        default_factory=list,
        description="A list of all measurements found in this dataset",
    )

    type: DatasetType = Field(
        default=DatasetType.MEASUREMENT,
        description="The type of the dataset.",
    )

    # =====================
    # Getters
    # =====================

    def get_measurement(self, id: str) -> Measurement:
        """Retrieve a measurement from the dataset by its ID.

        Args:
            id: The unique identifier of the measurement to retrieve

        Returns:
            The measurement object with the specified ID

        Raises:
            ValueError: If no measurement with the given ID exists
        """
        meas = next(
            (meas for meas in self.measurements if meas.id == id),
            None,
        )

        if meas is None:
            raise ValueError(f"Measurement with ID={id} not found.")

        return meas

    @deprecated("This method is deprecated. Use get_observable_states_order instead.")
    def get_observable_species_order(self) -> List[str]:
        """Get the ordered list of species that are observed across measurements."""
        return self.get_observable_states_order()

    def get_observable_states_order(self) -> List[str]:
        """Get the ordered list of states that are observed across measurements.

        This method determines which states have measurement data and ensures
        consistency across different measurements in the dataset.

        Returns:
            Ordered list of observable states names

        Raises:
            ValueError: If no common observable states exist across measurements
        """
        if not self.measurements:
            return []

        # Get observables from the first measurement
        observable_ids = set(self.measurements[0].data.keys())

        # Check if all measurements have the same observables
        for meas in self.measurements[1:]:
            meas_observables = set(meas.data.keys())
            if meas_observables != observable_ids:
                # Find common observables
                common_observables = observable_ids.intersection(meas_observables)
                warnings.warn(
                    f"Inconsistent observables across measurements. "
                    f"First measurement has {observable_ids}, "
                    f"but measurement {meas.id} has {meas_observables}. "
                    f"Using common observables: {common_observables}"
                )

                if not common_observables:
                    raise ValueError(
                        "No common observables found across measurements. "
                        "Please check the data for consistency."
                    )

                observable_ids = common_observables

        # Return sorted observable states
        return sorted(observable_ids)

    def get_observable_indices(self) -> List[int]:
        """Get the indices of observable states within the full states list.

        Returns:
            List of integer indices corresponding to the positions of observable
            states in the full states list
        """
        return [self.states.index(sp) for sp in self.get_observable_states_order()]

    # =====================
    # Data Addition Methods
    # =====================

    def add_initial(
        self,
        time: Array | list[float] | None = None,
        **kwargs,
    ) -> None:
        """Add an empty measurement with only initial conditions.

        Args:
            time: Time points for the initial condition. Useful if you want to predict
                  using neural ODEs.
            **kwargs: Initial condition values as keyword arguments where
                      keys are states names and values are concentrations
        """
        self.measurements.append(
            Measurement(
                time=time,
                initial_conditions=kwargs,
            )
        )

    def add_measurement(self, measurement: Measurement) -> None:
        """Add a complete measurement to the dataset.

        Args:
            measurement: The measurement object to add

        Raises:
            AssertionError: If a measurement with the same ID already exists
            ValueError: If states in the measurement are inconsistent with dataset states
        """
        # Check for duplicate measurement ID
        assert not any(meas.id == measurement.id for meas in self.measurements), (
            f"Measurement with ID={measurement.id} already exists."
        )

        # Check for states consistency
        unused_states = [
            state for state in measurement.data.keys() if state not in self.states
        ]

        if unused_states:
            warnings.warn(f"States {unused_states} are not used in the dataset.")

        self.measurements.append(measurement)

    def add_from_jax_array(
        self,
        state_order: List[str],
        initial_condition: Dict[str, float],
        data: Array,
        time: Array,
    ) -> None:
        """Add a measurement from JAX arrays directly.

        Args:
            state_order: Ordered list of state names matching data array columns
            initial_condition: Dictionary mapping states names to initial concentrations
            data: JAX array of concentration measurements (shape: time_points x states)
            time: JAX array of time points
        """
        measurement = Measurement(
            initial_conditions=initial_condition,
            time=time,
            data={state: data[:, i].squeeze() for i, state in enumerate(state_order)},
        )

        self.add_measurement(measurement)

    def pad(self) -> "Dataset":
        """Return a copy of the dataset with uniform array lengths, padded with NaN.

        This method deep-copies the dataset and ensures that each `Measurement`
        has:
        * A `data` entry for every state in `self.states`.
        * All per-state arrays right-padded to the maximum length with NaN.
        * A `time` array right-padded to the same maximum length with NaN
            (if present). If `time` is `None`, it is left unchanged.

        The maximum length is determined from the longest `time` array among
        all measurements. Measurements with `time=None` are ignored when
        computing this length.

        Returns:
            Dataset: A new dataset instance where all measurements have
            homogeneously shaped 1-D lists for all states and padded `time`
            arrays.

        Raises:
            ValueError: If any measurement's `initial_conditions` do not cover
            all states in `self.states`.
        """
        ds = deepcopy(self)
        required = set(self.states)

        # Get max length from data arrays and time arrays
        max_len = 0
        for meas in ds.measurements:
            # Check time length if time exists
            if meas.time is not None:
                max_len = max(max_len, len(meas.time))

        # Validate that measurements have required initial conditions
        for meas in ds.measurements:
            keys = set(meas.initial_conditions.keys())
            missing = required - keys
            if missing:
                raise ValueError(
                    f"Measurement {meas.id} missing definition of initial condition for states: {sorted(missing)}"
                )

        # Pad missing measurement data with NaNs
        for meas in ds.measurements:
            # Pad time if needed
            if meas.time is not None:
                time_arr = jnp.asarray(meas.time, dtype=float)
                if len(time_arr) < max_len:
                    pad_len = max_len - len(time_arr)
                    # Continue monotonically with +1.0
                    if time_arr.size > 0:
                        start_val = time_arr[-1] + 1.0
                    else:
                        start_val = 0.0
                    pad_vals = jnp.arange(start_val, start_val + pad_len, 1.0)
                    time_arr = jnp.concatenate((time_arr, pad_vals))
                meas.time = time_arr

            # Pad data for each state
            for sid in self.states:
                try:
                    arr = jnp.asarray(meas.data[sid], dtype=float)
                except KeyError:
                    # Species data missing - create full NaN array
                    arr = jnp.full(max_len, jnp.nan)
                    meas.data[sid] = arr
                    continue

                if arr.size < max_len:
                    pad_len = max_len - arr.size
                    arr = jnp.concatenate((arr, jnp.full(pad_len, jnp.nan)))

                meas.data[sid] = arr

        return ds

    # =====================
    # Data Export Methods
    # =====================

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Export dataset as two pandas DataFrames for data and initial conditions.

        Returns:
            Tuple containing:
            - DataFrame with time course data (columns: measurementId, time, states...)
            - DataFrame with initial conditions (columns: measurementId, states...)
        """
        data = pd.concat([meas.to_dataframe() for meas in self.measurements])
        inits = pd.DataFrame(
            {"measurementId": meas.id, **meas.initial_conditions}
            for meas in self.measurements
        )

        return data, inits

    def to_jax_arrays(
        self,
        state_order: List[str],
        inits_to_array: bool = False,
    ) -> tuple[Array, Array, Union[Array, List[Union[Array, Dict[str, float]]]]]:
        """Convert dataset to JAX arrays for computational use.

        Args:
            state_order: Ordered list of state names to ensure consistent array structure
            inits_to_array: Whether to convert initial conditions to a JAX array

        Returns:
            A tuple containing:
            - data: JAX array of shape (n_measurements, n_time_points, n_states)
            - time: JAX array of shape (n_measurements, n_time_points)
            - initial_conditions: Either a JAX array of shape (n_measurements, n_states)
              if inits_to_array=True, or a list of dictionaries mapping states names
              to initial concentrations
        """
        data = []
        time = []
        initial_conditions = []

        for meas in self.measurements:
            data_, time_, inits = meas.to_jax_arrays(
                state_order=state_order,
                inits_to_array=inits_to_array,
            )

            data.append(data_)
            time.append(time_)
            initial_conditions.append(inits)

        if inits_to_array:
            return (
                jnp.stack(data, axis=0),
                jnp.stack(time, axis=0),
                jnp.stack(initial_conditions, axis=0),  # type: ignore
            )
        else:
            return (
                jnp.stack(data, axis=0),
                jnp.stack(time, axis=0),
                initial_conditions,  # type: ignore
            )

    def to_y0_matrix(self, state_order: List[str]) -> Array:
        """Create a matrix of initial conditions for all measurements.

        Args:
            state_order: Ordered list of state names for the matrix columns

        Returns:
            JAX array of shape (n_measurements, n_states) containing initial
            conditions for all measurements
        """
        inits = []

        for meas in self.measurements:
            inits.append(meas.to_y0_array(state_order=state_order))

        return jnp.stack(inits, axis=0)

    def has_data(self) -> bool:
        """Check if the dataset has any data.

        Returns:
            True if the dataset has any data, False otherwise
        """
        return any(meas.has_data() for meas in self.measurements)

    def to_croissant(
        self,
        dirpath: str,
        license: str = "CC BY-SA 4.0",
        version: str = "1.0.0",
        name: Optional[str] = None,
        cite_as: Optional[str] = None,
        url: Optional[str] = None,
        date_published: datetime = datetime.now(),
    ) -> None:
        """Export dataset to a Croissant archive.

        The Croissant format is a JSON-LD standard for describing datasets that enables
        interoperability between different tools and platforms. This method exports
        the dataset to a directory structure compatible with the Croissant specification.

        Args:
            dirpath: Directory path where the Croissant archive will be saved
            license: License identifier for the dataset
            version: Version string for the dataset
            name: Optional custom name for the dataset archive
            cite_as: Citation information for the dataset
            url: URL where the dataset is available
            date_published: Publication date for the dataset
        """
        from .croissant import dataset_to_croissant

        os.makedirs(dirpath, exist_ok=True)

        if name is None and self.name is None:
            name = self.id
        elif name is None and self.name is not None:
            name = self.name.replace(" ", "_")

        dataset_to_croissant(
            dataset=self,
            dirpath=dirpath,
            name=name,
            license=license,
            version=version,
            cite_as=cite_as,
            url=url,
            date_published=date_published,
        )

    # =====================
    # Data Import Methods
    # =====================

    @classmethod
    def from_enzymeml(
        cls,
        enzmldoc: pe.EnzymeMLDocument,
    ) -> Dataset:
        """Create a dataset from an EnzymeML document.

        Args:
            enzmldoc: EnzymeML document containing experimental data

        Returns:
            A new Dataset object with measurements extracted from the EnzymeML document
        """

        measurements = [
            Measurement.from_enzymeml(meas)
            for meas in enzmldoc.measurements
            if any(sp.data is not None and len(sp.data) > 0 for sp in meas.species_data)
        ]

        small_molecules = [sp.id for sp in enzmldoc.small_molecules]
        proteins = [sp.id for sp in enzmldoc.proteins]
        complexes = [sp.id for sp in enzmldoc.complexes]
        all_states = small_molecules + proteins + complexes

        dataset = cls(
            id=enzmldoc.name,
            name=enzmldoc.name,
            states=all_states,
            measurements=measurements,
        )

        return dataset

    @classmethod
    def from_dataframe(
        cls,
        name: str,
        data: pd.DataFrame,
        inits: pd.DataFrame,
        meas_id: Optional[str] = None,
        description: Optional[str] = "",
    ) -> "Dataset":
        """Create a dataset from pandas DataFrames.

        Args:
            name: Name for the dataset
            data: DataFrame containing time course data
                  Expected columns: measurementId, time, state1, state2...
            inits: DataFrame containing initial conditions
                   Expected columns: measurementId, state1, state2...
            meas_id: Optional custom ID for the dataset
            description: Optional description for the dataset

        Returns:
            A new Dataset object populated with measurements from the DataFrames

        Raises:
            AssertionError: If required columns are missing from the DataFrames
            ValueError: If measurement IDs are inconsistent between data and inits
        """
        # Validate required columns
        assert "measurementId" in data.columns, (
            "Missing column in data table: 'measurementId'"
        )
        assert "time" in data.columns, "Missing column in data table: 'time'"
        assert "measurementId" in inits.columns, (
            "Missing column in inits table: 'measurementId'"
        )

        if meas_id is None:
            meas_id = str(uuid.uuid4())

        # Check ID consistency between data and initial conditions
        data_ids = set(data["measurementId"])
        init_ids = set(inits["measurementId"])
        id_diff = set(
            list(data_ids.difference(init_ids)) + list(init_ids.difference(data_ids))
        )

        if id_diff:
            missing_in_data = [
                id_ for id_ in id_diff if id_ not in data["measurementId"].to_list()
            ]
            missing_in_inits = [
                id_ for id_ in id_diff if id_ not in inits["measurementId"].to_list()
            ]

            raise ValueError(
                f"Measurement IDs are inconsistent between tables:\n"
                f"- Missing in data: {missing_in_data}\n"
                f"- Missing in initial conditions: {missing_in_inits}"
            )

        # Extract states names (all columns except measurementId)
        states = [sp for sp in inits.columns if sp != "measurementId"]

        # Create the dataset
        dataset = cls(
            states=states,
            name=name,
            description=description,
            id=meas_id,
        )

        # Add measurements for each unique ID
        for meas_id in data_ids:
            sub_inits = inits[inits.measurementId == meas_id].to_dict("records")[0]
            sub_data = data[data.measurementId == meas_id]
            meas_id = sub_inits.pop("measurementId")

            dataset.add_measurement(
                measurement=Measurement.from_dataframe(
                    df=sub_data,
                    initial_conditions=sub_inits,  # type: ignore
                    id=meas_id,
                )
            )

        return dataset

    @classmethod
    def from_model(cls, model: "Model") -> "Dataset":
        """Create an empty dataset with states from a model.

        Args:
            model: Model object containing states information

        Returns:
            A new Dataset object with states from the model but no measurements

        Raises:
            AssertionError: If the provided object is not a Model instance
        """
        from ..model import Model

        assert isinstance(model, Model), "Expected a Model object."

        return cls(
            id=model.name,
            name=model.name,
            states=model.get_state_order(modeled=False),
        )

    @classmethod
    def from_croissant(cls, path: str) -> "Dataset":
        """Create a dataset from a Croissant archive.

        Args:
            path: Path to the Croissant archive file

        Returns:
            A new Dataset object populated from the Croissant archive

        Raises:
            AssertionError: If measurements lack corresponding initial conditions
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the archive
            with zipfile.ZipFile(path, "r") as f:
                f.extractall(tmpdir)

            # Load the Croissant dataset
            croissant_path = os.path.join(tmpdir, "croissant.json")
            croissant_ds = mlc.Dataset(jsonld=croissant_path)

            # Extract record sets for initial conditions and measurements
            init_rs = extract_record_set(
                croissant_ds, lambda meas_id: "/inits" in meas_id
            )
            meas_rs = extract_record_set(
                croissant_ds, lambda meas_id: "/inits" not in meas_id
            )

            states: set[str] = set()
            measurements = []

            # Process each measurement
            for meas_uuid, rs in meas_rs.items():
                assert meas_uuid in init_rs, (
                    f"Initial conditions not found for {meas_uuid}"
                )

                # Extract initial conditions
                inits = {
                    key.replace(f"{meas_uuid}/inits/", ""): value
                    for key, value in list(init_rs[meas_uuid])[0].items()
                }

                # Extract measurement data
                meas_recs = croissant_ds.records(record_set=meas_uuid)
                data = json_lines_to_dict(meas_recs)
                time = data.pop("time")

                # Update states set
                states.update(inits.keys())

                # Create measurement
                measurements.append(
                    Measurement(
                        initial_conditions=inits,
                        data=data,  # type: ignore
                        time=time,
                        id=meas_uuid,
                    )
                )

        # Create dataset with metadata from Croissant
        return cls(
            id=croissant_ds.metadata.id,
            name=croissant_ds.metadata.name,
            description=croissant_ds.metadata.description,
            states=list(states),
            measurements=measurements,
        )

    @classmethod
    def from_jax_arrays(
        cls,
        state_order: List[str],
        data: Array,
        time: Array,
        y0s: Array,
    ) -> "Dataset":
        """Create a dataset directly from JAX arrays.

        Args:
            state_order: Ordered list of state names
            data: JAX array of concentrations with shape (n_measurements, n_timepoints, n_state)
            time: JAX array of timepoints with shape (n_measurements, n_timepoints)
            y0s: JAX array of initial conditions with shape (n_measurements, n_state)

        Returns:
            A new Dataset object with measurements constructed from the arrays

        Raises:
            AssertionError: If array shapes are incompatible
        """
        # Validate array shapes
        assert data.shape[0] == time.shape[0] == y0s.shape[0], (
            f"Incompatible shapes: data shape = {data.shape}, "
            f"time shape = {time.shape}, y0s shape = {y0s.shape}. "
            f"First dimensions must be equal."
        )

        assert y0s.shape[-1] == len(state_order), (
            f"Incompatible shapes: y0s shape = {y0s.shape}, "
            f"state_order length = {len(state_order)}. "
            f"Last state dimensions must be equal to state_order length."
        )

        # Create dataset
        dataset = cls(
            id=str(uuid.uuid4()),
            name=str(uuid.uuid4()),
            states=state_order,
        )

        # Add measurements
        for i in range(data.shape[0]):
            initial_conditions = {
                state: float(y0s[i, j]) for j, state in enumerate(state_order)
            }

            dataset.add_from_jax_array(
                state_order=state_order,
                initial_condition=initial_conditions,
                data=data[i],
                time=time[i],
            )

        return dataset

    # =====================
    # Validation Methods
    # =====================

    def leave_one_out(self):
        """Generator that yields dataset copies with one measurement left out.

        This method implements leave-one-out cross-validation by yielding tuples
        containing a copy of the dataset with one measurement removed and the ID
        of the removed measurement.

        Yields:
            Tuple[Dataset, str]: A tuple containing:
                - A copy of the dataset with one measurement removed
                - The ID of the measurement that was left out
        """
        for measurement in self.measurements:
            # Create a copy of the dataset
            rest_dataset = deepcopy(self)
            single_dataset = Dataset(
                **self.model_dump(exclude={"measurements"}),
                measurements=[measurement],
            )

            # Remove the current measurement from the copy
            rest_dataset.measurements = [
                m for m in rest_dataset.measurements if m.id != measurement.id
            ]

            yield single_dataset, rest_dataset

    def train_test_split(self, test_size: float = 0.2):
        """Split dataset into training and testing sets.

        This method splits the dataset into training and testing sets based on a specified test size.

        Args:
            test_size: Proportion of dataset to include in the test set (default: 0.2)

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing:
                - Training dataset
                - Testing dataset
        """

        assert test_size > 0 and test_size < 1, "Test size must be between 0 and 1"

        # Create shuffled indices
        indices = list(range(len(self.measurements)))
        random.shuffle(indices)

        # Calculate split index
        split_idx = int(len(self.measurements) * (1 - test_size))

        # Split indices
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Create training dataset
        train_dataset = deepcopy(self)
        train_dataset.measurements = [self.measurements[i] for i in train_indices]

        # Create testing dataset
        test_dataset = deepcopy(self)
        test_dataset.measurements = [self.measurements[i] for i in test_indices]

        return train_dataset, test_dataset

    # =====================
    # Plotting Methods
    # =====================

    @overload
    def plot(
        self,
        ncols: int = 2,
        show: Literal[True] = True,
        path: Optional[str] = None,
        measurement_ids: List[str] = [],
        figsize: Tuple[int, int] = (5, 3),
        predictor: Optional[Predictor] = None,
        n_steps: int = 100,
        xlim: Optional[Tuple[float, float | None]] = (0, None),
        **kwargs,
    ) -> None: ...

    @overload
    def plot(
        self,
        ncols: int = 2,
        show: Literal[False] = False,
        path: Optional[str] = None,
        measurement_ids: List[str] = [],
        figsize: Tuple[int, int] = (5, 3),
        predictor: Optional[Predictor] = None,
        n_steps: int = 100,
        xlim: Optional[Tuple[float, float | None]] = (0, None),
        **kwargs,
    ) -> Figure: ...

    def plot(
        self,
        ncols: int = 2,
        show: bool = False,
        path: Optional[str] = None,
        measurement_ids: List[str] = [],
        figsize: Tuple[int, int] = (5, 3),
        predictor: Optional[Predictor] = None,
        n_steps: int = 100,
        xlim: Optional[Tuple[float, float | None]] = (0, None),
        **kwargs,
    ) -> Optional[Figure]:
        """Plot all measurements in the dataset.

        Creates a multi-panel figure with one panel per measurement, with
        optional overlay of model predictions.

        Args:
            ncols: Number of columns in the figure grid
            show: Whether to display the figure
            path: Path to save the figure (if provided)
            measurement_ids: List of measurement IDs to plot (defaults to all measurements)
            figsize: Size of each individual subplot
            model: Optional model to overlay predictions
            n_steps: Number of points to use for model prediction curves
            **kwargs: Additional arguments passed to the measurement plot function

        Returns:
            The matplotlib figure object
        """
        from catalax.model import Model

        # Get measurement IDs if not provided
        measurement_ids = self._get_measurement_ids(measurement_ids)

        # Setup figure and axes
        fig, axs = self._setup_figure(measurement_ids, ncols, figsize)

        # Simulate model data if model is provided
        model_data = (
            predictor.predict(
                dataset=self,
                n_steps=n_steps,
            )
            if predictor
            else None
        )

        if predictor and isinstance(predictor, Model) and predictor.has_hdi():
            lower_95 = self._simulate_model_data(predictor, n_steps, "lower")
            upper_95 = self._simulate_model_data(predictor, n_steps, "upper")
            lower_50 = self._simulate_model_data(predictor, n_steps, "lower_50")
            upper_50 = self._simulate_model_data(predictor, n_steps, "upper_50")
        else:
            lower_95 = None
            upper_95 = None
            lower_50 = None
            upper_50 = None

        # Plot measurements
        self._plot_measurements(
            axs=axs,
            measurement_ids=measurement_ids,
            model_data=model_data,
            kwargs=kwargs,
            xlim=xlim,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_50=lower_50,
            upper_50=upper_50,
        )

        # Format legends
        self._format_legends(axs, ncols, measurement_ids)

        # Save or show figure
        self._save_or_show_figure(fig, path, show)

        if not show:
            return fig
        return None

    def _get_measurement_ids(self, measurement_ids: List[str]) -> List[str]:
        """Get measurement IDs to plot, using all measurements with data if none provided."""
        if not measurement_ids:
            return [meas.id for meas in self.measurements if meas.has_data()]
        return measurement_ids

    def _setup_figure(
        self, measurement_ids: List[str], ncols: int, figsize: Tuple[int, int]
    ):
        """Setup the figure and axes for plotting."""
        x, y = figsize
        ncols, nrows = self._get_rows_cols(measurement_ids, ncols)

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * x, nrows * y),
        )

        if len(measurement_ids) > 1:
            axs = axs.flatten()
        else:
            axs = [axs]

        return fig, axs

    def _simulate_model_data(
        self,
        model: Model,
        n_steps: int,
        use_hdi: Optional[HDIOptions] = None,
        hdi_prob: float = 0.95,
    ) -> Dataset:
        """Simulate model data for plotting."""
        # Import SimulationConfig here to avoid circular imports
        from catalax.model.simconfig import SimulationConfig

        _, saveat, _ = self.to_jax_arrays(
            state_order=[
                state for state in self.states if state not in model.constants
            ],
            inits_to_array=True,
        )

        t0 = saveat.min(axis=-1)
        t1 = saveat.max(axis=-1)

        return model.simulate(
            dataset=self,
            use_hdi=use_hdi,
            hdi_prob=hdi_prob,
            config=SimulationConfig(
                t0=t0,
                t1=t1,
                nsteps=n_steps,
            ),
        )

    def _plot_measurements(
        self,
        axs,
        measurement_ids: List[str],
        model_data: Optional[Dataset],
        kwargs: Dict[str, Any],
        xlim: Optional[Tuple[float, float | None]] = (0, None),
        lower_95: Optional[Dataset] = None,
        upper_95: Optional[Dataset] = None,
        lower_50: Optional[Dataset] = None,
        upper_50: Optional[Dataset] = None,
    ):
        """Plot each measurement on its corresponding axis."""
        index = 0
        for i, meas in enumerate(self.measurements):
            if meas.id not in measurement_ids:
                continue

            sim_meas = model_data.measurements[i] if model_data else None
            lower_95_meas = lower_95.measurements[i] if lower_95 else None
            upper_95_meas = upper_95.measurements[i] if upper_95 else None
            lower_50_meas = lower_50.measurements[i] if lower_50 else None
            upper_50_meas = upper_50.measurements[i] if upper_50 else None

            meas.plot(
                ax=axs[index],
                model_data=sim_meas,
                _lower_95=lower_95_meas,
                _upper_95=upper_95_meas,
                _lower_50=lower_50_meas,
                _upper_50=upper_50_meas,
                **kwargs,
                xlim=xlim,
            )

            index += 1

    def _format_legends(self, axs, ncols, measurement_ids):
        """Format legends for each subplot."""
        # Place legends only on rightmost plots in each row
        if len(axs) == 1:
            axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            return

        for i, ax in enumerate(axs):
            if i % ncols == ncols - 1:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().set_visible(False)

        # Hide unused axes if there's an odd number of measurements
        if len(measurement_ids) > 1 and len(measurement_ids) % 2 != 0:
            axs[-1].set_visible(False)

        plt.tight_layout(w_pad=4, h_pad=4)

    def _save_or_show_figure(self, fig, path, show):
        """Save figure to path if provided, otherwise show if requested."""
        if path:
            plt.savefig(
                path,
                dpi=300,
                format="png",
                bbox_inches="tight",
            )
        elif show:
            plt.show()

    @staticmethod
    def _get_rows_cols(
        measurement_ids: list[str],
        ncols: int,
    ) -> Tuple[int, int]:
        """Calculate the grid size needed for the plots."""
        if len(measurement_ids) == 1:
            return 1, 1
        else:
            return ncols, int(np.ceil(len(measurement_ids) / ncols))

    # =====================
    # Data Augmentation
    # =====================

    def augment(
        self,
        n_augmentations: int,
        sigma: float = 0.5,
        seed: int = 42,
        append: bool = True,
        multiplicative: bool = False,
    ) -> "Dataset":
        """Generate augmented versions of the dataset with added noise.

        This method creates artificial variants of the dataset by adding
        Gaussian noise to the measurements, which can be useful for
        uncertainty quantification and model robustness testing.

        Args:
            n_augmentations: Number of augmented copies to generate
            sigma: Standard deviation of the Gaussian noise
            seed: Random seed for reproducibility
            append: Whether to include original measurements in the result
            multiplicative: Whether to use multiplicative rather than additive noise

        Returns:
            A new Dataset containing the augmented measurements
        """
        # Create new dataset or copy existing
        if append:
            augmented_dataset = deepcopy(self)
        else:
            augmented_dataset = Dataset(
                id=self.id,
                name=self.name,
                description=self.description,
                states=self.states,
            )

        # Generate augmented measurements
        augmented_meas = []
        for i in range(n_augmentations):
            for meas in self.measurements:
                augmented_meas.append(
                    meas.augment(
                        sigma=sigma,
                        seed=seed + i,
                        multiplicative=multiplicative,
                    )
                )

        # Add augmented measurements to dataset
        augmented_dataset.measurements += augmented_meas

        return augmented_dataset

    # =====================
    # Utility Methods
    # =====================

    def metrics(
        self,
        predictor: Predictor,
        objective_fun: Callable[[Array, Array], Array] = l1_loss,  # type: ignore
    ) -> FitMetrics:
        """Calculate comprehensive fit metrics for evaluating predictor performance on this dataset.

        This method computes statistical metrics following the lmfit convention to assess
        how well a predictor fits the experimental data. The metrics are calculated by comparing
        predictions from the given predictor against the actual measurements in this dataset.

        The method performs the following steps:
        1. Generate predictions using the provided predictor
        2. Extract observable state data from both predictions and measurements
        3. Calculate chi-square, reduced chi-square, AIC, and BIC statistics

        Args:
            predictor (Predictor): The predictor model to evaluate. Must implement the predict()
                method and provide parameter count information.
            objective_fun: Objective function to use for calculating chi-square.
                Defaults to L2 loss.

        Returns:
            FitMetrics: A metrics object containing:
                - chisqr (float): Chi-square statistic
                - redchi (float): Reduced chi-square statistic
                - aic (float): Akaike Information Criterion
                - bic (float): Bayesian Information Criterion

        Raises:
            ValueError: If predictor cannot generate predictions for this dataset
            RuntimeError: If observable states orders don't match between dataset and predictions

        Note:
            Only observable states (those with actual measurement data) are included in the
            metric calculations. This ensures fair comparison when models predict additional
            unmeasured states.
        """
        pred = predictor.predict(self, use_times=True)
        state_order = self.get_observable_states_order()

        y_pred, _, _ = pred.to_jax_arrays(state_order=state_order)
        y_true, _, _ = self.to_jax_arrays(state_order=state_order)

        observable_indices = self.get_observable_indices()

        # Extract observable states data for metric calculation
        y_pred_obs = y_pred[:, :, observable_indices]
        y_true_obs = y_true[:, :, observable_indices]

        # Flatten arrays to compute metrics on all data points
        y_pred_flat = y_pred_obs.flatten()
        y_true_flat = y_true_obs.flatten()

        # Get model complexity metrics
        n_parameters = predictor.n_parameters()

        # Calculate and return comprehensive fit metrics
        return FitMetrics.from_predictions(
            y_true=y_true_flat,
            y_pred=y_pred_flat,
            n_parameters=n_parameters,
            objective_fun=objective_fun,
        )

    @staticmethod
    def get_vmap_dims(
        data: jax.Array,
        time: jax.Array,
        y0s: Union[jax.Array, List[Dict[str, float]]],
    ) -> Tuple[Optional[int], None, Optional[int]]:
        """Determines the dimensions for JAX's vmap operation based on input array shapes.

        This method analyzes the shapes of data, time, and initial condition arrays to determine
        which dimensions should be mapped over when using JAX's vectorized mapping.

        Args:
            data: JAX array of concentration time courses with shape
                 (n_measurements, n_timepoints, n_states) or (n_timepoints, n_states)
            time: JAX array of time points with shape (n_measurements, n_timepoints) or (n_timepoints,)
            y0s: Either a JAX array of initial conditions with shape (n_measurements, n_states)
                 or a list of dictionaries mapping state names to initial concentrations

        Returns:
            A tuple of three elements:
            - First element: 0 if data has 3 dimensions (batched), None otherwise
            - Second element: Always None (time dimension is not mapped over)
            - Third element: 0 if multiple initial conditions are present, None otherwise

        Raises:
            TypeError: If y0s is not a list or JAX array
        """
        # Check if data is batched (has 3 dimensions)
        data_in_batch = 0 if len(data.shape) == 3 else None

        # Check if there are multiple initial conditions
        if isinstance(y0s, list):
            multiple_y0s = 0 if len(y0s) > 1 else None
        elif isinstance(y0s, jax.Array):
            multiple_y0s = 0 if len(y0s.shape) == 2 else None
        else:
            raise TypeError(f"Expected list or JAX array for 'y0s' but got {type(y0s)}")

        # Return mapping dimensions for data, time, and y0s
        return (data_in_batch, None, multiple_y0s)

    def to_config(self, nsteps: int = 100) -> SimulationConfig:
        """Convert dataset to a SimulationConfig object."""
        from catalax.model.simconfig import SimulationConfig

        t0, t1 = [], []

        for meas in self.measurements:
            if meas.time is None:
                raise ValueError("Measurement has no time data.")
            t0.append(meas.time[0])
            t1.append(meas.time[-1])

        return SimulationConfig(
            t0=jnp.array(t0),
            t1=jnp.array(t1),
            nsteps=nsteps,
        )
