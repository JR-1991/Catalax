"""EnzymeML integration for Catalax datasets.

This module provides functionality to convert Catalax datasets to EnzymeML format,
enabling interoperability with the EnzymeML ecosystem for enzyme kinetics modeling.
"""

from typing import Optional

import jax
import pyenzyme as pe

from catalax.dataset import Dataset
from catalax.dataset.measurement import Measurement


def to_enzymeml(
    dataset: Dataset,
    enzmldoc: Optional[pe.EnzymeMLDocument] = None,
) -> pe.EnzymeMLDocument:
    """Convert a Catalax dataset to an EnzymeML document.

    This function transforms a Catalax Dataset into an EnzymeML document format,
    preserving all measurement data, time points, and initial conditions. If an
    existing EnzymeML document is provided, the dataset measurements are appended
    to it.

    Args:
        dataset: The Catalax dataset to convert
        enzmldoc: Optional existing EnzymeML document to append to. If None,
                 a new document is created with the dataset name

    Returns:
        EnzymeML document containing the converted dataset measurements

    Example:
        >>> dataset = Dataset(name="my_experiment")
        >>> enzymeml_doc = to_enzymeml(dataset)
        >>> print(enzymeml_doc.name)
        my_experiment
    """
    if enzmldoc is None:
        enzmldoc = pe.EnzymeMLDocument(name=dataset.name or "catalax_dataset")
    else:
        enzmldoc = enzmldoc.model_copy(deep=True)

    offset = len(enzmldoc.measurements)
    for i, measurement in enumerate(dataset.measurements):
        index = i + offset
        enzmldoc.measurements.append(_convert_measurement(measurement, index))

    return enzmldoc


def _convert_measurement(
    measurement: Measurement,
    index: int,
) -> pe.Measurement:
    """Convert a Catalax measurement to an EnzymeML measurement.

    This internal function handles the conversion of individual Catalax measurements
    to EnzymeML format, including proper handling of JAX arrays and data validation.

    Args:
        measurement: The Catalax measurement to convert
        index: Index for naming the measurement if no name is provided

    Returns:
        EnzymeML measurement object with converted data

    Raises:
        ValueError: If data or time arrays are not in supported formats (list or jax.Array)
    """
    species_data = []
    for species, data in measurement.data.items():
        assert species in measurement.initial_conditions, (
            f"Species {species} is not in the initial conditions of measurement {measurement.id}",
            [", ".join(measurement.initial_conditions.keys())],
        )

        if isinstance(data, jax.Array):
            data = data.tolist()
        elif isinstance(data, list):
            data = data
        else:
            raise ValueError(f"Data for species {species} is not a list or jax.Array.")

        if isinstance(measurement.time, jax.Array):
            time = measurement.time.tolist()
        elif isinstance(measurement.time, list):
            time = measurement.time
        else:
            raise ValueError(
                f"Time for measurement {measurement.id} is not a list or jax.Array."
            )

        species_data.append(
            pe.MeasurementData(
                species_id=species,
                initial=measurement.initial_conditions[species],
                time=time,
                data=data,
                data_type=pe.DataTypes.CONCENTRATION,
            )
        )

    return pe.Measurement(
        id=f"m{index}",
        name=measurement.name or f"m{index}",
        species_data=species_data,
    )
