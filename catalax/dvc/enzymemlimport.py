from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
import pyenzyme as pe
import zntrack


class EnzymeMLImport(zntrack.Node):
    """ZnTrack/DVC node used to import data from an EnzymeML file.

    Args:
        path (str): Path to the EnzymeML file.
        species_mapping (Dict[str, str]): Mapping from species in the EnzymeML file to species in the model.

    Returns:
        data (Array): The data to fit the model to.
        times (Array): The times at which the data has been measured.
        initial_conditions (List[Dict[str, float]]): The initial conditions of the model.
    """

    # Dependencies & Parameters
    path: str = zntrack.dvc.deps()  # type: ignore
    species_mapping: Dict[str, str] = zntrack.zn.params()  # type: ignore

    # Outputs
    data: jax.Array = zntrack.zn.outs()  # type: ignore
    times: jax.Array = zntrack.zn.outs()  # type: ignore
    initial_conditions: List[Dict[str, float]] = zntrack.zn.outs()  # type: ignore

    def run(self):
        """Loads the data from the given EnzymeML file."""

        # Load the EnzymeML file
        enzmldoc = pe.EnzymeMLDocument.fromFile(self.path)
        meausurements = enzmldoc.exportMeasurementData()

        # Extract data from measurements
        data, times, initial_conditions = self._extract_measurement_data(meausurements)

        # Add to Node
        self.data = data
        self.times = times
        self.initial_conditions = initial_conditions

    def _extract_measurement_data(
        self, measurements
    ) -> Tuple[jax.Array, jax.Array, List[Dict[str, float]]]:
        """Extracts measurement data and organizes it by species."""

        # Inverse mapping from species to source speciess
        inverse_species_mapping = {
            target: source for source, target in self.species_mapping.items()
        }

        data, times, initial_conditions = [], [], []
        species_order = sorted(list(self.species_mapping.values()))

        for measurement in measurements.values():
            data.append(
                [
                    self._extract_time_course(
                        measurement["data"], species, inverse_species_mapping
                    )
                    for species in species_order
                    if inverse_species_mapping[species] in measurement["data"]
                ]
            )

            times.append(measurement["data"]["time"].values.tolist())
            initial_conditions.append(
                {
                    species: measurement["data"][species].values[0]
                    for species in species_order
                }
            )

        return (
            jnp.swapaxes(jnp.array(data), 1, 2),
            jnp.array(times),
            initial_conditions,
        )

    @staticmethod
    def _extract_time_course(
        data: pd.DataFrame, species: str, inverse_mapping: Dict[str, str]
    ) -> List:
        """Extracts the time course of the given species from the given data."""

        source_species = inverse_mapping[species]

        return data[source_species].values.tolist()
