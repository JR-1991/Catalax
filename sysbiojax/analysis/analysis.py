from typing import Dict, List
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import numpy as np

from sysbiojax.model.model import Model
from sysbiojax.measurement.measurment import Experiment


class Analysis(BaseModel):
    data: Experiment
    model: Model = None

    def __post_init__(self):
        self.data_array = self._initialize_data_matrix()

    def _get_dimentions(self):
        """Returns dimentions of the experiment as needed for the simulate funcion of `Model`.
        shape = (measurements, species, repetitions, time)
        """

        n_measurements = len(self.data.measurements)

        species_keys = [
            list(measurement.reactants.keys()) for measurement in self.data.measurements
        ]
        unique_species = sorted(list({x for l in species_keys for x in l}))
        n_species = len(unique_species)

        n_replicates = max(
            measurement.get_max_replicates() for measurement in self.data.measurements
        )
        max_data_length = max(
            measurement.get_max_data_lengths() for measurement in self.data.measurements
        )

        n_measurements = len(self.data.measurements)

        return n_measurements, n_species, n_replicates, max_data_length

    def _initialize_data_matrix(self) -> jax.Array:
        """Initializes an array with the dimentions (measurements, species, repetitions, time),
        containing measurement data from 'data'.
        Array is initialized with nan values. Then, 'data' is inserted in the right dimension.
        Species are ordered alphabetically"""
        matrix = np.empty(self._get_dimentions())
        matrix[:] = np.nan

        for m, measurement in enumerate(self.data.measurements):
            for s, species_id in enumerate(sorted(measurement.reactants.keys())):
                if measurement.reactants[species_id].data is not None:
                    for r, replicate in enumerate(
                        measurement.reactants[species_id].data
                    ):
                        matrix[m, s, r] = replicate.values

        return jnp.array(matrix)
