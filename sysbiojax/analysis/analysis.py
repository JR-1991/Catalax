from typing import Dict, List
from pydantic import BaseModel, PrivateAttr, validator
import jax
import jax.numpy as jnp
import numpy as np

from sysbiojax.model.model import Model
from sysbiojax.tools.parameterestimation import ParameterEstimator
from sysbiojax.measurement.measurment import Experiment


class Analysis:
    def __init__(self, data: Experiment, model: Model = None) -> None:
        self.data = data
        self.model = model

        self._unique_species = self._get_unique_species()
        self._data_shape = self._get_dimensions()
        self._data_matrix = self._initialize_data_matrix()
        self._initial_condition = self._get_initial_conditions()
        self.time = self._get_time()

    def _get_dimensions(self) -> tuple:
        """Returns dimensions of the experiment as needed for the simulate funcion of `Model`.
        shape = (measurements, replicates, timepoints, species)
        """

        n_measurements = len(self.data.measurements)

        n_species = len(self._unique_species)

        n_replicates = max(
            measurement.get_max_replicates() for measurement in self.data.measurements
        )
        max_data_length = max(
            measurement.get_max_data_lengths() for measurement in self.data.measurements
        )

        n_measurements = len(self.data.measurements)

        return (n_measurements, n_replicates, max_data_length, n_species)

    def _get_time(self):
        """Extracts time from data and reshapes it according to the needs of the model.simulate function."""
        times = []
        time_shape = (self._data_shape[0] * self._data_shape[1], self._data_shape[2])
        for measurement in self.data.measurements:
            n_replicates = measurement.get_max_replicates()
            times.append([measurement.time] * n_replicates)

        return jnp.array(times).reshape(time_shape)

    def _initialize_data_matrix(self) -> jax.Array:
        """Initializes an array with the dimensions (measurements, species, repetitions, time),
        containing measurement data from 'data'.
        Array is initialized with nan values. Then, 'data' is inserted in the right dimension.
        Species are ordered alphabetically"""
        matrix = np.empty(self._data_shape)
        matrix[:] = np.nan

        for m, measurement in enumerate(self.data.measurements):
            for s, species_id in enumerate(sorted(measurement.reactants.keys())):
                if measurement.reactants[species_id].data is not None:
                    for r, replicate in enumerate(
                        measurement.reactants[species_id].data
                    ):
                        # (measurement, replicates, timepoints, species)
                        matrix[m, r, :, s] = replicate.values

        return jnp.array(matrix)

    def _get_unique_species(self) -> List[str]:
        """Iterates over all measurements and extracts species keys.
        Returns sorted list of keys."""
        species_keys = [
            list(measurement.reactants.keys()) for measurement in self.data.measurements
        ]
        unique_species = sorted(list({x for l in species_keys for x in l}))
        return unique_species

    def _get_initial_conditions(self) -> List[Dict[str, float]]:
        """Extracts initial conditions from data.
        Returns List of dicts, whereas length of lists corresponds to measurements and
        dict keys to species int data."""

        initial_conditions = []

        # Extract initial conditions of species
        for measurement in self.data.measurements:
            initial_conditions.append(
                [
                    {
                        species: measurement.reactants[species].init_conc
                        for species in self._unique_species
                    }
                ]
                * measurement.get_max_replicates()
            )

        # unpack list of lists
        initial_conditions = [l for s in initial_conditions for l in s]

        return initial_conditions

    def fit_data_to_model(self):
        """lmfit parameter estiamtion."""
        estimator = ParameterEstimator(
            model=self.model,
            data=self._data_matrix,
            time=self.time,
            initial_states=self._initial_condition,
        )

        return estimator.fit()
