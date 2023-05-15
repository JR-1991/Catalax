from typing import Dict, List
from pydantic import BaseModel, PrivateAttr
from lmfit import Parameters, minimize
import jax
import jax.numpy as jnp

from sysbiojax.model.model import Model


class ParameterEstimator:
    def __init__(
        self,
        model: Model,
        data: jax.Array,
        time: jax.Array,
        initial_states: List[Dict[str, float]],
        observables_mask: jax.Array,
    ):
        self.model = model
        self.data = data
        self.time = time
        self.initial_states = initial_states  # from data
        self.observables_mask = observables_mask

        self.lmfit_params = self._init_lmfit_params()

    def _init_lmfit_params(self) -> Parameters:
        """Creates lmfit Parameters object based on model.parameters."""
        lmfit_params = Parameters()

        # Get parameters from model
        param_keys = list(self.model.parameters.keys())
        initial_params = self.model.parameters

        # Map parameters to lmfit 'Parameters'
        for key in param_keys:
            lmfit_params.add(
                name=key,
                value=initial_params[key].value,
                min=initial_params[key].lower_bound,
                max=initial_params[key].upper_bound,
            )

        return lmfit_params

    def residuals(
        self,
        params: Parameters,
    ):
        # extract lmfit params and create array for simulation
        lmfit_params_dict = params.valuesdict()
        sorted_keys = sorted(list(lmfit_params_dict.keys()))
        parameters = jnp.array([lmfit_params_dict[key] for key in sorted_keys])

        # simulate
        _, sim_data = self.model.simulate(
            initial_conditions=self.initial_states,
            dt0=0.01,
            saveat=self.time,
            parameters=parameters,
            in_axes=(0, None, 0),
        )

        # calculate residuals
        residuals = sim_data[self.observables_mask] - self.data[self.observables_mask]

        return residuals

    def fit(self):
        """Fits data to model with given initial conditions and parameter guesses."""

        return minimize(
            fcn=self.residuals,
            params=self.lmfit_params,
        )
