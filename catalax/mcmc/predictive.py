"""Posterior-predictive trajectories and credible bands for :class:`HMCResults`.

This module turns a posterior into honest predictive bands in concentration
space by **pushing the full joint posterior through the model** and taking
quantiles *in trajectory space* -- never the marginal-corner shortcut (which
pins every parameter to its own HDI bound at once, ignoring correlations).

The trajectory is identical for both inference modes: a normal **Tsit5** ODE
solve of the mechanistic model with sampled parameters
(:func:`_integrate_ensemble`). The aleatoric noise is the directly-sampled
concentration-space std (:func:`_aleatoric_variance`):

* **Mechanistic** -- the sampled ``sigma`` (``Normal(state, sigma)``).
* **Surrogate (rate-matching)** -- the sampled per-observable ``sigma_y``. The
  rate-matching likelihood already pushes ``sigma_y`` *forward* to rate space
  via the surrogate Jacobian, so here it is read straight back as the
  concentration noise -- no inverse, no rate->concentration conversion.

:func:`euler_integrate_rates` is retained because :mod:`catalax.mcmc.loo` still
Euler-integrates the sampled rates for its concentration-space trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from catalax.dataset.dataset import Dataset
from catalax.mcmc.loo import (
    Integration,
    _resolve_yerrs,
    _stack_thetas,
    _subsample,
)
from catalax.model.simconfig import SimulationConfig

if TYPE_CHECKING:
    from catalax.mcmc.results import HMCResults

#: HDI band identifier -> percentile of the predictive distribution.
HDILevel = Literal["lower", "upper", "lower_50", "upper_50"]
_PERCENTILE: Dict[str, float] = {
    "lower": 2.5,
    "upper": 97.5,
    "lower_50": 25.0,
    "upper_50": 75.0,
}


def euler_integrate_rates(
    rates: Array,
    y0_obs: Array,
    dt: Array,
    rate_var: Array,
    *,
    integration: Integration,
    data_safe: Array,
) -> Tuple[Array, Array]:
    """Forward-Euler integrate per-draw observable rates into concentrations.

    Shared by the surrogate-mode LOO (:mod:`catalax.mcmc.loo`) and the
    posterior-predictive bands here, so the rate-matching integration lives in
    exactly one place. All arrays are for a *single* posterior draw.

    Args:
        rates: Observable rates at each point, ``(n_meas, n_time, n_obs)``.
        y0_obs: Measured initial observable state, ``(n_meas, n_obs)``.
        dt: Inter-measurement time steps, ``(n_meas, n_time - 1)``.
        rate_var: Rate-space variance at each point, ``(n_meas, n_time, n_obs)``.
        integration: ``"euler"`` (global forward Euler from the measured initial
            condition) or ``"euler_onestep"`` (one-step-ahead from each measured
            state, no error accumulation).
        data_safe: Measured concentrations with non-finite entries zeroed,
            ``(n_meas, n_time, n_obs)``. Only used by ``"euler_onestep"``.

    Returns:
        ``(yhat, var_y)`` -- the integrated concentration trajectory and its
        forward-propagated variance, each ``(n_meas, n_time, n_obs)``.
    """
    n_meas, _, n_obs = rates.shape

    if integration == "euler_onestep":
        # One-step-ahead: predict each point from the *previous measured* state
        # plus the local rate increment -- no error accumulation.
        step = rates[:, :-1, :] * dt[..., None]
        yhat_tail = data_safe[:, :-1, :] + step
        yhat = jnp.concatenate([y0_obs[:, None, :], yhat_tail], axis=1)
        var_tail = (dt[..., None] ** 2) * rate_var[:, :-1, :]
        var_y = jnp.concatenate([jnp.zeros((n_meas, 1, n_obs)), var_tail], axis=1)
    else:
        # Global forward Euler from the measured initial condition.
        inc = rates[:, :-1, :] * dt[..., None]
        cum = jnp.cumsum(inc, axis=1)
        yhat = jnp.concatenate(
            [y0_obs[:, None, :], y0_obs[:, None, :] + cum],
            axis=1,
        )
        var_inc = (dt[..., None] ** 2) * rate_var[:, :-1, :]
        var_cum = jnp.cumsum(var_inc, axis=1)
        var_y = jnp.concatenate([jnp.zeros((n_meas, 1, n_obs)), var_cum], axis=1)

    return yhat, var_y


def _integrate_ensemble(
    results: "HMCResults",
    dataset: Dataset,
    posterior: Dict[str, Array],
    *,
    n_steps: int,
) -> Tuple[Array, Array]:
    """Integrate the mechanistic model with Tsit5 once per posterior draw.

    Used for *both* inference modes -- the predictive trajectory is always a
    proper ODE solve of ``results.model`` with sampled parameters (in surrogate
    mode the parameters were inferred by rate-matching, but the trajectory is
    still the mechanistic ODE). Returns ``(times, yhat_ens)`` with ``times`` of
    shape ``(n_meas, n_steps)`` and ``yhat_ens`` the observable-state ensemble
    ``(n_draws, n_meas, n_steps, n_obs)``.
    """
    from diffrax import Tsit5

    from catalax.mcmc.mcmc import _prepare_mcmc_data

    bayes_model = results.bayesian_model
    # Force Tsit5: integrate the ODE normally regardless of how it was fit.
    config = SimulationConfig(
        t1=1, t0=0, dt0=bayes_model.dt0, solver=Tsit5, throw=False
    )
    data_prep = _prepare_mcmc_data(
        dataset, results.model, surrogate=None, config=config
    )

    observables = np.asarray(bayes_model.observables)
    sim_func = data_prep.sim_func
    y0s, constants, meas_times = data_prep.y0s, data_prep.constants, data_prep.times

    t0 = jnp.min(meas_times, axis=-1)
    t1 = jnp.max(meas_times, axis=-1)
    fractions = jnp.linspace(0.0, 1.0, n_steps)
    times = t0[:, None] + (t1 - t0)[:, None] * fractions[None, :]

    param_names = [name for name, _ in bayes_model.priors]
    thetas, _ = _stack_thetas(posterior, param_names)

    def per_draw(theta):
        states = sim_func(y0s, theta, constants, times)
        return states[..., observables]

    yhat_ens = jax.vmap(per_draw)(thetas)
    return times, yhat_ens


def _aleatoric_variance(
    results: "HMCResults",
    posterior: Dict[str, Array],
    yhat_shape: Tuple[int, ...],
) -> Array:
    """Per-draw concentration-space aleatoric variance, broadcast to ``yhat_shape``.

    Both modes read the directly-sampled concentration-space noise std
    (``sigma_y``) and square it -- there is no rate->concentration conversion in
    the predictive, because the surrogate model already pushes the noise *forward*
    (``sigma_y -> sigma_r``) inside the likelihood. A shared (scalar) ``sigma_y``
    is broadcast across observables.
    """
    n_draws = yhat_shape[0]
    sigma_y = np.asarray(posterior["sigma_y"]).reshape(n_draws, -1)  # (n_draws, n_obs|1)
    var = (sigma_y**2)[:, None, None, :]  # (n_draws, 1, 1, n_obs|1)
    return jnp.broadcast_to(jnp.asarray(var), yhat_shape)


def _ensemble_to_dataset(
    reference: Dataset,
    results: "HMCResults",
    times: Array,
    values: Array,
) -> Dataset:
    """Wrap a per-measurement observable trajectory into a plottable Dataset.

    ``times`` is ``(n_meas, n_time)`` and ``values`` is ``(n_meas, n_time,
    n_obs)``. Measurement ids and initial conditions are copied from
    ``reference`` so the result lines up positionally for plotting overlays.
    """
    from catalax.dataset.measurement import Measurement

    obs_states = list(results.model.get_observable_state_order())
    times = np.asarray(times)
    values = np.asarray(values)

    dataset = Dataset(
        id=reference.id,
        name=reference.name,
        states=reference.states,
    )
    for i, ref_meas in enumerate(reference.measurements):
        dataset.add_measurement(
            Measurement(
                id=ref_meas.id,
                initial_conditions=dict(ref_meas.initial_conditions),
                time=times[i],
                data={state: values[i, :, k] for k, state in enumerate(obs_states)},
            )
        )
    return dataset


def compute_ensemble(
    results: "HMCResults",
    dataset: Dataset,
    *,
    n_steps: int = 100,
    max_draws: Optional[int] = None,
    integration: Integration = "euler",
    yerrs: Union[float, Array, None] = None,
) -> Tuple[Array, Array, Array]:
    """Push the posterior through the model; return the raw trajectory ensemble.

    This is the expensive step (one integration per draw). The cheap quantile
    extraction is :func:`band_from_ensemble`, so a caller plotting several bands
    of the same fit computes this once and slices it repeatedly.

    Returns ``(times, yhat_ens, var_ens)`` with ``times`` of shape ``(n_meas,
    n_time)`` and the ensemble arrays of shape ``(n_draws, n_meas, n_time,
    n_obs)`` (``var_ens`` is the per-draw aleatoric variance).
    """
    posterior = _subsample(results.mcmc.get_samples(group_by_chain=True), max_draws)

    # Trajectory: always a normal Tsit5 ODE solve of the mechanistic model with
    # sampled parameters (both modes).
    times, yhat_ens = _integrate_ensemble(
        results, dataset, posterior, n_steps=n_steps
    )

    # Aleatoric noise: the directly-sampled concentration-space std (``sigma`` for
    # an integrated fit, ``sigma_y`` for a rate-matching fit -- the surrogate
    # model already pushes it forward to rate space, so nothing to invert here).
    if yerrs is not None:
        var_ens = jnp.broadcast_to(
            _resolve_yerrs(yerrs, results, yhat_ens.shape[1:]) ** 2,
            yhat_ens.shape,
        )
    else:
        var_ens = _aleatoric_variance(results, posterior, yhat_ens.shape)
    return times, yhat_ens, var_ens


def band_from_ensemble(
    results: "HMCResults",
    reference: Dataset,
    times: Array,
    yhat_ens: Array,
    var_ens: Array,
    *,
    hdi: Optional[HDILevel] = None,
    include_noise: bool = True,
) -> Dataset:
    """Slice a posterior-predictive trajectory/band out of a precomputed ensemble.

    Quantiles are taken per ``(state, time)`` in trajectory space. With
    ``include_noise`` the aleatoric component (sampled from ``var_ens``) is
    pooled in, giving a true predictive interval rather than an
    epistemic-only band.

    Args:
        hdi: ``None`` returns the posterior-median trajectory; otherwise the
            matching 2.5/97.5/25/75 percentile band.
        include_noise: Fold in aleatoric noise (predictive interval) vs.
            parameter uncertainty only (epistemic band).
    """
    yhat_ens = np.asarray(yhat_ens)

    if hdi is None:
        # Posterior-median trajectory -- zero-mean noise does not move the centre.
        values = np.nanmedian(yhat_ens, axis=0)
    else:
        if include_noise:
            # Sample the aleatoric noise per draw and pool, so the band is a true
            # predictive quantile without assuming Gaussian tails downstream.
            std = np.sqrt(np.asarray(var_ens))
            noise = np.asarray(
                jax.random.normal(jax.random.PRNGKey(0), shape=yhat_ens.shape)
            )
            pooled = yhat_ens + std * noise
        else:
            pooled = yhat_ens
        values = np.nanpercentile(pooled, _PERCENTILE[hdi], axis=0)

    return _ensemble_to_dataset(reference, results, times, values)


def posterior_predictive(
    results: "HMCResults",
    dataset: Dataset,
    *,
    hdi: Optional[HDILevel] = None,
    n_steps: int = 100,
    max_draws: Optional[int] = None,
    integration: Integration = "euler",
    include_noise: bool = True,
    yerrs: Union[float, Array, None] = None,
) -> Dataset:
    """One-shot posterior-predictive trajectory (``hdi=None``) or band as a Dataset.

    Convenience wrapper composing :func:`compute_ensemble` and
    :func:`band_from_ensemble`; see those for the argument semantics. Callers
    that need several bands of the same fit should reuse a single
    :func:`compute_ensemble` result instead.
    """
    times, yhat_ens, var_ens = compute_ensemble(
        results,
        dataset,
        n_steps=n_steps,
        max_draws=max_draws,
        integration=integration,
        yerrs=yerrs,
    )
    return band_from_ensemble(
        results,
        dataset,
        times,
        yhat_ens,
        var_ens,
        hdi=hdi,
        include_noise=include_noise,
    )
