"""Leave-one-out cross-validation (LOO) in concentration/integration space.

This module implements LOO-CV for :class:`HMCResults` whose held-out unit is a
real concentration measurement, valid regardless of which inference mode
produced the posterior. The **prediction** always reuses the fit's own sampled
quantities (no re-derivation from scratch); the **observation noise** defaults to
the known measurement error.

* **Mechanistic mode** -- the ``"y"`` site is already in concentration space
  (loc = integrated ODE states). We integrate per draw and score against the
  measured concentrations.
* **Surrogate mode (RM-NLL)** -- the ``"y"`` site is *mechanistic RHS vs
  surrogate RHS*, so its per-draw predictions are the **rates** ``v(yi; theta)``
  evaluated at the measured states. Running ``az.loo`` on that site scores rate
  residuals -- not a measurement-level validation statistic. Instead we *reuse
  those sampled rates*, **manually integrate them** along the measurement times
  (forward Euler), and score the resulting concentration trajectory against the
  measured concentrations.

The default noise (``sigma_source="reuse"``) is the fit's own inferred
concentration-space noise ``sigma_y`` (sampled by the error model). This is the
proper PSIS-LOO statistic -- it reuses the model's own likelihood, and the
mechanistic reconstruction reproduces native ``az.loo`` to ~1e-7. Because the
error model samples ``sigma_y`` directly in concentration space for both modes,
no separate measurement error is needed. Alternatively ``sigma_source="yerrs"``
scores the held-out concentration against a supplied instrument error instead,
asking "does the trajectory predict each measurement to within instrument
error". For a surrogate fit that path needs a concentration-space ``yerrs``,
since the stored one is rate-shaped.

The headline diagnostic is the **per-observation Pareto-k** (``pointwise=True``),
which flags high-influence measurements: exactly the points a naive train/test
split would wrongly discard.

.. note::
    LOO answers "does the fit generalize", not "is each parameter individually
    identifiable". Pair it with posterior-correlation / prior-sensitivity
    analysis for structurally traded-off parameters (e.g. an ``E0``-``kcat``
    trade-off), which LOO will not flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from catalax.dataset.dataset import Dataset
from catalax.model.simconfig import SimulationConfig

if TYPE_CHECKING:
    from catalax.mcmc.results import HMCResults

#: Source of the concentration-space noise scale used to score held-out points.
#:
#: * ``"reuse"`` (default) -- use the fit's own inferred concentration-space
#:   noise ``sigma_y`` (sampled by the error model). This is the proper PSIS-LOO
#:   likelihood and needs no separate measurement error.
#: * ``"yerrs"`` -- score the held-out concentration against a supplied
#:   instrument error instead. The prediction still comes from the fit (the
#:   integrated trajectory); only the observation noise is the supplied error.
#:   Asks "does the trajectory predict the measurement to within instrument
#:   error", and needs a concentration-space ``yerrs`` for a surrogate fit.
SigmaSource = Literal["yerrs", "reuse"]
LeaveOut = Literal["point", "curve"]
Integration = Literal["euler", "euler_onestep"]


def _build_eval_config(results: "HMCResults") -> SimulationConfig:
    """Reconstruct the integrator configuration used during the original fit.

    The solver and ``dt0`` are taken from the stored Bayesian model; integration
    bounds come from the measurement times inside the solver, so ``t1`` is a
    placeholder. ``throw`` is disabled so a failed solve yields non-finite states
    (which the mask then drops) rather than aborting LOO.
    """
    bayes_model = results.bayesian_model
    return SimulationConfig(
        t1=1,
        t0=0,
        dt0=bayes_model.dt0,
        solver=bayes_model.solver,
        throw=False,
    )


def _stack_thetas(
    posterior: Dict[str, Array], param_names
) -> Tuple[Array, Tuple[int, int]]:
    """Stack parameter draws into ``(n_chain*n_draw, n_params)`` in prior order."""
    cols = [np.asarray(posterior[name]) for name in param_names]
    n_chain, n_draw = cols[0].shape[:2]
    flat = [c.reshape(n_chain * n_draw, *c.shape[2:]) for c in cols]
    thetas = jnp.asarray(np.stack(flat, axis=-1))
    return thetas, (n_chain, n_draw)


def _sigma_y_draws(
    posterior: Dict[str, Array], n_chain: int, n_draw: int, n_obs: int
) -> Array:
    """Flattened per-observable concentration-space noise ``sigma_y`` draws.

    A surrogate fit samples ``sigma_y`` directly in concentration space (the
    Jacobian pushforward to rate space happens inside the likelihood), so this is
    the aleatoric std used to score the integrated trajectory. Returns
    ``(n_chain*n_draw, n_obs)``; a shared scalar ``sigma_y`` is broadcast across
    observables. Falls back to ones if the site is absent.
    """
    if "sigma_y" in posterior:
        flat = np.asarray(posterior["sigma_y"]).reshape(n_chain * n_draw, -1)
        return jnp.broadcast_to(jnp.asarray(flat), (n_chain * n_draw, n_obs))
    return jnp.ones((n_chain * n_draw, n_obs))


def _subsample(posterior: Dict[str, Array], max_draws: Optional[int]):
    """Evenly subsample the draw axis (axis 1) of a grouped-by-chain sample dict."""
    if max_draws is None:
        return posterior
    n_draw = next(iter(posterior.values())).shape[1]
    if n_draw <= max_draws:
        return posterior
    idx = np.linspace(0, n_draw - 1, max_draws).astype(int)
    return {k: jnp.asarray(np.asarray(v)[:, idx]) for k, v in posterior.items()}


def _mechanistic_loglik(
    results: "HMCResults",
    dataset: Dataset,
    posterior: Dict[str, Array],
    *,
    sigma_source: SigmaSource,
    yerrs: Union[float, Array, None],
    config: Optional[SimulationConfig],
):
    """Concentration-space pointwise log-lik by integrating the ODE per draw.

    Reuses the model's own concentration likelihood (loc = integrated states,
    scale = ``sqrt(sigma_total)``), reproducing the native ArviZ statistic.

    Returns ``(loglik, mask)`` with ``loglik`` of shape
    ``(n_chain, n_draw, n_meas, n_time, n_obs)`` and ``mask`` of shape
    ``(n_meas, n_time, n_obs)``.
    """
    from catalax.mcmc.mcmc import _prepare_mcmc_data

    bayes_model = results.bayesian_model
    if config is None:
        config = _build_eval_config(results)

    data_prep = _prepare_mcmc_data(
        dataset, results.model, surrogate=None, config=config
    )
    data = data_prep.data  # (n_meas, n_time, n_obs)
    mask = np.asarray(data_prep.mask)
    data_safe = jnp.where(jnp.asarray(mask), data, 0.0)

    observables = bayes_model.observables
    likelihood = bayes_model.likelihood
    sim_func = data_prep.sim_func
    y0s, constants, times = data_prep.y0s, data_prep.constants, data_prep.times

    n_obs = int(np.asarray(observables).size)
    param_names = [name for name, _ in bayes_model.priors]
    thetas, (n_chain, n_draw) = _stack_thetas(posterior, param_names)
    sigma_y = _sigma_y_draws(posterior, n_chain, n_draw, n_obs)

    # "reuse": the sampled concentration-space ``sigma_y`` is the noise scale.
    # "yerrs": score against a supplied measurement error instead.
    sigma_y_fixed = (
        _resolve_yerrs(yerrs, results, tuple(data.shape))
        if sigma_source == "yerrs"
        else None
    )

    def per_draw(theta, sig_y):
        states = sim_func(y0s, theta, constants, times)
        loc = states[..., observables]
        # The error model samples ``sigma_y`` directly in concentration space
        # (scored via ``sqrt(sigma_y**2)``), so it is the scale as-is. This
        # reproduces the native per-point log-likelihood to ~1e-7.
        scale = (
            sigma_y_fixed
            if sigma_y_fixed is not None
            else jnp.broadcast_to(sig_y, data.shape)
        )
        return likelihood(loc, scale).log_prob(data_safe)

    ll = jax.vmap(per_draw)(thetas, sigma_y)
    ll = np.asarray(ll).reshape(n_chain, n_draw, *data.shape)
    info = {
        "data": np.asarray(data),
        "times": np.asarray(times),
        "species": list(results.model.get_observable_state_order()),
    }
    return ll, mask, info


def _surrogate_loglik(
    results: "HMCResults",
    dataset: Dataset,
    posterior: Dict[str, Array],
    *,
    sigma_source: SigmaSource,
    yerrs: Union[float, Array, None],
    integration: Integration,
):
    """Concentration-space pointwise log-lik by Euler-integrating sampled rates.

    Reuses the surrogate-mode forward function (the mechanistic RHS, i.e. the
    ``"y"`` site's per-draw predictions) to obtain rates at every measured
    ``(measurement, time)`` point, then manually integrates them along each
    trajectory and scores the result against the measured concentrations.

    The noise is concentration-space: the fit's inferred ``sigma_y`` (default), or under
    ``sigma_source="reuse"`` the fit's own sampled ``sigma_y`` -- which the
    surrogate model already pushes *forward* to rate space in the likelihood, so
    it is used here directly as the concentration scale (no inverse mapping).

    Returns ``(loglik, mask)`` shaped as in :func:`_mechanistic_loglik`.
    """
    bayes_model = results.bayesian_model
    model = results.model

    # Full modeled-state trajectory at the measurement points -- the inputs the
    # rate function is evaluated at (mirrors the surrogate-mode data prep).
    full_data, full_times, _ = dataset.to_jax_arrays(model.get_state_order())
    n_meas, n_time, n_states = full_data.shape
    constants_full = dataset.to_y0_matrix(state_order=model.get_constants_order())

    points_flat = full_data.reshape(-1, n_states)
    times_flat = full_times.ravel()
    constants_flat = jnp.repeat(constants_full, n_time, axis=0)

    observables = np.asarray(bayes_model.observables)
    likelihood = bayes_model.likelihood
    rate_sim_func = bayes_model.sim_func  # vmapped over the flattened points

    # Measured concentrations to score against (observable columns only).
    data_obs = np.asarray(full_data)[..., observables]  # (n_meas, n_time, n_obs)
    mask = np.isfinite(data_obs)
    data_safe = jnp.where(jnp.asarray(mask), jnp.asarray(data_obs), 0.0)

    y0_full = full_data[:, 0, :]  # measured initial state per measurement
    dt = jnp.diff(full_times, axis=-1)  # (n_meas, n_time - 1)

    n_obs = observables.size
    use_reuse = sigma_source == "reuse"
    # In "yerrs" mode the noise is the supplied measurement error; in "reuse"
    # mode it is the fit's own sampled concentration-space noise ``sigma_y`` (the
    # surrogate model pushes it forward to rate space, so here it is used
    # directly as the concentration scale -- no rate->concentration conversion).
    yerrs_scale = (
        _resolve_yerrs(yerrs, results, (n_meas, n_time, n_obs))
        if not use_reuse
        else None
    )

    param_names = [name for name, _ in bayes_model.priors]
    thetas, (n_chain, n_draw) = _stack_thetas(posterior, param_names)
    sigma_y = _sigma_y_draws(posterior, n_chain, n_draw, n_obs)

    # Shared forward-Euler integrator -- also used for posterior-predictive
    # bands (imported lazily to avoid an import cycle with ``predictive``).
    from catalax.mcmc.predictive import euler_integrate_rates

    y0_obs = y0_full[:, observables]
    zero_rate_var = jnp.zeros((n_meas, n_time, n_obs))

    def per_draw(theta, sig_y):
        rates_flat = rate_sim_func(points_flat, theta, constants_flat, times_flat)
        rates = rates_flat.reshape(n_meas, n_time, n_states)[..., observables]
        # Euler-integrate the sampled rates into a concentration trajectory; the
        # noise is the concentration-space ``sigma_y`` (or supplied ``yerrs``).
        yhat, _ = euler_integrate_rates(
            rates,
            y0_obs,
            dt,
            zero_rate_var,
            integration=integration,
            data_safe=data_safe,
        )
        scale = (
            yerrs_scale
            if yerrs_scale is not None
            else jnp.broadcast_to(sig_y, (n_meas, n_time, n_obs))
        )
        return likelihood(yhat, scale).log_prob(data_safe)

    ll = jax.vmap(per_draw)(thetas, sigma_y)
    ll = np.asarray(ll).reshape(n_chain, n_draw, n_meas, n_time, n_obs)

    modeled = list(model.get_state_order(modeled=True))
    info = {
        "data": np.asarray(data_obs),
        "times": np.asarray(full_times),
        "species": [modeled[i] for i in observables],
    }
    return ll, mask, info


def _resolve_yerrs(
    yerrs: Union[float, Array, None],
    results: "HMCResults",
    data_shape: Tuple[int, ...],
) -> Array:
    """Broadcast a concentration-space measurement error to ``data_shape``.

    Only used for ``sigma_source="yerrs"``. ``yerrs=None`` falls back to the
    measurement error stored on the fit (valid only when it is already
    concentration-space, i.e. a mechanistic fit).
    """
    if yerrs is None:
        stored = results.bayesian_model.yerrs
        if tuple(stored.shape) == tuple(data_shape):
            return jnp.asarray(stored)
        raise ValueError(
            "`yerrs` was not provided and the stored measurement error does not "
            f"match the concentration-space data shape {data_shape} (stored "
            f"{tuple(stored.shape)}). Pass concentration-space `yerrs` explicitly."
        )
    if isinstance(yerrs, (float, int)):
        return jnp.broadcast_to(float(yerrs), data_shape)
    return jnp.broadcast_to(jnp.asarray(yerrs), data_shape)


def reconstruct_log_likelihood(
    results: "HMCResults",
    dataset: Dataset,
    *,
    yerrs: Union[float, Array, None] = None,
    sigma_source: SigmaSource = "reuse",
    leave_out: LeaveOut = "point",
    integration: Integration = "euler",
    max_draws: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
):
    """Compute per-observation, per-draw concentration-space log-likelihoods.

    Reuses the fit's own RHS and rate-space noise to score the integrated
    trajectory against the measured concentrations in ``dataset``, returning the
    held-out log densities ready for :func:`arviz.loo`.

    Args:
        results: The fitted :class:`HMCResults` (mechanistic or surrogate mode).
        dataset: Dataset of real concentration measurements to score against.
        sigma_source: ``"reuse"`` (default) uses the fit's inferred ``sigma_y``;
            against the supplied measurement error; ``"reuse"`` instead takes the
            noise from the sampled ``sigma`` (pushed forward through the
            integration in surrogate mode).
        yerrs: Concentration-space measurement error; used when
            ``sigma_source="yerrs"`` (ignored under ``"reuse"``). ``None`` falls
            back to the error stored on the fit (mechanistic fits only).
        leave_out: ``"point"`` holds out one species/timepoint; ``"curve"`` holds
            out a whole measurement series -- "predict a new experiment".
        integration: For surrogate fits, ``"euler"`` (global forward Euler) or
            ``"euler_onestep"`` (one-step-ahead from each measured state).
        max_draws: Optionally subsample the posterior draws for speed.
        config: Override the integrator :class:`SimulationConfig`.

    Returns:
        Tuple ``(loglik, posterior, meta)``: ``loglik`` numpy array of shape
        ``(chains, draws, n_obs_kept)``, ``posterior`` the (subsampled) parameter
        draws grouped by chain, and ``meta`` bookkeeping (kept indices, full
        shape, ``n_dropped``, mode).
    """
    from catalax.mcmc.mcmc import Modes

    posterior = results.mcmc.get_samples(group_by_chain=True)
    posterior = _subsample(posterior, max_draws)

    mode = results.bayesian_model.mode
    if mode == Modes.MECHANISTIC:
        ll, mask, info = _mechanistic_loglik(
            results,
            dataset,
            posterior,
            sigma_source=sigma_source,
            yerrs=yerrs,
            config=config,
        )
    else:
        ll, mask, info = _surrogate_loglik(
            results,
            dataset,
            posterior,
            sigma_source=sigma_source,
            yerrs=yerrs,
            integration=integration,
        )

    n_chain, n_draw = ll.shape[0], ll.shape[1]
    event_shape = ll.shape[2:]

    if leave_out == "point":
        flat = ll.reshape(n_chain, n_draw, -1)
        keep = mask.reshape(-1)
        kept_idx = np.flatnonzero(keep)
        loglik = flat[:, :, kept_idx]
        meta = {
            "leave_out": "point",
            "kept_idx": kept_idx,
            "full_shape": event_shape,
            "n_total": int(keep.size),
            "n_dropped": int(keep.size - kept_idx.size),
        }
    elif leave_out == "curve":
        n_meas = event_shape[0]
        # Sum over (timepoints, observables); masked points contribute 0.
        ll_masked = np.where(mask[None, None], ll, 0.0)
        per_curve = ll_masked.reshape(n_chain, n_draw, n_meas, -1).sum(axis=-1)
        curve_has_data = mask.reshape(n_meas, -1).any(axis=-1)
        kept_idx = np.flatnonzero(curve_has_data)
        loglik = per_curve[:, :, kept_idx]
        meta = {
            "leave_out": "curve",
            "kept_idx": kept_idx,
            "full_shape": (n_meas,),
            "n_total": int(n_meas),
            "n_dropped": int(n_meas - kept_idx.size),
        }
    else:
        raise ValueError(f"Unknown leave_out={leave_out!r}; use 'point' or 'curve'.")

    param_names = [name for name, _ in results.bayesian_model.priors]
    posterior_np = {
        name: np.asarray(posterior[name]) for name in param_names if name in posterior
    }
    meta["sigma_source"] = sigma_source
    meta["mode"] = mode.name
    meta["integration"] = integration
    meta["info"] = info
    return loglik, posterior_np, meta


def build_loo_idata(
    results: "HMCResults",
    dataset: Dataset,
    **kwargs,
) -> az.InferenceData:
    """Build an ``InferenceData`` whose ``log_likelihood`` group is the
    concentration-space, integrator-scored reconstruction.

    Accepts the same keyword arguments as :func:`reconstruct_log_likelihood`.
    """
    loglik, posterior_np, meta = reconstruct_log_likelihood(results, dataset, **kwargs)
    idata = az.from_dict(
        posterior=posterior_np if posterior_np else None,
        log_likelihood={"y": loglik},
    )
    idata.log_likelihood.attrs.update(
        {
            "catalax_leave_out": meta["leave_out"],
            "catalax_sigma_source": meta["sigma_source"],
            "catalax_mode": meta["mode"],
            "catalax_n_dropped": meta["n_dropped"],
        }
    )
    return idata


def loo(
    results: "HMCResults",
    dataset: Dataset,
    *,
    yerrs: Union[float, Array, None] = None,
    sigma_source: SigmaSource = "reuse",
    leave_out: LeaveOut = "point",
    integration: Integration = "euler",
    pointwise: bool = True,
    reloo: bool = False,
    max_draws: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
    scale: Literal["log", "negative_log", "deviance"] = "log",
):
    """Concentration-space LOO-CV for a fit produced in either inference mode.

    See the module docstring for the rationale. Returns an
    :class:`arviz.ELPDData` with ``elpd_loo``, ``p_loo`` and (when ``pointwise``)
    per-observation Pareto-:math:`k` diagnostics. High Pareto-k values flag
    high-influence observations -- the points a naive split would discard.

    Args:
        results: Fitted :class:`HMCResults`.
        dataset: Real concentration measurements to validate against.
        sigma_source: ``"reuse"`` (default) uses the fit's inferred ``sigma_y``;
            measurement error; ``"reuse"`` takes the noise from the sampled
            ``sigma``.
        yerrs: Concentration-space measurement error; used when
            ``sigma_source="yerrs"``. ``None`` falls back to the error stored on
            the fit (mechanistic fits only).
        leave_out: ``"point"`` or ``"curve"``.
        integration: Surrogate-mode integration scheme (``"euler"`` or
            ``"euler_onestep"``).
        pointwise: Return per-observation Pareto-k diagnostics.
        reloo: Not supported; passing ``True`` raises ``NotImplementedError``.
        max_draws: Optional posterior draw subsampling for speed.
        config: Override integrator :class:`SimulationConfig`.
        scale: ArviZ ELPD scale.

    Returns:
        arviz.ELPDData: The LOO result.
    """
    if reloo:
        raise NotImplementedError(
            "reloo=True requires refitting the model for flagged observations, "
            "which is out of scope for this single-fit diagnostic. Inspect the "
            "Pareto-k values (pointwise=True) to identify high-influence points."
        )

    idata = build_loo_idata(
        results,
        dataset,
        yerrs=yerrs,
        sigma_source=sigma_source,
        leave_out=leave_out,
        integration=integration,
        max_draws=max_draws,
        config=config,
    )
    return az.loo(idata, pointwise=pointwise, scale=scale)


@dataclass
class LooPointwise:
    """Per-observation LOO diagnostics mapped back onto the data grid.

    Every array is shaped ``(n_measurements, n_timepoints, n_observables)`` with
    ``NaN`` at points that were not scored (missing data, or the surrogate-mode
    ``t0`` anchor). This is the structure both LOO plots consume.

    Attributes:
        elpd: Pointwise expected log predictive density ``elpd_i`` (higher is a
            better-predicted point; ``-elpd`` is the per-point penalty).
        pareto_k: Pointwise PSIS Pareto-:math:`k` (influence / reliability;
            ``> 0.7`` is concerning, ``> 1`` unreliable).
        data: The measured concentrations being scored.
        times: Measurement times, shape ``(n_measurements, n_timepoints)``.
        species: Observable state symbols, length ``n_observables``.
        measurement_ids: Measurement ids, length ``n_measurements``.
        measurement_labels: Concise per-measurement labels (from initial
            conditions), length ``n_measurements`` -- used for plot titles.
        measurement_initial_conditions: Per-measurement initial-condition dicts,
            length ``n_measurements`` -- used to render plot titles in the shared
            library style.
        elpd_loo, p_loo, scale: The summary LOO statistics.
        elpd_data: The raw :class:`arviz.ELPDData` from :func:`arviz.loo`.
    """

    elpd: np.ndarray
    pareto_k: np.ndarray
    data: np.ndarray
    times: np.ndarray
    species: list
    measurement_ids: list
    measurement_labels: list
    measurement_initial_conditions: list
    elpd_loo: float
    p_loo: float
    scale: str
    elpd_data: "az.ELPDData"


def loo_pointwise(
    results: "HMCResults",
    dataset: Dataset,
    *,
    yerrs: Union[float, Array, None] = None,
    sigma_source: SigmaSource = "reuse",
    integration: Integration = "euler",
    max_draws: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
    scale: Literal["log", "negative_log", "deviance"] = "log",
) -> LooPointwise:
    """Pointwise (per species x timepoint) LOO diagnostics on the data grid.

    Runs ``leave_out="point"`` LOO and scatters ``elpd_i`` and Pareto-:math:`k`
    back onto the ``(measurement, time, species)`` grid (``NaN`` where a point
    was not scored). Used by :func:`plot_loo_heatmap` and
    :func:`plot_loo_influence`. See :func:`loo` for argument semantics.
    """
    loglik, posterior_np, meta = reconstruct_log_likelihood(
        results,
        dataset,
        yerrs=yerrs,
        sigma_source=sigma_source,
        leave_out="point",
        integration=integration,
        max_draws=max_draws,
        config=config,
    )
    idata = az.from_dict(
        posterior=posterior_np if posterior_np else None,
        log_likelihood={"y": loglik},
    )
    elpd_data = az.loo(idata, pointwise=True, scale=scale)

    full_shape = meta["full_shape"]
    kept_idx = meta["kept_idx"]
    n_total = int(np.prod(full_shape))

    elpd_flat = np.full(n_total, np.nan)
    k_flat = np.full(n_total, np.nan)
    elpd_flat[kept_idx] = np.asarray(elpd_data.loo_i).ravel()
    k_flat[kept_idx] = np.asarray(elpd_data.pareto_k).ravel()

    def _label(meas):
        ic = getattr(meas, "initial_conditions", {}) or {}
        return ", ".join(f"{k}={v:g}" for k, v in ic.items())

    info = meta["info"]
    return LooPointwise(
        elpd=elpd_flat.reshape(full_shape),
        pareto_k=k_flat.reshape(full_shape),
        data=info["data"],
        times=info["times"],
        species=info["species"],
        measurement_ids=[m.id for m in dataset.measurements],
        measurement_labels=[_label(m) for m in dataset.measurements],
        measurement_initial_conditions=[
            dict(getattr(m, "initial_conditions", {}) or {})
            for m in dataset.measurements
        ],
        elpd_loo=float(elpd_data.elpd_loo),
        p_loo=float(elpd_data.p_loo),
        scale=scale,
        elpd_data=elpd_data,
    )


def compare(
    results_map: Dict[str, "HMCResults"],
    dataset: Union[Dataset, Dict[str, Dataset]],
    *,
    yerrs: Union[float, Array, None] = None,
    sigma_source: SigmaSource = "reuse",
    leave_out: LeaveOut = "point",
    integration: Integration = "euler",
    max_draws: Optional[int] = None,
    scale: Literal["log", "negative_log", "deviance"] = "log",
    **compare_kwargs,
):
    """Compare several fits by concentration-space LOO via :func:`arviz.compare`.

    Each fit is scored with its own integrator-based ``log_likelihood`` group so
    that comparison is on the same concentration-space footing -- even when some
    fits are surrogate-mode and others mechanistic.

    Args:
        results_map: Mapping ``name -> HMCResults``.
        dataset: A single :class:`Dataset` used for all fits, or a mapping
            ``name -> Dataset`` matching ``results_map``.
        yerrs, sigma_source, leave_out, integration, max_draws, scale:
            Forwarded to :func:`build_loo_idata` / :func:`arviz.compare`.
        **compare_kwargs: Extra keyword arguments for :func:`arviz.compare`.

    Returns:
        pandas.DataFrame: The ArviZ comparison table.
    """
    idata_map = {}
    for name, res in results_map.items():
        ds = dataset[name] if isinstance(dataset, dict) else dataset
        idata_map[name] = build_loo_idata(
            res,
            ds,
            yerrs=yerrs,
            sigma_source=sigma_source,
            leave_out=leave_out,
            integration=integration,
            max_draws=max_draws,
        )
    return az.compare(idata_map, ic="loo", scale=scale, **compare_kwargs)


def consistency_check(
    results: "HMCResults",
    dataset: Dataset,
    *,
    yerrs: Union[float, Array, None] = None,
    leave_out: LeaveOut = "point",
    max_draws: Optional[int] = None,
    rtol: float = 1e-2,
    atol: float = 1e-1,
) -> dict:
    """Validate the reconstruction against ArviZ's native LOO (mechanistic only).

    For a mechanistic fit, native ``az.loo(az.from_numpyro(mcmc))`` and the
    reconstruction (which reuses the same integrated states and inferred
    ``sigma``) should agree on ``elpd_loo``. Agreement here validates the
    surrogate-mode Euler reconstruction, which has no native counterpart.

    Args:
        results: A mechanistic :class:`HMCResults`.
        dataset: The dataset that was fit.
        yerrs: Unused for ``sigma_source="reuse"`` (kept for signature symmetry).
        leave_out: Must be ``"point"`` to line up with native per-point LOO.
        max_draws: Optional draw subsampling (reconstruction only).
        rtol, atol: Tolerances for the ``elpd_loo`` agreement assertion.

    Returns:
        dict with ``native_elpd``, ``reconstructed_elpd``, ``abs_diff`` and
        ``agree``.
    """
    from catalax.mcmc.mcmc import Modes

    if results.bayesian_model.mode != Modes.MECHANISTIC:
        raise ValueError(
            "consistency_check compares against ArviZ's native log-likelihood, "
            "which is only concentration-space for mechanistic fits. There is no "
            "native counterpart for a surrogate-mode posterior -- that is exactly "
            "why the reconstruction exists."
        )
    if leave_out != "point":
        raise ValueError("consistency_check requires leave_out='point'.")

    native = az.loo(az.from_numpyro(results.mcmc), pointwise=False)
    reconstructed = loo(
        results,
        dataset,
        sigma_source="reuse",
        leave_out="point",
        pointwise=False,
        max_draws=max_draws,
    )

    native_elpd = float(native.elpd_loo)
    recon_elpd = float(reconstructed.elpd_loo)
    abs_diff = abs(native_elpd - recon_elpd)
    agree = bool(np.isclose(native_elpd, recon_elpd, rtol=rtol, atol=atol))
    return {
        "native_elpd": native_elpd,
        "reconstructed_elpd": recon_elpd,
        "abs_diff": abs_diff,
        "agree": agree,
    }
