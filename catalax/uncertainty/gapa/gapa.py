"""GAPA: Gaussian Process Activations for neural ODEs.

A post-hoc, mean-preserving epistemic uncertainty wrapper around a trained
:class:`~catalax.neural.neuralbase.NeuralBase`. It attaches a per-neuron
Gaussian process to one hidden layer (the GP prior mean *is* the activation, so
predictions are unchanged), reads a closed-form activation-space variance, and
propagates it to a trajectory covariance via the augmented Lyapunov ODE in
:mod:`~catalax.uncertainty.gapa.propagate`.

Two variants (arXiv:2502.20966):

* ``"empirical"`` (GAPA-Free) sets the kernel hyperparameters from the cached
  training activations with no optimisation.
* ``"variational"`` (GAPA-Variational) learns the per-neuron signal variance by
  gradient descent on the Gaussian NLL of the one-step-ahead prediction on
  held-out data. The mean stays pinned to the activation, so predictions are
  unchanged.
"""

from __future__ import annotations

from typing import Generic, List, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from catalax.dataset import Dataset
from catalax.neural.neuralbase import NeuralBase
from catalax.surrogate import Surrogate
from catalax.uncertainty.base import PredictiveDistribution, UncertaintyPredictor

from . import gp as _gp
from .activations import make_input, split_mlp
from .propagate import integrate_with_covariance, rate_covariance

T = TypeVar("T", bound=NeuralBase)


class _Decomposition(eqx.Module):
    """Field callables split around the GAPA layer (rebuilt from the model)."""

    rate_fn: object
    to_preact: object
    postact_to_rate: object
    activation: object
    max_time: float


def _decompose(model: NeuralBase, gp_layer: int) -> _Decomposition:
    """Split any ``NeuralBase`` vector field around the GAPA layer.

    Uses the common interface every model exposes -- :meth:`NeuralBase.get_mlp`
    for the network and :meth:`NeuralBase.mlp_output_to_rate` for the
    post-network tail (identity for ``NeuralODE``, the stoichiometry for
    ``RateFlowODE``, the mechanistic-plus-gate term for ``UniversalODE``). GAPA
    therefore never needs to know the concrete subclass: the rate is
    ``mlp_output_to_rate(net(t, y), t, y)`` and the GAPA-layer tail is the same
    map applied to the reconstructed MLP output.
    """
    net = model.get_mlp()
    to_preact, from_postact = split_mlp(net.mlp, gp_layer)

    def rate_fn(t, y):
        return model.mlp_output_to_rate(net(t, y, None), t, y)

    def postact_to_rate(h, t, y):
        return model.mlp_output_to_rate(from_postact(h), t, y)

    return _Decomposition(
        rate_fn=rate_fn,
        to_preact=to_preact,
        postact_to_rate=postact_to_rate,
        activation=net.mlp.activation,
        max_time=net.max_time,
    )


class GAPA(eqx.Module, UncertaintyPredictor, Surrogate, Generic[T]):
    """Gaussian Process Activations wrapper for a trained neural ODE.

    Attributes:
        model: The frozen neural ODE whose predictions are preserved exactly.
        Z: Inducing pre-activations, shape ``(width, M)``.
        log_lengthscale: Per-neuron log RBF lengthscales, shape ``(width,)``.
        log_signal_var: Per-neuron log RBF signal variances, shape ``(width,)``.
        var_chol: Cholesky factors of the optional variational covariance,
            shape ``(width, M, M)``. Held at zero in both variants and kept as
            an extension hook for a full sparse-variational correction.
        obs_noise: Per-state measurement-noise variance added to the band,
            estimated from training residuals unless given explicitly.
    """

    model: T
    Z: jax.Array
    log_lengthscale: jax.Array
    log_signal_var: jax.Array
    var_chol: jax.Array
    obs_noise: jax.Array
    gp_layer: int = eqx.field(static=True)
    variant: str = eqx.field(static=True)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_model(
        cls,
        model: T,
        train_data: Dataset,
        *,
        gp_layer: int = 0,
        n_inducing: int = 40,
        variant: str = "variational",
        val_data: Optional[Dataset] = None,
        signal_var_floor: float = 1.0,
        obs_noise: Optional[float] = None,
        n_iter: int = 300,
        lr: float = 1e-2,
    ) -> "GAPA[T]":
        """Fit GAPA on top of a trained model.

        Args:
            model: A trained ``NeuralBase`` model (``NeuralODE``, ``RateFlowODE``,
                or ``UniversalODE``).
            train_data: Data whose visited activations seed the GP (the inducing
                set and empirical hyperparameters).
            gp_layer: Hidden layer to attach the GP to (``0`` = first hidden layer).
            n_inducing: Inducing points per neuron.
            variant: ``"empirical"`` (GAPA-Free) or ``"variational"`` (GAPA-Variational).
            val_data: Held-out data for variational calibration (defaults to ``train_data``).
            signal_var_floor: Floor on the empirical signal variance, i.e. the
                prior epistemic scale far from any data. The default follows the
                paper (``sigma_f^2 = max(1, Var(activations))``). The shape of the
                band is data-driven, only its out-of-distribution height is set
                here, since in-distribution data cannot inform it.
            obs_noise: Measurement-noise variance folded into the band to make it
                a full predictive interval. When ``None`` (the default) it is
                estimated per state from the model's training residuals. Pass
                ``0.0`` for an epistemic-only band.
            n_iter: Variational optimisation steps.
            lr: Variational learning rate.
        """
        decomp = _decompose(model, gp_layer)
        state_order = model.get_state_order()

        # Cache the pre-activations the model actually visits on the training data.
        pre_acts = _cache_preactivations(train_data, decomp, state_order)

        Z, log_ls, log_var = _gp.fit_empirical(
            np.asarray(pre_acts),
            decomp.activation,
            n_inducing=n_inducing,
            signal_var_floor=signal_var_floor,
        )
        width, m = Z.shape
        var_chol = jnp.zeros((width, m, m))

        if obs_noise is None:
            obs_noise_var = _estimate_obs_noise(model, train_data, state_order)
        else:
            obs_noise_var = jnp.asarray(float(obs_noise))

        instance = cls(
            model=model,
            Z=Z,
            log_lengthscale=log_ls,
            log_signal_var=log_var,
            var_chol=var_chol,
            obs_noise=obs_noise_var,
            gp_layer=gp_layer,
            variant="empirical",
        )

        if variant == "empirical":
            return instance
        if variant != "variational":
            raise ValueError(f"Unknown GAPA variant: {variant!r}")

        return instance._calibrate(
            val_data if val_data is not None else train_data,
            n_iter=n_iter,
            lr=lr,
        )

    # ------------------------------------------------------------------ #
    # Variational calibration
    # ------------------------------------------------------------------ #
    def _calibrate(self, data: Dataset, *, n_iter: int, lr: float) -> "GAPA[T]":
        """Learn the per-neuron signal variance by one-step predictive NLL.

        The objective is the Gaussian negative log-likelihood of the
        one-step-ahead prediction on ``data``, the dynamical analogue of GAPA's
        output-NLL and cheap because it needs no full ODE solve in the loss.

        Only the signal variance is learned. The lengthscale stays at the
        empirical heuristic and the variational covariance at the prior. Because
        the GP posterior variance vanishes at the inducing points regardless of
        the signal variance, this tunes the off-distribution magnitude to the
        data scale while keeping the in-distribution variance near zero. The
        mean, and therefore the prediction, is unchanged.
        """
        decomp = _decompose(self.model, self.gp_layer)
        state_order = self.model.get_state_order()
        obs_idx = jnp.asarray(self.model.observable_indices)

        y_data, times, _ = data.to_jax_arrays(state_order)  # (n_meas, n_time, d)

        params = {"log_signal_var": self.log_signal_var}

        def step_nll(var_fn, yi, ti, dti, ynext):
            rate = decomp.rate_fn(ti, yi)
            mu = yi + dti * rate
            Q = rate_covariance(
                ti,
                yi,
                to_preact=decomp.to_preact,
                postact_to_rate=decomp.postact_to_rate,
                activation=decomp.activation,
                var_fn=var_fn,
                max_time=decomp.max_time,
            )
            var = (dti**2) * jnp.diag(Q) + self.obs_noise + 1e-8
            resid2 = (ynext - mu) ** 2
            term = 0.5 * (resid2 / var + jnp.log(2.0 * jnp.pi * var))
            return jnp.sum(term[obs_idx])

        def meas_nll(var_fn, y_meas, t_meas):
            yi, ynext = y_meas[:-1], y_meas[1:]
            ti = t_meas[:-1]
            dti = t_meas[1:] - t_meas[:-1]
            return jnp.sum(
                jax.vmap(lambda a, b, c, d: step_nll(var_fn, a, b, c, d))(
                    yi, ti, dti, ynext
                )
            )

        def loss(p):
            var_fn = _gp.variance_fn(
                self.log_lengthscale,
                p["log_signal_var"],
                self.Z,
                self.var_chol,
            )
            total = jnp.sum(
                jax.vmap(lambda y, t: meas_nll(var_fn, y, t))(y_data, times)
            )
            n_points = y_data.shape[0] * max(1, y_data.shape[1] - 1)
            return total / n_points

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        @jax.jit
        def update(p, opt_state):
            value, grads = jax.value_and_grad(loss)(p)
            updates, opt_state = optimizer.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)
            return p, opt_state, value

        for _ in range(n_iter):
            params, opt_state, _ = update(params, opt_state)

        return GAPA(
            model=self.model,
            Z=self.Z,
            log_lengthscale=self.log_lengthscale,
            log_signal_var=params["log_signal_var"],
            var_chol=self.var_chol,
            obs_noise=self.obs_noise,
            gp_layer=self.gp_layer,
            variant="variational",
        )

    # ------------------------------------------------------------------ #
    # Predictive distribution
    # ------------------------------------------------------------------ #
    def predict_distribution(
        self,
        dataset: Dataset,
        config=None,
        n_steps: int = 100,
        use_times: bool = False,
        **kwargs,
    ) -> PredictiveDistribution:
        """Propagate the GAPA covariance to a mean + std over each trajectory."""
        decomp = _decompose(self.model, self.gp_layer)
        state_order = self.model.get_state_order()
        times, y0s = _extract_arrays(dataset, config, n_steps, use_times, state_order)

        var_fn = _gp.variance_fn(
            self.log_lengthscale, self.log_signal_var, self.Z, self.var_chol
        )
        d = y0s.shape[-1]
        P0 = jnp.zeros((d, d))

        def per_measurement(ts, y0):
            return integrate_with_covariance(
                decomp.rate_fn,
                to_preact=decomp.to_preact,
                postact_to_rate=decomp.postact_to_rate,
                activation=decomp.activation,
                var_fn=var_fn,
                max_time=decomp.max_time,
                ts=ts,
                y0=y0,
                P0=P0,
            )

        mean, var = jax.vmap(per_measurement)(times, y0s)
        std = jnp.sqrt(jnp.maximum(var + self.obs_noise, 0.0))

        return PredictiveDistribution.from_moments(
            state_order=state_order,
            times=times,
            y0s=y0s,
            mean=mean,
            std=std,
        )

    # ------------------------------------------------------------------ #
    # Predictor / Surrogate interface (delegated to the wrapped model)
    # ------------------------------------------------------------------ #
    def get_state_order(self) -> List[str]:
        return self.model.get_state_order()

    def get_species_order(self) -> List[str]:
        return self.model.get_state_order()

    def n_parameters(self) -> int:
        return self.model.n_parameters()

    def rates(self, t, y, constants=None):
        return self.model.rates(t, y, constants)

    def predict_rates(self, dataset: Dataset, return_individual: bool = False):
        return self.model.predict_rates(dataset)

    def rate_sigma(self, dataset: Dataset) -> jax.Array:
        return self.model.rate_sigma(dataset)

    def rate_uncertainty(self, dataset: Dataset) -> jax.Array:
        """GAPA epistemic rate standard deviation at each measured point."""
        decomp = _decompose(self.model, self.gp_layer)
        state_order = self.model.get_state_order()
        data, times, _ = dataset.to_jax_arrays(state_order)
        n_meas, n_time, d = data.shape
        ys = data.reshape(-1, d)
        ts = times.ravel()
        var_fn = _gp.variance_fn(
            self.log_lengthscale, self.log_signal_var, self.Z, self.var_chol
        )

        def point_std(t, y):
            Q = rate_covariance(
                t,
                y,
                to_preact=decomp.to_preact,
                postact_to_rate=decomp.postact_to_rate,
                activation=decomp.activation,
                var_fn=var_fn,
                max_time=decomp.max_time,
            )
            return jnp.sqrt(jnp.maximum(jnp.diag(Q), 0.0))

        return jax.vmap(point_std)(ts, ys)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _estimate_obs_noise(
    model: NeuralBase, dataset: Dataset, state_order: List[str]
) -> jax.Array:
    """Per-state aleatoric variance from the model's training residuals.

    The mean-squared difference between the data and the model's own fit is a
    standard estimate of the measurement-noise variance for a well-fit model.
    Returns one variance per state, shape ``(n_state,)``.
    """
    data, _, _ = dataset.to_jax_arrays(state_order)
    pred, _, _ = model.predict(dataset, use_times=True).to_jax_arrays(state_order)
    resid = data - pred
    var = jnp.nanmean(resid**2, axis=(0, 1))
    return jnp.nan_to_num(var, nan=0.0)


def _cache_preactivations(
    dataset: Dataset, decomp: _Decomposition, state_order: List[str]
) -> jax.Array:
    """Pre-activations the model visits on ``dataset``, shape ``(n_points, width)``."""
    data, times, _ = dataset.to_jax_arrays(state_order)
    n_meas, n_time, d = data.shape
    ys = data.reshape(-1, d)
    ts = times.ravel()

    def preact(t, y):
        return decomp.to_preact(make_input(t, y, decomp.max_time))

    return jax.vmap(preact)(ts, ys)


def _extract_arrays(
    dataset: Dataset,
    config,
    n_steps: int,
    use_times: bool,
    state_order: List[str],
) -> Tuple[jax.Array, jax.Array]:
    """Time grid and initial conditions, mirroring ``NeuralBase.predict``."""
    if config is None and not use_times:
        config = dataset.to_config(nsteps=n_steps)
    if config and config.nsteps != n_steps:
        config.nsteps = n_steps

    if not dataset.has_data():
        assert config is not None, (
            "Dataset has only initial conditions; a simulation configuration is "
            "required to generate predictions."
        )
        y0s = dataset.to_y0_matrix(state_order)
        times = jnp.linspace(config.t0, config.t1, config.nsteps).T
    else:
        _, times, y0s = dataset.to_jax_arrays(state_order, inits_to_array=True)

    if config:
        times = jnp.linspace(config.t0, config.t1, config.nsteps).T

    return times, y0s
