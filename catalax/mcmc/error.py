"""User-configurable observation-noise (error) models for Bayesian inference.

An :class:`ErrorModel` specifies how the aleatoric measurement-noise standard
deviation is distributed in **concentration space**. Catalax samples it (the
``"sigma_y"`` site) and applies it as the likelihood scale -- directly for a
mechanistic (integrated) fit, or pushed forward through the surrogate
rate-Jacobian for a rate-matching fit. Error models never see the pushforward;
they only define the prior, so the same object works in either mode.

Select one via :class:`~catalax.mcmc.MCMCConfig`::

    import catalax.mcmc as cmm
    config = cmm.MCMCConfig(
        num_warmup=1000,
        num_samples=1000,
        error_model=cmm.error.LogNormal(sigma=0.7),   # funnel-safe, per species
    )
"""

from __future__ import annotations

import abc
from typing import Optional, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array

#: Name of the numpyro site every error model samples.
SITE = "sigma_y"


class ErrorModel(abc.ABC):
    """Concentration-space prior on the observation-noise std.

    Subclasses implement :meth:`sample`, which draws the ``"sigma_y"`` site and
    returns a per-observable standard deviation. ``shared=True`` collapses it to
    a single scalar broadcast across observables.
    """

    shared: bool = False

    @abc.abstractmethod
    def sample(self, n_obs: int, scale_hint: Array) -> Array:
        """Sample the per-observable concentration-space noise std.

        Args:
            n_obs: Number of observable states.
            scale_hint: A sensible default scale (Catalax passes ``mean(yerrs)``),
                used when the model's own scale is left unset.

        Returns:
            Array of shape ``(n_obs,)`` -- the concentration-space noise std.
        """

    def _event_shape(self, n_obs: int) -> tuple:
        return () if self.shared else (n_obs,)


class HalfNormal(ErrorModel):
    """Half-normal prior. Mode at 0 -- Catalax's historical default.

    Note: an unidentified ``sigma_y`` component can rail at the 0 boundary and
    cause divergences; prefer :class:`LogNormal` for per-species noise.

    Args:
        scale: Half-normal scale; defaults to ``mean(yerrs)``.
        shared: Use a single scalar instead of one std per observable.
    """

    def __init__(self, scale: Optional[float] = None, shared: bool = False):
        self.scale = scale
        self.shared = shared

    def sample(self, n_obs: int, scale_hint: Array) -> Array:
        scale = self.scale if self.scale is not None else jnp.mean(scale_hint)
        sigma_y = numpyro.sample(
            SITE, dist.HalfNormal(scale * jnp.ones(self._event_shape(n_obs)))
        )
        return jnp.broadcast_to(sigma_y, (n_obs,))


class LogNormal(ErrorModel):
    """Log-normal prior. Zero density at 0, sampled in unbounded log-space, so a
    weakly-identified component settles at the median instead of funneling into
    the boundary.

    Args:
        median: Median noise std; defaults to ``mean(yerrs)``.
        sigma: Log-space standard deviation (spread). ``0.7`` ~ a 4x band.
        shared: Use a single scalar instead of one std per observable.
    """

    def __init__(
        self,
        median: Optional[float] = None,
        sigma: float = 0.7,
        shared: bool = False,
    ):
        self.median = median
        self.sigma = sigma
        self.shared = shared

    def sample(self, n_obs: int, scale_hint: Array) -> Array:
        median = self.median if self.median is not None else jnp.mean(scale_hint)
        loc = jnp.log(median) * jnp.ones(self._event_shape(n_obs))
        sigma_y = numpyro.sample(SITE, dist.LogNormal(loc, self.sigma))
        return jnp.broadcast_to(sigma_y, (n_obs,))


class Gamma(ErrorModel):
    """Gamma prior. Zero density at 0 for ``concentration > 1``.

    Args:
        concentration: Gamma shape (``> 1`` keeps mass off 0).
        rate: Gamma rate; defaults so the mean equals ``mean(yerrs)``.
        shared: Use a single scalar instead of one std per observable.
    """

    def __init__(
        self,
        concentration: float = 2.0,
        rate: Optional[float] = None,
        shared: bool = False,
    ):
        self.concentration = concentration
        self.rate = rate
        self.shared = shared

    def sample(self, n_obs: int, scale_hint: Array) -> Array:
        rate = (
            self.rate
            if self.rate is not None
            else self.concentration / jnp.mean(scale_hint)
        )
        conc = self.concentration * jnp.ones(self._event_shape(n_obs))
        sigma_y = numpyro.sample(SITE, dist.Gamma(conc, rate))
        return jnp.broadcast_to(sigma_y, (n_obs,))


class Fixed(ErrorModel):
    """Fixed (non-sampled) noise std -- e.g. estimated from LOO residuals.

    Recorded as a deterministic ``"sigma_y"`` site so the posterior-predictive
    and LOO machinery read it like any sampled noise.

    Args:
        sigma_y: Scalar or per-observable concentration noise std.
    """

    def __init__(self, sigma_y: Union[float, Array]):
        self.sigma_y = sigma_y

    def sample(self, n_obs: int, scale_hint: Array) -> Array:
        sigma_y = jnp.broadcast_to(jnp.asarray(self.sigma_y, dtype=float), (n_obs,))
        numpyro.deterministic(SITE, sigma_y)
        return sigma_y


#: Used when ``MCMCConfig.error_model`` is left unset (preserves prior behavior).
DEFAULT_ERROR_MODEL: ErrorModel = HalfNormal()
