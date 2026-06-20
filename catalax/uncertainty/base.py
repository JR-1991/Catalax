"""Base abstractions for uncertainty-aware predictors.

This module formalises the *outputs* an uncertainty-quantification method must
provide on top of a normal point prediction. The :class:`NeuralODEEnsemble` is
the prototypical example (mean trajectory + HDI bands); this module distils that
contract into a reusable base class so that other methods -- e.g. GAPA
(Gaussian Process Activations) and the HMC posterior predictive -- expose the
exact same interface and therefore plug straight into ``Dataset.plot``.

The unifying object is :class:`PredictiveDistribution`, which can be backed
either by an *ensemble* of trajectories (sample-based methods such as the
ensemble and HMC) or by *moments* -- a mean and a standard deviation -- as
produced by moment-propagation methods such as GAPA. Both representations
expose the same ``lower``/``upper``/``lower_50``/``upper_50`` HDI bands, so the
plotting machinery does not need to know which method produced them.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional

import jax
import jax.numpy as jnp

from catalax.predictor import Predictor

if TYPE_CHECKING:
    from catalax.dataset import Dataset
    from catalax.model.model import SimulationConfig

HDILevel = Literal["lower", "upper", "lower_50", "upper_50"]

#: Gaussian z-multipliers for moment-backed bands. ``lower``/``upper`` bound the
#: 95% interval (+/- 1.96 sigma) and ``lower_50``/``upper_50`` the 50% interval
#: (+/- 0.674 sigma), matching the percentiles used for sample-backed bands.
_Z: dict[str, float] = {
    "lower": -1.959963984540054,
    "upper": 1.959963984540054,
    "lower_50": -0.6744897501960817,
    "upper_50": 0.6744897501960817,
}

#: Percentiles of the predictive distribution for sample-backed bands.
_PERCENTILE: dict[str, float] = {
    "lower": 2.5,
    "upper": 97.5,
    "lower_50": 25.0,
    "upper_50": 75.0,
}

# Predict results are memoised so that the five calls ``Dataset.plot`` makes
# (mean plus four HDI bands) only run the expensive integration once. The cache
# lives at module scope because predictors may be frozen ``equinox`` modules
# that cannot hold a mutable attribute. The key combines the dataset's memory
# address and its content ``id`` so that neither a deep copy (which shares the
# content id but has a new address) nor a reused address (which carries a new
# content id) collides. Only the same live object hits. Bounded to cap memory.
_DISTRIBUTION_CACHE_MAX = 64
_DISTRIBUTION_CACHE: "OrderedDict[tuple, PredictiveDistribution]" = OrderedDict()


@dataclass
class PredictiveDistribution:
    """A predictive distribution over trajectories for a set of measurements.

    Backed by *either* ``samples`` (an ensemble of trajectories) *or* ``mean``
    and ``std`` (Gaussian moments). The ``values``/``to_dataset`` accessors
    return the centre line (``hdi=None``) or an HDI band, transparently using
    empirical percentiles for the sample-backed case and Gaussian quantiles for
    the moment-backed case.

    Attributes:
        state_order: Ordered state names matching the last array axis.
        times: Time grid, shape ``(n_meas, n_time)``.
        y0s: Initial conditions, shape ``(n_meas, n_state)``.
        mean: Mean trajectory, shape ``(n_meas, n_time, n_state)``.
        std: Predictive standard deviation, same shape as ``mean`` (moment-backed).
        samples: Trajectory ensemble, shape ``(n_samples, n_meas, n_time, n_state)``.
    """

    state_order: List[str]
    times: jax.Array
    y0s: jax.Array
    mean: jax.Array
    std: Optional[jax.Array] = None
    samples: Optional[jax.Array] = None

    @classmethod
    def from_moments(
        cls,
        state_order: List[str],
        times: jax.Array,
        y0s: jax.Array,
        mean: jax.Array,
        std: jax.Array,
    ) -> "PredictiveDistribution":
        """Build a moment-backed distribution (mean + standard deviation)."""
        return cls(state_order=state_order, times=times, y0s=y0s, mean=mean, std=std)

    @classmethod
    def from_samples(
        cls,
        state_order: List[str],
        times: jax.Array,
        y0s: jax.Array,
        samples: jax.Array,
    ) -> "PredictiveDistribution":
        """Build a sample-backed distribution from a trajectory ensemble."""
        mean = jnp.nanmean(samples, axis=0)
        return cls(state_order=state_order, times=times, y0s=y0s, mean=mean, samples=samples)

    def values(self, hdi: Optional[HDILevel] = None) -> jax.Array:
        """Return the centre line (``hdi=None``) or an HDI band as an array."""
        if hdi is None:
            return self.mean
        if self.samples is not None:
            return jnp.nanpercentile(self.samples, _PERCENTILE[hdi], axis=0)
        if self.std is None:
            raise ValueError("Distribution has neither samples nor a standard deviation.")
        return self.mean + _Z[hdi] * self.std

    def to_dataset(self, hdi: Optional[HDILevel] = None) -> "Dataset":
        """Materialise the centre line or an HDI band as a :class:`Dataset`."""
        from catalax.dataset import Dataset

        return Dataset.from_jax_arrays(
            state_order=self.state_order,
            data=self.values(hdi),
            time=self.times,
            y0s=self.y0s,
        )


class UncertaintyPredictor(Predictor, abc.ABC):
    """A :class:`Predictor` that also reports calibrated predictive uncertainty.

    Subclasses implement a single method, :meth:`predict_distribution`, returning
    a :class:`PredictiveDistribution`. The HDI-band ``predict`` contract that
    ``Dataset.plot`` relies on (``has_hdi`` + ``predict(..., hdi=level)``) is then
    provided here once, on top of that method, so every uncertainty method shares
    identical band semantics and caching.
    """

    @abc.abstractmethod
    def predict_distribution(
        self,
        dataset: "Dataset",
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        **kwargs,
    ) -> PredictiveDistribution:
        """Compute the full predictive distribution for ``dataset``."""

    def _cached_distribution(
        self,
        dataset: "Dataset",
        config: Optional["SimulationConfig"],
        n_steps: int,
        use_times: bool,
        **kwargs,
    ) -> PredictiveDistribution:
        key = (id(self), id(dataset), dataset.id, n_steps, bool(use_times), id(config))
        cached = _DISTRIBUTION_CACHE.get(key)
        if cached is None:
            cached = self.predict_distribution(
                dataset, config=config, n_steps=n_steps, use_times=use_times, **kwargs
            )
            _DISTRIBUTION_CACHE[key] = cached
            if len(_DISTRIBUTION_CACHE) > _DISTRIBUTION_CACHE_MAX:
                _DISTRIBUTION_CACHE.popitem(last=False)
        return cached

    def predict(
        self,
        dataset: "Dataset",
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional[HDILevel] = None,
        **kwargs,
    ) -> "Dataset":
        """Predict the mean trajectory (``hdi=None``) or an HDI band.

        The expensive distribution is computed once per ``(dataset, grid)`` and
        cached, so the repeated calls ``Dataset.plot`` makes for the mean and the
        four HDI bands integrate the model only a single time.
        """
        dist = self._cached_distribution(dataset, config, n_steps, use_times, **kwargs)
        return dist.to_dataset(hdi)

    def has_hdi(self) -> bool:
        """Uncertainty predictors always provide HDI bands."""
        return True

    @property
    def has_uncertainty(self) -> bool:
        """Uncertainty predictors always quantify uncertainty."""
        return True
