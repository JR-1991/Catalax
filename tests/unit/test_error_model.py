"""Unit tests for the concentration-space error models.

These focus on how a per-observable ``scale_hint`` (the per-species mean of
``yerrs``) flows into each prior: in the default ``shared=False`` case every
species gets its own default scale, while ``shared=True`` collapses the hint to
a single scalar. The tests inspect the distribution object each model hands to
``numpyro.sample`` (captured via a trace) so the assertions are exact and do
not depend on random draws.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from catalax.mcmc.error import Gamma, HalfNormal, LogNormal
from catalax.mcmc.mcmc import _per_species_scale_hint


def _trace_sample(error_model, n_obs, scale_hint, seed=0):
    """Run ``error_model.sample`` under a seeded trace.

    Returns ``(sigma_y, fn)`` where ``fn`` is the distribution registered at the
    ``sigma_y`` site (``None`` for deterministic models).
    """
    with handlers.seed(rng_seed=seed), handlers.trace() as tr:
        sigma_y = error_model.sample(n_obs, scale_hint)
    site = tr["sigma_y"]
    return sigma_y, site.get("fn")


PER_SPECIES_HINT = jnp.array([0.1, 1.0, 5.0])
N_OBS = 3


class TestPerSpeciesScaleHint:
    """``yerrs`` -> per-species default scale reduction used at the call site."""

    def test_scalar_broadcasts_to_all_species(self):
        # A scalar yerrs is broadcast to the data shape upstream; the reduction
        # yields the same value for every species.
        hint = _per_species_scale_hint(jnp.asarray(0.5), N_OBS)
        assert hint.shape == (N_OBS,)
        np.testing.assert_allclose(hint, np.full(N_OBS, 0.5))

    def test_per_species_vector_kept(self):
        yerrs = jnp.array([0.1, 1.0, 5.0])
        hint = _per_species_scale_hint(yerrs, N_OBS)
        np.testing.assert_allclose(hint, yerrs)

    def test_full_data_array_reduces_to_per_species_mean(self):
        # Shape (n_meas, n_time, n_obs): mean over every leading axis per species.
        yerrs = jnp.stack(
            [
                jnp.full((4, N_OBS), 0.2),
                jnp.full((4, N_OBS), 0.4),
            ]
        )  # (2, 4, 3), all species share a column value here
        # Make species distinct: scale each column.
        yerrs = yerrs * jnp.array([1.0, 2.0, 3.0])
        hint = _per_species_scale_hint(yerrs, N_OBS)
        expected = np.asarray(yerrs).reshape(-1, N_OBS).mean(axis=0)
        assert hint.shape == (N_OBS,)
        np.testing.assert_allclose(hint, expected)

    def test_mismatched_trailing_axis_falls_back_to_global_mean(self):
        # Trailing axis != n_obs -> global mean broadcast across species.
        yerrs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])  # length 5, n_obs == 3
        hint = _per_species_scale_hint(yerrs, N_OBS)
        assert hint.shape == (N_OBS,)
        np.testing.assert_allclose(hint, np.full(N_OBS, float(np.mean(yerrs))))


class TestResolveHint:
    """The base-class helper that maps a hint onto the event shape."""

    @pytest.mark.parametrize("model", [HalfNormal(), LogNormal(), Gamma()])
    def test_per_species_hint_kept_as_vector(self, model):
        resolved = model._resolve_hint(PER_SPECIES_HINT, N_OBS)
        assert resolved.shape == (N_OBS,)
        np.testing.assert_allclose(resolved, PER_SPECIES_HINT)

    @pytest.mark.parametrize(
        "model",
        [HalfNormal(shared=True), LogNormal(shared=True), Gamma(shared=True)],
    )
    def test_shared_collapses_to_scalar_mean(self, model):
        resolved = model._resolve_hint(PER_SPECIES_HINT, N_OBS)
        assert resolved.shape == ()
        np.testing.assert_allclose(resolved, float(np.mean(PER_SPECIES_HINT)))

    @pytest.mark.parametrize("model", [HalfNormal(), LogNormal(), Gamma()])
    def test_scalar_hint_broadcasts(self, model):
        resolved = model._resolve_hint(2.0, N_OBS)
        assert resolved.shape == (N_OBS,)
        np.testing.assert_allclose(resolved, np.full(N_OBS, 2.0))


class TestPerSpeciesPrior:
    """A per-species hint produces a per-species prior scale (not a global one)."""

    def test_halfnormal_scale_is_per_species(self):
        _, fn = _trace_sample(HalfNormal(), N_OBS, PER_SPECIES_HINT)
        # HalfNormal scale equals the per-species hint, one entry per observable.
        assert fn.scale.shape == (N_OBS,)
        np.testing.assert_allclose(fn.scale, PER_SPECIES_HINT)

    def test_lognormal_median_is_per_species(self):
        _, fn = _trace_sample(LogNormal(sigma=0.7), N_OBS, PER_SPECIES_HINT)
        # LogNormal loc is log(median); median defaults to the per-species hint.
        assert fn.loc.shape == (N_OBS,)
        np.testing.assert_allclose(fn.loc, np.log(np.asarray(PER_SPECIES_HINT)))

    def test_gamma_rate_is_per_species(self):
        conc = 2.0
        _, fn = _trace_sample(Gamma(concentration=conc), N_OBS, PER_SPECIES_HINT)
        # Rate defaults so the prior mean (conc / rate) equals the hint per species.
        assert fn.rate.shape == (N_OBS,)
        np.testing.assert_allclose(fn.rate, conc / np.asarray(PER_SPECIES_HINT))

    def test_distinct_hints_give_distinct_scales(self):
        # Distinct per-species hints map to distinct prior scales rather than a
        # single shared value.
        _, fn = _trace_sample(HalfNormal(), N_OBS, PER_SPECIES_HINT)
        assert len(np.unique(np.asarray(fn.scale))) == N_OBS


class TestSharedPrior:
    """``shared=True`` uses a single scalar scale (the mean of the hint)."""

    def test_halfnormal_shared_scalar_scale(self):
        sigma_y, fn = _trace_sample(HalfNormal(shared=True), N_OBS, PER_SPECIES_HINT)
        assert fn.scale.shape == ()
        np.testing.assert_allclose(fn.scale, float(np.mean(PER_SPECIES_HINT)))
        # Returned std is still broadcast to one value per observable.
        assert sigma_y.shape == (N_OBS,)

    def test_lognormal_shared_scalar_loc(self):
        _, fn = _trace_sample(LogNormal(shared=True), N_OBS, PER_SPECIES_HINT)
        assert fn.loc.shape == ()
        np.testing.assert_allclose(fn.loc, float(np.log(np.mean(PER_SPECIES_HINT))))


class TestExplicitScaleOverridesHint:
    """An explicit model scale ignores the hint entirely (backward compatible)."""

    def test_halfnormal_explicit_scale(self):
        _, fn = _trace_sample(HalfNormal(scale=0.3), N_OBS, PER_SPECIES_HINT)
        np.testing.assert_allclose(fn.scale, np.full(N_OBS, 0.3))

    def test_gamma_explicit_rate(self):
        sigma_y, fn = _trace_sample(
            Gamma(concentration=2.0, rate=4.0), N_OBS, PER_SPECIES_HINT
        )
        # An explicit rate stays scalar (and broadcasts); the per-observable axis
        # comes from concentration, so the hint is ignored entirely.
        np.testing.assert_allclose(np.asarray(fn.rate), 4.0)
        assert sigma_y.shape == (N_OBS,)


class TestSampleShapeAndPositivity:
    """Sanity: every model returns a positive per-observable std."""

    @pytest.mark.parametrize(
        "model",
        [HalfNormal(), LogNormal(), Gamma(), HalfNormal(shared=True)],
    )
    def test_sample_is_positive_per_observable(self, model):
        sigma_y, _ = _trace_sample(model, N_OBS, PER_SPECIES_HINT)
        assert sigma_y.shape == (N_OBS,)
        assert np.all(np.asarray(sigma_y) > 0.0)
