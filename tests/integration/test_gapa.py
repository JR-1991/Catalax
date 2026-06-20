import jax.numpy as jnp
import matplotlib
import pytest

import catalax as ctx
import catalax.neural as ctn
from catalax.uncertainty import GAPA
from catalax.uncertainty.base import PredictiveDistribution, UncertaintyPredictor

matplotlib.use("Agg")


def _mm_setup():
    """Small Michaelis-Menten model + a short noisy training dataset."""
    model = ctx.Model(name="MM")
    model.add_state(s1="Substrate")
    model.add_constant(e="Enzyme")
    model.add_ode("s1", "-kcat * e * s1 / (k_m + s1)")
    model.parameters["kcat"].value = 3.0
    model.parameters["k_m"].value = 80.0

    train = ctx.Dataset.from_model(model)
    for s0 in (40.0, 80.0, 120.0):
        train.add_initial(s1=s0, e=1.0)
    config = ctx.SimulationConfig(nsteps=20, t0=0.0, t1=60.0)
    train = model.simulate(dataset=train, config=config)

    node = ctn.NeuralODE.from_model(model, width_size=8, depth=2)
    return model, train, node


class TestGAPA:
    def test_is_uncertainty_predictor(self):
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12)
        assert isinstance(gapa, UncertaintyPredictor)
        assert gapa.has_hdi() is True
        assert gapa.has_uncertainty is True

    def test_mean_preservation(self):
        """GAPA's mean trajectory matches the wrapped model's prediction."""
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12)

        gapa_mean, _, _ = gapa.predict(train, n_steps=20).to_jax_arrays(["s1"])
        node_mean, _, _ = node.predict(train, n_steps=20).to_jax_arrays(["s1"])

        # Mean-preserving by construction; only ODE-solver tolerance differs.
        rel = jnp.max(jnp.abs(gapa_mean - node_mean)) / jnp.max(jnp.abs(node_mean))
        assert float(rel) < 1e-2

    def test_band_widens_out_of_distribution(self):
        """The epistemic band is ~0 in-distribution and grows out-of-distribution."""
        model, train, node = _mm_setup()
        config = ctx.SimulationConfig(nsteps=20, t0=0.0, t1=60.0)
        gapa = GAPA.from_model(
            node, train, variant="empirical", n_inducing=20, obs_noise=0.0
        )

        eval_ds = ctx.Dataset.from_model(model)
        eval_ds.add_initial(s1=80.0, e=1.0)  # in-distribution
        eval_ds.add_initial(s1=300.0, e=1.0)  # out-of-distribution
        eval_ds = model.simulate(dataset=eval_ds, config=config)

        lo = gapa.predict(eval_ds, n_steps=20, hdi="lower").to_jax_arrays(["s1"])[0]
        hi = gapa.predict(eval_ds, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        width = (hi - lo)[:, :, 0].max(axis=1)

        # The epistemic band is small in-distribution and clearly wider OOD.
        assert float(width[1]) > 2.5 * float(width[0])

    def test_predict_distribution_moment_backed(self):
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12)
        dist = gapa.predict_distribution(train, n_steps=20)
        assert isinstance(dist, PredictiveDistribution)
        assert dist.std is not None  # moment-backed
        assert dist.mean.shape == dist.std.shape

    def test_variational_runs_and_preserves_mean(self):
        _, train, node = _mm_setup()
        noisy = train.augment(n_augmentations=3, sigma=4.0)
        gapa = GAPA.from_model(
            node, noisy, variant="variational", n_inducing=15,
            obs_noise=16.0, n_iter=40, lr=2e-2,
        )
        assert gapa.variant == "variational"
        gapa_mean, _, _ = gapa.predict(train, n_steps=20).to_jax_arrays(["s1"])
        node_mean, _, _ = node.predict(train, n_steps=20).to_jax_arrays(["s1"])
        rel = jnp.max(jnp.abs(gapa_mean - node_mean)) / jnp.max(jnp.abs(node_mean))
        assert float(rel) < 1e-2  # calibration never touches the mean

    def test_plots_through_dataset_plot(self):
        """GAPA plugs into Dataset.plot as a Predictor with HDI bands."""
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12)
        fig = train.plot(predictor=gapa, bands=True, show=False)
        assert fig is not None

    def test_band_zero_at_initial_time(self):
        """With a known initial condition (P0 = 0) the band starts at zero width."""
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12, obs_noise=0.0)
        lo = gapa.predict(train, n_steps=20, hdi="lower").to_jax_arrays(["s1"])[0]
        hi = gapa.predict(train, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        width_t0 = (hi - lo)[:, 0, 0]
        assert float(jnp.max(jnp.abs(width_t0))) < 1e-4

    def test_obs_noise_widens_band(self):
        """Adding observation noise widens the predictive band everywhere."""
        _, train, node = _mm_setup()
        kw = dict(variant="empirical", n_inducing=12)
        g0 = GAPA.from_model(node, train, obs_noise=0.0, **kw)
        gn = GAPA.from_model(node, train, obs_noise=100.0, **kw)

        def max_width(g):
            lo = g.predict(train, n_steps=20, hdi="lower").to_jax_arrays(["s1"])[0]
            hi = g.predict(train, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
            return float(jnp.max(hi - lo))

        assert max_width(gn) > max_width(g0)

    def test_signal_var_floor_scales_ood_band(self):
        """A larger signal-variance prior widens the OOD band; ID stays ~0."""
        model, train, node = _mm_setup()
        config = ctx.SimulationConfig(nsteps=20, t0=0.0, t1=60.0)
        eval_ds = ctx.Dataset.from_model(model)
        eval_ds.add_initial(s1=80.0, e=1.0)
        eval_ds.add_initial(s1=300.0, e=1.0)
        eval_ds = model.simulate(dataset=eval_ds, config=config)

        def ood_id_width(floor):
            g = GAPA.from_model(
                node, train, variant="empirical", n_inducing=20,
                signal_var_floor=floor, obs_noise=0.0,
            )
            lo = g.predict(eval_ds, n_steps=20, hdi="lower").to_jax_arrays(["s1"])[0]
            hi = g.predict(eval_ds, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
            w = (hi - lo)[:, :, 0].max(axis=1)
            return float(w[1]), float(w[0])

        ood_small, _ = ood_id_width(1.0)
        ood_large, _ = ood_id_width(50.0)
        assert ood_large > ood_small

    def test_rate_uncertainty_surrogate(self):
        """rate_uncertainty returns per-point std, larger out-of-distribution."""
        model, train, node = _mm_setup()
        config = ctx.SimulationConfig(nsteps=20, t0=0.0, t1=60.0)
        eval_ds = ctx.Dataset.from_model(model)
        eval_ds.add_initial(s1=80.0, e=1.0)  # ID
        eval_ds.add_initial(s1=300.0, e=1.0)  # OOD
        eval_ds = model.simulate(dataset=eval_ds, config=config)

        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=20)
        ru = gapa.rate_uncertainty(eval_ds)  # (n_meas * n_time, n_state)
        assert ru.shape == (2 * 20, 1)
        assert jnp.all(ru >= 0.0)
        id_max = jnp.max(ru[:20])
        ood_max = jnp.max(ru[20:])
        assert float(ood_max) > float(id_max)

    def test_delegates_to_wrapped_model(self):
        """Predictor/Surrogate passthroughs match the wrapped model."""
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12)
        assert gapa.model is node
        assert gapa.get_state_order() == node.get_state_order()
        assert gapa.n_parameters() == node.n_parameters()
        rates_g = gapa.predict_rates(train)
        rates_n = node.predict_rates(train)
        assert jnp.allclose(rates_g, rates_n)

    def test_empirical_has_no_variational_covariance(self):
        _, train, node = _mm_setup()
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=15)
        assert float(jnp.abs(gapa.var_chol).max()) == 0.0
        assert gapa.Z.shape[0] == node.func.mlp.width_size
        assert gapa.Z.shape[1] <= 15

    def test_unknown_variant_raises(self):
        _, train, node = _mm_setup()
        with pytest.raises(ValueError):
            GAPA.from_model(node, train, variant="bogus")

    def test_obs_noise_estimated_by_default(self):
        """obs_noise=None estimates a per-state variance from training residuals."""
        _, train, node = _mm_setup()
        auto = GAPA.from_model(node, train, variant="empirical")
        zero = GAPA.from_model(node, train, variant="empirical", obs_noise=0.0)
        assert auto.obs_noise.shape == (len(node.get_state_order()),)
        assert jnp.all(auto.obs_noise >= 0.0)
        assert float(jnp.sum(auto.obs_noise)) > 0.0
        # The estimated floor widens the band relative to an epistemic-only one.
        wide = auto.predict(train, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        tight = zero.predict(train, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        assert float(jnp.nanmax(wide - tight)) > 0.0

    def test_rateflow_ode_supported(self):
        """GAPA wraps a RateFlowODE, propagating through the stoichiometry."""
        model = ctx.Model(name="MM2")
        model.add_state(s1="Substrate", p="Product")
        model.add_constant(e="Enzyme")
        model.add_ode("s1", "-kcat * e * s1 / (k_m + s1)")
        model.add_ode("p", "kcat * e * s1 / (k_m + s1)")
        model.parameters["kcat"].value = 2.0
        model.parameters["k_m"].value = 80.0
        ds = ctx.Dataset.from_model(model)
        ds.add_initial(s1=80.0, p=0.0, e=1.0)
        ds.add_initial(s1=120.0, p=0.0, e=1.0)
        config = ctx.SimulationConfig(nsteps=15, t0=0.0, t1=40.0)
        sim = model.simulate(dataset=ds, config=config)

        net = ctn.RateFlowODE.from_model(model, width_size=6, depth=2, reaction_size=2)
        gapa = GAPA.from_model(net, sim, variant="empirical", n_inducing=10, obs_noise=0.0)

        order = model.get_state_order()
        gm, _, _ = gapa.predict(sim, n_steps=15).to_jax_arrays(order)
        nm, _, _ = net.predict(sim, n_steps=15).to_jax_arrays(order)
        rel = jnp.max(jnp.abs(gm - nm)) / (jnp.max(jnp.abs(nm)) + 1e-8)
        assert float(rel) < 1e-2
        lo, _, _ = gapa.predict(sim, n_steps=15, hdi="lower").to_jax_arrays(order)
        hi, _, _ = gapa.predict(sim, n_steps=15, hdi="upper").to_jax_arrays(order)
        assert bool(jnp.isfinite(hi - lo).all())
        assert float(jnp.max(hi - lo)) > 0.0

    def test_universal_ode_supported(self):
        """GAPA wraps a UniversalODE, injecting uncertainty only via the neural term."""
        model = ctx.Model(name="MMu")
        model.add_state(s1="Substrate")
        model.add_ode("s1", "-vmax * s1 / (km + s1)")
        model.parameters["vmax"].value = 2.0
        model.parameters["km"].value = 80.0
        ds = ctx.Dataset.from_model(model)
        ds.add_initial(s1=80.0)
        ds.add_initial(s1=120.0)
        config = ctx.SimulationConfig(nsteps=15, t0=0.0, t1=40.0)
        sim = model.simulate(dataset=ds, config=config)

        net = ctn.UniversalODE.from_model(model, width_size=6, depth=1)
        gapa = GAPA.from_model(net, sim, variant="empirical", n_inducing=10, obs_noise=0.0)

        gm, _, _ = gapa.predict(sim, n_steps=15).to_jax_arrays(["s1"])
        nm, _, _ = net.predict(sim, n_steps=15).to_jax_arrays(["s1"])
        rel = jnp.max(jnp.abs(gm - nm)) / (jnp.max(jnp.abs(nm)) + 1e-8)
        assert float(rel) < 1e-2
        lo, _, _ = gapa.predict(sim, n_steps=15, hdi="lower").to_jax_arrays(["s1"])
        hi, _, _ = gapa.predict(sim, n_steps=15, hdi="upper").to_jax_arrays(["s1"])
        assert bool(jnp.isfinite(hi - lo).all())

    def test_rateflow_variational_calibration(self):
        """Variational calibration runs through the stoichiometric tail."""
        model = ctx.Model(name="MM2v")
        model.add_state(s1="Substrate", p="Product")
        model.add_ode("s1", "-vmax * s1 / (km + s1)")
        model.add_ode("p", "vmax * s1 / (km + s1)")
        model.parameters["vmax"].value = 2.0
        model.parameters["km"].value = 80.0
        ds = ctx.Dataset.from_model(model)
        ds.add_initial(s1=80.0, p=0.0)
        config = ctx.SimulationConfig(nsteps=12, t0=0.0, t1=40.0)
        sim = model.simulate(dataset=ds, config=config)
        noisy = sim.augment(n_augmentations=3, sigma=3.0)

        net = ctn.RateFlowODE.from_model(model, width_size=6, depth=2, reaction_size=2)
        gapa = GAPA.from_model(
            net, noisy, variant="variational", n_inducing=10,
            obs_noise=9.0, n_iter=30, lr=2e-2,
        )
        assert gapa.variant == "variational"
        gm, _, _ = gapa.predict(sim, n_steps=12).to_jax_arrays(model.get_state_order())
        assert bool(jnp.isfinite(gm).all())

    def test_cache_distinguishes_deep_copies(self):
        """A deep copy shares the content id, so the predict cache must not collide."""
        import copy

        model, train, node = _mm_setup()
        config = ctx.SimulationConfig(nsteps=20, t0=0.0, t1=60.0)
        gapa = GAPA.from_model(node, train, variant="empirical", n_inducing=12, obs_noise=0.0)

        ds_a = ctx.Dataset.from_model(model)
        ds_a.add_initial(s1=80.0, e=1.0)
        ds_a = model.simulate(dataset=ds_a, config=config)

        ds_b = copy.deepcopy(ds_a)  # shares ds_a.id
        ds_b.measurements[0].initial_conditions["s1"] = 300.0
        assert ds_b.id == ds_a.id

        band_a = gapa.predict(ds_a, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        band_b = gapa.predict(ds_b, n_steps=20, hdi="upper").to_jax_arrays(["s1"])[0]
        # Different initial conditions must give different bands, not a stale hit.
        assert not jnp.allclose(band_a, band_b)
