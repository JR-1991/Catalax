import random

import jax.numpy as jnp
import numpy as np
import optax
import pytest

import catalax.mcmc as cmc
import catalax.mcmc as cmm
import catalax.neural as ctn
from catalax.dataset.dataset import Dataset
from catalax.mcmc.models import estimate_initials
from catalax.mcmc.priors import Uniform
from catalax.model.model import Model
from catalax.model.simconfig import SimulationConfig


class TestMCMC:
    def test_mcmc(self, generate_data):
        model, dataset = generate_data
        config = cmm.MCMCConfig(
            num_warmup=100,
            num_samples=100,
        )

        cmm.run_mcmc(
            model=model,
            dataset=dataset,
            config=config,
            yerrs=1e-5,
        )

    def test_surrogate_model(self, generate_data):
        model, dataset = generate_data
        config = cmm.MCMCConfig(
            num_warmup=100,
            num_samples=100,
        )

        dataset = dataset.augment(n_augmentations=10)

        # Create a neural ODE model
        rbf = ctn.RBFLayer(0.2)
        neural_ode = ctn.NeuralODE.from_model(
            model,
            width_size=10,
            depth=3,
            activation=rbf,  # type: ignore
        )

        penalties = ctn.Penalties.for_neural_ode(
            l2_alpha=1e-3,
            l1_alpha=1e-3,
        )

        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-2,
            length=0.1,
            steps=200,
            batch_size=15,
            penalties=penalties,
            loss=optax.log_cosh,
        )

        neural_ode = ctn.train_neural_ode(
            model=neural_ode,
            dataset=dataset,
            strategy=strategy,
            print_every=10,
            weight_scale=1e-7,
        )

        cmm.run_mcmc(
            model=model,
            dataset=dataset,
            config=config,
            surrogate=neural_ode,
            yerrs=1e-5,
        )

    def test_loo_mechanistic(self, generate_data):
        """Mechanistic LOO returns a valid ELPDData over concentration points."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)
        results = cmm.run_mcmc(model=model, dataset=dataset, config=config, yerrs=1e-2)

        # Reusing the inferred noise keeps every (measurement, time, obs) point.
        loo_point = results.loo(dataset, leave_out="point")
        n_obs = int(loo_point.n_data_points)
        assert (
            n_obs == dataset.to_jax_arrays(model.get_observable_state_order())[0].size
        )

        # Leave-one-curve-out collapses each measurement series to one unit.
        loo_curve = results.loo(dataset, leave_out="curve")
        assert int(loo_curve.n_data_points) == len(dataset.measurements)

    def test_loo_consistency_check(self, generate_data):
        """Eval-model reconstruction must match ArviZ native LOO (mechanistic)."""
        model, dataset = generate_data
        # Add observation noise so the discrepancy ``sigma`` is identifiable; on
        # noiseless data it collapses toward zero and the reuse reconstruction
        # divides a tiny re-integration residual by a near-zero scale.
        dataset = dataset.augment(n_augmentations=8, sigma=1e-2)
        config = cmm.MCMCConfig(num_warmup=100, num_samples=200, num_chains=2, verbose=0)
        results = cmm.run_mcmc(model=model, dataset=dataset, config=config, yerrs=1e-2)

        check = results.loo_consistency_check(dataset, yerrs=1e-2)
        assert check["agree"], check

    def test_loo_surrogate(self, generate_data):
        """Surrogate-mode posterior still yields concentration-space LOO."""
        model, dataset = generate_data
        aug = dataset.augment(n_augmentations=10)

        rbf = ctn.RBFLayer(0.2)
        neural_ode = ctn.NeuralODE.from_model(
            model,
            width_size=8,
            depth=2,
            activation=rbf,  # type: ignore
        )
        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-2, length=0.1, steps=100, batch_size=15, loss=optax.log_cosh
        )
        neural_ode = ctn.train_neural_ode(
            model=neural_ode,
            dataset=aug,
            strategy=strategy,
            print_every=1000,
            weight_scale=1e-7,
        )

        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)
        results = cmm.run_mcmc(
            model=model,
            dataset=aug,
            config=config,
            surrogate=neural_ode,
            yerrs=1e-2,
        )

        # Reuse the sampled rates, Euler-integrate, and score against the
        # *measured* concentrations -- not the surrogate rates. The stored yerrs
        # is rate-space for a surrogate fit, so pass a concentration-space one.
        loo_res = results.loo(dataset, yerrs=0.5)
        assert int(loo_res.n_data_points) > 0
        # One Pareto-k per held-out data point (the headline diagnostic).
        assert np.asarray(loo_res.pareto_k).shape[0] == int(loo_res.n_data_points)

        # One-step-ahead integration is also available.
        loo_onestep = results.loo(dataset, yerrs=0.5, integration="euler_onestep")
        assert int(loo_onestep.n_data_points) > 0

        # The reuse-the-inferred-noise variant is also still available.
        loo_reuse = results.loo(dataset, sigma_source="reuse")
        assert int(loo_reuse.n_data_points) > 0

    def test_loo_compare(self, generate_data):
        """compare() ranks two fits on the same concentration-space footing."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)

        res_a = cmm.run_mcmc(model=model, dataset=dataset, config=config, yerrs=1e-2)
        res_b = cmm.run_mcmc(model=model, dataset=dataset, config=config, yerrs=1e-2)

        table = res_a.compare({"other": res_b}, dataset)
        assert set(table.index) == {"self", "other"}

    def test_loo_plots(self, generate_data):
        """Pointwise mapping and both LOO diagnostic plots render."""
        import matplotlib

        matplotlib.use("Agg")

        model, dataset = generate_data
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)
        results = cmm.run_mcmc(model=model, dataset=dataset, config=config, yerrs=1e-2)

        pw = results.loo_pointwise(dataset, yerrs=0.5)
        n_meas = len(dataset.measurements)
        n_obs = len(model.get_observable_state_order())
        assert pw.elpd.shape[0] == n_meas
        assert pw.elpd.shape[2] == n_obs
        assert pw.pareto_k.shape == pw.elpd.shape

        # Influence overlay (marker size = influence) and both heatmaps.
        assert results.plot_loo_influence(dataset, yerrs=0.5) is not None
        assert results.plot_loo_heatmap(dataset, metric="elpd", yerrs=0.5) is not None
        assert (
            results.plot_loo_heatmap(dataset, metric="pareto_k", yerrs=0.5) is not None
        )

    @pytest.mark.parametrize(
        "error_model",
        [
            cmm.error.HalfNormal(),
            cmm.error.LogNormal(sigma=0.7),
            cmm.error.Gamma(concentration=2.0),
        ],
        ids=["halfnormal", "lognormal", "gamma"],
    )
    def test_error_model_samples_sigma_y(self, generate_data, error_model):
        """Every error model samples a positive concentration-space ``sigma_y``."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(
            num_warmup=50, num_samples=100, verbose=0, error_model=error_model
        )
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=1e-2
        )

        samples = results.get_samples()
        assert "sigma_y" in samples
        sigma_y = np.asarray(samples["sigma_y"])
        # One std per observable (per-species default), drawn for every sample.
        n_obs = len(model.get_observable_state_order())
        assert sigma_y.shape == (config.num_samples, n_obs)
        assert np.all(np.isfinite(sigma_y))
        assert np.all(sigma_y > 0.0)

    def test_error_model_fixed_is_deterministic(self, generate_data):
        """``Fixed`` records the supplied noise as a constant deterministic site."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(
            num_warmup=30,
            num_samples=60,
            verbose=0,
            error_model=cmm.error.Fixed(sigma_y=0.05),
        )
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=1e-2
        )

        sigma_y = np.asarray(results.get_samples()["sigma_y"])
        # Constant across every draw -- it is not sampled, just recorded.
        assert np.allclose(sigma_y, 0.05)

    def test_error_model_shared_collapses_to_scalar(self, generate_data):
        """``shared=True`` broadcasts a single scalar std across observables."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(
            num_warmup=30,
            num_samples=60,
            verbose=0,
            error_model=cmm.error.LogNormal(sigma=0.5, shared=True),
        )
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=1e-2
        )

        # A shared site has no per-observable axis (scalar event shape).
        sigma_y = np.asarray(results.get_samples()["sigma_y"])
        assert sigma_y.shape == (config.num_samples,)

    def test_per_species_yerrs_array(self, generate_data):
        """A per-species ``yerrs`` array runs end-to-end and yields per-obs sigma_y.

        The per-species means set the default ``sigma_y`` prior scale, and the
        array form is accepted by ``run_mcmc``, producing one positive std per
        observable.
        """
        model, dataset = generate_data
        n_obs = len(model.get_observable_state_order())
        yerrs = jnp.array([1e-2 * (i + 1) for i in range(n_obs)])

        config = cmm.MCMCConfig(
            num_warmup=30,
            num_samples=60,
            verbose=0,
            error_model=cmm.error.LogNormal(sigma=0.7),
        )
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=yerrs
        )

        sigma_y = np.asarray(results.get_samples()["sigma_y"])
        assert sigma_y.shape == (config.num_samples, n_obs)
        assert np.all(sigma_y > 0.0)

    def _stack_band(self, model, band_dataset):
        """Stack an observable band/trajectory Dataset to ``(n_meas, n_time, n_obs)``."""
        obs_states = model.get_observable_state_order()
        return np.stack(
            [
                np.stack(
                    [np.asarray(m.data[s]) for s in obs_states], axis=-1
                )
                for m in band_dataset.measurements
            ]
        )

    def test_predict_band_ordering_and_noise(self, generate_data):
        """Bands nest correctly and the predictive interval contains the epistemic one."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=1e-2
        )

        n_meas = len(dataset.measurements)
        n_obs = len(model.get_observable_state_order())

        # Median trajectory is a plottable Dataset over the observable states.
        median = results.predict(dataset, hdi=None, n_steps=40)
        median_arr = self._stack_band(model, median)
        assert median_arr.shape == (n_meas, 40, n_obs)
        assert np.all(np.isfinite(median_arr))

        # Epistemic-only band: lower <= upper everywhere.
        lo = self._stack_band(
            model, results.predict(dataset, hdi="lower", n_steps=40, include_noise=False)
        )
        hi = self._stack_band(
            model, results.predict(dataset, hdi="upper", n_steps=40, include_noise=False)
        )
        assert np.all(hi - lo >= -1e-8)

        # Folding in aleatoric noise can only widen the interval.
        lo_pred = self._stack_band(
            model, results.predict(dataset, hdi="lower", n_steps=40, include_noise=True)
        )
        hi_pred = self._stack_band(
            model, results.predict(dataset, hdi="upper", n_steps=40, include_noise=True)
        )
        epistemic_width = float(np.mean(hi - lo))
        predictive_width = float(np.mean(hi_pred - lo_pred))
        assert predictive_width >= epistemic_width - 1e-8

    def test_posterior_predictive_ensemble_shapes(self, generate_data):
        """The raw draw ensemble has one trajectory per (subsampled) posterior draw."""
        model, dataset = generate_data
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, verbose=0)
        results = cmm.run_mcmc(
            model=model, dataset=dataset, config=config, yerrs=1e-2
        )

        n_meas = len(dataset.measurements)
        n_obs = len(model.get_observable_state_order())
        times, values, obs_states = results.posterior_predictive_ensemble(
            dataset, n_steps=30, max_draws=25
        )

        assert times.shape == (n_meas, 30)
        n_draws = values.shape[0]
        assert 0 < n_draws <= 25
        assert values.shape == (n_draws, n_meas, 30, n_obs)
        assert obs_states == list(model.get_observable_state_order())
        assert np.all(np.isfinite(values))

    def test_initial_estimator(self):
        # Create a simple Michaelis-Menten model
        model = Model(name="test")
        model.add_state(s0="Substrate")
        model.add_ode("s0", "-v_max * s0 / (K_m + s0)")

        # Set parameter values and priors
        model.parameters["v_max"].value = 7.0
        model.parameters["v_max"].prior = Uniform(low=1e-6, high=10.0)
        model.parameters["K_m"].value = 100.0
        model.parameters["K_m"].prior = Uniform(low=1e-6, high=1000.0)

        # Create dataset with multiple initial conditions
        dataset = Dataset.from_model(model)
        dataset.add_initial(s0=100.0)
        dataset.add_initial(s0=200.0)
        dataset.add_initial(s0=300.0)

        # Simulate the model and add augmentations
        config = SimulationConfig(t1=100, nsteps=10)
        simulated = model.simulate(dataset, config).augment(
            n_augmentations=1, append=False
        )

        # Add noise to initial conditions to simulate measurement uncertainty
        multiplicative_noise = random.gauss(0, 0.1)

        for measurement in simulated.measurements:
            for initial in measurement.initial_conditions:
                value = measurement.initial_conditions[initial]
                corrupted = value + multiplicative_noise * value
                data = measurement.data[initial]
                measurement.initial_conditions[initial] = corrupted

                # Remove first time point
                if measurement.time is not None:
                    measurement.time = measurement.time[1:]

                # Remove first data point
                if isinstance(data, list):
                    del data[0]
                    measurement.data[initial] = data
                else:
                    data = data.tolist()
                    del data[0]
                    measurement.data[initial] = jnp.array(data)

        # Create an instance with the same behavior as the original function
        pre_model = estimate_initials()

        # Configure and run MCMC
        hmc_config = cmc.MCMCConfig(num_warmup=1000, num_samples=1000)
        hmc = cmc.HMC.from_config(hmc_config)

        hmc.run(
            model,
            simulated,
            pre_model=pre_model,
            yerrs=1.0,
        )

    def test_init_symbol_surrogate_tracks_sampled_y0s(self):
        """Surrogate rate path must resolve "{state}_init" from the live y0s.

        In the surrogate path y0s is repurposed as the flattened state points at
        which rates are evaluated. The init symbols must be derived from the t=0
        state of each measurement *as passed in*, so that when y0s are sampled or
        transformed (e.g. via ``estimate_initials``) the inits reflect those new
        values rather than the static dataset values.
        """
        from catalax.mcmc.mcmc import (
            _configure_simulation_function,
            _extract_dataset_components,
        )
        from catalax.surrogate import Surrogate

        # Model whose rate depends solely on s0_init (k * s0_init), so the rate
        # of each evaluated point is determined entirely by the initial value.
        model = Model(name="InitSurrogate")
        model.add_state(s0="A", s1="B")
        model.add_constant(s0_init="initial of s0")
        model.add_assignment(symbol="rate", equation="k * s0_init")
        model.add_ode("s0", "-rate")
        model.add_ode("s1", "rate")
        model.parameters["k"].value = 0.5

        dataset = Dataset.from_model(model)
        dataset.add_initial(s0=2.0, s1=0.0)
        dataset.add_initial(s0=4.0, s1=0.0)
        config = SimulationConfig(nsteps=3, t0=0, t1=2)
        simulated = model.simulate(dataset=dataset, config=config)

        n_time = 3

        class _StubSurrogate(Surrogate):
            """Minimal surrogate; only predict_rates is touched by the branch."""

            def rates(self, t, y, constants=None):
                return jnp.zeros_like(y)

            def predict_rates(self, dataset, return_individual=False):
                n_states = len(model.get_state_order())
                n_points = len(dataset.measurements) * n_time
                return jnp.zeros((n_points, n_states))

            @property
            def has_uncertainty(self):
                return False

            def rate_uncertainty(self, dataset):
                return jnp.zeros(())

            def rate_sigma(self, dataset):
                return jnp.zeros(())

        data, times, y0s, constants = _extract_dataset_components(simulated, model)
        sim_func, _, y0s, times, constants = _configure_simulation_function(
            model=model,
            surrogate=_StubSurrogate(),
            dataset=simulated,
            data=data,
            y0s=y0s,
            times=times,
            constants=constants,
            config=config,
        )

        theta = jnp.array(
            [model.parameters[p].value for p in model.get_parameter_order()]
        )

        def s1_rates(y0s_in):
            # Returns the s1 rate for every flattened point (== k * s0_init).
            return sim_func(y0s_in, theta, constants, times)[:, 1]

        # Baseline: rate == k * s0_init, constant across each measurement's
        # points (0.5 * 2.0 = 1.0 for meas 0, 0.5 * 4.0 = 2.0 for meas 1).
        base = s1_rates(y0s)
        assert jnp.allclose(base[:n_time], 1.0), base[:n_time]
        assert jnp.allclose(base[n_time:], 2.0), base[n_time:]

        n_states = y0s.shape[-1]
        grouped = y0s.reshape(-1, n_time, n_states)

        # "Sampling" the initial conditions: bump the t=0 state of measurement 0
        # to 10.0 -> every point of that measurement must now use s0_init=10.0.
        sampled = grouped.at[0, 0, 0].set(10.0).reshape(-1, n_states)
        sampled_rates = s1_rates(sampled)
        assert jnp.allclose(sampled_rates[:n_time], 5.0), sampled_rates[:n_time]
        assert jnp.allclose(sampled_rates[n_time:], 2.0), sampled_rates[n_time:]

        # Changing a NON-initial point (t>0) must NOT change the init term,
        # confirming the initial value (t=0) is what gets reused.
        non_init = grouped.at[0, 1, 0].set(999.0).reshape(-1, n_states)
        non_init_rates = s1_rates(non_init)
        assert jnp.allclose(non_init_rates, base), non_init_rates

    def test_init_symbol_with_estimated_initials(self):
        """Full-simulation MCMC with sampled y0s on an init-using model.

        ``estimate_initials`` samples the initial conditions (y0s). Because the
        simulation path forwards y0 as the source of "{state}_init" symbols, the
        sampled initial conditions must flow into the inits automatically. This
        exercises the whole pipeline end-to-end and guards against the
        "Array 'inits' for positional arguments is missing" regression.
        """
        model = Model(name="InitEstimated")
        model.add_state(s="Substrate", p="Product")
        model.add_constant(s_init="initial of s")
        model.add_assignment(symbol="rate", equation="k * s_init")
        model.add_ode("s", "-rate")
        model.add_ode("p", "rate")
        model.parameters["k"].value = 0.5
        model.parameters["k"].prior = Uniform(low=1e-6, high=5.0)

        dataset = Dataset.from_model(model)
        dataset.add_initial(s=2.0, p=0.0)
        dataset.add_initial(s=4.0, p=0.0)
        config = SimulationConfig(nsteps=5, t0=0, t1=2)
        simulated = model.simulate(dataset=dataset, config=config)

        hmc = cmc.HMC.from_config(cmc.MCMCConfig(num_warmup=20, num_samples=20))
        results = hmc.run(
            model,
            simulated,
            pre_model=estimate_initials(),
            yerrs=1.0,
        )

        # The sampled initial conditions must appear in the posterior, proving
        # y0s were sampled while the init-using model ran without error.
        assert "estimated_y0s" in results.get_samples()
