import random

import jax.numpy as jnp
import numpy as np
import optax

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
        config = cmm.MCMCConfig(num_warmup=50, num_samples=100, num_chains=2, verbose=0)
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
