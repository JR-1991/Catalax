import random

import jax.numpy as jnp
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
