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
