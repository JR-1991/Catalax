import catalax.mcmc as cmm
import catalax.neural as ctn
import optax


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

        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-2,
            length=0.1,
            steps=200,
            batch_size=15,
            alpha=0.1,
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
