import catalax as ctx
import catalax.neural as ctn
import rich


class TestNeuralODE:
    def test_neuralode(self, generate_data):
        model, dataset = generate_data
        dataset = dataset.augment(n_augmentations=4, sigma=1e-5)

        neural_ode = ctn.NeuralODE.from_model(
            model,
            width_size=3,
            depth=1,
        )

        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-3,
            steps=200,
            batch_size=5,
            alpha=0.01,
        )
        strategy.add_step(
            lr=1e-4,
            steps=200,
            batch_size=5,
            alpha=0.01,
        )
        neural_ode = ctn.train_neural_ode(
            model=neural_ode,
            dataset=dataset,
            strategy=strategy,
            print_every=10,
            weight_scale=1e-7,
        )

        pred = neural_ode.predict(dataset)

        assert len(pred.measurements) == len(dataset.measurements)

    def test_neuralode_with_rbf(self, generate_data):
        model, dataset = generate_data
        dataset = dataset.augment(n_augmentations=4, sigma=1e-5)

        neural_ode = ctn.NeuralODE.from_model(
            model,
            width_size=3,
            depth=1,
            activation=ctn.RBFLayer(0.2),  # type: ignore
        )

        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-3,
            steps=200,
            length=0.1,
            batch_size=5,
            alpha=0.01,
        )
        strategy.add_step(
            lr=1e-4,
            steps=200,
            batch_size=5,
            alpha=0.01,
        )

        neural_ode = ctn.train_neural_ode(
            model=neural_ode,
            dataset=dataset,
            strategy=strategy,
            print_every=10,
            weight_scale=1e-7,
        )

        pred = neural_ode.predict(dataset)

        assert len(pred.measurements) == len(dataset.measurements)
