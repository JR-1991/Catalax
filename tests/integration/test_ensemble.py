import jax.numpy as jnp

import catalax.neural as ctn


class TestNeuralODEEnsemble:
    def test_ensemble_predict(self, generate_data):
        model, dataset = generate_data
        dataset = dataset.augment(n_augmentations=2, sigma=1e-5)

        models = [
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=0),
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=1),
        ]
        ensemble = ctn.NeuralODEEnsemble(models)

        pred = ensemble.predict(dataset)

        assert len(pred.measurements) == len(dataset.measurements)
        assert ensemble.has_uncertainty
        assert ensemble.has_hdi()

    def test_ensemble_predict_with_hdi_and_individual(self, generate_data):
        model, dataset = generate_data

        models = [
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=2),
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=3),
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=4),
        ]
        ensemble = ctn.NeuralODEEnsemble(models)

        lower = ensemble.predict(dataset, use_times=True, hdi="lower")
        upper = ensemble.predict(dataset, use_times=True, hdi="upper")
        all_predictions, times, y0s = ensemble.predict(
            dataset,
            use_times=True,
            return_individual=True,
        )

        assert len(lower.measurements) == len(dataset.measurements)
        assert len(upper.measurements) == len(dataset.measurements)
        assert all_predictions.shape[0] == len(models)
        assert all_predictions.shape[1] == len(dataset.measurements)
        assert times.shape[0] == len(dataset.measurements)
        assert y0s.shape[0] == len(dataset.measurements)

    def test_ensemble_rates_and_serialization(self, generate_data, tmp_path):
        model, dataset = generate_data
        models = [
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=5),
            ctn.NeuralODE.from_model(model, width_size=3, depth=1, seed=6),
        ]
        ensemble = ctn.NeuralODEEnsemble(models)

        rates = ensemble.predict_rates(dataset)
        sigma = ensemble.rate_uncertainty(dataset)
        rate_mean = ensemble.rates(
            t=jnp.array([0.0]),
            y=jnp.array([[1.0]]),
        )

        ensemble.save_to_eqx(path=tmp_path, name="ensemble_test")
        loaded = ctn.NeuralODEEnsemble.from_eqx(tmp_path / "ensemble_test.zip")
        loaded_pred = loaded.predict(dataset, use_times=True)

        assert rates.shape[1] == len(ensemble.state_order)
        assert sigma.shape[1] == len(ensemble.state_order)
        assert rate_mean.shape == (1, len(ensemble.state_order))
        assert len(loaded.models) == len(models)
        assert len(loaded_pred.measurements) == len(dataset.measurements)
