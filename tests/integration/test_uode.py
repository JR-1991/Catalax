import os
import tempfile

import jax.numpy as jnp
import optax
import pytest

import catalax as ctx
import catalax.neural as ctn


class TestUODE:
    @pytest.mark.expensive
    def test_train_uode(self):
        model, dataset = self._load_model_and_data()

        penalties = ctn.Penalties.for_universal_ode(
            l1_gate_alpha=0.01,
            l2_gate_alpha=None,
        )

        neural_ode = ctn.UniversalODE.from_model(model, width_size=4, depth=1)

        strategy = ctn.Strategy()
        strategy.add_step(
            lr=1e-2,
            length=0.20,
            steps=100,  # 500
            batch_size=2,
            penalties=penalties,
            loss=optax.log_cosh,
        )

        strategy.add_step(
            lr=1e-3,
            steps=3500,  # 7000
            batch_size=2,
            penalties=penalties,
            loss=optax.log_cosh,
        )

        ctn.train_neural_ode(
            model=neural_ode,
            dataset=dataset,
            strategy=strategy,
            print_every=10,
            weight_scale=1e-8,
            save_milestones=False,
        )

    @pytest.mark.expensive
    def test_save_load_uode(self):
        model, dataset = self._load_model_and_data()

        neural_ode = ctn.UniversalODE.from_model(model, width_size=4, depth=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.eqx")
            neural_ode.save_to_eqx(temp_dir, name="model")
            loaded_neural_ode = ctn.UniversalODE.from_eqx(path)

            loaded_neural_ode.predict_rates(dataset)

    def _load_model_and_data(self):
        model = ctx.Model.load("tests/fixtures/uode/uode_model.json")

        times = jnp.load("tests/fixtures/uode/times.npy")
        data = jnp.load("tests/fixtures/uode/data.npy")
        y0 = jnp.load("tests/fixtures/uode/y0s.npy")

        dataset = ctx.Dataset.from_jax_arrays(
            state_order=model.get_state_order(),
            data=data,
            time=times,
            y0s=y0,
        )

        return model, dataset
