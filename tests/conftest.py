import jax.numpy as jnp
import pytest

import catalax as ctx
import catalax.mcmc as cmm


def menten_model():
    model = ctx.Model(name="Michaelis-Menten")
    model.add_state(s1="Substrate")
    model.add_constant(e="Enzyme")
    model.add_ode("s1", "kcat * e * s1 / (k_m + s1)")

    model.parameters["kcat"].value = 10.0
    model.parameters["kcat"].prior = cmm.priors.Uniform(low=9.0, high=11.0)
    model.parameters["k_m"].value = 100.0
    model.parameters["k_m"].prior = cmm.priors.Uniform(low=90.0, high=110.0)

    return model


@pytest.fixture
def model():
    """Sets up a simple model with one species and one reaction."""
    return menten_model()


@pytest.fixture
def generate_data():
    model = menten_model()

    # Create dataset with two different enzyme concentrations
    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(s1=1.0, e=0.001)
    dataset.add_initial(s1=10.0, e=0.001)

    # Configure simulation: 3 time points from t=0 to t=3
    config = ctx.SimulationConfig(nsteps=3, t0=0, t1=3)

    # Run simulation
    sim_dataset = model.simulate(dataset=dataset, config=config)

    return model, sim_dataset


@pytest.fixture
def time_states_inits():
    """Loads the pre-computed times and states."""

    return (
        jnp.load("./tests/fixtures/times.npy"),
        jnp.load("./tests/fixtures/data.npy"),
        [{"s1": 100.0}, {"s1": 200.0}],
    )


@pytest.fixture
def dataset():
    """Sets up a datasets with two measurements."""

    times = jnp.load("./tests/fixtures/times.npy")
    data = jnp.load("./tests/fixtures/data.npy")

    dataset = ctx.Dataset(states=["s1"], id="test")

    dataset.add_measurement(
        ctx.Measurement(
            id="meas1",
            time=times,
            initial_conditions={"s1": 100.0},
            data={"s1": data[0, :, 0]},
        )
    )

    dataset.add_measurement(
        ctx.Measurement(
            id="meas2",
            time=times,
            initial_conditions={"s1": 200.0},
            data={"s1": data[1, :, 0]},
        )
    )
