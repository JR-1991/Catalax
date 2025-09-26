import jax.numpy as jnp

import catalax as ctx

TOLERANCE = 1e-3


class TestSimulation:
    def test_simulation(self):
        """Test that model simulation produces expected results.

        This test verifies that:
        1. A simple exponential growth model (ds1/dt = e * s1 * q) simulates correctly
        2. Different initial conditions produce different but predictable results
        3. The simulation preserves initial conditions and time points
        """
        # Define a simple exponential growth model: ds1/dt = e * s1 * q
        model = ctx.Model(name="ExponentialGrowthModel")
        model.add_state(s1="Substrate")
        model.add_constant(e="Enzyme")
        model.add_ode("s1", "e * s1 * q")
        model.parameters["q"].value = 1.0

        # Create dataset with two different enzyme concentrations
        dataset = ctx.Dataset.from_model(model)
        dataset.add_initial(s1=1.0, e=1.0)  # Low enzyme concentration
        dataset.add_initial(s1=1.0, e=2.0)  # High enzyme concentration

        # Configure simulation: 3 time points from t=0 to t=3
        config = ctx.SimulationConfig(nsteps=3, t0=0, t1=3)

        # Run simulation
        sim_dataset = model.simulate(dataset=dataset, config=config)

        # Verify basic structure
        assert (
            len(sim_dataset.measurements) == len(dataset.measurements)
        ), "Simulated dataset should have the same number of measurements as input dataset"

        # Test first measurement (e=1.0): exponential growth with rate 1.0
        expected_data_1 = jnp.array([1.0, 4.481674, 20.08544])
        actual_data_1 = sim_dataset.measurements[0].data["s1"]
        assert (
            jnp.abs(actual_data_1 - expected_data_1).sum() < TOLERANCE
        ), f"First measurement data mismatch. Expected: {expected_data_1}, Got: {actual_data_1}"

        expected_times = [0.0, 1.5, 3.0]
        actual_times = sim_dataset.measurements[0].time.tolist()  # type: ignore
        assert (
            actual_times == expected_times
        ), f"Time points mismatch. Expected: {expected_times}, Got: {actual_times}"

        expected_initial_1 = {"s1": 1.0, "e": 1.0}
        actual_initial_1 = sim_dataset.measurements[0].initial_conditions
        assert (
            actual_initial_1 == expected_initial_1
        ), f"First measurement initial conditions mismatch. Expected: {expected_initial_1}, Got: {actual_initial_1}"

        # Test second measurement (e=2.0): exponential growth with rate 2.0
        expected_data_2 = jnp.array([1.0, 20.085447, 403.42603])
        actual_data_2 = sim_dataset.measurements[1].data["s1"]
        assert (
            jnp.abs(actual_data_2 - expected_data_2).sum() < TOLERANCE
        ), f"Second measurement data mismatch. Expected: {expected_data_2}, Got: {actual_data_2}"

        actual_times_2 = sim_dataset.measurements[1].time.tolist()  # type: ignore
        assert (
            actual_times_2 == expected_times
        ), f"Time points mismatch for second measurement. Expected: {expected_times}, Got: {actual_times_2}"

        expected_initial_2 = {"s1": 1.0, "e": 2.0}
        actual_initial_2 = sim_dataset.measurements[1].initial_conditions
        assert (
            actual_initial_2 == expected_initial_2
        ), f"Second measurement initial conditions mismatch. Expected: {expected_initial_2}, Got: {actual_initial_2}"
