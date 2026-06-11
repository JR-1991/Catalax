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
        assert len(sim_dataset.measurements) == len(dataset.measurements), (
            "Simulated dataset should have the same number of measurements as input dataset"
        )

        # Test first measurement (e=1.0): exponential growth with rate 1.0
        expected_data_1 = jnp.array([1.0, 4.481674, 20.08544])
        actual_data_1 = sim_dataset.measurements[0].data["s1"]
        assert (
            jnp.abs(actual_data_1 - expected_data_1).sum() < TOLERANCE  # type: ignore
        ), (
            f"First measurement data mismatch. Expected: {expected_data_1}, Got: {actual_data_1}"
        )

        expected_times = [0.0, 1.5, 3.0]
        actual_times = sim_dataset.measurements[0].time.tolist()  # type: ignore
        assert actual_times == expected_times, (
            f"Time points mismatch. Expected: {expected_times}, Got: {actual_times}"
        )

        expected_initial_1 = {"s1": 1.0, "e": 1.0}
        actual_initial_1 = sim_dataset.measurements[0].initial_conditions
        assert actual_initial_1 == expected_initial_1, (
            f"First measurement initial conditions mismatch. Expected: {expected_initial_1}, Got: {actual_initial_1}"
        )

        # Test second measurement (e=2.0): exponential growth with rate 2.0
        expected_data_2 = jnp.array([1.0, 20.085447, 403.42603])
        actual_data_2 = sim_dataset.measurements[1].data["s1"]
        assert (
            jnp.abs(actual_data_2 - expected_data_2).sum() < TOLERANCE  # type: ignore
        ), (
            f"Second measurement data mismatch. Expected: {expected_data_2}, Got: {actual_data_2}"
        )

        actual_times_2 = sim_dataset.measurements[1].time.tolist()  # type: ignore
        assert actual_times_2 == expected_times, (
            f"Time points mismatch for second measurement. Expected: {expected_times}, Got: {actual_times_2}"
        )

        expected_initial_2 = {"s1": 1.0, "e": 2.0}
        actual_initial_2 = sim_dataset.measurements[1].initial_conditions
        assert actual_initial_2 == expected_initial_2, (
            f"Second measurement initial conditions mismatch. Expected: {expected_initial_2}, Got: {actual_initial_2}"
        )

    def test_simulation_with_init_symbol(self):
        """Test that "{state}_init" symbols reuse a state's initial value.

        The model uses two states:
        - ``s`` decays exponentially: ds/dt = -k * s
        - ``p`` accumulates at a rate equal to the *initial* value of ``s``:
          dp/dt = s_init

        Because the rate of ``p`` is the constant initial value ``s_init`` (and
        NOT the decaying current value of ``s``), the analytical solution for
        ``p`` is exactly linear:  p(t) = p0 + s_init * t. This makes it trivial
        to verify that the initial value was used rather than the live state.
        """
        # Build the model. `s_init` should be recognised as an init reference,
        # not turned into a free parameter.
        model = ctx.Model(name="InitReuseModel")
        model.add_state(s="Substrate", p="Product")
        model.add_ode("s", "-k * s")
        model.add_ode("p", "s_init")
        model.parameters["k"].value = 1.0

        # `s_init` must be tracked as an init on the p-equation and must NOT
        # appear among the model parameters.
        assert "s" in model.odes["p"].inits, (
            "Expected 's' to be tracked as an init reference on the p-equation"
        )
        assert "s_init" not in model.parameters, (
            "'s_init' must not be registered as a fittable parameter"
        )

        # Two measurements with different initial substrate concentrations.
        dataset = ctx.Dataset.from_model(model)
        dataset.add_initial(s=2.0, p=0.0)
        dataset.add_initial(s=5.0, p=0.0)

        # 4 time points from t=0 to t=3 -> [0, 1, 2, 3]
        config = ctx.SimulationConfig(nsteps=4, t0=0, t1=3)
        sim_dataset = model.simulate(dataset=dataset, config=config)

        times = jnp.array([0.0, 1.0, 2.0, 3.0])

        # p grows linearly with slope == initial value of s.
        expected_p_1 = 2.0 * times  # s_init = 2.0
        actual_p_1 = sim_dataset.measurements[0].data["p"]
        assert jnp.abs(actual_p_1 - expected_p_1).sum() < TOLERANCE, (  # type: ignore
            f"p should grow linearly with slope s_init=2.0. "
            f"Expected: {expected_p_1}, Got: {actual_p_1}"
        )

        expected_p_2 = 5.0 * times  # s_init = 5.0
        actual_p_2 = sim_dataset.measurements[1].data["p"]
        assert jnp.abs(actual_p_2 - expected_p_2).sum() < TOLERANCE, (  # type: ignore
            f"p should grow linearly with slope s_init=5.0. "
            f"Expected: {expected_p_2}, Got: {actual_p_2}"
        )

        # Sanity check: s itself still decays exponentially (s = s0 * exp(-t)),
        # confirming the init symbol did not freeze the live state.
        expected_s_1 = 2.0 * jnp.exp(-times)
        actual_s_1 = sim_dataset.measurements[0].data["s"]
        assert jnp.abs(actual_s_1 - expected_s_1).sum() < TOLERANCE, (  # type: ignore
            f"s should decay exponentially. Expected: {expected_s_1}, Got: {actual_s_1}"
        )
