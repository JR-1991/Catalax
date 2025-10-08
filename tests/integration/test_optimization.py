import catalax as ctx


class TestOptimization:
    def test_optimize_runs(self):
        # Create a Michaelis-Menten enzyme kinetics model
        model = ctx.Model(name="Enzyme Kinetics")

        # Define states
        model.add_states(S="Substrate", P="Product")

        # Add reaction kinetics via schemes
        model.add_reaction(
            "S -> P",
            symbol="r1",
            equation="v_max * S / (K_m + S)",
        )

        # Set parameter values
        model.parameters["v_max"].value = 0.2
        model.parameters["v_max"].initial_value = 0.1
        model.parameters["K_m"].value = 0.1
        model.parameters["K_m"].initial_value = 0.05

        # Create dataset and simulate
        dataset = ctx.Dataset.from_model(model)
        dataset.add_initial(S=1.0, P=0.0)
        config = ctx.SimulationConfig(t1=10, nsteps=10)
        dataset = model.simulate(dataset=dataset, config=config)

        # Run optimization and check that it completes and returns expected types
        result, new_model = ctx.optimize(model, dataset)
        assert hasattr(result, "params")
        assert isinstance(new_model, type(model))
