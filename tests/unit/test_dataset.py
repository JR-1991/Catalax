import catalax as ctx


class TestDataset:
    def test_dataset_creation(self, time_states_inits):
        """Tests the creation of a datasets"""

        # Arrange
        times, data, initial_conditions = time_states_inits

        # Act
        dataset = ctx.Dataset(species=["s1"], id="test")
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=times,
                initial_conditions=initial_conditions[0],
                data={"s1": data[0, :, 0]},
            )
        )

        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=times,
                initial_conditions=initial_conditions[1],
                data={"s1": data[1, :, 0]},
            )
        )

        # Assert
        assert len(dataset.measurements) == 2, "Two measurements should be present"
        assert dataset.species == ["s1"], "Species should be ['s1']"
        assert dataset.id == "test", "Dataset ID should be 'test'"

        assert dataset.measurements[0].id == "meas1", "Measurement ID should be 'meas1'"
        assert dataset.measurements[1].id == "meas2", "Measurement ID should be 'meas2'"

        assert dataset.measurements[0].time.shape == (100,), "Time should be (100,)"  # type: ignore

        assert dataset.measurements[0].data["s1"].shape == (100,), (  # type: ignore
            "Data shape should be (100,)"
        )
        assert dataset.measurements[1].data["s1"].shape == (100,), (  # type: ignore
            "Data shape should be (100,)"
        )  # type: ignore

        assert dataset.measurements[0].initial_conditions == {"s1": 100.0}, (
            "Initial conditions should be {'s1': 100.0}"
        )
        assert dataset.measurements[1].initial_conditions == {"s1": 200.0}, (
            "Initial conditions should be {'s1': 200.0}"
        )
