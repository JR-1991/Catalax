import numpy as np
import pytest

import catalax as ctx


class TestDataset:
    def test_dataset_creation(self, time_states_inits):
        """Tests the creation of a datasets"""

        # Arrange
        times, data, initial_conditions = time_states_inits

        # Act
        dataset = ctx.Dataset(states=["s1"], id="test")
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
        assert dataset.states == ["s1"], "Species should be ['s1']"
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


class TestDatasetPad:
    """Test suite for the Dataset.pad() method."""

    def test_pad_no_changes_needed(self):
        """Test padding when all measurements already have consistent data shapes."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        # Both measurements have same-length data for all species
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1, 2],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1, 1.2], "s2": [2.0, 2.1, 2.2]},
            )
        )
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=[0, 1, 2],
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={"s1": [3.0, 3.1, 3.2], "s2": [4.0, 4.1, 4.2]},
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert
        assert len(padded_dataset.measurements) == 2

        # Check that data hasn't changed
        for i, meas in enumerate(padded_dataset.measurements):
            assert len(meas.data["s1"]) == 3
            assert len(meas.data["s2"]) == 3
            # Verify original data is preserved
            np.testing.assert_allclose(
                meas.data["s1"], dataset.measurements[i].data["s1"], rtol=1e-6
            )
            np.testing.assert_allclose(
                meas.data["s2"], dataset.measurements[i].data["s2"], rtol=1e-6
            )

    def test_pad_missing_species_data(self):
        """Test padding when a measurement has initial conditions but no data array for a species."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        # First measurement has data for both species
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1, 2, 3],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1, 1.2, 1.3], "s2": [2.0, 2.1, 2.2, 2.3]},
            )
        )

        # Second measurement missing data for s2 (only initial condition)
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=[0, 1],
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={"s1": [3.0, 3.1]},  # Missing s2 data
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert
        meas1 = padded_dataset.measurements[0]
        meas2 = padded_dataset.measurements[1]

        # First measurement should be unchanged
        assert len(meas1.data["s1"]) == 4
        assert len(meas1.data["s2"]) == 4
        assert not any(np.isnan(meas1.data["s1"]))
        assert not any(np.isnan(meas1.data["s2"]))

        # Second measurement should have s2 filled with NaNs
        assert len(meas2.data["s1"]) == 4  # Padded to max length
        assert len(meas2.data["s2"]) == 4  # Added as NaN array

        # Original s1 data should be preserved, then padded with NaNs
        np.testing.assert_allclose(meas2.data["s1"][:2], [3.0, 3.1], rtol=1e-6)
        assert all(np.isnan(meas2.data["s1"][2:]))

        # s2 should be all NaNs since it was missing
        assert all(np.isnan(meas2.data["s2"]))

    def test_pad_different_data_lengths(self):
        """Test padding when data arrays have different lengths across measurements."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        # Different length data arrays
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1], "s2": [2.0, 2.1]},  # Length 2
            )
        )
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=[0, 1, 2, 3, 4],
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={
                    "s1": [3.0, 3.1, 3.2, 3.3, 3.4],
                    "s2": [4.0, 4.1, 4.2, 4.3, 4.4],
                },  # Length 5
            )
        )
        dataset.add_measurement(
            ctx.Measurement(
                id="meas3",
                time=[0, 1, 2],
                initial_conditions={"s1": 5.0, "s2": 6.0},
                data={"s1": [5.0, 5.1, 5.2], "s2": [6.0, 6.1, 6.2]},  # Length 3
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert
        max_length = 5  # From meas2

        for i, meas in enumerate(padded_dataset.measurements):
            assert len(meas.data["s1"]) == max_length
            assert len(meas.data["s2"]) == max_length

            # Check that original data is preserved
            original_meas = dataset.measurements[i]
            original_len_s1 = len(original_meas.data["s1"])
            original_len_s2 = len(original_meas.data["s2"])

            # Original data should be at the beginning
            np.testing.assert_allclose(
                meas.data["s1"][:original_len_s1], original_meas.data["s1"], rtol=1e-6
            )
            np.testing.assert_allclose(
                meas.data["s2"][:original_len_s2], original_meas.data["s2"], rtol=1e-6
            )

            # Padding should be NaN
            if original_len_s1 < max_length:
                assert all(np.isnan(meas.data["s1"][original_len_s1:]))
            if original_len_s2 < max_length:
                assert all(np.isnan(meas.data["s2"][original_len_s2:]))

    def test_pad_different_data_lengths_across_measurements(self):
        """Test padding when different measurements have different data lengths for the same species."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        # First measurement with shorter data
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1], "s2": [2.0, 2.1]},  # Length 2
            )
        )

        # Second measurement with longer data
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=[0, 1, 2, 3, 4],
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={
                    "s1": [3.0, 3.1, 3.2, 3.3, 3.4],
                    "s2": [4.0, 4.1, 4.2, 4.3, 4.4],
                },  # Length 5
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert
        meas1 = padded_dataset.measurements[0]
        meas2 = padded_dataset.measurements[1]
        max_length = 5

        # First measurement should be padded
        assert len(meas1.data["s1"]) == max_length
        assert len(meas1.data["s2"]) == max_length
        np.testing.assert_allclose(meas1.data["s1"][:2], [1.0, 1.1], rtol=1e-6)
        np.testing.assert_allclose(meas1.data["s2"][:2], [2.0, 2.1], rtol=1e-6)
        assert all(np.isnan(meas1.data["s1"][2:]))
        assert all(np.isnan(meas1.data["s2"][2:]))

        # Second measurement should be unchanged
        assert len(meas2.data["s1"]) == max_length
        assert len(meas2.data["s2"]) == max_length
        np.testing.assert_allclose(
            meas2.data["s1"], [3.0, 3.1, 3.2, 3.3, 3.4], rtol=1e-6
        )
        np.testing.assert_allclose(
            meas2.data["s2"], [4.0, 4.1, 4.2, 4.3, 4.4], rtol=1e-6
        )

    def test_pad_empty_data_measurements(self):
        """Test padding when measurements have only initial conditions (no data)."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        # One measurement with data
        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1, 2],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1, 1.2], "s2": [2.0, 2.1, 2.2]},
            )
        )

        # One measurement with only initial conditions
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=None,
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={},  # Empty data
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert
        meas1 = padded_dataset.measurements[0]
        meas2 = padded_dataset.measurements[1]

        # First measurement unchanged
        assert len(meas1.data["s1"]) == 3
        assert len(meas1.data["s2"]) == 3

        # Second measurement should have NaN arrays for all species
        assert len(meas2.data["s1"]) == 3
        assert len(meas2.data["s2"]) == 3
        assert all(np.isnan(meas2.data["s1"]))
        assert all(np.isnan(meas2.data["s2"]))

    def test_pad_preserves_original_dataset(self):
        """Test that padding creates a new dataset and doesn't modify the original."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2"], id="test_pad")

        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1],
                initial_conditions={"s1": 1.0, "s2": 2.0},
                data={"s1": [1.0, 1.1], "s2": [2.0, 2.1]},
            )
        )
        dataset.add_measurement(
            ctx.Measurement(
                id="meas2",
                time=[0, 1, 2, 3],
                initial_conditions={"s1": 3.0, "s2": 4.0},
                data={"s1": [3.0, 3.1, 3.2, 3.3]},  # Missing s2
            )
        )

        # Store original state
        original_meas1_s1 = dataset.measurements[0].data["s1"].copy()
        original_meas1_s2 = dataset.measurements[0].data["s2"].copy()
        original_meas2_s1 = dataset.measurements[1].data["s1"].copy()
        original_meas2_has_s2 = "s2" in dataset.measurements[1].data

        # Act
        padded_dataset = dataset.pad()

        # Assert original dataset is unchanged
        np.testing.assert_allclose(
            dataset.measurements[0].data["s1"], original_meas1_s1, rtol=1e-6
        )
        np.testing.assert_allclose(
            dataset.measurements[0].data["s2"], original_meas1_s2, rtol=1e-6
        )
        np.testing.assert_allclose(
            dataset.measurements[1].data["s1"], original_meas2_s1, rtol=1e-6
        )
        assert ("s2" in dataset.measurements[1].data) == original_meas2_has_s2

        # Assert padded dataset is different
        assert id(padded_dataset) != id(dataset)
        assert len(padded_dataset.measurements[0].data["s1"]) == 4  # Padded
        assert len(padded_dataset.measurements[1].data["s2"]) == 4  # Added

    def test_pad_all_species_equal_length(self):
        """Test padding when all species already have equal length data."""
        # Arrange
        dataset = ctx.Dataset(states=["s1", "s2", "s3"], id="test_pad")

        dataset.add_measurement(
            ctx.Measurement(
                id="meas1",
                time=[0, 1, 2],
                initial_conditions={"s1": 1.0, "s2": 2.0, "s3": 3.0},
                data={
                    "s1": [1.0, 1.1, 1.2],
                    "s2": [2.0, 2.1, 2.2],
                    "s3": [3.0, 3.1, 3.2],
                },
            )
        )

        # Act
        padded_dataset = dataset.pad()

        # Assert - no padding should be needed
        meas = padded_dataset.measurements[0]
        assert "s1" in meas.data
        assert "s2" in meas.data
        assert "s3" in meas.data

        assert len(meas.data["s1"]) == 3
        assert len(meas.data["s2"]) == 3
        assert len(meas.data["s3"]) == 3

        # Verify data is unchanged
        np.testing.assert_allclose(meas.data["s1"], [1.0, 1.1, 1.2], rtol=1e-6)
        np.testing.assert_allclose(meas.data["s2"], [2.0, 2.1, 2.2], rtol=1e-6)
        np.testing.assert_allclose(meas.data["s3"], [3.0, 3.1, 3.2], rtol=1e-6)
