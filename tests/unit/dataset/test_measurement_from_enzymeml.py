"""Tests for Measurement._pad_species_arrays() and from_enzymeml()."""
import jax.numpy as jnp
import pyenzyme as pe
import pytest

from catalax.dataset.measurement import Measurement


def _make_pe_measurement(species: list[dict]) -> pe.Measurement:
    """Build a pe.Measurement with multiple species_data entries.

    Each entry in ``species`` is a dict with keys:
        species_id, initial, data, time (all required).
    """
    doc = pe.EnzymeMLDocument(name="test")
    meas = doc.add_to_measurements(id="m1", name="m1")
    for sp in species:
        sm = doc.add_to_small_molecules(id=sp["species_id"], name=sp["species_id"])
        meas.add_to_species_data(
            species_id=sm.id,
            name=sp["species_id"],
            initial=sp["initial"],
            data=sp["data"],
            time=sp["time"],
        )
    return meas


class TestPadSpeciesArrays:
    def test_homogeneous_no_padding_needed(self):
        """When all species already have the same length, arrays are unchanged."""
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6], "time": [0, 1, 2]},
            {"species_id": "s2", "initial": 0.0,  "data": [0, 2, 4],  "time": [0, 1, 2]},
        ])
        time, data = Measurement._pad_species_arrays(meas, global_max_len=3)
        assert len(time) == 3
        assert len(data["s1"]) == len(data["s2"]) == 3
        assert not jnp.any(jnp.isnan(data["s1"]))
        assert not jnp.any(jnp.isnan(data["s2"]))

    def test_subset_species_aligned_with_nan(self):
        """Shorter species (subset time) get NaN at positions they were not measured."""
        # s1: [0,1,2,3,4] — canonical (longest)
        # s2: [0,2,4]     — subset, measured at every other point
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6, 4, 2], "time": [0, 1, 2, 3, 4]},
            {"species_id": "s2", "initial": 0.0,  "data": [0, 4, 8],        "time": [0, 2, 4]},
        ])
        time, data = Measurement._pad_species_arrays(meas, global_max_len=5)
        assert list(time) == [0, 1, 2, 3, 4]
        assert float(data["s2"][0]) == 0.0
        assert jnp.isnan(data["s2"][1])      # t=1 not measured
        assert float(data["s2"][2]) == 4.0
        assert jnp.isnan(data["s2"][3])      # t=3 not measured
        assert float(data["s2"][4]) == 8.0

    def test_cross_measurement_padding_to_global_max(self):
        """When global_max_len > local canonical length, time and data are extended."""
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6], "time": [0, 1, 2]},
        ])
        time, data = Measurement._pad_species_arrays(meas, global_max_len=5)
        assert len(time) == 5
        assert float(time[3]) == 3.0      # monotonic continuation
        assert float(time[4]) == 4.0
        assert jnp.isnan(data["s1"][3])   # data padded with NaN
        assert jnp.isnan(data["s1"][4])

    def test_raises_when_species_time_not_subset(self):
        """Raises ValueError when a species has time points outside the canonical axis."""
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6, 4],  "time": [0, 1, 2, 3]},
            {"species_id": "s2", "initial": 0.0,  "data": [0, 2, 4],      "time": [0, 1.5, 3]},
            # t=1.5 is NOT in s1's time array -> should raise
        ])
        with pytest.raises(ValueError, match="not found in the canonical time"):
            Measurement._pad_species_arrays(meas, global_max_len=4)


class TestMeasurementFromEnzymeML:
    def test_homogeneous_roundtrip(self):
        """Homogeneous species (same time arrays) round-trip without NaN."""
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6], "time": [0, 1, 2]},
            {"species_id": "s2", "initial": 0.0,  "data": [0, 2, 4],  "time": [0, 1, 2]},
        ])
        m = Measurement.from_enzymeml(meas, global_max_len=3)
        assert list(m.time) == [0, 1, 2]
        assert not jnp.any(jnp.isnan(m.data["s1"]))
        assert not jnp.any(jnp.isnan(m.data["s2"]))

    def test_inhomogeneous_no_validation_error(self):
        """The original bug: inhomogeneous lengths must not raise ValidationError."""
        meas = _make_pe_measurement([
            {"species_id": "s1", "initial": 10.0, "data": [10, 8, 6, 4, 2], "time": [0, 1, 2, 3, 4]},
            {"species_id": "s2", "initial": 0.0,  "data": [0, 4, 8],        "time": [0, 2, 4]},
        ])
        m = Measurement.from_enzymeml(meas, global_max_len=5)  # must not raise
        assert len(m.time) == 5
        assert len(m.data["s1"]) == len(m.data["s2"]) == 5
        assert jnp.isnan(m.data["s2"][1])
        assert jnp.isnan(m.data["s2"][3])
