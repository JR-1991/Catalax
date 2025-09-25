"""Tests for Dataset.from_enzymeml() method and Model.from_enzymeml() with reactions."""

import pyenzyme as pe
import pytest

import catalax as ctx

ENZYMEML_DOC_PATH = "examples/datasets/enzymeml_inactivation.json"


class TestDatasetFromEnzymeML:
    """Test suite for the Dataset.from_enzymeml() method."""

    @pytest.fixture
    def dataset_from_enzymeml_basics(self):
        """Create a simple EnzymeML document for testing."""
        doc = pe.read_enzymeml(ENZYMEML_DOC_PATH)
        ds = ctx.Dataset.from_enzymeml(doc)

        protein_ids = [p.id for p in doc.proteins]
        small_molecules = [s.id for s in doc.small_molecules]

        all_species = protein_ids + small_molecules

        assert doc.name == ds.name
        assert set(all_species) == set(ds.states)
        assert len(doc.measurements) == len(ds.measurements)

    def test_from_enzymeml_missing_initial_conditions(self):
        """Test behavior when initial conditions are missing."""
        # Arrange
        doc = pe.EnzymeMLDocument(name="test_missing_initial")
        s1 = doc.add_to_small_molecules(id="s1", name="substrate1")

        meas1 = doc.add_to_measurements(id="meas1", name="measurement1")
        meas1.add_to_species_data(
            species_id=s1.id,
            name="s1_data",
            initial=None,  # Missing initial condition
            data=[10.0, 8.5, 7.2],
            time=[0.0, 1.0, 2.0],
        )

        # Act & Assert - should fail due to validation error (missing initial conditions)
        with pytest.raises(Exception):  # Could be ValidationError or similar
            ctx.Dataset.from_enzymeml(doc)

    def test_model_from_enzymeml_missing_kinetic_law(self):
        doc = pe.read_enzymeml(ENZYMEML_DOC_PATH)

        # delete kinetic law from reaction
        doc.reactions[0].kinetic_law = None

        # Act & Assert - should fail due to validation error (missing kinetic law)
        with pytest.raises(Exception):  # Could be ValidationError or similar
            ctx.Model.from_enzymeml(doc)
