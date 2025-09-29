"""Tests for Dataset.from_enzymeml() method and Model.from_enzymeml() with reactions."""

import pyenzyme as pe
import pytest

import catalax as ctx

ENZYMEML_DOC_PATH = "examples/datasets/enzymeml_inactivation.json"


class TestDatasetFromEnzymeML:
    """Test suite for the Dataset.from_enzymeml() method."""

    @pytest.fixture
    def dataset_from_enzymeml_basics(self):
        """Create a simple EnzymeML document for testing.

        This fixture loads an EnzymeML document from a JSON file and creates
        a Catalax Dataset from it. It then validates that the basic properties
        (name, states, measurements) are correctly transferred from the EnzymeML
        document to the Dataset object.
        """
        doc = pe.read_enzymeml(ENZYMEML_DOC_PATH)
        ds = ctx.Dataset.from_enzymeml(doc)

        protein_ids = [p.id for p in doc.proteins]
        small_molecules = [s.id for s in doc.small_molecules]

        all_species = protein_ids + small_molecules

        assert doc.name == ds.name
        assert set(all_species) == set(ds.states)
        assert len(doc.measurements) == len(ds.measurements)

    def test_from_enzymeml_missing_initial_conditions(self):
        """Test behavior when initial conditions are missing.

        This test verifies that the Dataset.from_enzymeml() method properly
        handles cases where measurement data lacks initial conditions. The
        method should raise an exception when trying to create a dataset
        from measurements that don't have initial concentration values.
        """
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
        with pytest.raises(ValueError):  # Could be ValidationError or similar
            ctx.Dataset.from_enzymeml(doc)

    def test_model_from_enzymeml_missing_kinetic_law(self):
        """Test Model.from_enzymeml() behavior when kinetic law is missing.

        This test ensures that the Model.from_enzymeml() method properly
        validates that reactions have kinetic laws defined. When a reaction
        lacks a kinetic law, the method should raise an exception since the
        kinetic law is required to generate the ODE system.
        """
        doc = pe.read_enzymeml(ENZYMEML_DOC_PATH)

        # delete kinetic law from reaction
        doc.reactions[0].kinetic_law = None

        # Act & Assert - should fail due to validation error (missing kinetic law)
        with pytest.raises(Exception):  # Could be ValidationError or similar
            ctx.Model.from_enzymeml(doc)

    def test_dataset_from_enzymeml(self):
        """Test successful creation of Dataset from EnzymeML document.

        This test verifies that a Dataset can be successfully created from
        a valid EnzymeML document. It checks that:
        - The number of measurements matches between the document and dataset
        - The number of states includes all small molecules and proteins
        - All measurements are properly transferred
        """
        doc = pe.read_enzymeml(ENZYMEML_DOC_PATH)
        ds = ctx.Dataset.from_enzymeml(doc)

        assert len(ds.measurements) == len(doc.measurements)
        assert len(ds.states) == len(doc.small_molecules) + len(doc.proteins)
        assert len(ds.measurements) == len(doc.measurements)
