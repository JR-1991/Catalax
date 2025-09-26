import pyenzyme as pe
import pyenzyme.equations as peq
from sympy import sympify

import catalax as ctx
from catalax.model.utils import LOCALS


class TestEnzymemlModel:
    def test_model_from_enzymeml_complete_enzymeml(self):
        doc = pe.read_enzymeml("examples/datasets/enzymeml_inactivation.json")
        dataset, model = ctx.from_enzymeml(doc)
        all_enzymeml_species = [
            sp.id for sp in doc.small_molecules + doc.proteins + doc.complexes
        ]

        assert dataset is not None
        assert model is not None

        ### MODEL CHECKS
        # model participants
        assert set(model.states.keys()) == set(all_enzymeml_species)
        assert set(model.parameters.keys()) == set(["k_cat", "K_M", "k_inact"])

        # model structure
        assert len(model.odes) == 3
        assert len(model.assignments) == 0

        # equations
        for idx, (ode_id, ode) in enumerate(model.odes.items()):
            assert ode_id == doc.equations[idx].species_id
            assert ode.equation == sympify(doc.equations[idx].equation, locals=LOCALS)

        ### DATASET CHECKS
        assert len(dataset.states) == len(all_enzymeml_species)
        assert len(dataset.measurements) == 3

        for idx, meas in enumerate(dataset.measurements):
            # initial conditions
            for species_id, value in meas.initial_conditions.items():
                assert (
                    value
                    == doc.measurements[idx]
                    .filter_species_data(species_id=species_id)[0]
                    .initial
                )

            # data
            for species_id, array in meas.data.items():
                species_data = doc.measurements[idx].filter_species_data(
                    species_id=species_id
                )[0]
                assert array.tolist() == species_data.data  # type: ignore
                assert meas.time.tolist() == species_data.time  # type: ignore

    def test_model_from_enzymeml_with_reactions(self):
        enzmldoc = pe.EnzymeMLDocument(name="test")
        enzmldoc.add_to_small_molecules(id="s1", name="s1")
        enzmldoc.add_to_small_molecules(id="s2", name="s2")
        enzmldoc.add_to_small_molecules(id="s3", name="s3")

        kinetic_law = peq.build_equation(
            equation="k*s1*s2",
            equation_type=pe.EquationType.RATE_LAW,
            enzmldoc=enzmldoc,
        )
        reaction = peq.build_reaction(
            name="test",
            scheme="s1 + s2 -> s3",
            kinetic_law=kinetic_law,
        )

        reaction.kinetic_law = kinetic_law

        enzmldoc.reactions.append(reaction)

        dataset, model = ctx.from_enzymeml(enzmldoc)

        assert dataset is not None
        assert model is not None

        assert len(dataset.states) == 3
        assert len(dataset.measurements) == 0
        assert len(model.reactions) == 1
        assert len(model.odes) == 0

    def test_model_from_enzymeml_no_model(self):
        enzmldoc = pe.EnzymeMLDocument(name="test")
        enzmldoc.add_to_small_molecules(id="s1", name="s1")
        enzmldoc.add_to_complexes(id="s2", name="s2")
        enzmldoc.add_to_proteins(id="s3", name="s3")

        dataset, model = ctx.from_enzymeml(enzmldoc)

        assert dataset is not None
        assert model is not None
        assert len(model.reactions) == 0
        assert len(model.odes) == 0
        assert len(model.states) == 3
        assert len(dataset.states) == 3
        assert len(dataset.measurements) == 0
