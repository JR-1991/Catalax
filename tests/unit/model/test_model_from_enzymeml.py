import pyenzyme as pe
from sympy import sympify

import catalax as ctx


def test_model_from_enzymeml_complete_enzymeml():
    doc = pe.read_enzymeml("tests/enzymeml_docs/enzymeml_complete.json")
    dataset, model = ctx.from_enzymeml(doc, from_reactions=False)

    assert dataset is not None
    assert model is not None

    ### MODEL CHECKS
    # model participants
    assert set(model.species.keys()) == set(
        ["s_lactate", "pyruvate", "l_lactate_dehydrogenase_cytochrome"]
    )
    assert set(model.parameters.keys()) == set(["k_cat", "K_M", "k_inact"])

    # model structure
    assert len(model.odes) == 3
    assert len(model.assignments) == 0

    # equations
    for idx, (ode_id, ode) in enumerate(model.odes.items()):
        assert ode_id == doc.equations[idx].species_id
        assert ode.equation == sympify(doc.equations[idx].equation)

    ### DATASET CHECKS
    assert len(dataset.species) == 3
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
            assert array.tolist() == species_data.data
            assert meas.time.tolist() == species_data.time
