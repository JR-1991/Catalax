import pyenzyme as pe

from catalax.dataset import Dataset
from catalax.model import Model


def dataset_and_model_from_enzymeml(
    enzmldoc: pe.EnzymeMLDocument,
    from_reactions: bool = False,
) -> tuple[Dataset, Model]:
    """
    Convert an EnzymeML document to a Catalax Dataset and/or Model.

    Args:
        enzmldoc: The EnzymeML document to convert.
        from_reactions: Whether to convert the reactions to ODEs. If False, the
            equations from the `equations` section are used.

    Returns:
        tuple[Dataset, Model]: The dataset and model.

    Raises:
        ValueError: If required elements are missing or if species differ
            between dataset and model.
    """
    dataset = Dataset.from_enzymeml(enzmldoc) if enzmldoc.measurements else None
    model = Model.from_enzymeml(enzmldoc, from_reactions=from_reactions)

    # if dataset and model, remove species from dataset that are not in model
    if dataset and model:
        all_model_species = set(model.species.keys()) | set(model.constants.keys())

        dataset.species = [sp for sp in dataset.species if sp in all_model_species]

        for meas in dataset.measurements:
            meas_species = set(meas.initial_conditions.keys())
            to_remove = meas_species - all_model_species
            for sp in to_remove:
                if sp in meas.initial_conditions:
                    del meas.initial_conditions[sp]
                if sp in meas.data:
                    del meas.data[sp]

        return dataset, model

    elif dataset and not model:
        raise ValueError("EnzymeML Document is missing reactions or equations.")

    elif not dataset and model:
        raise ValueError("EnzymeML Document is missing measurements.")

    else:
        raise ValueError(
            "EnzymeML Document is missing measurements and reactions or equations."
        )
