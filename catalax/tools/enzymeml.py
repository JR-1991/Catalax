import pyenzyme as pe

from catalax.dataset import Dataset
from catalax.model import Model


def dataset_and_model_from_enzymeml(
    enzmldoc: pe.EnzymeMLDocument,
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
    model = Model.from_enzymeml(enzmldoc)
    dataset = (
        Dataset.from_enzymeml(enzmldoc)
        if enzmldoc.measurements
        else Dataset.from_model(model)
    )

    return dataset, model
