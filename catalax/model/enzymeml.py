from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type

import pyenzyme as pe

from catalax.model.ode import ODE

if TYPE_CHECKING:
    from catalax.model import Model


def from_enzymeml(
    cls: Type[Model],
    enzmldoc: pe.EnzymeMLDocument,
    name: str | None = None,
) -> Model:
    """
    Convert a PyEnzyme EnzymeMLDocument to a Catalax Model.

    This function creates a Catalax model from an EnzymeML document, supporting both
    direct ODE conversion and reaction-based conversion. The resulting model includes
    all species, parameters, and equations from the original document.

    The conversion process:
    1. Extracts all species from the document (small molecules, proteins, complexes)
    2. Determines which species are observable based on measurement data
    3. Imports reactions and converts them to ODEs if present
    4. Imports direct ODE specifications if present
    5. Transfers parameter values and bounds from the document
    6. Cleans up unused species to maintain a minimal model

    Args:
        cls: The Model class to instantiate (catalax.model.Model)
        enzmldoc: The EnzymeML document to convert
        name: Optional name for the model. If None, uses the document's name

    Returns:
        A Catalax Model instance containing all species, parameters, and equations
        from the EnzymeML document.

    Raises:
        ValueError: If reactions are missing kinetic laws when from_reactions=True
        AssertionError: If a parameter from the document is not found in the model

    Example:
        >>> import pyenzyme as pe
        >>> import catalax as ctx
        >>>
        >>> doc = pe.read_enzymeml("enzyme_model.json")
        >>> model = ctx.Model.from_enzymeml(doc)
    """
    # Use document name if no name provided
    model_name = name if name is not None else enzmldoc.name
    model = cls(name=model_name)

    # get all species from document
    small_molecules = {sp.id: sp.name for sp in enzmldoc.small_molecules}
    proteins = {sp.id: sp.name for sp in enzmldoc.proteins}
    complexes = {sp.id: sp.name for sp in enzmldoc.complexes}
    all_species = (
        set(small_molecules.keys()) | set(proteins.keys()) | set(complexes.keys())
    )

    added_small_molecules = model.add_state(**small_molecules)
    added_proteins = model.add_state(**proteins)
    added_complexes = model.add_state(**complexes)

    for state in added_small_molecules:
        state.type = "small_molecule"
    for state in added_proteins:
        state.type = "protein"
    for state in added_complexes:
        state.type = "complex"

    # Identify observable and non-observable species
    observables, non_observables = _extract_species_observability(enzmldoc, all_species)

    assert all_species == observables | non_observables, (
        "all species should be either observable or non-observable"
    )

    # Convert equations based on the specified mode
    _import_reactions(model, enzmldoc)
    _import_odes(model, enzmldoc, observables, non_observables)

    _transfer_parameters(model, enzmldoc)

    return model


def _extract_species_observability(
    enzmldoc: pe.EnzymeMLDocument,
    all_species: set[str],
) -> tuple[set[str], set[str]]:
    """
    Determine which species are observable based on measurement data.

    A species is considered observable if it has measurement data in all experiments
    where it appears. Species with missing data in any experiment are considered
    non-observable.

    Args:
        enzmldoc: The EnzymeML document to analyze.
        all_species: A set of all species IDs present in the document.

    Returns:
        A tuple of (observable species, non-observable species).

    Note:
        If no measurements are present in the document, all species are considered
        non-observable.
    """
    if not enzmldoc.measurements:
        return set(), all_species

    # Collect species with measurement data
    species_with_data = {
        species_data.species_id
        for measurement in enzmldoc.measurements
        for species_data in measurement.species_data
        if species_data.data
    }

    # Species with incomplete data (some measurements missing)
    species_with_missing_data = {
        species_data.species_id
        for measurement in enzmldoc.measurements
        for species_data in measurement.species_data
        if not species_data.data
    }

    # Only species with complete data are observable
    observables = species_with_data - species_with_missing_data
    non_observables = all_species - observables

    return observables, non_observables


def _import_odes(
    model: Model,
    enzmldoc: pe.EnzymeMLDocument,
    observables: set[str],
    non_observables: set[str],
) -> None:
    """
    Convert equations directly from ODE specifications in the EnzymeML document.

    This function processes explicit ODE and assignment equations from the EnzymeML
    document. Species appearing on the left-hand side of ODEs are added as states,
    while species only appearing on the right-hand side are added as constants.

    Args:
        model: The Catalax model to populate
        enzmldoc: The EnzymeML document containing the equations
        observables: Set of species IDs that should be marked as observable
        non_observables: Set of species IDs that should not be marked as observable

    Raises:
        ValueError: If an equation has an invalid equation type (only ODE and
                   ASSIGNMENT types are supported)

    Note:
        Observable species are those with complete measurement data across all
        experiments, while non-observable species lack measurement data.
    """
    all_species = observables | non_observables
    lhs_species = set()
    rhs_species = set()

    # Collect species from equations
    for equation in enzmldoc.equations:
        if equation.equation_type == pe.EquationType.ODE:
            lhs_species.add(equation.species_id)
            # Find species mentioned in RHS
            rhs_species.update(
                species for species in all_species if species in equation.equation
            )

    # Add states and constants
    for species in lhs_species:
        model.add_state(species)

    for species in rhs_species - lhs_species:
        model.add_constant(species)

    # Add equations to model
    for equation in enzmldoc.equations:
        if equation.equation_type == pe.EquationType.ASSIGNMENT:
            model.add_assignment(equation.species_id, equation.equation)
        elif equation.equation_type == pe.EquationType.ODE:
            model.add_ode(
                equation.species_id,
                equation.equation,
                observable=equation.species_id in observables,
            )
        else:
            continue


def _import_reactions(
    model: Model,
    enzmldoc: pe.EnzymeMLDocument,
) -> None:
    """
    Convert reaction networks to ODEs using stoichiometry and kinetic laws.

    This function processes reaction networks from the EnzymeML document, converting
    them to a system of ODEs based on stoichiometry and kinetic laws. Each reaction
    is converted to an assignment equation for the reaction rate, and ODEs are
    constructed for each species based on their participation in reactions.

    All species participating in reactions (reactants and products) are automatically
    added as states to the model.

    Args:
        model: The Catalax model to populate
        enzmldoc: The EnzymeML document containing the reactions

    Raises:
        ValueError: If any reaction lacks a kinetic law (all reactions must have
                   kinetic laws for proper conversion)

    Note:
        This function validates that all reactions have kinetic laws before
        processing any reactions to ensure consistency.
    """
    # Validate that all reactions have kinetic laws
    missing_kinetic_laws = []

    for reaction in enzmldoc.reactions:
        if reaction.kinetic_law is None:
            missing_kinetic_laws.append(reaction.id)
            continue

        _import_reaction(model, reaction)

    # Check for missing kinetic laws and raise error if found
    if missing_kinetic_laws:
        reaction_list = ", ".join(missing_kinetic_laws)
        raise ValueError(
            f"The following reactions have no kinetic law: {reaction_list}. "
            f"All reactions must have kinetic laws for conversion from reactions."
        )


def _import_reaction(model: Model, enzml_reac: pe.Reaction) -> None:
    """
    Convert an EnzymeML reaction to a Catalax Reaction.

    This function extracts reaction information including reactants, products,
    stoichiometry, kinetic law, and reversibility from an EnzymeML reaction
    and adds it to the Catalax model.

    Args:
        model: The Catalax model to add the reaction to
        enzml_reac: The EnzymeML reaction to convert

    Note:
        All species participating in the reaction (reactants and products) are
        automatically added as states to the model.
    """
    assert enzml_reac.kinetic_law is not None
    assert enzml_reac.kinetic_law.equation is not None

    states = {r.species_id for r in enzml_reac.reactants} | {
        r.species_id for r in enzml_reac.products
    }

    for state in states:
        model.add_state(state)

    model.add_reaction(
        symbol=enzml_reac.id,
        reactants=[(r.stoichiometry, r.species_id) for r in enzml_reac.reactants],
        products=[(p.stoichiometry, p.species_id) for p in enzml_reac.products],
        equation=enzml_reac.kinetic_law.equation,
        reversible=enzml_reac.reversible,
    )


def _transfer_parameters(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """
    Transfer parameter values and bounds from the EnzymeML document to the model.

    This function copies parameter values, initial values, and bounds from the
    EnzymeML document to the corresponding parameters in the Catalax model.
    All transferred parameters are marked as non-constant (i.e., they can be
    optimized or varied).

    Args:
        model: The Catalax model to update
        enzmldoc: The EnzymeML document containing parameter information

    Raises:
        AssertionError: If a parameter from the document is not found in the model
                       (indicates inconsistency between document and model)

    Note:
        Parameters are automatically marked as non-constant to allow for
        optimization and parameter estimation workflows.
    """
    for parameter in enzmldoc.parameters:
        if parameter.id not in model.parameters:
            raise AssertionError(
                f"Parameter '{parameter.id}' not found in the model. "
                f"Please check the JSON file you are trying to load. "
                f"Available parameters: {list(model.parameters.keys())}"
            )

        model_param = model.parameters[parameter.id]

        # Transfer parameter properties
        model_param.value = parameter.value  # type: ignore
        model_param.initial_value = parameter.initial_value  # type: ignore
        model_param.constant = False  # Allow parameter to be optimized
        model_param.upper_bound = parameter.upper_bound  # type: ignore
        model_param.lower_bound = parameter.lower_bound  # type: ignore


def _cleanup_unused_states_and_constants(model: Model) -> None:
    """
    Remove species from the model that are not referenced in any equations.

    This function identifies species that are not used in any ODE, assignment
    equations, or reactions and removes them from the model. This cleanup helps
    maintain a minimal model with only the necessary species, improving
    computational efficiency and model clarity.

    The cleanup process:
    1. Identifies species that participate in reactions
    2. Identifies species that appear in ODE equations
    3. Removes unused states and constants that don't appear in either

    Args:
        model: The Catalax model to clean up

    Note:
        Uses regex with word boundaries to ensure exact species name matches,
        preventing false positives from partial string matches (e.g., 'A' vs 'ATP').
    """
    reacting_states = model.get_reacting_state_order()

    # Find unused states and constants
    unused_states = {
        species
        for species in model.states
        if species not in reacting_states and not _is_in_odes(species, model.odes)
    }

    unused_constants = {
        constant
        for constant in model.constants
        if constant not in reacting_states and not _is_in_odes(constant, model.odes)
    }

    # Remove unused species
    for species in unused_states:
        del model.states[species]

    for constant in unused_constants:
        del model.constants[constant]


def _is_in_odes(state: str, odes: dict[str, ODE]) -> bool:
    """
    Check if a state is referenced in any ODE equation.

    This function uses regex with word boundaries to ensure exact matches,
    preventing false positives from partial string matches.

    Args:
        state: The state/species name to search for
        odes: Dictionary of ODE objects to search through

    Returns:
        True if the state appears in any ODE equation, False otherwise

    Example:
        >>> odes = {'A': ODE('A', 'k1*B - k2*A')}
        >>> _is_in_odes('A', odes)  # True
        >>> _is_in_odes('B', odes)  # True
        >>> _is_in_odes('C', odes)  # False
    """
    pattern = r"\b" + re.escape(state) + r"\b"
    return any(re.search(pattern, str(ode.equation)) for ode in odes.values())
