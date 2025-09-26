from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

import pyenzyme as pe
from pydantic import BaseModel

if TYPE_CHECKING:
    from catalax.model import Model


def to_enzymeml(
    model: Model,
    enzmldoc: Optional[pe.EnzymeMLDocument] = None,
) -> pe.EnzymeMLDocument:
    """Convert a Catalax Model to a PyEnzyme EnzymeMLDocument.

    This function transforms a Catalax model into an EnzymeML document format,
    preserving all model components including states, parameters, reactions,
    ODEs, and assignments. If an existing document is provided, it will be
    updated with the model's information.

    Args:
        model: The Catalax model to convert
        enzmldoc: Optional existing EnzymeML document to update. If None,
                 creates a new document with the model's name

    Returns:
        An EnzymeMLDocument containing all the model's components

    Example:
        >>> import catalax as ctx
        >>> import pyenzyme as pe
        >>>
        >>> model = ctx.Model(name="Michaelis-Menten")
        >>> doc = to_enzymeml(model)
        >>> # Or update existing document
        >>> existing_doc = pe.EnzymeMLDocument(name="existing")
        >>> updated_doc = to_enzymeml(model, existing_doc)
    """
    if enzmldoc is None:
        enzmldoc = pe.EnzymeMLDocument(name=model.name)
    else:
        enzmldoc = enzmldoc.model_copy(deep=True)

    _add_states_to_document(model, enzmldoc)
    _add_parameters_to_document(model, enzmldoc)
    _add_reactions_to_document(model, enzmldoc)
    _add_odes_to_document(model, enzmldoc)
    _add_assignments_to_document(model, enzmldoc)

    return enzmldoc


def _add_states_to_document(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """Add model states to the EnzymeML document as appropriate species types.

    This function categorizes model states by their type and adds them to the
    appropriate species collections in the EnzymeML document. States are added
    as small molecules, proteins, or complexes based on their type attribute.

    Args:
        model: The Catalax model containing states to add
        enzmldoc: The EnzymeML document to add states to

    Note:
        For existing species, preserve their names if already defined and only update
        if the existing name is None, empty, or same as the ID.
    """
    for state in model.states.values():
        state_id = str(state.symbol)

        if state.type in ("small_molecule", "other", None):
            existing = next(iter(enzmldoc.filter_small_molecules(id=state_id)), None)
            if not existing:
                enzmldoc.add_to_small_molecules(id=state_id, name=state.name)

        elif state.type == "protein":
            existing = next(iter(enzmldoc.filter_proteins(id=state_id)), None)
            if not existing:
                enzmldoc.add_to_proteins(id=state_id, name=state.name)

        elif state.type == "complex":
            existing = next(iter(enzmldoc.filter_complexes(id=state_id)), None)
            if not existing:
                enzmldoc.add_to_complexes(id=state_id, name=state.name)


def _add_parameters_to_document(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """Add model parameters to the EnzymeML document with bounds and values.

    This function transfers all parameters from the Catalax model to the EnzymeML
    document, including their values, bounds, and uncertainty information. If HDI
    (Highest Density Interval) data is available, it is used for bounds; otherwise,
    the parameter's explicit bounds are used.

    Args:
        model: The Catalax model containing parameters to add
        enzmldoc: The EnzymeML document to add parameters to

    Note:
        For existing parameters, preserve string fields (name, symbol) but update
        numerical values (value, bounds).
    """
    for parameter in model.parameters.values():
        param_id = str(parameter.symbol)

        # Determine bounds from HDI if available, otherwise use parameter bounds
        if parameter.hdi:
            lower_bound = parameter.hdi.lower_50
            upper_bound = parameter.hdi.upper_50
        else:
            lower_bound = parameter.lower_bound
            upper_bound = parameter.upper_bound

        existing = next(iter(enzmldoc.filter_parameters(id=parameter.name)), None)
        if existing:
            # Only update numerical values, preserve existing name/symbol if they exist
            update_object(
                existing,
                name=parameter.name,
                symbol=param_id,
                value=parameter.value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                overwrite_strings=True,
            )
        else:
            enzmldoc.add_to_parameters(
                id=param_id,
                name=parameter.name,
                symbol=param_id,
                value=parameter.value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )


def _add_reactions_to_document(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """Add model reactions to the EnzymeML document with kinetic laws.

    This function converts Catalax reactions to EnzymeML format, including
    reactants, products, stoichiometry, kinetic laws, and reversibility
    information. Each reaction's equation is converted to a kinetic law.

    Args:
        model: The Catalax model containing reactions to add
        enzmldoc: The EnzymeML document to add reactions to

    Note:
        For existing reactions, preserve their names if already defined and update
        kinetic laws and stoichiometry.
    """
    for reaction in model.reactions.values():
        reaction_id = str(reaction.symbol)

        reactants = [
            pe.ReactionElement(
                species_id=str(reactant.state),
                stoichiometry=reactant.stoichiometry,
            )
            for reactant in reaction.reactants
        ]

        products = [
            pe.ReactionElement(
                species_id=str(product.state),
                stoichiometry=product.stoichiometry,
            )
            for product in reaction.products
        ]

        kinetic_law = pe.Equation(
            species_id=f"v_{reaction_id}",
            equation=str(reaction.equation),
            equation_type=pe.EquationType.RATE_LAW,
        )

        existing = next(iter(enzmldoc.filter_reactions(id=reaction_id)), None)
        if existing:
            # Preserve existing name, but always update kinetic law and stoichiometry
            update_object(
                existing,
                overwrite_strings=False,
                reversible=reaction.reversible,
                name=reaction.symbol,
                id=reaction_id,
                reactants=reactants,
                products=products,
                kinetic_law=kinetic_law,
            )
        else:
            enzmldoc.add_to_reactions(
                id=reaction_id,
                name=reaction.symbol,
                reactants=reactants,
                products=products,
                kinetic_law=kinetic_law,
                reversible=reaction.reversible,
            )


def _add_odes_to_document(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """Add model ODEs to the EnzymeML document as equations.

    This function converts Catalax ODE objects to EnzymeML equation format,
    preserving the differential equation structure and associating each ODE
    with its corresponding species.

    Args:
        model: The Catalax model containing ODEs to add
        enzmldoc: The EnzymeML document to add ODEs to

    Note:
        Always updates equations since they are the core content.
    """
    for ode in model.odes.values():
        species_id = str(ode.state.symbol)

        existing = next(iter(enzmldoc.filter_equations(species_id=species_id)), None)
        if existing:
            update_object(
                existing,
                equation=str(ode.equation),
                equation_type=pe.EquationType.ODE,
            )
        else:
            enzmldoc.add_to_equations(
                id=species_id,
                species_id=species_id,
                equation=str(ode.equation),
                equation_type=pe.EquationType.ODE,
            )


def _add_assignments_to_document(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """Add model assignments to the EnzymeML document as assignment equations.

    This function converts Catalax assignment rules to EnzymeML assignment
    equations, which define algebraic relationships between variables.

    Args:
        model: The Catalax model containing assignments to add
        enzmldoc: The EnzymeML document to add assignments to

    Note:
        Always updates equations since they are the core content.
    """
    for assignment in model.assignments.values():
        assignment_id = str(assignment.symbol)

        existing = next(iter(enzmldoc.filter_equations(species_id=assignment_id)), None)
        if existing:
            update_object(existing, equation=str(assignment.equation))
        else:
            enzmldoc.add_to_equations(
                id=assignment_id,
                species_id=assignment_id,
                equation_type=pe.EquationType.ASSIGNMENT,
                equation=str(assignment.equation),
            )


def update_object(
    object: BaseModel,
    overwrite_strings: bool = False,
    **kwargs,
) -> None:
    """Update a Pydantic BaseModel object with the given keyword arguments.

    This utility function selectively updates object attributes, with special
    handling for string fields to preserve existing values when desired.

    Args:
        object: The Pydantic BaseModel object to update
        overwrite_strings: If False, existing string values are preserved.
                          If True, string values are updated with new values.
        **kwargs: Keyword arguments representing attribute names and their new values

    Note:
        Non-string attributes are always updated regardless of the overwrite_strings
        parameter.
    """
    for key, value in kwargs.items():
        enzmld_value = getattr(object, key)

        if isinstance(enzmld_value, str) and not overwrite_strings:
            continue

        setattr(object, key, value)


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

    assert (
        all_species == observables | non_observables
    ), "all species should be either observable or non-observable"

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
