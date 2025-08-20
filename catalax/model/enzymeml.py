from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type

import pyenzyme as pe

if TYPE_CHECKING:
    from catalax.model import Model


def from_enzymeml(
    cls: Type[Model],
    enzmldoc: pe.EnzymeMLDocument,
    name: str | None = None,
    from_reactions: bool = False,
) -> Model:
    """
    Convert a PyEnzyme EnzymeMLDocument to a Catalax Model.

    This function creates a Catalax model from an EnzymeML document, supporting both
    direct ODE conversion and reaction-based conversion. The resulting model includes
    all species, parameters, and equations from the original document.

    Args:
        cls: The Model class to instantiate (catalax.model.Model)
        enzmldoc: The EnzymeML document to convert
        name: Optional name for the model. If None, uses the document's name
        from_reactions: If True, converts from reaction networks to ODEs.
                       If False, uses ODEs directly from the document.

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
        >>> model = ctx.Model.from_enzymeml(doc, from_reactions=True)
    """
    # Use document name if no name provided
    model_name = name if name is not None else enzmldoc.name
    model = cls(name=model_name)

    # get all species from document
    all_species = {sp.id for sp in enzmldoc.small_molecules} | {
        sp.id for sp in enzmldoc.proteins
    }

    # Identify observable and non-observable species
    observables, non_observables = _extract_species_observability(enzmldoc, all_species)

    assert all_species == observables | non_observables, (
        "all species should be either observable or non-observable"
    )

    # Convert equations based on the specified mode
    if from_reactions:
        _convert_from_reactions(model, enzmldoc, observables, non_observables)
    else:
        _convert_from_odes(model, enzmldoc, observables, non_observables)

    _transfer_parameters(model, enzmldoc)
    # _cleanup_unused_species(model)

    return model


def _extract_species_observability(
    enzmldoc: pe.EnzymeMLDocument,
    all_species: set[str],
) -> tuple[set[str], set[str]]:
    """
    Determine which species in the EnzymeML document are observable based on measurement data.

    This function analyzes the measurement data in the EnzymeML document to classify species
    into observable and non-observable categories. A species is considered observable if it
    has associated measurement data. If a species is only observable in some measurements,
    it is marked as non-observable. If no measurement data is available, all species are
    marked as non-observable by default.

    Args:
        enzmldoc: The EnzymeML document to analyze.
        all_species: A set of all species IDs present in the document.

    Returns:
        A tuple containing:
        - Set of observable species IDs (species with measurement data)
        - Set of non-observable species IDs (species without measurement data or all species if no measurements)
    """
    observables = set()
    non_observables = set()

    # If no measurements exist, all species are non-observable by default
    if not enzmldoc.measurements:
        return set(), all_species

    # Check each species against measurement data
    for species_id in all_species:
        species_has_data = False
        species_has_no_data = False

        for measurement in enzmldoc.measurements:
            for species_data in measurement.species_data:
                if species_data.species_id != species_id:
                    continue
                if species_data.data:
                    species_has_data = True
                else:
                    species_has_no_data = True

        # Classify based on measurement data
        if species_has_data and not species_has_no_data:
            # Species has data in all measurements where it appears
            observables.add(species_id)
        else:
            # Species has no data or mixed data (some with data, some without)
            non_observables.add(species_id)

    return observables, non_observables


def _convert_from_odes(
    model: Model,
    enzmldoc: pe.EnzymeMLDocument,
    observables: set[str],
    non_observables: set[str],
) -> None:
    """
    Convert equations directly from ODE specifications in the EnzymeML document.

    This function processes ODE and assignment equations directly from the EnzymeML
    document, adding them to the model with appropriate observability settings.

    Args:
        model: The Catalax model to populate
        enzmldoc: The EnzymeML document containing the equations
        observables: Set of species IDs that should be marked as observable
        non_observables: Set of species IDs that should not be marked as observable

    Raises:
        AssertionError: If a species mentioned in any right hand side of an ODE equation
        does not have a corresponding ODE equation.
        ValueError: If an equation has an invalid equation type
    """

    all_species = observables | non_observables
    rhs_species = set()
    lhs_species = set()

    # Identify species mentioned in ODEs
    for equation in enzmldoc.equations:
        if not equation.equation_type == pe.EquationType.ODE:
            continue

        # add LHS species as `species` to model
        lhs_species.add(equation.species_id)

        # bookkeep species are part of RHS of ODE
        for species in all_species:
            if species in equation.equation:
                rhs_species.add(species)

    # add LHS species as `species` to model
    for species in lhs_species:
        model.add_species(species)

    # add RHS species without ODE as constants to model
    for species in rhs_species:
        if species not in lhs_species:
            model.add_constant(species)

    # Populate model with ODEs and assignments
    for equation in enzmldoc.equations:
        if equation.equation_type == pe.EquationType.ASSIGNMENT:
            model.add_assignment(equation.species_id, equation.equation)
        elif equation.equation_type == pe.EquationType.ODE:
            is_observable = equation.species_id in observables
            model.add_ode(
                equation.species_id, equation.equation, observable=is_observable
            )
        else:
            raise ValueError(
                f"Equation {equation.id} has an invalid equation type: {equation.equation_type}"
                f"Only ODE and ASSIGNMENT equations are supported."
            )


def _convert_from_reactions(
    model: Model,
    enzmldoc: pe.EnzymeMLDocument,
    observables: set[str],
    non_observables: set[str],
) -> None:
    """
    Convert reaction networks to ODEs using stoichiometry and kinetic laws.

    This function processes reaction networks from the EnzymeML document, converting
    them to a system of ODEs based on stoichiometry and kinetic laws. Each reaction
    is converted to an assignment equation for the reaction rate, and ODEs are
    constructed for each species based on their participation in reactions.

    Args:
        model: The Catalax model to populate
        enzmldoc: The EnzymeML document containing the reactions
        observables: Set of species IDs that should be marked as observable
        non_observables: Set of species IDs that should not be marked as observable

    Raises:
        ValueError: If any reaction lacks a kinetic law
    """
    # Validate that all reactions have kinetic laws
    missing_kinetic_laws = []
    kinetic_laws = {}
    species_in_equations = set()
    species_ode_terms: dict[str, list[str]] = {
        species: [] for species in observables | non_observables
    }

    for reaction_index, reaction in enumerate(enzmldoc.reactions):
        if reaction.kinetic_law is None:
            missing_kinetic_laws.append(reaction.id)
            continue

        # aggregate all species mentioned in kinetic law
        for species in observables | non_observables:
            if (
                species == reaction.kinetic_law.species_id
                or species in reaction.kinetic_law.equation
            ):
                species_in_equations.add(species)

        # Create a unique identifier for this reaction rate
        rate_id = f"r{reaction_index}"
        kinetic_laws[rate_id] = reaction.kinetic_law.equation

        # Add stoichiometric contributions for reactants (negative)
        for reactant in reaction.reactants:
            species_id = reactant.species_id
            stoichiometry = reactant.stoichiometry
            species_ode_terms[species_id].append(f"(-{stoichiometry})*{rate_id}")

        # Add stoichiometric contributions for products (positive)
        for product in reaction.products:
            species_id = product.species_id
            stoichiometry = product.stoichiometry
            species_ode_terms[species_id].append(f"({stoichiometry})*{rate_id}")

    # Check for missing kinetic laws and raise error if found
    if missing_kinetic_laws:
        reaction_list = ", ".join(missing_kinetic_laws)
        raise ValueError(
            f"The following reactions have no kinetic law: {reaction_list}. "
            f"All reactions must have kinetic laws for conversion from reactions."
        )

    # Add reaction rate assignments to the model
    for rate_id, kinetic_law in kinetic_laws.items():
        model.add_assignment(rate_id, kinetic_law)

    # Add ODEs for each species based on their stoichiometric contributions
    for species_id, ode_terms in species_ode_terms.items():
        if ode_terms:  # Only add ODE if species participates in reactions
            ode_equation = " + ".join(ode_terms)
            is_observable = species_id in observables
            model.add_ode(species_id, ode_equation, observable=is_observable)

    missing_odes = species_in_equations - set(model.odes.keys())
    for species in missing_odes:
        model.add_constant(species)


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


def _cleanup_unused_species(model: Model) -> None:
    """
    Remove species from the model that are not referenced in any equations.

    This function identifies species that are not used in any ODE or assignment
    equations and removes them from the model. This cleanup helps maintain a
    minimal model with only the necessary species.

    Args:
        model: The Catalax model to clean up

    Note:
        Uses regex with word boundaries to ensure exact species name matches,
        preventing false positives from partial string matches.
    """
    equations = [
        obj.equation
        for obj in list(model.assignments.values()) + list(model.odes.values())
    ]
    to_delete = set()
    for species in model.species:
        # Use regex with word boundaries to find exact matches only
        pattern = r"\b" + re.escape(species) + r"\b"
        if not any(re.search(pattern, str(equation)) for equation in equations):
            to_delete.add(species)

    for species in to_delete:
        del model.species[species]
