from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type

import pyenzyme as pe

if TYPE_CHECKING:
    from catalax.model import Model


class DocumentAnalyzer:
    """Helper class to analyze EnzymeML documents and extract species information."""

    def __init__(self, enzmldoc: pe.EnzymeMLDocument):
        """
        Initialize the DocumentAnalyzer.

        Args:
            enzmldoc: The EnzymeML document to analyze
        """
        self.enzmldoc = enzmldoc
        self._all_species_ids = None
        self._constant_species_ids = None
        self._variable_species_ids = None

    @property
    def all_species_ids(self) -> set[str]:
        """Get all species IDs from the document."""
        if self._all_species_ids is None:
            self._all_species_ids = {
                sp.id
                for sp in (
                    self.enzmldoc.small_molecules
                    + self.enzmldoc.proteins
                    + self.enzmldoc.complexes
                )
            }
        return self._all_species_ids

    @property
    def constant_species_ids(self) -> set[str]:
        """Get IDs of species marked as constant."""
        if self._constant_species_ids is None:
            self._constant_species_ids = {
                sp.id
                for sp in (
                    self.enzmldoc.small_molecules
                    + self.enzmldoc.proteins
                    + self.enzmldoc.complexes
                )
                if sp.constant
            }
        return self._constant_species_ids

    @property
    def variable_species_ids(self) -> set[str]:
        """Get IDs of species that are not constant."""
        if self._variable_species_ids is None:
            self._variable_species_ids = (
                self.all_species_ids - self.constant_species_ids
            )
        return self._variable_species_ids

    def get_equations(self) -> list[str]:
        """
        Get all equation strings from the document.

        Returns:
            List of equation strings from both ODE/assignment specifications
            and reaction kinetic laws
        """
        equations = []

        # Add equations from direct ODE/assignment specifications
        for equation in self.enzmldoc.equations:
            equations.append(equation.equation)

        # Add equations from reaction kinetic laws
        for reaction in self.enzmldoc.reactions:
            if reaction.kinetic_law and reaction.kinetic_law.equation:
                equations.append(reaction.kinetic_law.equation)

        return equations

    def find_species_in_measurements(self) -> tuple[set[str], set[str]]:
        """
        Find which species appear in measurements and classify them by observability.

        Returns:
            Tuple of (observable_species, non_observable_species)
            - observable_species: Species with measurement data
            - non_observable_species: Species referenced in measurements but without data
        """
        observables = set()
        non_observables = set()

        if not self.enzmldoc.measurements:
            return observables, non_observables

        for species_id in self.variable_species_ids:
            species_has_data = False
            species_has_no_data = False
            species_appears_in_measurements = False

            for measurement in self.enzmldoc.measurements:
                for species_data in measurement.species_data:
                    if species_data.species_id != species_id:
                        continue
                    species_appears_in_measurements = True
                    if species_data.data:
                        species_has_data = True
                    else:
                        species_has_no_data = True

            if species_appears_in_measurements:
                if species_has_data and not species_has_no_data:
                    observables.add(species_id)
                else:
                    non_observables.add(species_id)

        return observables, non_observables


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
    only species that appear in measurements, plus any constant species referenced
    in equations.

    Args:
        cls: The Model class to instantiate (catalax.model.Model)
        enzmldoc: The EnzymeML document to convert
        name: Optional name for the model. If None, uses the document's name
        from_reactions: If True, converts from reaction networks to ODEs.
                       If False, uses ODEs directly from the document.

    Returns:
        A Catalax Model instance containing species that appear in measurements
        and constant species referenced in equations.

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
    model_name = name if name is not None else enzmldoc.name
    model = cls(name=model_name)

    analyzer = DocumentAnalyzer(enzmldoc)

    # Find species that appear in measurements
    observables, non_observables = analyzer.find_species_in_measurements()
    species_in_measurements = observables | non_observables

    # Add measurement species to model
    if species_in_measurements:
        model.add_species(species_string=f"{','.join(species_in_measurements)}")

    # Add constant species referenced in equations
    _add_required_constants(model, analyzer, species_in_measurements)

    # Convert equations to model format
    if from_reactions:
        _convert_from_reactions(model, analyzer, observables, non_observables)
    else:
        _convert_from_odes(model, analyzer, observables, non_observables)

    # Transfer parameter values and bounds
    _transfer_parameters(model, enzmldoc)

    return model


def _add_required_constants(
    model: Model,
    analyzer: DocumentAnalyzer,
    species_in_measurements: set[str],
) -> None:
    """
    Add constant species that are referenced in equations but not in measurements.

    Args:
        model: The Catalax model to add constants to
        analyzer: DocumentAnalyzer instance for the EnzymeML document
        species_in_measurements: Set of species IDs that appear in measurements
    """
    equations = analyzer.get_equations()

    for species_id in analyzer.constant_species_ids:
        if species_id not in species_in_measurements:
            # Check if this constant species is referenced in any equation
            for equation in equations:
                pattern = r"\b" + re.escape(species_id) + r"\b"
                if re.search(pattern, str(equation)):
                    model.add_constant(**{species_id: species_id})
                    break


def _convert_from_odes(
    model: Model,
    analyzer: DocumentAnalyzer,
    observables: set[str],
    non_observables: set[str],
) -> None:
    """
    Convert equations directly from ODE specifications in the EnzymeML document.

    Args:
        model: The Catalax model to add equations to
        analyzer: DocumentAnalyzer instance for the EnzymeML document
        observables: Set of observable species IDs
        non_observables: Set of non-observable species IDs
    """
    species_in_measurements = observables | non_observables

    # Add any additional species/constants needed for ODEs
    _add_ode_dependencies(model, analyzer, species_in_measurements)

    # Add equations to model
    for equation in analyzer.enzmldoc.equations:
        if equation.equation_type == pe.EquationType.ASSIGNMENT:
            model.add_assignment(equation.species_id, equation.equation)
        elif equation.equation_type == pe.EquationType.ODE:
            # Only add ODE if the species appears in measurements
            if equation.species_id in species_in_measurements:
                is_observable = equation.species_id in observables
                model.add_ode(
                    equation.species_id, equation.equation, observable=is_observable
                )
        else:
            raise ValueError(
                f"Equation {equation.species_id} has invalid type: {equation.equation_type}. "
                f"Only ODE and ASSIGNMENT equations are supported."
            )


def _add_ode_dependencies(
    model: Model,
    analyzer: DocumentAnalyzer,
    species_in_measurements: set[str],
) -> None:
    """
    Add species that are referenced in ODEs but not in measurements as constants.

    Args:
        model: The Catalax model to add dependencies to
        analyzer: DocumentAnalyzer instance for the EnzymeML document
        species_in_measurements: Set of species IDs that appear in measurements
    """
    rhs_species = set()
    lhs_species = set()

    # Find species referenced in ODEs
    for equation in analyzer.enzmldoc.equations:
        if equation.equation_type == pe.EquationType.ODE:
            lhs_species.add(equation.species_id)
            for species in analyzer.all_species_ids:
                if species in equation.equation:
                    rhs_species.add(species)

    # Add species that appear in measurement data to model
    for species in lhs_species:
        if species in species_in_measurements:
            model.add_species(species)

    # Add RHS species that aren't already handled as constants
    for species in rhs_species:
        if (
            species not in lhs_species
            and species not in species_in_measurements
            and species not in model.constants
        ):
            model.add_constant(species)


def _convert_from_reactions(
    model: Model,
    analyzer: DocumentAnalyzer,
    observables: set[str],
    non_observables: set[str],
) -> None:
    """
    Convert reaction networks to ODEs using stoichiometry and kinetic laws.

    Args:
        model: The Catalax model to add reactions to
        analyzer: DocumentAnalyzer instance for the EnzymeML document
        observables: Set of observable species IDs
        non_observables: Set of non-observable species IDs

    Raises:
        ValueError: If any reactions are missing kinetic laws
    """
    species_in_measurements = observables | non_observables

    # Validate all reactions have kinetic laws
    missing_kinetic_laws = [
        reaction.id
        for reaction in analyzer.enzmldoc.reactions
        if reaction.kinetic_law is None
    ]

    if missing_kinetic_laws:
        reaction_list = ", ".join(missing_kinetic_laws)
        raise ValueError(
            f"The following reactions have no kinetic law: {reaction_list}. "
            f"All reactions must have kinetic laws for conversion from reactions."
        )

    # Build stoichiometric equations
    kinetic_laws = {}
    species_ode_terms = {species: [] for species in species_in_measurements}
    species_in_equations = set()

    for reaction_index, reaction in enumerate(analyzer.enzmldoc.reactions):
        rate_id = f"r{reaction_index}"
        kinetic_laws[rate_id] = reaction.kinetic_law.equation

        # Track species mentioned in kinetic law
        for species in analyzer.all_species_ids:
            if (
                species == reaction.kinetic_law.species_id
                or species in reaction.kinetic_law.equation
            ):
                species_in_equations.add(species)

        # Add stoichiometric contributions
        for reactant in reaction.reactants:
            if reactant.species_id in species_ode_terms:
                species_ode_terms[reactant.species_id].append(
                    f"(-{reactant.stoichiometry})*{rate_id}"
                )

        for product in reaction.products:
            if product.species_id in species_ode_terms:
                species_ode_terms[product.species_id].append(
                    f"({product.stoichiometry})*{rate_id}"
                )

    # Add reaction rate assignments
    for rate_id, kinetic_law in kinetic_laws.items():
        model.add_assignment(rate_id, kinetic_law)

    # Add ODEs for species with reactions
    for species_id, ode_terms in species_ode_terms.items():
        if (
            ode_terms
            and species_id not in analyzer.constant_species_ids
            and species_id in species_in_measurements
        ):
            ode_equation = " + ".join(ode_terms)
            is_observable = species_id in observables
            model.add_ode(species_id, ode_equation, observable=is_observable)

    # Add missing species as constants
    missing_species = (
        species_in_equations
        - set(model.odes.keys())
        - species_in_measurements
        - set(model.constants.keys())
    )

    for species in missing_species:
        model.add_constant(species)


def _transfer_parameters(model: Model, enzmldoc: pe.EnzymeMLDocument) -> None:
    """
    Transfer parameter values and bounds from the EnzymeML document to the model.

    Args:
        model: The Catalax model to transfer parameters to
        enzmldoc: The EnzymeML document containing parameter information

    Raises:
        AssertionError: If a parameter from the document is not found in the model
    """
    for parameter in enzmldoc.parameters:
        if parameter.id not in model.parameters:
            available_params = list(model.parameters.keys())
            raise AssertionError(
                f"Parameter '{parameter.id}' not found in the model. "
                f"Available parameters: {available_params}"
            )

        model_param = model.parameters[parameter.id]
        model_param.value = parameter.value  # type: ignore
        model_param.initial_value = parameter.initial_value  # type: ignore
        model_param.constant = False  # Allow parameter to be optimized
        model_param.upper_bound = parameter.upper_bound  # type: ignore
        model_param.lower_bound = parameter.lower_bound  # type: ignore
