from __future__ import annotations

import re
from typing import List, Optional, Tuple

from sympy import Expr

from catalax.model.base import CatalaxBase
from catalax.model.equation import Equation


class Reaction(Equation):
    """A chemical reaction with reactants, products, and kinetic properties.

    This class represents a chemical reaction that can be reversible or irreversible,
    with specified reactants and products. It inherits from Equation to provide
    kinetic modeling capabilities.

    Attributes:
        name: The name or identifier of the reaction
        reactants: List of reactant states with their stoichiometric coefficients
        products: List of product states with their stoichiometric coefficients
        reversible: Whether the reaction can proceed in both directions
    """

    symbol: str
    reactants: List[ReactionElement]
    products: List[ReactionElement]
    reversible: bool

    @classmethod
    def from_scheme(
        cls,
        symbol: str,
        schema: str,
        equation: str | Expr,
        reversible: bool,
        states: Optional[List[str]] = None,
    ) -> Reaction:
        """Create a Reaction from a schema string.

        The schema should follow the format: "reactants -> products" where:
        - Reactants and products are separated by arrows
        - Each side contains species separated by '+'
        - Species can contain letters, numbers, and underscores
        - Stoichiometric coefficients are optional (defaults to 1.0)

        Supported arrow variants:
        - -> (irreversible)
        - <-> (reversible)
        - => (irreversible)
        - <=> (reversible)
        - <==> (reversible)
        - <===> (reversible)
        - > (irreversible)

        Examples:
            >>> Reaction.from_schema("r1", "1 s1 + 1 s2 -> 1 s3", "k*s1*s2", False)
            >>> Reaction.from_schema("r2", "A + B <-> C", "k*A*B", True)
            >>> Reaction.from_schema("r3", "2 H2 + O2 <=> 2 H2O", "k*H2*O2", True)

        Args:
            name: The name or identifier of the reaction
            schema: The reaction schema string
            equation: The kinetic equation as a string
            reversible: Whether the reaction is reversible

        Returns:
            A Reaction object with parsed reactants and products

        Raises:
            ValueError: If no supported arrow pattern is found in schema
            ValueError: If a term cannot be parsed
        """
        reactants_str, products_str = cls._split_schema(schema)
        reactants = cls._parse_side(reactants_str)
        products = cls._parse_side(products_str)

        # Validate that reaction has at least one reactant and one product
        if len(reactants) == 0 or len(products) == 0:
            raise ValueError("Reaction must have at least one reactant and one product")

        # Check if all the stoichiometries are positive and not zero
        if any(element.stoichiometry <= 0 for element in reactants + products):
            raise ValueError("Stoichiometries must be positive and not zero")

        if states is not None:
            undefined_states = []
            for element in reactants + products:
                if element.state not in states:
                    undefined_states.append(element.state)
            if undefined_states:
                raise ValueError(f"States {undefined_states} not found in states")

        return cls(
            symbol=symbol,
            reactants=reactants,
            products=products,
            reversible=reversible,
            equation=equation,  # type: ignore
        )

    @staticmethod
    def _split_schema(schema: str) -> tuple[str, str]:
        """Split schema into reactants and products parts.

        Args:
            schema: The reaction schema string

        Returns:
            Tuple of (reactants_string, products_string)

        Raises:
            ValueError: If no supported arrow pattern is found
        """
        # Order matters: longer patterns first to avoid partial matches
        arrow_patterns = [
            r"<===>",
            r"<==>",
            r"<=>",
            r"<->",
            r"=>",
            r"->",
            r">",
        ]

        for pattern in arrow_patterns:
            if re.search(pattern, schema):
                parts = re.split(pattern, schema, maxsplit=1)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

        raise ValueError(f"No supported arrow pattern found in schema: {schema}")

    @staticmethod
    def _parse_side(side_str: str) -> list[ReactionElement]:
        """Parse one side of the reaction (reactants or products).

        Args:
            side_str: String containing species separated by '+'

        Returns:
            List of ReactionElement objects
        """
        if not side_str.strip():
            return []

        terms = [term.strip() for term in side_str.split("+") if term.strip()]
        return [Reaction._parse_term(term) for term in terms]

    @staticmethod
    def _parse_term(term: str) -> ReactionElement:
        """Parse a single term into coefficient and species name.

        Args:
            term: A single term like "2 H2O" or "ATP"

        Returns:
            ReactionElement with parsed state and stoichiometry

        Raises:
            ValueError: If the term cannot be parsed
        """
        # Match optional coefficient followed by species name
        coefficient_pattern = r"^(\d+(?:\.\d+)?)?\s*([a-zA-Z_][a-zA-Z0-9_]*)"
        match = re.match(coefficient_pattern, term)

        if match:
            coeff_str, species = match.groups()
            coefficient = float(coeff_str) if coeff_str else 1.0
            return ReactionElement(state=species, stoichiometry=coefficient)

        # Fallback: try to match just species name
        species_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)"
        species_match = re.match(species_pattern, term)

        if species_match:
            species = species_match.group(1)
            return ReactionElement(state=species, stoichiometry=1.0)

        raise ValueError(f"Could not parse term: {term}")


class ReactionElement(CatalaxBase):
    """A state participating in a reaction with its stoichiometric coefficient.

    This class represents a single state in a chemical reaction along with
    its stoichiometric coefficient, which determines how many molecules of
    the state participate in the reaction.

    Attributes:
        state: The name or identifier of the chemical state
        stoichiometry: The stoichiometric coefficient (number of molecules)
    """

    state: str
    stoichiometry: float

    @classmethod
    def from_tuple(cls, tuple: Tuple[float, str]) -> ReactionElement:
        """Create a ReactionElement from a tuple of stoichiometry and state name."""
        return cls(state=tuple[1], stoichiometry=tuple[0])


Reaction.model_rebuild()
