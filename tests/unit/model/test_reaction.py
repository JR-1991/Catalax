import pytest

from catalax.model.reaction import Reaction, ReactionElement


class TestReactionFromSchema:
    """Test cases for Reaction.from_schema method."""

    def test_basic_irreversible_reaction(self):
        """Test basic irreversible reaction with -> arrow."""
        reaction = Reaction.from_scheme(
            symbol="r1",
            schema="A + B -> C",
            equation="k*A*B",
            reversible=False,
        )

        assert reaction.symbol == "r1"
        assert reaction.reversible is False
        assert len(reaction.reactants) == 2
        assert len(reaction.products) == 1
        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.reactants[1] == ReactionElement(state="B", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="C", stoichiometry=1.0)

    def test_basic_reversible_reaction(self):
        """Test basic reversible reaction with <-> arrow."""
        reaction = Reaction.from_scheme(
            symbol="r2",
            schema="A + B <-> C",
            equation="k*A*B",
            reversible=True,
        )

        assert reaction.symbol == "r2"
        assert reaction.reversible is True
        assert len(reaction.reactants) == 2
        assert len(reaction.products) == 1

    def test_arrow_variants(self):
        """Test all supported arrow variants."""
        arrow_tests = [
            ("A -> B", False),
            ("A <-> B", True),
            ("A => B", False),
            ("A <=> B", True),
            ("A <==> B", True),
            ("A <===> B", True),
            ("A > B", False),
        ]

        for schema, expected_reversible in arrow_tests:
            reaction = Reaction.from_scheme(
                symbol="test",
                schema=schema,
                equation="k*A",
                reversible=expected_reversible,
            )
            assert len(reaction.reactants) == 1
            assert len(reaction.products) == 1
            assert reaction.reactants[0].state == "A"
            assert reaction.products[0].state == "B"

    def test_stoichiometric_coefficients(self):
        """Test reactions with stoichiometric coefficients."""
        reaction = Reaction.from_scheme(
            symbol="r3",
            schema="2 A + 3 B -> 4 C",
            equation="k*A*B",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=2.0)
        assert reaction.reactants[1] == ReactionElement(state="B", stoichiometry=3.0)
        assert reaction.products[0] == ReactionElement(state="C", stoichiometry=4.0)

    def test_decimal_coefficients(self):
        """Test reactions with decimal stoichiometric coefficients."""
        reaction = Reaction.from_scheme(
            symbol="r4",
            schema="1.5 A + 2.5 B -> 3.0 C",
            equation="k*A*B",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.5)
        assert reaction.reactants[1] == ReactionElement(state="B", stoichiometry=2.5)
        assert reaction.products[0] == ReactionElement(state="C", stoichiometry=3.0)

    def test_species_with_underscores(self):
        """Test species names with underscores."""
        reaction = Reaction.from_scheme(
            symbol="r5",
            schema="reactant_1 + reactant_2 -> product_1",
            equation="k*reactant_1*reactant_2",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(
            state="reactant_1", stoichiometry=1.0
        )
        assert reaction.reactants[1] == ReactionElement(
            state="reactant_2", stoichiometry=1.0
        )
        assert reaction.products[0] == ReactionElement(
            state="product_1", stoichiometry=1.0
        )

    def test_species_with_numbers(self):
        """Test species names with numbers."""
        reaction = Reaction.from_scheme(
            symbol="r6",
            schema="H2 + O2 -> H2O",
            equation="k*H2*O2",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(state="H2", stoichiometry=1.0)
        assert reaction.reactants[1] == ReactionElement(state="O2", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="H2O", stoichiometry=1.0)

    def test_complex_reaction(self):
        """Test complex reaction with multiple reactants and products."""
        reaction = Reaction.from_scheme(
            symbol="r7",
            schema="2 H2 + O2 -> 2 H2O",
            equation="k*H2*O2",
            reversible=False,
        )

        assert len(reaction.reactants) == 2
        assert len(reaction.products) == 1
        assert reaction.reactants[0] == ReactionElement(state="H2", stoichiometry=2.0)
        assert reaction.reactants[1] == ReactionElement(state="O2", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="H2O", stoichiometry=2.0)

    def test_empty_reactants(self):
        """Test reaction with empty reactants."""
        with pytest.raises(
            ValueError, match="Reaction must have at least one reactant and one product"
        ):
            Reaction.from_scheme(
                symbol="r8",
                schema="-> A",
                equation="k",
                reversible=False,
            )

    def test_empty_products(self):
        """Test reaction with empty products."""
        with pytest.raises(
            ValueError, match="Reaction must have at least one reactant and one product"
        ):
            Reaction.from_scheme(
                symbol="r9",
                schema="A ->",
                equation="k*A",
                reversible=False,
            )

    def test_no_arrow(self):
        """Test schema without any arrow (should raise error)."""
        with pytest.raises(ValueError, match="No supported arrow pattern found"):
            Reaction.from_scheme(
                symbol="r10",
                schema="A + B",
                equation="k*A*B",
                reversible=False,
            )

    def test_multiple_arrows(self):
        """Test schema with multiple arrows (uses first arrow found)."""
        reaction = Reaction.from_scheme(
            symbol="r11",
            schema="A -> B -> C",
            equation="k*A",
            reversible=False,
        )

        assert reaction.symbol == "r11"
        assert len(reaction.reactants) == 1
        assert len(reaction.products) == 1
        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="B", stoichiometry=1.0)

    def test_invalid_species_name(self):
        """Test species name that doesn't match pattern."""
        with pytest.raises(ValueError, match="Could not parse term"):
            Reaction.from_scheme(
                symbol="r12",
                schema="A + @invalid -> B",
                equation="k*A",
                reversible=False,
            )

    def test_whitespace_handling(self):
        """Test proper whitespace handling."""
        reaction = Reaction.from_scheme(
            symbol="r13",
            schema="  A  +  B  ->  C  ",
            equation="k*A*B",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.reactants[1] == ReactionElement(state="B", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="C", stoichiometry=1.0)

    def test_single_reactant_single_product(self):
        """Test reaction with single reactant and single product."""
        reaction = Reaction.from_scheme(
            symbol="r14",
            schema="A -> B",
            equation="k*A",
            reversible=False,
        )

        assert len(reaction.reactants) == 1
        assert len(reaction.products) == 1
        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="B", stoichiometry=1.0)

    def test_multiple_products(self):
        """Test reaction with multiple products."""
        reaction = Reaction.from_scheme(
            symbol="r15",
            schema="A -> B + C + D",
            equation="k*A",
            reversible=False,
        )

        assert len(reaction.reactants) == 1
        assert len(reaction.products) == 3
        assert reaction.products[0] == ReactionElement(state="B", stoichiometry=1.0)
        assert reaction.products[1] == ReactionElement(state="C", stoichiometry=1.0)
        assert reaction.products[2] == ReactionElement(state="D", stoichiometry=1.0)

    def test_zero_coefficient_handling(self):
        """Test that zero coefficients are handled properly."""
        with pytest.raises(
            ValueError, match="Stoichiometries must be positive and not zero"
        ):
            Reaction.from_scheme(
                symbol="r16",
                schema="0 A + 1 B -> C",
                equation="k*B",
                reversible=False,
            )

    def test_very_long_arrow(self):
        """Test the longest supported arrow pattern."""
        reaction = Reaction.from_scheme(
            symbol="r17",
            schema="A <===> B",
            equation="k*A",
            reversible=True,
        )

        assert len(reaction.reactants) == 1
        assert len(reaction.products) == 1
        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.products[0] == ReactionElement(state="B", stoichiometry=1.0)

    def test_mixed_coefficient_types(self):
        """Test mixing integer and decimal coefficients."""
        reaction = Reaction.from_scheme(
            symbol="r18",
            schema="1 A + 2.5 B + 3 C -> 4.5 D",
            equation="k*A*B*C",
            reversible=False,
        )

        assert reaction.reactants[0] == ReactionElement(state="A", stoichiometry=1.0)
        assert reaction.reactants[1] == ReactionElement(state="B", stoichiometry=2.5)
        assert reaction.reactants[2] == ReactionElement(state="C", stoichiometry=3.0)
        assert reaction.products[0] == ReactionElement(state="D", stoichiometry=4.5)
