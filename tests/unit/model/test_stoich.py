import jax.numpy as jnp
import numpy as np

from catalax.model.reaction import Reaction, ReactionElement
from catalax.model.stoich import derive_stoich_matrix


class TestStoichMatrix:
    def test_simple_reaction(self):
        """Test stoichiometric matrix for a simple A -> B reaction."""

        # Arrange: A -> B reaction
        reactions = [
            Reaction(
                symbol="r1",
                reactants=[ReactionElement(state="A", stoichiometry=1.0)],
                products=[ReactionElement(state="B", stoichiometry=1.0)],
                reversible=False,
                equation="k1*A",  # type: ignore
            )
        ]
        species_order = ["A", "B"]

        # Expected stoichiometric matrix:
        # Species A: -1 (consumed in reaction r1)
        # Species B: +1 (produced in reaction r1)
        expected = jnp.array(
            [
                [-1.0],  # A
                [1.0],  # B
            ]
        )

        # Act
        result = derive_stoich_matrix(reactions, species_order)

        # Assert
        assert result.shape == (2, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_reactions_overlapping_species(self):
        """Test stoichiometric matrix for multiple reactions with overlapping species."""

        # Arrange:
        # Reaction r1: A + B -> C (with stoichiometry 2A + B -> C)
        # Reaction r2: C -> D + E
        reactions = [
            Reaction(
                symbol="r1",
                reactants=[
                    ReactionElement(state="A", stoichiometry=2.0),
                    ReactionElement(state="B", stoichiometry=1.0),
                ],
                products=[ReactionElement(state="C", stoichiometry=1.0)],
                reversible=False,
                equation="k1*A*A*B",  # type: ignore
            ),
            Reaction(
                symbol="r2",
                reactants=[ReactionElement(state="C", stoichiometry=1.0)],
                products=[
                    ReactionElement(state="D", stoichiometry=1.0),
                    ReactionElement(state="E", stoichiometry=2.0),
                ],
                reversible=False,
                equation="k2*C",  # type: ignore
            ),
        ]
        species_order = ["A", "B", "C", "D", "E"]

        # Expected stoichiometric matrix (species x reactions):
        #         r1    r2
        # A    [-2.0,  0.0]
        # B    [-1.0,  0.0]
        # C    [ 1.0, -1.0]
        # D    [ 0.0,  1.0]
        # E    [ 0.0,  2.0]
        expected = jnp.array(
            [
                [-2.0, 0.0],  # A
                [-1.0, 0.0],  # B
                [1.0, -1.0],  # C
                [0.0, 1.0],  # D
                [0.0, 2.0],  # E
            ]
        )

        # Act
        result = derive_stoich_matrix(reactions, species_order)

        # Assert
        assert result.shape == (5, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_reaction_ordering(self):
        """Test that reactions are properly ordered by name for consistent matrix construction."""

        # Arrange: Create reactions in non-alphabetical order
        reactions = [
            Reaction(
                symbol="z_reaction",
                reactants=[ReactionElement(state="A", stoichiometry=1.0)],
                products=[ReactionElement(state="B", stoichiometry=1.0)],
                reversible=False,
                equation="k1*A",  # type: ignore
            ),
            Reaction(
                symbol="a_reaction",
                reactants=[ReactionElement(state="B", stoichiometry=1.0)],
                products=[ReactionElement(state="C", stoichiometry=1.0)],
                reversible=False,
                equation="k2*B",  # type: ignore
            ),
        ]
        species_order = ["A", "B", "C"]

        # Expected: reactions should be ordered alphabetically (a_reaction, z_reaction)
        # So column 0 = a_reaction (B->C), column 1 = z_reaction (A->B)
        expected = jnp.array(
            [
                [0.0, -1.0],  # A: not in a_reaction, consumed in z_reaction
                [-1.0, 1.0],  # B: consumed in a_reaction, produced in z_reaction
                [1.0, 0.0],  # C: produced in a_reaction, not in z_reaction
            ]
        )

        # Act
        result = derive_stoich_matrix(reactions, species_order)

        # Assert
        assert result.shape == (3, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_species_not_in_reactions(self):
        """Test that species not involved in reactions get zero coefficients."""

        # Arrange: Only A -> B, but include species C in the order
        reactions = [
            Reaction(
                symbol="r1",
                reactants=[ReactionElement(state="A", stoichiometry=1.0)],
                products=[ReactionElement(state="B", stoichiometry=1.0)],
                reversible=False,
                equation="k1*A",  # type: ignore
            )
        ]
        species_order = ["A", "B", "C"]  # C is not in any reaction

        # Expected: C should have zero coefficient
        expected = jnp.array(
            [
                [-1.0],  # A
                [1.0],  # B
                [0.0],  # C (not involved)
            ]
        )

        # Act
        result = derive_stoich_matrix(reactions, species_order)

        # Assert
        assert result.shape == (3, 1)
        np.testing.assert_array_almost_equal(result, expected)
