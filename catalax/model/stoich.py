from typing import List

import jax
import jax.numpy as jnp

from catalax.model.reaction import Reaction


def derive_stoich_matrix(
    reactions: List[Reaction],
    state_order: List[str],
) -> jax.Array:
    """Derive the stoichiometric matrix from a list of reactions and a list of state.

    The stoichiometric matrix S has dimensions (n_state, n_reactions) where:
    - Each row represents a state
    - Each column represents a reaction
    - S[i,j] is the stoichiometric coefficient of state i in reaction j
    - Negative values indicate reactants (consumed)
    - Positive values indicate products (produced)
    - Zero values indicate the state is not involved in the reaction

    Args:
        reactions: List of reactions to include in the matrix
        state_order: Ordered list of state names defining the row order

    Returns:
        JAX array of shape (n_state, n_reactions) containing stoichiometric coefficients
    """
    reaction_order = _get_reaction_order(reactions)
    state_to_idx = _create_state_mapping(state_order)

    row_indices, col_indices, coefficients = _collect_matrix_entries(
        reaction_order=reaction_order,
        state_to_idx=state_to_idx,
    )

    return _build_jax_matrix(
        row_indices=row_indices,
        col_indices=col_indices,
        coefficients=coefficients,
        n_state=len(state_order),
        n_reactions=len(reaction_order),
    )


def _get_reaction_order(reactions: List[Reaction]) -> List[Reaction]:
    """Get the order of the reactions by name.

    Reactions are sorted alphabetically by name to ensure consistent
    column ordering in the stoichiometric matrix.

    Args:
        reactions: List of reactions to sort

    Returns:
        List of reactions sorted alphabetically by name
    """
    return sorted(reactions, key=lambda x: x.symbol)


def _create_state_mapping(state_order: List[str]) -> dict[str, int]:
    """Create a mapping from state names to matrix row indices.

    Args:
        state_order: Ordered list of state names

    Returns:
        Dictionary mapping state names to their row indices
    """
    return {name: idx for idx, name in enumerate(state_order)}


def _collect_matrix_entries(
    reaction_order: List[Reaction], state_to_idx: dict[str, int]
) -> tuple[list[int], list[int], list[float]]:
    """Collect row indices, column indices, and coefficients for stoichiometric matrix.

    Processes all reactions to extract the sparse matrix entries needed to
    construct the stoichiometric matrix.

    Args:
        reaction_order: Ordered list of reactions (defines column order)
        state_to_idx: Mapping from state names to row indices

    Returns:
        Tuple of (row_indices, col_indices, coefficients) for sparse matrix construction
    """
    row_indices = []
    col_indices = []
    coefficients = []

    for reaction_idx, reaction in enumerate(reaction_order):
        _process_reactants(
            reaction,
            reaction_idx,
            state_to_idx,
            row_indices,
            col_indices,
            coefficients,
        )
        _process_products(
            reaction,
            reaction_idx,
            state_to_idx,
            row_indices,
            col_indices,
            coefficients,
        )

    return row_indices, col_indices, coefficients


def _process_reactants(
    reaction: Reaction,
    reaction_idx: int,
    state_to_idx: dict[str, int],
    row_indices: list[int],
    col_indices: list[int],
    coefficients: list[float],
) -> None:
    """Process reactants and add their entries to the matrix arrays.

    Reactants are consumed in reactions, so their stoichiometric coefficients
    are negative.

    Args:
        reaction: The reaction being processed
        reaction_idx: Column index for this reaction
        state_to_idx: Mapping from state names to row indices
        row_indices: List to append row indices to
        col_indices: List to append column indices to
        coefficients: List to append stoichiometric coefficients to
    """
    for reactant in reaction.reactants:
        if reactant.state in state_to_idx:
            row_indices.append(state_to_idx[reactant.state])
            col_indices.append(reaction_idx)
            coefficients.append(-reactant.stoichiometry)


def _process_products(
    reaction: Reaction,
    reaction_idx: int,
    state_to_idx: dict[str, int],
    row_indices: list[int],
    col_indices: list[int],
    coefficients: list[float],
) -> None:
    """Process products and add their entries to the matrix arrays.

    Products are produced in reactions, so their stoichiometric coefficients
    are positive.

    Args:
        reaction: The reaction being processed
        reaction_idx: Column index for this reaction
        state_to_idx: Mapping from state names to row indices
        row_indices: List to append row indices to
        col_indices: List to append column indices to
        coefficients: List to append stoichiometric coefficients to
    """
    for product in reaction.products:
        if product.state in state_to_idx:
            row_indices.append(state_to_idx[product.state])
            col_indices.append(reaction_idx)
            coefficients.append(product.stoichiometry)


def _build_jax_matrix(
    row_indices: list[int],
    col_indices: list[int],
    coefficients: list[float],
    n_state: int,
    n_reactions: int,
) -> jax.Array:
    """Build the final JAX stoichiometric matrix using scatter operations.

    Constructs a dense matrix from sparse coordinate format using JAX's
    efficient scatter operations.

    Args:
        row_indices: Row indices for non-zero entries
        col_indices: Column indices for non-zero entries
        coefficients: Stoichiometric coefficients for non-zero entries
        n_state: Number of state (matrix height)
        n_reactions: Number of reactions (matrix width)

    Returns:
        Dense JAX array of shape (n_state, n_reactions)
    """
    # Convert to JAX arrays with proper dtypes
    row_indices_jax = jnp.array(row_indices, dtype=jnp.int32)
    col_indices_jax = jnp.array(col_indices, dtype=jnp.int32)
    coefficients_jax = jnp.array(coefficients, dtype=jnp.float32)

    # Use JAX's scatter operations for efficient matrix construction
    stoich_matrix = jnp.zeros((n_state, n_reactions))
    return stoich_matrix.at[row_indices_jax, col_indices_jax].add(coefficients_jax)
