import jax
import jax.numpy as jnp

from catalax.neural.rateflow import RateFlowODE


def penalize_null_space(model: RateFlowODE, alpha: float = 0.1, **kwargs) -> jax.Array:
    """Penalize the null space of the stoichiometric matrix.

    This penalty function encourages the stoichiometric matrix to have a null space of zero.
    """
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)
    if model.mass_constraint is not None:
        # Check shape compatibility for matrix multiplication
        if model.mass_constraint.shape[1] != stoich_matrix.shape[0]:
            raise ValueError(
                f"Incompatible shapes for matrix multiplication: "
                f"mass_constraint.shape={model.mass_constraint.shape}, "
                f"stoich_matrix.shape={stoich_matrix.shape}. "
                "Expected mass_constraint.shape[1] == stoich_matrix.shape[0]."
            )
        return alpha * jnp.mean(model.mass_constraint @ stoich_matrix)
    else:
        return jnp.array(0.0)


def penalize_density(model: RateFlowODE, alpha: float = 0.1, **kwargs) -> jax.Array:
    """Penalize dense stoichiometric matrices by encouraging sparsity.

    This penalty function encourages the stoichiometric matrix to have fewer
    non-zero entries by penalizing values greater than 0.5. This promotes
    sparse reaction networks which are often more interpretable and realistic.

    Args:
        stoich_matrix: The stoichiometric matrix to penalize, shape (n_state, n_reactions)
        alpha: Penalty strength coefficient, higher values increase sparsity pressure

    Returns:
        Scalar penalty value proportional to the density of the matrix
    """
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)
    greater_than_half = jnp.where(jnp.abs(stoich_matrix) > 0.5, 1, 0)
    return alpha * jnp.mean(greater_than_half)


def penalize_non_bipolar(model: RateFlowODE, alpha: float = 0.1, **kwargs) -> jax.Array:
    """Penalize stoichiometric matrices that violate mass balance constraints.

    This penalty function encourages reactions to be mass-balanced by penalizing
    columns (reactions) where the sum of stoichiometric coefficients is non-zero.
    In a bipolar representation, balanced reactions should have equal numbers of
    reactants (-1) and products (+1), resulting in zero column sums.

    Args:
        stoich_matrix: The stoichiometric matrix to penalize, shape (n_state, n_reactions)
        alpha: Penalty strength coefficient, higher values enforce stricter mass balance

    Returns:
        Scalar penalty value proportional to mass balance violations
    """
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)

    # The difference in each column should be roughly zero
    diffs = jnp.abs(jnp.sum(stoich_matrix, axis=0))
    return alpha * jnp.mean(diffs)


def penalize_non_conservative(
    model: RateFlowODE,
    alpha: float = 0.1,
    **kwargs,
) -> jax.Array:
    """Penalize non-conservative reactions.

    Args:
        stoich_matrix: The stoichiometric matrix to penalize
        alpha: Penalty strength coefficient
    """
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)

    # The sum of each row should be zero
    diffs = jnp.abs(jnp.sum(stoich_matrix, axis=1))
    return alpha * jnp.mean(diffs)


def penalize_duplicate_reactions(
    model: RateFlowODE,
    alpha: float = 1.0,
    **kwargs,
) -> jax.Array:
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)

    G = stoich_matrix @ stoich_matrix.T  # shape (n_reactions, n_reactions)

    # We want only i<j off‑diagonal terms; use a mask.
    n = G.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    off_diag = jnp.where(mask, G, 0.0)
    return alpha * 1e-1 * jnp.mean(off_diag**2)


def penalize_non_integer(model: RateFlowODE, alpha: float = 0.1, **kwargs) -> jax.Array:
    """Penalize non-integer values in the stoichiometric matrix.

    Args:
        stoich_matrix: The stoichiometric matrix to penalize
        alpha: Penalty strength coefficient

    Returns:
        Scalar penalty value proportional to deviation from integers
    """
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)

    # Calculate distance from nearest integer
    rounded = jnp.round(stoich_matrix)
    deviations = jnp.abs(stoich_matrix - rounded)
    return alpha * jnp.mean(deviations)


def l1_stoich_penalty(model: RateFlowODE, alpha: float = 0.1, **kwargs) -> jax.Array:
    """Penalize the L1 norm of the stoichiometric matrix."""
    assert isinstance(model, RateFlowODE), "Model must be a RateFlowODE"

    stoich_matrix = _normalize_matrix(model.stoich_matrix)

    return alpha * jnp.linalg.norm(stoich_matrix, ord=1)


def _normalize_matrix(stoich_matrix: jax.Array) -> jax.Array:
    """Normalize the stoichiometric matrix to have a gcd of 1."""
    norm = jnp.linalg.norm(jnp.abs(stoich_matrix), axis=0)
    return stoich_matrix / norm
