import jax.numpy as jnp

from catalax.neural.universalode import UniversalODE


def l2_reg_gate(model: UniversalODE, alpha: float, **kwargs):
    """Performs L2 regularization on UniversalODE gate matrix.

    This penalty function applies L2 regularization specifically to the gate matrix
    of a UniversalODE model. The gate matrix controls the gating mechanism that blends
    mechanistic and neural network terms. Regularizing these parameters helps prevent
    overfitting and promotes stable hybrid modeling.

    The penalty is computed as the sum of squared values of the gate matrix,
    scaled by the regularization strength coefficient. This encourages smaller
    parameter values, leading to more conservative gating behavior.

    Args:
        model: UniversalODE model containing gate_matrix parameter
        alpha: Regularization strength coefficient, higher values increase regularization pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L2 norm of gate matrix

    Raises:
        AssertionError: If the provided model is not an instance of UniversalODE
    """
    assert isinstance(model, UniversalODE), "Model must be a UniversalODE"
    return alpha * jnp.sum(model.gate_matrix**2)


def l2_reg_alpha(model: UniversalODE, alpha: float, **kwargs):
    """Performs L2 regularization on UniversalODE alpha residual parameters.

    This penalty function applies L2 regularization specifically to the alpha
    residual parameters of a UniversalODE model. The alpha residual parameters
    scale the neural network contribution. Regularizing these parameters helps
    prevent overfitting and promotes stable hybrid modeling.

    The penalty is computed as the sum of squared values of alpha residual
    parameters, scaled by the regularization strength coefficient. This encourages
    smaller parameter values, leading to more conservative neural network contributions.

    Args:
        model: UniversalODE model containing alpha_residual parameters
        alpha: Regularization strength coefficient, higher values increase regularization pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L2 norm of alpha residual parameters

    Raises:
        AssertionError: If the provided model is not an instance of UniversalODE
    """
    assert isinstance(model, UniversalODE), "Model must be a UniversalODE"
    return alpha * jnp.sum(model.alpha_residual**2)


def l1_reg_gate(model: UniversalODE, alpha: float, **kwargs):
    """Performs L1 regularization on UniversalODE gate matrix.

    This penalty function applies L1 regularization specifically to the gate matrix
    of a UniversalODE model. The gate matrix controls the gating mechanism that blends
    mechanistic and neural network terms. L1 regularization promotes sparsity by
    penalizing the absolute values of parameters, potentially setting some to exactly
    zero, which can lead to more interpretable hybrid models.

    The penalty is computed as the sum of absolute values of the gate matrix,
    scaled by the regularization strength coefficient. This encourages sparsity
    and can help select which components of the gating mechanism are most important.

    Args:
        model: UniversalODE model containing gate_matrix parameter
        alpha: Regularization strength coefficient, higher values increase sparsity pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L1 norm of gate matrix

    Raises:
        AssertionError: If the provided model is not an instance of UniversalODE
    """
    assert isinstance(model, UniversalODE), "Model must be a UniversalODE"
    return alpha * jnp.sum(jnp.abs(model.gate_matrix))


def l1_reg_alpha(model: UniversalODE, alpha: float, **kwargs):
    """Performs L1 regularization on UniversalODE alpha residual parameters.

    This penalty function applies L1 regularization specifically to the alpha
    residual parameters of a UniversalODE model. The alpha residual parameters
    scale the neural network contribution. L1 regularization promotes sparsity
    by penalizing the absolute values of parameters, potentially setting some
    to exactly zero, which can lead to more interpretable hybrid models.

    The penalty is computed as the sum of absolute values of alpha residual
    parameters, scaled by the regularization strength coefficient. This encourages
    sparsity and can help select which neural network contributions are most important.

    Args:
        model: UniversalODE model containing alpha_residual parameters
        alpha: Regularization strength coefficient, higher values increase sparsity pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L1 norm of alpha residual parameters

    Raises:
        AssertionError: If the provided model is not an instance of UniversalODE
    """
    assert isinstance(model, UniversalODE), "Model must be a UniversalODE"
    return alpha * jnp.sum(jnp.abs(model.alpha_residual))
