import jax.numpy as jnp

from catalax.neural.neuralbase import NeuralBase


def l2_regularisation(model: NeuralBase, alpha: float):
    """Performs L2 regularization to control weights of an MLP

    L2 regularization encourages small weights by penalizing the sum of squared
    values of all weights and biases. This can help prevent overfitting and create
    more stable models. The penalty is added to the loss function during training,
    effectively shrinking parameters toward zero.

    Args:
        model: Neural model to apply L2 regularization to
        alpha: Regularization strength coefficient, higher values increase sparsity pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L2 norm of all model parameters
    """
    return alpha * sum(jnp.sum(layer**2) for layer in model.get_weights_and_biases())


def l1_regularisation(model: NeuralBase, alpha: float, **kwargs):
    """Performs L1 regularization to control weights of an MLP

    L1 regularization encourages sparsity in model parameters by penalizing the sum
    of absolute values of all weights and biases. This can help prevent overfitting
    and create more interpretable models with fewer active parameters. The penalty
    is added to the loss function during training, effectively shrinking parameters
    toward zero and potentially setting some exactly to zero.

    Args:
        model: Neural model to apply L1 regularization to
        alpha: Regularization strength coefficient, higher values increase sparsity pressure
        **kwargs: Additional keyword arguments (unused but maintained for consistency)

    Returns:
        Scalar penalty value proportional to the L1 norm of all model parameters
    """
    return alpha * sum(
        jnp.sum(jnp.abs(layer)) for layer in model.get_weights_and_biases()
    )
