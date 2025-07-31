"""Loss functions for model fitting and evaluation.

This module provides loss functions commonly used in optimization and model
evaluation tasks, particularly for scientific computing and experimental
data fitting.
"""

import jax.numpy as jnp
from jax import Array


def l1_loss(y_true: Array, y_pred: Array) -> Array:
    """Calculate the L1 (Manhattan) loss between true and predicted values.

    The L1 loss, also known as Mean Absolute Error (MAE) or Manhattan distance,
    computes the absolute difference between true and predicted values. This loss
    function is robust to outliers compared to L2 loss and is commonly used in
    regression tasks where outlier robustness is desired.

    Args:
        y_true: True/observed values as a JAX array
        y_pred: Predicted values as a JAX array, must have the same shape as y_true

    Returns:
        JAX array containing the element-wise absolute differences between y_true
        and y_pred, with the same shape as the input arrays

    Note:
        This function returns element-wise losses rather than the mean loss.
        To get the mean L1 loss, use jnp.mean(l1_loss(y_true, y_pred)).

    Example:
        >>> import jax.numpy as jnp
        >>> y_true = jnp.array([1.0, 2.0, 3.0])
        >>> y_pred = jnp.array([1.1, 1.8, 3.2])
        >>> loss = l1_loss(y_true, y_pred)
        >>> print(loss)  # [0.1, 0.2, 0.2]
        >>> mean_loss = jnp.mean(loss)
        >>> print(mean_loss)  # 0.16666667
    """
    return jnp.abs(y_true - y_pred)


def mean_absolute_error(y_true: Array, y_pred: Array) -> Array:
    """Calculate the mean absolute error between true and predicted values.

    The mean absolute error computes the average absolute difference between
    true and predicted values. This loss function is robust to outliers compared
    to mean squared error and is commonly used in regression tasks where outlier
    robustness is desired.
    """
    return jnp.mean(jnp.abs(y_true - y_pred))
