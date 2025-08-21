from __future__ import annotations

from jax import Array
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Callable


class FitMetrics(BaseModel):
    """Statistical metrics for evaluating predictor performance on experimental datasets.

    This class provides a standardized way to compute and store model evaluation metrics
    following the lmfit convention. The metrics help assess both model fit quality and
    complexity, enabling model comparison and selection.

    The class supports custom objective functions for computing chi-square, defaulting to
    L2 loss but allowing any objective function that takes (y_true, y_pred) and returns
    loss values. For normalized objectives (like mean loss), the metrics automatically
    scale appropriately to maintain statistical interpretation.

    The class computes:
    - Chi-square: Objective function applied to residuals
    - Reduced chi-square: Chi-square normalized by degrees of freedom
    - AIC (Akaike Information Criterion): Balances model fit and complexity
    - BIC (Bayesian Information Criterion): Similar to AIC but with stronger penalty for complexity

    Attributes:
        chisqr: Chi-square statistic - lower values indicate better fit
        redchi: Reduced chi-square statistic - lower values indicate better fit
        aic: Akaike Information Criterion - lower values indicate better models
        bic: Bayesian Information Criterion - lower values indicate better models
    """

    model_config: ConfigDict = ConfigDict(
        extra="forbid",
    )

    n_parameters: int = Field(
        description="Number of parameters in the model",
    )
    n_points: int = Field(
        description="Number of data points",
    )
    chisqr: float = Field(
        description="Chi-square: χ² = Σᵢᴺ[Residᵢ]²",
    )
    redchi: float = Field(
        description="Reduced chi-square: χᵥ² = χ²/max(1, N - Nᵥₐᵣᵧₛ)",
    )
    weighted_mape: float = Field(
        description="Weighted mean absolute percentage error (L1-based): |Σ abs(y_true - y_pred)| / Σ abs(y_true)",
    )
    aic: float = Field(
        description="Akaike Information Criterion statistic: -2*loglikelihood + 2*Nᵥₐᵣᵧₛ",
    )
    bic: float = Field(
        description="Bayesian Information Criterion statistic: -2*loglikelihood + ln(N)*Nᵥₐᵣᵧₛ",
    )
    r2: float = Field(
        description="R² statistic: 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_true.mean())²)",
    )

    @staticmethod
    def _weighted_mape(y_true: Array, y_pred: Array) -> Array:
        """Calculate relative error with robust handling of small values.

        Args:
            y_true: True/observed values
            y_pred: Predicted values

        Returns:
            The calculated relative error value
        """
        residuals = jnp.abs(y_true - y_pred).sum()
        weights = jnp.abs(y_true).sum()
        return residuals / weights

    @staticmethod
    def _chisqr(residual: Array, n_points: int) -> Array:
        """Calculate chi-square statistic using a specified objective function.

        Chi-square is computed using the provided objective function. For normalized
        objectives (like mean loss), the result is scaled by the number of data points
        to maintain consistency with the traditional chi-square interpretation.

        Args:
            y_true: True/observed values
            y_pred: Predicted values
            objective_fn: Function that computes the objective/loss

        Returns:
            The calculated chi-square value

        Note:
            Lower chi-square values indicate better model fit to the data.
        """

        if residual.size > 1:
            return jnp.maximum(jnp.sum(jnp.pow(residual, 2)), 1.0e-250 * n_points)
        else:
            return jnp.maximum(residual, 1.0e-250 * n_points)

    @staticmethod
    def _redchi(chisqr: float, n_points: int, n_parameters: int) -> float:
        """Calculate reduced chi-square statistic.

        Reduced chi-square normalizes the chi-square by the degrees of freedom.
        It is calculated as: χᵥ² = χ²/(N - Nᵥₐᵣᵧₛ)

        Args:
            chisqr: Chi-square statistic
            n_points: Number of data points
            n_parameters: Number of variable parameters

        Returns:
            The calculated reduced chi-square value

        Note:
            Values close to 1.0 indicate good model fit. Values much larger than 1.0
            suggest poor fit or underestimated uncertainties.
        """
        nfree = n_points - n_parameters
        return chisqr / max(1, nfree)

    @staticmethod
    def _aic(chisqr: Array, n_points: int, n_parameters: int) -> Array:
        """Calculate Akaike Information Criterion (AIC).

        AIC estimates the relative quality of statistical models by balancing
        goodness of fit against model complexity. Following lmfit convention,
        it is calculated as: AIC = -2*loglikelihood + 2*Nᵥₐᵣᵧₛ
        where -2*loglikelihood = N * ln(χ²/N)

        Args:
            chisqr: Chi-square statistic
            n_points: Number of data points
            n_parameters: Number of variable parameters

        Returns:
            The calculated AIC value

        Note:
            Lower AIC values indicate better models. AIC tends to favor more
            complex models compared to BIC.
        """
        # This is -2*loglikelihood following lmfit convention
        neg2_log_likel = n_points * jnp.log(chisqr / n_points)
        return neg2_log_likel + 2 * n_parameters

    @staticmethod
    def _bic(chisqr: Array, n_points: int, n_parameters: int) -> Array:
        """Calculate Bayesian Information Criterion (BIC).

        BIC is similar to AIC but includes a stronger penalty for model complexity
        that depends on the sample size. Following lmfit convention,
        it is calculated as: BIC = -2*loglikelihood + ln(N)*Nᵥₐᵣᵧₛ
        where -2*loglikelihood = N * ln(χ²/N)

        Args:
            chisqr: Chi-square statistic
            n_points: Number of data points
            n_parameters: Number of variable parameters

        Returns:
            The calculated BIC value

        Note:
            Lower BIC values indicate better models. BIC tends to favor simpler
            models compared to AIC, especially with larger datasets.
        """
        # This is -2*loglikelihood following lmfit convention
        neg2_log_likel = n_points * jnp.log(chisqr / n_points)
        return neg2_log_likel + jnp.log(n_points) * n_parameters

    @staticmethod
    def _r2(y_true: Array, y_pred: Array) -> Array:
        """Calculate R² statistic.

        R² is the coefficient of determination, which measures the proportion of variance in the dependent variable that is explained by the independent variable.
        """
        return 1 - (
            jnp.sum(jnp.pow(y_true - y_pred, 2))
            / jnp.sum(jnp.pow(y_true - y_true.mean(), 2))
        )

    @classmethod
    def from_predictions(
        cls,
        y_true: Array,
        y_pred: Array,
        n_parameters: int,
        objective_fun: Callable[[Array, Array], Any] = lambda y_true, y_pred: jnp.abs(
            y_true - y_pred
        ),  # type: ignore
    ) -> FitMetrics:
        """Create FitMetrics from predicted and observed values.

        This factory method computes all statistical metrics from the provided
        predicted and observed values along with model characteristics, following
        the lmfit convention for statistical analysis.

        Args:
            y_true: True/observed values as a JAX array
            y_pred: Predicted values as a JAX array, must have the same shape as y_true
            n_parameters: The number of variable parameters in the model being evaluated
            objective_fun: Function to compute the objective/loss. Defaults to optax.l2_loss.
                         For normalized objectives (like mean loss), the result will be
                         scaled appropriately for chi-square calculation.

        Returns:
            A FitMetrics instance with computed chi-square, reduced chi-square, AIC, and BIC values

        Raises:
            ValueError: If y_true and y_pred have different shapes

        Example:
            >>> import jax.numpy as jnp
            >>> # Evaluate a model with 3 parameters on experimental data
            >>> y_observed = jnp.array([1.2, 2.1, 2.9, 4.2, 5.1])
            >>> y_predicted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> metrics = FitMetrics.from_predictions(
            ...     y_true=y_observed,
            ...     y_pred=y_predicted,
            ...     n_parameters=3
            ... )
            >>> print(f"Model fit: χ² = {metrics.chisqr:.3f}, AIC = {metrics.aic:.2f}")
            >>>
            >>> # Using a custom objective function
            >>> import optax
            >>> metrics_custom = FitMetrics.from_predictions(
            ...     y_true=y_observed,
            ...     y_pred=y_predicted,
            ...     n_parameters=3,
            ...     objective_fn=optax.huber_loss
            ... )
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}"
            )

        # Calculate chi-square using the objective function
        loss = objective_fun(y_true, y_pred)
        n_points = loss.size
        chisqr = cls._chisqr(loss, n_points)

        if isinstance(loss, float):
            loss = jnp.array(loss)

        return cls(
            n_parameters=n_parameters,
            n_points=n_points,
            chisqr=float(chisqr),
            redchi=cls._redchi(float(chisqr), n_points, n_parameters),
            weighted_mape=float(cls._weighted_mape(y_true, y_pred)),
            aic=float(cls._aic(chisqr, n_points, n_parameters)),
            bic=float(cls._bic(chisqr, n_points, n_parameters)),
            r2=float(cls._r2(y_true, y_pred)),
        )


def l1_loss(y_true: Array, y_pred: Array) -> Array:
    return jnp.abs(y_true - y_pred)
