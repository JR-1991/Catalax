from __future__ import annotations

from jax import Array
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field


class FitMetrics(BaseModel):
    """Statistical metrics for evaluating predictor performance on experimental datasets.

    This class provides a standardized way to compute and store model evaluation metrics
    following the lmfit convention. The metrics help assess both model fit quality and
    complexity, enabling model comparison and selection.

    The class computes:
    - Chi-square: Sum of squared residuals
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
        populate_by_name=True,
        extra="forbid",
    )

    n_parameters: int = Field(
        alias="Parameters",
        description="Number of parameters in the model",
    )
    n_points: int = Field(
        alias="Data Points",
        description="Number of data points",
    )
    chisqr: float = Field(
        alias="Chi-Square",
        description="Chi-square: χ² = Σᵢᴺ[Residᵢ]²",
    )
    redchi: float = Field(
        alias="Reduced Chi-Square",
        description="Reduced chi-square: χᵥ² = χ²/max(1, N - Nᵥₐᵣᵧₛ)",
    )
    aic: float = Field(
        alias="AIC",
        description="Akaike Information Criterion statistic: -2*loglikelihood + 2*Nᵥₐᵣᵧₛ",
    )
    bic: float = Field(
        alias="BIC",
        description="Bayesian Information Criterion statistic: -2*loglikelihood + ln(N)*Nᵥₐᵣᵧₛ",
    )

    @staticmethod
    def _chisqr(y_true: Array, y_pred: Array) -> float:
        """Calculate chi-square statistic.

        Chi-square is the sum of squared residuals between predicted and observed values.
        It is calculated as: χ² = Σᵢᴺ[Residᵢ]²

        Args:
            y_true: True/observed values
            y_pred: Predicted values

        Returns:
            The calculated chi-square value

        Note:
            Lower chi-square values indicate better model fit to the data.
        """
        residuals = y_true - y_pred
        chisqr = float(jnp.sum(residuals**2))
        # Apply minimum threshold as in lmfit
        ndata = y_true.size
        return max(chisqr, 1.0e-250 * ndata)

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
    def _aic(chisqr: float, n_points: int, n_parameters: int) -> float:
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
        neg2_log_likel = n_points * float(jnp.log(chisqr / n_points))
        return neg2_log_likel + 2 * n_parameters

    @staticmethod
    def _bic(chisqr: float, n_points: int, n_parameters: int) -> float:
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
        neg2_log_likel = n_points * float(jnp.log(chisqr / n_points))
        return neg2_log_likel + float(jnp.log(n_points)) * n_parameters

    @classmethod
    def from_predictions(
        cls,
        y_true: Array,
        y_pred: Array,
        n_parameters: int,
    ) -> FitMetrics:
        """Create FitMetrics from predicted and observed values.

        This factory method computes all statistical metrics from the provided
        predicted and observed values along with model characteristics, following
        the lmfit convention for statistical analysis.

        Args:
            y_true: True/observed values as a JAX array
            y_pred: Predicted values as a JAX array, must have the same shape as y_true
            n_parameters: The number of variable parameters in the model being evaluated

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
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}"
            )

        # Calculate chi-square from residuals
        chisqr = cls._chisqr(y_true, y_pred)
        n_points = y_true.size

        return cls(
            n_parameters=n_parameters,  # type: ignore
            n_points=n_points,  # type: ignore
            chisqr=chisqr,  # type: ignore
            redchi=cls._redchi(chisqr, n_points, n_parameters),  # type: ignore
            aic=cls._aic(chisqr, n_points, n_parameters),  # type: ignore
            bic=cls._bic(chisqr, n_points, n_parameters),  # type: ignore
        )
