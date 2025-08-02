"""
Initial condition estimation pre-model classes for MCMC inference.

This module provides classes that implement the PreModel protocol to handle
uncertain initial conditions in Bayesian inference.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Callable, Any
import numpyro
import numpyro.distributions as dist
from jax import Array

if TYPE_CHECKING:
    from catalax.model.model import Model


class InitialConditionEstimator:
    """
    Pre-model class for estimating true initial conditions from uncertain measurements.

    This class implements the PreModel protocol and follows the same pattern as the
    pre_model_initial_condition_noise function in test.py, but allows users to
    configure the distributions used for both the measurement uncertainty (sigma)
    and the true initial conditions.

    Parameters:
        sigma_dist: Distribution for the measurement uncertainty. Default: HalfNormal(10.0)
        initial_dist_fn: Optional function that takes (y0s, sigma) and returns a distribution
                        for the true initial conditions. Default: Independent Normal
        sigma_param_name: Name for the sigma parameter in MCMC trace
        initial_param_name: Name for the estimated initial conditions in MCMC trace

    Examples:
        Basic usage (matches test.py behavior):
        >>> estimator = InitialConditionEstimator()
        >>> result = hmc.run(model, dataset, pre_model=estimator)

        Custom sigma distribution:
        >>> estimator = InitialConditionEstimator(
        ...     sigma_dist=dist.HalfNormal(5.0)
        ... )

        Custom parameter names:
        >>> estimator = InitialConditionEstimator(
        ...     sigma_param_name="measurement_error",
        ...     initial_param_name="true_concentrations"
        ... )
    """

    def __init__(
        self,
        sigma_dist: Optional[dist.Distribution] = None,
        initial_dist_fn: Optional[Callable[[Any, Any], dist.Distribution]] = None,
        sigma_param_name: str = "estimated_y0s_sigma",
        initial_param_name: str = "estimated_y0s",
    ):
        """
        Initialize the initial condition estimator.

        Args:
            sigma_dist: Distribution for measurement uncertainty (must be positive-valued)
            initial_dist_fn: Function taking (y0s, sigma) returning distribution for true initials
            sigma_param_name: Parameter name for sigma in MCMC trace
            initial_param_name: Parameter name for estimated initials in MCMC trace
        """
        self.sigma_dist = (
            sigma_dist if sigma_dist is not None else dist.HalfNormal(10.0)
        )
        self.initial_dist_fn = initial_dist_fn or self._default_initial_dist
        self.sigma_param_name = sigma_param_name
        self.initial_param_name = initial_param_name

    def _default_initial_dist(self, y0s: Any, sigma: Any) -> dist.Distribution:
        """Default distribution: Independent Normal centered on observed values."""
        return dist.Independent(dist.Normal(y0s, sigma), y0s.ndim)

    def __call__(
        self,
        *,
        model: "Model",
        y0s: Array,
        constants: Array,
        times: Array,
        data: Optional[Array],
        theta: Array,
    ) -> Tuple[Any, Array, Array, Optional[Array], Array]:
        """
        Estimate true initial conditions using Independent distribution.

        This follows the exact same pattern as the function in test.py but allows
        for configurable distributions.
        """
        # Sample uncertainty in initial conditions
        estimated_y0s_sigma = numpyro.sample(
            self.sigma_param_name,
            self.sigma_dist,
        )

        # Sample corrected initial conditions
        estimated_y0s = numpyro.sample(
            self.initial_param_name,
            self.initial_dist_fn(y0s, estimated_y0s_sigma),
        )

        return estimated_y0s, constants, times, data, theta
