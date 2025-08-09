from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from catalax.mcmc.protocols import PreModel, pre_model, PreModelContext


def estimate_initials(
    y0_sigma_dist: dist.Distribution = dist.HalfNormal(10.0),
) -> PreModel:
    """
    Create a pre-model to estimate true initial conditions from uncertain measurements.

    This function returns a pre-model that handles measurement uncertainty in initial
    conditions by:
    1. Sampling measurement uncertainty (sigma) from a specified distribution
    2. Sampling corrected initial conditions from a distribution centered on the
       observed values with the estimated uncertainty

    Args:
        y0_sigma_dist: Distribution for sampling measurement uncertainty.
                      Defaults to HalfNormal(10.0).

    Returns:
        PreModel: A pre-model function that can be used with MCMC inference.

    Example:
        >>> import catalax.mcmc as cmc
        >>> from catalax.mcmc.models import estimate_initials
        >>>
        >>> # Create pre-model with custom uncertainty distribution
        >>> pre_model = estimate_initials(
        ...     y0_sigma_dist=dist.HalfNormal(5.0),
        ...     y0_dist=dist.Normal
        ... )
        >>>
        >>> # Use with HMC
        >>> result = hmc.run(
        ...     model,
        ...     dataset,
        ...     pre_model=pre_model,
        ...     yerrs=1.0
        ... )
    """

    @pre_model
    def _estimate_initials(ctx: PreModelContext):
        estimated_y0s_sigma = numpyro.sample(
            "estimated_y0s_sigma",
            y0_sigma_dist,  # type: ignore
        )

        with numpyro.plate("y0s", ctx.y0s.shape[0]):
            ctx.y0s = numpyro.sample(  # type: ignore
                "estimated_y0s",
                dist.Normal(ctx.y0s, estimated_y0s_sigma),  # type: ignore
            )

    return _estimate_initials
