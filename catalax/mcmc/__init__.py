from .mcmc import run_mcmc, MCMCConfig, BayesianModel, HMC, HMCResults
from .loo import (
    loo,
    compare,
    consistency_check,
    build_loo_idata,
    reconstruct_log_likelihood,
    loo_pointwise,
    LooPointwise,
)
from .plotting import (
    plot_corner,
    plot_posterior,
    plot_trace,
    plot_forest,
    summary,
    plot_loo_influence,
    plot_loo_heatmap,
)
from .protocols import (
    PreModel,
    PostModel,
    PreModelContext,
    PostModelContext,
    pre_model,
    post_model,
)
from . import priors
from . import models
from . import error
from .error import ErrorModel

import arviz as az  # noqa: F401

# Set plotting style
az.style.use("arviz-doc")  # type: ignore

__all__ = [
    "run_mcmc",
    "plot_corner",
    "plot_posterior",
    "plot_trace",
    "plot_forest",
    "summary",
    "priors",
    "models",
    "error",
    "ErrorModel",
    "MCMCConfig",
    "BayesianModel",
    "HMC",
    "HMCResults",
    "PreModel",
    "PostModel",
    "PreModelContext",
    "PostModelContext",
    "pre_model",
    "post_model",
    "loo",
    "compare",
    "consistency_check",
    "build_loo_idata",
    "reconstruct_log_likelihood",
    "loo_pointwise",
    "LooPointwise",
    "plot_loo_influence",
    "plot_loo_heatmap",
]
