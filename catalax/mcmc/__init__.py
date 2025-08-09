from .mcmc import run_mcmc, MCMCConfig, BayesianModel, HMC, HMCResults
from .plotting import plot_corner, plot_posterior, plot_trace, plot_forest, summary
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
]
