from .mcmc import run_mcmc
from .plotting import plot_corner, plot_posterior, plot_trace, plot_forest, summary
from . import priors
from .mcmc import MCMCConfig

import arviz as az

__all__ = [
    "run_mcmc",
    "plot_corner",
    "plot_posterior",
    "plot_trace",
    "plot_forest",
    "summary",
    "priors",
    "MCMCConfig",
]

# Set plotting style
az.style.use("arviz-doc")
