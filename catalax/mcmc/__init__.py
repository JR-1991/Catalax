import arviz as az

from . import priors
from .mcmc import run_mcmc
from .plotting import plot_corner, plot_forest, plot_posterior, plot_trace

# Set plotting style
az.style.use("arviz-doc")

__all__ = [
    "run_mcmc",
    "plot_corner",
    "plot_posterior",
    "plot_trace",
    "plot_forest",
    "priors",
]
