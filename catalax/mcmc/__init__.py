from .mcmc import run_mcmc
from .plotting import plot_corner, plot_posterior, plot_trace
from . import priors

import arviz as az

# Set plotting style
az.style.use("arviz-doc")
