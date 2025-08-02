from typing import Dict, Tuple, Union

import arviz as az
import corner
import jax
import matplotlib.pyplot as plt
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC
import pandas as pd
import xarray


def plot_corner(
    mcmc: MCMC,
    quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
):
    """Plots the correlation between the parameters.

    Args:
        mcmc (MCMC): The MCMC object to plot.
        quantiles (Tuple[float, float, float]): Quantiles to display in the corner plot.
            Default is (0.16, 0.5, 0.84).

    Returns:
        matplotlib.figure.Figure: The corner plot figure.
    """
    data = az.from_numpyro(mcmc)
    fig = corner.corner(
        data,
        plot_contours=False,
        quantiles=list(quantiles),
        bins=20,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        divergences=True,
        use_math_text=False,
        var_names=[var for var in mcmc.get_samples().keys() if var != "sigma"],
    )

    fig.tight_layout()
    return fig


def plot_posterior(mcmc: MCMC, model, **kwargs):
    """Plots the posterior distribution of the parameters.

    Args:
        mcmc (MCMC): The MCMC object containing posterior samples.
        model: The model object with parameter information.
        **kwargs: Additional keyword arguments to pass to arviz.plot_posterior.

    Returns:
        matplotlib.figure.Figure: The posterior plot figure.
    """
    inf_data = az.from_numpyro(mcmc)
    az.plot_posterior(inf_data, **kwargs)
    return plt.gcf()


def plot_credibility_interval(
    mcmc: MCMC,
    model,
    initial_condition: Dict[str, float],
    time: jax.Array,
    dt0: float = 0.1,
):
    """Plots the credibility interval for model simulations.

    Simulates the model using posterior parameter samples and plots the mean
    trajectory with credibility intervals.

    Args:
        mcmc (MCMC): The MCMC object containing posterior samples.
        model: The model to simulate.
        initial_condition (Dict[str, float]): Initial conditions for the simulation.
        time (jax.Array): Time points for the simulation.
        dt0 (float): Time step for the simulation. Default is 0.1.

    Returns:
        matplotlib.figure.Figure: The credibility interval plot.
    """
    samples = mcmc.get_samples()

    # Evaluate all parameters from the distribution to gather hpdi
    _, post_states = model.simulate(
        initial_conditions=initial_condition,
        dt0=dt0,
        saveat=time,
        parameters=samples["theta"],
        in_axes=(None, 0, None),
    )

    # Simulate system at the mean of posterior parameters
    _, states = model.simulate(
        initial_conditions=initial_condition,
        dt0=dt0,
        saveat=time,
        parameters=samples["theta"].mean(0),
        in_axes=None,
    )

    # Get HPDI (Highest Posterior Density Interval)
    hpdi_mu = hpdi(post_states, 0.9)

    fig, ax = plt.subplots()
    for i, species in enumerate(model.get_species_order()):
        ax.plot(time, states[:, i], label=f"{species} simulation")
        ax.fill_between(
            time,
            hpdi_mu[0, :, i],  # type: ignore
            hpdi_mu[1, :, i],  # type: ignore
            alpha=0.3,
            interpolate=True,
            label=f"{species} CI",
        )

    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    return fig


def plot_trace(mcmc: MCMC, model, **kwargs):
    """Plots the MCMC trace for each parameter.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        model: The model object with parameter information.
        **kwargs: Additional keyword arguments to pass to arviz.plot_trace.

    Returns:
        matplotlib.figure.Figure: The trace plot figure.
    """
    inf_data = az.from_numpyro(mcmc)
    az.plot_trace(inf_data, **kwargs)
    return plt.gcf()


def plot_forest(mcmc: MCMC, model, **kwargs):
    """Plots a forest plot of parameter distributions.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        model: The model object with parameter information.
        **kwargs: Additional keyword arguments to pass to arviz.plot_forest.

    Returns:
        matplotlib.figure.Figure: The forest plot figure.
    """
    inf_data = az.from_numpyro(mcmc)
    az.plot_forest(inf_data, var_names=model.get_parameter_order(), **kwargs)
    return plt.gcf()


def summary(mcmc: MCMC, hdi_prob: float = 0.95) -> Union[pd.DataFrame, xarray.Dataset]:
    """Generates a summary of the MCMC results.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        hdi_prob (float): The probability mass to include in the highest density interval.
            Default is 0.95.

    Returns:
        Union[pd.DataFrame, az.Dataset]: Summary statistics of the posterior distributions.
    """
    inf_data = az.from_numpyro(mcmc)
    return az.summary(inf_data, hdi_prob=hdi_prob)
