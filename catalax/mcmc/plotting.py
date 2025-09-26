from typing import Dict, Optional, Tuple, Union

import arviz as az
import corner
import jax
import matplotlib.pyplot as plt
import pandas as pd
import xarray
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC

# Backend type hints
BackendType = Union[str, None]


def plot_corner(
    mcmc: MCMC,
    quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
):
    """Plots the correlation between the parameters.

    Args:
        mcmc (MCMC): The MCMC object to plot.
        quantiles (Tuple[float, float, float]): Quantiles to display in the corner plot.
            Default is (0.16, 0.5, 0.84).
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The corner plot figure.
    """
    data = az.from_numpyro(mcmc)

    # Create figure with specified size if provided
    fig = None
    if figsize is not None:
        fig = plt.figure(figsize=figsize)

    fig = corner.corner(
        data,
        fig=fig,
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


def plot_posterior(
    mcmc: MCMC,
    model,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
    **kwargs,
):
    """Plots the posterior distribution of the parameters.

    Args:
        mcmc (MCMC): The MCMC object containing posterior samples.
        model: The model object with parameter information.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.
        **kwargs: Additional keyword arguments to pass to arviz.plot_posterior.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The posterior plot figure.
    """
    inf_data = az.from_numpyro(mcmc)

    # Set figure size if provided and using matplotlib
    if figsize is not None and (backend is None or backend == "matplotlib"):
        plt.figure(figsize=figsize)

    # Add backend to kwargs if specified
    if backend is not None:
        kwargs["backend"] = backend

    plot_result = az.plot_posterior(inf_data, **kwargs)

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
        return plt.gcf()


def plot_credibility_interval(
    mcmc: MCMC,
    model,
    initial_condition: Dict[str, float],
    time: jax.Array,
    dt0: float = 0.1,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
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
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend. Note: This function currently only
            supports matplotlib backend regardless of the backend parameter value.

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

    fig, ax = plt.subplots(figsize=figsize)
    for i, state in enumerate(model.get_state_order()):
        ax.plot(time, states[:, i], label=f"{state} simulation")
        ax.fill_between(
            time,
            hpdi_mu[0, :, i],  # type: ignore
            hpdi_mu[1, :, i],  # type: ignore
            alpha=0.3,
            interpolate=True,
            label=f"{state} CI",
        )

    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    return fig


def plot_trace(
    mcmc: MCMC,
    model,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
    **kwargs,
):
    """Plots the MCMC trace for each parameter.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        model: The model object with parameter information.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.
        **kwargs: Additional keyword arguments to pass to arviz.plot_trace.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The trace plot figure.
    """
    inf_data = az.from_numpyro(mcmc)

    # Set figure size if provided and using matplotlib
    if figsize is not None and (backend is None or backend == "matplotlib"):
        plt.figure(figsize=figsize)

    # Add backend to kwargs if specified
    if backend is not None:
        kwargs["backend"] = backend

    plot_result = az.plot_trace(inf_data, **kwargs)

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
        return plt.gcf()


def plot_forest(
    mcmc: MCMC,
    model,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
    **kwargs,
):
    """Plots a forest plot of parameter distributions.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        model: The model object with parameter information.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.
        **kwargs: Additional keyword arguments to pass to arviz.plot_forest.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The forest plot figure.
    """
    inf_data = az.from_numpyro(mcmc)

    # Set figure size if provided and using matplotlib
    if figsize is not None and (backend is None or backend == "matplotlib"):
        plt.figure(figsize=figsize)

    # Add backend to kwargs if specified
    if backend is not None:
        kwargs["backend"] = backend

    plot_result = az.plot_forest(
        inf_data, var_names=model.get_parameter_order(), **kwargs
    )

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
        return plt.gcf()


def plot_mcse(
    mcmc: MCMC,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
    **kwargs,
):
    """Plots the Monte Carlo standard error for each parameter.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.
        **kwargs: Additional keyword arguments to pass to arviz.plot_mcse.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The MCSE plot figure.
    """
    inf_data = az.from_numpyro(mcmc)

    # Set figure size if provided and using matplotlib
    if figsize is not None and (backend is None or backend == "matplotlib"):
        plt.figure(figsize=figsize)

    # Add backend to kwargs if specified
    if backend is not None:
        kwargs["backend"] = backend

    plot_result = az.plot_mcse(inf_data, rug=True, extra_methods=True, **kwargs)

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
        return plt.gcf()


def plot_ess(
    mcmc: MCMC,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
):
    """Plots the effective sample size for each parameter.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib' or 'bokeh').
            If None, uses default backend.

    Returns:
        matplotlib.figure.Figure or bokeh plot: The ESS plot figure.
    """
    inf_data = az.from_numpyro(mcmc)

    # Set figure size if provided and using matplotlib
    if figsize is not None and (backend is None or backend == "matplotlib"):
        plt.figure(figsize=figsize)

    extra_kwargs = {"color": "lightsteelblue"}

    # Prepare kwargs for plot_ess
    ess_kwargs = {
        "kind": "evolution",
        "color": "royalblue",
        "extra_kwargs": extra_kwargs,
    }

    # Add backend if specified
    if backend is not None:
        ess_kwargs["backend"] = backend

    plot_result = az.plot_ess(inf_data, **ess_kwargs)

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
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
