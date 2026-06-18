from typing import Dict, Optional, Tuple, Union

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC

# Backend type hints
BackendType = Union[str, None]


def plot_corner(
    mcmc: MCMC,
    hdi_prob: float = 0.94,
    figsize: Optional[Tuple[float, float]] = None,
    backend: BackendType = None,
):
    """Plots a pair plot of the posterior parameters.

    Builds an ArviZ ``plot_pair`` with KDE marginals on the diagonal showing the
    posterior median and the HDI interval, scatter joint plots on the lower
    triangle, and divergent transitions highlighted on top of the scatter
    plots.

    Args:
        mcmc (MCMC): The MCMC object to plot.
        hdi_prob (float): Probability mass of the HDI marked on each marginal.
            Default is 0.94.
        figsize (Tuple[float, float], optional): Figure size as (width, height) in inches.
            If None, uses default size.
        backend (str, optional): Plotting backend ('matplotlib', 'bokeh', or
            'plotly'). If None, uses arviz default backend.

    Returns:
        matplotlib.figure.Figure or backend-specific figure: The pair plot figure.
    """
    inf_data = az.from_numpyro(mcmc)
    samples = mcmc.get_samples()
    var_names = [name for name in samples.keys() if name != "sigma"]

    plot_kwargs = {}
    if backend is not None:
        plot_kwargs["backend"] = backend

    with az.rc_context(
        {
            "stats.ci_kind": "hdi",
            "stats.ci_prob": hdi_prob,
            "stats.point_estimate": "median",
        }
    ):
        plot_matrix = az.plot_pair(
            inf_data,
            var_names=var_names,
            marginal=True,
            marginal_kind="kde",
            triangle="lower",
            visuals={
                "scatter": False,
                "divergence": True,
                "credible_interval": True,
                "point_estimate": True,
                "point_estimate_text": True,
            },
            **plot_kwargs,
        )

    if backend == "bokeh" or backend == "plotly":
        return plot_matrix

    flat = {k: np.asarray(samples[k]).reshape(-1) for k in var_names}
    for r in range(len(var_names)):
        for c in range(r):
            ax = plot_matrix.iget_target(r, c)
            ax.scatter(
                flat[var_names[c]],
                flat[var_names[r]],
                s=2,
                alpha=0.25,
                color="0.55",
                linewidths=0,
                zorder=1,
            )
            sns.kdeplot(
                x=flat[var_names[c]],
                y=flat[var_names[r]],
                ax=ax,
                levels=6,
                fill=True,
                cmap="Blues",
                thresh=0.05,
                zorder=2,
            )
            sns.kdeplot(
                x=flat[var_names[c]],
                y=flat[var_names[r]],
                ax=ax,
                levels=6,
                fill=False,
                cmap="Blues",
                thresh=0.05,
                linewidths=0.6,
                zorder=3,
            )

    fig = plt.gcf()
    if figsize is not None:
        fig.set_size_inches(figsize)
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
        **kwargs: Additional keyword arguments to pass to arviz.plot_dist.

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

    plot_result = az.plot_dist(inf_data, **kwargs)

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

    plot_kwargs = {}
    if backend is not None:
        plot_kwargs["backend"] = backend

    plot_result = az.plot_ess_evolution(inf_data, **plot_kwargs)

    # Return appropriate object based on backend
    if backend == "bokeh":
        return plot_result
    else:
        return plt.gcf()


def summary(mcmc: MCMC, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Generates a summary of the MCMC results.

    Args:
        mcmc (MCMC): The MCMC object containing samples.
        hdi_prob (float): The probability mass to include in the highest density interval.
            Default is 0.95.

    Returns:
        pd.DataFrame: Summary statistics of the posterior distributions.
    """
    inf_data = az.from_numpyro(mcmc)
    return az.summary(inf_data, ci_prob=hdi_prob, ci_kind="hdi")
