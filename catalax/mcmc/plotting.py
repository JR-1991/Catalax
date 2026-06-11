import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import arviz as az
import corner
import jax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC

if TYPE_CHECKING:
    from catalax.mcmc.loo import LooPointwise

# Backend type hints
BackendType = Union[str, None]

# Discrete Pareto-k bands (ArviZ-style): good / ok / bad / very bad.
_PARETO_K_BOUNDS = [0.0, 0.5, 0.7, 1.0]
_PARETO_K_COLORS = ["#8cb369", "#f4e285", "#f4a259", "#bc4b51"]


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


def _species_label(model, symbol: str) -> str:
    """Human-readable ``Name (symbol)`` label for a state, falling back to symbol."""
    try:
        name = model.states[symbol].name
    except (KeyError, AttributeError):
        name = None
    if name and str(name) != str(symbol):
        return f"{name} ({symbol})"
    return str(symbol)


def _select_measurements(
    pointwise: "LooPointwise", measurements: Optional[List[int]]
) -> List[int]:
    n_meas = pointwise.data.shape[0]
    if measurements is None:
        return list(range(n_meas))
    return list(measurements)


def _panel_title(pointwise: "LooPointwise", m: int) -> str:
    """Concise per-measurement title (initial conditions, else short id)."""
    label = pointwise.measurement_labels[m]
    if label:
        return label
    return str(pointwise.measurement_ids[m])[:8]


def _time_ticklabels(times) -> List[str]:
    """Compact tick labels for a 1-D time vector."""
    return [f"{float(t):.3g}" for t in times]


def plot_loo_influence(
    pointwise: "LooPointwise",
    model,
    influence: str = "pareto_k",
    k_threshold: float = 0.7,
    measurements: Optional[List[int]] = None,
    ncols: int = 2,
    figsize: Tuple[float, float] = (5.0, 3.5),
    min_marker: float = 15.0,
    max_marker: float = 450.0,
):
    """Plot the measured trajectories with each point sized by its LOO influence.

    The data are drawn as usual (one panel per measurement, one colour per
    species); on top, every observation is overlaid with a marker whose **area
    grows with its influence** so the points that dominate the fit are obvious at
    a glance. Points above ``k_threshold`` are ringed in black.

    Args:
        pointwise: Result of :func:`loo_pointwise`.
        model: The mechanistic model (for species display names).
        influence: ``"pareto_k"`` (default) sizes by Pareto-k; ``"elpd"`` sizes by
            the per-point penalty ``-elpd_i``.
        k_threshold: Pareto-k above which a point is ringed/flagged.
        measurements: Indices of measurements to plot (default: all).
        ncols: Number of subplot columns.
        figsize: Per-panel size; the figure scales with the grid.
        min_marker, max_marker: Marker-area range (points^2) for the influence
            scaling.

    Returns:
        matplotlib.figure.Figure
    """
    if influence == "pareto_k":
        weight = pointwise.pareto_k
        wlabel = "Pareto $k$"
    elif influence == "elpd":
        weight = -pointwise.elpd
        wlabel = r"$-$elpd$_{loo}$"
    else:
        raise ValueError(f"influence must be 'pareto_k' or 'elpd', got {influence!r}")

    data = pointwise.data
    times = pointwise.times
    species = pointwise.species
    n_obs = data.shape[-1]

    wpos = np.where(np.isfinite(weight), np.clip(weight, 0.0, None), np.nan)
    wmax = np.nanmax(wpos) if np.isfinite(wpos).any() else 1.0
    wmax = wmax if wmax > 0 else 1.0

    def marker_area(w):
        return min_marker + (max_marker - min_marker) * (w / wmax)

    meas_idx = _select_measurements(pointwise, measurements)
    ncols = min(ncols, len(meas_idx))
    nrows = math.ceil(len(meas_idx) / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize[0], nrows * figsize[1])
    )
    axs = np.atleast_1d(axs).flatten()

    colors = list(mcolors.TABLEAU_COLORS.values())

    for panel, m in enumerate(meas_idx):
        ax = axs[panel]
        for j in range(n_obs):
            color = colors[j % len(colors)]
            t = times[m]
            y = data[m, :, j]
            finite = np.isfinite(y)
            if not finite.any():
                continue
            label = f"${_species_label(model, species[j])}$"
            ax.plot(t[finite], y[finite], "-", color=color, alpha=0.35, lw=1.0)

            w = wpos[m, :, j]
            scored = finite & np.isfinite(w)
            sizes = marker_area(np.where(scored, w, 0.0))
            edgecolors = np.where(
                scored & (weight[m, :, j] > k_threshold), "black", "none"
            )
            ax.scatter(
                t[scored],
                y[scored],
                s=sizes[scored],
                facecolor=color,
                alpha=0.6,
                edgecolors=edgecolors[scored],
                linewidths=1.2,
                label=label,
                zorder=3,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(_panel_title(pointwise, m), fontsize="medium")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize="small")

    for ax in axs[len(meas_idx) :]:
        ax.set_visible(False)

    # Size legend: representative influence values -> marker areas.
    ref = [v for v in (0.5, k_threshold, 1.0, 2.0) if v <= max(wmax, 2.0)]
    handles = [
        plt.scatter([], [], s=marker_area(v), color="grey", alpha=0.6, label=f"{v:g}")
        for v in ref
    ]
    if handles:
        fig.legend(
            handles=handles,
            title=wlabel,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            labelspacing=1.4,
            frameon=True,
        )
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig


def plot_loo_heatmap(
    pointwise: "LooPointwise",
    model,
    metric: str = "elpd",
    measurements: Optional[List[int]] = None,
    k_threshold: float = 0.7,
    ncols: int = 1,
    figsize: Tuple[float, float] = (6.0, 2.6),
    cmap: Optional[str] = None,
):
    """Species x time heatmap of the LOO penalty (or Pareto-k), per measurement.

    Reproduces the diagnostic where each cell is a measured observation coloured
    by how much it costs the model (``-elpd_loo``, darker = larger penalty) or by
    its PSIS reliability (``pareto_k``). Cells with ``pareto_k > k_threshold`` are
    marked with a dot.

    Args:
        pointwise: Result of :func:`loo_pointwise`.
        model: The mechanistic model (for species display names).
        metric: ``"elpd"`` (default, plots ``-elpd_loo``) or ``"pareto_k"``.
        measurements: Indices of measurements to plot (default: all).
        k_threshold: Pareto-k above which a cell is dotted.
        ncols: Number of measurement panels per row.
        figsize: Per-panel size; the figure scales with the grid.
        cmap: Override the colormap (default ``"YlOrRd"`` for elpd; a discrete
            ArviZ-style scale for pareto_k).

    Returns:
        matplotlib.figure.Figure
    """
    elpd = pointwise.elpd
    karr = pointwise.pareto_k
    n_obs = elpd.shape[-1]

    if metric == "elpd":
        grid_all = -elpd
        used_cmap = cmap or "YlOrRd"
        norm = None
        finite = grid_all[np.isfinite(grid_all)]
        # Anchor at 0 (no penalty) when possible; stay valid for an all-good fit
        # where every penalty is negative (elpd_i > 0).
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        vmin = min(0.0, lo)
        vmax = max(vmin + 1e-9, hi)
        cbar_label = r"$-$elpd$_{loo}$"
        discrete = False
    elif metric == "pareto_k":
        grid_all = karr
        finite_k = karr[np.isfinite(karr)]  # excludes NaN *and* inf
        kmin = float(finite_k.min()) if finite_k.size else 0.0
        kmax = float(finite_k.max()) if finite_k.size else 1.0001
        bounds = [min(0.0, kmin), 0.5, 0.7, 1.0, max(1.0001, kmax)]
        listed = mcolors.ListedColormap(_PARETO_K_COLORS)
        norm = mcolors.BoundaryNorm(bounds, listed.N)
        used_cmap = listed
        vmin = vmax = None
        cbar_label = "Pareto $k$"
        discrete = True
    else:
        raise ValueError(f"metric must be 'elpd' or 'pareto_k', got {metric!r}")

    # Keep only species observed somewhere (drop all-NaN rows).
    keep = np.array(
        [np.isfinite(grid_all[:, :, j]).any() for j in range(n_obs)], dtype=bool
    )
    sp_idx = np.flatnonzero(keep)
    sp_labels = [_species_label(model, pointwise.species[j]) for j in sp_idx]

    meas_idx = _select_measurements(pointwise, measurements)
    ncols = min(ncols, len(meas_idx))
    nrows = math.ceil(len(meas_idx) / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        squeeze=False,
        constrained_layout=True,
    )
    axs = axs.flatten()

    im = None
    for panel, m in enumerate(meas_idx):
        ax = axs[panel]
        grid = grid_all[m][:, sp_idx].T  # (n_species, n_time)
        masked = np.ma.masked_invalid(grid)
        plot_cmap = plt.get_cmap(used_cmap) if isinstance(used_cmap, str) else used_cmap
        plot_cmap = plot_cmap.copy()
        plot_cmap.set_bad("white")
        im = ax.imshow(
            masked,
            aspect="auto",
            cmap=plot_cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )

        t = pointwise.times[m]
        # Thin tick labels if there are many timepoints, to avoid overlap.
        stride = max(1, len(t) // 10)
        ticks = list(range(0, len(t), stride))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [_time_ticklabels(t)[i] for i in ticks], rotation=45, ha="right"
        )
        ax.set_yticks(range(len(sp_idx)))
        ax.set_yticklabels(sp_labels)
        ax.set_xlabel("Time")
        ax.set_title(_panel_title(pointwise, m), fontsize="medium")

        kk = karr[m][:, sp_idx].T
        ys, xs = np.where(np.isfinite(kk) & (kk > k_threshold))
        if xs.size:
            ax.scatter(xs, ys, s=28, c="black", zorder=3)

    for ax in axs[len(meas_idx) :]:
        ax.set_visible(False)

    if im is not None:
        cbar_axes = axs[: len(meas_idx)].tolist()
        if discrete:
            cbar = fig.colorbar(
                im,
                ax=cbar_axes,
                boundaries=bounds,
                spacing="uniform",
                ticks=[0.5, 0.7, 1.0],  # internal band edges
                fraction=0.046,
                pad=0.04,
            )
        else:
            cbar = fig.colorbar(im, ax=cbar_axes, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    return fig
