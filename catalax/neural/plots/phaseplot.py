from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from catalax.dataset.dataset import Dataset
from catalax.model.model import Model

if TYPE_CHECKING:
    from catalax.neural.rateflow import RateFlowODE


def phase_plot(
    rateflow_ode: RateFlowODE,
    dataset: Dataset,
    model: Model,
    rate_indices: List[int] | None = None,
    state_identifiers: List[str] | None = None,
    state_pairs: List[Tuple[str | int, str | int]] | None = None,
    representative_time: float = 0.0,
    grid_resolution: int = 30,
    figsize_per_subplot: Tuple[int, int] = (5, 4),
    save_path: str | None = None,
    range_extension: float = 0.2,
    show: bool = True,
) -> Figure:
    """
    Create static visualizations showing how dynamics depend on concentration inputs.

    Args:
        rateflow_ode: RateFlowODE instance to use for calculations
        dataset: Dataset containing the measurements
        model: Model to use for the analysis
        rate_indices: List of rate indices to plot. If None, plots all rates.
        state_identifiers: List of state to include (by name, symbol, or index).
                        If None, uses all state.
        state_pairs: List of tuples specifying which state pairs to compare.
                        Each tuple should contain two state identifiers (name, symbol, or index).
                        If None, compares all possible pairs from state_identifiers.
        representative_time: Time point to use for the analysis
        grid_resolution: Number of points in each dimension of the concentration grid
        figsize_per_subplot: Size of each subplot (width, height)
        save_path: Path to save the figure
        range_extension: Factor to extend concentration ranges beyond data bounds (0.0-1.0).
                        Default 0.2 extends ranges by 20% above and below data bounds.
                        Lower bound is clamped to 0.0.
    """

    # Validate range_extension parameter
    if not (0.0 <= range_extension <= 1.0):
        raise ValueError(
            f"range_extension must be between 0.0 and 1.0, got {range_extension}"
        )

    # Prepare data and resolve indices
    plot_data = _prepare_plot_data(
        rateflow_ode, dataset, model, state_identifiers, range_extension
    )
    rate_indices = _resolve_rate_indices(
        rate_indices,
        plot_data["inner_func"],
        plot_data["mean_concentrations"],
        representative_time,
    )
    state_pairs_resolved = _create_state_pairs(
        plot_data["resolved_state_indices"],
        state_pairs,
        plot_data["state_order"],
        plot_data["state_names"],
    )

    if not state_pairs_resolved:
        raise ValueError("Need at least 2 state to create pairs")

    # Create figure and axes
    fig, axes_array = _create_figure_and_axes(
        len(rate_indices), len(state_pairs_resolved), figsize_per_subplot
    )

    # Generate plots
    for rate_idx_pos, rate_idx in enumerate(rate_indices):
        _add_rate_label(
            fig, axes_array, rate_idx_pos, rate_idx, len(state_pairs_resolved)
        )

        for col_idx, (state1_idx, state2_idx) in enumerate(state_pairs_resolved):
            # Create heatmap
            heatmap_ax = _get_axis(axes_array, rate_idx_pos, col_idx)
            _create_single_heatmap(
                heatmap_ax,
                plot_data,
                state1_idx,
                state2_idx,
                rate_idx,
                grid_resolution,
                representative_time,
                dataset,
            )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def _prepare_plot_data(
    rateflow_ode: RateFlowODE,
    dataset: Dataset,
    model: Model,
    state_identifiers: List[str] | None,
    range_extension: float,
):
    """Prepare all the data needed for plotting."""
    inner_func = jax.vmap(rateflow_ode.func, in_axes=(0, 0, None))
    state_order = rateflow_ode.state_order
    state_names = _get_state_names(model, state_order)
    resolved_state_indices = _resolve_state_indices(
        state_identifiers, state_order, state_names
    )
    min_concentrations, max_concentrations = _get_state_data_ranges(
        dataset, state_order, range_extension
    )
    mean_concentrations = [
        (min_c + max_c) / 2
        for min_c, max_c in zip(min_concentrations, max_concentrations)
    ]

    return {
        "inner_func": inner_func,
        "state_order": state_order,
        "state_names": state_names,
        "resolved_state_indices": resolved_state_indices,
        "min_concentrations": min_concentrations,
        "max_concentrations": max_concentrations,
        "mean_concentrations": mean_concentrations,
    }


def _resolve_rate_indices(
    rate_indices: List[int] | None,
    inner_func,
    mean_concentrations,
    representative_time: float,
):
    """Resolve and validate rate indices."""

    # Determine total number of rates
    test_state = jnp.array(mean_concentrations)
    test_rates = inner_func(
        jnp.array([representative_time]), test_state.reshape(1, -1), ()
    )[0]
    n_total_rates = test_rates.shape[0]

    if rate_indices is None:
        return list(range(n_total_rates))

    # Validate rate indices
    for idx in rate_indices:
        if not (0 <= idx < n_total_rates):
            raise ValueError(f"Rate index {idx} out of range (0-{n_total_rates - 1})")

    return rate_indices


def _create_figure_and_axes(
    n_rows: int, n_cols: int, figsize_per_subplot: Tuple[int, int]
):
    """Create matplotlib figure and properly handle axes array."""

    figsize = (
        figsize_per_subplot[0] * n_cols + 2.5,
        figsize_per_subplot[1] * n_rows,
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
    )
    fig.suptitle("Rate Magnitude Heatmaps", fontsize=16)

    # Ensure axes is always a 2D array for consistent access
    if n_rows == 1 and n_cols == 1:
        axes_array = np.array([[axes]])
    elif n_rows == 1:
        axes_array = np.array([axes])
    elif n_cols == 1:
        axes_array = np.array([[ax] for ax in axes])
    else:
        axes_array = axes

    return fig, axes_array


def _get_axis(axes_array, row_idx: int, col_idx: int):
    """Get the correct axis from the axes array."""
    if axes_array.ndim == 2:
        return axes_array[row_idx, col_idx]
    else:
        # Handle edge cases for 1D arrays
        return axes_array[row_idx] if axes_array.ndim == 1 else axes_array


def _add_rate_label(fig, axes_array, row_idx: int, rate_idx: int, n_cols: int):
    """Add rotated rate label on the left side of each row."""
    # Get the first axis in the row to calculate position
    first_ax = _get_axis(axes_array, row_idx, 0)

    # Calculate the vertical center of the row
    row_center_y = first_ax.get_position().y0 + first_ax.get_position().height / 2

    # Add rotated text on the left side
    fig.text(
        -0.02,
        row_center_y,
        f"Reaction {rate_idx}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=90,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )


def _create_single_heatmap(
    ax,
    plot_data: dict,
    state1_idx: int,
    state2_idx: int,
    rate_idx: int,
    grid_resolution: int,
    representative_time: float,
    dataset: Dataset,
):
    """Create a single heatmap subplot."""

    # Create concentration grids
    x_vals = jnp.linspace(
        plot_data["min_concentrations"][state1_idx],
        plot_data["max_concentrations"][state1_idx],
        grid_resolution,
    )
    y_vals = jnp.linspace(
        plot_data["min_concentrations"][state2_idx],
        plot_data["max_concentrations"][state2_idx],
        grid_resolution,
    )
    X, Y = jnp.meshgrid(x_vals, y_vals)

    # Calculate rate magnitudes
    rate_magnitudes = _calculate_rate_magnitudes(
        plot_data["inner_func"],
        X,
        Y,
        state1_idx,
        state2_idx,
        plot_data["mean_concentrations"],
        representative_time,
        rate_idx,
    )

    # Create contour plot
    im = ax.contourf(X, Y, rate_magnitudes, levels=20, cmap="viridis")
    ax.contour(
        X, Y, rate_magnitudes, levels=10, colors="white", alpha=0.5, linewidths=0.5
    )

    # Find and mark min and max points with stars
    min_idx = jnp.unravel_index(jnp.argmin(rate_magnitudes), rate_magnitudes.shape)
    max_idx = jnp.unravel_index(jnp.argmax(rate_magnitudes), rate_magnitudes.shape)

    # Get the corresponding X, Y coordinates
    min_x, min_y = X[min_idx], Y[min_idx]
    max_x, max_y = X[max_idx], Y[max_idx]

    # Add star markers for min and max points
    ax.scatter(
        min_x,
        min_y,
        marker="*",
        s=200,
        c="cyan",
        edgecolors="black",
        linewidth=2,
        label="Min Rate",
        zorder=10,
    )
    ax.scatter(
        max_x,
        max_y,
        marker="*",
        s=200,
        c="yellow",
        edgecolors="black",
        linewidth=2,
        label="Max Rate",
        zorder=10,
    )

    # Add colorbar and labels
    plt.colorbar(im, ax=ax, label="Rate Magnitude")

    state1_name = plot_data["state_names"][state1_idx]
    state2_name = plot_data["state_names"][state2_idx]

    ax.set_xlabel(f"{state1_name} Concentration", fontsize=12)
    ax.set_ylabel(f"{state2_name} Concentration", fontsize=12)
    ax.set_title(f"{state1_name} vs {state2_name}", fontsize=12)

    # Add data points from actual measurements
    _add_data_points(
        ax,
        dataset,
        plot_data["state_order"],
        state1_idx,
        state2_idx,
        representative_time,
    )


def _get_state_data_ranges(
    dataset: Dataset,
    state_order,
    range_extension: float,
):
    """Get min and max concentrations for each state from the dataset with range extension."""
    min_concentrations = []
    max_concentrations = []

    for state in state_order:
        state_data = []
        for measurement in dataset.measurements:
            state_data.extend(measurement.data[state])

        data_min = min(state_data)
        data_max = max(state_data)
        data_range = data_max - data_min

        # Extend range by the specified factor
        extended_min = data_min - (data_range * range_extension)
        extended_max = data_max + (data_range * range_extension)

        # Don't go below 0 for minimum concentration
        extended_min = max(0.0, extended_min)

        min_concentrations.append(extended_min)
        max_concentrations.append(extended_max)

    return min_concentrations, max_concentrations


def _get_state_names(model: Model, state_order: List[str]):
    """Get state names from the model in the correct order."""
    return [state.name for state in model.states.values()]


def _resolve_state_indices(
    state_identifiers: List[str] | None,
    state_order: List[str],
    state_names: List[str],
):
    """Resolve state identifiers (names or indices) to actual indices."""
    if state_identifiers is None:
        return list(range(len(state_order)))

    resolved_indices = []
    for identifier in state_identifiers:
        if isinstance(identifier, int):
            if 0 <= identifier < len(state_order):
                resolved_indices.append(identifier)
            else:
                raise ValueError(f"Species index {identifier} out of range")
        elif isinstance(identifier, str):
            # Try to find by state name
            if identifier in state_names:
                resolved_indices.append(state_names.index(identifier))
            # Try to find by state symbol
            elif identifier in state_order:
                resolved_indices.append(state_order.index(identifier))
            else:
                raise ValueError(f"Species '{identifier}' not found")
        else:
            raise ValueError(f"Invalid state identifier: {identifier}")

    return resolved_indices


def _create_state_pairs(
    state_indices, custom_pairs=None, state_order=None, state_names=None
):
    """Create state pairs from given indices or custom pair specifications."""
    if custom_pairs is None:
        # Original behavior: create all combinations
        state_pairs = []
        for i in range(len(state_indices)):
            for j in range(i + 1, len(state_indices)):
                state_pairs.append((state_indices[i], state_indices[j]))
        return state_pairs

    # New behavior: process custom pairs
    resolved_pairs = []
    seen_pairs = set()

    for pair in custom_pairs:
        if len(pair) != 2:
            raise ValueError(
                f"Each state pair must contain exactly 2 elements, got {len(pair)}"
            )

        state1, state2 = pair

        # Resolve state identifiers to indices
        idx1 = _resolve_single_state_identifier(state1, state_order, state_names)
        idx2 = _resolve_single_state_identifier(state2, state_order, state_names)

        # Check for self-comparison
        if idx1 == idx2:
            raise ValueError(f"Cannot compare state with itself: {state1} vs {state2}")

        # Create normalized pair (smaller index first) to check for duplicates
        normalized_pair = (min(idx1, idx2), max(idx1, idx2))

        # Check for duplicates
        if normalized_pair in seen_pairs:
            continue  # Ignore duplicate

        seen_pairs.add(normalized_pair)
        resolved_pairs.append(normalized_pair)

    return resolved_pairs


def _resolve_single_state_identifier(identifier, state_order, state_names):
    """Resolve a single state identifier to its index."""
    if isinstance(identifier, int):
        if 0 <= identifier < len(state_order):
            return identifier
        else:
            raise ValueError(
                f"Species index {identifier} out of range (0-{len(state_order) - 1})"
            )
    elif isinstance(identifier, str):
        # Try to find by state name
        if identifier in state_names:
            return state_names.index(identifier)
        # Try to find by state symbol
        elif identifier in state_order:
            return state_order.index(identifier)
        else:
            raise ValueError(f"Species '{identifier}' not found")
    else:
        raise ValueError(f"Invalid state identifier: {identifier}")


def _calculate_rate_magnitudes(
    inner_func,
    X,
    Y,
    state1_idx,
    state2_idx,
    mean_concentrations,
    representative_time,
    rate_idx,
):
    """Calculate rate magnitudes for a given state pair and rate index using vectorized operations."""
    # Flatten the grids to create batch inputs
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)

    # Create base state array for all points
    mean_concentrations_array = jnp.array(mean_concentrations)

    # Create batch of states by broadcasting and updating specific state
    states_batch = jnp.tile(mean_concentrations_array, (n_points, 1))
    states_batch = states_batch.at[:, state1_idx].set(X_flat)
    states_batch = states_batch.at[:, state2_idx].set(Y_flat)

    # Create time array for all points
    times_batch = jnp.full(n_points, representative_time)

    # Compute all rates at once using the already vmapped inner_func
    all_rates = inner_func(times_batch, states_batch, ())

    # Extract the specific rate index and take absolute value
    rate_magnitudes_flat = jnp.abs(all_rates[:, rate_idx])

    # Reshape back to grid shape
    rate_magnitudes = rate_magnitudes_flat.reshape(X.shape)

    return rate_magnitudes


def _add_data_points(
    ax,
    dataset,
    state_order,
    state1_idx,
    state2_idx,
    representative_time,
):
    """Add scatter points from actual measurements for reference."""
    for measurement in dataset.measurements:
        # Get concentrations at the representative time (or closest)
        time_idx = jnp.argmin(
            jnp.abs(jnp.array(measurement.time) - representative_time)
        )

        conc1 = measurement.data[state_order[state1_idx]][time_idx]
        conc2 = measurement.data[state_order[state2_idx]][time_idx]

        ax.scatter(
            conc1, conc2, c="red", s=30, alpha=0.8, edgecolors="white", linewidth=1
        )
