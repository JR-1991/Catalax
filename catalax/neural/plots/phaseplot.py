from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from catalax.dataset.dataset import Dataset
from catalax.model.model import Model

if TYPE_CHECKING:
    from catalax.neural.rateflow import RateFlowODE


def phase_plot(
    rateflow_ode: RateFlowODE,
    dataset: Dataset,
    model: Model,
    rate_indices: List[int] | None = None,
    species_identifiers: List[str] | None = None,
    species_pairs: List[Tuple[str | int, str | int]] | None = None,
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
        species_identifiers: List of species to include (by name, symbol, or index).
                        If None, uses all species.
        species_pairs: List of tuples specifying which species pairs to compare.
                        Each tuple should contain two species identifiers (name, symbol, or index).
                        If None, compares all possible pairs from species_identifiers.
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
        rateflow_ode, dataset, model, species_identifiers, range_extension
    )
    rate_indices = _resolve_rate_indices(
        rate_indices,
        plot_data["inner_func"],
        plot_data["mean_concentrations"],
        representative_time,
    )
    species_pairs_resolved = _create_species_pairs(
        plot_data["resolved_species_indices"],
        species_pairs,
        plot_data["species_order"],
        plot_data["species_names"],
    )

    if not species_pairs_resolved:
        raise ValueError("Need at least 2 species to create pairs")

    # Create figure and axes - now with 2 rows per rate (heatmap + ratio plot)
    fig, axes_array = _create_figure_and_axes(
        len(rate_indices) * 2, len(species_pairs_resolved), figsize_per_subplot
    )

    # Generate plots
    for rate_idx_pos, rate_idx in enumerate(rate_indices):
        # Calculate row indices for heatmap and ratio plot
        heatmap_row = rate_idx_pos * 2
        ratio_row = rate_idx_pos * 2 + 1

        _add_rate_label(
            fig, axes_array, heatmap_row, rate_idx, len(species_pairs_resolved)
        )

        for col_idx, (species1_idx, species2_idx) in enumerate(species_pairs_resolved):
            # Create heatmap
            heatmap_ax = _get_axis(axes_array, heatmap_row, col_idx)
            _create_single_heatmap(
                heatmap_ax,
                plot_data,
                species1_idx,
                species2_idx,
                rate_idx,
                grid_resolution,
                representative_time,
                dataset,
            )

            # Create ratio plot
            ratio_ax = _get_axis(axes_array, ratio_row, col_idx)
            _create_ratio_plot(
                ratio_ax,
                plot_data,
                species1_idx,
                species2_idx,
                rate_idx,
                grid_resolution,
                representative_time,
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
    species_identifiers: List[str] | None,
    range_extension: float,
):
    """Prepare all the data needed for plotting."""
    inner_func = jax.vmap(rateflow_ode.func, in_axes=(0, 0, None))
    species_order = rateflow_ode.species_order
    species_names = _get_species_names(model, species_order)
    resolved_species_indices = _resolve_species_indices(
        species_identifiers, species_order, species_names
    )
    min_concentrations, max_concentrations = _get_species_data_ranges(
        dataset, species_order, range_extension
    )
    mean_concentrations = [
        (min_c + max_c) / 2
        for min_c, max_c in zip(min_concentrations, max_concentrations)
    ]

    return {
        "inner_func": inner_func,
        "species_order": species_order,
        "species_names": species_names,
        "resolved_species_indices": resolved_species_indices,
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

    # Adjust figure size to account for ratio plots (half height)
    # Each rate now takes 1.5x the height (1x for heatmap + 0.5x for ratio plot)
    adjusted_height = figsize_per_subplot[1] * (n_rows // 2) * 1.5
    figsize = (
        figsize_per_subplot[0] * n_cols + 2.5,
        adjusted_height,
    )

    # Create subplots with height ratios (2:1 for heatmap:ratio)
    height_ratios = []
    for i in range(n_rows // 2):
        height_ratios.extend([2, 1])  # heatmap height : ratio plot height

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle("Rate Magnitude Heatmaps with Ratio Analysis", fontsize=16)

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
    species1_idx: int,
    species2_idx: int,
    rate_idx: int,
    grid_resolution: int,
    representative_time: float,
    dataset: Dataset,
):
    """Create a single heatmap subplot."""

    # Create concentration grids
    x_vals = jnp.linspace(
        plot_data["min_concentrations"][species1_idx],
        plot_data["max_concentrations"][species1_idx],
        grid_resolution,
    )
    y_vals = jnp.linspace(
        plot_data["min_concentrations"][species2_idx],
        plot_data["max_concentrations"][species2_idx],
        grid_resolution,
    )
    X, Y = jnp.meshgrid(x_vals, y_vals)

    # Calculate rate magnitudes
    rate_magnitudes = _calculate_rate_magnitudes(
        plot_data["inner_func"],
        X,
        Y,
        species1_idx,
        species2_idx,
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

    species1_name = plot_data["species_names"][species1_idx]
    species2_name = plot_data["species_names"][species2_idx]

    ax.set_xlabel(f"{species1_name} Concentration", fontsize=12)
    ax.set_ylabel(f"{species2_name} Concentration", fontsize=12)
    ax.set_title(f"{species1_name} vs {species2_name}", fontsize=12)

    # Add data points from actual measurements
    _add_data_points(
        ax,
        dataset,
        plot_data["species_order"],
        species1_idx,
        species2_idx,
        representative_time,
    )


def _create_ratio_plot(
    ax,
    plot_data: dict,
    species1_idx: int,
    species2_idx: int,
    rate_idx: int,
    grid_resolution: int,
    representative_time: float,
):
    """Create a ratio plot showing rate magnitude vs species concentration ratio."""

    # Calculate ratio range
    min1, max1 = (
        plot_data["min_concentrations"][species1_idx],
        plot_data["max_concentrations"][species1_idx],
    )
    min2, max2 = (
        plot_data["min_concentrations"][species2_idx],
        plot_data["max_concentrations"][species2_idx],
    )

    # Create range of ratios, avoiding division by zero
    min_ratio = min1 / max2 if max2 > 0 else 0.01
    max_ratio = max1 / min2 if min2 > 0 else max1 / 0.01

    # Use log scale for better visualization if ratios span multiple orders of magnitude
    if max_ratio / min_ratio > 100:
        ratios = jnp.logspace(
            jnp.log10(max(min_ratio, 1e-6)), jnp.log10(max_ratio), grid_resolution
        )
    else:
        ratios = jnp.linspace(max(min_ratio, 1e-6), max_ratio, grid_resolution)

    # Calculate corresponding concentrations and rate magnitudes using vectorized operations
    # For each ratio, we'll vary the absolute concentration levels to find min/max rate magnitudes
    mean_conc1 = plot_data["mean_concentrations"][species1_idx]
    mean_conc2 = plot_data["mean_concentrations"][species2_idx]
    base_ref_total = (mean_conc1 + mean_conc2) / 2

    # Create a range of reference totals to explore different absolute concentration levels
    min_total = base_ref_total * 0.1  # 10% of base
    max_total = base_ref_total * 10.0  # 10x base
    n_total_samples = 20  # Number of different total concentrations to sample
    ref_totals = jnp.linspace(min_total, max_total, n_total_samples)

    # Create meshgrid of ratios and reference totals
    ratios_mesh, ref_totals_mesh = jnp.meshgrid(ratios, ref_totals, indexing="ij")
    ratios_flat = ratios_mesh.flatten()
    ref_totals_flat = ref_totals_mesh.flatten()

    # Vectorized calculation of concentrations for all combinations
    # Solve: conc1/conc2 = ratio and conc1 + conc2 = ref_total
    # This gives: conc2 = ref_total/(1 + ratio), conc1 = ratio * conc2
    conc2_batch = ref_totals_flat / (1 + ratios_flat)
    conc1_batch = ratios_flat * conc2_batch

    # Create batch of states for all combinations
    mean_concentrations_array = jnp.array(plot_data["mean_concentrations"])
    n_combinations = len(ratios_flat)
    states_batch = jnp.tile(mean_concentrations_array, (n_combinations, 1))
    states_batch = states_batch.at[:, species1_idx].set(conc1_batch)
    states_batch = states_batch.at[:, species2_idx].set(conc2_batch)

    # Create time array for all combinations
    times_batch = jnp.full(n_combinations, representative_time)

    # Compute all rates at once using the already vmapped inner_func
    all_rates = plot_data["inner_func"](times_batch, states_batch, ())

    # Extract the specific rate index and take absolute value
    rate_magnitudes_flat = jnp.abs(all_rates[:, rate_idx])

    # Reshape to (n_ratios, n_total_samples) to find min/max for each ratio
    rate_magnitudes_matrix = rate_magnitudes_flat.reshape(len(ratios), n_total_samples)

    # Find min and max rate magnitudes for each ratio
    min_rate_magnitudes = jnp.min(rate_magnitudes_matrix, axis=1)
    max_rate_magnitudes = jnp.max(rate_magnitudes_matrix, axis=1)
    mean_rate_magnitudes = jnp.mean(rate_magnitudes_matrix, axis=1)

    # Create the plot with filled area between min and max
    ax.fill_between(
        ratios,
        min_rate_magnitudes,
        max_rate_magnitudes,
        alpha=0.3,
        color="lightblue",
        label="Min-Max Range",
    )
    ax.plot(ratios, mean_rate_magnitudes, "b-", linewidth=2, alpha=0.8, label="Mean")
    ax.plot(ratios, min_rate_magnitudes, "g--", linewidth=1, alpha=0.6, label="Minimum")
    ax.plot(ratios, max_rate_magnitudes, "r--", linewidth=1, alpha=0.6, label="Maximum")

    # Add legend
    ax.legend(fontsize=8, loc="best")

    # Labels and formatting
    species1_name = plot_data["species_names"][species1_idx]
    species2_name = plot_data["species_names"][species2_idx]

    ax.set_xlabel(f"{species1_name}/{species2_name} Ratio", fontsize=10)
    ax.set_ylabel("Rate Magnitude", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if ratios span multiple orders of magnitude
    if max_ratio / min_ratio > 100:
        ax.set_xscale("log")


def _add_ratio_data_points(
    ax,
    dataset: Dataset,
    plot_data: dict,
    species1_idx: int,
    species2_idx: int,
    rate_idx: int,
    representative_time: float,
):
    """Add scatter points from actual measurements for the ratio plot."""

    for measurement in dataset.measurements:
        # Get concentrations at the representative time (or closest)
        time_idx = jnp.argmin(
            jnp.abs(jnp.array(measurement.time) - representative_time)
        )

        conc1 = measurement.data[plot_data["species_order"][species1_idx]][time_idx]
        conc2 = measurement.data[plot_data["species_order"][species2_idx]][time_idx]

        # Calculate ratio, avoiding division by zero
        if conc2 > 0:
            ratio = conc1 / conc2

            # Calculate rate magnitude for this measurement
            state = jnp.array(plot_data["mean_concentrations"])
            state = state.at[species1_idx].set(conc1)
            state = state.at[species2_idx].set(conc2)

            rates = plot_data["inner_func"](
                jnp.array([representative_time]), state.reshape(1, -1), ()
            )[0]
            rate_magnitude = jnp.abs(rates[rate_idx])

            ax.scatter(
                ratio,
                rate_magnitude,
                c="red",
                s=20,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )


def _get_species_data_ranges(
    dataset: Dataset,
    species_order,
    range_extension: float,
):
    """Get min and max concentrations for each species from the dataset with range extension."""
    min_concentrations = []
    max_concentrations = []

    for species in species_order:
        species_data = []
        for measurement in dataset.measurements:
            species_data.extend(measurement.data[species])

        data_min = min(species_data)
        data_max = max(species_data)
        data_range = data_max - data_min

        # Extend range by the specified factor
        extended_min = data_min - (data_range * range_extension)
        extended_max = data_max + (data_range * range_extension)

        # Don't go below 0 for minimum concentration
        extended_min = max(0.0, extended_min)

        min_concentrations.append(extended_min)
        max_concentrations.append(extended_max)

    return min_concentrations, max_concentrations


def _get_species_names(model, species_order):
    """Get species names from the model in the correct order."""
    return [species.name for species in model.species.values()]


def _resolve_species_indices(species_identifiers, species_order, species_names):
    """Resolve species identifiers (names or indices) to actual indices."""
    if species_identifiers is None:
        return list(range(len(species_order)))

    resolved_indices = []
    for identifier in species_identifiers:
        if isinstance(identifier, int):
            if 0 <= identifier < len(species_order):
                resolved_indices.append(identifier)
            else:
                raise ValueError(f"Species index {identifier} out of range")
        elif isinstance(identifier, str):
            # Try to find by species name
            if identifier in species_names:
                resolved_indices.append(species_names.index(identifier))
            # Try to find by species symbol
            elif identifier in species_order:
                resolved_indices.append(species_order.index(identifier))
            else:
                raise ValueError(f"Species '{identifier}' not found")
        else:
            raise ValueError(f"Invalid species identifier: {identifier}")

    return resolved_indices


def _create_species_pairs(
    species_indices, custom_pairs=None, species_order=None, species_names=None
):
    """Create species pairs from given indices or custom pair specifications."""
    if custom_pairs is None:
        # Original behavior: create all combinations
        species_pairs = []
        for i in range(len(species_indices)):
            for j in range(i + 1, len(species_indices)):
                species_pairs.append((species_indices[i], species_indices[j]))
        return species_pairs

    # New behavior: process custom pairs
    resolved_pairs = []
    seen_pairs = set()

    for pair in custom_pairs:
        if len(pair) != 2:
            raise ValueError(
                f"Each species pair must contain exactly 2 elements, got {len(pair)}"
            )

        species1, species2 = pair

        # Resolve species identifiers to indices
        idx1 = _resolve_single_species_identifier(
            species1, species_order, species_names
        )
        idx2 = _resolve_single_species_identifier(
            species2, species_order, species_names
        )

        # Check for self-comparison
        if idx1 == idx2:
            raise ValueError(
                f"Cannot compare species with itself: {species1} vs {species2}"
            )

        # Create normalized pair (smaller index first) to check for duplicates
        normalized_pair = (min(idx1, idx2), max(idx1, idx2))

        # Check for duplicates
        if normalized_pair in seen_pairs:
            continue  # Ignore duplicate

        seen_pairs.add(normalized_pair)
        resolved_pairs.append(normalized_pair)

    return resolved_pairs


def _resolve_single_species_identifier(identifier, species_order, species_names):
    """Resolve a single species identifier to its index."""
    if isinstance(identifier, int):
        if 0 <= identifier < len(species_order):
            return identifier
        else:
            raise ValueError(
                f"Species index {identifier} out of range (0-{len(species_order) - 1})"
            )
    elif isinstance(identifier, str):
        # Try to find by species name
        if identifier in species_names:
            return species_names.index(identifier)
        # Try to find by species symbol
        elif identifier in species_order:
            return species_order.index(identifier)
        else:
            raise ValueError(f"Species '{identifier}' not found")
    else:
        raise ValueError(f"Invalid species identifier: {identifier}")


def _calculate_rate_magnitudes(
    inner_func,
    X,
    Y,
    species1_idx,
    species2_idx,
    mean_concentrations,
    representative_time,
    rate_idx,
):
    """Calculate rate magnitudes for a given species pair and rate index using vectorized operations."""
    # Flatten the grids to create batch inputs
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)

    # Create base state array for all points
    mean_concentrations_array = jnp.array(mean_concentrations)

    # Create batch of states by broadcasting and updating specific species
    states_batch = jnp.tile(mean_concentrations_array, (n_points, 1))
    states_batch = states_batch.at[:, species1_idx].set(X_flat)
    states_batch = states_batch.at[:, species2_idx].set(Y_flat)

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
    species_order,
    species1_idx,
    species2_idx,
    representative_time,
):
    """Add scatter points from actual measurements for reference."""
    for measurement in dataset.measurements:
        # Get concentrations at the representative time (or closest)
        time_idx = jnp.argmin(
            jnp.abs(jnp.array(measurement.time) - representative_time)
        )

        conc1 = measurement.data[species_order[species1_idx]][time_idx]
        conc2 = measurement.data[species_order[species2_idx]][time_idx]

        ax.scatter(
            conc1, conc2, c="red", s=30, alpha=0.8, edgecolors="white", linewidth=1
        )
