import math
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from catalax.model.model import Model


def visualize(
    model: Model,
    data: jax.Array,
    times: jax.Array,
    initial_conditions: Optional[List[Dict[str, float]]] = None,
    n_cols: int = 2,
    colors: Optional[Dict] = None,
    use_names=True,
    time_unit: str = "s",
    concentration_unit: str = "mM",
    heading_fontsize=12,
    figsize=(3, 6),
    resolution: int = 1000,
    max_steps=64**4,
    save: Optional[str] = None,
):
    if colors is None:
        color_iter = iter(mcolors.TABLEAU_COLORS)  # type: ignore
        colors = {species: next(color_iter) for species in model._get_species_order()}

    # Check if this mode has a fit already
    has_fit = model.parameters and all(
        parameter.value is not None for parameter in model.parameters.values()
    )

    # Get the number of rows
    n_rows = math.ceil(data.shape[0] / 2)

    # Simulate the model
    if has_fit and initial_conditions:
        times_, states = model.simulate(
            initial_conditions=initial_conditions,  # type: ignore
            dt0=0.01,
            in_axes=(0, None, None),
            saveat=jnp.linspace(0, times.max(), resolution),
            max_steps=max_steps,
        )
    elif has_fit and not initial_conditions:
        print(f"Cannot plot fit without 'initial_conditions'.")
    else:
        times_, states = None, None

    # Create sub-plot
    f, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_rows * figsize[0], n_cols * figsize[1]),
        constrained_layout=True,
    )

    _check_color_species_consistency(colors, model)

    # Initialize column/row iter
    row, col = 0, 0

    for dataset in range(data.shape[0]):
        for species, color in colors.items():
            index = model._get_species_order().index(species)

            if use_names:
                species = model.species[species].name

            ax[row, col].plot(
                times[dataset],
                data[dataset, :, index],
                "o",
                label=f"{species} (Data)",
                c=color,
            )

            if times_ is not None and states is not None:
                ax[row, col].plot(
                    times_[dataset],
                    states[dataset, :, index],
                    label=f"{species} (Fit)",
                    c=color,
                )

        if initial_conditions:
            # Add initial conditions as heading
            init_as_string = "  ".join(
                f"{model.species[species].name}: ${value:.2f}$"
                for species, value in initial_conditions[dataset].items()
            )
            ax[row, col].set_title(init_as_string, fontsize=heading_fontsize)

        # Add Grids and other stuff
        ax[row, col].set_xlabel(f"Time [${time_unit}$]", fontsize=12)
        ax[row, col].set_ylabel(f"Concentration [${concentration_unit}$]", fontsize=12)
        ax[row, col].grid(alpha=0.7, linestyle=":")

        col += 1

        if col == n_cols:
            col = 0
            row += 1

    ax[0, n_cols - 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        if not save.endswith(".png"):
            save = f"{save}.png"

        plt.savefig(save, dpi=300, format="png")

    return f


def _check_color_species_consistency(
    color_mapping: Dict[str, str],
    model: Model,
) -> None:
    """Checks whether the given color mapping is consistent"""

    # Check whether all species in the color mapping are in the model and vice versa
    diff1 = set(color_mapping.keys()).difference(model._get_species_order())
    diff2 = set(model.species.keys()).difference(color_mapping.keys())

    if len(diff1) > 0 or len(diff2) > 0:
        print(f"Species {list(diff1)} are in the color mapping but not in the model")
        print(f"Species {list(diff2)} are in the model but not in the color mapping")

        raise ValueError(
            "Color mapping and model species are not consistent. Please check the color mapping."
        )
