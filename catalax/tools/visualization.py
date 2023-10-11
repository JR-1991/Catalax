import math
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpyro.infer import MCMC
from numpyro.diagnostics import hpdi

from catalax.model.model import Model
from catalax.neural.neuralbase import NeuralBase


def visualize(
    model: Model,
    data: jax.Array,
    times: jax.Array,
    neural_ode: Optional[NeuralBase] = None,
    initial_conditions: Optional[List[Dict[str, float]]] = None,
    mcmc: Optional[MCMC] = None,
    n_cols: int = 2,
    title: Optional[str] = None,
    colors: Optional[Dict] = None,
    use_names=True,
    time_unit: str = "s",
    concentration_unit: str = "mM",
    heading_fontsize=12,
    figsize: Tuple[int, int] = (4, 2),
    resolution: int = 1000,
    max_steps: int = 64**4,
    mass: float = 0.9,
    save: Optional[str] = None,
):
    # Reset simulation function due to varying vmaps
    model.reset()

    if colors is None:
        color_iter = iter(mcolors.TABLEAU_COLORS)  # type: ignore
        colors = {species: next(color_iter) for species in model._get_species_order()}

    has_fit = all(
        parameter.value is not None for parameter in model.parameters.values()
    )

    if not neural_ode and initial_conditions:
        # Check if this mode has a fit already
        assert model.parameters and all(
            parameter.value is not None for parameter in model.parameters.values()
        ), f"Model has an incomplete parameter value set. Cant simulate!"

        func = lambda: (
            model.simulate(
                initial_conditions=initial_conditions,  # type: ignore
                dt0=0.01,
                in_axes=(0, None, None),
                saveat=jnp.linspace(0, times.max(), resolution),
                max_steps=max_steps,
            )
        )
    elif neural_ode and initial_conditions:
        y0s = jnp.stack(
            [
                jnp.array(
                    [float(init[species]) for species in sorted(list(init.keys()))]
                )
                for init in initial_conditions
            ]
        )
        func = lambda: (
            neural_ode.predict(  # type: ignore
                y0s=y0s,
                times=jnp.linspace(0, times.max(), resolution),
            )
        )
    else:
        func = None

    # Get the number of rows
    n_rows = math.ceil(data.shape[0] / 2)

    # Simulate the model
    if func and initial_conditions:
        times_, states = func()
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

    if mcmc and initial_conditions and has_fit:
        # Get boundaries from mcmc
        lower_quantiles, upper_quantiles = _get_quantile_preds(
            model=model,
            mcmc=mcmc,
            initial_conditions=initial_conditions,
            time=times_,  # type: ignore
            mass=mass,
        )

        has_quantiles = True
    else:
        lower_quantiles = None
        upper_quantiles = None
        has_quantiles = False

    # Initialize column/row iter
    row, col = 0, 0
    remaining_axes = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    for dataset in range(data.shape[0]):
        # Remove from remaining axes
        remaining_axes.remove((row, col))

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

            if has_quantiles:
                ax[row, col].fill_between(
                    times_[dataset],
                    lower_quantiles[dataset, :, index],  # type: ignore
                    upper_quantiles[dataset, :, index],  # type: ignore
                    alpha=0.3,
                    color=color,
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

    legend = ax[0, n_cols - 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Clean up remaining axes
    for row, col in remaining_axes:
        ax[row, col].axis("off")

    if title:
        bbox_extra_artists = [
            f.suptitle(
                title,
                x=0.05,
                y=1.05,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=15,
            ),
            legend,
        ]
    else:
        bbox_extra_artists = [legend]

    if save:
        if not save.endswith(".png"):
            save = f"{save}.png"

        plt.savefig(
            save,
            dpi=300,
            format="png",
            bbox_inches="tight",
            bbox_extra_artists=bbox_extra_artists,
        )

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


def _get_quantile_preds(
    model: Model,
    mcmc: MCMC,
    initial_conditions: List[Dict[str, float]],
    time: jax.Array,
    mass: float = 0.9,
) -> Tuple[jax.Array, jax.Array]:
    samples = mcmc.get_samples()
    hdpi_range = [
        hpdi(mcmc.get_samples()[param], mass) for param in model._get_parameter_order()
    ]

    # Construct upper and lower parameters
    lower_quantile_params = jnp.array([value[0] for value in hdpi_range])
    upper_quantile_params = jnp.array([value[1] for value in hdpi_range])

    lower_quantile_pred = _simulate_quantile(
        model,
        lower_quantile_params,
        time,
        initial_conditions,
    )

    upper_quantile_pred = _simulate_quantile(
        model,
        upper_quantile_params,
        time,
        initial_conditions,
    )

    # Stack both and get the minimum across the stacked dim
    stacked_quantiles = jnp.stack([lower_quantile_pred, upper_quantile_pred])
    lower_end = stacked_quantiles.min(axis=0)
    upper_end = stacked_quantiles.max(axis=0)

    return (lower_end, upper_end)


def _simulate_quantile(
    model: Model,
    quantile_params: jax.Array,
    times: jax.Array,
    initial_conditions: List[Dict[str, float]],
):
    # Perform simulation with these parameters
    _, states = model.simulate(
        initial_conditions=initial_conditions,
        saveat=times,
        in_axes=(0, None, 0),
        parameters=quantile_params,
    )

    return states
