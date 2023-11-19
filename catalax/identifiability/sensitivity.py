from typing import Dict, List, Tuple
import jax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from sympy import Symbol, latex
from tqdm import tqdm

import jax.numpy as jnp
from catalax.model.model import Model
from catalax import INITS, PARAMETERS


def sensitivity_analysis(
    model: Model,
    initial_conditions: List[Dict[str, float]],
    t0: float,
    t1: float,
    nsteps: int = 100,
    scale: Tuple[float, float] = (1 / 3, 4 / 3),
    nsamples: int = 10,
    gap: int = 10,
    plot: bool = True,
    figsize: Tuple[int, int] = (6, 5),
    threshold: float = 1e-9,
    ignore_criteria: bool = False,
):
    _validate_input(model, initial_conditions, t0, t1, scale)
    _print_log(t0, t1, nsteps, scale, gap, nsamples)

    sensitivity_matrix = _calculate_senisitivity_matrix(
        model=model,
        t0=t0,
        t1=t1,
        initial_conditions=initial_conditions,
        lower_scale=scale[0],
        upper_scale=scale[1],
        nsamples=nsamples,
        nsteps=nsteps,
    )

    # Calulate the singular values
    _, singular_vals, vs = jax.vmap(
        jnp.linalg.svd,
        in_axes=(0, None, None),
        out_axes=0,
    )(sensitivity_matrix, False, True)

    vs = jnp.abs(vs)

    # Find singular values that fulfill the gap criteria
    mean_singular_values = singular_vals.mean(axis=0)
    diffs = jnp.diff(jnp.log10(mean_singular_values))
    zero_cols = jnp.where(diffs <= -1 * gap)[0] + 1
    has_criteria = zero_cols.size > 0

    if not has_criteria and not ignore_criteria:
        raise ValueError(
            "❌ No singular values fulfill the gap criteria! If you want to ignore this, set ignore_criteria=True."
        )
    elif not has_criteria and ignore_criteria:
        print(
            "\n❌ No singular values fulfill the gap criteria! Only printing the singular vectors to identify a better gap criteria."
        )
        singular_vectors = jnp.expand_dims(jnp.mean(vs, axis=0)[-1, :], axis=0)
    else:
        singular_vectors = jnp.mean(vs, axis=0)[zero_cols, :]

    singular_vectors = singular_vectors.at[singular_vectors < threshold].set(0)

    print("\n🎉 Finished!")

    if not plot:
        return (
            singular_vectors,
            singular_vals,
            None,
        )

    f = _plot_sensitivity_analysis(
        model=model,
        mean_singular_values=mean_singular_values,
        singular_vectors=singular_vectors,
        figsize=figsize,
        gap=gap,
        has_criteria=has_criteria,
    )

    return (
        singular_vectors,
        singular_vals,
        f,
    )


def _calculate_senisitivity_matrix(
    model: Model,
    t0: float,
    t1: float,
    initial_conditions: List[Dict[str, float]],
    lower_scale: float,
    upper_scale: float,
    nsamples: int,
    nsteps: int,
):
    """
    Calculates the sensitivity matrix for a given model and initial conditions.

    Args:
        model (Model): The model object.
        initial_conditions (List[Dict[str, float]]): The list of initial conditions.
        lower_scale (float): The lower scale value for parameter scaling.
        upper_scale (float): The upper scale value for parameter scaling.
        n_samples (int): The number of samples for parameter scaling.

    Returns:
        jnp.ndarray: The sensitivity matrix.
    """

    sensitivities = []
    pbar = tqdm(range(nsamples), desc="╰── sampling: ")

    for _ in pbar:
        scale = np.random.uniform(
            low=lower_scale,
            high=upper_scale,
            size=len(model._get_parameter_order()),
        )

        _, sensitivity = model.simulate(
            nsteps=nsteps,
            t0=t0,
            t1=t1,
            initial_conditions=initial_conditions,
            in_axes=INITS,
            sensitivity=PARAMETERS,
            parameters=model._get_parameters() * scale,
        )

        sensitivities += [sensitivity]

    sensitivities = jnp.concatenate(sensitivities, axis=0)
    n_inits, n_times, n_species, n_params = sensitivities.shape

    return jnp.reshape(
        sensitivities,
        (n_inits, n_times * n_species, n_params),
    )


def _validate_input(model, initial_conditions, t0, t1, scale):
    """
    Validates the input parameters for sensitivity analysis.

    Args:
        model: The model object.
        initial_conditions: The initial conditions for the model.
        t0: The starting time for the analysis.
        t1: The ending time for the analysis.
        scale: The scale for the analysis.

    Raises:
        AssertionError: If any of the input parameters are invalid.
    """
    assert t0 < t1, "t0 must be smaller than t1"
    assert len(scale) == 2, "scale must be a tuple of length 2"
    assert scale[0] < scale[1], "scale[0] must be smaller than scale[1]"
    assert all(
        parameter.value is not None for parameter in model.parameters.values()
    ), "All parameters must have a value"

    _validate_initial_conditions(model, initial_conditions)


def _validate_initial_conditions(
    model: Model,
    initial_conditions: List[Dict[str, float]],
) -> None:
    """
    Validates the initial conditions for a given model.

    Args:
        model (Model): The model to validate the initial conditions for.
        initial_conditions (List[Dict[str, float]]): The list of initial conditions to validate.

    Raises:
        ValueError: If any initial condition is missing species from the model.
    """
    all_species = model._get_species_order()
    for initial_condition in initial_conditions:
        if not all(species in initial_condition for species in all_species):
            raise ValueError(
                f"Initial condition {initial_condition} is missing species {set(all_species) - set(initial_condition)}"
            )


def _plot_sensitivity_analysis(
    model: Model,
    mean_singular_values: jax.Array,
    singular_vectors: jax.Array,
    figsize: Tuple[int, int],
    gap: int,
    has_criteria: bool,
):
    """
    Plot the sensitivity analysis of the model.

    Args:
        model (Model): The model object.
        mean_singular_values (jax.Array): The mean singular values.
        singular_vectors (jax.Array): The zero vectors.
        figsize (Tuple[int, int]): The size of the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    if not has_criteria:
        figsize = (figsize[0], int(figsize[1] / 2))

    f, ax = plt.subplots(2 - int(not has_criteria), 1, figsize=figsize)

    _plot_singular_values(
        ax[0] if has_criteria else ax,
        model,
        mean_singular_values,
        gap,
        has_criteria,
    )

    if has_criteria:
        _plot_singular_vectors(
            ax[1],
            model,
            singular_vectors,
        )

    return f


def _plot_singular_values(
    ax: plt.Axes,
    model: Model,
    mean_singular_values: jax.Array,
    gap: int,
    has_criteria: bool,
):
    """
    Plot the singular vectors of a model.

    Args:
        ax (plt.Axes): The matplotlib axes object to plot on.
        model (Model): The model object.
        mean_singular_values (jax.Array): The array of mean singular values.

    Returns:
        None
    """

    if has_criteria:
        ax.axhspan(
            ymin=jnp.min(mean_singular_values),
            ymax=mean_singular_values[jnp.argmin(mean_singular_values) - 1],
            xmin=0,
            xmax=11,
            color="gray",
            alpha=0.1,
            linewidth=0,
        )

        ax.text(
            x=0.1,
            y=0.2,
            s="Gap greater than $log_{10}(\Delta_{s})$ = $" + str(gap) + "$",
            fontsize=10,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    ax.plot(mean_singular_values, "o")
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["left"].set_position(("outward", 8))
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":")

    ax.set_xticks(jnp.arange(len(model._get_parameter_order())))
    ax.set_title("Singular values", pad=20)
    ax.set_xlabel("Index")


def _plot_singular_vectors(
    ax: plt.Axes,
    model: Model,
    zero_vectors: jax.Array,
):
    """
    Plot the singular vectors of a model.

    Args:
        ax (plt.Axes): The matplotlib axes object to plot on.
        model (Model): The model object.
        zero_vectors (jax.Array): The array of zero vectors.

    Returns:
        None
    """

    x = jnp.arange(len(model._get_parameter_order()))
    width = 0.3
    distance = 0.15
    n_bars = zero_vectors.shape[0]

    # Singular vectors
    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 5))

    for col, zero_vector in enumerate(zero_vectors):
        ax.bar(
            x=x + col * distance,
            height=zero_vector,
            width=width / n_bars,
        )

    latex_symbols = [
        "${symbol}$".format(symbol=latex(Symbol(model.parameters[param].symbol)))
        for param in model._get_parameter_order()
    ]

    ax.set_xticks(x + (distance * (n_bars - 1)) / n_bars)
    ax.set_xticklabels(latex_symbols)
    ax.set_title("Singular vectors", pad=20)
    ax.set_xlabel("Parameters")

    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":")


def _print_log(
    t0: int,
    t1: int,
    nsteps: int,
    scale: Tuple[float, float],
    gap: int,
    nsamples: int,
):
    statements = [
        f"\n🚀 Sensitivity matrix analysis",
    ]
    fun = lambda name, value: f"├── \033[1m{name}\033[0m: {value}"

    statements += [
        "│",
        fun("t0/t1/steps", f"{t0}/{t1}/{nsteps}"),
        fun("scale", f"{round(scale[0], 2)}/{round(scale[1], 2)}"),
        fun("gap criteria", str(gap)),
        fun("n_samples", nsamples),
        "│",
    ]

    print("\n".join(statements))
