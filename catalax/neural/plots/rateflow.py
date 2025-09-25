from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from catalax.dataset.dataset import Dataset
from catalax.model.model import Model

if TYPE_CHECKING:
    from catalax.neural.rateflow import RateFlowODE


def plot_learned_rates(
    rateflow_ode: RateFlowODE,
    dataset: Dataset,
    model: Model,
    show: bool = True,
    save_path: str | None = None,
    round_stoich: bool = True,
) -> Figure:
    """
    Plot learned stoichiometry matrix, rates, and dataset for a trained RateFlowODE.

    Args:
        trained: Trained RateFlowODE instance
        dataset: Dataset containing measurements
        model: Model instance with state information
    """
    pred = rateflow_ode.predict(dataset)
    f, ax = plt.subplots(
        len(pred.measurements),
        3,
        figsize=(15, 4 * len(pred.measurements)),
    )

    for i, measurement in enumerate(pred.measurements):
        data, t, _ = measurement.to_jax_arrays(model.get_state_order())
        sub_ax = ax[i]

        inner = jax.vmap(rateflow_ode.func, in_axes=(0, 0, None))
        rates = inner(t, data, ())

        # Stoichiometry Matrix (left panel)
        stoich_matrix = rateflow_ode.stoich_matrix.T
        if round_stoich:
            stoich_matrix = jnp.round(stoich_matrix)

        sub_ax[0].set_title(
            f"Learned Stoichiometry Matrix {i + 1}", fontsize=12, pad=15
        )
        sns.heatmap(
            stoich_matrix,
            annot=True,
            cmap="RdBu",
            xticklabels=["PGME", "7-ADCA", "CEX", "PG"],
            yticklabels=[f"R{j + 1}" for j in range(stoich_matrix.shape[0])],
            ax=sub_ax[0],
            cbar_kws={"shrink": 0.8},
            square=True,
            linewidths=0.5,
            annot_kws={"size": 10, "weight": "bold"},
        )
        sub_ax[0].set_xlabel("Species", fontsize=12, labelpad=10)
        sub_ax[0].set_ylabel("Reactions", fontsize=12, labelpad=10)
        sub_ax[0].tick_params(axis="y", rotation=0)

        # Rates (middle panel)
        sub_ax[1].set_title(f"Learned Rates {i + 1}", fontsize=12, pad=15)

        for reaction in range(rates.shape[1]):
            sub_ax[1].plot(
                t,
                rates[:, reaction],
                label=f"Reaction {reaction + 1}",
            )

        sub_ax[1].grid(True, which="both", linestyle="--")
        sub_ax[1].grid(True, which="minor", alpha=0.3)
        sub_ax[1].minorticks_on()

        sub_ax[1].legend(fontsize="small", frameon=True, fancybox=True, shadow=True)
        sub_ax[1].set_xlabel("Time", fontsize=12, labelpad=10)
        sub_ax[1].set_ylabel("Rate Magnitude", fontsize=12, labelpad=10)
        sub_ax[1].spines["top"].set_visible(False)
        sub_ax[1].spines["right"].set_visible(False)

        # Model fit (right panel)
        dataset.measurements[i].plot(ax=sub_ax[2], model_data=pred.measurements[i])

        sub_ax[2].set_title(f"Dataset {i + 1}", fontsize=12, pad=15)
        sub_ax[2].legend(fontsize="small", frameon=True, fancybox=True, shadow=True)
        sub_ax[2].grid(alpha=0.2, linestyle="--", linewidth=0.8)
        sub_ax[2].set_xlabel("Time", fontsize=12, labelpad=10)
        sub_ax[2].set_ylabel("Concentration", fontsize=12, labelpad=10)
        sub_ax[2].spines["top"].set_visible(False)
        sub_ax[2].spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return f
