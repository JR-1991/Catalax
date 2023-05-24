from typing import Dict, Tuple

import arviz as az
import corner
import jax
import matplotlib.pyplot as plt
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC
from sympy import Symbol

from sysbiojax.model.model import Model


def plot_corner(
    mcmc: MCMC, model: Model, quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84)
):
    """Plots the correlation between the parameters.

    Args:
        mcmc (MCMC): The MCMC object to plot.
        model (Model): The model to infer the parameters of.
    """

    data = az.from_numpyro(mcmc)
    _ = corner.corner(
        data,
        plot_contours=False,
        quantiles=list(quantiles),
        title_quantiles=list(quantiles),
        bins=20,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        divergences=True,
        labels=[
            r"$sigma$",
            *[
                model.parameters[param]
                .symbol._repr_latex_()
                .replace("\\displaystyle ", "")
                for param in model._get_parameter_order()
            ],
        ],
        use_math_text=False,
        truths={"theta": _get_truths(model), "sigma": None},
    )

    plt.tight_layout()


def _get_truths(model: Model):
    """Returns the true values of the parameters"""

    return [model.parameters[param].value for param in model._get_parameter_order()]


def plot_posterior(
    mcmc,
    model,
    title: str = "Posterior distribution",
) -> None:
    """Plots the posterior distribution of the given bayes analysis"""

    inf_data = az.from_numpyro(mcmc)
    ax = az.plot_posterior(inf_data)

    for row in ax:
        for a in row:
            title = a.title.get_text()

            if "theta" not in title.lower():
                continue

            index = int(title.split("\n")[-1])
            a.title.set_text(
                Symbol(model._get_parameter_order()[index])
                ._repr_latex_()
                .replace("\\displaystyle ", "")
            )

    plt.tight_layout()


def plot_credibility_interval(
    mcmc, model, initial_condition: Dict[str, float], time: jax.Array, dt0: float = 0.1
) -> None:
    """Plots the credibility interval for a single simulation"""

    samples = mcmc.get_samples()
    samples["theta"].shape

    # Evaluate all parameters from the distribution to gather hpdi
    _, post_states = model.simulate(
        initial_conditions=initial_condition,
        dt0=dt0,
        saveat=time,
        parameters=samples["theta"],
        in_axes=(None, 0, None),
    )

    # Simulate system at the given evaluation point
    _, states = model.simulate(
        initial_conditions=initial_condition,
        dt0=dt0,
        saveat=time,
        parameters=samples["theta"].mean(0),
        in_axes=None,
    )

    # Get HPDI
    hpdi_mu = hpdi(post_states, 0.9)

    for i, species in enumerate(model._get_species_order()):
        plt.plot(time, states[:, i], label=f"{species} simulation")
        plt.fill_between(
            time[0],
            hpdi_mu[0, :, i],  # type: ignore
            hpdi_mu[1, :, i],  # type: ignore
            alpha=0.3,
            interpolate=True,
            label=f"{species} CI",
        )

    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
