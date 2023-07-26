from typing import Dict, Tuple

import arviz as az
import corner
import jax
import matplotlib.pyplot as plt
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC


def plot_corner(
    mcmc: MCMC,
    quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
):
    """Plots the correlation between the parameters.

    Args:
        mcmc (MCMC): The MCMC object to plot.
        model (Model): The model to infer the parameters of.
    """

    data = az.from_numpyro(mcmc)
    fig = corner.corner(
        data,
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


def plot_posterior(mcmc, model, **kwargs) -> None:
    """Plots the posterior distribution of the given bayes analysis"""

    inf_data = az.from_numpyro(mcmc)
    fig = az.plot_posterior(inf_data, var_names=model._get_parameter_order(), **kwargs)

    return fig


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


def plot_trace(mcmc, model, **kwargs) -> None:
    """Plots the trace of the given bayes analysis"""

    inf_data = az.from_numpyro(mcmc)
    f = az.plot_trace(inf_data, var_names=model._get_parameter_order(), **kwargs)

    plt.tight_layout()

    return f


def plot_forest(mcmc, model, **kwargs) -> None:
    """Plots a forest plot of the given bayes analysis"""

    inf_data = az.from_numpyro(mcmc)
    f = az.plot_forest(inf_data, var_names=model._get_parameter_order(), **kwargs)

    plt.tight_layout()

    return f
