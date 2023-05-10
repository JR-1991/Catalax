from typing import Tuple
import arviz as az
import corner
import matplotlib.pyplot as plt
from numpyro.infer import MCMC

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
