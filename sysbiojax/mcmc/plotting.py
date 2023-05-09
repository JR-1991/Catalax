import arviz as az
import corner
import matplotlib.pyplot as plt
from numpyro.infer import MCMC

from sysbiojax.model.model import Model


def plot_corner(mcmc: MCMC, model: Model):
    """Plots the correlation between the parameters.

    Args:
        mcmc (MCMC): _description_
        model (Model): _description_
    """

    data = az.from_numpyro(mcmc)
    _ = corner.corner(
        data,
        plot_contours=False,
        quantiles=[0.05, 0.5, 0.95],
        title_quantiles=[0.05, 0.5, 0.95],
        bins=20,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        divergences=True,
        colot="b",
        labels=[
            r"$sigma$",
            *[
                model.parameters[param]
                .symbol._repr_latex_()
                .replace("\\displaystyle ", "")
                for param in model._get_parameter_order()
            ],
        ],
        use_math_text=True,
    )

    plt.tight_layout()
