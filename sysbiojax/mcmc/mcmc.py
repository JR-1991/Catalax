from typing import Callable, Dict, List, Optional, Tuple, Union
import jax

import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS
from pydantic import BaseModel
from sysbiojax.model.model import Model
from sysbiojax.model.parameter import Parameter

PRIOR_REQUIRED_ATTRS = ["initial_value", "lower_bound", "upper_bound"]


class Prior(BaseModel):
    """Prior distribution of a parameter"""

    initial_guess: float
    lower_bound: float
    upper_bound: float
    stdev: Optional[float] = None

    @classmethod
    def from_parameter(cls, parameter: Parameter):
        if any(getattr(parameter, attr) is None for attr in PRIOR_REQUIRED_ATTRS):
            raise ValueError(
                f"Parameter {parameter.name} does not have the required attributes {PRIOR_REQUIRED_ATTRS}. Missing attributes: {', '.join([attr for attr in PRIOR_REQUIRED_ATTRS if getattr(parameter, attr) is None])}"
            )

        return cls(
            initial_guess=parameter.initial_value,  # type: ignore
            lower_bound=parameter.lower_bound,  # type: ignore
            upper_bound=parameter.upper_bound,  # type: ignore
            stdev=parameter.stdev,  # type: ignore
        )


def run_mcmc(
    model: Model,
    data: Array,
    initial_conditions: List[Dict[str, float]],
    times: Array,
    yerrs: Union[float, Array],
    num_warmup: int,
    num_samples: int,
    prior_dist: str = "normal",
    dense_mass: bool = True,
    thinning: int = 1,
    max_tree_depth: int = 10,
    dt0: float = 0.1,
    chain_method: str = "sequential",
    num_chains: int = 1,
    seed: int = 420,
    in_axes: Optional[Tuple] = (0, None, 0),
    verbose: int = 1,
    max_steps: int = 4096,
):
    """Runs an MCMC simulation to infer the posterior distribution of parameters.

    This function is using the NumPyro package to execute a Markov Chain Monte Carlo simulation
    using the No-U-Turn Sampler (NUTS) algorithm. Priors for the parameters are automatically extracted
    from the model such that the user only needs to specify the initial conditions, the data, and the
    standard deviation of the data. The simulation function of the model is used to simulate the model
    for each set of parameters sampled from the prior distribution. The simulated data is then compared
    to the observed data and the posterior distribution of parameters is inferred.

    Args:
        model (Model): The model to fit.
        data (Array): The data against which the model is fitted.
        initial_conditions (List[Dict[str, float]]): The initial conditions of the model.
        yerrs (Array, float): The standard deviation of the observed data.
        times (Array): The times at which the data has been measured.
        num_warmup (int): Number of warmup steps.
        num_samples (int): Number of samples.
        dense_mass (bool, optional): Whether to use a dense mass matrix or not. Defaults to True.
        dt0 (float, optional): Resolution of the simulation. Defaults to 0.1.
        chain_method (str, optional): Choose from 'vectorized', 'parallel' or 'sequential'. Defaults to "sequential".
        num_chains (int, optional): Number of chains. Defaults to 1.
        seed (int, optional): Random number seed to reproduce results. Defaults to 0.
        progress_bar (bool, optional): Whether to show a progress bar or not. Defaults to True.

    Returns:
        MCMC: Result of the MCMC simulation.
    """

    if verbose:
        print("<<< Priors >>>")
        _print_priors(model)

    # Extract priors from the model
    p_loc, p_lbounds, p_ubounds, p_scales = _assemble_priors(model)

    # Assemble the initial conditions
    y0s = model._assemble_y0_array(initial_conditions, in_axes=in_axes)

    # Compile the model to obtain the simulation function
    model._setup_system(in_axes=in_axes, dt0=dt0, max_steps=max_steps)

    # Setup the bayes model
    bayes_model = _setup_model(
        yerrs=yerrs,
        priors_loc=p_loc,
        priors_lower_bounds=p_lbounds,
        priors_upper_bounds=p_ubounds,
        priors_scale=p_scales,
        sim_func=model._sim_func,  # type: ignore
    )

    mcmc = MCMC(
        NUTS(bayes_model, dense_mass=dense_mass, max_tree_depth=max_tree_depth),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=bool(verbose),
        chain_method=chain_method,
        num_chains=num_chains,
        jit_model_args=True,
        thinning=thinning,
    )

    if verbose:
        print("<<< Running MCMC >>>")

    mcmc.run(
        PRNGKey(seed),
        data=data,
        y0s=y0s,
        times=times,
    )

    # Print a nice summary
    mcmc.print_summary()

    return mcmc, bayes_model


def _assemble_priors(model: Model) -> Tuple[Array, Array, Array, Array]:
    """Converts the parameters of a model to priors, if all necessary attributes are present.

    Args:
        model (Model): The model to convert the parameters of.

    Returns:
        Tuple[Array, Array, Array, Array]: Location, lower bound, upper bound, and scale of the priors.
    """

    # Convert the parameters to priors
    priors = [
        Prior.from_parameter(model.parameters[param])
        for param in model._get_parameter_order()
    ]

    prior_loc = jnp.array([prior.initial_guess for prior in priors])
    prior_scale = jnp.array([prior.stdev for prior in priors])
    prior_lower_bounds = jnp.array([prior.lower_bound for prior in priors])
    prior_upper_bounds = jnp.array([prior.upper_bound for prior in priors])

    return (
        prior_loc,
        prior_lower_bounds,
        prior_upper_bounds,
        prior_scale,
    )


def _setup_model(
    yerrs: Union[float, Array],
    priors_loc: Array,
    priors_lower_bounds: Array,
    priors_upper_bounds: Array,
    priors_scale: Array,
    sim_func: Callable,
):
    """Function to setup the model for the MCMC simulation.

    This is done, to not have to pass the priors and the simulation function to the MCMC.
    """

    def _bayes_model(
        y0s: Array,
        times: Array,
        data: Optional[Array] = None,
    ):
        """Generalized bayesian model to infer the posterior distribution of parameters.

        This function is used to sample from the posterior distribution of parameters by
        sampling from the prior distribution and comparing the simulated data with the
        observations. Theta is the vector of parameters, sigma is the standard deviation of
        the noise, and states is the simulated data.

        Args:
            data (Array): The data against which the model is fitted.
            y0s (Array): The initial conditions of the model.
            yerrs (Array): The standard deviation of the observed data.
            priors_loc (Array): The times at which the data is sampled.
            prior_low_bounds (Array): The lower bounds of the priors.
            prior_upper_bounds (Array): The upper bounds of the priors.
            priors_scale (Array): The stdev of the priors.
            sim_func (Callable): The simulation function of the model.
        """

        theta = numpyro.sample(
            "theta",
            dist.Uniform(
                low=priors_lower_bounds,  # type: ignore
                high=priors_upper_bounds,  # type: ignore
            ),
        )

        _, states = sim_func(y0s, theta, times)

        sigma = numpyro.sample("sigma", dist.Normal(0, yerrs))  # type: ignore

        numpyro.sample("y", dist.Normal(states, sigma), obs=data)  # type: ignore

    return _bayes_model


def _print_priors(model: Model):
    """Prints all priors of the model into a dataframe for an overview"""

    df = pd.DataFrame(
        [
            {"name": param, **Prior.from_parameter(model.parameters[param]).dict()}
            for param in model._get_parameter_order()
        ]
    )

    try:
        from IPython.display import display

        display(df)

        return ""

    except ImportError:
        return df
