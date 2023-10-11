from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax import Array
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from catalax.model.parameter import Parameter


def run_mcmc(
    model: "Model",
    data: Array,
    initial_conditions: List[Dict[str, float]],
    times: Array,
    yerrs: Union[float, Array],
    num_warmup: int,
    num_samples: int,
    dense_mass: bool = True,
    thinning: int = 1,
    max_tree_depth: int = 10,
    dt0: float = 0.1,
    chain_method: str = "sequential",
    num_chains: int = 1,
    seed: int = 420,
    in_axes: Optional[Tuple] = (0, None, 0),
    verbose: int = 1,
    max_steps: int = 64**4,
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

    # Check if all paramaters have priors
    assert all(
        param.prior is not None for param in model.parameters.values()
    ), f"Parameters {', '.join([param.name for param in model.parameters.values() if param.prior is None])} do not have priors. Please specify priors for all parameters."

    if verbose:
        _print_priors(model.parameters.values())

    if isinstance(data, np.ndarray):
        data = jnp.array(data)

    if len(data.shape) != 3:
        data = jnp.expand_dims(data, -1)

    # Assemble the initial conditions
    y0s = model._assemble_y0_array(initial_conditions, in_axes=in_axes)

    # Compile the model to obtain the simulation function
    model._setup_system(
        in_axes=in_axes,
        dt0=dt0,
        max_steps=max_steps,
    )

    # Get all priors
    priors = [
        (model.parameters[param].name, model.parameters[param].prior._distribution_fun)
        for param in model._get_parameter_order()
    ]

    # Setup the bayes model
    bayes_model = _setup_model(
        yerrs=yerrs,
        priors=priors,  # type: ignore
        sim_func=model._sim_func,  # type: ignore
        model=model,
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
        print("\n🚀 Running MCMC\n")

    mcmc.run(
        PRNGKey(seed),
        data=data,
        y0s=y0s,
        times=times,
    )

    # Print a nice summary
    if verbose:
        print("\n🎉 Finished")
        mcmc.print_summary()

    return mcmc, bayes_model


def _setup_model(
    yerrs: Union[float, Array],
    sim_func: Callable,
    priors: List[Tuple[str, dist.Distribution]],
    model: "Model",
):
    """Function to setup the model for the MCMC simulation.

    This is done, to not have to pass the priors and the simulation function to the MCMC.
    """

    # Set up the observables to extract from the simulation
    observables = jnp.array(
        [
            i
            for i, species in enumerate(model._get_species_order())
            if model.odes[species].observable
        ]
    )

    def _bayes_model(y0s: Array, times: Array, data: Optional[Array] = None):
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

        theta = jnp.array(
            [numpyro.sample(name, distribution) for name, distribution in priors]
        )

        _, states = sim_func(y0s, theta, times)

        sigma = numpyro.sample("sigma", dist.Normal(0, yerrs))  # type: ignore

        numpyro.sample("y", dist.Normal(states[..., observables], sigma), obs=data)  # type: ignore

    return _bayes_model


def _print_priors(parameters: List[Parameter]):
    fun = lambda name, value: f"├── \033[1m{name}\033[0m: {value}"
    statements = [
        f"🔸 Priors",
        *[fun(param.name, param.prior._print_str) for param in parameters],
    ]

    print("\n".join(statements))
