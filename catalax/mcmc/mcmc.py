from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from catalax.dataset.dataset import Dataset
from catalax.surrogate import Surrogate

if TYPE_CHECKING:
    from catalax.model.model import Model


@dataclass
class MCMCConfig:
    """Configuration parameters for MCMC simulation.

    Args:
        num_warmup: Number of warmup steps before sampling
        num_samples: Number of posterior samples to collect
        likelihood: Likelihood distribution class (default: SoftLaplace)
        dense_mass: Whether to use dense mass matrix (default: True)
        thinning: Factor by which to thin samples (default: 1)
        max_tree_depth: Maximum depth of NUTS binary tree (default: 10)
        dt0: Time step resolution for simulation (default: 0.1)
        chain_method: Chain execution method - "sequential", "parallel", or "vectorized"
        num_chains: Number of Markov chains to run (default: 1)
        seed: Random seed for reproducibility (default: 420)
        verbose: Verbosity level; 0 for silent, 1 for progress (default: 1)
        max_steps: Maximum number of integration steps (default: 64^4)
    """

    num_warmup: int
    num_samples: int
    likelihood: Type[dist.Distribution] = dist.SoftLaplace
    dense_mass: bool = True
    thinning: int = 1
    max_tree_depth: int = 10
    dt0: float = 0.1
    chain_method: str = "sequential"
    num_chains: int = 1
    seed: int = 420
    verbose: int = 1
    max_steps: int = 64**4


def run_mcmc(
    model: Model,
    dataset: Dataset,
    yerrs: Union[float, Array],
    config: MCMCConfig,
    surrogate: Optional[Surrogate] = None,
) -> Tuple[MCMC, Any]:
    """Run MCMC simulation to infer posterior distribution of parameters.

    Uses NumPyro to perform Markov Chain Monte Carlo simulation with the No-U-Turn
    Sampler (NUTS) algorithm. Parameter priors are automatically extracted from the model.
    The simulation compares model output against observed data to infer the posterior
    distribution of parameters.

    Args:
        model: Model to fit
        dataset: Dataset containing observations to fit against
        yerrs: Standard deviation of observed data
        config: MCMC configuration parameters
        surrogate: Optional surrogate model for rate prediction

    Returns:
        Tuple of (MCMC object with results, Bayesian model function)

    Raises:
        AssertionError: If any parameters lack prior distributions
    """
    # Validate inputs
    _validate_parameter_priors(model)

    # Setup model and prepare data
    data_prep = _prepare_mcmc_data(dataset, model, config, surrogate)

    # Create Bayesian model
    bayes_model = _create_bayesian_model(
        model=model,
        yerrs=yerrs,
        likelihood=config.likelihood,
        sim_func=data_prep.sim_func,
    )

    # Initialize and run MCMC
    mcmc = _initialize_mcmc_sampler(bayes_model, config)
    _run_mcmc_simulation(mcmc, data_prep, config)

    return mcmc, bayes_model


@dataclass
class ModelSetup:
    """Container for model setup results."""

    priors: List[Tuple[str, dist.Distribution]]


@dataclass
class DataPreparation:
    """Container for prepared MCMC data."""

    data: Array
    times: Array
    y0s: Array
    constants: Array
    sim_func: Callable


def _prepare_mcmc_data(
    dataset: Dataset, model: Model, config: MCMCConfig, surrogate: Optional[Surrogate]
) -> DataPreparation:
    """Prepare all data components needed for MCMC simulation.

    Args:
        dataset: Dataset containing observations
        model: Model to be fit
        config: MCMC configuration
        surrogate: Optional surrogate model

    Returns:
        DataPreparation containing all prepared data
    """
    # Extract basic dataset components
    data, times, y0s, constants = _extract_dataset_components(dataset, model)

    # Setup simulation function
    sim_func, data, y0s, times = _configure_simulation_function(
        model=model,
        surrogate=surrogate,
        dataset=dataset,
        data=data,
        y0s=y0s,
        times=times,
    )

    return DataPreparation(
        data=data,
        times=times,
        y0s=y0s,
        constants=constants,
        sim_func=sim_func,
    )


def _initialize_mcmc_sampler(bayes_model: Callable, config: MCMCConfig) -> MCMC:
    """Initialize the MCMC sampler with configuration.

    Args:
        bayes_model: Bayesian model function
        config: MCMC configuration parameters

    Returns:
        Initialized MCMC sampler
    """
    return MCMC(
        NUTS(
            bayes_model,
            dense_mass=config.dense_mass,
            max_tree_depth=config.max_tree_depth,
        ),
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        progress_bar=bool(config.verbose),
        chain_method=config.chain_method,
        num_chains=config.num_chains,
        jit_model_args=True,
        thinning=config.thinning,
    )


def _run_mcmc_simulation(
    mcmc: MCMC, data_prep: DataPreparation, config: MCMCConfig
) -> None:
    """Execute the MCMC simulation.

    Args:
        mcmc: Initialized MCMC sampler
        data_prep: Prepared data for simulation
        config: MCMC configuration
    """
    if config.verbose:
        print("\n🚀 Running MCMC\n")

    mcmc.run(
        PRNGKey(config.seed),
        data=data_prep.data,
        y0s=data_prep.y0s,
        times=data_prep.times,
        constants=data_prep.constants,
    )

    if config.verbose:
        print("\n\n🎉 Finished")
        mcmc.print_summary()


def _validate_parameter_priors(model: Model) -> None:
    """Ensure all model parameters have specified prior distributions.

    Args:
        model: Model whose parameters should be validated

    Raises:
        AssertionError: If any parameters lack prior distributions
    """
    missing_priors = [
        param.name for param in model.parameters.values() if param.prior is None
    ]
    assert not missing_priors, (
        f"Parameters {', '.join(missing_priors)} do not have priors. Please specify priors for all parameters."
    )


def _extract_dataset_components(
    dataset: Dataset, model: Model
) -> Tuple[Array, Array, Array, Array]:
    """Extract and prepare dataset components for MCMC simulation.

    Args:
        dataset: Dataset containing observations
        model: Model to be fit to the data

    Returns:
        Tuple of (data, times, initial conditions, constants)
    """
    # Extract data, times, and initial conditions
    data, times, y0s = dataset.to_jax_arrays(
        model.get_species_order(),
        inits_to_array=True,
    )

    # Get constants from dataset
    constants = dataset.to_y0_matrix(species_order=model.get_constants_order())

    # Create initial conditions for all species, including non-observable ones
    all_species = model.get_species_order()
    full_y0s = []

    for meas in dataset.measurements:
        y0 = jnp.array([meas.initial_conditions[species] for species in all_species])
        full_y0s.append(y0)

    y0s = jnp.stack(full_y0s)

    return data, times, y0s, constants


def _configure_simulation_function(
    model: Model,
    surrogate: Optional[Surrogate],
    dataset: Dataset,
    data: Array,
    y0s: Array,
    times: Array,
) -> Tuple[Callable, Array, Array, Array]:
    """Configure the appropriate simulation function based on model type.

    Args:
        model: Model to simulate
        surrogate: Optional surrogate model
        dataset: Dataset for simulation
        data: Observation data array
        y0s: Initial conditions array
        times: Time points array

    Returns:
        Tuple of (simulation function, modified data, modified y0s, modified times)
    """
    data, times, _ = dataset.to_jax_arrays(model.get_species_order())

    if surrogate is not None:
        # Use surrogate model for rate prediction
        rate_fun = model._setup_rate_function(in_axes=None)
        y0s = data.reshape(-1, data.shape[-1])

        def sim_func(y0s, theta, constants, times):
            return rate_fun(times, y0s, (theta, constants))

        sim_func = jax.jit(jax.vmap(sim_func, in_axes=(0, None, None, 0)))
        times = times.ravel()
        data = surrogate.predict_rates(dataset=dataset)
    else:
        # Use standard simulation function
        sim_func = model._sim_func  # type: ignore

    return sim_func, data, y0s, times  # type: ignore


def _create_bayesian_model(
    model: Model,
    yerrs: Union[float, Array],
    likelihood: Type[dist.Distribution],
    sim_func: Callable,
) -> Callable:
    """Create the Bayesian model for MCMC simulation.

    Args:
        model: Model being fit
        yerrs: Standard deviation of observed data
        likelihood: Likelihood distribution class
        sim_func: Function to simulate model with given parameters

    Returns:
        Bayesian model function for MCMC sampling
    """
    # Get parameter priors
    priors = [
        (model.parameters[param].name, model.parameters[param].prior._distribution_fun)
        for param in model.get_parameter_order()
    ]

    # Extract observable species indices
    observables = jnp.array(
        [
            i
            for i, species in enumerate(model.get_species_order())
            if model.odes[species].observable
        ]
    )

    def _bayes_model(
        y0s: Array,
        constants: Array,
        times: Array,
        data: Optional[Array] = None,
    ):
        """Bayesian model for parameter posterior sampling.

        Samples parameters from priors, simulates the model, and compares
        against observed data to build the posterior distribution.

        Args:
            y0s: Initial conditions
            constants: System constants
            times: Time points for simulation
            data: Observed data to fit against

        Returns:
            Sampled posterior distribution
        """
        # Sample parameters from priors
        theta = jnp.array(
            [numpyro.sample(name, distribution) for name, distribution in priors]
        )

        # Simulate model with sampled parameters
        states = sim_func(y0s, theta, constants, times)

        # Sample noise parameter
        sigma = numpyro.sample("sigma", dist.Normal(0, yerrs))  # type: ignore

        # Compare simulation to observed data
        numpyro.sample("y", likelihood(states[..., observables], sigma), obs=data)

    return _bayes_model
