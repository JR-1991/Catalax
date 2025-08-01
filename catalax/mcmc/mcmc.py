from __future__ import annotations
from copy import deepcopy
import inspect
import matplotlib.pyplot as plt

from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Dict,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import Array
from jax.random import PRNGKey
import pandas as pd
import xarray

from catalax.dataset.dataset import Dataset
from catalax.surrogate import Surrogate
from catalax.mcmc.protocols import PreModel, PostModel

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
    solver: str = "scipy"


class HMC:
    """Hamiltonian Monte Carlo sampler wrapper.

    This class provides a high-level interface for running HMC/NUTS sampling
    with integrated configuration and result handling.
    """

    def __init__(
        self,
        num_warmup: int,
        num_samples: int,
        likelihood: Type[dist.Distribution] = dist.SoftLaplace,
        dense_mass: bool = True,
        thinning: int = 1,
        max_tree_depth: int = 10,
        dt0: float = 0.1,
        chain_method: str = "sequential",
        num_chains: int = 1,
        seed: int = 420,
        verbose: int = 1,
        max_steps: int = 64**4,
    ):
        """Initialize HMC sampler.

        Args:
            num_warmup: Number of warmup steps before sampling
            num_samples: Number of posterior samples to collect
            likelihood: Likelihood distribution class
            dense_mass: Whether to use dense mass matrix
            thinning: Factor by which to thin samples
            max_tree_depth: Maximum depth of NUTS binary tree
            dt0: Time step resolution for simulation
            chain_method: Chain execution method - "sequential", "parallel", or "vectorized"
            num_chains: Number of Markov chains to run
            seed: Random seed for reproducibility
            verbose: Verbosity level; 0 for silent, 1 for progress
            max_steps: Maximum number of integration steps
        """
        self.config = MCMCConfig(
            num_warmup=num_warmup,
            num_samples=num_samples,
            likelihood=likelihood,
            dense_mass=dense_mass,
            thinning=thinning,
            max_tree_depth=max_tree_depth,
            dt0=dt0,
            chain_method=chain_method,
            num_chains=num_chains,
            seed=seed,
            verbose=verbose,
            max_steps=max_steps,
        )
        self.likelihood = likelihood

    def run(
        self,
        model: "Model",
        dataset: Dataset,
        yerrs: Union[float, Array],
        surrogate: Optional[Surrogate] = None,
        pre_model: Optional[PreModel] = None,
        post_model: Optional[PostModel] = None,
    ) -> HMCResults:
        """Run MCMC simulation to infer posterior distribution of parameters.

        Uses NumPyro to perform Markov Chain Monte Carlo simulation with the No-U-Turn
        Sampler (NUTS) algorithm. Parameter priors are automatically extracted from the model.
        The simulation compares model output against observed data to infer the posterior
        distribution of parameters.

        Args:
            model: Model to fit
            dataset: Dataset containing observations to fit against
            yerrs: Standard deviation of observed data
            surrogate: Optional surrogate model for rate prediction
            pre_model: Optional function to transform inputs and parameters before simulation
            post_model: Optional function to transform outputs after simulation

        Returns:
            Tuple of (MCMC object with results, Bayesian model function)

        Raises:
            AssertionError: If any parameters lack prior distributions
            TypeError: If pre_model or post_model functions are not callable
            ValueError: If pre_model or post_model signatures don't match Protocol requirements
        """
        # Validate inputs
        _validate_parameter_priors(model)
        _validate_model_functions(pre_model, post_model)

        # Setup model and prepare data
        data_prep = _prepare_mcmc_data(dataset, model, self.config, surrogate)

        # Create Bayesian model
        bayes_model = _create_bayesian_model(
            model=model,
            yerrs=yerrs,
            likelihood=self.config.likelihood,
            sim_func=data_prep.sim_func,
            pre_model=pre_model,
            post_model=post_model,
        )

        # Initialize and run MCMC
        mcmc = _initialize_mcmc_sampler(bayes_model, self.config)
        _run_mcmc_simulation(mcmc, data_prep, self.config)

        return HMCResults(mcmc, bayes_model, model)

    @classmethod
    def from_config(cls, config: MCMCConfig) -> HMC:
        """Create HMC instance from configuration.

        Args:
            config: MCMC configuration parameters
        """
        return cls(
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            likelihood=config.likelihood,
            dense_mass=config.dense_mass,
            thinning=config.thinning,
            max_tree_depth=config.max_tree_depth,
        )


class HMCResults:
    """Results container for HMC sampling with integrated visualization methods."""

    def __init__(self, mcmc: MCMC, bayesian_model: "BayesianModel", model: "Model"):
        """Initialize HMC results.

        Args:
            mcmc: MCMC object containing samples
            bayesian_model: Bayesian model used for sampling
            model: Original model that was fit
        """
        self.mcmc = mcmc
        self.bayesian_model = bayesian_model
        self.model = model

    def get_samples(self) -> Dict[str, Array]:
        """Get posterior samples."""
        return self.mcmc.get_samples()

    def print_summary(self):
        """Print MCMC summary."""
        self.mcmc.print_summary()

    # Visualization methods from plotting.py
    def plot_corner(
        self,
        quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Plot corner plot of parameter correlations.

        Args:
            quantiles: Quantiles to display in the corner plot

        Returns:
            matplotlib figure
        """
        from catalax.mcmc.plotting import plot_corner

        f = plot_corner(self.mcmc, quantiles)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None

        return plt.gcf()

    def plot_posterior(self, show: bool = False, path: Optional[str] = None, **kwargs):
        """Plot posterior distributions.

        Args:
            **kwargs: Additional arguments passed to arviz.plot_posterior

        Returns:
            matplotlib figure
        """
        from catalax.mcmc.plotting import plot_posterior

        f = plot_posterior(self.mcmc, self.model, **kwargs)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None

        return plt.gcf()

    def plot_credibility_interval(
        self,
        initial_condition: Dict[str, float],
        time: Array,
        dt0: float = 0.1,
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Plot credibility intervals for model simulations.

        Args:
            initial_condition: Initial conditions for simulation
            time: Time points for simulation
            dt0: Time step for simulation

        Returns:
            matplotlib figure
        """
        from catalax.mcmc.plotting import plot_credibility_interval

        f = plot_credibility_interval(
            self.mcmc, self.model, initial_condition, time, dt0
        )
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None
        return plt.gcf()

    def plot_trace(self, show: bool = False, path: Optional[str] = None, **kwargs):
        """Plot MCMC trace.

        Args:
            **kwargs: Additional arguments passed to arviz.plot_trace

        Returns:
            matplotlib figure
        """
        from catalax.mcmc.plotting import plot_trace

        f = plot_trace(self.mcmc, self.model, **kwargs)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()

    def plot_forest(self, show: bool = False, path: Optional[str] = None, **kwargs):
        """Plot forest plot of parameter distributions.

        Args:
            **kwargs: Additional arguments passed to arviz.plot_forest

        Returns:
            matplotlib figure
        """
        from catalax.mcmc.plotting import plot_forest

        f = plot_forest(self.mcmc, self.model, **kwargs)
        if path is not None:
            f.savefig(path)

        if show:
            plt.show()
            return None

        return plt.gcf()

    def summary(self, hdi_prob: float = 0.95) -> Union[pd.DataFrame, xarray.Dataset]:
        """Generate summary statistics.

        Args:
            hdi_prob: Probability mass for highest density interval

        Returns:
            Summary statistics
        """
        from catalax.mcmc.plotting import summary

        return summary(self.mcmc, hdi_prob)


class BayesianModel:
    """Bayesian model for MCMC parameter inference.

    This class creates a callable Bayesian model that can be used with NumPyro's
    MCMC samplers. It encapsulates the model parameters, priors, and likelihood
    configuration.
    """

    def __init__(
        self,
        model: "Model",
        yerrs: Union[float, Array],
        likelihood: Type[dist.Distribution],
        sim_func: Callable,
        pre_model: Optional[PreModel] = None,
        post_model: Optional[PostModel] = None,
    ):
        """Initialize the Bayesian model.

        Args:
            model: The model being fit
            yerrs: Standard deviation of observed data
            likelihood: Likelihood distribution class
            sim_func: Function to simulate model with given parameters
            pre_model: Optional function to transform inputs and parameters before simulation
            post_model: Optional function to transform outputs after simulation
        """
        self.model = deepcopy(model)
        self.model._setup_system()
        self.yerrs = yerrs
        self.likelihood = likelihood
        self.sim_func = sim_func
        self.pre_model = pre_model
        self.post_model = post_model

        # Pre-compute priors and observables
        self.priors = [
            (
                model.parameters[param].name,
                model.parameters[param].prior._distribution_fun(),
            )
            for param in model.get_parameter_order()
        ]

        self.observables = jnp.array(
            [
                i
                for i, species in enumerate(model.get_species_order())
                if model.odes[species].observable
            ]
        )

    def __call__(
        self,
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
            [numpyro.sample(name, distribution) for name, distribution in self.priors]
        )

        # Apply a pre-model which can be used to modify
        # the initial conditions, constants, times, data, and parameters
        if self.pre_model is not None:
            y0s, constants, times, data, theta = self.pre_model(
                model=self.model,
                y0s=y0s,
                constants=constants,
                times=times,
                data=data,
                theta=theta,
            )

        # Simulate model with sampled parameters
        states = self.sim_func(y0s, theta, constants, times)

        # Apply a post-model which can be used to modify
        # the states, data, and times
        if self.post_model is not None:
            states, data, times = self.post_model(
                model=self.model,
                states=states,
                data=data,
                times=times,
            )

        # Sample noise parameter
        sigma = numpyro.sample("sigma", dist.Normal(0, self.yerrs))  # type: ignore

        # Compare simulation to observed data
        numpyro.sample(
            "y", self.likelihood(states[..., self.observables], sigma), obs=data
        )


def run_mcmc(
    model: "Model",
    dataset: Dataset,
    yerrs: Union[float, Array],
    config: MCMCConfig,
    surrogate: Optional[Surrogate] = None,
    pre_model: Optional[PreModel] = None,
    post_model: Optional[PostModel] = None,
) -> HMCResults:
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
        pre_model: Optional function to transform inputs and parameters before simulation
        post_model: Optional function to transform outputs after simulation

    Returns:
        Tuple of (MCMC object with results, Bayesian model function)

    Raises:
        AssertionError: If any parameters lack prior distributions
        TypeError: If pre_model or post_model functions are not callable
        ValueError: If pre_model or post_model signatures don't match Protocol requirements
    """
    # Validate inputs
    hmc = HMC.from_config(config)
    return hmc.run(model, dataset, yerrs, surrogate, pre_model, post_model)


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


def _create_bayesian_model(
    model: "Model",
    yerrs: Union[float, Array],
    likelihood: Type[dist.Distribution],
    sim_func: Callable,
    pre_model: Optional[PreModel] = None,
    post_model: Optional[PostModel] = None,
) -> BayesianModel:
    """Create the Bayesian model for MCMC simulation.

    Args:
        model: Model being fit
        yerrs: Standard deviation of observed data
        likelihood: Likelihood distribution class
        sim_func: Function to simulate model with given parameters
        pre_model: Optional function to transform inputs and parameters before simulation
        post_model: Optional function to transform outputs after simulation

    Returns:
        BayesianModel instance for MCMC sampling
    """
    return BayesianModel(model, yerrs, likelihood, sim_func, pre_model, post_model)


def _prepare_mcmc_data(
    dataset: Dataset,
    model: "Model",
    config: MCMCConfig,
    surrogate: Optional[Surrogate],
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
    model = deepcopy(model)
    model._setup_system()
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
        mcmc.print_summary()


def _validate_parameter_priors(model: "Model") -> None:
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


def _validate_model_functions(
    pre_model: Optional[PreModel], post_model: Optional[PostModel]
) -> None:
    """Validate that pre_model and post_model functions conform to expected signatures.

    Args:
        pre_model: Optional pre-model transformation function
        post_model: Optional post-model transformation function

    Raises:
        TypeError: If functions don't conform to expected Protocol signatures
        ValueError: If functions have incorrect parameter signatures
    """
    if pre_model is not None:
        _validate_pre_model_signature(pre_model)

    if post_model is not None:
        _validate_post_model_signature(post_model)


def _validate_pre_model_signature(pre_model: PreModel) -> None:
    """Validate PreModel function signature.

    Args:
        pre_model: Pre-model function to validate

    Raises:
        TypeError: If function is not callable
        ValueError: If function signature doesn't match Protocol
    """
    if not callable(pre_model):
        raise TypeError("pre_model must be callable")

    try:
        sig = inspect.signature(pre_model)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not inspect pre_model signature: {e}")

    # Expected parameters for PreModel
    expected_params = {"model", "y0s", "constants", "times", "data", "theta"}

    # Check that function accepts keyword arguments
    actual_params = set(sig.parameters.keys())

    # Check for **kwargs (accepts any keyword args) or exact match
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not has_var_keyword and not expected_params.issubset(actual_params):
        missing_params = expected_params - actual_params
        raise ValueError(
            f"pre_model missing required parameters: {missing_params}. "
            f"Expected signature: def pre_model(*, model, y0s, constants, times, data, theta)"
        )

    # Check that parameters are keyword-only (if not using **kwargs)
    if not has_var_keyword:
        for param_name, param in sig.parameters.items():
            if param_name in expected_params and param.kind not in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"pre_model parameter '{param_name}' must be keyword-only. "
                    f"Use: def pre_model(*, {param_name}, ...)"
                )


def _validate_post_model_signature(post_model: PostModel) -> None:
    """Validate PostModel function signature.

    Args:
        post_model: Post-model function to validate

    Raises:
        TypeError: If function is not callable
        ValueError: If function signature doesn't match Protocol
    """
    if not callable(post_model):
        raise TypeError("post_model must be callable")

    try:
        sig = inspect.signature(post_model)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not inspect post_model signature: {e}")

    # Expected parameters for PostModel
    expected_params = {"model", "states", "data", "times"}

    # Check that function accepts keyword arguments
    actual_params = set(sig.parameters.keys())

    # Check for **kwargs (accepts any keyword args) or exact match
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not has_var_keyword and not expected_params.issubset(actual_params):
        missing_params = expected_params - actual_params
        raise ValueError(
            f"post_model missing required parameters: {missing_params}. "
            f"Expected signature: def post_model(*, model, states, data, times)"
        )

    # Check that parameters are keyword-only (if not using **kwargs)
    if not has_var_keyword:
        for param_name, param in sig.parameters.items():
            if param_name in expected_params and param.kind not in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"post_model parameter '{param_name}' must be keyword-only. "
                    f"Use: def post_model(*, {param_name}, ...)"
                )


def _extract_dataset_components(
    dataset: Dataset, model: "Model"
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
    model: "Model",
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
    if surrogate is not None:
        # Use surrogate model for rate prediction
        data, times, _ = dataset.to_jax_arrays(model.get_species_order())
        rate_fun = model._setup_rate_function(in_axes=None)
        y0s = data.reshape(-1, data.shape[-1])

        def sim_func(y0s, theta, constants, times):
            return rate_fun(times, y0s, (theta, constants))

        sim_func = jax.jit(jax.vmap(sim_func, in_axes=(0, None, None, 0)))
        times = times.ravel()
        data = surrogate.predict_rates(dataset=dataset)
    else:
        # Create a new simulation function with correct axes for MCMC
        # We need to vmap over measurements (axis 0) for y0s, constants, and times
        # but not over theta (same parameters for all measurements)
        from catalax.tools.simulation import Simulation

        odes = [model.odes[species] for species in model.get_species_order()]
        simulation_setup = Simulation(
            odes=odes,
            parameters=model.get_parameter_order(),
            stoich_mat=model._get_stoich_mat(),
            constants=model.get_constants_order(),
            dt0=0.1,  # Default dt0
            rtol=1e-5,
            atol=1e-5,
            max_steps=64**4,
        )

        # Set up simulation with correct in_axes for MCMC
        sim_func, _ = simulation_setup._prepare_func(in_axes=(0, None, 0, 0))

    return sim_func, data, y0s, times  # type: ignore
