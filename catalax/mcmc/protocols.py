from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Tuple, Optional
from jax import Array

if TYPE_CHECKING:
    from catalax.model.model import Model


class PreModel(Protocol):
    """Protocol for pre-model transformation functions.

    Pre-model functions are applied before simulation to transform inputs and parameters.
    They receive the sampled parameters from priors along with initial conditions,
    constants, times, and observed data, allowing for custom transformations before
    the model simulation step.

    Example use cases:
    - Parameter transformations (e.g., log-space to linear)
    - Dynamic initial condition modifications
    - Time-dependent parameter adjustments
    - Data preprocessing or filtering

    Examples:
        Inferring true initial concentrations when given values are uncertain:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> def infer_initial_concentrations(*, model, y0s, constants, times, data, theta):
        ...     '''Sample true initial concentrations when measured values are uncertain'''
        ...     # Assume the first species has uncertain initial concentration
        ...     # The measured value y0s[0] is treated as a noisy observation
        ...     measured_y0 = y0s[0]
        ...     measurement_error = 0.1 * measured_y0  # 10% measurement uncertainty
        ...     
        ...     # Sample the true initial concentration around the measured value
        ...     true_y0_0 = numpyro.sample("true_initial_conc_0", 
        ...                               dist.Normal(measured_y0, measurement_error))
        ...     
        ...     # Ensure positive concentration
        ...     positive_y0_0 = numpyro.deterministic("positive_initial_conc_0", 
        ...                                          jnp.maximum(true_y0_0, 1e-6))
        ...     
        ...     # Update initial conditions with inferred value
        ...     corrected_y0s = numpyro.deterministic("corrected_initial_conditions",
        ...                                          y0s.at[0].set(positive_y0_0))
        ...     
        ...     return corrected_y0s, constants, times, data, theta

        Using PreModel with HMC:

        >>> import catalax.mcmc as cmc
        >>> # Setup your model and dataset first
        >>> config = cmc.MCMCConfig(num_warmup=500, num_samples=1000)
        >>> results = cmc.run_mcmc(
        ...     model=model,
        ...     dataset=dataset,
        ...     yerrs=2.0,
        ...     config=config,
        ...     pre_model=infer_initial_concentrations
        ... )
    """

    def __call__(
        self,
        *,
        model: "Model",
        y0s: Array,
        constants: Array,
        times: Array,
        data: Optional[Array],
        theta: Array,
    ) -> Tuple[Array, Array, Array, Optional[Array], Array]:
        """Transform inputs and parameters before simulation.

        Args:
            model: The model being fit
            y0s: Initial conditions for each species
            constants: System constants
            times: Time points for simulation
            data: Observed data to fit against (may be None)
            theta: Sampled parameters from priors

        Returns:
            Tuple of (y0s, constants, times, data, theta) after transformation
        """
        ...


class PostModel(Protocol):
    """Protocol for post-model transformation functions.

    Post-model functions are applied after simulation to transform the model outputs.
    They receive the simulated states along with the observed data and time points,
    allowing for custom transformations of the simulation results before likelihood
    evaluation.

    Example use cases:
    - Unit conversions on simulation outputs
    - Observable transformations (e.g., concentrations to measurements)  
    - Time-dependent observation corrections
    - Data alignment or interpolation

    Examples:
        Algebraic conversions when observable is the sum of multiple species:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> def sum_species_observable(*, model, states, data, times):
        ...     '''Convert individual species concentrations to observable total'''
        ...     # Example: Total protein concentration is sum of bound and unbound forms
        ...     # states[:, 0] = unbound protein, states[:, 1] = bound protein
        ...     unbound_protein = numpyro.deterministic("unbound_protein", states[:, 0])
        ...     bound_protein = numpyro.deterministic("bound_protein", states[:, 1])
        ...     
        ...     # Observable is the total protein concentration
        ...     total_protein = numpyro.deterministic("total_protein_observable", 
        ...                                         unbound_protein + bound_protein)
        ...     
        ...     # Create observable states array with only the measurable quantity
        ...     observable_states = numpyro.deterministic("observable_states",
        ...                                              total_protein[:, None])  # Shape: (n_times, 1)
        ...     
        ...     return observable_states, data, times

        Multiple algebraic observables from species combinations:

        >>> def multiple_observables(*, model, states, data, times):
        ...     '''Create multiple observables from species combinations'''
        ...     # Example: Enzyme kinetics with substrate (S), enzyme (E), complex (ES), product (P)
        ...     # states[:, 0] = S, states[:, 1] = E, states[:, 2] = ES, states[:, 3] = P
        ...     
        ...     # Observable 1: Total substrate (free + bound)
        ...     total_substrate = numpyro.deterministic("total_substrate",
        ...                                            states[:, 0] + states[:, 2])
        ...     
        ...     # Observable 2: Total enzyme (free + bound) 
        ...     total_enzyme = numpyro.deterministic("total_enzyme",
        ...                                         states[:, 1] + states[:, 2])
        ...     
        ...     # Observable 3: Product formation (direct measurement)
        ...     product_conc = numpyro.deterministic("product_concentration", states[:, 3])
        ...     
        ...     # Combine into observable states matrix
        ...     observable_states = numpyro.deterministic("combined_observables",
        ...         jnp.stack([total_substrate, total_enzyme, product_conc], axis=1))
        ...     
        ...     return observable_states, data, times

        Using PostModel with HMC:

        >>> import catalax.mcmc as cmc
        >>> # Setup your model and dataset first  
        >>> config = cmc.MCMCConfig(num_warmup=500, num_samples=1000)
        >>> results = cmc.run_mcmc(
        ...     model=model,
        ...     dataset=dataset,
        ...     yerrs=2.0,
        ...     config=config,
        ...     post_model=sum_species_observable
        ... )

        Combining PreModel and PostModel:

        >>> config = cmc.MCMCConfig(num_warmup=500, num_samples=1000)
        >>> results = cmc.run_mcmc(
        ...     model=model,
        ...     dataset=dataset,
        ...     yerrs=2.0,
        ...     config=config,
        ...     pre_model=infer_initial_concentrations,  # Infer uncertain initial conditions
        ...     post_model=sum_species_observable        # Observable is sum of species
        ... )
    """

    def __call__(
        self,
        *,
        model: "Model",
        states: Array,
        data: Optional[Array],
        times: Array,
    ) -> Tuple[Array, Optional[Array], Array]:
        """Transform outputs after simulation.

        Args:
            model: The model being fit
            states: Simulated model states for all species and time points
            data: Observed data to fit against (may be None)
            times: Time points for simulation

        Returns:
            Tuple of (states, data, times) after transformation
        """
        ...
