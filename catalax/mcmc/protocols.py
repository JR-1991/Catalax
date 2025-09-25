from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    overload,
)

from jax import Array

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from catalax.mcmc.mcmc import Modes

    P = ParamSpec("P")

# Type variables for generic decorators
PreModelFunc = TypeVar("PreModelFunc", bound=Callable[[Any], Any])
PostModelFunc = TypeVar("PostModelFunc", bound=Callable[[Any], Any])

if TYPE_CHECKING:
    from catalax.model.model import Model


@dataclass
class Shapes:
    """Shapes of the data, initial conditions, constants, and times.

    This is a helper class to store the shapes of the data, initial conditions,
    constants, and times. It is used to pass the shapes to the pre_model functions
    to manipulate the data before simulation.
    """

    y0s: Tuple[int, ...]
    constants: Tuple[int, ...]
    times: Tuple[int, ...]
    data: Tuple[int, ...]

    @property
    def y0s_surrogate(self) -> Tuple[int, ...]:
        """Shapes of the initial conditions for the surrogate model."""
        n_meas, n_time, n_state = self.data
        return (n_meas * n_time, n_state)

    @property
    def data_surrogate(self) -> Tuple[int, ...]:
        """Shapes of the data for the surrogate model."""
        n_meas, n_time, n_state = self.data
        return (n_meas * n_time, n_state)

    @property
    def times_surrogate(self) -> Tuple[int, ...]:
        """Shapes of the times for the surrogate model."""
        n_meas, n_time, _ = self.data
        return (n_meas * n_time,)

    @property
    def constants_surrogate(self) -> Tuple[int, ...]:
        """Shapes of the constants for the surrogate model."""
        n_meas, n_time, _ = self.data
        _, n_constants = self.constants
        return (n_meas * n_time, n_constants)


@dataclass
class PreModelContext:
    """Mutable context for pre-model transformations.

    Users can modify any of these attributes directly:
    - y0s: Initial conditions
    - constants: System constants
    - times: Time points
    - data: Observed data
    - theta: Model parameters

    The pre_model function can optionally return an Array that will be
    passed to the post_model as pre_output.
    """

    model: "Model"
    y0s: Array
    constants: Array
    times: Array
    data: Optional[Array]
    theta: Array
    shapes: Shapes
    mode: Modes


@dataclass
class PostModelContext:
    """Mutable context for post-model transformations.

    Users can modify any of these attributes directly:
    - states: Simulated model states
    - data: Observed data
    - times: Time points
    - pre_output: Output from pre_model (if any)
    """

    model: "Model"
    states: Array
    data: Optional[Array]
    times: Array
    pre_output: Optional[Array] = None


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
        ...     # Assume the first state has uncertain initial concentration
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
        shapes: Shapes,
        mode: Modes,
    ) -> Tuple[
        Array,  # y0s
        Array,  # constants
        Array,  # times
        Optional[Array],  # data
        Array,  # theta
        Array | None,  # pre_output
    ]:
        """Transform inputs and parameters before simulation.

        Args:
            model: The model being fit
            y0s: Initial conditions for each state
            constants: System constants
            times: Time points for simulation
            data: Observed data to fit against (may be None)
            theta: Sampled parameters from priors

        Returns:
            Tuple of (y0s, constants, times, data, theta, pre_output) after transformation
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
        Algebraic conversions when observable is the sum of multiple state:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> def sum_state_observable(*, model, states, data, times):
        ...     '''Convert individual state concentrations to observable total'''
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

        Multiple algebraic observables from state combinations:

        >>> def multiple_observables(*, model, states, data, times):
        ...     '''Create multiple observables from state combinations'''
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
        ...     post_model=sum_state_observable
        ... )

        Combining PreModel and PostModel:

        >>> config = cmc.MCMCConfig(num_warmup=500, num_samples=1000)
        >>> results = cmc.run_mcmc(
        ...     model=model,
        ...     dataset=dataset,
        ...     yerrs=2.0,
        ...     config=config,
        ...     pre_model=infer_initial_concentrations,  # Infer uncertain initial conditions
        ...     post_model=sum_state_observable        # Observable is sum of state
        ... )
    """

    def __call__(
        self,
        *,
        model: "Model",
        states: Array,
        data: Optional[Array],
        times: Array,
        pre_output: Array | None,
    ) -> Tuple[
        Array,  # states
        Optional[Array],  # data
        Array,  # times
    ]:
        """Transform outputs after simulation.

        Args:
            model: The model being fit
            states: Simulated model states for all state and time points
            data: Observed data to fit against (may be None)
            times: Time points for simulation

        Returns:
            Tuple of (states, data, times) after transformation
        """
        ...


# Simple, mutation-based decorators using context classes with proper type inference


@overload
def pre_model(func: Callable[[PreModelContext], None]) -> PreModel: ...


@overload
def pre_model(func: Callable[[PreModelContext], Array]) -> PreModel: ...


@overload
def pre_model(func: Callable[[PreModelContext], Optional[Array]]) -> PreModel: ...


def pre_model(func: Callable[[PreModelContext], Any]) -> PreModel:
    """Decorator to create a PreModel from a simple user function.

    The user function must be type annotated and take a PreModelContext parameter.
    It can:
    1. Modify the context attributes directly (ctx.y0s, ctx.theta, etc.)
    2. Optionally return an Array that will be passed to post_model

    Args:
        func: User function that takes PreModelContext and optionally returns Array

    Returns:
        Function that conforms to PreModel protocol

    Examples:
        Simple initial condition modification:

        >>> @pre_model
        ... def estimate_initials(ctx: PreModelContext):
        ...     import numpyro
        ...     import numpyro.distributions as dist
        ...     ctx.y0s = numpyro.sample("estimated_y0s", dist.Normal(ctx.y0s, 0.1))

        Parameter transformation:

        >>> @pre_model
        ... def log_transform_params(ctx: PreModelContext):
        ...     import jax.numpy as jnp
        ...     ctx.theta = jnp.exp(ctx.theta)

        With pre_output for post_model:

        >>> @pre_model
        ... def compute_scaling(ctx: PreModelContext):
        ...     import jax.numpy as jnp
        ...     ctx.y0s = ctx.y0s * 2.0  # Scale initial conditions
        ...     return jnp.array([2.0])   # Return scaling factor for post_model
    """

    def wrapper(
        *,
        model: "Model",
        y0s: Array,
        constants: Array,
        times: Array,
        data: Optional[Array],
        theta: Array,
        shapes: Shapes,
        mode: Modes,
    ) -> Tuple[Array, Array, Array, Optional[Array], Array, Array | None]:
        # Create mutable context
        ctx = PreModelContext(model, y0s, constants, times, data, theta, shapes, mode)

        # Call user function - may modify context and/or return pre_output
        pre_output = func(ctx)

        # Return modified context values
        return ctx.y0s, ctx.constants, ctx.times, ctx.data, ctx.theta, pre_output

    # Preserve function metadata without copying signature
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@overload
def post_model(func: Callable[[PostModelContext], None]) -> PostModel: ...


@overload
def post_model(func: Callable[[PostModelContext], Any]) -> PostModel: ...


def post_model(func: Callable[[PostModelContext], Any]) -> PostModel:
    """Decorator to create a PostModel from a simple user function.

    The user function must be type annotated and take a PostModelContext parameter.
    It can modify the context attributes directly (ctx.states, ctx.data, ctx.times).

    Args:
        func: User function that takes PostModelContext

    Returns:
        Function that conforms to PostModel protocol

    Examples:
        Simple state transformation:

        >>> @post_model
        ... def sum_state(ctx: PostModelContext):
        ...     import jax.numpy as jnp
        ...     ctx.states = jnp.sum(ctx.states, axis=1, keepdims=True)

        Multiple transformations:

        >>> @post_model
        ... def combine_observables(ctx: PostModelContext):
        ...     import jax.numpy as jnp
        ...     import numpyro
        ...
        ...     # ctx.states[:, 0] = state A, ctx.states[:, 1] = state B
        ...     total = numpyro.deterministic("total_AB", ctx.states[:, 0] + ctx.states[:, 1])
        ...     ratio = numpyro.deterministic("ratio_AB", ctx.states[:, 0] / ctx.states[:, 1])
        ...     ctx.states = jnp.stack([total, ratio], axis=1)

        Using pre_output:

        >>> @post_model
        ... def scale_with_pre_output(ctx: PostModelContext):
        ...     if ctx.pre_output is not None:
        ...         scaling = ctx.pre_output[0]  # Get scaling from pre_model
        ...         ctx.states = ctx.states / scaling  # Undo scaling
    """

    def wrapper(
        *,
        model: "Model",
        states: Array,
        data: Optional[Array],
        times: Array,
        pre_output: Array | None,
    ) -> Tuple[Array, Optional[Array], Array]:
        # Create mutable context
        ctx = PostModelContext(model, states, data, times, pre_output)

        # Call user function - modifies context in place
        func(ctx)

        # Return modified context values
        return ctx.states, ctx.data, ctx.times

    # Preserve function metadata without copying signature
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
