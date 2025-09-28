from copy import deepcopy
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from catalax.dataset.dataset import Dataset
from catalax.model import Model, SimulationConfig


def optimize(
    model: Model,
    dataset: Dataset,
    global_upper_bound: Optional[float] = 1e5,
    global_lower_bound: Optional[float] = 1e-6,
    dt0: float = 0.01,
    objective_fun: Callable = optax.l2_loss,  # type: ignore
    max_steps: int = 64**4,
    method: str = "bfgs",
) -> Tuple[MinimizerResult, Model]:
    """Optimization functions to determine the parameters of a model of a system, given data.

    This method uses the LMFit library to perform parameter inference and thus
    supports all methods provided by LMFit.

    In order to perform an optimization, first specify your ODE model and configure
    the parameter objects in your model. This means to add a lower and upper bound
    for each parameter. You can also set a global upper/lower bound if you want to
    use the same bounds for all parameters. When you pass the model to this function,
    everything will be setup and a parameter optimization will be performed.

    This function returns a results object as well as an update version of your model
    including the inferred parameters. Please note, that the returned model is a modified
    COPY of your input system plus the parameter values. The input system will not be modified.

    Args:
        model (Model): The model to fit.
        dataset (Dataset): The dataset to fit the model to.
        global_upper_bound (Optional[float], optional): Global upper bound - Only applies to unspecified params. Defaults to 1e5.
        global_lower_bound (Optional[float], optional): Global lower bound - Only applies to unspecified params. Defaults to 1e-6.
        dt0 (float, optional): Integration step width. Defaults to 0.01.
        max_steps (int, optional): Maximum number of integration steps. Defaults to 64**4.
        method (str, optional): Optimization method. Defaults to "bfgs".

    Returns:
        Tuple[MinimizerResult, Model]: Results object and updated model with inferred parameters
    """

    params = _initialize_params(model, global_upper_bound, global_lower_bound)
    observables = jnp.array(
        [
            index
            for index, ode in enumerate(model.odes.values())
            if ode.observable is True
        ]
    )

    assert set(dataset.states) == set(model.get_state_order()), (
        "States in dataset and model do not match."
    )

    # Extract data arrays for the residual computation
    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())

    # Create simulation config from dataset
    config = dataset.to_config()
    config.dt0 = dt0
    config.max_steps = max_steps
    config.nsteps = None

    minimize_args = (
        model,
        dataset,
        config,
        data,
        times,
        observables,
        objective_fun,
    )

    _ipython_print(params)

    result = minimize(
        _residual,
        params,
        args=minimize_args,
        method=method,
    )

    _ipython_flush()
    _ipython_print(result)

    # Update the model with the inferred parameters
    new_model = deepcopy(model)

    for name, parameter in result.params.items():  # type: ignore
        new_model.parameters[name].value = parameter.value

    return (
        result,
        new_model,
    )


def _ipython_print(obj):
    """Print Jupyter things when available"""
    try:
        from IPython.display import display

        display(obj)
    except ImportError:
        pass


def _ipython_flush():
    """Flush Jupyter things when available"""
    try:
        from IPython.display import clear_output

        clear_output()
    except ImportError:
        pass


def _initialize_params(
    model: Model,
    global_upper_bound: Optional[float],
    global_lower_bound: Optional[float],
):
    """Initialize the Parameter object for the optimization."""

    params = Parameters()

    for param in model.parameters.values():
        if param.lower_bound is None:
            assert global_lower_bound is not None, (
                f"Neither a global lower bound nor lower bound for parameter '{param.name}' is given. \
                Please specify either a global lower bound or a lower bound for the parameter."
            )

            param.lower_bound = global_lower_bound

        if param.upper_bound is None:
            assert global_upper_bound is not None, (
                f"Neither a global upper bound nor upper bound for parameter '{param.name}' is given. \
                Please specify either a global upper bound or a upper bound for the parameter."
            )

            param.upper_bound = global_upper_bound

        if param.initial_value is None and param.value is None:
            raise ValueError(
                f"Neither 'value' nor 'initial_value' given for parameter '{param.name}'. \
                    Please add an initial value for the optimization to work."
            )
        elif param.initial_value is None and param.value is not None:
            param.initial_value = param.value

        params.add(
            param.name,
            value=param.initial_value,
            min=param.lower_bound,
            max=param.upper_bound,
            vary=not param.constant,
        )

    return params


def _residual(
    params: Parameters,
    model: Model,
    dataset: Dataset,
    config: SimulationConfig,
    data: jax.Array,
    times: jax.Array,
    observables: jax.Array,
    objective_fun: Callable[[jax.Array, jax.Array], float],
):
    """Performs a simulation of the model and returns the residual between the
    data and the simulation.

    Args:
        params (Parameters): Parameters to optimize.
        model (Model): Model to used to simulate the data.
        dataset (Dataset): Dataset containing initial conditions and time points.
        config (SimulationConfig): Simulation configuration.
        data (jax.Array): Data to fit.
        times (jax.Array): Time points of the data.
        observables (jax.Array): Indices of observable states.
        objective_fun (Callable): Objective function for residual calculation.

    Returns:
        jax.Array: Residuals between data and simulation
    """

    param_dict = {**params.valuesdict()}  # type: ignore
    parameters = jnp.array([param_dict[param] for param in model.get_parameter_order()])

    # Use the new simulate API
    _, states = model.simulate(
        dataset=dataset,
        config=config,
        saveat=times,
        parameters=parameters,
        return_array=True,
    )

    return objective_fun(data, states[:, :, observables])  # type: ignore
