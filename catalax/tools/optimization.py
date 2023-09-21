import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from catalax.model import Model


def optimize(
    model: Model,
    initial_conditions: List[Dict[str, float]],
    data: jax.Array,
    times: jax.Array,
    global_upper_bound: Optional[float] = 1e5,
    global_lower_bound: Optional[float] = 1e-6,
    dt0: float = 0.01,
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
        initial_conditions (List[Dict[str, float]]): List of initial condition objects.
        data (jax.Array): Data to fit against.
        times (jax.Array): Time steps of the data.
        global_upper_bound (Optional[float], optional): Global upper bound - Only applies to unspecified params. Defaults to 1e5.
        global_lower_bound (Optional[float], optional): Global lower bound - Only applies to unspecified params. Defaults to 1e-6.
        dt0 (float, optional): Inetgration step width. Defaults to 0.01.
        max_steps (int, optional): Maximum number of integration steps. Defaults to 64**4.
        method (str, optional): _description_. Defaults to "bfgs".

    Returns:
        _type_: _description_
    """

    params = _initialize_params(model, global_upper_bound, global_lower_bound)
    observables = jnp.array(
        [
            index
            for index, ode in enumerate(model.odes.values())
            if ode.observable == True
        ]
    )
    minimize_args = (
        model,
        initial_conditions,
        data,
        times,
        observables,
        dt0,
        max_steps,
    )

    _iPython_print(params)

    result = minimize(
        _residual,
        params,
        args=minimize_args,
        method=method,
    )

    _iPython_flush()
    _iPython_print(result)

    # Update the model with the inferred parameters
    new_model = deepcopy(model)

    for name, parameter in result.params.items():  # type: ignore
        new_model.parameters[name].value = parameter.value

    return (
        result,
        new_model,
    )


def _iPython_print(obj):
    """Print Jupyter things when available"""
    try:
        from IPython.display import display

        display(obj)
    except ImportError:
        pass


def _iPython_flush():
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
            assert (
                global_lower_bound is not None
            ), f"Neither a global lower bound nor lower bound for parameter '{param.name}' is given. \
                Please specify either a global lower bound or a lower bound for the parameter."

            param.lower_bound = global_lower_bound

        if param.upper_bound is None:
            assert (
                global_upper_bound is not None
            ), f"Neither a global upper bound nor upper bound for parameter '{param.name}' is given. \
                Please specify either a global upper bound or a upper bound for the parameter."

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
    y0s: List[Dict[str, float]],
    data: jax.Array,
    times: jax.Array,
    observables: jax.Array,
    dt0: float = 0.01,
    max_steps: int = 64**4,
):
    """Performs a simulation of the model and returns the residual between the
    data and the simulation.

    Args:
        model (Model): Model to used to simulate the data.
        params (Parameters): Parameters to optimize.
        y0s (List[Dict[str, float]]): _description_
        data (jax.Array): Data to fit.
        times (jax.Array): Time points of the data.
        dt0 (float, optional): Time step of the simulation. Defaults to 0.01.
        max_steps (int, optional): Maximum number of integration steps. Defaults to 64**4.

    Returns:
        jax.Array: Residuals between data and simulation
    """

    params = {**params.valuesdict()}  # type: ignore
    parameters = jnp.array([params[param] for param in model._get_parameter_order()])

    _, states = model.simulate(
        initial_conditions=y0s,
        dt0=dt0,
        in_axes=(0, None, 0),
        saveat=times,  # type: ignore
        max_steps=max_steps,
        parameters=parameters,
    )

    return data - states[:, :, observables]
