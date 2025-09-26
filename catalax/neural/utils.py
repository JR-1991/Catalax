import json
from textwrap import dedent

import equinox as eqx
import jax

# We need to keep track of the activation functions that are supported
# because when writing to eqx files, we cannot serialize custom functions
# Hence, when an activation function is not in this map, we will raise a warning
# once saving to eqx
ACTIVATION_MAP = {
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "relu": jax.nn.relu,
    "celu": jax.nn.celu,
    "selu": jax.nn.selu,
    "silu": jax.nn.silu,
    "soft_sign": jax.nn.soft_sign,
    "mish": jax.nn.mish,
    "swish": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "elu": jax.nn.elu,
    "leaky_relu": jax.nn.leaky_relu,
    "hard_swish": jax.nn.hard_swish,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "hard_tanh": jax.nn.hard_tanh,
    "identity": jax.nn.identity,
}

INVERSE_ACTIVATION_MAP = {v: k for k, v in ACTIVATION_MAP.items()}

SERIALISATION_WARNING = dedent("""
Warning: Custom activation functions detected. EQX serialisation cannot persist custom activation functions automatically. 

You can still use custom activation functions, but when saving/loading models:
- The model weights will be saved correctly
- The activation function itself won't be serialized
- When loading, you'll need to recreate the model with the same custom activation function

For automatic serialization support, consider using supported functions listed at:
'https://github.com/JR-1991/Catalax/blob/master/catalax/neural/utils.py'

To persist models with custom activations, use the 'load_weights' method and ensure your model architecture code is available when loading.
""")


def _get_activation_name(activation_func) -> str:
    """Get the string name of an activation function, handling JIT-compiled functions.

    This function handles both regular and JIT-compiled activation functions by trying
    multiple approaches to find the correct mapping in INVERSE_ACTIVATION_MAP.

    Custom activation functions are allowed but won't be serialized automatically.

    Args:
        activation_func: The activation function to look up

    Returns:
        The string name of the activation function

    Raises:
        KeyError: If the activation function is not found in the mapping (indicates custom function)
    """
    # Try direct lookup first (works for most cases)
    if activation_func in INVERSE_ACTIVATION_MAP:
        return INVERSE_ACTIVATION_MAP[activation_func]

    # If direct lookup fails, try unwrapping JIT-compiled functions
    if hasattr(activation_func, "__wrapped__"):
        wrapped_func = activation_func.__wrapped__
        if wrapped_func in INVERSE_ACTIVATION_MAP:
            return INVERSE_ACTIVATION_MAP[wrapped_func]

    # Custom activation function detected - this is allowed but won't be serialized
    raise KeyError(
        f"Custom activation function detected: {activation_func}. "
        f"This is allowed but won't be automatically serialized. "
        f"Supported functions for auto-serialization: {list(ACTIVATION_MAP.keys())}"
    )


def save(filename, hyperparams, model):
    """Save a model with hyperparameters to an eqx file."""
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams, default=str)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
