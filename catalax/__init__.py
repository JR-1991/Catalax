import arviz as az
from sympy import Symbol  # noqa: F401

from .dataset import Dataset, Measurement
from .model import InAxes, Model, SimulationConfig
from .objectives import l1_loss, mean_absolute_error
from .tools.enzymeml import dataset_and_model_from_enzymeml as from_enzymeml
from .tools.optimization import optimize

# Set plotting style
az.style.use("arviz-doc")  # type: ignore

__all__ = [
    "SimulationConfig",
    "Dataset",
    "Measurement",
    "InAxes",
    "Model",
    "optimize",
    "mean_absolute_error",
    "l1_loss",
    "from_enzymeml",
]

__version__ = "0.5.0"

PARAMETERS = InAxes.PARAMETERS
TIME = InAxes.TIME
INITS = InAxes.Y0


def set_host_count(n: int = 1):
    """
    Sets the number of hosts to be used by JAX for parallel execution.

    Args:
        n (int): The number of hosts to use. Defaults to 1.
    """
    import numpyro

    numpyro.set_host_device_count(n)


def set_platform(platform: str = "cpu"):
    """
    Sets the platform for JAX.

    Args:
        platform (str): The platform to use. Must be one of 'cpu' or 'gpu'. Defaults to 'cpu'.

    Raises:
        AssertionError: If the platform is not 'cpu' or 'gpu'.
    """
    import numpyro

    assert platform in ["cpu", "gpu"], "platform must be one of 'cpu' or 'gpu'"

    numpyro.set_platform(platform)


def enable_x64(use_x64: bool = True):
    """
    Enables the use of 64-bit precision in JAX.

    Args:
        use_x64 (bool, optional): Whether to enable 64-bit precision. Defaults to True.
    """
    import numpyro

    numpyro.enable_x64(use_x64=use_x64)
