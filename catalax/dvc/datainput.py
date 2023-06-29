from typing import Dict, List

import jax
import zntrack


class DataInput(zntrack.Node):
    """ZnTrack/DVC node used to store the data for optimiztion.

    Args:
        data (Array): The data to fit the model to.
        times (Array): The times at which the data has been measured.
        initial_conditions (List[Dict[str, float]]): The initial conditions of the model.
    """

    data: jax.Array = zntrack.zn.deps()  # type: ignore
    times: jax.Array = zntrack.zn.deps()  # type: ignore
    initial_conditions: List[Dict[str, float]] = zntrack.zn.deps()  # type: ignore
