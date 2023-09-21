from typing import Optional
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx


class RBFLayer(eqx.Module):
    mu: Optional[jax.Array]
    gamma: float

    def __init__(
        self,
        gamma: float,
        width_size: Optional[int] = None,
        *,
        key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if width_size:
            assert key is not None, "Must provide key if width_size is given."
            mukey = jax.random.split(key, 1)
            self.mu = jax.random.uniform(mukey, (width_size,))
        else:
            self.mu = None

        self.gamma = gamma

    def __call__(self, x):
        return jnp.exp(-self.gamma * jnp.square(x - self.mu))
