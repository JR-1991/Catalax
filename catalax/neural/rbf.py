from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


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
            mukey = jax.random.split(key, 1)[0]
            self.mu = jax.random.uniform(mukey, (width_size,))
        else:
            self.mu = None

        self.gamma = gamma

    def __call__(self, x):
        # For vector inputs, compute pairwise distances
        # x shape: (batch_size, input_dim)
        # mu shape: (n_centers, input_dim)
        # Broadcast to compute distances between all pairs
        assert self.mu is not None, "Mu must be set before calling the RBF layer."
        diff = jnp.expand_dims(x, axis=1) - jnp.expand_dims(self.mu, axis=0)
        sq_dist = jnp.sum(jnp.square(diff), axis=-1)
        return jnp.exp(-self.gamma * sq_dist)
