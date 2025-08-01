from typing import Callable, Optional
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from .rbf import RBFLayer


class MLP(eqx.Module):
    mlp: eqx.nn.MLP
    max_time: float

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        use_final_bias: bool,
        activation=jnn.softplus,
        final_activation: Optional[Callable] = None,
        max_time: float = 1.0,
        out_size: Optional[int] = None,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_time = max_time
        self.mlp = eqx.nn.MLP(
            in_size=data_size + 1,
            out_size=out_size if out_size else data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation if final_activation else lambda x: x,
            use_final_bias=use_final_bias,
            key=key,
        )  # type: ignore

        if isinstance(activation, RBFLayer):
            self.mlp = self._mutate_to_rbf(key, activation)

    def __call__(self, t, y, args) -> jax.Array:
        t = t / self.max_time
        y = jnp.concatenate((y, jnp.array([t])), axis=-1)
        return self.mlp(y)

    def _mutate_to_rbf(self, key, rbf):
        """Constructs an RBF layer and replaces the last layer of the MLP with it."""

        new_layers = []
        keys = list(jrandom.split(key, len(self.mlp.layers)))

        for layer in self.mlp.layers[:-1]:
            new_layers += [
                layer,
                RBFLayer(
                    width_size=int(layer.out_features),
                    gamma=rbf.gamma,
                    key=keys.pop(-1),
                ),
            ]

        new_layers.append(self.mlp.layers[-1])

        return eqx.tree_at(
            lambda tree: (tree.layers, tree.activation),
            self.mlp,
            (new_layers, lambda x: x),
        )
