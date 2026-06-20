"""Decompose a trained MLP around the GAPA layer.

GAPA attaches a Gaussian process to the neurons of a single hidden layer
``l*``. To both *read* that layer's pre-activations and *propagate* the
activation-space variance to the field output, we split the network's forward
pass into two halves at ``l*``:

* ``to_preact(x)`` runs the input up to and including the affine map of layer
  ``l*``, returning the **pre-activation** vector ``u = W^{l*} h^{l*-1} + b``.
* ``from_postact(h)`` continues from the **post-activation** ``h = phi(u)`` of
  layer ``l*`` through the remaining layers to the field output.

The composition ``from_postact(phi(to_preact(x)))`` reproduces the original
forward pass exactly, so nothing about the trained network changes.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def split_mlp(
    mlp,
    layer: int,
) -> Tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """Split an ``equinox.nn.MLP`` into pre- and post-GAPA-layer halves.

    Args:
        mlp: The ``equinox.nn.MLP`` instance (``catalax``'s ``MLP.mlp``).
        layer: Index of the hidden layer the GP is attached to (``0`` is the
            first hidden layer, the default GAPA placement).

    Returns:
        ``(to_preact, from_postact)`` callables operating on flat vectors.
    """
    layers = mlp.layers
    activation = mlp.activation
    final_activation = mlp.final_activation
    n_hidden = len(layers) - 1

    if not 0 <= layer < n_hidden:
        raise ValueError(
            f"GAPA layer index {layer} is out of range for an MLP with "
            f"{n_hidden} hidden layer(s) (valid: 0..{n_hidden - 1})."
        )

    def to_preact(x: jax.Array) -> jax.Array:
        """Run the input to the pre-activation of the GAPA layer."""
        h = x
        for lyr in layers[:layer]:
            h = activation(lyr(h))
        return layers[layer](h)

    def from_postact(h: jax.Array) -> jax.Array:
        """Run the GAPA-layer post-activation through to the field output."""
        for lyr in layers[layer + 1 : -1]:
            h = activation(lyr(h))
        return final_activation(layers[-1](h))

    return to_preact, from_postact


def make_input(t: jax.Array, y: jax.Array, max_time: float) -> jax.Array:
    """Assemble the MLP input ``[y, t / max_time]`` exactly as ``MLP.__call__``."""
    return jnp.concatenate((y, jnp.atleast_1d(t / max_time)), axis=-1)
