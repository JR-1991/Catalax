import equinox as eqx
import jax.nn as jnn


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        activation=jnn.softplus,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )  # type: ignore

    def __call__(self, t, y, args):
        return self.mlp(y)
