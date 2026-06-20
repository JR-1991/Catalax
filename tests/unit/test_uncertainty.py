import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from catalax.uncertainty.base import _Z, PredictiveDistribution
from catalax.uncertainty.gapa.activations import make_input, split_mlp


class TestPredictiveDistribution:
    def _moments(self):
        times = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 5), (2, 5))
        y0s = jnp.array([[1.0], [2.0]])
        mean = jnp.ones((2, 5, 1))
        std = jnp.full((2, 5, 1), 0.5)
        return PredictiveDistribution.from_moments(["a"], times, y0s, mean, std)

    def test_moment_band_ordering(self):
        d = self._moments()
        lo, lo50 = d.values("lower"), d.values("lower_50")
        m = d.values(None)
        hi50, hi = d.values("upper_50"), d.values("upper")
        assert jnp.all(lo < lo50)
        assert jnp.all(lo50 < m)
        assert jnp.all(m < hi50)
        assert jnp.all(hi50 < hi)
        # Gaussian bands are symmetric about the mean.
        assert jnp.allclose(m - lo, hi - m)

    def test_moment_band_values(self):
        d = self._moments()
        assert jnp.allclose(d.values("upper"), 1.0 + _Z["upper"] * 0.5)
        assert jnp.allclose(d.values("lower"), 1.0 + _Z["lower"] * 0.5)

    def test_to_dataset_shapes(self):
        d = self._moments()
        data, _, _ = d.to_dataset("upper").to_jax_arrays(["a"])
        assert data.shape == (2, 5, 1)

    def test_samples_backed_bands(self):
        times = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 4), (1, 4))
        y0s = jnp.zeros((1, 1))
        samples = 2.0 + jax.random.normal(jax.random.PRNGKey(0), (200, 1, 4, 1))
        d = PredictiveDistribution.from_samples(["a"], times, y0s, samples)
        assert jnp.allclose(d.mean, jnp.mean(samples, axis=0))
        assert jnp.all(d.values("lower") < d.mean)
        assert jnp.all(d.mean < d.values("upper"))


class TestSplitMLP:
    def test_reconstructs_forward_pass(self):
        """from_postact(phi(to_preact(x))) reproduces the MLP exactly."""
        mlp = eqx.nn.MLP(
            in_size=3, out_size=2, width_size=6, depth=3, key=jax.random.PRNGKey(1)
        )
        x = jnp.array([0.3, -1.2, 0.7])
        n_hidden = len(mlp.layers) - 1
        for layer in range(n_hidden):
            to_preact, from_postact = split_mlp(mlp, layer)
            out = from_postact(mlp.activation(to_preact(x)))
            assert jnp.allclose(out, mlp(x), atol=1e-5)

    def test_preactivation_width(self):
        mlp = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.PRNGKey(2)
        )
        to_preact, _ = split_mlp(mlp, 0)
        assert to_preact(jnp.zeros(2)).shape == (4,)

    def test_invalid_layer_raises(self):
        mlp = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.PRNGKey(3)
        )
        with pytest.raises(ValueError):
            split_mlp(mlp, 5)

    def test_make_input_matches_mlp_convention(self):
        x = make_input(jnp.array(4.0), jnp.array([1.0, 2.0]), max_time=2.0)
        assert jnp.allclose(x, jnp.array([1.0, 2.0, 2.0]))  # [y, t / max_time]
