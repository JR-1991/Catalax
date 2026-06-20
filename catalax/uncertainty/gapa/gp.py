"""Per-neuron Gaussian-process activation variance (GPJax-backed).

GAPA replaces each neuron's deterministic activation ``phi`` with an independent
1-D Gaussian process whose prior mean *is* ``phi`` (so the posterior mean, and
hence the network's predictions, are unchanged) and whose posterior variance
supplies a closed-form, distance-aware epistemic signal:

    s_d^2(u) = k(u, u) - k(u, Z) [k(Z, Z) + sigma_n^2 I]^{-1} k(Z, u)
                       + k(u, Z) K_zz^{-1} S_d K_zz^{-1} k(Z, u)

where ``Z`` are inducing pre-activations and ``S_d`` is an optional variational
covariance (held at zero in the current variants, reserved as a hook for a full
sparse-variational correction). The kernel is the GPJax RBF. Hyperparameters follow the paper
(arXiv:2502.20966): empirical lengthscale = 0.25-quantile of pairwise
pre-activation distances, signal variance = ``max(floor, Var(phi(U)))``,
inducing points quantile-spaced along the empirical CDF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from gpjax.kernels import RBF

#: Jitter added to the inducing Gram matrix for a stable solve.
_JITTER = 1e-6


def empirical_lengthscale(
    u_col: np.ndarray, q: float = 0.25, max_points: int = 512
) -> float:
    """Empirical lengthscale: the ``q``-quantile of pairwise abs distances.

    Computed once, at setup time, on (a subsample of) the cached pre-activations
    of a single neuron. A subsample bounds the O(N^2) pairwise computation.
    """
    u = np.asarray(u_col).ravel()
    if u.shape[0] > max_points:
        idx = np.linspace(0, u.shape[0] - 1, max_points).astype(int)
        u = u[idx]
    diffs = np.abs(u[:, None] - u[None, :])
    iu = np.triu_indices(u.shape[0], k=1)
    pairwise = diffs[iu]
    if pairwise.size == 0:
        return 1.0
    ell = float(np.quantile(pairwise, q))
    # Guard against degenerate (near-constant) pre-activations.
    return ell if ell > 1e-6 else 1.0


def inducing_points(u_col: np.ndarray, n_inducing: int) -> np.ndarray:
    """Quantile-spaced inducing inputs along the empirical CDF of ``u_col``.

    Always includes the min and max; the remaining ``M - 2`` points sit at
    quantile levels ``p_m = (m + 1) / (M - 1)`` (arXiv:2502.20966, appendix).
    """
    u = np.unique(np.asarray(u_col).ravel())
    m = int(min(n_inducing, u.shape[0]))
    if m <= 2:
        return np.array([u.min(), u.max()], dtype=float)
    interior = [(j + 1) / (m - 1) for j in range(m - 2)]
    quantiles = np.quantile(u, interior)
    pts = np.concatenate([[u.min()], quantiles, [u.max()]])
    return np.unique(pts).astype(float)


def fit_empirical(
    pre_activations: np.ndarray,
    activation,
    *,
    n_inducing: int = 40,
    signal_var_floor: float = 1.0,
    quantile: float = 0.25,
):
    """Empirical (``GAPA-Free``) per-neuron hyperparameters and inducing inputs.

    Args:
        pre_activations: Cached pre-activations, shape ``(n_points, width)``.
        activation: The network's hidden activation ``phi``.
        n_inducing: Number of inducing points per neuron.
        signal_var_floor: Lower bound on the signal variance (paper uses 1.0).
        quantile: Quantile for the lengthscale heuristic.

    Returns:
        ``(Z, log_lengthscale, log_signal_var)`` with ``Z`` padded to a uniform
        ``(width, M)`` array and the log-params shaped ``(width,)``.
    """
    u = np.asarray(pre_activations)
    width = u.shape[1]

    lengthscales = np.array(
        [empirical_lengthscale(u[:, d], q=quantile) for d in range(width)]
    )

    acts = np.asarray(activation(jnp.asarray(u)))
    signal_vars = np.array(
        [max(signal_var_floor, float(np.var(acts[:, d]))) for d in range(width)]
    )

    z_cols = [inducing_points(u[:, d], n_inducing) for d in range(width)]
    m_max = max(z.shape[0] for z in z_cols)
    # Pad shorter inducing sets by repeating the last point; duplicate inputs
    # only add jitter-regularised redundancy and never reduce the variance.
    Z = np.stack(
        [np.pad(z, (0, m_max - z.shape[0]), mode="edge") for z in z_cols], axis=0
    )

    return (
        jnp.asarray(Z),
        jnp.log(jnp.asarray(lengthscales)),
        jnp.log(jnp.asarray(signal_vars)),
    )


def _neuron_variance(
    log_lengthscale: jax.Array,
    log_signal_var: jax.Array,
    Z: jax.Array,
    var_chol: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """GP posterior variance for one neuron at a scalar pre-activation ``u``."""
    kernel = RBF(
        lengthscale=jnp.exp(log_lengthscale),
        variance=jnp.exp(log_signal_var),
        n_dims=1,
    )
    Zc = Z[:, None]
    m = Z.shape[0]
    Kzz = kernel.cross_covariance(Zc, Zc) + _JITTER * jnp.eye(m)
    kzu = kernel.cross_covariance(Zc, u.reshape(1, 1))[:, 0]  # (M,)
    B = jnp.linalg.solve(Kzz, kzu)  # K_zz^{-1} k(Z, u)
    kuu = jnp.exp(log_signal_var)  # RBF: k(u, u) = signal variance
    var = kuu - kzu @ B
    # Variational correction k(u,Z) K_zz^{-1} S K_zz^{-1} k(Z,u) with S = L L^T.
    LB = var_chol.T @ B
    var = var + LB @ LB
    return jnp.maximum(var, 0.0)


def variance_fn(log_lengthscale, log_signal_var, Z, var_chol):
    """Build a vectorised ``u -> diag(Sigma_GP)`` closure over all neurons.

    Args:
        log_lengthscale: ``(width,)`` log lengthscales.
        log_signal_var: ``(width,)`` log signal variances.
        Z: ``(width, M)`` inducing pre-activations.
        var_chol: ``(width, M, M)`` Cholesky factors of the variational
            covariance (zeros for the empirical variant).

    Returns:
        A function mapping a ``(width,)`` pre-activation vector to the
        ``(width,)`` diagonal of the activation-space covariance.
    """

    def fn(u: jax.Array) -> jax.Array:
        return jax.vmap(_neuron_variance, in_axes=(0, 0, 0, 0, 0))(
            log_lengthscale, log_signal_var, Z, var_chol, u
        )

    return fn
