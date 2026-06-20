"""Propagate GAPA activation variance to a trajectory covariance.

Two steps, following the neural-ODE adaptation of GAPA:

1. **Field covariance (delta method).** The diagonal activation-space variance
   ``Sigma_GP(z) = diag(s_d^2(u_d(z)))`` at the GAPA layer is pushed through the
   rest of the field by its Jacobian ``J_post = d f / d h^{l*}``:

       Q(z) = J_post Sigma_GP J_post^T          (a rate-space covariance)

2. **Moment propagation along the ODE.** The state mean follows the original
   (mean-preserving) field and the covariance obeys the Lyapunov differential
   equation driven by the field Jacobian ``J_f`` and the injected ``Q``:

       d z_bar/dt = f(z_bar)
       d P/dt     = J_f P + P J_f^T + Q(z_bar)

The mean and covariance are integrated jointly as one augmented ODE, so a single
solve yields the full mean trajectory and the marginal predictive variance
``diag P(t)``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Type

import diffrax
import jax
import jax.numpy as jnp

from .activations import make_input


def rate_covariance(
    t: jax.Array,
    y: jax.Array,
    *,
    to_preact: Callable[[jax.Array], jax.Array],
    postact_to_rate: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    activation: Callable[[jax.Array], jax.Array],
    var_fn: Callable[[jax.Array], jax.Array],
    max_time: float,
) -> jax.Array:
    """Rate-space covariance ``Q(t, y) = J_post Sigma_GP J_post^T``."""
    x = make_input(t, y, max_time)
    u = to_preact(x)
    h = activation(u)
    sigma2 = var_fn(u)  # (width,) diagonal of Sigma_GP
    J_post = jax.jacfwd(lambda hh: postact_to_rate(hh, t, y))(h)  # (n_state, width)
    return (J_post * sigma2) @ J_post.T


def integrate_with_covariance(
    rate_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    to_preact: Callable[[jax.Array], jax.Array],
    postact_to_rate: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    activation: Callable[[jax.Array], jax.Array],
    var_fn: Callable[[jax.Array], jax.Array],
    max_time: float,
    ts: jax.Array,
    y0: jax.Array,
    P0: jax.Array,
    solver: Type[diffrax.AbstractSolver] = diffrax.Tsit5,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    dt0: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Integrate the augmented (mean, covariance) system for one trajectory.

    Returns:
        ``(mean, var)`` with ``mean`` shaped ``(n_time, n_state)`` and ``var``
        the marginal predictive variance ``diag P(t)`` of the same shape.
    """
    d = y0.shape[0]

    def rhs(t, aug, args):
        y = aug[:d]
        P = aug[d:].reshape(d, d)
        dy = rate_fn(t, y)
        J_f = jax.jacfwd(lambda yy: rate_fn(t, yy))(y)  # (d, d)
        Q = rate_covariance(
            t,
            y,
            to_preact=to_preact,
            postact_to_rate=postact_to_rate,
            activation=activation,
            var_fn=var_fn,
            max_time=max_time,
        )
        dP = J_f @ P + P @ J_f.T + Q
        return jnp.concatenate([dy, dP.reshape(-1)])

    aug0 = jnp.concatenate([y0, P0.reshape(-1)])
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0 if dt0 is not None else ts[1] - ts[0],
        y0=aug0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=16**4,
    )
    aug = solution.ys
    mean = aug[:, :d]
    P = aug[:, d:].reshape(-1, d, d)
    var = jax.vmap(jnp.diag)(P)
    return mean, var
