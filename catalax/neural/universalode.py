import json
from typing import List, Optional, Type

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from catalax import Model
from catalax.dataset.dataset import Dataset
from catalax.tools import ODEStack

from .neuralbase import NeuralBase


class UniversalODE(NeuralBase):
    parameters: jax.Array
    vector_field: ODEStack
    alpha_residual: jax.Array
    gate_matrix: jax.Array
    use_gate: bool = eqx.field(static=True)
    model: dict = eqx.field(static=True)

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model | str | dict,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        use_final_bias: bool = False,
        activation=jax.nn.celu,
        final_activation=jax.nn.identity,
        use_gate: bool = True,
        *,
        key,
        **kwargs,
    ):
        if isinstance(model, str):
            model = Model.from_dict(json.loads(model))
        elif isinstance(model, dict):
            model = Model.from_dict(model)

        mlp_key, alpha_key, gate_key = jrandom.split(key, 3)
        state_order = model.get_state_order()
        super().__init__(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            state_order=state_order,
            observable_indices=observable_indices,
            solver=solver,
            activation=activation,
            key=mlp_key,
            use_final_bias=use_final_bias,
            final_activation=final_activation,
            model=model.to_dict(),
        )

        if use_gate and len(state_order) > 1:
            self.use_gate = use_gate
        else:
            self.use_gate = True

        # Resolve assignments before creating the stack
        resolved_model = model._replace_assignments()
        self.model = resolved_model.to_dict()
        self.parameters = jnp.array(resolved_model._get_parameters())
        self.vector_field = eqx.filter_jit(resolved_model.stack)  # type: ignore
        self.alpha_residual = 0.05 * jrandom.normal(alpha_key, (len(state_order),))
        self.gate_matrix = 0.01 * jrandom.normal(
            gate_key,
            (
                len(state_order),
                len(state_order),
            ),
        )

    def _corrective_term(self, t, y, args):
        """Get the corrective term of the model at the given states.

        The corrective term is the neural network contribution to the rates.
        """
        gate = jax.nn.sigmoid(10 * (self.gate_matrix @ y))
        alpha_residual = jax.nn.softplus(self.alpha_residual)
        ann_rates = self.func(t, y, args)
        return alpha_residual * gate * ann_rates

    def _combined_term(self, t, y, args):
        """Merges neural network terms and vector field"""
        mech_rates = self.vector_field(t, y, (self.parameters, None))

        if not self.use_gate:
            return mech_rates + self.func(t, y, args)

        return mech_rates + self._corrective_term(t, y, args)

    def __call__(
        self,
        ts,
        y0,
        solver: Optional[Type[diffrax.AbstractSolver]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        dt0: Optional[float] = None,
    ):
        stepsize_controller = self._create_controller(solver, rtol, atol)
        solver_instance = self._instantiate_solver(solver)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self._combined_term),  # type: ignore
            solver_instance,  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=dt0 or ts[1] - ts[0],
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),  # type: ignore
            max_steps=64**4,
        )
        return solution.ys

    def corrective_term(self, dataset: Dataset) -> tuple[jax.Array, jax.Array]:
        """Get the corrective term of the model at the given states.

        The corrective term is the neural network contribution to the rates.
        """

        data, times, _ = dataset.to_jax_arrays(state_order=self.state_order)
        y = data.reshape(-1, data.shape[-1])
        t = times.ravel()

        corrections = jax.vmap(self._corrective_term, in_axes=(0, 0, None))(t, y, None)
        return corrections.reshape(data.shape), times

    def plot_corrections_over_time(
        self,
        dataset: Dataset,
        show: bool = True,
        path: Optional[str] = None,
        n_steps: int = 100,
        figsize: tuple[float, float] = (10, 3.5),
    ) -> Optional[Figure]:
        # Calculate proper number of rows for odd/even measurements
        uode_pred = self.predict(dataset, n_steps=n_steps)
        corrections, times = self.corrective_term(uode_pred)
        n_meas, _, n_state = corrections.shape

        width, height = figsize

        if n_state == 1:
            # If there is only one state, we can plot in a grid of (n_measurements // 2, 2)
            fig, ax = plt.subplots(
                (n_meas + 1) // 2,
                2,
                figsize=(width, height * ((n_meas + 1) // 2)),
            )
        else:
            fig, ax = plt.subplots(
                n_meas,
                n_state,
                figsize=(width * 0.5 * n_state, height * 0.7 * n_meas),
            )

        ax = ax.flatten()

        color_iter = iter(mcolors.TABLEAU_COLORS)
        state_names = [(state, next(color_iter)) for state in dataset.states]

        ax_index = 0
        for i in range(n_meas):
            corrective_term = corrections[i]

            for j in range(n_state):
                name, color = state_names[j]
                ax[ax_index].plot(
                    times[i],
                    corrective_term[:, j],
                    color=color,
                    linewidth=2,
                    label="Correction",
                )

                ax[ax_index].set_title(f"{name} (Meas. {i + 1})", fontsize=12)
                ax[ax_index].set_xlabel("Time", fontsize=10)
                ax[ax_index].set_ylabel("Corrective Term", fontsize=10)
                ax[ax_index].grid(True, which="both")
                ax[ax_index].grid(True, which="minor", alpha=0.3, linestyle="--")
                ax[ax_index].minorticks_on()
                ax[ax_index].legend(fontsize=10)
                ax[ax_index].set_ylim(
                    jnp.min(corrections[:, :, j]) * 1.2,
                    max(float(jnp.max(corrections[:, :, j]) * 1.2), 0.5),
                )

                ax_index += 1

        # Hide unused subplots when odd number of measurements
        total_subplots = len(ax)
        for i in range(n_meas * n_state, total_subplots):
            ax[i].set_visible(False)

        plt.tight_layout()

        if path:
            plt.savefig(path)

        if show:
            return plt.show()

        return fig

    def plot_corrections_over_input(
        self,
        dataset: Dataset,
        show: bool = True,
        path: Optional[str] = None,
        figsize: tuple[float, float] = (10, 3.5),
    ) -> Optional[Figure]:
        uode_pred = self.predict(dataset, use_times=True)
        data, _, _ = dataset.to_jax_arrays(state_order=self.state_order)
        corrections, _ = self.corrective_term(uode_pred)

        # Unravel data and corrections
        data = data.reshape(-1, data.shape[-1])
        corrections = corrections.reshape(-1, corrections.shape[-1])

        # Get number of state and colors
        n_state = data.shape[-1]
        width, height = figsize
        fig, ax = plt.subplots(n_state, 1, figsize=(width, height * n_state))

        if n_state == 1:
            ax = [ax]

        color_iter = iter(mcolors.TABLEAU_COLORS)
        state_names = [(state, next(color_iter)) for state in dataset.states]

        for i, (state, color) in enumerate(state_names):
            ax[i].scatter(
                data[:, i],
                corrections[:, i],
                color=color,
            )
            ax[i].set_xlabel("Concentration", fontsize=10)
            ax[i].set_ylabel("Corrective Term", fontsize=10)
            ax[i].set_title(f"{state}", fontsize=12)
            ax[i].grid(True, which="both")
            ax[i].grid(True, which="minor", alpha=0.3, linestyle="--")
            ax[i].minorticks_on()

        # Hide unused subplots when odd number of state
        if n_state > 1:
            total_subplots = len(ax)
            for i in range(n_state, total_subplots):
                ax[i].set_visible(False)

        plt.tight_layout()

        if path:
            plt.savefig(path)

        if show:
            return plt.show()

        return fig

    def gate_activation(self, y: jax.Array) -> jax.Array:
        """Get the gate of the model at the given states.

        The gate is a smooth function that determines if the neural network
        contribution is active or not. The gate is triggered by state concentrations
        that flow into a sigmoid function. The row weights of the gate matrix determine
        the sensitivity of the gate to each state. weights can cancel out each other and
        thus cross-talk between states can be controlled.

        Args:
            y: jax.Array of shape (n_state,) or (n_time, n_state)

        Returns:
            Gate
        """

        def _get_gate(y: jax.Array) -> jax.Array:
            return jax.nn.sigmoid(10 * (self.gate_matrix @ y))

        if y.ndim == 1:
            return _get_gate(y)
        elif y.ndim == 2:
            return jax.vmap(_get_gate)(y)
        else:
            raise ValueError("y must be a 1D or 2D array")

    def get_alpha_residual(self) -> jax.Array:
        """Get the alpha residual of the model.

        The alpha residual is a scalar that scales the neural network contribution.
        This way and in combination with L1 regularization can introduce sparsity
        in the neural network contribution. This is particularly useful for symbolic
        regression to recover corrective terms.

        Returns:
            Alpha residual
        """
        return jax.nn.softplus(self.alpha_residual)

    def get_rates(
        self,
        t: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        """Get the rates of the model.

        This basically evaluates the MLP at the given time points and states, which
        is useful for quiver plots and MCMC surrogates.

        Args:
            t: Time points
            y: States

        Returns:
            Rates
        """
        t, y, _ = self._validate_rate_input(t, y, None)
        res: jax.Array = jax.vmap(self._combined_term, in_axes=(0, 0, None))(t, y, None)
        return res
