from typing import List, Optional, Tuple

import diffrax
import jax
import jax.tree_util as jtn
import jax.random as jrandom
from matplotlib.figure import Figure

from catalax.dataset.dataset import Dataset
from catalax.model.model import Model

from .neuralbase import NeuralBase
from .plots.phaseplot import phase_plot
from .plots.rateflow import plot_learned_rates


class RateFlowODE(NeuralBase):
    reaction_size: int
    stoich_matrix: jax.Array
    learn_stoich: bool

    def __init__(
        self,
        data_size: int,
        reaction_size: int,
        width_size: int,
        depth: int,
        species_order: List[str],
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        activation=jax.nn.softplus,
        use_final_bias: bool = False,
        learn_stoich: bool = True,
        stoich_matrix: jax.Array | None = None,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            data_size=data_size,
            out_size=reaction_size,
            width_size=width_size,
            depth=depth,
            species_order=species_order,
            observable_indices=observable_indices,
            solver=solver,
            activation=activation,
            final_activation=jax.nn.relu,
            key=key,
            use_final_bias=use_final_bias,
            reaction_size=reaction_size,
            learn_stoich=learn_stoich,
        )

        self.reaction_size = reaction_size
        self.learn_stoich = learn_stoich

        if stoich_matrix is None:
            self.stoich_matrix = jrandom.normal(key, (data_size, reaction_size))
        else:
            self.learn_stoich = False
            self.stoich_matrix = stoich_matrix

    def stoich_func(self, t, y, args):
        return self.stoich_matrix @ self.func(t, y, args)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.stoich_func),  # type: ignore
            self.solver(),  # type: ignore
            t0=ts[0],  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),  # type: ignore
            saveat=diffrax.SaveAt(ts=ts),  # type: ignore
        )
        return solution.ys

    def rates(
        self,
        t: jax.Array,
        y: jax.Array,
        constants: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Get the rates of the model.

        This basically evaluates the MLP at the given time points and states, which
        is useful for quiver plots and MCMC surrogates.

        Args:
            t: Time points
            y: States
            constants: Constants, if any

        Returns:
            Rates
        """
        t, y, _ = self._validate_rate_input(t, y, None)
        res: jax.Array = jax.vmap(self.stoich_func, in_axes=(0, 0, None))(t, y, None)
        return res

    def reaction_rates(
        self,
        dataset: Dataset,
    ) -> jax.Array:
        """Get the reaction rates of the model for a given dataset.

        Args:
            dataset: Dataset containing measurements

        Returns:
            Reaction rates
        """
        data, t, _ = dataset.to_jax_arrays(self.species_order)
        data = data.reshape(-1, data.shape[-1])
        t = t.reshape(-1)
        return jax.vmap(self.func, in_axes=(0, 0, None))(t, data, ())

    def get_weights_and_biases(self) -> List[jax.Array]:
        """Get all weights and biases from the model.

        Returns:
            List of weights and biases
        """
        return [
            layer
            for layer in jtn.tree_flatten(self.func)[0]
            if isinstance(layer, jax.Array)
        ]

    def plot_learned_rates(
        self,
        dataset: Dataset,
        model: Model,
        show: bool = True,
        save_path: str | None = None,
        round_stoich: bool = True,
    ) -> Figure:
        """
        Plot learned rates for a trained RateFlowODE.
        """
        return plot_learned_rates(
            rateflow_ode=self,
            dataset=dataset,
            model=model,
            show=show,
            save_path=save_path,
            round_stoich=round_stoich,
        )

    def plot_rate_grid(
        self,
        dataset: Dataset,
        model: Model,
        rate_indices: List[int] | None = None,
        species_identifiers: List[str] | None = None,
        species_pairs: List[Tuple[str | int, str | int]] | None = None,
        representative_time: float = 0.0,
        grid_resolution: int = 30,
        figsize_per_subplot: Tuple[int, int] = (5, 4),
        save_path: str | None = None,
        range_extension: float = 0.2,
        show: bool = True,
    ) -> Figure:
        """
        Create static visualizations showing how dynamics depend on concentration inputs.

        Args:
            dataset: Dataset containing the measurements
            model: Model to use for the analysis
            rate_indices: List of rate indices to plot. If None, plots all rates.
            species_identifiers: List of species to include (by name, symbol, or index).
                            If None, uses all species.
            species_pairs: List of tuples specifying which species pairs to compare.
                          Each tuple should contain two species identifiers (name, symbol, or index).
                          If None, compares all possible pairs from species_identifiers.
            representative_time: Time point to use for the analysis
            grid_resolution: Number of points in each dimension of the concentration grid
            figsize_per_subplot: Size of each subplot (width, height)
            save_path: Path to save the figure
            range_extension: Factor to extend concentration ranges beyond data bounds (0.0-1.0).
                           Default 0.2 extends ranges by 20% above and below data bounds.
                           Lower bound is clamped to 0.0.
        """

        return phase_plot(
            rateflow_ode=self,
            dataset=dataset,
            model=model,
            rate_indices=rate_indices,
            species_identifiers=species_identifiers,
            species_pairs=species_pairs,
            representative_time=representative_time,
            grid_resolution=grid_resolution,
            figsize_per_subplot=figsize_per_subplot,
            save_path=save_path,
            range_extension=range_extension,
            show=show,
        )
