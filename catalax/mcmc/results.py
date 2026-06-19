from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpyro
import pandas as pd
import xarray
import jax.numpy as jnp
from jax import Array

from catalax.mcmc.mcmc import MCMC
from catalax.mcmc.plotting import plot_corner, plot_ess, plot_mcse, plot_posterior
from catalax.predictor import Predictor

if TYPE_CHECKING:
    from catalax.dataset.dataset import Dataset
    from catalax.mcmc import BayesianModel
    from catalax.mcmc.loo import Integration, LeaveOut, SigmaSource
    from catalax.mcmc.predictive import HDILevel
    from catalax.model.model import Model
    from catalax.model.simconfig import SimulationConfig

try:
    import ipywidgets as widgets
    from IPython.core.getipython import get_ipython
    from IPython.display import display

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


class HMCResults(Predictor):
    """Results container for HMC sampling with integrated visualization methods.

    This class provides a comprehensive interface for analyzing and visualizing
    the results of Hamiltonian Monte Carlo (HMC) sampling. It wraps the MCMC
    object and provides convenient methods for diagnostics, plotting, and
    exporting results.

    The class integrates with ArviZ for advanced diagnostics and provides
    methods for generating various plots including corner plots, trace plots,
    posterior distributions, and credibility intervals.

    Attributes:
        mcmc: The MCMC object containing posterior samples
        bayesian_model: The Bayesian model used for sampling
        model: The original mechanistic model that was fit
    """

    def __init__(
        self,
        mcmc: MCMC,
        bayesian_model: BayesianModel,
        model: "Model",
    ):
        """Initialize HMC results.

        Args:
            mcmc: MCMC object containing samples and sampling metadata
            bayesian_model: Bayesian model used for sampling, containing
                prior definitions and likelihood specification
            model: Original mechanistic model that was fit to data
        """
        self.mcmc = mcmc
        self.bayesian_model = bayesian_model
        self.model = model

        # Defaults for posterior-predictive plotting via the Predictor interface.
        # Tunable on the instance, e.g. ``results.predictive_include_noise = False``
        # for an epistemic-only (parameter-uncertainty) band.
        self.predictive_max_draws: Optional[int] = 200
        self.predictive_integration: "Integration" = "euler"
        self.predictive_include_noise: bool = True
        self._ensemble_cache: Dict = {}

    @property
    def divergences(self) -> Array:
        """Boolean mask of diverging HMC transitions, shape ``(n_chains, n_draws)``.

        ``True`` marks draws where NUTS reported a divergence (a sign the sampler
        could not faithfully integrate the posterior geometry there). Sum it for
        the total, or use :attr:`num_divergences`. Empty array if the sampler was
        run without collecting the ``diverging`` field.
        """
        extra = self.mcmc.get_extra_fields(group_by_chain=True)
        diverging = extra.get("diverging")
        if diverging is None:
            return jnp.zeros((0,), dtype=bool)
        return diverging

    @property
    def num_divergences(self) -> int:
        """Total number of diverging transitions across all chains."""
        return int(jnp.sum(self.divergences))

    @property
    def diverging_samples(self) -> pd.DataFrame:
        """Parameter values at the diverging draws only, as a DataFrame.

        One row per diverging transition, with ``chain``/``draw`` indices and a
        column per parameter (vector sites are expanded to ``name[0]``,
        ``name[1]`` ...). This pinpoints *where* in parameter space the sampler
        diverged -- e.g. a noise scale collapsing toward 0. Empty if there were
        no divergences (or the field was not collected).
        """
        import numpy as np

        diverging = np.asarray(self.divergences)
        samples = self.mcmc.get_samples(group_by_chain=True)
        if diverging.size == 0 or not diverging.any():
            return pd.DataFrame()

        chain_idx, draw_idx = np.nonzero(diverging)
        columns: Dict[str, Array] = {"chain": chain_idx, "draw": draw_idx}
        for name, arr in samples.items():
            selected = np.asarray(arr)[diverging]  # (n_div, *event_shape)
            if selected.ndim == 1:
                columns[name] = selected
            else:
                flat = selected.reshape(selected.shape[0], -1)
                for k in range(flat.shape[1]):
                    columns[f"{name}[{k}]"] = flat[:, k]
        return pd.DataFrame(columns)

    # ------------------------------------------------------------------ #
    # Predictor interface -- lets an HMCResults be passed anywhere a
    # Predictor is accepted (e.g. ``dataset.plot(predictor=results)``),
    # producing an honest posterior-predictive band: the full joint
    # posterior pushed through the model, quantiled in trajectory space.
    # ------------------------------------------------------------------ #
    def get_state_order(self) -> List[str]:
        """Return the model's state order (see :meth:`Model.get_state_order`)."""
        return self.model.get_state_order()

    def get_species_order(self) -> List[str]:
        """Deprecated alias for :meth:`get_state_order`."""
        return self.model.get_state_order()

    def n_parameters(self) -> int:
        """Number of model parameters carried by the posterior."""
        return self.model.n_parameters()

    def has_hdi(self) -> bool:
        """Always ``True`` -- the posterior draws are always available."""
        return True

    def predict(
        self,
        dataset: "Dataset",
        config: Optional["SimulationConfig"] = None,
        n_steps: int = 100,
        use_times: bool = False,
        hdi: Optional["HDILevel"] = None,
        max_draws: Optional[int] = ...,  # type: ignore[assignment]
        integration: Optional["Integration"] = None,
        include_noise: Optional[bool] = None,
    ) -> "Dataset":
        """Posterior-predictive trajectory (``hdi=None``) or credible band.

        Pushes the full joint posterior through the model and takes per
        ``(state, time)`` quantiles in trajectory space -- never the
        marginal-corner shortcut. With ``include_noise`` the band is a true
        predictive interval (aleatoric noise folded in): the fit's concentration
        ``sigma`` in mechanistic mode, or the rate-space noise forward-propagated
        through the Euler integrator in surrogate (rate-matching) mode.

        The expensive per-draw integration is cached, so the five calls
        :meth:`Dataset.plot` makes (median + four HDI bands) integrate the
        posterior only once.

        Args:
            dataset: Provides the initial conditions / measurement times.
            config: Accepted for interface compatibility; unused.
            n_steps: Dense time-grid resolution (mechanistic mode; surrogate mode
                is fixed to the measurement times).
            use_times: Accepted for interface compatibility; the grid is always
                dense (mechanistic) or the measurement times (surrogate).
            hdi: Band selector (``None`` -> median trajectory).
            max_draws: Posterior subsample size; defaults to
                ``self.predictive_max_draws``.
            integration: Surrogate-mode integrator; defaults to
                ``self.predictive_integration``.
            include_noise: Predictive (``True``) vs epistemic-only band; defaults
                to ``self.predictive_include_noise``.

        Returns:
            Dataset of the requested trajectory/band over the observable states.
        """
        from catalax.mcmc.predictive import band_from_ensemble, compute_ensemble

        if max_draws is ...:
            max_draws = self.predictive_max_draws
        if integration is None:
            integration = self.predictive_integration
        if include_noise is None:
            include_noise = self.predictive_include_noise

        # Cache the (expensive) ensemble so repeated band slices are cheap.
        cache_key = (id(dataset), n_steps, max_draws, integration)
        if cache_key not in self._ensemble_cache:
            self._ensemble_cache[cache_key] = compute_ensemble(
                self,
                dataset,
                n_steps=n_steps,
                max_draws=max_draws,
                integration=integration,
            )
        times, yhat_ens, var_ens = self._ensemble_cache[cache_key]

        return band_from_ensemble(
            self,
            dataset,
            times,
            yhat_ens,
            var_ens,
            hdi=hdi,
            include_noise=include_noise,
        )

    def posterior_predictive_ensemble(
        self,
        dataset: "Dataset",
        n_steps: int = 100,
        max_draws: Optional[int] = ...,  # type: ignore[assignment]
        integration: Optional["Integration"] = None,
    ):
        """Raw posterior-predictive trajectories, one per draw (for spaghetti plots).

        Shares the cache with :meth:`predict`, so requesting bands and the draw
        ensemble of the same fit integrates the posterior only once.

        Returns:
            ``(times, values, obs_states)`` with ``times`` of shape ``(n_meas,
            n_time)``, ``values`` (the integrated trajectories, no aleatoric
            noise) of shape ``(n_draws, n_meas, n_time, n_obs)`` and
            ``obs_states`` the observable state names.
        """
        import numpy as np

        from catalax.mcmc.predictive import compute_ensemble

        if max_draws is ...:
            max_draws = self.predictive_max_draws
        if integration is None:
            integration = self.predictive_integration

        cache_key = (id(dataset), n_steps, max_draws, integration)
        if cache_key not in self._ensemble_cache:
            self._ensemble_cache[cache_key] = compute_ensemble(
                self,
                dataset,
                n_steps=n_steps,
                max_draws=max_draws,
                integration=integration,
            )
        times, yhat_ens, _ = self._ensemble_cache[cache_key]
        return (
            np.asarray(times),
            np.asarray(yhat_ens),
            list(self.model.get_observable_state_order()),
        )

    def get_fitted_model(
        self,
        hdi_prob: float = 0.95,
        set_bounds: bool = False,
    ) -> "Model":
        """Get the fitted model with parameter estimates from posterior samples.

        Creates a new model instance with parameters set to posterior estimates.
        Parameter values are typically set to posterior means or medians, with
        uncertainty bounds optionally derived from highest density intervals.

        Args:
            hdi_prob: Probability mass for highest density interval calculation.
                Used to determine uncertainty bounds for parameters.
            set_bounds: Whether to set parameter bounds based on HDI. If True,
                parameter bounds will be updated to reflect posterior uncertainty.

        Returns:
            Model: New model instance with parameters updated from posterior samples.
                The returned model can be used for predictions with fitted parameters.
        """
        return self.model.from_arviz(
            az.from_numpyro(self.mcmc),
            hdi_prob=hdi_prob,
            set_bounds=set_bounds,
        )

    def get_samples(self) -> Dict[str, Array]:
        """Get posterior samples for all parameters.

        Returns the raw posterior samples as generated by the MCMC sampler.
        These samples represent the posterior distribution over model parameters
        and can be used for custom analysis or uncertainty quantification.

        Returns:
            Dict[str, Array]: Dictionary mapping parameter names to arrays of
                posterior samples. Each array has shape (num_samples, ...) where
                the parameter dimensions depend on the parameter structure.
        """
        return self.mcmc.get_samples()

    def print_summary(self):
        """Print MCMC summary statistics to console.

        Displays a tabular summary of the MCMC results including:
        - Parameter posterior means and standard deviations
        - Effective sample size (ESS) for each parameter
        - R-hat convergence diagnostics
        - Monte Carlo standard error estimates

        This provides a quick overview of sampling quality and parameter estimates.
        """
        self.mcmc.print_summary()

    def render_model(
        self,
        path: str,
        render_distributions: bool = False,
    ):
        """Render the Bayesian model structure as a graphical representation.

        Creates a visual representation of the model's probabilistic structure,
        showing the relationships between parameters, priors, and likelihood.
        Useful for understanding model structure and debugging.

        Args:
            path: File path where the rendered model graph will be saved.
                Should include appropriate file extension (e.g., '.pdf', '.png').
            render_distributions: Whether to include distribution shapes in the
                rendered graph. If True, adds visual representation of prior
                and likelihood distributions.

        Returns:
            The result of numpyro.render_model, typically None but may vary
            based on the rendering backend used.
        """
        return numpyro.render_model(
            self.bayesian_model,
            filename=path,
            render_distributions=render_distributions,
        )

    # Visualization methods from plotting.py
    def plot_corner(
        self,
        quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Plot corner plot of parameter correlations and marginal distributions.

        Creates a matrix plot showing pairwise parameter correlations in the
        lower triangle, marginal posterior distributions on the diagonal, and
        optionally scatter plots in the upper triangle. This is essential for
        understanding parameter dependencies and posterior structure.

        Args:
            quantiles: Three quantiles to display in marginal distributions,
                typically representing lower bound, median, and upper bound.
                Default (0.16, 0.5, 0.84) corresponds to 68% credible intervals.
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.

        Returns:
            matplotlib.figure.Figure or bokeh plot: The corner plot figure, or None if show=True.
                Can be used for further customization or saving.
        """

        f = plot_corner(self.mcmc, quantiles, figsize, backend)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None

        return plt.gcf()

    def plot_posterior(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Plot posterior distributions for all parameters.

        Creates individual plots showing the posterior distribution for each
        model parameter. This provides a clear view of parameter uncertainty
        and can help identify multimodal distributions or convergence issues.

        Args:
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.
            **kwargs: Additional keyword arguments passed to arviz.plot_posterior.
                Common options include 'hdi_prob' for credible interval probability,
                'point_estimate' for central tendency measure, and 'ref_val' for
                reference values.

        Returns:
            matplotlib.figure.Figure or bokeh plot: The posterior plot figure, or None if show=True.
                Contains subplots for each parameter's posterior distribution.
        """

        f = plot_posterior(self.mcmc, self.model, figsize, backend, **kwargs)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None

        return plt.gcf()

    def plot_credibility_interval(
        self,
        initial_condition: Dict[str, float],
        time: Array,
        dt0: float = 0.1,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Plot credibility intervals for model simulations.

        Generates model predictions using posterior parameter samples and plots
        credibility intervals around the mean trajectory. This shows prediction
        uncertainty arising from parameter uncertainty and is crucial for
        understanding model reliability.

        Args:
            initial_condition: Dictionary mapping state names to initial
                concentrations for the simulation. Must include all state
                defined in the model.
            time: Array of time points at which to evaluate the model.
                Should span the desired simulation duration.
            dt0: Time step size for numerical integration. Smaller values
                increase accuracy but require more computation.
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
                Note: Credibility interval plots currently only support matplotlib backend.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.

        Returns:
            matplotlib.figure.Figure: The credibility interval plot figure,
                or None if show=True. Shows model trajectories with uncertainty bands.
        """
        from catalax.mcmc.plotting import plot_credibility_interval

        f = plot_credibility_interval(
            self.mcmc, self.model, initial_condition, time, dt0, figsize, backend
        )
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None
        return plt.gcf()

    def plot_trace(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Plot MCMC trace plots for convergence diagnostics.

        Shows the evolution of parameter values across MCMC iterations for
        each chain. Essential for diagnosing convergence issues, identifying
        burn-in periods, and detecting problematic sampling behavior such
        as poor mixing or trend.

        Args:
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.
            **kwargs: Additional keyword arguments passed to arviz.plot_trace.
                Common options include 'var_names' to select specific parameters,
                'compact' for layout style, and 'combined' for overlaying chains.

        Returns:
            matplotlib.figure.Figure or bokeh plot: The trace plot figure, or None if show=True.
                Contains time series plots for each parameter across all chains.
        """
        from catalax.mcmc.plotting import plot_trace

        f = plot_trace(self.mcmc, self.model, figsize, backend, **kwargs)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()

    def plot_forest(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Plot forest plot of parameter distributions.

        Creates a horizontal plot showing credible intervals for all parameters
        in a compact format. Useful for comparing parameter estimates and
        uncertainties across many parameters simultaneously.

        Args:
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.
            **kwargs: Additional keyword arguments passed to arviz.plot_forest.
                Common options include 'hdi_prob' for credible interval probability,
                'combined' for pooling chains, and 'ess' for showing effective
                sample size.

        Returns:
            matplotlib.figure.Figure or bokeh plot: The forest plot figure, or None if show=True.
                Shows horizontal credible intervals for each parameter.
        """
        from catalax.mcmc.plotting import plot_forest

        f = plot_forest(self.mcmc, self.model, figsize, backend, **kwargs)
        if path is not None:
            f.savefig(path)

        if show:
            plt.show()
            return None

        return plt.gcf()

    def summary(self, hdi_prob: float = 0.95) -> Union[pd.DataFrame, xarray.Dataset]:
        """Generate comprehensive summary statistics for posterior samples.

        Computes and returns detailed summary statistics including central
        tendencies, credible intervals, effective sample sizes, and convergence
        diagnostics. This provides a comprehensive numerical summary of the
        MCMC results.

        Args:
            hdi_prob: Probability mass for highest density interval calculation.
                Determines the width of credible intervals in the summary.
                Default 0.95 corresponds to 95% credible intervals.

        Returns:
            Union[pd.DataFrame, xarray.Dataset]: Summary statistics table
                containing means, standard deviations, credible intervals,
                effective sample sizes, and R-hat diagnostics for each parameter.
                Format depends on the underlying implementation.
        """
        from catalax.mcmc.plotting import summary

        return summary(self.mcmc, hdi_prob)

    def plot_mcse(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Plot Monte Carlo standard error diagnostics.

        Visualizes the Monte Carlo standard error (MCSE) for parameter estimates,
        which quantifies the uncertainty in posterior estimates due to finite
        sample size. Lower MCSE indicates more reliable estimates.

        Args:
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.
            **kwargs: Additional keyword arguments (currently unused but
                maintained for API consistency).

        Returns:
            matplotlib.figure.Figure or bokeh plot: The MCSE plot figure, or None if show=True.
                Shows Monte Carlo standard error for each parameter.
        """
        f = plot_mcse(self.mcmc, figsize, backend)
        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None
        return plt.gcf()

    def plot_ess(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        show: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Plot effective sample size diagnostics.

        Visualizes the effective sample size (ESS) for each parameter, which
        measures how many independent samples the MCMC chains provide. Higher
        ESS indicates better sampling efficiency and more reliable estimates.

        Args:
            figsize: Figure size as (width, height) in inches. If None, uses default size.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            show: Whether to display the plot immediately using plt.show().
                If False, the figure is returned without displaying.
            path: Optional file path to save the plot. If provided, the figure
                will be saved to this location.
            **kwargs: Additional keyword arguments (currently unused but
                maintained for API consistency).

        Returns:
            matplotlib.figure.Figure or bokeh plot: The ESS plot figure, or None if show=True.
                Shows effective sample size for each parameter.
        """
        f = plot_ess(self.mcmc, figsize, backend)

        if path is not None:
            f.savefig(path)
        if show:
            plt.show()
            return None
        return plt.gcf()

    def loo(
        self,
        dataset: "Dataset",
        *,
        yerrs: Optional[Union[float, Array]] = None,
        sigma_source: "SigmaSource" = "reuse",
        leave_out: "LeaveOut" = "point",
        integration: "Integration" = "euler",
        pointwise: bool = True,
        reloo: bool = False,
        max_draws: Optional[int] = None,
        config: Optional["SimulationConfig"] = None,
        scale: str = "log",
    ):
        """Concentration-space leave-one-out cross-validation (LOO-CV).

        Computes LOO whose held-out unit is a *real concentration measurement*,
        scored against the integrated trajectory -- valid whether the posterior
        came from mechanistic integration or from a rate-space surrogate
        (RM-NLL) fit. It reuses the fit's own likelihood: for surrogate fits the
        sampled rates ``v(yi; theta)`` are Euler-integrated along the measurement
        times and the reused rate noise is pushed forward to concentration space;
        for mechanistic fits the integrated states are scored directly
        (reproducing the native ArviZ statistic).

        Because ``HMCResults`` does not store the training data, the measured
        ``dataset`` (and, where relevant, ``yerrs``) must be supplied here.

        Args:
            dataset: Real concentration measurements to validate against.
            sigma_source: ``"reuse"`` (default) uses the fit's inferred ``sigma_y``;
                against the supplied measurement error (the prediction still
                comes from the sampled rates); ``"reuse"`` instead takes the noise
                from the sampled ``sigma`` (pushed forward through the
                integration in surrogate mode).
            yerrs: Concentration-space measurement error; used when
                ``sigma_source="yerrs"`` (ignored under ``"reuse"``). ``None``
                falls back to the error stored on the fit (mechanistic only).
            leave_out: ``"point"`` (one species/timepoint) or ``"curve"``
                (a whole measurement series -- "predict a new experiment").
            integration: Surrogate-mode integration scheme -- ``"euler"`` (global
                forward Euler) or ``"euler_onestep"`` (one-step-ahead from each
                measured state).
            pointwise: Return per-observation Pareto-k diagnostics, which flag
                high-influence observations.
            reloo: Not supported; passing ``True`` raises ``NotImplementedError``.
            max_draws: Optional posterior draw subsampling for speed.
            config: Override the integrator ``SimulationConfig``.
            scale: ArviZ ELPD scale (``"log"``, ``"negative_log"``, ``"deviance"``).

        Returns:
            arviz.ELPDData: LOO result with ``elpd_loo``, ``p_loo`` and pointwise
            Pareto-k diagnostics.
        """
        from catalax.mcmc.loo import loo as _loo

        return _loo(
            self,
            dataset,
            yerrs=yerrs,
            sigma_source=sigma_source,
            leave_out=leave_out,
            integration=integration,
            pointwise=pointwise,
            reloo=reloo,
            max_draws=max_draws,
            config=config,
            scale=scale,
        )

    def compare(
        self,
        others: Dict[str, "HMCResults"],
        dataset: "Union[Dataset, Dict[str, Dataset]]",
        *,
        yerrs: Optional[Union[float, Array]] = None,
        sigma_source: "SigmaSource" = "reuse",
        leave_out: "LeaveOut" = "point",
        integration: "Integration" = "euler",
        max_draws: Optional[int] = None,
        scale: str = "log",
        **compare_kwargs,
    ):
        """Compare this fit against others by concentration-space LOO.

        Thin wrapper over ``arviz.compare`` that scores every fit with its own
        integrator-based ``log_likelihood`` group, so models trained in
        different modes are compared on the same concentration-space footing.

        Args:
            others: Mapping ``name -> HMCResults`` for the competing fits. This
                fit is added automatically under the name ``"self"``.
            dataset: A single :class:`Dataset` used for all fits, or a mapping
                ``name -> Dataset`` (must include ``"self"``).
            yerrs, sigma_source, leave_out, integration, max_draws, scale:
                Forwarded to the LOO reconstruction / ``arviz.compare``.
            **compare_kwargs: Extra keyword arguments for ``arviz.compare``.

        Returns:
            pandas.DataFrame: The ArviZ comparison table.
        """
        from catalax.mcmc.loo import compare as _compare

        results_map: Dict[str, "HMCResults"] = {"self": self, **others}
        return _compare(
            results_map,
            dataset,
            yerrs=yerrs,
            sigma_source=sigma_source,
            leave_out=leave_out,
            integration=integration,
            max_draws=max_draws,
            scale=scale,
            **compare_kwargs,
        )

    def loo_consistency_check(
        self,
        dataset: "Dataset",
        *,
        yerrs: Optional[Union[float, Array]] = None,
        max_draws: Optional[int] = None,
        rtol: float = 1e-2,
        atol: float = 1e-1,
    ) -> dict:
        """Check the eval-model reconstruction against ArviZ's native LOO.

        For a mechanistic fit, native ``az.loo(az.from_numpyro(mcmc))`` and the
        eval-model reconstruction should agree on ``elpd_loo``. Agreement here
        validates the surrogate-mode reconstruction, which has no native
        counterpart. Raises if called on a surrogate-mode fit.

        Args:
            dataset: The dataset that was fit.
            yerrs: Measurement error (used for the noise prior scale).
            max_draws: Optional draw subsampling for the reconstruction.
            rtol, atol: Tolerances for the ``elpd_loo`` agreement check.

        Returns:
            dict with ``native_elpd``, ``reconstructed_elpd``, ``abs_diff`` and
            ``agree``.
        """
        from catalax.mcmc.loo import consistency_check

        return consistency_check(
            self,
            dataset,
            yerrs=yerrs,
            leave_out="point",
            max_draws=max_draws,
            rtol=rtol,
            atol=atol,
        )

    def loo_pointwise(
        self,
        dataset: "Dataset",
        *,
        yerrs: Optional[Union[float, Array]] = None,
        sigma_source: "SigmaSource" = "reuse",
        integration: "Integration" = "euler",
        max_draws: Optional[int] = None,
        config: Optional["SimulationConfig"] = None,
        scale: str = "log",
    ):
        """Per-(species, timepoint) LOO diagnostics mapped onto the data grid.

        Returns a :class:`~catalax.mcmc.loo.LooPointwise` holding ``elpd`` and
        ``pareto_k`` arrays shaped ``(n_measurements, n_timepoints,
        n_observables)`` (``NaN`` where a point was not scored), together with the
        measured data, times and species. This is the structure the LOO plots
        consume; see :meth:`loo` for the argument semantics.
        """
        from catalax.mcmc.loo import loo_pointwise

        return loo_pointwise(
            self,
            dataset,
            yerrs=yerrs,
            sigma_source=sigma_source,
            integration=integration,
            max_draws=max_draws,
            config=config,
            scale=scale,
        )

    def plot_loo_influence(
        self,
        dataset: "Dataset",
        *,
        influence: str = "pareto_k",
        k_threshold: float = 0.7,
        measurements: Optional[List[int]] = None,
        ncols: int = 2,
        figsize: Tuple[float, float] = (5.0, 3.5),
        yerrs: Optional[Union[float, Array]] = None,
        sigma_source: "SigmaSource" = "reuse",
        integration: "Integration" = "euler",
        max_draws: Optional[int] = None,
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Plot the data with each observation sized by its LOO influence.

        Draws the measured trajectories as usual and overlays every point with a
        marker whose area grows with its influence (Pareto-k by default), ringing
        the points above ``k_threshold`` so the high-leverage observations -- the
        ones a naive split would discard -- are obvious.

        Args:
            dataset: The measured dataset to score and plot.
            influence: ``"pareto_k"`` (default) or ``"elpd"`` (per-point penalty).
            k_threshold: Pareto-k above which a point is ringed.
            measurements: Indices of measurements to plot (default: all).
            ncols: Subplot columns.
            figsize: Per-panel size.
            yerrs, sigma_source, integration, max_draws: Forwarded to the LOO
                computation (see :meth:`loo`).
            show: Display the figure and return ``None``.
            path: Save the figure to this path.

        Returns:
            matplotlib.figure.Figure (or ``None`` if ``show=True``).
        """
        from catalax.mcmc.plotting import plot_loo_influence

        pointwise = self.loo_pointwise(
            dataset,
            yerrs=yerrs,
            sigma_source=sigma_source,
            integration=integration,
            max_draws=max_draws,
        )
        f = plot_loo_influence(
            pointwise,
            self.model,
            influence=influence,
            k_threshold=k_threshold,
            measurements=measurements,
            ncols=ncols,
            figsize=figsize,
        )
        if path is not None:
            f.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
            return None
        return f

    def plot_loo_heatmap(
        self,
        dataset: "Dataset",
        *,
        metric: str = "elpd",
        k_threshold: float = 0.7,
        measurements: Optional[List[int]] = None,
        ncols: int = 1,
        figsize: Tuple[float, float] = (6.0, 2.6),
        cmap: Optional[str] = None,
        yerrs: Optional[Union[float, Array]] = None,
        sigma_source: "SigmaSource" = "reuse",
        integration: "Integration" = "euler",
        max_draws: Optional[int] = None,
        show: bool = False,
        path: Optional[str] = None,
    ):
        """Species x time heatmap of the LOO penalty (or Pareto-k) per measurement.

        Each cell is a measured observation coloured by its penalty
        (``-elpd_loo``, darker = worse) or PSIS reliability (``pareto_k``); cells
        with ``pareto_k > k_threshold`` are dotted.

        Args:
            dataset: The measured dataset to score and plot.
            metric: ``"elpd"`` (default) or ``"pareto_k"``.
            k_threshold: Pareto-k above which a cell is dotted.
            measurements: Indices of measurements to plot (default: all).
            ncols: Measurement panels per row.
            figsize: Per-panel size.
            cmap: Override the colormap.
            yerrs, sigma_source, integration, max_draws: Forwarded to the LOO
                computation (see :meth:`loo`).
            show: Display the figure and return ``None``.
            path: Save the figure to this path.

        Returns:
            matplotlib.figure.Figure (or ``None`` if ``show=True``).
        """
        from catalax.mcmc.plotting import plot_loo_heatmap

        pointwise = self.loo_pointwise(
            dataset,
            yerrs=yerrs,
            sigma_source=sigma_source,
            integration=integration,
            max_draws=max_draws,
        )
        f = plot_loo_heatmap(
            pointwise,
            self.model,
            metric=metric,
            k_threshold=k_threshold,
            measurements=measurements,
            ncols=ncols,
            figsize=figsize,
            cmap=cmap,
        )
        if path is not None:
            f.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
            return None
        return f

    def to_arviz(self):
        """Convert MCMC results to ArviZ InferenceData format.

        Transforms the numpyro MCMC results into ArviZ's standardized format,
        enabling access to advanced diagnostic tools, visualization methods,
        and interoperability with other Bayesian analysis packages.

        Returns:
            arviz.InferenceData: MCMC results in ArviZ format containing
                posterior samples, sample statistics, and metadata. This format
                provides access to comprehensive Bayesian analysis tools and
                standardized visualization methods.
        """
        return az.from_numpyro(self.mcmc)

    def to_netcdf(self, path: str):
        """Save MCMC results to NetCDF format for persistent storage.

        Exports the complete MCMC results to a self-describing NetCDF file
        that preserves all sample data, metadata, and structure. This format
        is ideal for archiving results, sharing analyses, or loading results
        in different computing sessions.

        Args:
            path: File path where the NetCDF file will be saved. Should include
                the '.nc' extension. The file will contain all posterior samples,
                diagnostics, and metadata in a standardized format.

        Returns:
            None: Results are saved to the specified path. The file can later
                be loaded using arviz.from_netcdf() or similar tools.
        """
        return az.from_numpyro(self.mcmc).to_netcdf(path)

    def _is_jupyter_environment(self) -> bool:
        """Check if running in Jupyter environment.

        Returns:
            bool: True if running in Jupyter notebook/lab, False otherwise.
        """
        if not WIDGETS_AVAILABLE:
            return False

        try:
            # Check if we're in IPython and if it's a kernel
            ipython = get_ipython()
            if ipython is None:
                return False

            # Check for Jupyter-specific attributes
            return hasattr(ipython, "kernel") or ipython.__class__.__name__ in [
                "ZMQInteractiveShell",
                "TerminalInteractiveShell",
            ]
        except Exception:
            return False

    def _create_plot_tab(
        self,
        plot_name: str,
        plot_func,
        *args,
        figsize: Optional[Tuple[float, float]] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """Create a tab widget for a specific plot.

        Args:
            plot_name: Name of the plot for tab title
            plot_func: Function to call for generating the plot
            *args: Arguments to pass to plot function
            figsize: Figure size for the plot
            backend: Backend to use ('matplotlib' or 'bokeh')
            **kwargs: Additional keyword arguments for plot function

        Returns:
            ipywidgets.Output: Output widget containing the plot
        """
        output = widgets.Output()

        with output:
            try:
                # Add figsize and backend to kwargs if provided
                if figsize is not None:
                    kwargs["figsize"] = figsize
                if backend is not None:
                    kwargs["backend"] = backend

                plot_func(*args, **kwargs)

                # Only show plot if using matplotlib backend
                if backend != "bokeh":
                    plt.show()
            except Exception as e:
                print(f"Error creating {plot_name} plot: {str(e)}")

        return output

    def create_plots_widget(
        self,
        initial_condition: Optional[Dict[str, float]] = None,
        time: Optional[Array] = None,
        dt0: float = 0.1,
        figsize: Optional[Tuple[float, float]] = (10, 6),
        backend: Optional[str] = None,
        tabs_to_include: Optional[List[str]] = None,
    ) -> Optional[widgets.Widget]:
        """Create an interactive tabbed widget with all diagnostic plots.

        This method creates a comprehensive tabbed interface for exploring
        MCMC results with different diagnostic plots. Each tab contains
        a different type of visualization.

        Args:
            initial_condition: Initial conditions for credibility interval plot.
                If None, credibility interval tab will be excluded.
            time: Time points for credibility interval plot.
                If None, credibility interval tab will be excluded.
            dt0: Time step for credibility interval simulation.
            figsize: Default figure size for all plots.
            backend: Backend to use ('matplotlib' or 'bokeh'). If None, uses default.
            tabs_to_include: List of tab names to include. If None, includes all available.
                Available options: ['posterior', 'trace', 'corner', 'forest',
                'credibility', 'ess', 'mcse']

        Returns:
            ipywidgets.Tab: Tabbed widget with diagnostic plots, or None if widgets
                are not available or not in Jupyter environment.

        Example:
            >>> # Create widget with default settings
            >>> results.create_plots_widget()

            >>> # Create widget with specific tabs and backend
            >>> results.create_plots_widget(
            ...     figsize=(8, 6),
            ...     backend='bokeh',
            ...     tabs_to_include=['posterior', 'trace', 'corner']
            ... )
        """
        if not self._is_jupyter_environment():
            print("Interactive widgets are only available in Jupyter environments.")
            return None

        # Default tabs to include
        available_tabs = {
            "posterior": ("Posterior", self.plot_posterior),
            "trace": ("Trace", self.plot_trace),
            "corner": ("Corner", self.plot_corner),
            "forest": ("Forest", self.plot_forest),
            "ess": ("ESS", self.plot_ess),
            "mcse": ("MCSE", self.plot_mcse),
        }

        # Add credibility interval tab if data is provided
        if initial_condition is not None and time is not None:
            available_tabs["credibility"] = (
                "Credibility Interval",
                self.plot_credibility_interval,
            )

        # Filter tabs if specified
        if tabs_to_include is not None:
            available_tabs = {
                k: v for k, v in available_tabs.items() if k in tabs_to_include
            }

        # Create tab outputs
        tab_outputs = []
        tab_titles = []

        for tab_key, (title, plot_method) in available_tabs.items():
            tab_titles.append(title)

            # Special handling for credibility interval plot
            if tab_key == "credibility":
                output = self._create_plot_tab(
                    title,
                    plot_method,
                    initial_condition,
                    time,
                    dt0,
                    figsize=figsize,
                    backend=backend,
                    show=False,  # Don't show immediately in widget
                )
            else:
                output = self._create_plot_tab(
                    title,
                    plot_method,
                    figsize=figsize,
                    backend=backend,
                    show=False,  # Don't show immediately in widget
                )

            tab_outputs.append(output)

        # Create tabbed interface
        tab_widget = widgets.Tab(children=tab_outputs)

        # Set tab titles
        for i, title in enumerate(tab_titles):
            tab_widget.set_title(i, title)

        # Display the widget
        display(tab_widget)

        return tab_widget
