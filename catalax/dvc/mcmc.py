import zntrack
import numpy as np

from typing import Dict

from catalax import Model
from catalax.mcmc import run_mcmc

from .datainput import DataInput


class MCMC(zntrack.Node):
    """ZnTrack/DVC node to perform MCMC sampling.

    Args:
        data (DataInput): The data to fit the model to.
        model (Model): The model to fit.
        yerrs (float): The standard deviation of the observed data.
        num_warmup (int): Number of warmup steps. Defaults to 5000.
        num_samples (int): Number of samples to draw from the posterior. Defaults to 5000.
        dt0 (float): Resolution of the simulation. Defaults to 0.1.
        max_tree_depth (int): Maximum depth of the tree. Defaults to 7.
        max_steps (int): Maximum number of integration steps. Defaults to 64**4.
        verbose (bool): Whether to show a progress bar or not. Defaults to True.
        seed (int): Random number seed to reproduce results. Defaults to 420.

    Returns:
        posterior_statistics (Dict): The posterior statistics of the parameters.
    """

    # Inputs
    data: DataInput = zntrack.zn.deps()  # type: ignore
    model: Model = zntrack.zn.deps()  # type: ignore

    # Parameters
    yerrs: float = zntrack.zn.params()  # type: ignore
    num_warmup: int = zntrack.zn.params(5000)  # type: ignore
    num_samples: int = zntrack.zn.params(5000)  # type: ignore
    dt0: float = zntrack.zn.params(0.1)  # type: ignore
    max_tree_depth: int = zntrack.zn.params(7)  # type: ignore
    max_steps: int = zntrack.zn.params(64**4)  # type: ignore
    verbose: bool = zntrack.zn.params(True)  # type: ignore
    seed: int = zntrack.zn.params(420)  # type: ignore

    # Metrics
    posterior_statistics: Dict = zntrack.zn.metrics()  # type: ignore

    def run(self):
        """Performs MCMC sampling based on the given parameters."""

        # Unpack the data
        times = self.data.times
        data = self.data.data
        initial_conditions = self.data.initial_conditions

        # Perform MCMC simulation
        mcmc, _ = run_mcmc(
            model=self.model,
            data=data,
            initial_conditions=initial_conditions,
            times=times,
            yerrs=self.yerrs,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            dt0=self.dt0,
            max_tree_depth=self.max_tree_depth,
            max_steps=self.max_steps,
            verbose=self.verbose,
            seed=self.seed,
        )

        # Save the posterior statistics
        self.posterior_statistics = {
            parameter: {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "prior": self.model.parameters[parameter].prior.dict(),
            }
            for parameter, samples in mcmc.get_samples().items()  # type: ignore
            if parameter in self.model.parameters
        }
