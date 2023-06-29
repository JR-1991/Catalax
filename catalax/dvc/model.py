import os
import zntrack

from catalax import Model


class ModelLoader(zntrack.Node):
    """ZnTrack/DVC Node to load a model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        species (Dict[str, Dict[str, str]]): Metadata for the species of the model.
        model (Model): The loaded model.
    """

    # Dependencies
    model_path: str = zntrack.dvc.deps()  # type: ignore

    # Metrics
    species: dict = zntrack.zn.metrics()  # type: ignore

    # Outputs
    model: Model = zntrack.zn.outs()  # type: ignore

    def run(self):
        """Run method to initialize the model"""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

        self.model = Model.load(self.model_path)

        # Document species of the model
        self.species = {
            species: {"observable": ode.observable}
            for species, ode in self.model.odes.items()
        }
