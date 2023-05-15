from typing import List, Dict, Optional
from pydantic import BaseModel
import numpy as np

from sysbiojax.model.parameter import Parameter
from sysbiojax.model.species import Species


class Series(BaseModel):
    values: List[float]


class Reactant(Species):
    init_conc: float
    unit: str
    data: Optional[List[Series]] = None

    def get_max_data_length(self):
        """Checks the maximal length of data array for a measurement of a reactant."""
        if self.data is not None:
            return max([len(d.values) for d in self.data])
        return 0

    def get_n_replicates(self):
        """Checks the ammount of replicates for a measurement of a reactant."""
        if self.data is not None:
            return len(self.data)
        return 0


class Measurement(BaseModel):
    reactants: Dict[str, Reactant]
    time: list
    time_unit: str

    def get_max_data_lengths(self):
        return max(
            [reactant.get_max_data_length() for reactant in self.reactants.values()]
        )

    def get_max_replicates(self):
        return max(reactant.get_n_replicates() for reactant in self.reactants.values())


class Experiment(BaseModel):
    name: str
    measurements: List[Measurement]
