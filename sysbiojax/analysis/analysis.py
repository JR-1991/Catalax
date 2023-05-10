from typing import Dict, List
from pydantic import BaseModel

from sysbiojax.model.model import Model
from sysbiojax.measurement.measurment import Experiment


class Analysis(BaseModel):
    data: Experiment
    model: Model

    def _initialize_measurement_data(self):
        n_measurements = len(self.data.measurements)

        reactant_keys = sorted(
            [measurement.reactants.keys() for measurement in self.data.measurements]
        )
        n_reactants = len(reactant_keys)

        for measurement in self.data.measurements:
            for key in reactant_keys:
                if measurement.reactants[key].data != None:
                    a
