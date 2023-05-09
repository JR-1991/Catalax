from typing import List, Dict, Optional
from pydantic import BaseModel

from sysbiojax.model.parameter import Parameter
from sysbiojax.model.species import Species


class Series(BaseModel):
    values: List[float]


class Reactant(Species):
    init_conc: float
    unit: str
    data: Optional[List[Series]] = None


class Catalyst(Species):
    init_conc: float
    unit: str
    data: Optional[List[Series]] = None


class Measurement(BaseModel):
    catalysts: Dict[str, Catalyst]
    reactants: Dict[str, Reactant]
    time: list
    time_unit: str


class Experiment(BaseModel):
    name: str
    measurements: List[Measurement]

    def _init_measurements_as_array(self):
        pass
