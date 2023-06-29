from typing import Dict, Optional, Union

from dotted_dict import DottedDict
from pydantic import BaseModel, Field, PrivateAttr, validator
from sympy import Expr, Symbol, sympify

from catalax.model.base import CatalaxBase

from .species import Species
from .parameter import Parameter
from .utils import parameter_exists


class ODE(CatalaxBase):
    class Config:
        arbitrary_types_allowed = True
        fields = {
            "parameters": {"exclude": True},
            "species": {"exclude": True},
        }

    species: Species
    equation: Expr
    observable: bool = True
    parameters: Dict[Union[str, Expr], Parameter] = Field(default_factory=DottedDict)

    __model__: Optional["Model"] = PrivateAttr(default=None)  # type: ignore

    @validator("equation", pre=True)
    def converts_ode_to_sympy(cls, value):
        """Convertes a string"""

        return sympify(value)

    def __setattr__(self, name, value):
        """Override of the __setattr__ method to add parameters to the model"""

        super().__setattr__(name, value)

        if name == "__model__":
            self.add_parameters_to_model()

    def add_parameters_to_model(self):
        """Adds parameters given by the ODE to the model and ODE

        This step is necessary to ensure that the parameters are available
        model-wide and not just within the ODE. Also, this step can only be
        done upon addition of the model due to no given knowledge of the species.
        """

        if self.__model__ is None:
            return None

        for symbol in self.equation.free_symbols:
            if str(symbol) in self.__model__.species or str(symbol) == "t":
                # Skip species and time symbol
                continue
            elif parameter_exists(str(symbol), self.__model__.parameters):
                # Assign parameter if it is already present in the model
                self.parameters[str(symbol)] = self.__model__.parameters[str(symbol)]
                continue

            # Create a new one and add it to the model and ODE
            parameter = Parameter(name=str(symbol), symbol=symbol)  # type: ignore

            self.parameters[str(symbol)] = parameter
            self.__model__.parameters[str(symbol)] = parameter
