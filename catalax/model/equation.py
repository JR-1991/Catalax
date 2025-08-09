from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union  # noqa: F401

from dotted_dict import DottedDict
from pydantic import Field, PrivateAttr
from pydantic.config import ConfigDict
from pydantic.functional_validators import field_validator
from sympy import Expr, sympify

from catalax.model.base import CatalaxBase
from .parameter import Parameter
from .utils import parameter_exists

if TYPE_CHECKING:
    from catalax.model import Model


class Equation(CatalaxBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    equation: Expr
    parameters: Dict[Union[str, Expr], Parameter] = Field(
        default_factory=DottedDict,
        exclude=True,
    )

    _model: Optional[Model] = PrivateAttr(default=None)  # type: ignore

    @field_validator("equation", mode="before")
    def converts_assignment_to_sympy(cls, value):
        """Convertes a string to a sympy expression"""

        return sympify(value)

    def __setattr__(self, name, value):
        """Override of the __setattr__ method to add parameters to the model"""

        super().__setattr__(name, value)

        if name == "_model":
            self.add_parameters_to_model()

    def reparametrize(self, **replacements: Dict[str, Expr | str | float]):
        """Reparametrizes the equation"""

        self.equation = self.equation.subs(replacements)  # type: ignore
        self.add_parameters_to_model()

    def add_parameters_to_model(self):
        """Adds parameters given by the ODE to the model and ODE

        This step is necessary to ensure that the parameters are available
        model-wide and not just within the ODE. Also, this step can only be
        done upon addition of the model due to no given knowledge of the species.
        """

        if self._model is None:
            return None

        for symbol in self.equation.free_symbols:
            if str(symbol) in self._model.species or str(symbol) == "t":
                # Skip species and time symbol
                continue
            elif str(symbol) in self._model.constants:
                # Skip constants
                continue
            elif str(symbol) in self._model.assignments:
                # Skip assignments
                continue
            elif parameter_exists(str(symbol), self._model.parameters):
                # Assign parameter if it is already present in the model
                self.parameters[str(symbol)] = self._model.parameters[str(symbol)]
                continue

            # Create a new one and add it to the model and ODE
            parameter = Parameter(name=str(symbol), symbol=symbol)  # type: ignore

            self.parameters[str(symbol)] = parameter
            self._model.parameters[str(symbol)] = parameter
