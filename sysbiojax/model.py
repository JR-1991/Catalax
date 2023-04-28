from typing import Dict, List, Optional, Union
from dotted_dict import DottedDict
from pydantic import BaseModel, Field, PrivateAttr, validator
from sympy import Expr, sympify, symbols

from .utils import odeprint


def _parameter_exists(name: str, parameters: Dict[str, "Parameter"]) -> bool:
    """Checks whether a parameter is already present in a model"""

    return any(param.name == name for param in parameters.values())


class Parameter(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    value: Optional[float] = None
    initial_value: Optional[float] = None
    equation: Union[str, Expr, None] = None


class Species(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: Expr

    @validator("name", pre=True)
    def convert_string_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        if isinstance(value, str):
            value = symbols(value)

        return value


class ODE(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    species: Species
    equation: Expr
    parameters: List[Parameter] = Field(default_factory=list)

    __model__: Optional["Model"] = PrivateAttr(default=None)

    @validator("equation", pre=True)
    def converts_ode_to_sympy(cls, value):
        """Convertes a string"""

        return sympify(value)

    def add_parameter(
        self,
        name: str,
        value: Optional[float] = None,
        initial_value: Optional[float] = None,
        equation: Union[str, Expr, None] = None,
    ):
        """Adds a parameter to an ODE"""

        parameter = Parameter(
            name=name, value=value, initial_value=initial_value, equation=equation
        )

        if self.__model__ and not _parameter_exists(name, self.__model__.parameters):
            self.__model__.parameters[parameter.name] = parameter

        self.parameters.append(parameter)


class Model(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    odes: List[ODE] = Field(default_factory=list)
    species: Dict[str, Species] = Field(default_factory=DottedDict)
    parameters: Dict[str, Parameter] = Field(default_factory=DottedDict)

    @validator("species", pre=True)
    def _convert_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        symbols_ = []

        for symbol in value:
            if isinstance(symbol, str):
                symbol = symbols(symbol)

            symbols_ += list(symbol)

        return symbols_

    def add_ode(
        self,
        equation: str,  # type: ignore
        species: str,
        parameters: List[Union[str, Parameter]] = [],
    ):  # type: ignore
        """Adds a an ODE to the model"""

        if isinstance(equation, str):
            equation: Expr = sympify(equation)

        # Turn all parameters into Parameter objects
        converted_params: List[Parameter] = [
            parameter if isinstance(parameter, Parameter) else Parameter(name=parameter)
            for parameter in parameters
        ]

        # Add parameters to model
        for parameter in converted_params:
            if not isinstance(parameter, str) and not _parameter_exists(
                parameter.name, self.parameters
            ):
                self.parameters[parameter.name] = parameter

        # Add species
        if species not in self.species:
            self.add_species(name=species)

        self.odes.append(
            ODE(
                equation=equation,
                parameters=converted_params,
                species=self.species[species],
                __model__=self,
            )
        )

    def add_species(self, name: str) -> None:
        """Adds a species to a model"""

        symbol = symbols(name)

        if isinstance(symbol, tuple):
            self.species.update({str(sym): Species(name=sym) for sym in symbol})
        else:
            self.species[str(symbol)] = Species(name=symbol)

    def add_parameter(
        self,
        name: str,
        value: Optional[float] = None,
        initial_value: Optional[float] = None,
        equation: Union[str, Expr, None] = None,
    ):
        """Adds a parameter to an ODE"""

        parameter = Parameter(
            name=name, value=value, initial_value=initial_value, equation=equation
        )

        if not _parameter_exists(name, self.parameters):
            self.parameters[parameter.name] = Parameter(
                name=name, value=value, initial_value=initial_value, equation=equation
            )
        else:
            print("Parameter already exists. Skipping...")

    def __repr__(self):
        """Prints a summary of the model"""

        for ode in self.odes:
            odeprint(y=ode.species.name, expr=ode.equation)

        return ""
