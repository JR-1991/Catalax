from typing import Dict, List, Optional, Union
from dotted_dict import DottedDict
from pydantic import BaseModel, Field, PrivateAttr, validator
from sympy import Expr, Symbol, sympify, symbols

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

    name: str
    symbol: Expr

    @validator("symbol", pre=True)
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
    parameters: Dict[Union[str, Expr], Parameter] = Field(default_factory=DottedDict)

    __model__: Optional["Model"] = PrivateAttr(default=None)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name == "__model__":
            self.add_parameters_to_model()

        return

    @validator("equation", pre=True)
    def converts_ode_to_sympy(cls, value):
        """Convertes a string"""

        return sympify(value)

    def add_parameters_to_model(self):
        """Adds parameters given by the ODE to the model and ODE

        This step is necessary to ensure that the parameters are available
        model-wide and not just within the ODE. Also, this step can only be
        done upon addition of the model due to no given knowledge of the species.
        """

        if self.__model__ is None:
            return None

        for symbol in self.equation.free_symbols:
            if str(symbol) in self.__model__.species:
                # Skip species
                continue
            elif _parameter_exists(str(symbol), self.__model__.parameters):
                # Assign parameter if it is already present in the model
                self.parameters[str(symbol)] = self.__model__.parameters[str(symbol)]
                continue

            # Create a new one and add it to the model and ODE
            parameter = Parameter(name=str(symbol))

            self.parameters[str(symbol)] = parameter
            self.__model__.parameters[str(symbol)] = parameter

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
            self.__model__.parameters[name] = parameter

        self.parameters[name] = parameter


class Model(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    odes: Dict[str, ODE] = Field(default_factory=DottedDict)
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
        species: str,
        equation: str,  # type: ignore
        species_map: Optional[Dict[str, str]] = None,
    ):  # type: ignore
        """Adds a an ODE to the model"""

        if any(str(ode.species.name) == species for ode in self.odes.values()):
            raise ValueError(f"Species {species} already exists in model")

        if isinstance(equation, str):
            equation: Expr = sympify(equation)

        # Add species
        if species not in self.species:
            self.add_species(name=species, species_map=species_map)

        self.odes[species] = ODE(
            equation=equation,
            species=self.species[species],
        )

        self.odes[species].__model__ = self

    def add_species(self, species_string: str = "", **species_map):
        """Adds single or multiple species to a model. If multiple species are provided as
        a string where species are separated by a comma, the species will be added one by one.
        """

        if not all(isinstance(value, str) for value in species_map.values()):
            raise TypeError("Species names must be of type str")

        if species_string:
            species_map.update(
                {str(species): str(species) for species in symbols(species_string)}
            )

        for symbol, name in species_map.items():
            if not isinstance(name, str):
                raise TypeError("Species names must be of type str")

            self.species[symbol] = Species(name=name, symbol=Symbol(symbol))

    def _add_single_species(
        self, name: str, species_map: Optional[Dict[str, str]] = None
    ) -> None:
        """Adds a species to a model"""

        if species_map is None:
            self.species[name] = Species(name=name, symbol=symbols(name))
            return

        if self._is_symbol(name, species_map):
            symbol = name
            name = species_map[str(symbol)]
        else:
            symbol = self._gather_symbol_from_species_map(name, species_map)

        self.species[str(symbol)] = Species(name=name, symbol=Symbol(symbol))

    @staticmethod
    def _is_symbol(name: str, species_map: Dict[str, str]) -> bool:
        """Checks whether the given name is an identifer (key of species_map)"""

        inverse_dict = {v: k for k, v in species_map.items()}

        if str(name) not in species_map and name not in inverse_dict:
            raise ValueError(f"Species {name} not found in species map")

        return name in species_map

    @staticmethod
    def _gather_symbol_from_species_map(name: str, species_map: Dict[str, str]) -> str:
        """Converts a name to a sympy symbol"""

        inverse_dict = {v: k for k, v in species_map.items()}

        if name not in inverse_dict:
            raise ValueError(f"Species {name} not found in species map")

        return str(inverse_dict[name])

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

        for ode in self.odes.values():
            odeprint(y=ode.species.name, expr=ode.equation)

        return ""
