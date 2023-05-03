from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from diffrax import Kvaerno5, ODETerm, PIDController, SaveAt
from dotted_dict import DottedDict
from pydantic import BaseModel, Field, PrivateAttr, validator
from sympy import Expr, Matrix, Symbol, symbols, sympify
from sympy2jax import SymbolicModule

from sysbiojax.tools import Stack
from sysbiojax.tools.simulation import Simulation

from .ode import ODE
from .parameter import Parameter, ParameterDict
from .species import Species
from .utils import check_symbol, eqprint, odeprint, parameter_exists


class Model(BaseModel):

    """
    Model class for storing ODEs, species, and parameters, which is used
    to describe a biological system. Model classes can be passed to analysis
    and optimisation methods, which will utilize the defined model. The workflow
    of using the Model class is as follows:

        (1) Instantiate a Model object
        (2) Add species to the model
        (3) Add ODEs to the model
        (4) Further modify parameters as needed (values, equations, etc.)

    Example usage:

        model = Model(name="model")
        model.add_species("s1, s2, e1, c1") # see 'add_species' method for more details

        # Add ODEs
        model.add_ode("s1", "k1 * s1")
        model.add_ode("s2", "k2 * s2")

        # Add custom equations (equilibrium constants etc.)
        model.add_parameter(name="E_tot", equation="c1 + e1")
    """

    class Config:
        arbitrary_types_allowed = True

    name: str
    odes: Dict[str, ODE] = Field(default_factory=DottedDict)
    species: Dict[str, Species] = Field(default_factory=DottedDict)
    parameters: Dict[str, Parameter] = Field(default_factory=ParameterDict)
    term: Optional[ODETerm] = Field(default=None)

    _sim_func: Optional[Callable] = PrivateAttr(default=None)
    _in_axes: Optional[Tuple] = PrivateAttr(default=None)

    def add_ode(
        self,
        species: str,
        equation: str,  # type: ignore
        species_map: Optional[Dict[str, str]] = None,
    ):
        """Adds an ODE to the model and converts the equation to a SymPy expression.

        This method will add the new ODE to the model and the model's ODEs attribute,
        which can be accessed by object dot-notation. For example, if the ODE is set
        for species 's1' and the model is named 'model', the ODE can be accessed by:

        model = Model(name="model")
        model.add_ode(species="s1", equation="k1*s1")
        model.odes.s1 -> ODE(equation=k1*s1, species=s1)

        Parameters will be inferred from the equation and added to the model automatically
        by comparing free symbols in the equation to the model's species. If a symbol is not
        present as a species, it will be added as a parameter. If the symbol is already present
        as a parameter the already defined one will be referenced instead of creating a new one.

        If there already exists an ODE for the given species, a ValueError will be raised.
        Due to the theoretical nature of dynamic systems, there can only be one ODE per
        species. If a species hasnt been added to the model, it will be added automatically.

        Args:
            species (str): The species to be modelled within this ODE.
            equation (str): The equation that describes the dynamics of the species.

        Raises:
            ValueError: _description_
        """

        if any(str(ode.species.name) == species for ode in self.odes.values()):
            raise ValueError(
                f"There already exists an ODE for species '{species}'. Please edit the existing ODE instead."
            )

        if isinstance(equation, str):
            equation: Expr = sympify(equation)

        if species not in self.species:
            self.add_species(name=species, species_map=species_map)

        self.odes[species] = ODE(
            equation=equation,
            species=self.species[species],
        )

        self.odes[species].__model__ = self

    def add_species(self, species_string: str = "", **species_map):
        """Adds a single or multiple species to the model, which can later be used in ODEs.

        This method will add new species to the model and the model's species dictionary,
        which can be accessed by object dot-notation. For example, if the species is named 's1'
        and the model is named 'model', the species can be accessed by:

            model = Model(name="model")
            model.add_species("s1")
            model.species.s1 -> Species(name="s1", symbol=s1)

        Species can be added in three ways. The first is by passing a string of species names that
        are separated by a comma. Please note, symbols will be also be used as names. The second is
        by passing a dictionary of species symbols (key) and names (values) and unpacking them as
        keyword arguments. The third is by passing each as a keyword argument.

        The following are all valid ways to add species:

            (1) model.add_species("s1, s2, s3")
            (2) model.add_species(**{"s1": "species1", "s2": "species2", "s3": "species3"})
            (3) model.add_species(s1="species1", s2="species2", s3="species3")

        If a species already exists, a ValueError will be raised.

        Args:
            species_string (str, optional): String of comma-separated species symbols. Defaults to "".
            **species_map (Dict[str, str]): Dictionary of species symbols (key) and names (values).
        """

        if not all(isinstance(value, str) for value in species_map.values()):
            raise TypeError("Species names must be of type str")

        if species_string:
            species_map.update(
                {
                    str(species): str(species)
                    for species in self._split_species_string(species_string)
                }
            )

        for symbol, name in species_map.items():
            if not isinstance(name, str):
                raise TypeError("Species names must be of type str")

            # Make sure the symbol is valid
            check_symbol(symbol)

            self.species[symbol] = Species(name=name, symbol=Symbol(symbol))

    @staticmethod
    def _split_species_string(species_string: str) -> List[Symbol]:
        """Helper method to split a string of species into a list of species"""

        if len(species_string.split(",")) == 1:
            return [Symbol(species_string)]

        return symbols(species_string)

    def _add_single_species(
        self, name: str, species_map: Optional[Dict[str, str]] = None
    ) -> None:
        """Helper method to add a single species to a model"""

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

        if not parameter_exists(name, self.parameters):
            self.parameters[parameter.name] = Parameter(
                name=name, value=value, initial_value=initial_value, equation=equation
            )
        else:
            print("Parameter already exists. Skipping...")

    # ! Simulation methods

    def simulate(
        self,
        initial_conditions: List[Dict[str, float]],
        t0: int,
        t1: int,
        dt0: float,
        solver=Kvaerno5(),
        nsteps: Optional[int] = None,
        saveat: Optional[SaveAt] = None,
        stepsize_controller: PIDController = PIDController(rtol=1e-5, atol=1e-5),
        parameters: Optional[jax.Array] = None,
        in_axes: Tuple = (0, None, None),
    ):
        """Simulates the given model"""

        if nsteps and saveat:
            raise ValueError(
                "Cannot specify both nsteps and saveat. Please choose one."
            )
        elif nsteps is None and saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        if isinstance(initial_conditions, dict):
            initial_conditions = [initial_conditions]

        # Get the order of the species
        species_order = self._get_species_order()

        parameter_maps, parameters = self._get_parameters(parameters)

        # Setup the initial conditions
        y0 = self._assemble_y0_array(initial_conditions)

        # Setup save points
        if nsteps is not None:
            saveat = jnp.linspace(t0, t1, nsteps)  # type: ignore
        elif saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        if in_axes != self._in_axes or self._sim_func is None:
            self._setup_system(
                in_axes=in_axes,
                t0=t0,
                t1=t1,
                dt0=dt0,
                solver=solver,
                stepsize_controller=stepsize_controller,
                species_maps={symbol: i for i, symbol in enumerate(species_order)},
                parameter_maps={symbol: i for i, symbol in enumerate(parameter_maps)},
            )

            return self._sim_func(y0, parameters, saveat)

        return self._sim_func(y0, parameters, saveat)

    def _get_species_order(self) -> List[str]:
        """Returns the order of the species in the model"""

        return sorted(self.species.keys())

    def _setup_system(self, in_axes: Tuple = (0, None, None), **kwargs) -> Callable:
        """Converts given SymPy equations into Equinox modules, used for simulation.

        This method will prepare the simulation function, jit it and vmap it across
        the initial condition vector by default.

        Args:
            in_axes: Specifies the axes to map the simulation function across. Defaults to (0, None, None).
        """

        self.term = ODETerm(
            Stack(
                modules=[
                    SymbolicModule(self.odes[species].equation) for species in self._get_species_order()  # type: ignore
                ]
            ).__call__
        )

        simulation_setup = Simulation(term=self.term, **kwargs)
        simulation_setup._prepare_func(in_axes=in_axes)

        # Attach to the model to prevent re-modelling
        self._sim_func = simulation_setup._simulation_func
        self._in_axes = in_axes

    def _get_parameters(
        self, parameters: Optional[jax.Array]
    ) -> Tuple[List[str], jax.Array]:
        """Gets all the parameters for the model"""

        if any(param.value is None for param in self.parameters.values()):
            raise ValueError("Missing values for parameters")

        param_order = sorted(self.parameters.keys())

        if parameters is not None:
            return param_order, parameters

        return (
            param_order,
            jnp.array([self.parameters[param].value for param in param_order]),
        )  # type: ignore

    def _assemble_y0_array(
        self, initial_conditions: List[Dict[str, float]]
    ) -> jax.Array:
        """Assembles the initial conditions into an array"""

        # Check that all initial conditions are valid
        deque(map(self._check_initial_condition, initial_conditions))

        return jnp.array(
            [
                [initial_condition[species] for species in self.species.keys()]
                for initial_condition in initial_conditions
            ]
        )

    def _check_initial_condition(self, initial_condition: Dict[str, float]) -> None:
        """Checks that the initial conditions are valid"""

        if not all(species in initial_condition for species in self.species.keys()):
            raise ValueError(
                f"Not all species have initial conditions specified. Please specify initial conditions for the following species: {set(self.species.keys()) - set(initial_condition.keys())}"
            )

    # ! Helper methods

    def __repr__(self):
        """Prints a summary of the model"""

        eqprint(
            Symbol("x"), Matrix([species.symbol for species in self.species.values()]).T
        )

        eqprint(
            Symbol("theta"),
            Matrix([param.name for param in self.parameters.values()]).T,
        )

        for ode in self.odes.values():
            odeprint(y=ode.species.name, expr=ode.equation)

        return ""

    @validator("species", pre=True)
    def _convert_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        symbols_ = []

        for symbol in value:
            if isinstance(symbol, str):
                symbol = symbols(symbol)

            symbols_ += list(symbol)

        return symbols_
