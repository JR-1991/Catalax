from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from diffrax import Kvaerno5, ODETerm, PIDController, SaveAt
from dotted_dict import DottedDict
from pydantic import BaseModel, Field, validator
from sympy import Expr, Matrix, Symbol, symbols, sympify
from sympy2jax import SymbolicModule

from sysbiojax.tools.simulation import Stack, simulate

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
                {str(species): str(species) for species in symbols(species_string)}
            )

        for symbol, name in species_map.items():
            if not isinstance(name, str):
                raise TypeError("Species names must be of type str")

            # Make sure the symbol is valid
            check_symbol(symbol)

            self.species[symbol] = Species(name=name, symbol=Symbol(symbol))

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
        initial_conditions: Dict[str, float],
        t0: int,
        t1: int,
        dt0: float,
        solver=Kvaerno5(),
        nsteps: Optional[int] = None,
        saveat: Optional[SaveAt] = None,
        stepsize_controller: PIDController = PIDController(rtol=1e-5, atol=1e-5),
        constants: Dict[str, float] = {},
    ):
        """Simulates the given model"""

        if nsteps and saveat:
            raise ValueError(
                "Cannot specify both nsteps and saveat. Please choose one."
            )
        elif nsteps is None and saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        # Get the order of the species
        species_order = sorted(self.species.keys())

        if self.term is None:
            # Setup the jitted system if not done yet
            self._setup_system(species_order)

        parameters = self._get_parameters()

        # Setup the initial conditions
        y0 = self._assemble_y0_array(initial_conditions)

        # Setup save points
        if nsteps is not None:
            saveat = SaveAt(ts=jnp.linspace(t0, t1, nsteps))  # type: ignore
        elif saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        return simulate(
            term=self.term,
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=dt0,
            parameters=parameters,
            solver=solver,
            constants=constants,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            maps={symbol: i for i, symbol in enumerate(species_order)},
        )

    def _setup_system(self, species_order: List[str]) -> None:
        """Converts given SymPy equations into Equinox modules, used for simulation

        In order to keep generality, species are sorted alphabetically by name. This is done to
        ensure that the order of the equations is the same as the order of the initial conditions.

        """

        system = Stack(
            modules=[
                SymbolicModule(self.odes[species].equation) for species in species_order  # type: ignore
            ]
        )

        self.term = ODETerm(jax.jit(system.__call__))

    def _get_parameters(self) -> Dict[str, float]:
        """Gets all the parameters for the model"""

        if any(param.value is None for param in self.parameters.values()):
            raise ValueError("Missing values for parameters")

        return {param.name: param.value for param in self.parameters.values()}  # type: ignore

    def _assemble_y0_array(self, initial_conditions: Dict[str, float]) -> jnp.ndarray:
        """Assembles the initial conditions into an array"""

        assert all(
            species in initial_conditions for species in self.species.keys()
        ), f"Not all species have initial conditions specified. Please specify initial conditions for the following species: {set(self.species.keys()) - set(initial_conditions.keys())}"

        y0 = jnp.array([initial_conditions[species] for species in self.species.keys()])

        return y0

    # ! Helper methods

    def __repr__(self):
        """Prints a summary of the model"""

        print("# Species", end="\n\n")
        eqprint(
            Symbol("x"), Matrix([species.symbol for species in self.species.values()]).T
        )

        print("\n# Parameters", end="\n\n")
        eqprint(
            Symbol("theta"),
            Matrix([param.name for param in self.parameters.values()]).T,
        )

        print("\n# Equations", end="\n\n")
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
