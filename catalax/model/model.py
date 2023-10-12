import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pandas as pd
from diffrax import ODETerm, SaveAt, Tsit5
from dotted_dict import DottedDict
from pydantic import Field, PrivateAttr, validator
from sympy import Expr, Matrix, Symbol, symbols, sympify

from catalax.model.base import CatalaxBase
from catalax.mcmc import priors
from catalax.tools import Stack
from catalax.tools.simulation import Simulation

from .ode import ODE
from .parameter import Parameter
from .species import Species
from .utils import PrettyDict, check_symbol, eqprint, odeprint
from .inaxes import InAxes


class Model(CatalaxBase):

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
        fields = {
            "term": {"exclude": True},
        }

    name: str
    odes: Dict[str, ODE] = Field(default_factory=DottedDict)
    species: Dict[str, Species] = Field(default_factory=PrettyDict)
    parameters: Dict[str, Parameter] = Field(default_factory=PrettyDict)
    term: Optional[ODETerm] = Field(default=None)

    _sim_func: Optional[Callable] = PrivateAttr(default=None)
    _in_axes: Optional[Tuple] = PrivateAttr(default=None)
    _dt0: Optional[Tuple] = PrivateAttr(default=None)
    _jacobian_parameters: Optional[Callable] = PrivateAttr(default=None)
    _jacobian_states: Optional[Callable] = PrivateAttr(default=None)
    _sensitivity: Optional[InAxes] = PrivateAttr(default=None)

    def add_ode(
        self,
        species: str,
        equation: str,  # type: ignore
        observable: bool = True,
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
            equation=equation, species=self.species[species], observable=observable
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
                }  # type: ignore
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

    # ! Simulation methods

    def simulate(
        self,
        initial_conditions: List[Dict[str, float]],
        dt0: float = 0.1,
        solver=Tsit5,
        t0: Optional[int] = None,
        t1: Optional[int] = None,
        nsteps: Optional[int] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        saveat: Union[SaveAt, jax.Array] = None,
        parameters: Optional[jax.Array] = None,
        in_axes: Optional[Tuple] = None,
        max_steps: int = 4096,
        sensitivity: Optional[InAxes] = None,
    ):
        """Simulates the given model"""

        if isinstance(in_axes, InAxes):
            in_axes = in_axes.value

        if nsteps and saveat:
            raise ValueError(
                "Cannot specify both nsteps and saveat. Please choose one."
            )
        elif nsteps is None and saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        if isinstance(initial_conditions, dict):
            initial_conditions = [initial_conditions]

        parameters = self._get_parameters(parameters)

        # Setup the initial conditions
        y0 = self._assemble_y0_array(initial_conditions, in_axes)

        # Setup save points
        if nsteps is not None:
            saveat = jnp.linspace(t0, t1, nsteps)  # type: ignore
            t0 = saveat[0]  # type: ignore
            t1 = saveat[-1]  # type: ignore
        elif saveat is None:
            raise ValueError("Must specify either nsteps or saveat.")

        if self._model_changed(in_axes, dt0, sensitivity) or self._sim_func is None:
            self._setup_system(
                in_axes=in_axes,
                t0=t0,
                t1=t1,
                dt0=dt0,
                solver=solver,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                sensitivity=sensitivity,
            )

            # Set markers to check whether the conditions have changed
            # This is done to avoid recompilation of the simulation function
            self._in_axes = in_axes
            self._dt0 = dt0
            self._sensitivity = sensitivity

            # Warmup the simulation to make use of jit compilation
            self._warmup_simulation(y0, parameters, saveat, in_axes)

            return self._sim_func(y0, parameters, saveat)

        return self._sim_func(y0, parameters, saveat)

    def _get_stoich_mat(self) -> jax.Array:
        """Creates the stoichiometry matrx based on the given model.

        Models can be defined in both ways - First by using plain ODE models where each species
        has a single assigned equation. Here, the resulting stoichiometry matrx is a unity matrix.
        In the case of a model, that is defined by reactions, the stoichioemtry matrix accordingly
        is constructed per species and reaction.
        """

        if hasattr(self, "reactions"):
            raise NotImplementedError(
                f"So far stoichioemtry matrix construction is exclusive to ODE models and not implemented yet."
            )

        stoich_mat = jnp.zeros((len(self.odes), len(self.odes)))
        diag_indices = jnp.diag_indices_from(stoich_mat)

        return stoich_mat.at[diag_indices].set(1)

    def _model_changed(self, in_axes: Tuple, dt0, sensitivity: InAxes) -> bool:
        """Checks whether the model has changed since the last simulation"""

        if self._in_axes != in_axes:
            return True

        if self._dt0 != dt0:
            return True

        if self._sensitivity != sensitivity:
            return True

        return False

    def _setup_system(self, in_axes: Tuple = (0, None, None), **kwargs):
        """Converts given SymPy equations into Equinox modules, used for simulation.

        This method will prepare the simulation function, jit it and vmap it across
        the initial condition vector by default.

        Args:
            in_axes: Specifies the axes to map the simulation function across.
                     Defaults to (0, None, None).
        """

        # Retrieve ODEs based on species order
        odes = [self.odes[species] for species in self._get_species_order()]
        simulation_setup = Simulation(
            odes=odes,
            parameters=self._get_parameter_order(),
            stoich_mat=self._get_stoich_mat(),
            **kwargs,
        )

        self._sim_func = simulation_setup._prepare_func(in_axes=in_axes)

    def _get_parameters(self, parameters: Optional[jax.Array] = None) -> jax.Array:
        """Gets all the parameters for the model"""

        if (
            any(param.value is None for param in self.parameters.values())
            and parameters is None
        ):
            # If no 'custom' parameters are given, raise an exception
            raise ValueError("Missing values for parameters")

        if parameters is not None:
            return parameters

        return jnp.array([self.parameters[param].value for param in self._get_parameter_order()])  # type: ignore

    def _get_parameter_order(self) -> List[str]:
        """Returns the order of the parameters in the model"""
        return sorted(self.parameters.keys())

    def _get_species_order(self) -> List[str]:
        """Returns the order of the species in the model"""
        return sorted(self.species.keys())

    def _assemble_y0_array(
        self, initial_conditions: List[Dict[str, float]], in_axes: Tuple
    ) -> jax.Array:
        """Assembles the initial conditions into an array"""

        # Turn initial conditions dict into a dataframe
        df_inits = pd.DataFrame(initial_conditions)

        # Check whether all species have initial conditions and
        # if there are more than in the model
        if not all(species in self.species.keys() for species in df_inits.columns):
            raise ValueError(
                f"Not all species have initial conditions specified or there are ones that havent been specified yet. Please specify initial conditions or remove these for the following species: {set(self.species.keys()) - set(df_inits.columns)}"
            )

        if in_axes is not None and len(initial_conditions) > 1:
            return jnp.stack(
                [df_inits[s].values for s in self._get_species_order()], axis=-1  # type: ignore
            )
        elif in_axes and len(initial_conditions) > 1:
            raise ValueError(
                "If in_axes is set to None, only one initial condition can be specified."
            )

        return jnp.array(
            [initial_conditions[0][species] for species in self._get_species_order()]
        )

    def _warmup_simulation(self, y0, parameters, saveat, in_axes):
        """Warms up the simulation to make use of jit compilation"""

        if self._sim_func is None:
            raise ValueError(
                "Simulation function not found. Please run 'simulate' first."
            )

        if in_axes is None:
            self._sim_func(y0, parameters, saveat)
            return

        y0_axis, parameters_axis, saveat_axis = in_axes

        if y0_axis is not None:
            y0 = jnp.expand_dims(y0[0, :], axis=0)
        if parameters_axis is not None:
            parameters = jnp.expand_dims(parameters[0, :], axis=0)
        if saveat_axis is not None:
            saveat = jnp.expand_dims(saveat[0, :], axis=0)

        self._sim_func(y0, parameters, saveat)

    def _setup_rate_function(self, in_axes=None):
        """Prepares a function to evaluate the rates of the vector field.

        This method in particualar is useful to fit rates to the neural ODE and
        thus connect an abstract neural ODE to a physical model. Mainly, this is
        a very efficient approach that mitigates the performance issues of MCMC
        sampling that necessitates full simulations. Thus, by obtaining the rates
        predicted by the neural ODE fitting these to the physical model is faster
        and can be done in parallel.

        Args:
            model (Model): Catalax model to prepare the rate function for.
        """

        assert (
            in_axes is None or len(in_axes) == 3
        ), "Got invalid dimension for 'in_axes' - Needs to have three ints/Nones"

        odes = [self.odes[species] for species in self._get_species_order()]
        fun = Stack(odes=odes, parameters=self._get_parameter_order())

        if in_axes is None:
            return fun

        return jax.jit(jax.vmap(fun, in_axes=in_axes))

    def reset(self):
        """Resets the model"""

        self._sim_func = None
        self._in_axes = None

    # ! Derivatives

    def jacobian_parameters(self, y, parameters, t=0):
        """Sets up and calculates the jacobian of the model with respect to the parameters.

        Args:
            y (Array): The inital conditions of the model.
            parameters (Array): The parameters that are inserted into the model.

        Returns:
            Array: The jacobian of the model with respect to the parameters.
        """

        if self._jacobian_parameters is not None:
            return self._jacobian_parameters(t, y, parameters)

        stoich_mat = self._get_stoich_mat()

        def _vector_field(t, y, parameters):
            return self._setup_rate_function()(t, y, (parameters, stoich_mat))

        self._jacobian_parameters = jax.jacfwd(_vector_field, argnums=2)

        return self._jacobian_parameters(t, y, parameters)

    def jacobian_states(self, y, parameters):
        """Sets up and calculates the jacobian of the model with respect to the states.

        Args:
            y (Array): The inital conditions of the model.
            parameters (Array): The parameters that are inserted into the model.

        Returns:
            Array: The jacobian of the model with respect to the states.
        """

        if self._jacobian_states is not None:
            return self._jacobian_states(y, parameters)
        elif self.term is None:
            self._setup_term()

        species_maps = {symbol: i for i, symbol in enumerate(self._get_species_order())}
        parameter_maps = {
            symbol: i for i, symbol in enumerate(self._get_parameter_order())
        }

        def _vector_field(y, parameters):
            return self.term.vector_field(
                0, y, (species_maps, parameter_maps, parameters)
            )

        self._jacobian_states = jax.jacfwd(_vector_field, argnums=0)

        return self._jacobian_states(y, parameters)

    # ! Helper methods

    def __repr__(self):
        """Prints a summary of the model"""

        print("Model summary")

        eqprint(
            Symbol("x"), Matrix([species.symbol for species in self.species.values()]).T
        )

        eqprint(
            Symbol("theta"),
            Matrix([param.name for param in self.parameters.values()]).T,
        )

        for ode in self.odes.values():
            odeprint(y=ode.species.name, expr=ode.equation)

        # Parameter
        self.parameters.__repr__()

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

    # ! Exporters
    def save(self, path: str, name: Optional[str] = None, **json_kwargs):
        """Saves a model to a JSON file.

        Args:
            path (str): Path to which the model will be serialized
        """

        if name is None:
            name = self.name.replace(" ", "_")

        fpath = os.path.join(path, f"{name}.json")

        with open(fpath, "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(),
                    default=str,
                    sort_keys=False,
                    indent=2,
                    **json_kwargs,
                )
            )

    def to_dict(self):
        """Converts the model into a serializable dictionary."""
        model_dict = DottedDict(self.dict(exclude_none=True))
        return {
            "name": model_dict.name,
            "species": [species.to_dict() for species in model_dict.species.values()],
            "odes": [
                {**ode.to_dict(), "species": species}
                for species, ode in model_dict.odes.items()
            ],
            "parameters": [
                parameter.to_dict() for parameter in model_dict.parameters.values()
            ],
        }

    # ! Metrics

    def calculate_aic(
        self,
        data: jax.Array,
        initial_conditions: List[Dict[str, float]],
        times: jax.Array,
    ):
        """Calculates the AIC value to the given data if parameters have values.

        Args:
            data (jax.Array): Data to check against.
            initial_conditions (List[Dict[str, float]]): Initial conditions to perform the integration.
            times (jax.Array): Time points to evaluate

        Returns:
            float: AIC criterion of the model given the data.
        """

        assert self.parameters and all(
            parameter.value is not None for parameter in self.parameters.values()
        ), "Cannot calculate AIC, because this model hasnt been fitted yet."

        _, states = self.simulate(
            initial_conditions=initial_conditions, saveat=times, in_axes=(0, None, 0)
        )

        residual = states - data
        chisqr = (residual**2).sum()
        ndata = len(residual.ravel())
        _neg2_log_likel = ndata * jnp.log(chisqr / ndata)
        aic = _neg2_log_likel + 2 * len(self.parameters)

        return aic

    # ! Importers
    @classmethod
    def load(cls, path: str):
        """Loads a model from a JSON file and initializes a model.

        Args:
            path (str): Path to the model JSON file.

        Raises:
            ValueError: If parameters given in the JSON file are not present in the model.

        Returns:
            Model: Resulting model instance.
        """

        with open(path, "r") as f:
            data = DottedDict(json.load(f))
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, model_dict: Dict):
        """Initializes a model from a dictionary."""

        if not isinstance(model_dict, DottedDict):
            model_dict = DottedDict(model_dict)

        # Initialize the model
        model = cls(name=model_dict.name)

        # Add Species
        model.add_species(**{sp.symbol: sp.name for sp in model_dict.species})

        # Add ODEs
        for ode in model_dict.odes:
            model.add_ode(**ode)

        # Update given parameters
        for parameter in model_dict.parameters:
            if parameter.symbol not in model.parameters:
                raise ValueError(
                    f"Parameter [symbol: {parameter.symbol}, name: {parameter.name}] not found in the model and thus inconsistent with the given model. Please check the JSON file you are trying to load."
                )

            if "prior" in parameter:
                prior = parameter.pop("prior")
                prior_cls = priors.__dict__[prior.pop("type")]
                model.parameters[parameter.symbol].prior = prior_cls(**prior)

            model.parameters[parameter.symbol].__dict__.update(parameter)

        return model
