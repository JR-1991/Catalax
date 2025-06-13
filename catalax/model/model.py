from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import arviz as az
from dotted_dict import DottedDict
from jax import Array
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from pyenzyme import EnzymeMLDocument
from sympy import Expr, Matrix, Symbol, symbols, sympify

from catalax.dataset.dataset import Dataset
from catalax.mcmc import priors
from catalax.model.base import CatalaxBase
from catalax.model.simconfig import SimulationConfig
from catalax.tools import Stack
from catalax.tools.simulation import Simulation
from .constant import Constant
from .inaxes import InAxes
from .ode import ODE
from .parameter import HDI, Parameter
from .species import Species
from .utils import PrettyDict, check_symbol, eqprint, odeprint
from ..predictor import Predictor

Y0_INDEX = 0
PARAMETERS_INDEX = 1
CONSTANTS_INDEX = 2
TIME_INDEX = 3


class Model(CatalaxBase, Predictor):
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    name: str
    odes: Dict[str, ODE] = Field(default_factory=DottedDict)
    species: Dict[str, Species] = Field(default_factory=PrettyDict)
    parameters: Dict[str, Parameter] = Field(default_factory=PrettyDict)
    constants: Dict[str, Constant] = Field(default_factory=PrettyDict)

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
        species. If a species hasn't been added to the model, it will be added automatically.

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
            self.add_species(species)

        self.odes[species] = ODE(
            equation=equation,
            species=self.species[species],
            observable=observable,
        )

        self.odes[species]._model = self

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

    def add_constant(self, constant_string: str = "", **constants_map):
        """
        Adds a constant or multiple constants to the model.

        This method will add new constants to the model and the model's constants dictionary,
        which can be accessed by object dot-notation. For example, if the constant is named 'c1'
        and the model is named 'model', the constant can be accessed by:

            model = Model(name="model")
            model.add_constant("c1")
            model.constants.c1 -> Constant(name="c1", symbol=c1)

        Constants can be added in two ways. The first is by passing a string of constant names that
        are separated by a comma. The second is by passing a dictionary of constant symbols (key)
        and names (values) and unpacking them as keyword arguments.

        The following are both valid ways to add constants:

            (1) model.add_constant("c1, c2, c3")
            (2) model.add_constant(**{"c1": "constant1", "c2": "constant2", "c3": "constant3"})

        If a constant already exists, a ValueError will be raised.

        Args:
            constant_string (str, optional): String of comma-separated constant symbols. Defaults to "".
            **constants_map (Dict[str, str]): Dictionary of constant symbols (key) and names (values).

        Raises:
            TypeError: If the constant names are not of type str.
        """
        if constant_string:
            constants_map.update(
                {
                    str(constant): str(constant)
                    for constant in self._split_species_string(constant_string)
                }
            )

        for symbol, name in constants_map.items():
            if not isinstance(name, str):
                raise TypeError("Constants names must be of type str")

            # Make sure the symbol is valid
            check_symbol(symbol)

            self.constants[symbol] = Constant(name=name, symbol=Symbol(symbol))

    # ! Simulation methods

    def simulate(
        self,
        dataset: Dataset,
        config: SimulationConfig,
        saveat: Union[jax.Array, None] = None,
        parameters: Optional[jax.Array] = None,
        in_axes: InAxes = InAxes.Y0,
        return_array: bool = False,
    ) -> Dataset:
        """
        Simulate the model with given dataset and parameters.

        Args:
            dataset: Dataset containing initial conditions and simulation data
            config: Optional simulation configuration parameters
            saveat: Optional array of time points to save results at
            parameters: Optional array of model parameters
            in_axes: Axes specification for vectorized operations
            return_array: Whether to return raw arrays instead of Dataset

        Returns:
            Either a tuple of (time points, results) arrays or a Dataset object
        """

        if config.nsteps is not None and isinstance(saveat, jax.Array):
            raise ValueError("Cannot specify both nsteps and saveat")

        # Initialize simulation data from dataset
        y0, constants = self._initialize_simulation_data(dataset)
        parameters = self._get_parameters(parameters)
        saveat = self._setup_saveat(config, saveat)

        if saveat.ndim != 1:
            inaxes: Tuple = in_axes + InAxes.TIME  # type: ignore
        else:
            inaxes: Tuple = in_axes.value  # type: ignore

        # Setup and run simulation
        if self._model_changed(inaxes, config.dt0) or self._sim_func is None:
            self._setup_simulation(config, inaxes)

        result = self._run_simulation(y0, parameters, constants, saveat)

        return self._format_simulation_results(
            result,
            saveat,
            return_array,
            y0,
            constants,
        )  # type: ignore

    def predict(
        self,
        dataset: Dataset,
        config: SimulationConfig | None = None,
        n_steps: int = 100,
    ) -> Dataset:
        """Predict model behavior using the given dataset.

        This is a convenience wrapper around the `simulate` method that automatically
        creates a simulation configuration if one is not provided.

        Args:
            dataset: Dataset containing initial conditions for prediction
            config: Optional simulation configuration parameters. If None,
                   a configuration will be created from the dataset.
            nsteps: Number of time steps for the simulation. This will override
                   the nsteps value in the provided config if both are specified.

        Returns:
            A Dataset object containing the prediction results
        """
        if config is None:
            config = dataset.to_config(nsteps=n_steps)

        if config.nsteps != n_steps:
            config.nsteps = n_steps

        return self.simulate(dataset, config)

    def _initialize_simulation_data(
        self,
        dataset: Dataset,
    ) -> Tuple[jax.Array, jax.Array]:
        """Initialize simulation data from dataset."""
        dataset = copy.deepcopy(dataset)
        return (
            dataset.to_y0_matrix(species_order=self.get_species_order()),
            dataset.to_y0_matrix(species_order=self.get_constants_order()),
        )

    def _setup_saveat(
        self,
        config: SimulationConfig,
        saveat: Optional[jax.Array],
    ) -> jax.Array:
        """Setup save points for simulation."""
        if config.nsteps is not None:
            saveat = jnp.linspace(config.t0, config.t1, config.nsteps).T

        if isinstance(saveat, jax.Array):
            return saveat
        else:
            raise ValueError(
                "No saveat or config.nsteps provided. Please provide one of the two."
            )

    def _setup_simulation(
        self,
        config: SimulationConfig,
        in_axes: Tuple,
    ) -> None:
        """Setup the simulation function with given configuration."""
        self._setup_system(
            in_axes=in_axes,
            t0=config.t0,
            t1=config.t1,
            dt0=config.dt0,
            solver=config.solver,
            rtol=config.rtol,
            atol=config.atol,
            max_steps=config.max_steps,
        )

        self._in_axes = in_axes
        self._dt0 = config.dt0  # type: ignore

    def _run_simulation(
        self,
        y0: jax.Array,
        parameters: jax.Array,
        constants: jax.Array,
        saveat: jax.Array,
    ) -> jax.Array:
        """Run the simulation with given parameters."""
        if self._sim_func is None:
            raise RuntimeError("Simulation function not initialized")

        result = self._sim_func(y0, parameters, constants, saveat)
        if len(result.shape) != 3:
            result = jnp.expand_dims(result, axis=0)
        return result

    def _format_simulation_results(
        self,
        result: jax.Array,
        saveat: jax.Array,
        return_array: bool,
        y0: jax.Array,
        constants: jax.Array,
    ) -> Union[tuple[Array, Array], Dataset]:
        """Format simulation results according to return type."""
        if return_array:
            return saveat, result

        dataset = Dataset.from_model(self)
        for y0_i in range(result.shape[0]):
            init_conc = {
                species: float(y0[y0_i, sp_j])
                for sp_j, species in enumerate(self.get_species_order())
            }

            for cons_i, constant in enumerate(self.get_constants_order()):
                init_conc[constant] = float(constants[y0_i, cons_i])

            if saveat.ndim == 1:
                local_saveat = saveat
            else:
                local_saveat = saveat[y0_i, :]

            dataset.add_from_jax_array(
                species_order=self.get_species_order(),
                data=result[y0_i, :, :],
                time=local_saveat,
                initial_condition=init_conc,
            )
        return dataset

    def _model_changed(
        self,
        in_axes: Tuple,
        dt0: float,
    ) -> bool:
        """Check if model configuration has changed."""
        return self._in_axes != in_axes or self._dt0 != dt0

    def _get_stoich_mat(self) -> jax.Array:
        """Creates the stoichiometry matrx based on the given model.

        Models can be defined in both ways - First by using plain ODE models where each species
        has a single assigned equation. Here, the resulting stoichiometry matrx is a unity matrix.
        In the case of a model, that is defined by reactions, the stoichioemtry matrix accordingly
        is constructed per species and reaction.
        """

        if hasattr(self, "reactions"):
            raise NotImplementedError(
                "So far stoichioemtry matrix construction is exclusive to ODE models and not implemented yet."
            )

        stoich_mat = jnp.zeros((len(self.odes), len(self.odes)))
        diag_indices = jnp.diag_indices_from(stoich_mat)

        return stoich_mat.at[diag_indices].set(1)

    def _setup_system(self, in_axes: Tuple = (0, None, None), **kwargs):
        """Converts given SymPy equations into Equinox modules, used for simulation.

        This method will prepare the simulation function, jit it and vmap it across
        the initial condition vector by default.

        Args:
            in_axes: Specifies the axes to map the simulation function across.
                     Defaults to (0, None, None).
        """

        # Retrieve ODEs based on species order
        odes = [self.odes[species] for species in self.get_species_order()]
        simulation_setup = Simulation(
            odes=odes,
            parameters=self.get_parameter_order(),
            stoich_mat=self._get_stoich_mat(),
            constants=self.get_constants_order(),
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

        return jnp.array(
            [self.parameters[param].value for param in self.get_parameter_order()]
        )  # type: ignore

    def get_parameter_order(self) -> List[str]:
        """Returns the order of the parameters in the model"""
        return sorted(self.parameters.keys())

    def get_species_order(self) -> List[str]:
        """Returns the order of the species ad constants in the model"""
        return sorted(list(self.species.keys()))

    def get_observable_species_order(self) -> List[str]:
        """Returns the order of the observable species in the model"""
        if not self.odes:
            # When there are no ODEs, all species are observable
            return self.get_species_order()

        return sorted([key for key in self.species.keys() if self.odes[key].observable])

    def get_constants_order(self) -> List[str]:
        """Returns the order of the constants in the model"""
        return sorted(list(self.constants.keys()))

    def _warmup_simulation(
        self,
        y0,
        parameters,
        constants,
        saveat,
        in_axes,
    ):
        """Warms up the simulation to make use of jit compilation"""

        if self._sim_func is None:
            raise ValueError(
                "Simulation function not found. Please run 'simulate' first."
            )

        if in_axes is None:
            self._sim_func(y0, parameters, constants, saveat)
            return

        y0_axis, parameters_axis, constants_axis, saveat_axis = in_axes

        # if y0_axis is not None:
        #     y0 = jnp.expand_dims(y0[0, :], axis=0)
        # if parameters_axis is not None:
        #     parameters = jnp.expand_dims(parameters[0, :], axis=0)
        # if saveat_axis is not None:
        #     saveat = jnp.expand_dims(saveat[0, :], axis=0)

        self._sim_func(
            y0=y0,
            parameters=parameters,
            constants=constants,
            time=saveat,
        )

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

        assert in_axes is None or len(in_axes) == 4, (
            "Got invalid dimension for 'in_axes' - Needs to have four ints/Nones"
        )

        odes = [self.odes[species] for species in self.get_species_order()]
        fun = Stack(
            odes=odes,
            parameters=self.get_parameter_order(),
            constants=self.get_constants_order(),
        )

        if in_axes is None:
            return fun

        return jax.jit(jax.vmap(fun, in_axes=in_axes))

    def reset(self):
        """Resets the model"""

        self._sim_func = None
        self._in_axes = None

    # ! Derivatives

    def jacobian_parameters(
        self,
        dataset: Dataset,
        parameters: jax.Array,
    ):
        """Sets up and calculates the jacobian of the model with respect to the parameters.

        Args:
            y (Array): The inital conditions of the model.
            parameters (Array): The parameters that are inserted into the model.

        Returns:
            Array: The jacobian of the model with respect to the parameters.
        """

        _, y0s, time = dataset.to_jax_arrays(
            self.get_species_order(),
            inits_to_array=True,
        )

        if self._jacobian_parameters is not None:
            return self._jacobian_parameters(time, y0s, parameters)

        stoich_mat = self._get_stoich_mat()

        def _vector_field(t, y, parameters):
            return self._sim_func(y, parameters, constants, saveat)

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

        if self._sim_func is None:
            self._setup_system(in_axes=InAxes.Y0.value)

        def _vector_field(y, parameters):
            return self._sim_func(y, parameters, constants, saveat)

        self._jacobian_states = jax.jacfwd(_vector_field, argnums=0)

        return self._jacobian_states(y, parameters)

    # ! Helper methods
    def has_hdi(self):
        """Checks whether the model has HDI values for any of the parameters"""

        return any(parameter.hdi is not None for parameter in self.parameters.values())

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

    @field_validator("species", mode="before")
    def _convert_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        symbols_ = []

        for symbol in value:
            if isinstance(symbol, str):
                symbol = symbols(symbol)

            symbols_ += list(symbol)

        return symbols_

    def y0_array_to_dict(self, y0_array: jax.Array) -> List[Dict[str, float]]:
        """
        Converts a 1D or 2D array of initial values to a list of dictionaries.

        Args:
            y0_array (jax.Array): The array of initial values.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, where each dictionary represents
            the initial values for each species in the model.

        Raises:
            AssertionError: If the length of y0 array does not match the number of species in the model.
        """

        if len(y0_array.shape) == 1:
            y0_array = jnp.expand_dims(y0_array, axis=0)

        assert y0_array.shape[-1] == len(self.species), (
            "Length of y0 array does not match the number of species in the model."
        )

        return [
            {
                species: float(value)
                for species, value in zip(self.get_species_order(), y0)
            }
            for y0 in y0_array
        ]

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
        model_dict = DottedDict(self.model_dump(exclude_none=True))
        res = {
            "name": model_dict.name,
            "species": [species.to_dict() for species in model_dict.species.values()],
            "constants": [
                constant.to_dict() for constant in model_dict.constants.values()
            ],
            "odes": [
                {**ode.to_dict(), "species": species}
                for species, ode in model_dict.odes.items()
            ],
            "parameters": [
                parameter.to_dict() for parameter in model_dict.parameters.values()
            ],
        }

        if not res["constants"]:
            res.pop("constants")

        return res

    # ! Metrics
    def calculate_aic(
        self,
        dataset: Dataset,
    ) -> float:
        """Calculates the AIC value to the given data if parameters have values.

        Args:
            dataset (Dataset): Dataset to calculate the AIC for.

        Returns:
            float: AIC criterion of the model given the data.

        Raises:
            AssertionError: If the model hasn't been fitted yet (parameters have no values).
        """
        assert self.parameters and all(
            parameter.value is not None for parameter in self.parameters.values()
        ), "Cannot calculate AIC, because this model hasn't been fitted yet."

        # Simulate using the dataset
        _, states = self.simulate(
            dataset=dataset,
            saveat=dataset.time,
            in_axes=(0, None, 0),
            return_array=True,
        )

        # Get observables
        observables = [
            index
            for index, species in enumerate(self.get_species_order())
            if self.odes[species].observable
        ]

        # Get the actual data from the dataset
        data = dataset.to_array()

        # Calculate residuals and AIC
        residual = states[..., observables] - data
        chisqr = (residual**2).sum()
        ndata = len(residual.ravel())
        _neg2_log_likel = ndata * jnp.log(chisqr / ndata)
        aic = _neg2_log_likel + 2 * len(self.parameters)

        return aic

    # ! Updaters
    def from_samples(
        self,
        samples: Dict[str, Array],
        hdi_prob: float = 0.95,
    ) -> "Model":
        """Create a new model from samples drawn from the posterior distribution.

        Args:
            samples (Dict[str, Array]): The samples to update the parameters with.
        """

        new_model = copy.deepcopy(self)
        hdi = az.hdi(samples, hdi_prob=hdi_prob)

        for name, parameter in new_model.parameters.items():
            parameter.value = samples[name].mean()
            parameter.initial_value = samples[name].mean()
            parameter.hdi = HDI(lower=hdi[name][0], upper=hdi[name][1], q=hdi_prob)

        return new_model

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

        if "constants" in model_dict and model_dict.constants:
            model.add_constant(**{sp.symbol: sp.name for sp in model_dict.constants})

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

    @classmethod
    def from_enzymeml(cls, path: Path | str, name: str | None = None):
        with open(path, "r") as f:
            enzmldoc = EnzymeMLDocument.model_validate_json(f.read())

        if name is None:
            name = enzmldoc.name

        model = cls(name=name)

        # get observables
        observables = set()
        non_observables = set()
        for measurement in enzmldoc.measurements:
            for species_data in measurement.species_data:
                if species_data.data:
                    observables.add(species_data.species_id)
                else:
                    non_observables.add(species_data.species_id)

        all_species = observables | non_observables

        # create regex match objects for all species
        species_regex = re.compile(r"|".join(map(re.escape, all_species)))

        for reaction in enzmldoc.reactions:
            # get species from reaction
            match_species = species_regex.findall(reaction.kinetic_law.equation)

            # add species to model
            for species in set(match_species):
                model.add_species(species)

            # add ode to model
            model.add_ode(
                species=reaction.kinetic_law.species_id,
                equation=reaction.kinetic_law.equation,
                observable=True
                if reaction.kinetic_law.species_id in observables
                else False,
            )

        # add for all model.species that don't have an ode, a constant rate of 0
        for species in model.species:
            if species not in model.odes:
                model.add_ode(species=species, equation="0", observable=False)

        return model

    def update_enzymeml_parameters(self, enzmldoc: EnzymeMLDocument):
        """Updates model parameters of enzymeml document with model parameters.
        Existing parameters will be updated, non-existing parameters will be added.

        Args:
            enzmldoc (EnzymeMLDocument): EnzymeML document to update.
        """

        enzml_param_ids = [param.id for param in enzmldoc.parameters]

        for parameter in self.parameters.values():
            # update existing parameter
            if parameter.name in enzml_param_ids:
                enzymeml_param = next(
                    p for p in enzmldoc.parameters if p.id == parameter.name
                )
                enzymeml_param.value = parameter.value
                enzymeml_param.initial_value = parameter.initial_value
                enzymeml_param.constant = parameter.constant
                enzymeml_param.upper = parameter.upper_bound
                enzymeml_param.lower = parameter.lower_bound

            # add new parameter
            else:
                enzmldoc.add_to_parameters(
                    id=parameter.name,
                    symbol=parameter.name,
                    name=parameter.name,
                    value=parameter.value,
                    initial_value=parameter.initial_value,
                    constant=parameter.constant,
                    upper=parameter.upper_bound,
                    lower=parameter.lower_bound,
                )
