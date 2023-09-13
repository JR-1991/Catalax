from enum import Enum
from typing import Dict, List
from importlib import resources as pkg_resources
from jinja2 import Template

import pandas as pd

from catalax import Model
from catalax.identifiability import templates as jinja_templates

try:
    from julia import Main
except ImportError:
    raise ImportError(
        "The julia package is required for the identifiability analysis. "
        "Please install it using `pip install julia jill` and the following command:\n\n"
        "python -c 'import julia; julia.install()'\n\n"
    )


class IdentPackages(Enum):
    SIAN = "SIAN"
    SCIML = "StructuralIdentifiability"


def SIAN(model: Model, update_model: bool = True) -> pd.DataFrame:
    """Structural identifiability analysis using SIAN.

    The SIAN package is a Julia package for structural identifiability analysis
    of ODE models. It is capable of analyzing nonlinear ODE models and assess
    local and global identifiability. The package is available at:

    https://github.com/alexeyovchinnikov/SIAN-Julia

    For more information on the SIAN method, see:

    Hoon Hong, Alexey Ovchinnikov, Gleb Pogudin, Chee Yap,
    SIAN: software for structural identifiability analysis of ODE models,
    Bioinformatics, Volume 35, Issue 16, August 2019, Pages 2873–2874,
    https://doi.org/10.1093/bioinformatics/bty1069

    Args:
        model (Model): The model to be analyzed.
        update_model (bool, optional): If True, the model will be updated with the results of the analysis. Defaults to True.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the analysis.
    """

    try:
        from julia.SIAN import get_parameters, identifiability_ode
    except ImportError:
        print("Installing SIAN...")
        Main.eval(
            'using Pkg; Pkg.add("https://github.com/alexeyovchinnikov/SIAN-Julia.git")'
        )
        from julia.SIAN import get_parameters, identifiability_ode

    all_species = list(model.species)
    odemodel = _prepare_jl_input(model, IdentPackages.SIAN)
    result = {
        Main.string(param): res
        for res, params in identifiability_ode(
            odemodel, get_parameters(odemodel)
        ).items()
        for param in Main.collect(params)
        if not Main.string(param) in all_species
    }

    if update_model:
        for param, res in result.items():
            model.parameters[param].identifiability = {
                "result": res,
                "method": "SIAN",
                "package": "https://github.com/alexeyovchinnikov/SIAN-Julia",
            }  # type: ignore

    return _create_result_df(result, IdentPackages.SIAN)


def SciML(model: Model, update_model: bool = True) -> pd.DataFrame:
    """Structural identifiability using SciML.

    The StructuralIdentifiability package is a Julia package for structural
    identifiability analysis of ODE models. It is capable of analyzing
    nonlinear ODE models and assess local and global identifiability.
    The package is available at:

    https://github.com/SciML/StructuralIdentifiability.jl

    Args:
        model (Model): The model to be analyzed.
        update_model (bool, optional): If True, the model will be updated with the results of the analysis. Defaults to True.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the analysis.
    """

    try:
        from julia.StructuralIdentifiability import assess_identifiability
    except ImportError:
        print("Installing StructuralIdentifiability...")
        Main.eval('using Pkg; Pkg.add("StructuralIdentifiability")')
        from julia.StructuralIdentifiability import assess_identifiability

    odemodel = _prepare_jl_input(model, IdentPackages.SCIML)
    result = {
        Main.string(param): res
        for param, res in assess_identifiability(odemodel).items()
    }

    if update_model:
        for param, res in result.items():
            model.parameters[param].identifiability = {
                "result": res,
                "method": "SciML",
                "package": "https://github.com/SciML/StructuralIdentifiability.jl",
            }  # type: ignore

    return _create_result_df(result, IdentPackages.SCIML)


def _prepare_jl_input(model: Model, ident_pkg: IdentPackages):
    """Prepares the Julia input for the identifiability analysis"""

    print(f"Assessing identifiability using {ident_pkg.name}...\n")

    # Convert to Julia compatible strings
    all_species = list(model.species)

    jl_obs = [
        (index, str(species))
        for index, (species, ode) in enumerate(model.odes.items())
        if ode.observable
    ]

    jl_odes = {
        species: _to_jl_compatible_eq(str(ode.equation), all_species)
        for species, ode in model.odes.items()
    }

    # Generate Julia input
    template = Template(pkg_resources.read_text(jinja_templates, "odemodel.jinja2"))
    jl_input = template.render(
        package=str(ident_pkg.value),
        odes=jl_odes,
        observables=jl_obs,
    )

    return Main.eval(jl_input)


def _to_jl_compatible_eq(equation: str, all_species: List[str]):
    """Converts Catalax equations to Julia compatible equations

    For example:
        s0 * p11**2 -> s0(t) * p11^2
    """

    equation = equation.replace("**", "^")
    for species in all_species:
        equation = equation.replace(species, f"{species}(t)")

    return equation


def _create_result_df(results: Dict[str, str], ident_pkg: IdentPackages):
    """Creates a DataFrame from the results of the identifiability analysis"""
    return pd.DataFrame(
        {
            "Parameter": list(results.keys()),
            "Identifiability": list(results.values()),
            "Method": ident_pkg.name,
        }
    )
