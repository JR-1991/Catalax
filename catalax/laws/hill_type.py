"""
Hill-type Rate Law kinetic law functions.

This module contains kinetic law functions for the "Hill-type Rate Law" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS


def hill_type_microscopic_form(
    r: Annotated[str, "Concentration of reactant"],
    V_max: Annotated[str, "Maximal velocity"] = "V_max",
    K: Annotated[str, "Pseudo-dissociation constant"] = "K",
) -> Expr:
    """Hill equation rewritten by creating a pseudo-microscopic constant, equal to the Hill constant powered to the opposite of the Hill coefficient.

    Args:
        r (str): Concentration of reactant
        V_max (str): Maximal velocity
        K (str): Pseudo-dissociation constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V_max*R^h)/(K^h+R^h)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("V_max", locals=LOCALS)] = sympify(V_max, locals=LOCALS)
    substitutions[sympify("K", locals=LOCALS)] = sympify(K, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def hill_type_reduced_form(
    r: Annotated[str, "Concentration of reactant"],
    V_max: Annotated[str, "Maximal velocity"] = "V_max",
) -> Expr:
    """Hill equation rewritten by replacing the concentration of reactant with its reduced form, that is the concentration divide by a pseudo-microscopic constant, equal to the Hill constant powered to the opposite of the Hill coefficient.

    Args:
        r (str): Concentration of reactant
        V_max (str): Maximal velocity

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V_max*R^h)/(1+R^h)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("V_max", locals=LOCALS)] = sympify(V_max, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation
