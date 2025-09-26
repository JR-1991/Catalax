"""
Activation kinetic law functions.

This module contains kinetic law functions for the "Activation" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS


def irreversible_substrate_activation(
    s: Annotated[str, "Concentration of substrate"],
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_sc: Annotated[str, "Dissociation constant"] = "K_sc",
    K_sa: Annotated[str, "Dissociation constant"] = "K_sa",
) -> Expr:
    """This enzymatic rate law is available only for irreversible reactions, with one substrate and one product. There is a second binding site for the enzyme which, when occupied, activates the enzyme. Substrate binding at either site can occur at random.


    Args:
        s (str): Concentration of substrate
        V (str): Forward maximal velocity
        K_sc (str): Dissociation constant
        K_sa (str): Dissociation constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V*((S)/(K_sa))^2)/(1+(S)/(K_sc)+(S)/(K_sa)+((S)/(K_sa))^2)", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_sc", locals=LOCALS)] = sympify(K_sc, locals=LOCALS)
    substitutions[sympify("K_sa", locals=LOCALS)] = sympify(K_sa, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irrreversible_mixed_activation(
    s: Annotated[str, "Concentration of substrate"],
    a: Annotated[str, "Concentration of activator"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_as: Annotated[str, "Activation constant"] = "K_as",
    K_ac: Annotated[str, "Activation constant"] = "K_ac",
) -> Expr:
    """Enzymatic rate law where the activator enhances the rate of reaction through specific and catalytic effects, which increase the apparent limiting rate and decrease apparent Michaelis constant. The activator can bind irreversibly both free enzyme and enzyme-substrate complex, while the substrate can bind only to enzyme-activator complex. Catalytic activity is seen only when enzyme, substrate and activator are complexed.

    Args:
        s (str): Concentration of substrate
        a (str): Concentration of activator
        K_ms (str): Michaelis constant for substrate
        V (str): Forward maximal velocity
        K_as (str): Activation constant
        K_ac (str): Activation constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V*S*A)/(K_ms*(K_as+A)+S*(K_ac+A))", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("A", locals=LOCALS)] = sympify(a, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_as", locals=LOCALS)] = sympify(K_as, locals=LOCALS)
    substitutions[sympify("K_ac", locals=LOCALS)] = sympify(K_ac, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_catalytic_activation_with_one_activator(
    s: Annotated[str, "Concentration of substrate"],
    a: Annotated[str, "Concentration of activator"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_a: Annotated[str, "Activation constant"] = "K_a",
) -> Expr:
    """Enzymatic rate law where an activator enhances the rate of reaction by increasing the apparent limiting rate; The activator binding to the enzyme-substrate complex (irreversibly) is required for enzyme catalytic activity (to generate the product).

    Args:
        s (str): Concentration of substrate
        a (str): Concentration of activator
        K_ms (str): Michaelis constant for substrate
        V (str): Forward maximal velocity
        K_a (str): Activation constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V*S*A)/((K_ms+S)*(K_a+A))", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("A", locals=LOCALS)] = sympify(a, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_a", locals=LOCALS)] = sympify(K_a, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_specific_activation(
    s: Annotated[str, "Concentration of substrate"],
    a: Annotated[str, "Concentration of activator"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_a: Annotated[str, "Activation constant"] = "K_a",
) -> Expr:
    """Enzymatic rate law for one substrate, one product and one modifier which acts as an activator. The activator enhances the rate of reaction by decreasing the apparent Michaelis constant. The activator must bind to the enzyme before the enzyme can bind the substrate.

    Args:
        s (str): Concentration of substrate
        a (str): Concentration of activator
        K_ms (str): Michaelis constant for substrate
        V (str): Forward maximal velocity
        K_a (str): Activation constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V*S*A)/(K_ms*K_a+(K_ms+S)*A)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("A", locals=LOCALS)] = sympify(a, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_a", locals=LOCALS)] = sympify(K_a, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation
