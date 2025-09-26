"""
Mass Action Rate Law kinetic law functions.

This module contains kinetic law functions for the "Mass Action Rate Law" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS


def zeroth_order_irreversible_reactions(
    k: Annotated[str, "Forward zeroth order rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is constant.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        k (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def monoexponential_decay(
    r: Annotated[str, "Concentration of reactant"],
    l: Annotated[str, "Half-life of an exponential decay"] = "l",
) -> Expr:
    """Monotonic decrease of a quantity proportionally to its value.

    Args:
        r (str): Concentration of reactant
        l (str): Half-life of an exponential decay

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(R)/(l)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("l", locals=LOCALS)] = sympify(l, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_irreversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    k: Annotated[str, "Forward bimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the square of one reactant quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        k (str): Forward bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*R^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_irreversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    k: Annotated[str, "Forward bimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the product of two reactant quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        k (str): Forward bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*R_1*R_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_irreversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    k: Annotated[str, "Forward trimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products, and the change of a product quantity is proportional to the cube of one reactant quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        k (str): Forward trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*R^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_irreversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    k: Annotated[str, "Forward trimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the quantity of one reactant and the square of the quantity of the other reactant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        k (str): Forward trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*R_1^2*R_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_irreversible_reactions_three_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    k: Annotated[str, "Forward trimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products, and the change of a product quantity is proportional to the product of three reactant quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        k (str): Forward trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*R_1*R_2*R_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_irreversible_reactions_single_essential_stimulator(
    a: Annotated[str, "Concentration of activator"],
    k: Annotated[str, "Forward unimolecular rate constant."] = "k",
) -> Expr:
    """Reaction scheme in which the reaction velocity is direct proportional to the activity or concentration of a single molecular species. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the quantity of the stimulator. It is to be used in a reaction modelled using a continuous framework.

    Args:
        a (str): Concentration of activator
        k (str): Forward unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k*A", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("A", locals=LOCALS)] = sympify(a, locals=LOCALS)
    substitutions[sympify("k", locals=LOCALS)] = sympify(k, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_first_order_reverse_reversible_reactions(
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        p (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_second_order_reverse_reversible_reactions_one_product(
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the square of one product quantity. It is to be used in a reaction modelled using a continuous framework.



    Args:
        p (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_second_order_reverse_reversible_reactions_two_products(
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the product of two product quantities. It is to be used in a reaction modelled using a continuous framework.


    Args:
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_third_order_reverse_reversible_reactions_one_product(
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        p (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_third_order_reverse_reversible_reactions_two_products(
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P_1*P_2^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def zeroth_order_forward_third_order_reverse_reversible_reactions_three_products(
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward zeroth order rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities.  The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is constant. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward zeroth order rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_zeroth_order_reverse_reversible_reactions(
    r: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_first_order_reverse_reversible_reactions(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_second_order_reverse_reversible_reactions_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_second_order_reverse_reversible_reactions_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_third_order_reverse_reversible_reactions_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_third_order_reverse_reversible_reactions_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def first_order_forward_third_order_reverse_reversible_reactions_three_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward unimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward unimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_zeroth_order_reverse_reversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_first_order_reverse_reversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_second_order_reverse_reversible_reactions_one_reactant_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_second_order_reverse_reversible_reactions_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_one_reactant_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_one_reactant_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_one_reactant_three_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the square of one reactant quantity. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^2-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_zeroth_order_reverse_reversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_first_order_reverse_reversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_second_order_reverse_reversible_reactions_two_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_second_order_reverse_reversible_reactions_two_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_two_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_two_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def second_order_forward_third_order_reverse_reversible_reactions_two_reactants_three_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward bimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of two reactant quantities. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward bimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_zeroth_order_reverse_reversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_first_order_reverse_reversible_reactions_two_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_two_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_two_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_two_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_two_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_two_reactants_three_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the  quantity of one reactant and the square of quantity of the other reactant. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_zeroth_order_reverse_reversible_reactions_three_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1^2*R_2-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_first_order_reverse_reversible_reactions_three_reactants(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_three_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_three_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_three_reactants_one_product(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_three_reactants_two_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_three_reactants_three_products(
    r_1: Annotated[str, "Concentration of reactant"],
    r_2: Annotated[str, "Concentration of reactant"],
    r_3: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the product of three reactant quantities. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r_1 (str): Concentration of reactant
        r_2 (str): Concentration of reactant
        r_3 (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R_1*R_2*R_3-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R_1", locals=LOCALS)] = sympify(r_1, locals=LOCALS)
    substitutions[sympify("R_2", locals=LOCALS)] = sympify(r_2, locals=LOCALS)
    substitutions[sympify("R_3", locals=LOCALS)] = sympify(r_3, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_zeroth_order_reverse_reversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Forward zeroth order rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is constant. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Forward zeroth order rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_first_order_reverse_reversible_reactions_one_reactant(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse unimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the quantity of one product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse unimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_one_reactant_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the square of one product quantity.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P^2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_second_order_reverse_reversible_reactions_one_reactant_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse bimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the product of two product quantities.  It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse bimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P_1*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_one_reactant_one_product(
    r: Annotated[str, "Concentration of reactant"],
    p: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the cube of one product quantity. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P^3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_one_reactant_two_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the quantity of one product and the square of the quantity of the other product. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P_1^2*P_2", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def third_order_forward_third_order_reverse_reversible_reactions_one_reactant_three_products(
    r: Annotated[str, "Concentration of reactant"],
    p_1: Annotated[str, "Concentration of product"],
    p_2: Annotated[str, "Concentration of product"],
    p_3: Annotated[str, "Concentration of product"],
    k_f: Annotated[str, "Forward trimolecular rate constant."] = "k_f",
    k_r: Annotated[str, "Reverse trimolecular rate constant."] = "k_r",
) -> Expr:
    """Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does include a reverse process that creates the reactants from the products. The rate of the forward process is proportional to the cube of one reactant quantity. The rate of the reverse process is proportional to the product of three product quantities. It is to be used in a reaction modelled using a continuous framework.

    Args:
        r (str): Concentration of reactant
        p_1 (str): Concentration of product
        p_2 (str): Concentration of product
        p_3 (str): Concentration of product
        k_f (str): Forward trimolecular rate constant.
        k_r (str): Reverse trimolecular rate constant.

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("k_f*R^3-k_r*P_1*P_2*P_3", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("R", locals=LOCALS)] = sympify(r, locals=LOCALS)
    substitutions[sympify("P_1", locals=LOCALS)] = sympify(p_1, locals=LOCALS)
    substitutions[sympify("P_2", locals=LOCALS)] = sympify(p_2, locals=LOCALS)
    substitutions[sympify("P_3", locals=LOCALS)] = sympify(p_3, locals=LOCALS)
    substitutions[sympify("k_f", locals=LOCALS)] = sympify(k_f, locals=LOCALS)
    substitutions[sympify("k_r", locals=LOCALS)] = sympify(k_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation
