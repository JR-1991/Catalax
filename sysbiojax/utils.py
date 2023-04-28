import pickle
import re
import os

from IPython.display import display
from copy import deepcopy
from typing import Dict
from sympy import solve, Derivative, Eq, Symbol, sympify


def save_model(model: Dict, name: str, dir: str) -> None:
    """Saves a model to a pickle file for re-use"""

    path = os.path.join(dir, f"{name}.pkl")

    with open(path, "wb") as file:
        pickle.dump(model, file)


def odeprint(y, expr):
    """Displays an ODE with a left-hand side derivative"""

    display(Eq(Derivative(y, "t"), expr))


def derive(expr, steps, ode=False):
    """Chain applies methods to an expression and prints every intermediate step

    Example:

        steps = [
            ("Solve for c1", "c1", lambda x: solve(x, "c1"))
        ]

        derive(x, steps)

        Out:

            1) Solve for c1

            --> the solution
    """

    expr = deepcopy(expr)

    for index, step in enumerate(steps):
        message, left, fun = step
        expr = fun(expr)

        print(f"{index + 1}) {message}")

        if ode:
            odeprint(left, expr)
        else:
            eqprint(left, expr)

    return expr


def eqprint(y, expr):
    """Displays an equation with a left-hand side expression"""

    display(Eq(Symbol(y), expr))

    return expr


def equation(eq):
    """Creates an equilibrium constant equation"""

    assert "=" in eq, "No equal sign within the equation, yet it is required."

    y, x = eq.split("=")

    return Eq(sympify(y.strip()), sympify(x.strip()))


def substitute(equation, substitutes, eq_consts, params, ignore=[]):
    """Algebraically solves for a symbol and substitutes it into an equation"""

    if not isinstance(substitutes, list):
        substitutes = [substitutes]
    if not isinstance(ignore, list):
        ignore = [ignore]

    nu_equation = deepcopy(equation)

    for symbol in equation.free_symbols:
        if symbol in params or not _to_substitute(symbol, substitutes):
            continue

        for eq_const in eq_consts:
            solved = solve(eq_const, symbol)

            if len(solved) == 0:
                continue
            elif _has_ignored(solved[0], ignore):
                continue

            nu_equation = nu_equation.subs(symbol, solved[0])

    return nu_equation


def _to_substitute(symbol, substitutes):
    """Checks whether the given symbol is one to substitute"""

    return any(re.match(pattern, str(symbol)) for pattern in substitutes)


def _has_ignored(expr, ignore):
    """Checks whether the solution contains symbols or patterns not wished to include"""

    return any(
        bool(re.match(pattern, str(var)))
        for var in expr.free_symbols
        for pattern in ignore
    )
