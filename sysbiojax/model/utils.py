import re

from typing import Dict

from sympy import Derivative, Eq


def odeprint(y, expr):
    """Displays an ODE with a left-hand side derivative"""

    try:
        from IPython.display import display

        display(Eq(Derivative(y, "t"), expr))

    except ImportError:
        print(f"{str(y)}' = {str(expr)}")


def parameter_exists(name: str, parameters: Dict[str, "Parameter"]) -> bool:  # type: ignore
    """Checks whether a parameter is already present in a model"""

    return any(param.name == name for param in parameters.values())


def check_symbol(symbol: str) -> None:
    """Checks whether the given symbol is a valid symbol"""

    ERROR_MESSAGE = f"""Symbol '{symbol}' is not a valid symbol. The following rules apply:
    
    (1) The first character must be a letter
    (2) The remaining characters can be letters and numbers
    (3) The symbol cannot end with an underscore
    (4) The symbol can contain at most one underscore followed by letters and numbers
    
    These are valid symbols:
    
    k1, k_12, k_max, k_max1
    
    """

    if symbol.endswith("_"):
        raise ValueError(ERROR_MESSAGE)

    pattern = r"""
        ^[A-Za-z]    # First character must be a letter
        [A-Za-z0-9]* # Remaining characters can be letters and numbers
        \_?          # Optional underscore
        [A-Za-z0-9]* # Remaining characters can be letters and numbers
    """

    regex = re.compile(pattern, re.X)

    if not bool(regex.match(symbol)):
        raise ValueError(ERROR_MESSAGE)
