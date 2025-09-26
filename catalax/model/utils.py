"""Utility functions and classes for the catalax model module."""

import re
from typing import Dict

import pandas as pd
from dotted_dict import DottedDict
from sympy import Derivative, Eq, Symbol

from catalax.model.base import CatalaxBase
from catalax.model.parameter import Parameter

LOCALS = {
    "I": Symbol("I"),
    "E": Symbol("E"),
    "oo": Symbol("oo"),
    "OO": Symbol("OO"),
    "O": Symbol("O"),
    "S": Symbol("S"),
    "N": Symbol("N"),
}


class PrettyDict(DottedDict):
    """A dictionary that displays its contents as a formatted pandas DataFrame.

    This class extends DottedDict to provide a more readable representation
    of model components when displayed in Jupyter notebooks or printed to console.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the PrettyDict with the same arguments as DottedDict."""
        super().__init__(*args, **kwargs)

    def __repr__(self):
        """Return a formatted representation of the dictionary contents.

        Returns:
            str: Empty string if in IPython (displays DataFrame), otherwise string representation of DataFrame.
        """
        sorted_items = sorted(self.items(), key=lambda x: x[0])
        df = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in cls.__dict__.items()
                    if key in self._fields_to_print(cls)
                }
                for _, cls in sorted_items
            ]
        )

        try:
            from IPython.display import display

            display(df)

            return ""

        except ImportError:
            return df

    @staticmethod
    def _fields_to_print(cls_: CatalaxBase) -> Dict[str, str]:
        """Determine which fields should be displayed for a given class.

        Args:
            cls_: The CatalaxBase instance to extract fields from.

        Returns:
            Dict[str, str]: Dictionary mapping field names to themselves for display.
        """
        if len(cls_._repr_fields) == 1 and cls_._repr_fields[0] == "__all__":
            return {key: key for key in cls_.__dict__}

        return {key: key for key in cls_._repr_fields}


def odeprint(y, expr):
    """Display an ODE with a left-hand side derivative.

    Args:
        y: The dependent variable (state).
        expr: The right-hand side expression of the ODE.
    """

    try:
        from IPython.display import display

        display(Eq(Derivative(y, "t"), expr, evaluate=False))

    except ImportError:
        print(f"d{str(y)}_dt' = {str(expr)}")


def eqprint(y, expr):
    """Display an equation in a formatted way.

    Args:
        y: The left-hand side variable.
        expr: The right-hand side expression.
    """

    try:
        from IPython.display import display

        display(Eq(y, expr, evaluate=False))

    except ImportError:
        print(f"{str(y)}' = {str(expr)}")


def parameter_exists(name: str, parameters: Dict[str, Parameter]) -> bool:  # type: ignore
    """Check whether a parameter is already present in a model.

    Args:
        name: The name of the parameter to check for.
        parameters: Dictionary of existing parameters.

    Returns:
        bool: True if parameter exists, False otherwise.
    """

    return any(param.name == name for param in parameters.values())


def check_symbol(symbol: str) -> None:
    """Check whether the given symbol is a valid symbol according to catalax naming rules.

    Valid symbols must follow these rules:
    1. The first character must be a letter
    2. The remaining characters can be letters and numbers
    3. The symbol cannot end with an underscore
    4. The symbol can contain at most one underscore followed by letters and numbers

    Valid examples: k1, k_12, k_max, k_max1

    Args:
        symbol: The symbol string to validate.

    Raises:
        ValueError: If the symbol does not meet the validation criteria.
    """

    ERROR_MESSAGE = f"""Symbol '{symbol}' is not a valid symbol. The following rules apply:

    (1) The first character must be a letter
    (2) The remaining characters can be letters and numbers
    (3) The symbol cannot end with an underscore
    (4) The symbol can contain at most one underscore followed by letters and numbers

    These are valid symbols:

    k1, k_12, k_max, k_max1

    """

    # Convert to string to use string methods
    symbol = str(symbol)

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
