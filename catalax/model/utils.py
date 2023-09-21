import re
from typing import Dict, List

import pandas as pd
from sympy import Derivative, Eq
from dotted_dict import DottedDict

from catalax.model.base import CatalaxBase


class PrettyDict(DottedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
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
        if len(cls_.__repr_fields__) == 1 and cls_.__repr_fields__[0] == "__all__":
            return {key: key for key in cls_.__dict__}

        return cls_.__repr_fields__


def odeprint(y, expr):
    """Displays an ODE with a left-hand side derivative"""

    try:
        from IPython.display import display

        display(Eq(Derivative(y, "t"), expr, evaluate=False))

    except ImportError:
        print(f"d{str(y)}_dt' = {str(expr)}")


def eqprint(y, expr):
    """Displays an ODE with a left-hand side derivative"""

    try:
        from IPython.display import display

        display(Eq(y, expr, evaluate=False))

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
