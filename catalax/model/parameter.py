import pandas as pd

from typing import Optional, Union

from dotted_dict import DottedDict
from pydantic import BaseModel
from sympy import Expr


class Parameter(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    symbol: Expr
    value: Optional[float] = None
    constant: bool = False
    initial_value: Optional[float] = None
    equation: Union[str, Expr, None] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    stdev: Optional[float] = None
