import pandas as pd

from typing import Any, List, Optional, Tuple, Union
from numpyro.distributions import Distribution

from pydantic import PrivateAttr, root_validator, validator
from sympy import Expr

from catalax.model.base import CatalaxBase


class Parameter(CatalaxBase):
    name: str
    symbol: Expr
    value: Optional[float] = None
    constant: bool = False
    initial_value: Optional[float] = None
    equation: Union[str, Expr, None] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    prior: Any = None  # TODO: Fix this typing
    _prior_str_: Optional[str] = None

    @root_validator()
    def _assign_prior_string(cls, values):
        if isinstance(values["prior"], tuple):
            prior, prior_str = values["prior"]
            values["prior"] = prior
            values["_prior_str_"] = prior_str

        return values

    __repr_fields__: List[str] = PrivateAttr(
        default={
            "name": "name",
            "symbol": "symbol",
            "value": "value",
            "_prior_str_": "prior",
            "initial_value": "initial_value",
            "equation": "equation",
            "lower_bound": "lower_bound",
            "upper_bound": "upper_bound",
            "constant": "constant",
        }
    )
