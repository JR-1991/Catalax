from typing import Any, List, Optional, Union

from pydantic import PrivateAttr, model_validator
from sympy import Expr

from catalax.model.base import CatalaxBase


class Identifiability(CatalaxBase):
    result: str
    method: str
    package: str


class HDI(CatalaxBase):
    lower: float
    upper: float
    lower_50: float
    upper_50: float
    q: float


class Parameter(CatalaxBase):
    name: str
    symbol: Expr
    value: Optional[float] = None
    constant: bool = False
    identifiability: Optional[Identifiability] = None
    initial_value: Optional[float] = None
    equation: Union[str, Expr, None] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    hdi: Optional[HDI] = None
    prior: Any = None  # TODO: Fix this typing
    _prior_str_: Optional[str] = None

    @model_validator(mode="after")
    def _assign_prior_string(self):
        if isinstance(self.prior, tuple):
            prior, prior_str = self.prior
            prior, prior_str = self.prior
            self.prior = prior
            self._prior_str_ = prior_str

        return self

    _repr_fields: List[str] = PrivateAttr(
        default=[
            "name",
            "symbol",
            "value",
            "_prior_str_",
            "initial_value",
            "equation",
            "lower_bound",
            "upper_bound",
            "constant",
            "hdi",
        ]
    )
