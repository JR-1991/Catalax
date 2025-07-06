from typing import List

from pydantic import PrivateAttr
from sympy import Expr

from catalax.model.base import CatalaxBase


class Constant(CatalaxBase):
    name: str
    symbol: Expr

    _repr_fields: List[str] = PrivateAttr(
        default=[
            "name",
            "symbol",
        ]
    )
