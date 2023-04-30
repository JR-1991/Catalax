from typing import Optional, Union

from pydantic import BaseModel
from sympy import Expr


class Parameter(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    value: Optional[float] = None
    initial_value: Optional[float] = None
    equation: Union[str, Expr, None] = None
