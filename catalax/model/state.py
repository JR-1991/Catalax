from typing import Literal, Optional

from pydantic import ConfigDict, field_validator
from sympy import Expr, symbols

from catalax.model.base import CatalaxBase

STATE_TYPE_CHOICES = Literal[
    "small_molecule", "protein", "complex", "process_variable", "other"
]


class State(CatalaxBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    name: str
    symbol: Expr
    type: Optional[STATE_TYPE_CHOICES] = None

    @field_validator("symbol", mode="before")
    def convert_string_state_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        if isinstance(value, str):
            value = symbols(value)

        return value
