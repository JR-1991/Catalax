from pydantic import field_validator, ConfigDict
from sympy import Expr, symbols

from catalax.model.base import CatalaxBase


class Species(CatalaxBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    name: str
    symbol: Expr

    @field_validator("symbol", mode="before")
    def convert_string_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        if isinstance(value, str):
            value = symbols(value)

        return value
