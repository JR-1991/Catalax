from pydantic import BaseModel, validator
from sympy import Expr, symbols


class Species(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    symbol: Expr

    @validator("symbol", pre=True)
    def convert_string_species_to_sympy(cls, value):
        """Converts given strings of unit definitions into SymPy symbols"""

        if isinstance(value, str):
            value = symbols(value)

        return value
