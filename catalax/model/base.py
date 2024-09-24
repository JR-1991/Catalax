from typing import List
from pydantic import BaseModel, PrivateAttr


class CatalaxBase(BaseModel):
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        validate_assignment = True

    _repr_fields: List[str] = PrivateAttr(default=["__all__"])
