from typing import List
from pydantic import BaseModel, PrivateAttr


class CatalaxBase(BaseModel):
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        allow_mutation = True
        validate_assignment = True

    __repr_fields__: List[str] = PrivateAttr(default=["__all__"])
