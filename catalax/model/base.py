from typing import List
from pydantic import BaseModel, PrivateAttr, ConfigDict


class CatalaxBase(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    _repr_fields: List[str] = PrivateAttr(default=["__all__"])
