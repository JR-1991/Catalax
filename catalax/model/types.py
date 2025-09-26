from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema
from sympy import Expr, sympify

from catalax.model.utils import LOCALS

# Ensure Expressions are denoted as "string" in
# JSON schema upon schema export
AnnotatedExpr = Annotated[
    Expr,
    BeforeValidator(lambda x: sympify(x, locals=LOCALS) if isinstance(x, str) else x),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}),
]
