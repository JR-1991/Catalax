from enum import Enum
from typing import Self


class InAxes(Enum):
    Y0 = (0, None, 0, None)  # Index 0
    PARAMETERS = (None, 0, None, None)  # Index 1
    CONSTANTS = (None, None, 0, None)  # Index 2
    TIME = (None, None, None, 0)  # Index 3

    # Override add function to combine axes
    def __add__(
        self, other: Self
    ) -> tuple[int | None, int | None, int | None, int | None]:
        assert isinstance(other, InAxes), "Other spec must be an instance of InAxes"

        def merge(i: int) -> int | None:
            return (
                0 if any(spec.value[i] is not None for spec in [self, other]) else None
            )

        return (
            merge(0),
            merge(1),
            merge(2),
            merge(3),
        )
