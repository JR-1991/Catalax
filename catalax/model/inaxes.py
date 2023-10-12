from enum import Enum


class InAxes(Enum):
    PARAMETERS = (None, 0, None)
    TIME = (None, None, 0)
    Y0 = (0, None, None)

    # Override add function to combine axes
    def __add__(self, other):
        assert isinstance(other, InAxes), "Other spec must be an instance of InAxes"

        merge = (
            lambda i: 0
            if any(spec.value[i] is not None for spec in [self, other])
            else None
        )
        return (
            merge(0),
            merge(1),
            merge(2),
        )
