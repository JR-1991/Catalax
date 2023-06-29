from numpyro import distributions
from typing import Optional
from pydantic import BaseModel, PrivateAttr


class Prior(BaseModel):
    type: str
    _print_str: str = PrivateAttr(default="")
    _distribution_fun: Optional[distributions.Distribution] = PrivateAttr(default=None)

    def __init__(self, **kwargs) -> None:
        super().__init__(type=self.__class__.__name__, **kwargs)

    def __str__(self) -> str:
        return self._print_str


class Normal(Prior):
    mu: float
    sigma: float

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._distribution_fun = distributions.Normal(self.mu, self.sigma)
        self._print_str = f"N(μ={self.mu}, σ={self.sigma})"


class TruncatedNormal(Prior):
    mu: float
    sigma: float
    low: float = 1e-6
    high: float = 1e6

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.low is None and self.high is None:
            raise ValueError(
                "Both low and high cannot be None. At least one of them must be specified."
            )

        if self.low:
            assert (
                self.low > 0
            ), "low must be greater than 0. Otherwise the integration will probably fail"

        self._distribution_fun = distributions.TruncatedNormal(
            loc=self.mu, scale=self.sigma, low=self.low, high=self.high
        )
        self._print_str = (
            f"N(μ={self.mu}, σ={self.sigma}, high={self.high} low={self.low})"
        )


class Uniform(Prior):
    low: float
    high: float

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._distribution_fun = distributions.Uniform(self.low, self.high)
        self._print_str = f"U(low={self.low}, high={self.high})"
