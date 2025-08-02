from numpyro import distributions
from typing import Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field


class Prior(BaseModel):
    """Base class for all prior distributions used in Bayesian inference.

    This class defines the interface for all prior distributions and provides
    common functionality for serialization and representation.
    """

    _print_str: str = PrivateAttr(default="")
    _distribution_fun: Optional[distributions.Distribution] = PrivateAttr(default=None)

    @computed_field()
    def type(self) -> str:
        """Return the name of the prior distribution class."""
        return self.__class__.__name__


class Normal(Prior):
    """Normal (Gaussian) prior distribution.

    Represents a normal distribution with mean μ and standard deviation σ.

    Attributes:
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mu: float
    sigma: float

    def _distribution_fun(self) -> distributions.Distribution:
        """Create the NumPyro distribution object."""
        return distributions.Normal(self.mu, self.sigma)

    def _print_str(self) -> str:
        """String representation of the distribution."""
        return f"N(μ={self.mu}, σ={self.sigma})"


class TruncatedNormal(Prior):
    """Truncated Normal prior distribution.

    Represents a normal distribution truncated to a specific range.

    Attributes:
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution
        low: Lower bound of the distribution (default: 1e-6)
        high: Upper bound of the distribution (default: 1e6)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mu: float
    sigma: float
    low: float = 1e-6
    high: float = 1e6

    def _distribution_fun(self) -> distributions.Distribution:
        """Create the NumPyro distribution object."""
        return distributions.TruncatedNormal(
            loc=self.mu, scale=self.sigma, low=self.low, high=self.high
        )

    def _print_str(self) -> str:
        """String representation of the distribution."""
        return f"N(μ={self.mu}, σ={self.sigma}, low={self.low}, high={self.high})"


class Uniform(Prior):
    """Uniform prior distribution.

    Represents a uniform distribution between low and high values.

    Attributes:
        low: Lower bound of the distribution
        high: Upper bound of the distribution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    low: float
    high: float

    def _distribution_fun(self) -> distributions.Distribution:
        """Create the NumPyro distribution object."""
        return distributions.Uniform(self.low, self.high)

    def _print_str(self) -> str:
        """String representation of the distribution."""
        return f"U(low={self.low}, high={self.high})"


class LogUniform(Prior):
    """Log-Uniform prior distribution.

    Represents a distribution that is uniform in log space.

    Attributes:
        low: Lower bound of the distribution
        high: Upper bound of the distribution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    low: float
    high: float

    def _distribution_fun(self) -> distributions.Distribution:
        """Create the NumPyro distribution object."""
        return distributions.LogUniform(self.low, self.high)

    def _print_str(self) -> str:
        """String representation of the distribution."""
        return f"LogU(low={self.low}, high={self.high})"
