from numpyro import distributions


def Normal(mu, sigma):
    return (distributions.Normal(mu, sigma), f"N(μ={mu}, σ={sigma})")


def TruncatedNormal(mu, sigma, low=1.0, high=1000.0):
    if low is None and high is None:
        raise ValueError(
            "Both low and high cannot be None. At least one of them must be specified."
        )

    if low:
        assert (
            low > 0
        ), "low must be greater than 0. Otherwise the integration will probably fail"

    return (
        distributions.TruncatedNormal(mu, sigma, low=low, high=high),
        f"N(μ={mu}, σ={sigma}, high={high} low={low})",
    )


def Uniform(low, high):
    return (distributions.Uniform(low, high), f"U(low={low}, high={high})")
