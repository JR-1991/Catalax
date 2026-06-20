"""Uncertainty-quantification methods for catalax predictors.

Exposes a common base (:class:`UncertaintyPredictor` + :class:`PredictiveDistribution`)
that formalises the outputs an uncertainty method provides on top of a point
prediction, and concrete methods built on it -- currently GAPA (Gaussian Process
Activations). The :class:`~catalax.neural.NeuralODEEnsemble` and
:class:`~catalax.mcmc.HMCResults` predictors also conform to this base.

``GAPA`` is imported lazily so that the lightweight base (used by the ensemble
and HMC predictors) does not pull in the optional ``gpjax`` dependency.
"""

from .base import HDILevel, PredictiveDistribution, UncertaintyPredictor

__all__ = [
    "UncertaintyPredictor",
    "PredictiveDistribution",
    "HDILevel",
    "GAPA",
]


def __getattr__(name: str):
    if name == "GAPA":
        from .gapa import GAPA

        return GAPA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
