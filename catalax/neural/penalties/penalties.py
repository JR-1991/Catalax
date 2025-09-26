from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field, field_validator

import jax
import jax.numpy as jnp

from catalax.neural.neuralbase import NeuralBase
from catalax.neural.penalties.uode import (
    l1_reg_alpha,
    l1_reg_gate,
    l2_reg_alpha,
    l2_reg_gate,
)
from catalax.neural.penalties.weight import l1_regularisation, l2_regularisation

from .stoich_mat import (
    l1_stoich_penalty,
    penalize_density,
    penalize_non_bipolar,
    penalize_duplicate_reactions,
    penalize_non_conservative,
    penalize_non_integer,
    penalize_null_space,
)


class Penalties:
    """A collection of penalty functions that can be applied to neural models during training.

    This class manages multiple penalty functions and applies them collectively to a model.
    Penalties are used to enforce constraints or encourage certain behaviors in the model
    parameters during optimization.

    Attributes:
        penalties: List of Penalty objects to apply to the model
    """

    def __init__(
        self,
        penalties: Optional[List[Penalty]] = None,
    ):
        """Initialize the Penalties collection.

        Args:
            penalties: Optional list of Penalty objects. If None, initializes empty list.
        """
        if penalties is None:
            self.penalties = []
        else:
            self.penalties = penalties

    def add_penalty(
        self,
        name: str,
        fun: Callable,
        alpha: float,
        **kwargs,
    ) -> Penalties:
        """Add a penalty function to the collection.

        Args:
            name: Name of the penalty
            fun: The penalty function to apply
            alpha: Strength coefficient for the penalty
            **kwargs: Additional keyword arguments for the penalty function

        Returns:
            Self for method chaining
        """
        self.penalties.append(
            Penalty(
                name=name,
                fun=fun,
                alpha=alpha,
                **kwargs,
            )
        )
        return self

    def __call__(self, model: Type[NeuralBase]) -> jax.Array:
        """Apply all penalties to the model and return the total penalty value.

        Args:
            model: Neural model to apply penalties to

        Returns:
            Sum of all penalty values as a JAX array
        """
        return jnp.sum(jnp.array([penalty(model) for penalty in self.penalties]))

    def update_alpha(self, alpha: Optional[float], **kwargs) -> Penalties:
        """Update the alpha parameter for all penalties.

        Args:
            alpha: New alpha value to set for all penalties. If None, only penalties
                specified in kwargs will be updated.
            **kwargs: Specific alpha values for individual penalties by name

        Returns:
            New Penalties instance with updated alpha values
        """
        new_penalties = deepcopy(self)
        for penalty in new_penalties.penalties:
            if penalty.name in kwargs:
                assert isinstance(
                    kwargs[penalty.name], float
                ), f"Alpha for {penalty.name} must be a float"
                penalty.alpha = kwargs[penalty.name]
            elif alpha is not None:
                penalty.alpha = alpha
            else:
                continue

        return Penalties(penalties=new_penalties.penalties)

    @classmethod
    def for_neural_ode(
        cls,
        l2_alpha: Optional[float] = 1e-3,
        l1_alpha: Optional[float] = None,
    ) -> Penalties:
        """Create a collection of penalties specifically designed for NeuralODE models."""
        penalties = cls()

        if l2_alpha is not None:
            penalties.add_penalty(
                name="l2",
                fun=l2_regularisation,
                alpha=l2_alpha,
            )

        if l1_alpha is not None:
            penalties.add_penalty(
                name="l1",
                fun=l1_regularisation,
                alpha=l1_alpha,
            )
        return penalties

    @classmethod
    def for_universal_ode(
        cls,
        l2_gate_alpha: Optional[float] = 1e-3,
        l1_gate_alpha: Optional[float] = None,
        l2_residual_alpha: Optional[float] = 1e-3,
        l1_residual_alpha: Optional[float] = None,
        l2_mlp_alpha: Optional[float] = 1e-3,
        l1_mlp_alpha: Optional[float] = None,
    ) -> Penalties:
        """Create a collection of penalties specifically designed for UniversalODE models."""
        penalties = cls()
        if l2_gate_alpha is not None:
            penalties.add_penalty(
                name="l2_gate",
                fun=l2_reg_gate,
                alpha=l2_gate_alpha,
            )

        if l1_gate_alpha is not None:
            penalties.add_penalty(
                name="l1_gate",
                fun=l1_reg_gate,
                alpha=l1_gate_alpha,
            )

        if l2_residual_alpha is not None:
            penalties.add_penalty(
                name="l2_residual",
                fun=l2_reg_alpha,
                alpha=l2_residual_alpha,
            )

        if l1_residual_alpha is not None:
            penalties.add_penalty(
                name="l1_residual",
                fun=l1_reg_alpha,
                alpha=l1_residual_alpha,
            )

        if l2_mlp_alpha is not None:
            penalties.add_penalty(
                name="l2_mlp",
                fun=l2_regularisation,
                alpha=l2_mlp_alpha,
            )

        if l1_mlp_alpha is not None:
            penalties.add_penalty(
                name="l1_mlp",
                fun=l1_regularisation,
                alpha=l1_mlp_alpha,
            )

        return penalties

    @classmethod
    def for_rateflow(
        cls,
        alpha: float = 0.1,
        density_alpha: Optional[float] = None,
        bipolar_alpha: Optional[float] = None,
        conservation_alpha: Optional[float] = None,
        duplicate_reactions_alpha: Optional[float] = None,
        integer_alpha: Optional[float] = None,
        sparsity_alpha: Optional[float] = None,
        l2_alpha: Optional[float] = None,
        null_space_alpha: Optional[float] = None,
    ) -> Penalties:
        """Create a collection of penalties specifically designed for NeuralRDE models.

        This factory method creates a comprehensive set of penalties that are commonly
        used with NeuralRDE models, including stoichiometric matrix penalties and
        weight regularization penalties.

        Args:
            alpha: Default penalty strength coefficient for all penalties
            density_alpha: Specific alpha for density penalty (sparsity of stoich matrix)
            bipolar_alpha: Specific alpha for bipolar penalty (mass balance)
            conservation_alpha: Specific alpha for conservation penalty
            duplicate_reactions_alpha: Specific alpha for duplicate reactions penalty
            integer_alpha: Specific alpha for integer penalty (encouraging integer coefficients)
            sparsity_alpha: Specific alpha for L1 sparsity penalty on stoich matrix
            l2_alpha: Specific alpha for L2 weight regularization

        Returns:
            Penalties instance configured for NeuralRDE models
        """
        penalties = cls()

        # Stoichiometric matrix penalties
        penalties.add_penalty(
            name="density",
            fun=penalize_density,
            alpha=density_alpha or alpha,
        )

        if conservation_alpha is not None:
            # This is a very strong penalty, so we only add it if it is explicitly set
            penalties.add_penalty(
                name="conservation",
                fun=penalize_non_conservative,
                alpha=conservation_alpha,
            )

        penalties.add_penalty(
            name="bipolar",
            fun=penalize_non_bipolar,
            alpha=bipolar_alpha or alpha,
        )

        penalties.add_penalty(
            name="duplicate_reactions",
            fun=penalize_duplicate_reactions,
            alpha=duplicate_reactions_alpha or alpha,
        )

        penalties.add_penalty(
            name="integer",
            fun=penalize_non_integer,
            alpha=integer_alpha or alpha,
        )

        penalties.add_penalty(
            name="sparsity",
            fun=l1_stoich_penalty,
            alpha=sparsity_alpha or alpha,
        )

        # Weight penalties
        penalties.add_penalty(
            name="l2",
            fun=l2_regularisation,
            alpha=l2_alpha or alpha,
        )

        if null_space_alpha is not None:
            penalties.add_penalty(
                name="null_space",
                fun=penalize_null_space,
                alpha=null_space_alpha,
            )
        return penalties


class Penalty(BaseModel):
    """A single penalty function with associated parameters.

    This class wraps a penalty function along with its strength parameter (alpha)
    and any additional keyword arguments needed for the function. It provides
    validation to ensure the penalty function has the correct signature.

    Attributes:
        name: The name of the penalty for identification
        fun: The penalty function to apply (must accept 'model' and 'alpha' parameters)
        alpha: Strength coefficient for the penalty
        kwargs: Additional keyword arguments for the penalty function
    """

    name: str
    fun: Callable
    alpha: float
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __call__(self, model: Type[NeuralBase], **kwargs) -> jax.Array:
        """Apply the penalty function to the model.

        Args:
            model: Neural model to apply penalty to
            **kwargs: Additional keyword arguments that override instance kwargs

        Returns:
            Penalty value as a JAX array
        """
        return self.fun(
            model=model,
            alpha=self.alpha,
            **self.kwargs,
            **kwargs,
        )

    @field_validator("fun")
    @classmethod
    def validate_function_signature(cls, v: Callable) -> Callable:
        """Validate that the penalty function has the required signature.

        Ensures the penalty function can accept the required 'model' and 'alpha'
        parameters that are needed for proper penalty computation.

        Args:
            v: The penalty function to validate

        Returns:
            The validated function

        Raises:
            ValueError: If the function doesn't have required 'model' and 'alpha' parameters
        """
        if not cls.validate_penalty_signature(v):
            raise ValueError(
                f"Penalty function {v.__name__} must accept 'model' and 'alpha' parameters"
            )
        return v

    @staticmethod
    def has_model_alpha_signature(func: Callable) -> bool:
        """Check if a function has 'model' and 'alpha' parameters.

        This is a simple check that looks for the presence of both required
        parameter names in the function signature.

        Args:
            func: Function to check

        Returns:
            True if function has both 'model' and 'alpha' parameters
        """
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        return "model" in param_names and "alpha" in param_names

    @staticmethod
    def validate_penalty_signature(func: Callable) -> bool:
        """Validate that a function has the required signature for penalties.

        This method performs a more robust validation by attempting to bind
        the required parameters to the function signature, which handles
        cases with *args, **kwargs, and other parameter variations.

        Args:
            func: Function to validate

        Returns:
            True if function can accept 'model' and 'alpha' parameters
        """
        try:
            # Test if we can bind the required parameters
            inspect.signature(func).bind_partial(model=None, alpha=0.0)
            return True
        except TypeError:
            return False
