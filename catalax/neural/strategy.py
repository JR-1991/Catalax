from enum import Enum
from typing import Any, List

import optax
import jax.tree_util as jtu
import equinox as eqx

from pydantic import Field, PrivateAttr
from typing import Tuple

from catalax.model.base import CatalaxBase
from catalax.neural.neuralbase import NeuralBase


class Modes(Enum):
    BOTH = "both"
    VECTOR_FIELD = "vector_field"
    MLP = "mlp"


class Step(CatalaxBase):
    steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    length: float = Field(default=1.0, le=1.0, gt=0.0)
    alpha: float = Field(default=0.0, le=1.0, ge=0.0)
    loss: Any = optax.l2_loss
    mode: Modes = Modes.MLP

    def _partition_model(self, model: NeuralBase) -> Tuple[NeuralBase, NeuralBase]:
        """
        Partitions the given uode based on the mode specified.

        Args:
            uode (ClosureODE): The model to partition.
            mode (str): The mode to partition the model with. Can be one of "both", "mlp", or "vector_field".

        Returns:
            Tuple[ClosureODE, ClosureODE]: The partitioned models.
        """

        assert issubclass(
            type(model), NeuralBase
        ), "Model must be a subclass of NeuralBase."

        if isinstance(self.mode, Modes):
            mode = self.mode.value
        else:
            mode = self.mode

        filter_spec = jtu.tree_map(lambda _: False, model)
        mlp_filter = jtu.tree_map(lambda _: True, model.func)
        vfield_filter = jtu.tree_map(lambda _: True, model.vector_field)  # type: ignore

        if mode == Modes.BOTH.value:
            filter_spec = jtu.tree_map(lambda _: True, model)

        elif mode == Modes.MLP.value:
            assert hasattr(
                model, "func"
            ), "Mode is set to MLP, but the model does not have an MLP."

            filter_spec = eqx.tree_at(
                lambda tree: tree.func,
                filter_spec,
                replace=mlp_filter,
            )

        elif mode == Modes.VECTOR_FIELD.value:
            assert hasattr(
                model, "vector_field"
            ), "Mode is set to vector field, but the model does not have a vector field."

            filter_spec = eqx.tree_at(
                lambda tree: (tree.vector_field, tree.parameters),
                filter_spec,
                replace=(vfield_filter, True),
            )

        else:
            raise ValueError(
                f"Mode {self.mode} is not supported. Please choose one of the following\n {list(Modes.__members__.values())}"
            )

        return eqx.partition(model, filter_spec)


class Strategy(CatalaxBase):
    steps: List[Step] = []
    _step: int = PrivateAttr(0)

    def add_step(
        self,
        lr: float,
        steps: int,
        batch_size: int,
        length: float = 1.0,
        alpha: float = 0.0,
        loss: Any = optax.l2_loss,
        train_weights: Modes = Modes.BOTH,
    ):
        self.steps.append(
            Step(
                lr=lr,
                steps=steps,
                batch_size=batch_size,
                length=length,
                alpha=alpha,
                loss=loss,
                train_weights=train_weights,
            )
        )

    def __iter__(self):
        self._step = 0
        return self

    def __next__(self):
        if self._step >= len(self.steps):
            raise StopIteration
        else:
            step = self.steps[self._step]
            self._step += 1
            self._print_step(self._step, step)

            return step

    @staticmethod
    def _print_step(index: int, step: Step):
        print(
            f"<< Strategy #{index+1}: Learning rate: {step.lr} | Steps: {step.steps} Length: {step.length*100}% >>\n"
        )
