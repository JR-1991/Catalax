from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import jax.tree_util as jtu
import optax
from pydantic import Field, PrivateAttr

from catalax.model.base import CatalaxBase
from catalax.neural.neuralbase import NeuralBase
from catalax.neural.penalties.penalties import Penalties
from catalax.neural.rateflow import RateFlowODE
from catalax.neural.universalode import UniversalODE


class Modes(Enum):
    BOTH = "both"
    VECTOR_FIELD = "vector_field"
    MLP = "mlp"


class Step(CatalaxBase):
    steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    length: float = Field(default=1.0, le=1.0, gt=0.0)
    penalties: Penalties = Field(default=Penalties())
    loss: Any = optax.l2_loss
    train: Modes = Modes.MLP

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

        if isinstance(self.train, Modes):
            train = self.train.value
        else:
            train = self.train

        filter_spec = jtu.tree_map(lambda _: False, model)
        mlp_filter = jtu.tree_map(lambda _: True, model.func)
        vfield_filter = jtu.tree_map(lambda _: True, model.vector_field)

        if isinstance(model, RateFlowODE):
            stoich_filter = jtu.tree_map(
                lambda _: model.learn_stoich,
                model.stoich_matrix,
            )

            filter_spec = eqx.tree_at(
                lambda tree: tree.stoich_matrix,
                filter_spec,
                replace=stoich_filter,
            )

        elif isinstance(model, UniversalODE):
            learn_gate = train == Modes.BOTH.value or train == Modes.MLP.value
            gate_filter = jtu.tree_map(
                lambda _: (
                    learn_gate and model.use_gate,
                    learn_gate and model.use_gate,
                ),
                model.gate_matrix,
            )

            filter_spec = eqx.tree_at(
                lambda tree: (tree.gate_matrix, tree.alpha_residual),
                filter_spec,
                replace=gate_filter,
            )

        if train == Modes.BOTH.value:
            filter_spec = jtu.tree_map(lambda _: True, model)

        elif train == Modes.MLP.value:
            assert hasattr(
                model, "func"
            ), "Mode is set to MLP, but the model does not have an MLP."

            filter_spec = eqx.tree_at(
                lambda tree: tree.func,
                filter_spec,
                replace=mlp_filter,
            )

        elif train == Modes.VECTOR_FIELD.value:
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
                f"Mode {self.train} is not supported. Please choose one of the following\n {list(Modes.__members__.values())}"
            )

        return eqx.partition(model, filter_spec)


class Strategy(CatalaxBase):
    steps: List[Step] = []
    penalties: Penalties = Penalties()
    batch_size: int = 5
    loss: Optional[Callable] = optax.l2_loss
    _step: int = PrivateAttr(0)

    def add_step(
        self,
        lr: float,
        steps: int,
        batch_size: Optional[int] = None,
        length: float = 1.0,
        penalties: Optional[Penalties] = None,
        loss: Optional[Callable] = None,
        train: Modes = Modes.MLP,
    ):
        penalties = penalties or self.penalties or Penalties()
        batch_size = batch_size or self.batch_size
        loss = loss or self.loss

        if batch_size is None:
            raise ValueError(
                "Batch size must be provided. Please provide a batch size or set the batch size in the strategy initialization."
            )

        if loss is None:
            raise ValueError(
                "No loss function provided. Please provide a loss function or set the loss function in the strategy initialization."
            )

        assert (
            penalties is not None
        ), "Penalties must be provided. Please provide penalties or set the penalties in the strategy initialization."

        self.steps.append(
            Step(
                lr=lr,
                steps=steps,
                batch_size=batch_size,
                length=length,
                penalties=penalties,
                loss=loss,
                train=train,
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
        statements = [
            f"🔸 Step #{index}",
        ]
        fun = lambda name, value: f"├── \033[1m{name}\033[0m: {value}"  # noqa: E731

        statements += [
            fun("lr", step.lr),
            fun("batch size", step.batch_size),
            fun("length", f"{step.length * 100}%"),
            fun("train", step.train),
            "│",
        ]

        print("\n".join(statements))
