from enum import Enum
from typing import Any, List

import optax
from pydantic import Field, PrivateAttr

from catalax.model.base import CatalaxBase


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
    train_weights: Modes = Modes.BOTH


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
        train_weights: str = "both",
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
            print(
                f"<< Strategy #{self._step}: Learning rate: {step.lr} | Steps: {step.steps} Length: {step.length*100}% >>\n"
            )
            return step

    @staticmethod
    def _print_step(index: int, step: Step):
        print(
            f"<< Strategy #{index+1}: Learning rate: {step.lr} | Steps: {step.steps} Length: {step.length*100}% >>\n"
        )
