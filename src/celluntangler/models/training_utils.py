from dataclasses import dataclass
from typing import Optional, Callable
import torch
import os


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"
    best_value: float = float("-inf")
    best_epoch: Optional[int] = None
    epochs_no_improve: int = 0
    save_fn: Optional[Callable] = None

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        else:
            return value > self.best_value + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_no_improve = 0

            if self.save_fn is not None:
                self.save_fn(epoch, value)
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve >= self.patience
