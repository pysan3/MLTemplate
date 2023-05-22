from __future__ import annotations

from typing import TYPE_CHECKING

from mltemplate.libs.loader.dataset_loader import DatasetTypeBatch

if TYPE_CHECKING:
    from torch import Tensor

    from utils.manager import Manager


class Evaluator:
    def __init__(self, mgr: Manager) -> None:
        self.mgr = mgr

    def evaluate(self, pred: Tensor, gt: DatasetTypeBatch, epoch: int, iteration: int):
        pass


def make_evaluator(mgr: Manager):
    ev = Evaluator(mgr)
    mgr.debug(f"Using evaluator: {ev}")
    return ev
