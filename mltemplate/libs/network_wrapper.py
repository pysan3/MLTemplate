from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from mltemplate.libs.loader.dataset_loader import DatasetTypeBatch
from mltemplate.libs.renderer import make_renderer
from utils.manager import Manager

if TYPE_CHECKING:
    from torch import Tensor

CRT_T = Callable[[Tensor, Tensor], Tensor]


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class ScalarStats:
    loss: Tensor
    loss_diff: Tensor

    is_mean: bool = False

    def get_means(self):
        return ScalarStats(**{k: torch.mean(v) for k, v in self.to_dict().items()}, is_mean=True)

    def to_dict(self):
        result = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if not isinstance(v, Tensor):
                continue
            result[field.name] = v.detach().cpu()
        return result


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class RenderStats:
    pass


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class NetworkWrapperResult:
    result: Tensor
    loss: Tensor
    scalar: ScalarStats
    render: RenderStats


class NetworkWrapper(nn.Module):
    def __init__(self, mgr: Manager, net: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.mgr = mgr
        self.device = device
        self.renderer = make_renderer(mgr, net, self.device)
        self.criteria: CRT_T = nn.MSELoss()

    def forward(self, batch: DatasetTypeBatch):
        result = self.renderer.render(batch)

        loss_diff = self.criteria(batch.img_batch, result)
        loss = loss_diff

        return NetworkWrapperResult(
            result.cpu(),
            loss.cpu(),
            ScalarStats(
                loss.detach().cpu(),
                loss_diff.detach().cpu(),
            ),
            RenderStats(),
        )
