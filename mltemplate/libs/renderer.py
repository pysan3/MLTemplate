from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .loader.dataset_loader import DatasetTypeBatch

if TYPE_CHECKING:
    from torch import Tensor

    from mltemplate.utils.manager import Manager


class Renderer:
    def __init__(self, mgr: Manager, net: torch.nn.Module, device: torch.device):
        self.mgr = mgr
        self.device = device
        self.net = net.to(self.device)

    def render(self, batch: DatasetTypeBatch) -> Tensor:
        b, c, h, w = batch.img_batch.shape
        if self.mgr.veryverbose:
            print(f"Renderer.render: {b=}, {c=}, {h=}, {w=}")
        img = batch.img_batch.to(self.device)
        return self.net(img)


def make_renderer(mgr: Manager, net: torch.nn.Module, device: torch.device):
    return Renderer(mgr, net, device)
