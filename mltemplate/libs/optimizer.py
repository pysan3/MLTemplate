from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import Module

if TYPE_CHECKING:
    from utils.manager import Manager


def make_optimizer(mgr: Manager, net: Module, lr=None, weight_decay=None):
    params = []
    if lr is None:
        lr = mgr.TRAIN.LR
    if weight_decay is None:
        weight_decay = mgr.TRAIN.WEIGHT_DECAY

    for _, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    if "adam" == mgr.TRAIN.OPTIM.lower():
        from torch.optim import Adam

        mgr.debug(f"Using optimizer Adam: {lr=}, {weight_decay=}")
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    elif "adamw" == mgr.TRAIN.OPTIM.lower():
        from torch.optim import AdamW

        mgr.debug(f"Using optimizer AdamW: {lr=}, {weight_decay=}")
        optimizer = AdamW(params, lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Unknown optimizer: {mgr.TRAIN.OPTIM=}")

    return optimizer
