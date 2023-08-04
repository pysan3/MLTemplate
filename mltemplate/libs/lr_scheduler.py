from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim import Optimizer, lr_scheduler

if TYPE_CHECKING:
    from mltemplate.utils.manager import Manager


def make_lr_scheduler(mgr: Manager, optimizer: Optimizer):
    mgr_sch = mgr.TRAIN.SCHEDULER
    if mgr_sch.TYPE == "steplr":
        mgr.info(f"Using StepLR: step_size={mgr_sch.STEP_SIZE}, gamma={mgr_sch.GAMMA}")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=mgr_sch.STEP_SIZE, gamma=mgr_sch.GAMMA)
    else:
        raise NotImplementedError(f"Unknown scheduler: {mgr_sch.TYPE}")
    return scheduler


def set_lr_scheduler(mgr: Manager, scheduler: lr_scheduler.LRScheduler):
    mgr_sch = mgr.TRAIN.SCHEDULER
    if mgr_sch.TYPE == "steplr":
        mgr.info(f"Using StepLR: step_size={mgr_sch.STEP_SIZE}, gamma={mgr_sch.GAMMA}")
        assert isinstance(scheduler, lr_scheduler.StepLR), f"{type(scheduler)}"
        scheduler.step_size = mgr_sch.STEP_SIZE
        scheduler.gamma = mgr_sch.GAMMA
    else:
        raise NotImplementedError(f"Unknown scheduler: {mgr_sch.TYPE}")
