from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim import lr_scheduler

if TYPE_CHECKING:
    from utils.manager import Manager


def make_lr_scheduler(mgr: Manager, optimizer):
    mgr_sch = mgr.TRAIN.SCHEDULER
    if mgr_sch.TYPE == "steplr":
        mgr.info(f"Using StepLR: step_size={mgr_sch.STEP_SIZE}, gamma={mgr_sch.GAMMA}")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=mgr_sch.STEP_SIZE, gamma=mgr_sch.GAMMA)
    else:
        raise NotImplementedError(f"Unknown scheduler: {mgr_sch.TYPE}")
    return scheduler


def set_lr_scheduler(mgr: Manager, scheduler):
    mgr_sch = mgr.TRAIN.SCHEDULER
    if mgr_sch.TYPE == "steplr":
        mgr.info(f"Using StepLR: step_size={mgr_sch.STEP_SIZE}, gamma={mgr_sch.GAMMA}")
        scheduler.step_size = mgr_sch.STEP_SIZE
    else:
        raise NotImplementedError(f"Unknown scheduler: {mgr_sch.TYPE}")
    scheduler.gamma = mgr_sch.GAMMA
