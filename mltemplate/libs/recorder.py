from __future__ import annotations

import os
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from torch import Tensor

if TYPE_CHECKING:
    from mltemplate.utils.manager import Manager


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over
    a window or the global series average.
    """

    def __init__(self, window_size: int = 20) -> None:
        self.deque = deque(maxlen=window_size)
        self.removed = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value):
        if len(self.deque) == self.deque.maxlen:
            self.removed += self.deque.popleft()
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def sum(self):
        return self.total - self.removed

    @property
    def mean(self):
        return self.sum / len(self.deque)

    @property
    def global_avg(self):
        return self.total / self.count


STATS_T = dict[str, Tensor | SmoothedValue]


class Recorder:
    def __init__(self, mgr: Manager) -> None:
        self.mgr = mgr
        log_dir = self.mgr.RECORD_DIR
        if not self.mgr.RESUME and self.mgr.LOCAL_RANK == 0:
            self.mgr.warn(f"Removing contents in {log_dir}")
            os.system(f"rm -rf {str(log_dir)}/*")
        from torch.utils.tensorboard.writer import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = {}
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = {}
        self.processor = None

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def update_loss_stats(self, loss_dict: dict[str, Tensor]):
        for k, v in loss_dict.items():
            if not isinstance(v, Tensor):
                continue
            self.loss_stats[k].update(v.detach().cpu().item())

    def record(
        self,
        prefix: str,
        step: int = -1,
        loss_stats: Optional[STATS_T] = None,
        render_stats: Optional[dict[str, Any]] = None,
    ):
        if self.mgr.LOCAL_RANK > 0:
            return
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats is not None else self.loss_stats
        for k, v in loss_stats.items():
            self.writer.add_scalar(f"{prefix}/{k}", v.mean if isinstance(v, SmoothedValue) else v, step)
        if self.processor is None or render_stats is None:
            return
        for k, v in render_stats.items():
            self.writer.add_image(f"{prefix}/{k}", v, step)

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict["step"]

    def __str__(self) -> str:
        states: list[str] = []
        states.append(f"ep {self.epoch: 4d}")
        states.append(f"st {self.step:>5}")
        states.extend([f"{k} {v.mean:.5e}" for k, v in self.loss_stats.items()])
        states.append(f"data_sec {self.data_time.mean:.4f}")
        states.append(f"batch_sec {self.batch_time.mean:.4f}")
        return ", ".join(states)


def make_recorder(mgr):
    return Recorder(mgr)
