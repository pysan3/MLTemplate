from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import torch

from mltemplate.libs.network_wrapper import RenderStats

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from mltemplate.libs.evaluator import Evaluator
    from mltemplate.libs.loader.dataset_loader import DatasetType, DatasetTypeBatch
    from mltemplate.libs.network_wrapper import NetworkWrapper, NetworkWrapperResult
    from mltemplate.libs.recorder import Recorder
    from mltemplate.utils.manager import Manager


def now():
    return datetime.now()


def get_cuda_memory(force_cpu=False):
    if force_cpu:
        return []
    if not torch.cuda.is_available():
        return []
    from torch.cuda import max_memory_allocated

    memory = max_memory_allocated() / 1024 / 1024
    return [f"cuda {memory:.0f} MB"]


class Trainer(object):
    def __init__(self, mgr: Manager, net: Module):
        self.mgr = mgr
        if self.mgr.FORCE_CPU:
            self.device = torch.device("cpu")
        else:
            self.deivce = torch.device(f"cuda:{self.mgr.LOCAL_RANK}")
        net_wrap = NetworkWrapper(self.mgr, net, self.device)
        if self.mgr.DISTRIBUTED:
            net_wrap = torch.nn.parallel.DistributedDataParallel(
                net_wrap,
                device_ids=self.mgr.LOCAL_RANK,
                output_device=self.mgr.LOCAL_RANK,
                find_unused_parameters=True,
            )
        self.netwrapper = net_wrap

    def train(self, epoch: int, data_loader: DataLoader[DatasetType], optimizer: Optimizer, recorder: Recorder):
        max_iter = len(data_loader)
        recorder.set_epoch(epoch)
        self.netwrapper.train()
        start = now()
        for iteration, batch in enumerate(data_loader):
            batch: DatasetTypeBatch
            data_time = now() - start
            # Call network
            result: NetworkWrapperResult = self.netwrapper(batch)
            loss = result.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # End step
            if self.mgr.LOCAL_RANK > 0:
                continue
            # Update recorder loss status
            recorder.update_loss_stats(result.scalar.get_means().to_dict())
            # Record time elapsed
            batch_time = now() - data_time - start
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            # print training state
            if (iteration + 1) % self.mgr.LOG_INTERVAL == 0 or (iteration + 1) == max_iter:
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration - 1)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]["lr"]
                self.mgr.info(
                    f"[bold]{self.mgr.EXP_NAME}[/]: "
                    + ", ".join([f"eta {eta_string}", str(recorder), f"lr {lr}"] + get_cuda_memory(self.mgr.FORCE_CPU))
                )
            # record loss_stats and image_dict
            if (iteration + 1) % self.mgr.RECORD_INTERVAL == 0 or (iteration + 1) == max_iter:
                recorder.record("train")
            recorder.step += 1
            start = now()

    def val(
        self,
        epoch: int,
        data_loader: DataLoader[DatasetType],
        evaluator: Evaluator,
        recorder: Optional[Recorder] = None,
        max_iter: int = -1,
    ):
        self.netwrapper.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        render_stats: Optional[RenderStats] = None
        for iteration, batch in enumerate(data_loader):
            with torch.no_grad():
                result: NetworkWrapperResult = self.netwrapper(batch)
                if evaluator is not None:
                    evaluator.evaluate(result.result, batch, epoch, iteration)
            for k, v in result.scalar.get_means().to_dict().items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v
            if max_iter > 0 and iteration >= max_iter:
                break

        loss_state = ", ".join([f"{k}: {v / data_size:.4f}" for k, v in val_loss_stats.items()])
        self.mgr.info(loss_state)

        if render_stats is None:
            raise RuntimeError(f"Train.val dataloader must iterate more than once. {data_size=}")
        if recorder is not None:
            recorder.record("val", epoch, val_loss_stats, render_stats)  # type: ignore


def make_trainer(mgr: Manager, net: Module):
    return Trainer(mgr, net)
