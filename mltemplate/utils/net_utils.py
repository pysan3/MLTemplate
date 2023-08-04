from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from logging import Logger

    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from mltemplate.libs.recorder import Recorder


def search_int_pths(model_dir: Path):
    return [int(pth.stem) for pth in model_dir.iterdir() if pth.stem.isnumeric()]


def latest_exists(model_dir: Path):
    return any(pth.name == "latest.pth" for pth in model_dir.iterdir())


def get_best_model_path(model_dir: Path, epoch=-1):
    if epoch > 0:
        return model_dir / f"{epoch:03}.pth"
    if latest_exists(model_dir):
        return model_dir / "latest.pth"
    pths = search_int_pths(model_dir)
    if len(pths) == 0:
        return None
    return model_dir / f"{max(pths):03}.pth"


def load_model(
    net: Module,
    optim: Optimizer,
    scheduler: LRScheduler,
    recorder: Recorder,
    model_dir: Path,
    log: Logger,
    resume: bool = True,
    epoch: int = -1,
) -> int:
    if not resume:
        os.system(f"rm -rf {str(model_dir)}")
    if not model_dir.exists():
        return 0
    model_path = get_best_model_path(model_dir, epoch)
    if model_path is None:
        log.warn(f"Cannot find best model in {model_dir}")
        return 0
    log.warn(f"Loading model: {model_path}")
    pretrained_model = torch.load(str(model_path), "cpu")
    net.load_state_dict(pretrained_model["net"])
    optim.load_state_dict(pretrained_model["optim"])
    scheduler.load_state_dict(pretrained_model["scheduler"])
    recorder.load_state_dict(pretrained_model["recorder"])
    return pretrained_model["epoch"] + 1


def save_model(
    net: Module,
    optim: Optimizer,
    scheduler: LRScheduler,
    recorder: Recorder,
    model_dir: Path,
    log: Logger,
    epoch: int = -1,
    last=False,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "net": net.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "recorder": recorder.state_dict(),
        "epoch": epoch,
    }
    model_name = "latest" if last or epoch < 0 else f"{epoch:03}"
    model_path = model_dir / f"{model_name}.pth"
    torch.save(model, str(model_path))
    log.info(f"Model saved to {model_path}")

    # remove previous pretrained model if the number ofmodels is too big
    pths = search_int_pths(model_dir)
    if len(pths) <= 20:
        return
    min_pth = model_dir / f"{min(pths):03}.pth"
    os.system(f"rm {str(min_pth)}")
    return min_pth


def load_network_pth(log: Logger, net: Module, model_path: Path):
    log.warn(f"Loading model: {model_path}")
    pretrained_model = torch.load(str(model_path), "cpu")
    net.load_state_dict(pretrained_model["net"])
    return pretrained_model["epoch"] + 1


def load_network(log: Logger, net: Module, model_dir: Path, resume=True, epoch=-1):
    if not resume:
        return 0
    if not model_dir.exists():
        log.warn("Pretrained model does not exist.")
        return 0
    if model_dir.is_dir():
        model_path = get_best_model_path(model_dir, epoch)
    elif model_dir.is_file() and model_dir.suffix == ".pth":
        model_path = model_dir
    else:
        log.error(f"Cannot find appropriate model at {model_dir}")
        return 0
    if model_path is None:
        log.warn(f"Cannot find best model in {model_dir}")
        return 0
    return load_network_pth(log, net, model_path)
