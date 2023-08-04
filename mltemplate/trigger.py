from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.traceback import install

from .libs.evaluator import make_evaluator
from .libs.loader.loader_base import make_data_loader
from .libs.lr_scheduler import make_lr_scheduler, set_lr_scheduler
from .libs.network import Network
from .libs.optimizer import make_optimizer
from .libs.recorder import make_recorder
from .libs.trainer import make_trainer
from .utils.data_utils import local_iter
from .utils.net_utils import load_model, load_network, save_model

if TYPE_CHECKING:
    from .utils.manager import Manager


install()


def train(mgr: Manager):
    train_loader = make_data_loader(mgr, is_test=False, is_distributed=mgr.DISTRIBUTED, max_iter=mgr.NUM_ITER_PER_EP)
    val_loader = make_data_loader(mgr, is_test=True)

    net = Network(mgr)
    trainer = make_trainer(mgr, net)
    optimizer = make_optimizer(mgr, net)
    scheduler = make_lr_scheduler(mgr, optimizer)
    recorder = make_recorder(mgr)
    evaluator = make_evaluator(mgr)

    begin_epoch = load_model(net, optimizer, scheduler, recorder, mgr.TRAINED_MODEL_DIR, mgr, resume=mgr.RESUME)
    mgr.info(f"Resume: {mgr.RESUME}. Starting from epoch: {begin_epoch}.")
    set_lr_scheduler(mgr, scheduler)

    tensorboard_desc = f"Run: [green]tensorboard --logdir {mgr.RECORD_DIR_BASE}[/] to see progress..."
    for epoch in local_iter(begin_epoch, mgr.TRAIN.EPOCH, 1, mgr.LOCAL_RANK, tensorboard_desc):
        recorder.set_epoch(epoch)
        if mgr.DISTRIBUTED:
            train_loader.batch_sampler.sampler.set_epoch(epoch)  # type: ignore
        # train and update
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()
        # save model
        if (epoch + 1) % mgr.SAVE_EP == 0 and mgr.LOCAL_RANK == 0:
            save_model(net, optimizer, scheduler, recorder, mgr.TRAINED_MODEL_DIR, mgr, epoch)
        if (epoch + 1) % mgr.SAVE_LATEST_EP == 0 and mgr.LOCAL_RANK == 0:
            save_model(net, optimizer, scheduler, recorder, mgr.TRAINED_MODEL_DIR, mgr, epoch, last=True)
        if (epoch + 1) % mgr.EVAL_EP == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, 20)
    return net


def test(mgr: Manager):
    val_loader = make_data_loader(mgr, is_test=True)

    net = Network(mgr)
    trainer = make_trainer(mgr, net)
    evaluator = make_evaluator(mgr)
    begin_epoch = load_network(mgr, net, mgr.TRAINED_MODEL_DIR)
    trainer.val(begin_epoch, val_loader, evaluator)


def main(mgr: Manager):
    mgr.info(f"PWD={Path.cwd()}")
    if mgr.IS_TEST:
        mgr.info("Running Test")
        test(mgr)
    else:
        mgr.info("Running Train")
        train(mgr)


def run():
    from mltemplate.utils.manager import Manager

    mgr = Manager.argparse()
    mgr.print_whole_config()
    main(mgr)
