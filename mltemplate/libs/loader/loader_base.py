from __future__ import annotations

from typing import TYPE_CHECKING, Type

import torch.utils.data as data

from . import samplers
from .collator import collate
from .dataset_loader import DataAugment

if TYPE_CHECKING:
    from mltemplate.utils.manager import Manager

    from .dataset_loader import AbstractDataset


def make_dataset(
    mgr: Manager,
    ds_name: str,
    ds_class: Type[AbstractDataset],
    aug_class: Type[DataAugment] = DataAugment,
    is_test=False,
):
    ds = ds_class(mgr, ds_name, aug_class, is_test)
    mgr.debug(f"Using dataset: {ds}")
    return ds


def make_data_sampler(mgr: Manager, dataset: AbstractDataset, shuffle: bool, is_distributed: bool, is_test: bool):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle and not is_test:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    mgr: Manager, sampler: data.Sampler, batch_size: int, drop_last: bool, max_iter: int, is_test: bool
):
    batch_sampler = mgr.train_or_test(is_test).BATCH_SAMPLER
    if batch_sampler == "default":
        batch_sampler = data.BatchSampler(sampler, batch_size, drop_last)
    else:
        raise NotImplementedError(f"Unknown sampler: {batch_sampler}")
    if max_iter > 0:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    import time

    import numpy as np

    return np.random.seed(worker_id * (int(round(time.time() * 1000) % 2**16)))


def make_data_loader(
    mgr: Manager,
    is_test=True,
    is_distributed=False,
    max_iter=-1,
    ds_class: Type[AbstractDataset] = AbstractDataset,
    aug_class: Type[DataAugment] = DataAugment,
):
    if not is_test:
        batch_size = mgr.TRAIN.BATCH_SIZE
        shuffle = mgr.TRAIN.SHUFFLE
        drop_last = False
    else:
        batch_size = mgr.TEST.BATCH_SIZE
        shuffle = True if is_distributed else False
        drop_last = False
    ds_name = mgr.train_or_test(is_test).DATASET
    dataset = make_dataset(mgr, ds_name, ds_class, aug_class, is_test)
    sampler = make_data_sampler(mgr, dataset, shuffle, is_distributed, is_test)
    batch_sampler = make_batch_data_sampler(mgr, sampler, batch_size, drop_last, max_iter, is_test)
    num_workers = mgr.train_or_test(is_test).NUM_WORKERS
    return data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
