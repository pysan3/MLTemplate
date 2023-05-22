from __future__ import annotations

from typing import TYPE_CHECKING, Type

import torch.utils.data as data

from . import samplers

if TYPE_CHECKING:
    from mltemplate.libs.loader.dataset_loader import AbstractDataset
    from utils.manager import Manager


def make_dataset(mgr: Manager, ds_name: str, ds_class: Type[AbstractDataset], is_train=True):
    ds = ds_class(mgr, ds_name, is_train)
    mgr.debug(f"Using dataset: {ds}")
    return ds


def make_data_sampler(mgr: Manager, dataset: AbstractDataset, shuffle: bool, is_distributed: bool, is_train: bool):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle and is_train:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    mgr: Manager, sampler: data.Sampler, batch_size: int, drop_last: bool, max_iter: int, is_train: bool
):
    batch_sampler = (mgr.TRAIN if is_train else mgr.TEST).BATCH_SAMPLER
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
    mgr: Manager, is_train=True, is_distributed=False, max_iter=-1, ds_class: Type[AbstractDataset] = AbstractDataset
):
    if is_train:
        batch_size = mgr.TRAIN.BATCH_SIZE
        shuffle = mgr.TRAIN.SHUFFLE
        drop_last = False
    else:
        batch_size = mgr.TEST.BATCH_SIZE
        shuffle = True if is_distributed else False
        drop_last = False
    ds_name = mgr.train_or_test(is_train).DATASET
    dataset = make_dataset(mgr, ds_name, ds_class)
    sampler = make_data_sampler(mgr, dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(mgr, sampler, batch_size, drop_last, max_iter, is_train)
    num_workers = mgr.train_or_test(is_train).NUM_WORKERS
    return data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
