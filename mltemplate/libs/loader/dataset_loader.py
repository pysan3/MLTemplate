from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from utils.dataset_util import DatasetFile
    from utils.manager import Manager


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetType:
    index: int
    "file data index"
    img: Tensor
    "img loaded with torchvision.read_image. #[RGB, h, w]"
    h: int
    "img height"
    w: int
    "img width"

    @property
    def hw(self):
        return (self.h, self.w)

    @property
    def hw_tn(self):
        return torch.tensor(self.hw, dtype=torch.int32)

    @classmethod
    def from_datasetfile(cls, file: DatasetFile):
        from torchvision.io import read_image

        img = read_image(str(file.image))
        _, h, w = img.shape
        self = cls(
            index=file.index,
            img=img,
            h=h,
            w=w,
        )
        return self

    def coord2pixel(self, coords: Tensor):
        return coords * self.hw_tn[..., None]

    def pixel2index(self, pixels: Tensor):
        return pixels.clip_(torch.zeros(pixels.shape), self.coord2pixel(torch.ones(pixels.shape)))


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetTypeBatch:
    index_batch: Tensor
    "file data index"
    img_batch: Tensor
    "img loaded with torchvision.read_image. #[RGB, h, w]"
    h_batch: Tensor
    "img height"
    w_batch: Tensor
    "img width"

    def __len__(self):
        return self.index_batch.shape[0]

    @classmethod
    def from_list(cls, data: list[DatasetType]):
        return cls(
            torch.tensor([d.index for d in data], dtype=torch.int32),
            torch.stack([d.img for d in data]),
            torch.tensor([d.h for d in data]),
            torch.tensor([d.w for d in data]),
        )


class DataAugment:
    data_per_img = 1


class AbstractDataset(Dataset[DatasetType]):
    def __init__(self, mgr: Manager, dataset: Optional[str] = None, is_train: Optional[bool] = None, post_args=None):
        self.mgr = mgr
        self.is_train = is_train if is_train is not None else not mgr.IS_TEST
        self.ds_mgr = mgr.TRAIN if self.is_train else mgr.TEST
        self.ds_name = dataset or self.ds_mgr.DATASET
        self.data_files: list[DatasetFile] = []
        self.datas: list[DatasetType] = []
        self.post_init(post_args)
        if self.mgr.veryverbose:
            self.mgr.info(f"Found {len(self) // DataAugment.data_per_img} datas.")
        super().__init__()

    def post_init(self, post_args: Any):
        raise NotImplementedError

    def get_online(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> DatasetType:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DatasetBase(AbstractDataset):
    def get_online(self, index):
        file = self.data_files[index]
        data = DatasetType.from_datasetfile(file)
        return data

    def __getitem__(self, index) -> DatasetType:
        if self.mgr.LOAD_DATA_ON_MEMORY:
            return self.datas[index]
        else:
            return self.get_online(index)

    def __len__(self) -> int:
        return len(self.data_files) * DataAugment.data_per_img
