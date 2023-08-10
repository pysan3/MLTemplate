from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch import Tensor

    from mltemplate.utils.dataset_util import DatasetFile
    from mltemplate.utils.manager import Manager


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetType:
    index: int
    "file data index"
    img: Tensor
    "img loaded with torchvision.read_image. #[RGB, h, w]"

    @property
    def hw(self):
        return torch.tensor(self.img.shape[1:], torch.int)

    @classmethod
    def from_datasetfile(cls, file: DatasetFile):
        from torchvision.io import read_image

        img = read_image(str(file.image))
        self = cls(index=file.index, img=img)
        return self

    def coord2pixel(self, coords: Tensor):
        return coords * self.hw.to(coords.device)[..., None]

    def pixel2index(self, pixels: Tensor):
        return pixels.clip_(torch.zeros(pixels.shape), self.coord2pixel(torch.ones(pixels.shape)))


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetTypeBatch:
    index_batch: Tensor
    "file data index"
    img_batch: Tensor
    "img loaded with torchvision.read_image. #[RGB, h, w]"

    def __len__(self):
        return self.index_batch.shape[0]

    @classmethod
    def from_list(cls, data: list[DatasetType]):
        return cls(
            torch.tensor([d.index for d in data], dtype=torch.int32),
            torch.stack([d.img for d in data]),
        )


class DataAugment:
    data_per_img = 1


class AbstractDataset(Dataset[DatasetType]):
    def __init__(
        self,
        mgr: Manager,
        dataset: Optional[str] = None,
        aug_class: Type[DataAugment] = DataAugment,
        is_test: Optional[bool] = None,
        post_args: Any = None,
    ):
        self.mgr = mgr
        self.aug_class = aug_class
        self.is_test = is_test if is_test is not None else self.mgr.IS_TEST
        self.ds_mgr = self.mgr.TEST if self.is_test else self.mgr.TRAIN
        self.ds_name = dataset or self.ds_mgr.DATASET
        self.data_files: list[DatasetFile] = []
        self.datas: list[DatasetType] = []
        self.post_init(post_args)
        if self.mgr.veryverbose:
            self.mgr.info(f"Found {len(self.data_files)} datas.")
        super().__init__()

    def post_init(self, post_args: Any):
        raise NotImplementedError("Implement code to fill `self.data_files")

    def get_online(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> DatasetType:
        if self.mgr.LOAD_DATA_ON_MEMORY:
            return self.datas[index]
        else:
            return self.get_online(index)

    def __len__(self) -> int:
        return len(self.data_files) * self.aug_class.data_per_img


class DatasetBase(AbstractDataset):
    def get_online(self, index):
        file = self.data_files[index]
        data = DatasetType.from_datasetfile(file)
        return data
