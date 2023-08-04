from mltemplate.libs.loader.dataset_loader import DatasetType, DatasetTypeBatch


def collate(batch: list[DatasetType]):
    return DatasetTypeBatch.from_list(batch)
