from dataclasses import dataclass
from pathlib import Path


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetFile:
    base_dir: Path
    "dataset base directory"
    index: int
    "file index"

    def filename(self, post: str, suffix: str):
        post = ("_" if len(post) > 0 else "") + post
        return (self.base_dir / f"{self.index:05}{post}").with_suffix(suffix)

    @property
    def image(self):
        return self.filename("img", "png")

    @property
    def mask(self):
        return self.filename("msk", "png")

    @property
    def data(self):
        return self.filename("data", "json")
