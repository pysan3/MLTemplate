from pathlib import Path
from typing import Optional

from rich.progress import track


def local_iter(start: int, end: int, step: int, local_rank: Optional[int] = None, desc: Optional[str] = None):
    if local_rank == 0:
        return track(range(start, end, step), description=desc or "Working...")
    else:
        return range(start, end, step)


def valid_dir(dir_path: Path):
    result = dir_path.expanduser().absolute()
    result.mkdir(parents=True, exist_ok=True)
    return result
