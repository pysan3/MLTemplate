from pathlib import Path
from typing import Optional


def local_iter(start: int, end: int, step: int, local_rank: Optional[int] = None, desc: Optional[str] = None):
    if local_rank == 0:
        from rich.progress import track

        return track(range(start, end, step), description=desc or "Working...")
    else:
        return range(start, end, step)


def valid_dir(dir_path: Path | str, do_mkdir=True):
    result = Path(dir_path).expanduser().absolute()
    if do_mkdir:
        result.mkdir(parents=True, exist_ok=True)
    return result
