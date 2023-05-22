from typing import Optional

from rich.progress import track


def local_iter(start: int, end: int, step: int, local_rank: Optional[int] = None, desc: Optional[str] = None):
    if local_rank == 0:
        return track(range(start, end, step), description=desc or "Working...")
    else:
        return range(start, end, step)
