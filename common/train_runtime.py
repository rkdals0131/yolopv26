"""Exclusive writer ownership for one durable training run."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator


@contextmanager
def training_run_lock(run_dir: str | Path) -> Iterator[None]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Retain the file so concurrent writers always lock the same inode.
    with (directory / ".train.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"training run already has a writer: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
