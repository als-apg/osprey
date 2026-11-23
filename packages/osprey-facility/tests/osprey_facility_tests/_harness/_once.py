"""A session fixture's result, made once per run and shared by every xdist worker.

The framework-free twin of the root suite's ``tests._builds._built_once``: the
facility suite cannot import that one. Outside xdist the session makes the
thing in a ``mktemp`` directory. Under xdist the first worker to ask makes it,
holding a lock, in a directory in the run's root that every worker reads, and
pickles the result beside it; every later worker reads that record. The run's
root is the one ``_run_temp`` (or the root suite's harness) claimed, and goes
with it.
"""

from __future__ import annotations

import fcntl
import os
import pickle
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

__all__ = ["built_once"]


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` for the block."""
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def built_once(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    record_name: str,
    make: Callable[[Path], Any],
) -> Any:
    """What ``make`` returns for a directory of its own, made once per run.

    Args:
        tmp_path_factory: The session's temp path factory.
        name: The name the directory handed to ``make`` starts with.
        record_name: The record's file name in the run's root.
        make: Makes the thing in the directory it is handed.

    Returns:
        What ``make`` returned.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        return make(tmp_path_factory.mktemp(name))
    shared = tmp_path_factory.getbasetemp().parent
    record = shared / record_name
    with _exclusive(shared / f"{record_name}.lock"):
        if record.is_file():
            return pickle.loads(record.read_bytes())
        made = make(Path(tempfile.mkdtemp(prefix=f"{name}-", dir=shared)))
        record.write_bytes(pickle.dumps(made))
        return made
