"""Hold the state directories of the archive composites a test builds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from osprey_connectors.simulation import archive as archive_module


class _HeldTemporaryDirectory(TemporaryDirectory):
    """A temporary directory kept alive until an explicit cleanup removes it."""

    held: list[TemporaryDirectory[str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).held.append(self)


@contextmanager
def held_state_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[TemporaryDirectory[str]]]:
    """Every state directory ``archive.build`` makes, none released by the collector.

    Each is held until exit, so a directory that is gone was removed by an
    explicit cleanup; exit removes what is left.
    """
    held: list[TemporaryDirectory[str]] = []
    monkeypatch.setattr(_HeldTemporaryDirectory, "held", held)
    monkeypatch.setattr(archive_module.tempfile, "TemporaryDirectory", _HeldTemporaryDirectory)
    try:
        yield held
    finally:
        for state in held:
            state.cleanup()
