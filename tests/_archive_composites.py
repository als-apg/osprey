"""Hold and close the archive composites, and their state directories, a test builds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from osprey_connectors.simulation import archive as archive_module

if TYPE_CHECKING:
    from osprey_connectors.simulation.archive import ArchiveComposite


@contextmanager
def archives_closed(
    monkeypatch: pytest.MonkeyPatch, *namespaces: ModuleType
) -> Iterator[list[ArchiveComposite]]:
    """Record every composite ``archive.build`` returns and close each one on exit.

    ``build`` is replaced on the archive module, which every caller that
    imports it at call time reads, and on each of ``namespaces`` that bound
    the name at import.
    """
    built: list[ArchiveComposite] = []
    real = archive_module.build

    def recording(*args: Any, **kwargs: Any) -> ArchiveComposite:
        composite = real(*args, **kwargs)
        built.append(composite)
        return composite

    monkeypatch.setattr(archive_module, "build", recording)
    for namespace in namespaces:
        monkeypatch.setattr(namespace, "build", recording)
    try:
        yield built
    finally:
        for composite in built:
            composite.close()


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
