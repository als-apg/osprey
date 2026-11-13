"""A fake installed application: the fixture package on the path, an entry point naming it.

``tests/apps/fixtures/installed_app/`` stands for a distribution's site-packages:
its ``fixture_installed_app/`` package is the application directory. The entry
point is faked rather than installed, so the tests choose the distribution and
version each one needs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from osprey.apps.origin import ENTRY_POINT_GROUP

#: The directory that stands for site-packages.
SITE = Path(__file__).resolve().parent / "fixtures" / "installed_app"
#: The fixture application: its name, package and directory.
NAME = "fixture-installed"
MODULE = "fixture_installed_app"
DIRECTORY = SITE / MODULE
DIST = "fixture-installed-app"
VERSION = "1.0.0"


@dataclass(frozen=True)
class FakeDist:
    name: str
    version: str


@dataclass(frozen=True)
class FakeEntryPoint:
    """The attributes discovery reads of an :class:`importlib.metadata.EntryPoint`."""

    name: str
    value: str
    dist: FakeDist | None
    group: str = ENTRY_POINT_GROUP
    extras: list[str] = field(default_factory=list)

    @property
    def module(self) -> str:
        return self.value.split(":")[0]


def entry_point(
    name: str = NAME, module: str = MODULE, dist: str = DIST, version: str = VERSION
) -> FakeEntryPoint:
    """One ``osprey.apps`` entry point, declared by *dist* at *version*."""
    return FakeEntryPoint(name, module, FakeDist(dist, version))


def install(monkeypatch: pytest.MonkeyPatch, *entry_points: FakeEntryPoint) -> list[FakeEntryPoint]:
    """Put the fixture package on the path and make the environment declare *entry_points*.

    With none given, the fixture application at :data:`VERSION`. Every other group
    reads through to the real registry.
    """
    eps = list(entry_points) or [entry_point()]
    monkeypatch.syspath_prepend(str(SITE))
    real: Callable[..., Any] = metadata.entry_points

    def fake(**params: Any) -> Iterable[Any]:
        if params.get("group") == ENTRY_POINT_GROUP:
            return list(eps)
        return real(**params)

    monkeypatch.setattr(metadata, "entry_points", fake)
    return eps
