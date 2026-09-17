"""Fixtures shared by the Microsoft Teams adapter suite.

The adapter's standing promise is that its package root and every module under
it import with neither ``azure-servicebus`` nor ``Pillow`` installed: both live
in the optional ``teams`` extra, and each is imported inside the single function
that needs it. A promise like that is worth exactly as much as its proof, so the
meta-path blockers the proofs run under live here rather than in one suite — the
Service Bus block is read by the ingestion tests and the posting tests alike.

Modelled on ``no_google`` in the Google Chat suite.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest


class BlockAzureImports:
    """A meta-path finder that refuses every ``azure*`` import.

    Placed at the front of ``sys.meta_path``, so it is consulted before the
    installed distribution is found. Raising rather than returning ``None`` is
    what makes the block visible: the importer surfaces the raise as the
    ``ImportError`` the caller would see on a machine without the extra.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "azure" or fullname.startswith("azure."):
            raise ImportError(f"blocked in this test: {fullname}")
        return None


@pytest.fixture
def no_servicebus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``azure*`` import fail, cached ones included.

    Purging the module cache is the load-bearing half: a module another test
    already imported would otherwise satisfy the import without ever reaching
    the finder, and the proof would pass vacuously on a machine that has the
    extra installed. ``monkeypatch`` restores both the cache and ``meta_path``.
    """
    for name in list(sys.modules):
        if name == "azure" or name.startswith("azure."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockAzureImports(), *sys.meta_path])


class BlockPillowImports:
    """A meta-path finder that refuses every ``PIL*`` import.

    The Pillow half of the same promise: the delivery path imports
    ``PIL.Image`` inside ``deliver_files`` and nowhere else, so a deployment
    without the extra loses its images and keeps its answers.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ImportError(f"blocked in this test: {fullname}")
        return None


@pytest.fixture
def no_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``PIL*`` import fail, cached ones included.

    The cache purge matters more here than for Service Bus: Pillow's own
    plugins (``PIL.PngImagePlugin`` and friends) are imported lazily on first
    encode, so a test running under this fixture must not build a real PNG
    itself — the block is deliberately strict enough to stop it.
    """
    for name in list(sys.modules):
        if name == "PIL" or name.startswith("PIL."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockPillowImports(), *sys.meta_path])
