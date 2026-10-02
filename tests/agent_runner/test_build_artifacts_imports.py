"""Verify the build-artifact package's public API, and that no copy of it remains at a former path."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


def _has_real_home(name: str) -> bool:
    """Report whether importable code lives at a dotted module name.

    A name has a home when a module file, a package with ``__init__.py``, or a
    namespace directory holding importable source resolves at it. A directory
    holding only ``__pycache__/`` is not a home: bytecode there is never imported
    without its source beside it.

    Args:
        name: Dotted module name to resolve without importing it.

    Returns:
        True when importable code resolves at ``name``.
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False
    if spec.origin is not None:
        return True
    suffixes = tuple(importlib.machinery.all_suffixes())
    return any(
        path.is_file()
        and path.name.endswith(suffixes)
        and "__pycache__" not in path.relative_to(location).parts
        for location in spec.submodule_search_locations or ()
        for path in Path(location).rglob("*")
    )


class _FormerPathRoot:
    """A temporary package root mirroring ``osprey/services/``."""

    def __init__(self, base: str, services: Path) -> None:
        self.base = base
        self.services = services


@pytest.fixture
def former_path_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FormerPathRoot]:
    top = re.sub(r"[^0-9a-zA-Z]", "_", f"formerpath_{tmp_path.name}")
    services = tmp_path / top / "services"
    services.mkdir(parents=True)
    (tmp_path / top / "__init__.py").write_text("")
    (services / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield _FormerPathRoot(f"{top}.services", services)
    for key in [k for k in sys.modules if k == top or k.startswith(f"{top}.")]:
        sys.modules.pop(key, None)


def test_public_api_surface() -> None:
    from osprey.agent_runner.build_artifacts import BuildArtifact, BuildArtifactCatalog

    catalog = BuildArtifactCatalog.default()
    names = catalog.all_names()
    assert len(names) > 0

    sample = catalog.get(names[0])
    assert isinstance(sample, BuildArtifact)
    assert sample.canonical_name == names[0]


def test_legacy_package_removed() -> None:
    assert not _has_real_home("osprey.services.prompts"), (
        "importable code still lives at osprey.services.prompts"
    )


def test_the_catalog_has_no_home_outside_the_harness_adapter() -> None:
    assert not _has_real_home("osprey.services.build_artifacts"), (
        "importable code still lives at osprey.services.build_artifacts"
    )


def test_a_bytecode_only_leftover_is_not_a_home(former_path_root: _FormerPathRoot) -> None:
    pycache = former_path_root.services / "build_artifacts" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "catalog.cpython-313.pyc").write_bytes(b"\x00")
    importlib.invalidate_caches()
    dotted = f"{former_path_root.base}.build_artifacts"

    spec = importlib.util.find_spec(dotted)
    assert spec is not None
    assert spec.origin is None
    assert not _has_real_home(dotted)
    assert not _has_real_home(f"{former_path_root.base}.prompts")


def test_a_package_with_an_init_is_a_home(former_path_root: _FormerPathRoot) -> None:
    package = former_path_root.services / "build_artifacts"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    importlib.invalidate_caches()

    assert _has_real_home(f"{former_path_root.base}.build_artifacts")


def test_a_namespace_dir_holding_source_is_a_home(former_path_root: _FormerPathRoot) -> None:
    package = former_path_root.services / "build_artifacts"
    package.mkdir()
    (package / "catalog.py").write_text("")
    importlib.invalidate_caches()

    assert _has_real_home(f"{former_path_root.base}.build_artifacts")
