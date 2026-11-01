#!/usr/bin/env python3
"""Print each workspace package whose installed version the tree no longer describes.

An editable install is stamped when it is built, so after a rebase past a release
tag it still answers with the old stream. For the root and every workspace member
this prints ``dist<TAB>installed<TAB>described`` when the release or pre-release of
the two differ, never for the distance, the dev or the local segment. The packages
and their describe commands come from the pyprojects. A package not installed and
a tag that does not describe print nothing: neither can be judged here.

Run inside the project's venv, whose metadata it reads:

    uv run --frozen --no-sync python scripts/installed_versions.py

Stdlib-only.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTORS_VERSION = REPO_ROOT / "scripts" / "va" / "connectors_version.py"

#: The release and pre-release of a normalized version.
_STREAM_RE = re.compile(r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?")


def _load_describe_reader() -> ModuleType:
    """The describe reader, loaded by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("connectors_version", CONNECTORS_VERSION)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {CONNECTORS_VERSION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reader = _load_describe_reader()


def _stream(version_text: str) -> str:
    """The part of *version_text* a tag sets."""
    match = _STREAM_RE.match(version_text)
    return match.group() if match else version_text


def _described(pyproject: Path, root: Path) -> str | None:
    """The version the tree at *root* describes for the package at *pyproject*.

    Every stream's tag ends ``v<calendar>``, so the prefix is the tag up to its last ``v``.
    """
    try:
        command = _reader.describe_command(pyproject)
    except KeyError:
        return None
    done = subprocess.run(["git", "-C", str(root), *command[1:]], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    described = done.stdout.strip()
    prefix, v, _rest = described.rpartition("v")
    return _reader.version_from_describe(described, prefix=prefix + v)


def _pyprojects(root: Path) -> list[Path]:
    """The root's pyproject and each workspace member's."""
    with (root / "pyproject.toml").open("rb") as handle:
        members = tomllib.load(handle)["tool"]["uv"]["workspace"]["members"]
    return [root / "pyproject.toml", *(root / member / "pyproject.toml" for member in members)]


def stale_rows(root: Path = REPO_ROOT) -> list[tuple[str, str, str]]:
    """``(dist, installed, described)`` for every stale package under *root*."""
    rows: list[tuple[str, str, str]] = []
    for pyproject in _pyprojects(root):
        with pyproject.open("rb") as handle:
            dist = tomllib.load(handle)["project"]["name"]
        try:
            installed = version(dist)
        except PackageNotFoundError:
            continue
        described = _described(pyproject, root)
        if described is not None and _stream(installed) != _stream(described):
            rows.append((dist, installed, described))
    return rows


def main() -> int:
    for row in stale_rows(REPO_ROOT):
        print("\t".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
