"""The VA image's connectors version is derived the way the package build derives it.

``scripts/va/connectors_version.py`` reads the describe command from the
package's own pyproject.toml, so there is no second glob to drift; what is
pinned here is that it does read that one, and that the conversion matches the
backend's ``post-release`` / ``node-and-date`` schemes.
"""

from __future__ import annotations

import shlex
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests._by_path import load_module

REPO_ROOT = Path(__file__).resolve().parents[2]
cv = load_module("connectors_version", REPO_ROOT / "scripts" / "va" / "connectors_version.py")

NODE = "0123456789abcdef0123456789abcdef01234567"


def test_the_describe_command_is_the_packages_own():
    with (REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml").open("rb") as handle:
        options = tomllib.load(handle)["tool"]["hatch"]["version"]["raw-options"]
    assert cv.describe_command() == shlex.split(options["scm"]["git"]["describe_command"])
    assert "osprey-connectors-v2[0-9][0-9][0-9].*" in cv.describe_command()


@pytest.mark.parametrize(
    ("described", "version"),
    [
        (f"osprey-connectors-v2026.10.0-0-g{NODE}", "2026.10.0"),
        (f"osprey-connectors-v2026.10.0b1-0-g{NODE}", "2026.10.0b1"),
        (f"osprey-connectors-v2026.10.0-3-g{NODE}", "2026.10.0.post3+g012345678"),
        (f"osprey-connectors-v2026.10.0-3-g{NODE}-dirty", "2026.10.0.post3+g012345678.d20261020"),
        (f"osprey-connectors-v2026.10.0-0-g{NODE}-dirty", "2026.10.0.post0+g012345678.d20261020"),
    ],
)
def test_the_conversion_matches_the_backend_schemes(described, version):
    today = datetime(2026, 10, 20, tzinfo=UTC)
    assert cv.version_from_describe(described, today=today) == version


@pytest.mark.parametrize(
    "described", [f"v2026.10.0-3-g{NODE}", f"osprey-connectors-vx-3-g{NODE}", "", "garbage"]
)
def test_anything_but_a_connectors_calendar_describe_is_refused(described):
    assert cv.version_from_describe(described) is None


@pytest.mark.parametrize(
    ("tag_version", "normal"),
    [
        ("2026.10.0", "2026.10.0"),
        ("2026.10.0-rc1", "2026.10.0rc1"),
        ("2026.10.0.RC.1", "2026.10.0rc1"),
        ("2026.10.0beta2", "2026.10.0b2"),
        ("2026.10.0-alpha", "2026.10.0a0"),
    ],
)
def test_the_tag_version_is_normalized(tag_version, normal):
    assert cv.normalize(tag_version) == normal
    described = f"osprey-connectors-v{tag_version}-2-g{NODE}"
    assert cv.version_from_describe(described) == f"{normal}.post2+g012345678"


@pytest.mark.parametrize(
    ("tag_version", "normal"),
    [
        ("2026.10.0-rc1", "2026.10.0rc1"),
        ("2026.10.0.RC.1", "2026.10.0rc1"),
        ("2026.10.0beta2", "2026.10.0b2"),
    ],
)
def test_the_fallback_normalizer_agrees_without_packaging(monkeypatch, tag_version, normal):
    import builtins

    real_import = builtins.__import__

    def no_packaging(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_packaging)
    assert cv.normalize(tag_version) == normal


def test_a_describe_command_may_sit_under_the_flat_key(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.hatch.version]\nraw-options = { git_describe_command = "git describe --long" }\n',
        encoding="utf-8",
    )
    assert cv.describe_command(pyproject) == ["git", "describe", "--long"]


def test_another_stream_converts_under_its_own_prefix():
    assert cv.version_from_describe(f"v2026.10.0b1-0-g{NODE}", prefix="v") == "2026.10.0b1"
    assert cv.version_from_describe(f"v2026.10.0-3-g{NODE}", prefix="v") == (
        "2026.10.0.post3+g012345678"
    )
