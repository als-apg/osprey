"""The VA image's connectors version is derived the way the package build derives it.

``scripts/va/connectors_version.py`` reads the describe command from the
package's own pyproject.toml, so there is no second glob to drift; what is
pinned here is that it does read that one, and that the conversion matches the
backend's ``post-release`` / ``node-and-date`` schemes.
"""

from __future__ import annotations

import importlib.util
import shlex
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "connectors_version", REPO_ROOT / "scripts" / "va" / "connectors_version.py"
)
assert _spec and _spec.loader
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

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
