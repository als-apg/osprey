"""The built osprey-facility wheel ships the published schema and its compat module.

A fork or a tool outside Python validates a facility.json against the JSON
Schema the package hands out through
:func:`osprey_facility.schema.published_schema`. That accessor reads package
data, so the files must be inside the wheel at the path it resolves, not only in
the source tree. The test reads one built wheel: the one ``OSPREY_FACILITY_WHEEL``
names (the package's standalone CI job builds it once and sets the variable), or,
when the variable is unset, one it builds itself into a temporary directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import pytest

import osprey_facility
from osprey_facility.schema import published_schema
from osprey_facility.validate import FACILITY_HEADER
from osprey_facility_tests._harness._paths import REPO_ROOT

pytestmark = pytest.mark.wheel_build

#: The variable naming an already built wheel.
WHEEL_VARIABLE = "OSPREY_FACILITY_WHEEL"

#: The command that builds the wheel when the variable is unset.
BUILD = ("uv", "build", "--package", "osprey-facility", "--wheel")

#: The compatibility classifier the schema test reads, as a wheel member.
COMPAT_MODULE = "osprey_facility/schema/compat.py"

#: The directory the installed ``osprey_facility`` package sits in.
_SITE = Path(osprey_facility.__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The built wheel: the one the environment names, or one built here."""
    named = os.environ.get(WHEEL_VARIABLE)
    if named:
        path = Path(named)
        assert path.is_file(), f"{WHEEL_VARIABLE}={named} names no file"
        return path
    uv = shutil.which("uv")
    assert uv, f"{WHEEL_VARIABLE} is unset and uv is not on PATH to build the wheel"
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [uv, *BUILD[1:], "--out-dir", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    built = sorted(out.glob("osprey_facility-*.whl"))
    assert len(built) == 1, f"{' '.join(BUILD)} wrote {[p.name for p in built]}"
    return built[0]


@pytest.fixture(scope="module")
def members(wheel: Path) -> dict[str, bytes]:
    """Every file in the wheel, by its archive path."""
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _member(schema) -> tuple[str, bytes]:
    """The wheel path ``published_schema`` resolved to, and the bytes it reads there."""
    path = Path(str(schema)).resolve()
    return PurePosixPath(path.relative_to(_SITE)).as_posix(), schema.read_bytes()


def _current_version() -> int:
    return int(FACILITY_HEADER.rsplit("/", 1)[-1])


@pytest.mark.parametrize(
    "version",
    [None, "floor"],
    ids=["current", "floor-of-current-header"],
)
def test_the_wheel_ships_each_published_schema_where_the_accessor_resolves(
    members: dict[str, bytes], version: str | None
) -> None:
    schema = published_schema(None if version is None else _current_version())
    member, content = _member(schema)
    assert member.startswith("osprey_facility/schema/_generated/"), member
    assert member in members, f"the wheel ships no {member}"
    assert members[member] == content, f"the wheel's {member} differs from the package's"


def test_the_wheel_ships_the_compat_module(members: dict[str, bytes]) -> None:
    assert COMPAT_MODULE in members, f"the wheel ships no {COMPAT_MODULE}"


def test_the_wheel_ships_nothing_from_the_repository_checks(members: dict[str, bytes]) -> None:
    """Control: the reader sees archive paths, so an absent file reads as absent."""
    assert "repo_checks.py" not in members
    assert not any(name.startswith("osprey/") for name in members)
