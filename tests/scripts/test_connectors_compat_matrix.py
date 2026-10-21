"""The connectors compatibility matrix reads the declared state and refuses an inconsistent one."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "ci" / "connectors_compat_matrix.py"
)
_spec = importlib.util.spec_from_file_location("connectors_compat_matrix", _MODULE_PATH)
compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat)


def _connectors(frameworks: Any) -> dict[str, Any]:
    return {"tool": {"osprey-connectors": {"compat": {"frameworks": frameworks}}}}


def _root(*dependencies: str) -> dict[str, Any]:
    return {"project": {"dependencies": list(dependencies)}}


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("2026.6.2", "2026.10.0"),
        ("2026.9.0b3", "2026.9.0"),
        ("2026.9.0b3", "2026.9.0b3.post3499"),
        ("2026.9.0b3.post3499", "2026.9.0rc1"),
        ("2026.10.0.dev1", "2026.10.0a1"),
        ("2026.10.0", "2026.10.0.post1"),
        ("2026.9.0", "2026.9.1"),
    ],
)
def test_versions_order_as_pep_440_orders_them(lower: str, higher: str) -> None:
    assert compat.version_key(lower) < compat.version_key(higher)


@pytest.mark.parametrize("version", ["latest", "2026.9.0-beta", "v2026.9.0", ""])
def test_a_non_release_version_is_refused(version: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        compat.version_key(version)


def test_the_listed_frameworks_are_read_as_declared() -> None:
    assert compat.supported_frameworks(_connectors(["2026.10.0", "2026.11.0b1"])) == [
        "2026.10.0",
        "2026.11.0b1",
    ]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"tool": {}}, "not declared"),
        (_connectors("2026.10.0"), "list of version strings"),
        (_connectors(["2026.10.0", "2026.10.0"]), "twice"),
        (_connectors(["newest"]), "not a release version"),
    ],
)
def test_a_malformed_list_is_refused(document: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compat.supported_frameworks(document)


def test_the_floor_is_the_one_bare_connectors_floor() -> None:
    root = _root("pyyaml>=6", "osprey-connectors>=2026.10.0", "pyaml-cs-osprey>=2026.6.2")
    assert compat.connectors_floor(root) == "2026.10.0"


@pytest.mark.parametrize(
    "dependencies",
    [("pyyaml>=6",), ("osprey-connectors>=2026.10.0,<2027",), ("osprey-connectors==2026.10.0",)],
)
def test_a_floor_that_is_not_one_bare_floor_is_refused(dependencies: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="expected one"):
        compat.connectors_floor(_root(*dependencies))


@pytest.mark.parametrize(
    ("floor", "applicable"),
    [("2026.6.2", False), ("2026.9.0b3", False), ("2026.10.0", True), ("2026.11.1", True)],
)
def test_the_floor_lane_applies_from_the_first_independent_release(
    floor: str, applicable: bool
) -> None:
    assert compat.floor_applicable(floor) is applicable


def test_an_empty_list_is_one_empty_cell() -> None:
    """GitHub refuses a matrix with no cell; the empty one is the stand-down."""
    assert compat.matrix([]) == {"include": [{"framework": ""}]}


def test_a_listed_release_is_one_cell_each() -> None:
    assert compat.matrix(["2026.10.0", "2026.11.0"]) == {
        "include": [{"framework": "2026.10.0"}, {"framework": "2026.11.0"}]
    }


def test_an_independent_floor_with_nothing_listed_is_refused() -> None:
    assert compat.problems([], compat.FIRST_INDEPENDENT_CONNECTORS)
    assert compat.problems(["2026.10.0"], compat.FIRST_INDEPENDENT_CONNECTORS) == []
    assert compat.problems([], "2026.6.2") == []


def test_the_plan_job_output_names_every_key(tmp_path: Path) -> None:
    output = tmp_path / "github_output"
    result = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "github-output"],
        env={"GITHUB_OUTPUT": str(output), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    pairs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert set(pairs) == {"matrix", "listed", "floor", "floor-applicable"}
    assert json.loads(pairs["matrix"]) == compat.matrix(compat.supported_frameworks())
    assert pairs["floor"] == compat.connectors_floor()
