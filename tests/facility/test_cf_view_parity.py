"""Channel-finder view parity against the frozen pre-LINE goldens.

The goldens under ``tests/facility/golden/`` are captured from the demo's
committed sources by ``scripts/facility_demo/fingerprint.py`` (the limits
golden by ``scripts/facility_demo/_limits.py``); each names the command that
rewrites it in its ``_reproduce`` line. This module's loader
reads them, and the tests below hold the frozen invariants every later
comparison relies on: the fingerprint's size, role split and pinned sha256,
the in_context size, the standalone address set, the limits projection and
the byte identity of the pre-LINE channel-finder index copies.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
CF_INDEX_DIR = GOLDEN_DIR / "cf_index_pre_line"

#: The benchmark queries the pre-LINE indexes are scored against; every query
#: names the addresses it targets in ``targeted_pv``.
PRE_LINE_QUERIES = (
    REPO_ROOT
    / "src/osprey/templates/apps/control_assistant/data/benchmarks"
    / "cross_paradigm/queries/tier3_queries.json"
)

#: The channel-finder pipelines with an index file, keyed by their index name.
PRE_LINE_INDEXES = ("hierarchical", "in_context", "middle_layer")

FINGERPRINT_SHA256 = "644f41850784e0251ac4be0081c6891b7f933b119efc037a7455ebe89c54e9c0"
FINGERPRINT_ROW_KEYS = ["address", "role", "value_type", "names", "description"]


@cache
def load_golden(name: str) -> Any:
    """Parse ``tests/facility/golden/<name>``."""
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def fingerprint_rows() -> list[dict[str, Any]]:
    """The frozen demo fingerprint rows, sorted by address."""
    return load_golden("demo_fingerprint.json")["rows"]


def fingerprint_addresses() -> set[str]:
    return {row["address"] for row in fingerprint_rows()}


@cache
def pre_line_index(index: str) -> Any:
    """Parse the frozen pre-LINE channel-finder index ``index``."""
    return json.loads((CF_INDEX_DIR / f"{index}.json").read_text(encoding="utf-8"))


@cache
def pre_line_queries() -> list[dict[str, Any]]:
    """The benchmark queries the pre-LINE indexes answer."""
    return json.loads(PRE_LINE_QUERIES.read_text(encoding="utf-8"))


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    compact = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _golden_files() -> list[str]:
    return sorted(
        path.relative_to(GOLDEN_DIR).as_posix()
        for path in GOLDEN_DIR.rglob("*.json")
        if path.parent != CF_INDEX_DIR or path.name == "MANIFEST.json"
    )


@pytest.mark.parametrize("name", _golden_files())
def test_every_golden_names_its_reproduce_command(name: str) -> None:
    golden = load_golden(name)
    assert isinstance(golden, dict)
    assert golden.get("_reproduce"), f"{name} has no _reproduce line"


def test_fingerprint_is_frozen() -> None:
    golden = load_golden("demo_fingerprint.json")
    rows = fingerprint_rows()
    assert len(rows) == golden["count"] == 2908
    assert [row["address"] for row in rows] == sorted(fingerprint_addresses())
    assert all(list(row) == FINGERPRINT_ROW_KEYS for row in rows)
    roles = {"setpoint": 0, "readback": 0}
    for row in rows:
        roles[row["role"]] += 1
    assert roles == golden["roles"] == {"setpoint": 396, "readback": 2512}
    assert {row["value_type"] for row in rows} == {"bool", "float"}
    assert sum(row["value_type"] == "bool" for row in rows) == 1246
    assert _rows_sha256(rows) == golden["sha256"] == FINGERPRINT_SHA256


def test_fingerprint_additions_are_new_addresses_in_the_row_shape() -> None:
    rows = load_golden("demo_fingerprint_additions.json")["rows"]
    addresses = [row["address"] for row in rows]
    assert addresses == sorted(set(addresses))
    assert all(list(row) == FINGERPRINT_ROW_KEYS for row in rows)
    assert not set(addresses) & fingerprint_addresses()


def test_in_context_size_is_frozen() -> None:
    golden = load_golden("in_context_size.json")
    assert golden["size"] == len(golden["rows"]) == 569
    assert {row["address"] for row in golden["rows"]} <= fingerprint_addresses()


def test_standalone_address_set_is_frozen() -> None:
    golden = load_golden("cf_standalone_addresses.json")
    addresses = golden["addresses"]
    assert golden["count"] == len(addresses) == 1228
    assert addresses == sorted(set(addresses))


def test_limits_name_only_setpoints_of_the_fingerprint() -> None:
    golden = load_golden("limits.json")
    channels = golden["channels"]
    assert "defaults" not in golden
    assert golden["count"] == len(channels) == 3
    setpoints = {row["address"] for row in fingerprint_rows() if row["role"] == "setpoint"}
    assert set(channels) <= setpoints
    assert all(
        list(record) == ["min_value", "max_value", "max_step", "writable", "confirm"]
        for record in channels.values()
    )


def test_pre_line_index_copies_match_their_manifest() -> None:
    manifest = load_golden("cf_index_pre_line/MANIFEST.json")["files"]
    assert sorted(manifest) == [f"{index}.json" for index in PRE_LINE_INDEXES]
    for name, entry in manifest.items():
        content = (CF_INDEX_DIR / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], name


def test_pre_line_in_context_index_holds_the_fingerprint_addresses() -> None:
    rows = pre_line_index("in_context")["channels"]
    assert {row["address"] for row in rows} == fingerprint_addresses()


def test_pre_line_queries_target_fingerprint_addresses() -> None:
    queries = pre_line_queries()
    assert len(queries) == 60
    addresses = fingerprint_addresses()
    for query in queries:
        assert query["targeted_pv"], query["user_query"]
        assert set(query["targeted_pv"]) <= addresses, query["user_query"]
