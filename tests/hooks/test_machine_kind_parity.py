"""The published machine kind: one reader in the package, one in the hook twin.

C-KIND: the controls server publishes ``machine_kind`` per target in its
report, and :func:`osprey_connectors.types.published_machine_kind` reads it
back. The stdlib-only hook restates that reader; both are driven here over the
same report fixtures, so a ``standin`` reads ``rehearsal`` on one side exactly
when it does on the other.

The round trip at the bottom starts at the writer: ``target_display_metadata``
decides the kind with the predicate that picks the label, the report writer's
normalizer keeps the key, and both readers read it off disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import target_display_metadata
from osprey_connectors import posture_store
from osprey_connectors.types import published_machine_kind
from tests.mcp_server.test_control_target_roster import standin_config

#: A PID no test host runs.
DEAD_PID = 2_147_483_646

#: PID 1 is alive on every host this suite runs on.
INIT_PID = 1

_REHEARSAL = {"standin": {"machine_kind": "rehearsal"}}

#: (id, reports as (pid, payload-or-raw-text), expected kind of ``standin``).
#: ``None`` as the pid means this test process, which is alive.
REPORT_FIXTURES: list[tuple[str, list[tuple[int | None, Any]], str]] = [
    ("live-standin", [(None, {"targets": _REHEARSAL})], "rehearsal"),
    ("standin-elsewhere", [(None, {"targets": {"standin": {"machine_kind": "real"}}})], "real"),
    ("no-report", [], "real"),
    ("no-key", [(None, {"targets": {"standin": {"label": "LIVE MACHINE (stand-in)"}}})], "real"),
    ("dead-server", [(DEAD_PID, {"targets": _REHEARSAL})], "real"),
    ("odd-value", [(None, {"targets": {"standin": {"machine_kind": "Rehearsal"}}})], "real"),
    ("meta-not-object", [(None, {"targets": {"standin": "rehearsal"}})], "real"),
    ("targets-not-object", [(None, {"targets": ["standin"]})], "real"),
    ("corrupt-report", [(None, "{not json")], "real"),
    (
        "first-live-with-targets-wins",
        [
            (INIT_PID, {"targets": {}}),
            (DEAD_PID, {"targets": {"standin": {"machine_kind": "real"}}}),
            (None, {"targets": _REHEARSAL}),
        ],
        "rehearsal",
    ),
    (
        "name-order-decides",
        [
            (INIT_PID, {"targets": {"standin": {"machine_kind": "real"}}}),
            (None, {"targets": _REHEARSAL}),
        ],
        "real",
    ),
]


def _write_reports(directory: Path, reports: list[tuple[int | None, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pid, payload in reports:
        pid = os.getpid() if pid is None else pid
        text = payload if isinstance(payload, str) else json.dumps({"server_pid": pid, **payload})
        (directory / f"server_{pid}.json").write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("reports", "expected"),
    [(reports, expected) for _, reports, expected in REPORT_FIXTURES],
    ids=[fixture_id for fixture_id, _, _ in REPORT_FIXTURES],
)
def test_both_readers_agree_over_the_report_fixtures(tmp_path, reports, expected):
    _write_reports(tmp_path, reports)

    package = published_machine_kind("standin", state_dir=tmp_path)
    twin = reader.published_machine_kind("standin", state_dir=str(tmp_path))

    assert package == twin == expected


@pytest.mark.parametrize(("target", "kind"), [("live", "real"), ("va", "simulator"), ("x", "real")])
def test_no_report_makes_any_other_target_a_rehearsal(tmp_path, target, kind):
    _write_reports(tmp_path, [(None, {"targets": {target: {"machine_kind": "rehearsal"}}})])

    assert published_machine_kind(target, state_dir=tmp_path) == kind
    assert reader.published_machine_kind(target, state_dir=str(tmp_path)) == kind


def test_both_readers_resolve_the_same_directory_by_default(tmp_path, monkeypatch):
    """With no ``state_dir``, both read the bound control-context directory."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(tmp_path))
    _write_reports(tmp_path, [(None, {"targets": _REHEARSAL})])

    assert published_machine_kind("standin") == reader.published_machine_kind("standin")
    assert reader.published_machine_kind("standin") == "rehearsal"


def test_the_name_tables_match():
    assert reader._MACHINE_KINDS == {
        target: posture_store.machine_kind(
            {"type": "epics", "connector": {"live_standin": {}}}, target
        )
        for target in ("live", "standin", "va")
    }
    assert reader._MACHINE_KINDS["standin"] == "real"
    assert "published_machine_kind" in reader.__all__


# -- round trip: the server writes, both readers read ------------------------


@pytest.fixture
def report_dir(tmp_path, monkeypatch) -> Path:
    """The writer's and both readers' state directory, bound to one path."""
    directory = tmp_path / "state"
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    return directory


@pytest.mark.parametrize(
    ("config_kwargs", "expected"),
    [
        ({}, "rehearsal"),
        ({"standin_port": None}, "real"),
        ({"gateway_host": "gateway.example.org"}, "real"),
    ],
    ids=["deployed-standin", "no-standin-deployed", "standin-off-loopback"],
)
def test_a_round_trip_keeps_the_key(report_dir, config_kwargs, expected):
    config = standin_config(**config_kwargs)
    metadata = target_display_metadata(config)
    assert metadata["standin"]["machine_kind"] == expected
    assert metadata["va"]["machine_kind"] == "simulator"
    assert metadata["live"]["machine_kind"] == "real"
    # The label and the kind come from one predicate.
    assert (metadata["standin"]["label"] == "Rehearsal — nothing moves") is (
        expected == "rehearsal"
    )
    assert "LIVE MACHINE (stand-in)" not in metadata["standin"]["label"]

    target_state.write_server_record(metadata)

    written = json.loads(target_state.report_file_path().read_text(encoding="utf-8"))
    assert written["targets"]["standin"]["machine_kind"] == expected
    assert published_machine_kind("standin") == expected
    assert reader.published_machine_kind("standin") == expected
    assert reader.published_machine_kind("standin", state_dir=str(report_dir)) == expected
