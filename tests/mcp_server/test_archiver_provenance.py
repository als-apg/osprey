"""Archiver reads carry their provenance, and never depend on the control system.

Two behaviours, one task:

* **Provenance.** ``archiver_read`` stamps the session's control-system target
  and the archiver's own identity onto both the saved query block and the
  artifact's metadata — the ``bin_size_source`` honesty rule applied to
  identity. The keys are always present: a deployment with no readable record
  stamps ``target_source="baseline"`` rather than leaving a reader to guess.
* **The carve-out.** The archiver connector is HTTP/pymongo-class, never
  Channel Access, so an archiver read is not routed through the connector-host
  child and must serve identically while no child is alive. Pinned here by a
  full round trip that leaves the host supervisor unbuilt and the
  control-system connector unconstructed.

The third behaviour covered here is the health suite's side of the same fact:
``HealthRuntime`` reports on the deployment as configured, and says so with one
informational row rendered from the shared ``target_banner`` helper.

The agent-data root is stamped into ``tmp_path`` by the ``control_context_root``
fixture (the shape ``test_phoebus_baseline_guard`` uses) so no test can see — or
write — a real deployment's control-context record.
"""

import json
import os

import pytest
import yaml

from osprey.health.runner import run_health_suite
from osprey.health.runtime import (
    BASELINE_ROW_CATEGORY,
    BASELINE_ROW_NAME,
    HealthRuntime,
)
from osprey.mcp_server.control_system.server_context import (
    get_server_context,
    initialize_server_context,
)
from osprey.stores.artifact_store import get_artifact_store
from osprey_connectors import control_context
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

#: A PID no kernel hands out — ``os.kill(pid, 0)`` reports it gone. Stands in
#: for a connector-host child that died.
_DEAD_PID = 2_147_483_646

#: The window and channel the real ``MockArchiverConnector`` serves.
_READ = {
    "channels": ["SR:DCCT"],
    "start_time": "2024-01-15T10:00:00",
    "end_time": "2024-01-15T10:05:00",
}


# ── fixtures / helpers ──────────────────────────────────────────────────────
def write_corrupt_record(root, raw):
    """Put *raw* where the record belongs, bypassing the schema writer."""
    path = control_context.record_path_under(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    control_context.invalidate_cache()
    return path


@pytest.fixture
def archiver_project(tmp_path, monkeypatch):
    """A project CWD wired to the mock archiver on an ``epics`` (live) baseline.

    Both the server context and ``target_banner``'s baseline resolution read
    ``./config.yml``, so one file answers for both. ``epics`` rather than
    ``virtual_accelerator`` because a VA deployment paired with a mock archiver
    is the one pairing the server context refuses outright.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        yaml.dump({"archiver": {"type": "mock_archiver"}, "control_system": {"type": "epics"}})
    )
    initialize_server_context()
    return tmp_path


def archiver_read_fn():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    return get_tool_fn(archiver_read)


async def read_and_load(tmp_path, **overrides):
    """Run one archiver read; return ``(response, saved query block)``."""
    result = await archiver_read_fn()(**{**_READ, **overrides})
    response = extract_response_dict(result)
    assert response["status"] == "success"
    payload = json.loads((tmp_path / response["data_file"]).read_text())
    return response, payload["query"]


# ── the stamp ───────────────────────────────────────────────────────────────
@pytest.mark.unit
async def test_query_block_names_the_recorded_target_and_the_archiver(
    archiver_project, control_context_root, write_control_context
):
    """Switched to va on a live baseline: the query block says so, and names the archiver."""
    write_control_context(control_context_root, target="va")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "va"
    assert query["target_source"] == "session_switch"
    assert query["archiver_type"] == "mock_archiver"
    assert query["archiver_backend"] == "MockArchiverConnector"


@pytest.mark.unit
async def test_stamp_is_additive_and_leaves_the_existing_query_keys_intact(
    archiver_project, control_context_root, write_control_context
):
    """The provenance keys are added beside the existing ones, replacing none."""
    write_control_context(control_context_root, target="va")

    _, query = await read_and_load(archiver_project, bin_size=60)

    assert query["channels"] == ["SR:DCCT"]
    assert query["processing"] == "raw"
    assert query["bin_size"] == 60
    assert query["bin_size_source"] == "requested"


@pytest.mark.unit
async def test_artifact_metadata_carries_the_same_stamp(
    archiver_project, control_context_root, write_control_context
):
    """The artifact outlives the session, so the stamp travels in its metadata too."""
    write_control_context(control_context_root, target="va")

    response, query = await read_and_load(archiver_project)

    entry = get_artifact_store().get_entry(response["artifact_id"])
    assert entry is not None
    stamp_keys = ("target", "target_source", "archiver_type", "archiver_backend")
    assert {k: entry.metadata[k] for k in stamp_keys} == {k: query[k] for k in stamp_keys}
    # The pre-existing metadata key survives the addition.
    assert entry.metadata["data_type"] == "timeseries"


@pytest.mark.unit
async def test_absent_record_stamps_the_baseline_spelling(archiver_project, control_context_root):
    """No record at all — the read happened on the deployment baseline."""
    assert not control_context.record_path_under(control_context_root).exists()

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


@pytest.mark.unit
async def test_unreadable_record_stamps_the_baseline_spelling(
    archiver_project, control_context_root
):
    """A corrupt record is not an answer, and "baseline" is the honest spelling of that."""
    write_corrupt_record(control_context_root, "{not json")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


@pytest.mark.unit
async def test_a_record_sitting_on_the_baseline_is_spelled_baseline(
    archiver_project, control_context_root, write_control_context
):
    """A record naming the baseline target is the same claim as no record."""
    write_control_context(control_context_root, target="live")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


# ── the carve-out (CC-2) ────────────────────────────────────────────────────
@pytest.mark.unit
async def test_serves_while_the_connector_host_child_is_dead(
    archiver_project, control_context_root, write_control_context, write_server_report
):
    """A switched deployment whose connector-host child is gone still reads history.

    The record names a target and the server report names a child PID that no
    longer exists — the fail-closed situation in which every
    control-system-routed op refuses. The archiver read must complete anyway,
    with real points, because it never went near that child.
    """
    write_control_context(control_context_root, target="va")
    write_server_report(
        control_context_root, os.getpid(), applied_target="va", children=[_DEAD_PID]
    )
    assert not control_context.is_process_alive(_DEAD_PID)

    response, query = await read_and_load(archiver_project)

    # 1 Hz over 300 s at the auto bin: real data, not an empty success.
    assert response["summary"]["per_channel"]["SR:DCCT"]["points"] > 0
    assert query["target"] == "va"


@pytest.mark.unit
async def test_read_never_touches_the_control_system(
    archiver_project, control_context_root, write_control_context, write_server_report
):
    """The path builds no host supervisor and no control-system connector.

    Asserting on the registry's own state rather than on a patched-out call:
    a supervisor that was never constructed cannot have been asked whether a
    child is alive, which is precisely the gate this tool must not have.
    """
    write_control_context(control_context_root, target="va")
    write_server_report(
        control_context_root, os.getpid(), applied_target="va", children=[_DEAD_PID]
    )

    await read_and_load(archiver_project)

    registry = get_server_context()
    assert registry._connector_hosts is None
    assert registry._connectors["control_system"].instance is None


# ── the health row ──────────────────────────────────────────────────────────
def _config(tmp_path, monkeypatch, cs_type):
    """Point the config loader at a config.yml declaring *cs_type*."""
    config_file = tmp_path / "osprey_config.yml"
    config_file.write_text(yaml.dump({"control_system": {"type": cs_type}}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.chdir(tmp_path)


@pytest.mark.unit
def test_health_row_names_both_targets_while_switched(
    tmp_path, monkeypatch, control_context_root, write_control_context
):
    """A VA deployment switched to live: the row says which target is which."""
    _config(tmp_path, monkeypatch, "virtual_accelerator")
    write_control_context(control_context_root, target="live")

    row = HealthRuntime.baseline_pinned_row()

    assert row is not None
    assert row.name == BASELINE_ROW_NAME
    assert row.category == BASELINE_ROW_CATEGORY
    assert row.status == "skip"
    assert row.message == (
        "HealthRuntime is pinned to the deployment baseline (va); the deployment is on the live target"
    )


@pytest.mark.unit
def test_health_row_is_absent_on_the_baseline(tmp_path, monkeypatch, control_context_root):
    """Nothing to announce, so nothing is added — an unswitched report is unchanged."""
    _config(tmp_path, monkeypatch, "epics")

    assert HealthRuntime.baseline_pinned_row() is None


@pytest.mark.unit
async def test_suite_opens_with_the_row_while_switched(
    tmp_path, monkeypatch, control_context_root, write_control_context
):
    """The runner puts the banner first: every row below it describes the baseline."""
    _config(tmp_path, monkeypatch, "epics")
    write_control_context(control_context_root, target="va")

    report = await run_health_suite([], runtime=HealthRuntime({"type": "mock"}))

    assert [r.name for r in report.results] == [BASELINE_ROW_NAME]
    assert report.results[0].message.startswith("HealthRuntime is pinned to the deployment")
    # A banner is not a failing check: the run's verdict is untouched.
    assert report.exit_code == 0


@pytest.mark.unit
async def test_suite_adds_no_row_on_the_baseline(tmp_path, monkeypatch, control_context_root):
    """On the baseline the report is byte-identical to what it was before the row."""
    _config(tmp_path, monkeypatch, "epics")

    report = await run_health_suite([], runtime=HealthRuntime({"type": "mock"}))

    assert report.results == []
