"""Archiver reads carry their provenance, and never depend on the control system.

Two behaviours, one task:

* **Provenance.** ``archiver_read`` stamps the session's control-system target
  and the archiver's own identity onto both the saved query block and the
  artifact's metadata — the ``bin_size_source`` honesty rule applied to
  identity. The keys are always present: a session with no readable state
  stamps ``target_source="baseline"`` rather than leaving a reader to guess.
* **The carve-out.** The archiver connector is HTTP/pymongo-class, never
  Channel Access, so an archiver read is not routed through the connector-host
  child and must serve identically while no child is alive. Pinned here by a
  full round trip that leaves the host supervisor unbuilt and the
  control-system connector unconstructed.

The third behaviour covered here is the health suite's side of the same fact:
``HealthRuntime`` reports on the deployment as configured, and says so with one
informational row rendered from the shared ``target_banner`` helper.

The target-state directory is redirected into ``tmp_path`` (the fixture shape
``test_phoebus_baseline_guard`` uses) so no test can see — or write — real
session state.
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
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.server_context import (
    get_server_context,
    initialize_server_context,
)
from osprey.stores.artifact_store import get_artifact_store
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
@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Redirect the target-state directory into ``tmp_path``.

    Rebinding the one name ``target_state.state_dir()`` resolves through keeps
    the real deployment's ``var/agent_data`` invisible in both directions.
    """
    root = tmp_path / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    return root / target_state.STATE_DIR_NAME


def write_state(state_dir, *, target, children=None):
    """Write this process's state file naming *target* and owning *children*.

    ``owner_ppid`` is this process's real parent and ``server_pid`` this
    process, which is what makes the resolver match the record: patching those
    lookups away would stop testing the match.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{target_state.STATE_FILE_PREFIX}{os.getpid()}.json"
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": os.getpid(),
                "owner_ppid": os.getppid(),
                "targets": {
                    "live": {"label": "live", "endpoint": "gw:5064", "real_machine": True},
                    "va": {"label": "virtual accelerator", "endpoint": "localhost:5074"},
                },
                "children": list(children or []),
            }
        )
    )
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
async def test_query_block_names_the_session_target_and_the_archiver(archiver_project, state_root):
    """Switched to va on a live baseline: the query block says so, and names the archiver."""
    write_state(state_root, target="va")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "va"
    assert query["target_source"] == "session_switch"
    assert query["archiver_type"] == "mock_archiver"
    assert query["archiver_backend"] == "MockArchiverConnector"


@pytest.mark.unit
async def test_stamp_is_additive_and_leaves_the_existing_query_keys_intact(
    archiver_project, state_root
):
    """The provenance keys are added beside the existing ones, replacing none."""
    write_state(state_root, target="va")

    _, query = await read_and_load(archiver_project, bin_size=60)

    assert query["channels"] == ["SR:DCCT"]
    assert query["processing"] == "raw"
    assert query["bin_size"] == 60
    assert query["bin_size_source"] == "requested"


@pytest.mark.unit
async def test_artifact_metadata_carries_the_same_stamp(archiver_project, state_root):
    """The artifact outlives the session, so the stamp travels in its metadata too."""
    write_state(state_root, target="va")

    response, query = await read_and_load(archiver_project)

    entry = get_artifact_store().get_entry(response["artifact_id"])
    assert entry is not None
    stamp_keys = ("target", "target_source", "archiver_type", "archiver_backend")
    assert {k: entry.metadata[k] for k in stamp_keys} == {k: query[k] for k in stamp_keys}
    # The pre-existing metadata key survives the addition.
    assert entry.metadata["data_type"] == "timeseries"


@pytest.mark.unit
async def test_absent_state_stamps_the_baseline_spelling(archiver_project, state_root):
    """No state file at all — the read happened on the deployment baseline."""
    assert not state_root.exists()

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


@pytest.mark.unit
async def test_unreadable_state_stamps_the_baseline_spelling(archiver_project, state_root):
    """A corrupt record is not an answer, and "baseline" is the honest spelling of that."""
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / f"{target_state.STATE_FILE_PREFIX}{os.getpid()}.json").write_text("{not json")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


@pytest.mark.unit
async def test_a_session_sitting_on_the_baseline_is_spelled_baseline(archiver_project, state_root):
    """A state file naming the baseline target is the same claim as no file."""
    write_state(state_root, target="live")

    _, query = await read_and_load(archiver_project)

    assert query["target"] == "live"
    assert query["target_source"] == "baseline"


# ── the carve-out (CC-2) ────────────────────────────────────────────────────
@pytest.mark.unit
async def test_serves_while_the_connector_host_child_is_dead(archiver_project, state_root):
    """A switched session whose connector-host child is gone still reads history.

    The state file names a target and a child PID that no longer exists — the
    fail-closed situation in which every control-system-routed op refuses. The
    archiver read must complete anyway, with real points, because it never went
    near that child.
    """
    write_state(state_root, target="va", children=[_DEAD_PID])
    assert not target_state.is_process_alive(_DEAD_PID)

    response, query = await read_and_load(archiver_project)

    # 1 Hz over 300 s at the auto bin: real data, not an empty success.
    assert response["summary"]["per_channel"]["SR:DCCT"]["points"] > 0
    assert query["target"] == "va"


@pytest.mark.unit
async def test_read_never_touches_the_control_system(archiver_project, state_root):
    """The path builds no host supervisor and no control-system connector.

    Asserting on the registry's own state rather than on a patched-out call:
    a supervisor that was never constructed cannot have been asked whether a
    child is alive, which is precisely the gate this tool must not have.
    """
    write_state(state_root, target="va", children=[_DEAD_PID])

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
def test_health_row_names_both_targets_while_switched(tmp_path, monkeypatch, state_root):
    """A VA deployment whose session went to live: the row says which is which."""
    _config(tmp_path, monkeypatch, "virtual_accelerator")
    write_state(state_root, target="live")

    row = HealthRuntime.baseline_pinned_row()

    assert row is not None
    assert row.name == BASELINE_ROW_NAME
    assert row.category == BASELINE_ROW_CATEGORY
    assert row.status == "skip"
    assert row.message == (
        "HealthRuntime is pinned to the deployment baseline (va); the session target is live"
    )


@pytest.mark.unit
def test_health_row_is_absent_on_the_baseline(tmp_path, monkeypatch, state_root):
    """Nothing to announce, so nothing is added — an unswitched report is unchanged."""
    _config(tmp_path, monkeypatch, "epics")

    assert HealthRuntime.baseline_pinned_row() is None


@pytest.mark.unit
async def test_suite_opens_with_the_row_while_switched(tmp_path, monkeypatch, state_root):
    """The runner puts the banner first: every row below it describes the baseline."""
    _config(tmp_path, monkeypatch, "epics")
    write_state(state_root, target="va")

    report = await run_health_suite([], runtime=HealthRuntime({"type": "mock"}))

    assert [r.name for r in report.results] == [BASELINE_ROW_NAME]
    assert report.results[0].message.startswith("HealthRuntime is pinned to the deployment")
    # A banner is not a failing check: the run's verdict is untouched.
    assert report.exit_code == 0


@pytest.mark.unit
async def test_suite_adds_no_row_on_the_baseline(tmp_path, monkeypatch, state_root):
    """On the baseline the report is byte-identical to what it was before the row."""
    _config(tmp_path, monkeypatch, "epics")

    report = await run_health_suite([], runtime=HealthRuntime({"type": "mock"}))

    assert report.results == []
