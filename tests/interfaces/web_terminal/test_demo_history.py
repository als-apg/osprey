"""The demo history a fresh preset deploy looks back on, end to end without a model.

A project holding the example facility's REAL simulator view activates the
preset's default set, ``[rf-thermal, demo-history]``, through
``activate_default_scenarios`` -- the deploy's one writer of the seed file -- and
a real hub app (no Docker, no model) starts over it. The agent record is the
record's own tailer and reader over an embedded store
(:class:`~docs.screenshots.contact_sheet.HermeticRecord`), so what the hub lists
and replays is what a deployed record makes of the laid-down transcripts.

Pinned, all through the hub's and the gallery's HTTP APIs:

- the sidebar lists four rows (session 3 is ``/clear``-ed into two), each with
  the artifacts its conversation made, and nothing seeded in "Earlier";
- ``/api/session-summary`` lists the seeded artifacts, and the store holds no
  example plot;
- session 2 replays its corrector write as approved, with a rehearsal result
  and no executed-write marker; each row of session 3 replays its own half;
- the seeded ``arrange_workspace`` call ran no panel op;
- session 3's key resumes its post-``/clear`` transcript;
- the seed file is gone once the hub is up;
- an operator's own session, artifact and turn on a seeded session come
  through a hub restart and later ``apply_scenarios`` calls unchanged, with one
  copy of everything;
- the record ingests the transcripts plainly: fed twice, it does not grow.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from docs.screenshots.contact_sheet import HermeticRecord, _append_transcript, _exchange
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import session_meta
from osprey.interfaces.web_terminal.session_handoff import SURFACE_CHAT, _spawn_request
from osprey.simulation.apply import (
    SCENARIO_SESSIONS_FILENAME,
    activate_default_scenarios,
    apply_scenarios,
)
from osprey.stores import session_ledger
from osprey.stores.artifact_store import EXAMPLE_ORIGIN, ArtifactStore
from osprey_connectors import posture_store
from tests.interfaces.web_terminal._started_app import terminal_client
from tests.simulation.test_apply_scenario_sessions import (  # noqa: F401
    HOST,
    VIEW_CONFIG,
    _preset_web_terminals,
    built_view,
)

#: The roster's first readwrite user: the identity this hub serves.
OPERATOR = "alice"

#: The preset's default set.
DEFAULTS = ["rf-thermal", "demo-history"]

SESSION_1 = "sector-7-beam-loss"
SESSION_2 = "sector-3-corrector-nudge"
SESSION_3 = "orbit-table-and-stats"

#: What the operator's own work says, so it can be found again.
OPERATOR_PROMPT = "What is the stored current right now?"
OPERATOR_TURN = "And the sector 7 vacuum right now?"


def _config() -> dict[str, Any]:
    return {
        **VIEW_CONFIG,
        "deployment": {"host_identity": HOST},
        "modules": {"web_terminals": _preset_web_terminals()},
        "simulation": {"default_scenarios": list(DEFAULTS)},
    }


class Deployment:
    """A deployed project, its seed, its record and the means to (re)start its hub."""

    def __init__(self, project: Path, claude_dir: Path, record: HermeticRecord) -> None:
        self.project = project
        self.agent_data = project / "var" / "agent_data"
        self.state_dir = self.agent_data / posture_store.STATE_DIR_NAME / OPERATOR
        self.seed_path = self.state_dir / SCENARIO_SESSIONS_FILENAME
        self.claude_dir = claude_dir
        self.record = record
        self.seed: dict[str, Any] = {}
        self.app: Any = None

    @property
    def sessions_dir(self) -> Path:
        from osprey.agent_runner.project_paths import claude_project_dir

        return claude_project_dir(self.project)

    def session(self, slug: str) -> dict[str, Any]:
        return next(s for s in self.seed["sessions"] if s["slug"] == slug)

    def transcripts(self) -> dict[str, bytes]:
        return {p.stem: p.read_bytes() for p in sorted(self.sessions_dir.glob("*.jsonl"))}

    def store(self) -> ArtifactStore:
        return ArtifactStore(workspace_root=self.agent_data, auto_launch=False)

    def ledger_rows(self) -> list[tuple[str, str, str]]:
        return [tuple(r) for r in session_ledger.load(self.state_dir).rows]

    @contextmanager
    def hub(self) -> Iterator[TestClient]:
        """The hub, started over this project; leaving the block stops it."""
        app_mod = "osprey.interfaces.web_terminal.app"
        patches = (
            patch(
                "osprey_connectors.workspace.resolve_shared_data_root",
                return_value=self.agent_data,
            ),
            patch(f"{app_mod}._load_panel_config", return_value=(set(), [], None)),
            patch(f"{app_mod}._launch_panel_server", lambda *_a, **_k: None),
            patch(
                "osprey.interfaces.web_terminal.routes.session.record_client",
                new=lambda: self.record,
            ),
        )
        with terminal_client(self.agent_data, *patches, project_dir=self.project) as client:
            self.app = client.app
            yield client
        self.app = None

    @contextmanager
    def gallery(self) -> Iterator[TestClient]:
        """The artifacts backend the sidebar reads its rows' artifacts from."""
        from osprey.interfaces.artifacts.app import create_app

        with TestClient(create_app(workspace_root=self.agent_data)) as client:
            yield client


@pytest.fixture
def deployment(built_view, tmp_path, monkeypatch) -> Iterator[Deployment]:  # noqa: F811
    """The preset's default set activated over the real example view; hub not started."""
    from osprey.agent_record.backends.embedded import EmbeddedBackend

    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(built_view, project / "data" / "simulator")
    config = _config()
    (project / "config.yml").write_text(yaml.safe_dump(config))
    claude_dir = tmp_path / "claude"
    for name in ("OSPREY_SESSION_ID", "OSPREY_DISPATCH_RUN_ID", "OSPREY_ARTIFACT_ORIGIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(project / "var" / "agent_data"))
    monkeypatch.setenv("OSPREY_TERMINAL_USER", OPERATOR)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))
    posture_store.invalidate_cache()

    assert activate_default_scenarios(config, project) == ("nominal", *DEFAULTS)

    with ExitStack() as stack:
        backend = stack.enter_context(EmbeddedBackend(tmp_path / "record.duckdb"))
        from osprey.agent_runner.project_paths import claude_project_dir

        sessions_dir = claude_project_dir(project)
        sessions_dir.mkdir(parents=True)
        dep = Deployment(project, claude_dir, HermeticRecord(backend, sessions_dir, OPERATOR))
        dep.seed = json.loads(dep.seed_path.read_text())
        yield dep
    posture_store.invalidate_cache()


# ---------------------------------------------------------------------------
# Reading the hub
# ---------------------------------------------------------------------------


def _rows(client: TestClient) -> list[dict[str, Any]]:
    body = client.get("/api/sessions", params={"drain": "1"}).json()
    assert body["more"] is False
    return body["sessions"]


def _chat(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    body = client.get("/api/session-chat", params={"session_id": session_id, "events": "1"}).json()
    return body["turns"]


def _prompts(turns: list[dict[str, Any]]) -> list[str]:
    return [t["content"] for t in turns if t.get("role") == "user"]


def _events(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for t in turns for e in t.get("events") or []]


def _authored_prompts(part: dict[str, Any]) -> list[str]:
    return [turn["prompt"].strip() for turn in part["turns"]]


# ---------------------------------------------------------------------------
# A fresh deploy
# ---------------------------------------------------------------------------


def test_demo_history_fresh_hub_lists_and_replays_the_seeded_sessions(deployment):
    dep = deployment
    s1, s2, s3 = (dep.session(slug) for slug in (SESSION_1, SESSION_2, SESSION_3))
    part_ids = {
        slug: [p["transcript_id"] for p in dep.session(slug)["parts"]]
        for slug in (SESSION_1, SESSION_2, SESSION_3)
    }
    seeded = {tid for ids in part_ids.values() for tid in ids}
    entry_owner = {
        a["entry_id"]: a["transcript_id"] for s in dep.seed["sessions"] for a in s["artifacts"]
    }
    assert len(seeded) == 4
    assert len(entry_owner) == 3

    with dep.hub() as client:
        # The seed file is gone once the hub is up.
        assert not dep.seed_path.exists()

        # Four rows, one per transcript, each under its session's key.
        rows = _rows(client)
        assert {r["transcript_id"] for r in rows} == seeded
        keys = {r["transcript_id"]: r["session_key"] for r in rows}
        for session in (s1, s2, s3):
            for part in session["parts"]:
                assert keys[part["transcript_id"]] == session["key"]
        titles = {r["transcript_id"]: r["title"] for r in rows}
        for session in (s1, s2, s3):
            for part in session["parts"]:
                assert titles[part["transcript_id"]] == part["title"]

        # Every artifact under the row whose conversation made it.
        with dep.gallery() as gallery:
            summary = gallery.post(
                "/api/artifacts/session-summary", json={"sessions": sorted(seeded)}
            ).json()
        counts = {tid: summary["counts"].get(tid, {}).get("total", 0) for tid in seeded}
        expected = dict.fromkeys(seeded, 0)
        for owner in entry_owner.values():
            expected[owner] += 1
        assert counts == expected
        assert summary["owned_total"] == len(entry_owner)

        # /api/session-summary lists the seeded artifacts, and nothing else.
        listed = client.get("/api/session-summary").json()
        assert {e["id"] for e in listed["entries"]} == set(entry_owner)

        # The store holds no example plot.
        entries = dep.store().list_entries()
        assert {e.id for e in entries} == set(entry_owner)
        assert not [e for e in entries if e.origin == EXAMPLE_ORIGIN]

        # Session 2: the write approved, its result a rehearsal, never executed.
        turns = _chat(client, s2["key"])
        assert _prompts(turns) == _authored_prompts(s2["parts"][0])
        events = _events(turns)
        calls = {e["tool_use_id"]: e for e in events if e["type"] == "tool_use"}
        writes = [c for c in calls.values() if c["tool_name_raw"].endswith("channel_write")]
        assert len(writes) == 1
        (result,) = [
            e
            for e in events
            if e["type"] == "tool_result" and e["tool_use_id"] == writes[0]["tool_use_id"]
        ]
        assert result["decision"] == "allowed"
        assert not result["is_error"]
        assert "Rehearsal" in result["preview"]
        assert "executed" not in json.dumps(events).lower()

        # Each row of session 3 replays its own half, opened as the sidebar opens
        # a row: the key's current row by its key, the other by entering it.
        before, after = s3["parts"]
        current = {r["transcript_id"]: r["current_for_key"] for r in rows}
        assert current[after["transcript_id"]] and not current[before["transcript_id"]]
        assert _prompts(_chat(client, s3["key"])) == _authored_prompts(after)
        entered = client.post(f"/api/sessions/{before['transcript_id']}/enter").json()
        assert entered["session_key"] not in (None, s3["key"])
        assert _prompts(_chat(client, entered["session_key"])) == _authored_prompts(before)

        # The seeded arrange_workspace call ran no panel op.
        app = client.app
        assert not getattr(app.state, "panel_ops", None)
        assert not getattr(app.state, "panel_ops_unanswered", None)

        # Session 3's key resumes the conversation after its /clear.
        request = client.portal.call(_spawn_request, app, s3["key"], SURFACE_CHAT)
        assert request.resume_id == after["transcript_id"]

    # The record ingests the transcripts plainly: fed twice, four and no growth.
    dep.record.ingest()
    dep.record.ingest()
    sessions = dep.record.list_sessions().data
    assert sorted(s["session_id"] for s in sessions) == sorted(seeded)


# ---------------------------------------------------------------------------
# The operator's work survives restarts and later applies
# ---------------------------------------------------------------------------


def _add_operator_work(dep: Deployment) -> tuple[str, str]:
    """An operator session with an artifact, and a turn on seeded session 1."""
    import uuid

    from osprey.stores import session_ledger as ledger

    key = str(uuid.uuid4())
    now = datetime.now(UTC).replace(microsecond=0)
    _append_transcript(dep.sessions_dir, key, _exchange(key, OPERATOR_PROMPT, "About 500 mA.", now))
    ledger.append(dep.state_dir, key, key, now - timedelta(seconds=1))
    entry = dep.store().save_file(
        b"500 mA",
        "current.txt",
        "text",
        "Stored current now",
        mime_type="text/plain",
        tool_source="channel_read",
        session_id=key,
    )
    s1 = dep.session(SESSION_1)["key"]
    _append_transcript(
        dep.sessions_dir,
        s1,
        _exchange(s1, OPERATOR_TURN, "3.7e-10 Torr.", now + timedelta(seconds=5)),
    )
    return key, entry.id


def _snapshot(dep: Deployment) -> dict[str, Any]:
    entries = dep.store().list_entries()
    return {
        "transcripts": dep.transcripts(),
        "ledger": sorted(dep.ledger_rows()),
        "artifacts": sorted((e.id, e.session_id, e.title) for e in entries),
    }


def _assert_one_copy(dep: Deployment, client: TestClient, expected: dict[str, Any]) -> None:
    assert not dep.seed_path.exists()
    assert _snapshot(dep) == expected
    rows = _rows(client)
    tids = [r["transcript_id"] for r in rows]
    assert len(tids) == len(set(tids)) == len(expected["transcripts"])
    listed = client.get("/api/session-summary").json()
    ids = [e["id"] for e in listed["entries"]]
    assert sorted(ids) == sorted(a[0] for a in expected["artifacts"])


def test_demo_history_operator_work_survives_restart_and_reapply(deployment):
    dep = deployment
    with dep.hub():
        pass
    key, entry_id = _add_operator_work(dep)
    expected = _snapshot(dep)
    s1 = dep.session(SESSION_1)["key"]
    assert OPERATOR_TURN in expected["transcripts"][s1].decode()
    assert key in expected["transcripts"]
    assert entry_id in {a[0] for a in expected["artifacts"]}

    with dep.hub() as client:
        _assert_one_copy(dep, client, expected)
        assert any(r["transcript_id"] == key for r in _rows(client))
        turns = _chat(client, s1)
        assert _prompts(turns)[-1] == OPERATOR_TURN

    for names in (["vacuum-burst"], DEFAULTS):
        apply_scenarios(dep.project, names, seed_logbook=False, seed_archive=False)
        assert not dep.seed_path.exists(), f"apply_scenarios({names}) wrote a seed"
        with dep.hub() as client:
            _assert_one_copy(dep, client, expected)

    # The record ingests plainly: five transcripts, fed twice, no growth.
    dep.record.ingest()
    first = sorted(s["session_id"] for s in dep.record.list_sessions().data)
    dep.record.ingest()
    second = sorted(s["session_id"] for s in dep.record.list_sessions().data)
    assert first == second == sorted(expected["transcripts"])


def test_demo_history_ledger_names_every_seeded_key_once(deployment):
    """Restarting the hub over a laid-down history appends no ledger row."""
    dep = deployment
    with dep.hub():
        pass
    rows = dep.ledger_rows()
    with dep.hub():
        pass
    assert dep.ledger_rows() == rows
    assert len(rows) == len(set(rows)) == 4
    assert session_meta.ledger_state_dir() == dep.state_dir
