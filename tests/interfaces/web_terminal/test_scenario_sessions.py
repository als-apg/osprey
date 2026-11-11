"""Tests for ``osprey.interfaces.web_terminal.scenario_sessions`` (the hub lay-down).

The seed file is built the way the deploy builds it: a ``sessions`` block resolved
against an anchor and written with :func:`to_dict`, plus the ids and artifact
payloads the deploy adds.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.interfaces.web_terminal import scenario_sessions, session_meta, transcript_map
from osprey.simulation.apply import SCENARIO_SESSIONS_FILENAME
from osprey.simulation.sessions import parse_sessions, resolve, to_dict
from osprey.stores import session_ledger
from osprey.stores.artifact_store import ArtifactStore, operator_visible, owner_of
from osprey_connectors import control_context, posture_store

ANCHOR = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
SCENARIO = "demo-history"

KEY_1 = "11111111-1111-5111-8111-111111111111"
KEY_3 = "33333333-3333-5333-8333-333333333333"
KEY_3_PART_2 = "33333333-3333-5333-8333-444444444444"
ENTRY_1 = "a1a1a1a1a1a1"
ENTRY_3 = "b3b3b3b3b3b3"


def _turn(prompt: str, answer: str, offset: int = 0) -> dict:
    return {"prompt": prompt, "at_offset_s": offset, "steps": [{"answer": answer}]}


BLOCK = [
    {
        "slug": "first",
        "title": "First look",
        "when": {"days_ago": 3, "time": "10:00:00"},
        "parts": [
            {
                "title": "Looking",
                "turns": [_turn("How is the beam?", "Stable."), _turn("And now?", "Same.", 120)],
            }
        ],
    },
    {
        "slug": "cleared",
        "title": "Cleared and continued",
        "when": {"days_ago": 1, "time": "14:00:00"},
        "parts": [
            {"title": "Before clear", "turns": [_turn("Check the orbit.", "Orbit is fine.")]},
            {"title": "After clear", "turns": [_turn("Now the tunes.", "Tunes ok.", 600)]},
        ],
    },
]


def _seed_document() -> dict:
    resolved = resolve(parse_sessions(BLOCK), ANCHOR)
    ids = [[KEY_1], [KEY_3, KEY_3_PART_2]]
    entries = [ENTRY_1, ENTRY_3]
    sessions = []
    for rs, tids, entry in zip(resolved, ids, entries, strict=True):
        plain = to_dict(rs)
        parts = [
            {"transcript_id": tid, **part}
            for tid, part in zip(tids, plain.pop("parts"), strict=True)
        ]
        sessions.append(
            {
                "scenario": SCENARIO,
                "key": tids[0],
                **plain,
                "parts": parts,
                "artifacts": [
                    {
                        "entry_id": entry,
                        "transcript_id": tids[-1],
                        "at": rs.parts[-1].turns[0].at.isoformat(),
                        "filename": f"{entry}.txt",
                        "title": f"Plot {entry}",
                        "artifact_type": "text",
                        "mime_type": "text/plain",
                        "kind": "",
                        "tool_source": "archiver_read",
                        "content_b64": base64.b64encode(entry.encode()).decode(),
                    }
                ],
            }
        )
    return {"version": 1, "anchor": ANCHOR.isoformat(), "sessions": sessions}


def _app(**state) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(**state))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A hub app over a pinned state dir, a transcript dir and a store root."""
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(root))
    control_context.invalidate_cache()
    sessions_dir = tmp_path / "claude-projects" / "proj"
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.session_discovery.claude_project_dir",
        lambda _project: sessions_dir,
    )
    store_root = tmp_path / "store-root"
    store_root.mkdir()
    app = _app(project_cwd=str(tmp_path / "project"), panel_status_root=store_root)
    state_dir = session_meta.ledger_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    yield SimpleNamespace(
        app=app,
        sessions_dir=sessions_dir,
        store_root=store_root,
        state_dir=state_dir,
        seed=state_dir / SCENARIO_SESSIONS_FILENAME,
    )
    control_context.invalidate_cache()


def _write_seed(path: Path) -> None:
    path.write_text(json.dumps(_seed_document()))


def _store(h) -> ArtifactStore:
    return ArtifactStore(workspace_root=h.store_root, auto_launch=False)


def _rows(h) -> list[tuple]:
    return [tuple(r) for r in session_ledger.load(h.state_dir).rows]


def _first_turn_at(session: int, part: int) -> str:
    resolved = resolve(parse_sessions(BLOCK), ANCHOR)
    return resolved[session].parts[part].turns[0].at


def test_scenario_sessions_fresh_lay_down_fills_every_store_and_deletes_seed(hub):
    _write_seed(hub.seed)

    report = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report.laid == (KEY_1, KEY_3)
    assert report.incomplete == ()
    assert report.seed_removed
    assert not hub.seed.exists()
    for tid in (KEY_1, KEY_3, KEY_3_PART_2):
        path = hub.sessions_dir / f"{tid}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert {line["sessionId"] for line in lines} == {tid}
        last = datetime.fromisoformat(lines[-1]["timestamp"])
        assert path.stat().st_mtime == pytest.approx(last.timestamp(), abs=1)
    assert {r[0] for r in _rows(hub)} == {KEY_1, KEY_3}
    meta = session_meta.get(hub.app, KEY_3_PART_2)
    assert meta["scenario"] == SCENARIO
    assert meta["title"] == "After clear"
    assert session_meta.get(hub.app, KEY_1)["scenario"] == SCENARIO
    entries = {e.id: e for e in _store(hub).list_entries()}
    assert set(entries) == {ENTRY_1, ENTRY_3}
    entry = entries[ENTRY_3]
    assert entry.session_id == KEY_3
    assert entry.metadata["transcript_id"] == KEY_3_PART_2
    assert entry.metadata["scenario"] == SCENARIO
    assert entry.timestamp.startswith(str(_first_turn_at(1, 1).date()))


def test_scenario_sessions_artifacts_belong_to_the_seeded_session_that_made_them(hub):
    """The agent sees seeded artifacts by the ownership rule, as any session's:
    each is the transcript's that laid it down, never hidden and never unowned."""
    _write_seed(hub.seed)
    scenario_sessions.lay_down_scenario_sessions(hub.app)

    ledger = session_ledger.load(hub.state_dir)
    entries = {e.id: e for e in _store(hub).list_entries()}
    assert all(operator_visible(e) for e in entries.values())
    assert owner_of(entries[ENTRY_1], ledger) == KEY_1
    assert owner_of(entries[ENTRY_3], ledger) == KEY_3_PART_2


def test_scenario_sessions_refuses_a_transcript_id_that_is_not_a_uuid(hub):
    document = _seed_document()
    document["sessions"][0]["parts"][0]["transcript_id"] = "../../escaped"
    hub.seed.write_text(json.dumps(document))

    report = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report.incomplete == (KEY_1,)
    assert not report.seed_removed
    assert not (hub.sessions_dir.parent.parent / "escaped.jsonl").exists()
    assert list(hub.sessions_dir.parent.parent.rglob("escaped.jsonl")) == []


def test_scenario_sessions_without_seed_file_is_a_no_op(hub):
    report = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report == scenario_sessions.LayDownReport()
    assert not hub.sessions_dir.exists()
    assert not session_ledger.exists(hub.state_dir)
    assert _store(hub).list_entries() == []


def test_scenario_sessions_interrupted_lay_down_completes_next_run(hub, monkeypatch):
    _write_seed(hub.seed)
    real_append = session_ledger.append_many

    with monkeypatch.context() as m:
        m.setattr(session_ledger, "append_many", lambda *_a, **_k: [])
        report = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report.laid == ()
    assert set(report.incomplete) == {KEY_1, KEY_3}
    assert hub.seed.exists()
    assert (hub.sessions_dir / f"{KEY_1}.jsonl").exists()
    assert _store(hub).list_entries() == []
    assert session_ledger.append_many is real_append

    report = scenario_sessions.lay_down_scenario_sessions(hub.app)
    again = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report.seed_removed and not hub.seed.exists()
    assert again == scenario_sessions.LayDownReport()
    rows = _rows(hub)
    assert len(rows) == len(set(rows)) == 3
    assert sorted(e.id for e in _store(hub).list_entries()) == [ENTRY_1, ENTRY_3]


def test_scenario_sessions_second_run_never_rewrites_extended_transcript(hub):
    _write_seed(hub.seed)
    scenario_sessions.lay_down_scenario_sessions(hub.app)
    path = hub.sessions_dir / f"{KEY_1}.jsonl"
    extended = path.read_text() + json.dumps({"type": "user", "note": "operator"}) + "\n"
    path.write_text(extended)

    _write_seed(hub.seed)
    report = scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert report.seed_removed
    assert path.read_text() == extended
    assert len(_rows(hub)) == 3
    assert len(_store(hub).list_entries()) == 2


def test_scenario_sessions_map_points_cleared_key_at_its_last_transcript(hub):
    _write_seed(hub.seed)

    scenario_sessions.lay_down_scenario_sessions(hub.app)

    assert transcript_map.get(hub.app, KEY_3) == KEY_3_PART_2
    assert KEY_1 not in transcript_map._mapping(hub.app)
    on_disk = json.loads(transcript_map.store_path().read_text())
    assert on_disk[KEY_3] == KEY_3_PART_2


def test_scenario_sessions_ledger_rows_start_at_each_parts_first_turn(hub):
    _write_seed(hub.seed)

    scenario_sessions.lay_down_scenario_sessions(hub.app)

    ledger = session_ledger.load(hub.state_dir)
    rows = {(r.session_key, r.transcript_id): r.from_ts for r in ledger.rows}
    assert rows[(KEY_1, KEY_1)] == _first_turn_at(0, 0).isoformat()
    assert rows[(KEY_3, KEY_3)] == _first_turn_at(1, 0).isoformat()
    assert rows[(KEY_3, KEY_3_PART_2)] == _first_turn_at(1, 1).isoformat()
    assert ledger.current(KEY_3) == KEY_3_PART_2


def test_scenario_sessions_artifacts_ignore_dispatch_environment(hub, monkeypatch):
    monkeypatch.setenv("OSPREY_SESSION_ID", "some-other-session")
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-123")
    monkeypatch.setenv("OSPREY_ARTIFACT_ORIGIN", "input")
    _write_seed(hub.seed)

    with patch("osprey.infrastructure.server_launcher.ensure_web_server") as launch:
        scenario_sessions.lay_down_scenario_sessions(hub.app)

    launch.assert_not_called()
    entries = _store(hub).list_entries()
    assert {e.session_id for e in entries} == {KEY_1, KEY_3}
    assert {e.run_id for e in entries} == {""}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0o000 file")
def test_scenario_sessions_unreadable_seed_file_is_left_with_a_warning(hub, caplog):
    _write_seed(hub.seed)
    hub.seed.chmod(0o000)
    try:
        with caplog.at_level("WARNING"):
            report = scenario_sessions.lay_down_scenario_sessions(hub.app)
        assert report == scenario_sessions.LayDownReport()
        assert hub.seed.exists()
        assert str(hub.seed) in caplog.text
    finally:
        hub.seed.chmod(0o600)
    assert not session_ledger.exists(hub.state_dir)


def test_scenario_sessions_lay_down_runs_after_backfill_at_startup(tmp_path):
    from tests.interfaces.web_terminal._started_app import terminal_client

    calls: list[str] = []

    def lay_down(_app):
        calls.append("lay_down")
        raise RuntimeError("boom")  # never fatal

    patches = (
        patch.object(session_meta, "backfill_ledger", lambda _app: calls.append("backfill")),
        patch.object(scenario_sessions, "lay_down_scenario_sessions", lay_down),
    )
    with terminal_client(tmp_path, *patches) as client:
        assert client.get("/health").status_code in (200, 404)

    assert calls == ["backfill", "lay_down"]
