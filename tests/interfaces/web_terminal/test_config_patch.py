"""Tests for PATCH /api/config endpoint (comment-preserving config updates).

Uses a minimal FastAPI app with just the config routes to avoid lifespan
complexity (PTY, file watchers, etc.) that can crash in test environments.

These cover the *mechanics* of the patch — type handling, comment preservation,
the backup, the error shapes. They therefore drive the endpoint with keys it may
actually write: ``control_system.*`` and ``approval.*`` are in the protected set
and now come back 403, which is exercised in test_config_routes.py. Using one
here would turn a mechanics test into a refusal test without saying so, and
several of these assert on the file rather than the status, so they would pass
vacuously against a file nothing had touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_log
from osprey.agent_record.client import ClientAnswer
from osprey.interfaces.web_terminal.routes import config as config_routes
from osprey.interfaces.web_terminal.routes import router
from osprey.utils.config_writer import config_update_fields
from tests.agent_record.fakes import FakeRecordClient

SAMPLE_CONFIG = """\
# ============================================================
# Test Config
# ============================================================
# Comments must survive PATCH operations.

project_name: "test-project"

control_system:
  type: "virtual_accelerator"  # Options: virtual_accelerator | epics
  writes_enabled: false  # Master safety switch
  limits_checking:
    enabled: false
    on_violation: "skip"

approval:
  enabled: true

artifact_server:
  host: "127.0.0.1"
  port: 10200
  auto_launch: true
"""


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project with config.yml."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(project_dir):
    """Minimal FastAPI test client with just the routes router and config state."""
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    # The lifespan resolves this tier flag; a routes-only app states it.
    app.state.config_panel_enabled = True
    with TestClient(app) as c:
        yield c


class TestPatchEndpoint:
    """Test PATCH /api/config for structured field updates."""

    def test_patch_multiple_fields(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={
                "updates": {
                    "artifact_server.auto_launch": False,
                    "artifact_server.host": "0.0.0.0",
                    "artifact_server.port": 7777,
                    "project_name": "renamed-project",
                }
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["fields_updated"] == 4

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["artifact_server"]["auto_launch"] is False
        assert data["artifact_server"]["host"] == "0.0.0.0"
        assert data["artifact_server"]["port"] == 7777
        assert data["project_name"] == "renamed-project"

    def test_patch_preserves_comments(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.auto_launch": False}},
        )
        # Asserted explicitly: every other assertion here is about the file, so a
        # refused patch would satisfy them all against untouched bytes.
        assert resp.status_code == 200
        text = (project_dir / "config.yml").read_text()
        assert "# ============================================================" in text
        assert "# Test Config" in text
        assert "# Comments must survive PATCH operations." in text
        assert "# Options: virtual_accelerator | epics" in text
        assert "# Master safety switch" in text

    def test_patch_empty_updates_rejected(self, client):
        resp = client.patch("/api/config", json={"updates": {}})
        assert resp.status_code == 422

    def test_patch_no_config_file(self, client):
        client.app.state.config_path = Path("/nonexistent/config.yml")
        resp = client.patch(
            "/api/config",
            json={"updates": {"key": "value"}},
        )
        assert resp.status_code == 404

    def test_patch_preserves_key_order(self, client, project_dir):
        original = yaml.safe_load((project_dir / "config.yml").read_text())
        original_keys = list(original.keys())

        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.port": 1234}},
        )

        # A refused or failed patch leaves the file untouched and the order equal.
        assert resp.status_code == 200
        updated = yaml.safe_load((project_dir / "config.yml").read_text())
        assert updated["artifact_server"]["port"] == 1234
        updated_keys = list(updated.keys())
        assert original_keys == updated_keys


class TestGetEndpoint:
    """Verify GET /api/config still works."""

    def test_get_returns_sections_and_raw(self, client, project_dir):
        """The Form view gets only the allowlisted sections; Raw YAML gets the file."""
        resp = client.get("/api/config")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["sections"]) == {"control_system", "approval", "artifact_server"}
        assert "project_name" not in body["sections"]
        assert body["path"] == str(project_dir / "config.yml")
        assert "# Test Config" in body["raw"]


class TestPutEndpointStillWorks:
    """Ensure the existing PUT /api/config still works for raw YAML saves."""

    def test_put_raw_yaml(self, client, project_dir):
        # The replacement keeps SAMPLE_CONFIG's protected blocks (control_system,
        # approval) exactly as they are and moves only unprotected keys. PUT is
        # gated on a protected-set *document diff* now, so a body that dropped
        # them -- as this one used to -- is a refusal, and this test would be
        # asserting the gate rather than the raw-save mechanics it is about.
        new_yaml = SAMPLE_CONFIG.replace('project_name: "test-project"', "project_name: updated")
        new_yaml += "key: value\n"
        resp = client.put(
            "/api/config",
            json={"raw": new_yaml},
        )
        assert resp.status_code == 200
        assert resp.json()["requires_restart"] is True

        text = (project_dir / "config.yml").read_text()
        assert "project_name: updated" in text

    def test_put_invalid_yaml_rejected(self, client):
        resp = client.put(
            "/api/config",
            json={"raw": "invalid: yaml: [unterminated"},
        )
        assert resp.status_code == 422


class TestHookDebugEndpoints:
    """``/api/hooks/debug-status`` and ``/api/hooks/debug-log``.

    The status reads ``app.state.config_path``; the log reads hook ``debug``
    entries from the agent record through the module's ``record_client``,
    replaced here by a fake. The lifespan that sets the app state is pinned by
    the app suites.
    """

    @pytest.mark.parametrize("debug", [True, False])
    def test_debug_status_reads_config(self, client, project_dir, debug):
        config_update_fields(project_dir / "config.yml", {"hooks.debug": debug})

        resp = client.get("/api/hooks/debug-status")

        assert resp.status_code == 200
        assert resp.json()["enabled"] is debug

    def test_debug_status_is_off_without_a_hooks_block(self, client):
        resp = client.get("/api/hooks/debug-status")

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_debug_log_reads_hook_debug_entries_from_the_record(self, client, monkeypatch):
        items = [
            _debug_item("2026-03-02T10:00:01Z", hook="approval", tool="Write", status="ask"),
            _debug_item("2026-03-02T10:00:00Z", hook="writes_check", tool="Bash", status="ok"),
        ]
        fake = FakeRecordClient({"entries": ClientAnswer(data=items, complete=True)})
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        resp = client.get("/api/hooks/debug-log?limit=50")

        assert resp.status_code == 200
        body = resp.json()
        assert body["record"] == "complete"
        assert [e["tool"] for e in body["entries"]] == ["Write", "Bash"]
        assert body["entries"][0] == items[0]["body"]
        (call,) = fake.calls_to("entries")
        assert call["kind"] == "debug"
        assert call["source"] == "hook"
        assert call["limit"] == 50

    def test_debug_log_clamps_the_limit(self, client, monkeypatch):
        fake = FakeRecordClient()
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        client.get("/api/hooks/debug-log?limit=0")
        client.get("/api/hooks/debug-log?limit=100000")

        assert [c["limit"] for c in fake.calls_to("entries")] == [1, 500]

    def test_debug_log_says_record_unavailable_when_the_record_does_not_answer(
        self, client, monkeypatch
    ):
        fake = FakeRecordClient(unavailable="connection refused")
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        resp = client.get("/api/hooks/debug-log")

        assert resp.status_code == 200
        assert resp.json() == {
            "entries": [],
            "record": "unavailable",
            "reason": "connection refused",
        }

    def test_debug_log_marks_an_incomplete_answer(self, client, monkeypatch):
        items = [_debug_item("2026-03-02T10:00:00Z", hook="approval", tool="Write")]
        fake = FakeRecordClient(
            {"entries": ClientAnswer(data=items, complete=False, reason="scan limit")}
        )
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        body = client.get("/api/hooks/debug-log").json()

        assert body["record"] == "incomplete"
        assert body["reason"] == "scan limit"
        assert len(body["entries"]) == 1

    def test_debug_log_drops_any_debug_entry_not_written_by_a_hook(self, client, monkeypatch):
        transcript = {"kind": "debug", "ts": "2026-03-02T10:00:02Z", "body": {"type": "x"}}
        items = [transcript, _debug_item("2026-03-02T10:00:00Z", hook="approval", tool="Write")]
        fake = FakeRecordClient({"entries": ClientAnswer(data=items, complete=True)})
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        entries = client.get("/api/hooks/debug-log").json()["entries"]

        assert [e["hook"] for e in entries] == ["approval"]

    def test_transcript_debug_entries_never_appear_in_the_panel(
        self, client, monkeypatch, tmp_path, spool_zone
    ):
        """Hook and transcript both produce ``debug`` entries; only the hook's are served.

        Both are written and ingested for real, and read back through the
        record's own reader with the ``source`` the route asks for.
        """
        from osprey.agent_record import contract, writer
        from osprey.agent_record.reader import RecordReader

        monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
        hook_log._debug_from_config = None
        hook_log.log_hook("approval", {"tool_name": "channel_write"}, status="ask")
        # The transcript's debug body: the line type and its raw text, no source.
        writer.write(contract.KIND_DEBUG, {"type": "progress", "raw": "{}"})
        assert len(spool_zone.recorded_entries("debug")) == 2

        reader = RecordReader(spool_zone._backend, spool_zone.config())

        def entries(kind, since=None, limit=None, source=None, **_):
            answer = reader.entries(
                contract.Own(spool_zone.terminal), kind, since, limit, source=source
            )
            return ClientAnswer(
                data=[item.to_json() for item in answer.data],
                complete=answer.complete,
                reason=answer.reason,
            )

        fake = FakeRecordClient({"entries": entries})
        monkeypatch.setattr(config_routes, "record_client", lambda: fake)

        body = client.get("/api/hooks/debug-log").json()

        assert body["record"] == "complete"
        assert [(e["hook"], e["status"]) for e in body["entries"]] == [("approval", "ask")]


def _debug_item(ts, **fields):
    """One hook debug entry as the record client serves it."""
    body = {"source": "hook", "ts": ts, **fields}
    return {"kind": "debug", "ts": ts, "body": body}


class TestPanelGateFailsClosed:
    """An app that never resolved ``web.config_panel.enabled`` is refused.

    The lifespan always sets the flag; an app mounted without it (an embedder,
    a routes-only app) has made no tier decision, and a tier gate that has no
    decision to read refuses rather than opens.
    """

    @pytest.fixture
    def flagless_client(self, project_dir):
        app = FastAPI()
        app.include_router(router)
        app.state.config_path = project_dir / "config.yml"
        app.state.project_cwd = str(project_dir)
        with TestClient(app) as c:
            yield c

    @pytest.mark.parametrize(
        "send",
        [
            pytest.param(lambda c: c.get("/api/config"), id="get-config"),
            pytest.param(lambda c: c.put("/api/config", json={"raw": SAMPLE_CONFIG}), id="put"),
            pytest.param(
                lambda c: c.patch("/api/config", json={"updates": {"project_name": "x"}}),
                id="patch",
            ),
            pytest.param(lambda c: c.get("/api/claude-setup"), id="get-claude-setup"),
            pytest.param(
                lambda c: c.put("/api/claude-setup", json={"path": "CLAUDE.md", "content": "x"}),
                id="put-claude-setup",
            ),
            pytest.param(
                lambda c: c.post(
                    "/api/claude-setup", json={"path": ".claude/agents/x.md", "content": "x"}
                ),
                id="post-claude-setup",
            ),
        ],
    )
    def test_every_verb_is_refused_without_the_flag(self, flagless_client, project_dir, send):
        before = (project_dir / "config.yml").read_bytes()

        resp = send(flagless_client)

        assert resp.status_code == 403
        assert "web.config_panel.enabled" in resp.json()["detail"]
        assert (project_dir / "config.yml").read_bytes() == before
        assert not (project_dir / ".claude").exists()
