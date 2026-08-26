"""Tests for the ``/api/config`` write surface: the protected-set gate, and regen.

Two things are under test here.

**The protected-set gate.** ``config.yml`` carries the write gate, the approval
gate, the agent's rendered permission surface and the paths the safety layers
derive their allow and deny areas from. A surface that can set those keys is a
surface that can un-gate the agent, so neither of the panel's two write paths
into the file may, whoever is driving them:

* ``PATCH /api/config`` is the structured path, judged key by key against the
  protected set -- :class:`TestPatchProtectedKeys`.
* ``PUT /api/config`` is the Raw YAML path, which replaces the whole document
  verbatim and so names nothing. It is judged by a document diff: the protected
  *view* of the incoming bytes must equal the protected view of the bytes on
  disk. One comparison, every mutation class -- a key added, deleted or changed,
  a family removed wholesale, a subtree reshaped in either direction -- and it
  is done on segment tuples, never dotted strings, so a raw key containing a
  ``.`` cannot collide with a real nested one and hide a change behind it.
  :class:`TestPutProtectedDocument`.

Three properties are load-bearing on both paths and each is asserted directly:

* **Nothing moves.** A refused PATCH leaves ``config.yml`` byte-identical and
  creates no backup. Byte comparison rather than a parsed one, because a
  round-trip that rewrote the file with the same values would still be a write;
  and the backup counts too, since it is a copy of a file this request turned out
  not to be allowed to touch. The backup no longer lands beside ``config.yml`` --
  it goes into the agent-data state zone, which is the tree that stays writable
  once the render is root-owned -- so these pins look for it there.
* **All-or-nothing.** One protected key refuses the whole body. A mixed request
  does not get to land its cosmetic half.
* **It leaves a trace.** One ``http_config`` ledger record per refused key
  and one activity frame, so the attempt is visible to the operator afterwards
  and not only to whoever saw the 403. The frame is stamped in-process, which is
  the point: no ``OSPREY_PANEL_TOKEN`` is needed to publish it.

**Regeneration.** The GitHub #244 fix: a config write re-renders the ``.claude/``
artifacts so safety-critical fields take effect on the next terminal restart.
That path now belongs to PUT, because no key that shapes the render survives the
PATCH gate -- which is the protected set doing exactly its job. The PATCH regen
tests below therefore run on ``claude_code.default_model``, an agent-relevant key
the panel may still write, and assert the honest outcome: the write lands, and
there is nothing to re-render. PUT keeps a live regen because unprotected keys
that *do* shape the render still exist -- ``channel_finder.pipeline_mode`` is
one, and it moves ``settings.json`` -- while the safety-critical fields it used
to be demonstrated with are now refused before any of it happens.

Uses a minimal FastAPI app over the routes router (no lifespan) pointed at a real
built project so ``regenerate_claude_code`` has artifacts to update.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.audit import writer
from osprey.audit.protected import SURFACE_HTTP_CONFIG
from osprey.cli.profile_conventions import (
    PROTECTED_CONFIG_KEYS,
    is_protected_key_path,
    protected_view,
)
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX
from osprey.interfaces.web_terminal.routes.config import _changed_protected_keys
from osprey.utils.config_writer import config_update_fields
from osprey.utils.identity import acting_identity

#: An unprotected key the Config panel may still write. ``claude_code`` is an
#: agent-relevant section, but only its ``permissions``/``hooks``/``servers``
#: subtrees are protected -- which is the shape of the whole rule: the panel
#: still tunes the agent, it just cannot rewrite the surface that constrains it.
COSMETIC_KEY = "claude_code.default_model"

#: An unprotected key that still shapes the render, so a PUT carrying only this
#: proves both halves of the surface at once: the gate lets it through, and the
#: #244 regen fires. ``control_system.writes_enabled`` used to play this role and
#: no longer can -- it is the write gate, which is the whole point of the gate.
RENDER_SHAPING_KEY = ("channel_finder", "pipeline_mode")


@pytest.fixture
def built_project(tmp_path):
    """A real OSPREY project with writes disabled (channel_write baked into deny)."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="route-regen",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project_dir / "config.yml", {"control_system.writes_enabled": False})
    manager.regenerate_claude_code(project_dir)
    return project_dir


@pytest.fixture
def audit_zone(tmp_path, monkeypatch):
    """Redirect the audit zone. ``writer.audit_dir`` is the ledger's one seam."""
    zone = tmp_path / "audit-zone" / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    return zone


@pytest.fixture
def client(built_project, audit_zone):
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = built_project / "config.yml"
    app.state.project_cwd = str(built_project)
    # The real app carries this ring; a refusal frame is only observable with it.
    app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    with TestClient(app) as c:
        yield c


def _deny(project_dir):
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    return settings["permissions"]["deny"]


def _audit_records(zone):
    """Every ``http_config`` record this identity filed, oldest first."""
    path = zone / acting_identity() / f"{SURFACE_HTTP_CONFIG}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _refusal_frames(client):
    return [
        frame
        for frame in client.app.state.agent_activity_ring
        if frame["tool"] == "config_patch_refused"
    ]


def _backup_dir(project_dir):
    """Where a config backup must land, resolved the way the route resolves it.

    Deliberately re-derived from ``agent_data.base_dir`` through the same public
    pair the app uses rather than spelled as a literal: the point of the move is
    that the location follows the configured state zone, and a hard-coded
    ``var/agent_data`` here would pass whether the route read the config or not.
    ``test_patch_backup_follows_a_relocated_agent_data_root`` is the case that
    actually separates the two.
    """
    from osprey.utils.workspace import agent_data_base_dir, anchored_path

    try:
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    except yaml.YAMLError:
        # Mirrors the route: a config too broken to name its own state zone still
        # gets backed up, into the framework default. Needed because one pin
        # deliberately corrupts the file and then asserts *no* backup appeared.
        config = None
    return anchored_path(agent_data_base_dir(config), project_dir) / "config-backups"


def _backup_path(project_dir):
    """The single backup slot for ``config.yml``, in the state zone."""
    return _backup_dir(project_dir) / "config.yml.bak"


def _put_raw(project_dir, mutate):
    """The project's config.yml, parsed, mutated, and re-serialized for a PUT body.

    PUT replaces the document verbatim, so a test states its case as a change to
    the *parsed* document and lets this render the bytes. Round-tripping through
    ``safe_dump`` drops the comments the file ships with -- which is itself worth
    having under test, because the resulting bytes differ everywhere while the
    protected view does not, and the gate must judge the view.

    Args:
        project_dir: The built project holding ``config.yml``.
        mutate: Callable applied to the parsed document in place. Return value
            ignored, so a mutation reads as the statement it is.

    Returns:
        The YAML text to send as ``{"raw": ...}``.
    """
    doc = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    mutate(doc)
    return yaml.safe_dump(doc, sort_keys=False)


def _flip_write_gate(doc):
    """Change one protected key: the kill-switch itself."""
    doc["control_system"]["writes_enabled"] = True


def _switch_pipeline(doc):
    """Change one unprotected key that still shapes the render."""
    doc[RENDER_SHAPING_KEY[0]][RENDER_SHAPING_KEY[1]] = "in_context"


def _add_protected_key(doc):
    """Add a protected key the file does not carry at all."""
    doc["agent_data"]["scratch_dir"] = "/tmp/elsewhere"


def _delete_protected_key(doc):
    """Delete one protected key, leaving its family in place."""
    del doc["control_system"]["writes_enabled"]


def _delete_protected_family(doc):
    """Delete a whole protected family in one stroke."""
    del doc["approval"]


def _collapse_subtree(doc):
    """Reshape a protected block into a scalar (dict -> scalar)."""
    doc["control_system"]["limits_checking"] = "off"


def _expand_scalar(doc):
    """Reshape a protected scalar into a block (scalar -> dict)."""
    doc["control_system"]["writes_enabled"] = {"enabled": True}


def _empty_the_document(doc):
    """Replace the file with nothing at all -- every protected key deleted at once."""
    doc.clear()


#: One case per class of change a whole-document replace can make to the
#: protected set. Each must come back 403 with the file byte-identical: the
#: single view comparison is what covers all seven, and a regression in any one
#: of them would show up as exactly one of these turning green-with-a-200.
PUT_MUTATIONS = [
    pytest.param(_add_protected_key, id="add-protected-key"),
    pytest.param(_delete_protected_key, id="delete-protected-key"),
    pytest.param(_flip_write_gate, id="change-protected-key"),
    pytest.param(_collapse_subtree, id="reshape-dict-to-scalar"),
    pytest.param(_expand_scalar, id="reshape-scalar-to-dict"),
    pytest.param(_delete_protected_family, id="delete-whole-family"),
    pytest.param(_empty_the_document, id="empty-document"),
]


#: One case per reason a key lands in the protected set, spelled the way a PATCH
#: body spells it. ``control_system`` is the ancestor rule: aiming at the parent
#: block rewrites every protected leaf beneath it, so the parent is refused too.
PROTECTED_CASES = [
    pytest.param("control_system.limits_checking.enabled", False, id="limits-gate"),
    pytest.param("control_system.writes_enabled", True, id="write-gate"),
    pytest.param("approval.mode", "disabled", id="approval-gate"),
    pytest.param("claude_code.permissions.deny", [], id="permission-surface"),
    pytest.param("agent_data.base_dir", "/tmp/elsewhere", id="agent-data-root"),
    pytest.param("control_system", {"type": "mock"}, id="ancestor-block"),
]


class TestPatchProtectedKeys:
    """``PATCH /api/config`` refuses the protected set before touching disk."""

    @pytest.mark.parametrize("key,value", PROTECTED_CASES)
    def test_patch_refuses_protected_key(self, client, built_project, key, value):
        before = (built_project / "config.yml").read_bytes()

        resp = client.patch("/api/config", json={"updates": {key: value}})

        assert resp.status_code == 403
        assert (built_project / "config.yml").read_bytes() == before

    def test_patch_protected_refusal_names_the_key_and_says_nothing_changed(
        self, client, built_project
    ):
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.limits_checking.enabled": False}},
        )

        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "control_system.limits_checking.enabled" in detail
        assert "config.yml is unchanged" in detail
        # The operator is pointed at the channel that *can* carry the change.
        assert "`config:` block" in detail

    def test_patch_protected_refusal_touches_nothing_on_disk(self, client, built_project):
        """No write, and no backup either -- a backup is a copy of a file it may not touch."""
        before = (built_project / "config.yml").read_bytes()
        backup = _backup_path(built_project)
        assert not backup.exists()

        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.limits_checking.enabled": False}},
        )

        assert resp.status_code == 403
        assert (built_project / "config.yml").read_bytes() == before
        assert not backup.exists()

    def test_patch_protected_key_refuses_the_whole_body(self, client, built_project):
        """One protected key refuses everything -- the cosmetic half does not land."""
        before = (built_project / "config.yml").read_bytes()

        resp = client.patch(
            "/api/config",
            json={
                "updates": {
                    COSMETIC_KEY: "sonnet",
                    "control_system.limits_checking.enabled": False,
                }
            },
        )

        assert resp.status_code == 403
        assert (built_project / "config.yml").read_bytes() == before
        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg["claude_code"]["default_model"] != "sonnet"
        assert not _backup_path(built_project).exists()

    def test_patch_unprotected_key_still_applies_beside_the_gate(self, client, built_project):
        """The gate refuses the protected set, not the panel."""
        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        assert resp.json()["fields_updated"] == 1
        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg["claude_code"]["default_model"] == "sonnet"

    def test_patch_protected_refusal_is_recorded_for_audit(self, client, audit_zone):
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.limits_checking.enabled": False}},
        )
        assert resp.status_code == 403

        records = _audit_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "http_config"
        assert records[0]["subject"] == "control_system.limits_checking.enabled"
        assert "target=config.yml" in records[0]["detail"]
        assert records[0]["reason"] == "protected_key"
        assert "`config:` block" in records[0]["detail"]

    def test_patch_protected_refusal_is_published_as_activity(self, client):
        """In-process, which is the success criterion: no panel token anywhere."""
        assert "OSPREY_PANEL_TOKEN" not in os.environ

        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.limits_checking.enabled": False}},
        )
        assert resp.status_code == 403

        frames = _refusal_frames(client)
        assert len(frames) == 1
        assert frames[0]["target"]["kind"] == "config"
        detail = frames[0]["target"]["detail"]
        assert detail.startswith("BLOCKED a protected config key")
        assert "config.yml: control_system.limits_checking.enabled" in detail
        assert "OSPREY_PANEL_TOKEN" not in os.environ

    def test_patch_protected_refusal_leaks_no_value(self, client, audit_zone):
        """Config values are secrets; a refusal reports the key, never the value."""
        sentinel = "qqzzSENTINELvalue77"

        resp = client.patch("/api/config", json={"updates": {"agent_data.base_dir": sentinel}})

        assert resp.status_code == 403
        assert sentinel not in resp.text
        assert sentinel not in json.dumps(_audit_records(audit_zone))
        assert sentinel not in json.dumps(_refusal_frames(client))


class TestConfigRouteRegen:
    def test_patch_writes_and_reports_nothing_to_regenerate(self, client, built_project):
        """No key the panel may still PATCH shapes the render, so regen is a no-op."""
        assert "mcp__controls__channel_write" in _deny(built_project)

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        assert resp.json()["regenerated"] == []
        # The kill-switch stays baked into the artifact the respawned agent reads.
        assert "mcp__controls__channel_write" in _deny(built_project)

    def test_patch_protected_write_gate_key_leaves_the_artifact_alone(self, client, built_project):
        """The kill-switch cannot be flipped through PATCH at all any more."""
        assert "mcp__controls__channel_write" in _deny(built_project)

        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )

        assert resp.status_code == 403
        assert "mcp__controls__channel_write" in _deny(built_project)

    def test_patch_fails_open_when_regen_raises(self, client, built_project, monkeypatch):
        """A regen error must never undo a config write that already succeeded."""

        def boom(self, project_dir):
            raise RuntimeError("regen exploded")

        monkeypatch.setattr(TemplateManager, "regen_if_drift", boom)
        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})
        assert resp.status_code == 200
        assert resp.json()["regenerated"] == []
        # The config write persisted despite the regen failure.
        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg["claude_code"]["default_model"] == "sonnet"

    def test_put_regenerates_artifacts(self, client, built_project):
        """PUT is the last surface whose writes can still move a rendered artifact.

        Driven with ``channel_finder.pipeline_mode`` rather than the write gate
        it used to use: the gate is protected now, so a PUT carrying it never
        reaches the render at all. This says the honest thing instead -- an
        unprotected key that shapes ``settings.json`` still re-renders it, and
        the kill-switch that the same file carries is untouched on the way past.
        """
        assert "mcp__controls__channel_write" in _deny(built_project)

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        body = resp.json()
        assert "regenerated" in body
        assert any("settings.json" in f for f in body["regenerated"])
        # Re-rendered, and the write gate came back out of config.yml unchanged.
        assert "mcp__controls__channel_write" in _deny(built_project)


class TestRenderZoneReadonlyRegen:
    """A read-only render zone takes the in-process regen off the success path.

    In the privilege-split container ``config.yml`` and ``.claude/`` are
    root-owned: the root entrypoint renders them and *then* drops to the
    non-root app user, so the server process may write neither. The admin image
    is the one where this panel's write to ``config.yml`` still lands -- and the
    regen that normally follows a successful write would then try to move a tree
    this process cannot touch.

    So the route skips it, and *says* it skipped it. That second half is the
    load-bearing one: ``regenerated: []`` on its own is indistinguishable from a
    render that simply had nothing to do, and an operator reading it that way
    would believe an edit to a render-shaping key had already taken effect. The
    ``detail`` is what tells them the truth -- the edit is in ``config.yml``, and
    the derived artifacts follow on the container restart.

    The flag is read with ``getattr(app.state, ..., False)``, so an app that
    never sets it -- a bare host, and the fixture app below -- keeps today's
    behaviour unchanged. Both halves are pinned here, because the value of the
    skip is entirely in it being conditional.
    """

    #: The exact string the panel renders its banner from. Spelled literally
    #: rather than imported from the route: it is a user-facing contract, and a
    #: test that imports the constant would follow a reworded detail silently.
    RESTART_DETAIL = "derived artifacts re-render on container restart"

    @staticmethod
    def _recording_regen(monkeypatch):
        """Replace the regen seam with a recorder; returns the call log.

        Patched on ``TemplateManager`` rather than on the route helper, which is
        where the route reaches it -- so "not invoked" means the render machinery
        was never entered, not merely that a wrapper returned early.
        """
        calls = []

        def record(self, project_dir):
            calls.append(project_dir)
            return ["settings.json"]

        monkeypatch.setattr(TemplateManager, "regen_if_drift", record)
        return calls

    def test_put_render_readonly_skips_regen_and_reports_restart(
        self, client, built_project, monkeypatch
    ):
        """PUT: the write lands, the regen never runs, the payload says why."""
        calls = self._recording_regen(monkeypatch)
        client.app.state.render_zone_readonly = True

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        body = resp.json()
        assert body["regenerated"] == []
        assert body["detail"] == self.RESTART_DETAIL
        assert calls == []
        # The config write itself is untouched by any of this -- it is only the
        # derived render that waits for the restart.
        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg[RENDER_SHAPING_KEY[0]][RENDER_SHAPING_KEY[1]] == "in_context"

    def test_patch_render_readonly_skips_regen_and_reports_restart(
        self, client, built_project, monkeypatch
    ):
        """PATCH: same skip, same detail, and it still reports what it wrote."""
        calls = self._recording_regen(monkeypatch)
        client.app.state.render_zone_readonly = True

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        body = resp.json()
        assert body["regenerated"] == []
        assert body["detail"] == self.RESTART_DETAIL
        assert body["fields_updated"] == 1
        assert calls == []
        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg["claude_code"]["default_model"] == "sonnet"

    def test_put_render_readonly_leaves_the_rendered_artifact_untouched(
        self, client, built_project
    ):
        """No seam patched: the real render is asked for and must not happen.

        ``_switch_pipeline`` is the one unprotected change that genuinely moves
        ``settings.json``; under the flag the file must come back byte-identical,
        which is the end-to-end statement the recorder tests cannot make.
        """
        settings_path = built_project / ".claude" / "settings.json"
        before = settings_path.read_bytes()
        client.app.state.render_zone_readonly = True

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        assert resp.json()["regenerated"] == []
        assert settings_path.read_bytes() == before

    def test_put_render_readonly_absent_keeps_todays_regen(
        self, client, built_project, monkeypatch
    ):
        """Bare host: the flag is unset, so the regen still runs and no detail is added."""
        calls = self._recording_regen(monkeypatch)
        assert getattr(client.app.state, "render_zone_readonly", False) is False

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        body = resp.json()
        assert body["regenerated"] == ["settings.json"]
        assert "detail" not in body
        assert len(calls) == 1

    def test_patch_render_readonly_absent_keeps_todays_regen(
        self, client, built_project, monkeypatch
    ):
        """The PATCH half of the same pin."""
        calls = self._recording_regen(monkeypatch)

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        body = resp.json()
        assert body["regenerated"] == ["settings.json"]
        assert "detail" not in body
        assert len(calls) == 1


class TestPutProtectedDocument:
    """``PUT /api/config`` diffs the protected view of the whole document.

    The Raw YAML view hands over bytes, not keys, so there is no field list to
    check. Every test below therefore builds a *document* mutation and asserts
    the diff catches it -- one per class of change a whole-file replace can make.
    """

    @pytest.mark.parametrize("mutate", PUT_MUTATIONS)
    def test_put_refuses_a_protected_document_change(self, client, built_project, mutate):
        before = (built_project / "config.yml").read_bytes()

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, mutate)})

        assert resp.status_code == 403
        assert (built_project / "config.yml").read_bytes() == before

    def test_put_protected_refusal_touches_nothing_on_disk(self, client, built_project):
        """No write, and no backup either -- a backup is a copy of a file it may not touch."""
        before = (built_project / "config.yml").read_bytes()
        backup = _backup_path(built_project)
        assert not backup.exists()

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _flip_write_gate)})

        assert resp.status_code == 403
        assert (built_project / "config.yml").read_bytes() == before
        assert not backup.exists()

    def test_put_protected_refusal_names_the_changed_key_and_says_nothing_changed(
        self, client, built_project
    ):
        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _flip_write_gate)})

        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "control_system.writes_enabled" in detail
        assert "config.yml is unchanged" in detail
        # The operator is pointed at the channel that *can* carry the change.
        assert "`config:` block" in detail

    def test_put_protected_refusal_is_recorded_for_audit(self, client, built_project, audit_zone):
        """One record per changed key, on the same surface name PATCH records."""
        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _flip_write_gate)})
        assert resp.status_code == 403

        records = _audit_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "http_config"
        assert records[0]["subject"] == "control_system.writes_enabled"
        assert "target=config.yml" in records[0]["detail"]
        assert records[0]["reason"] == "protected_key"
        assert "`config:` block" in records[0]["detail"]

    def test_put_protected_refusal_is_published_as_activity(self, client, built_project):
        """In-process, which is the success criterion: no panel token anywhere."""
        assert "OSPREY_PANEL_TOKEN" not in os.environ

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _flip_write_gate)})
        assert resp.status_code == 403

        frames = _refusal_frames(client)
        assert len(frames) == 1
        assert frames[0]["target"]["kind"] == "config"
        detail = frames[0]["target"]["detail"]
        assert detail.startswith("BLOCKED a protected config key")
        assert "config.yml: control_system.writes_enabled" in detail
        assert "OSPREY_PANEL_TOKEN" not in os.environ

    def test_put_protected_refusal_leaks_no_value(self, client, built_project, audit_zone):
        """Config values are secrets; a refusal reports the key, never the value."""
        sentinel = "qqzzSENTINELvalue77"

        def _plant(doc):
            doc["agent_data"]["base_dir"] = sentinel

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _plant)})

        assert resp.status_code == 403
        assert sentinel not in resp.text
        assert sentinel not in json.dumps(_audit_records(audit_zone))
        assert sentinel not in json.dumps(_refusal_frames(client))

    def test_put_unprotected_edit_passes_the_protected_gate_and_regenerates(
        self, client, built_project
    ):
        """The gate refuses the protected set, not the Raw YAML view."""
        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        doc = yaml.safe_load((built_project / "config.yml").read_text())
        assert doc["channel_finder"]["pipeline_mode"] == "in_context"
        assert any("settings.json" in f for f in resp.json()["regenerated"])

    def test_put_verbatim_rewrite_passes_the_protected_gate(self, client, built_project):
        """Different bytes, identical protected view: the diff is on keys, not text."""
        resp = client.put("/api/config", json={"raw": _put_raw(built_project, lambda doc: None)})

        assert resp.status_code == 200

    def test_put_hooks_debug_is_exempt_from_the_protected_gate(self, client, built_project):
        """The exemption is inherited: PUT consults the same one choke point PATCH does.

        ``hooks.*`` covers the hook *wiring*, and over-matched this one key --
        the shipped Hook Debug switch. It is exempt in ``is_protected_key_path``,
        so it is exempt here too, without this surface knowing the exemption
        exists.
        """
        assert yaml.safe_load((built_project / "config.yml").read_text())["hooks"]["debug"] is True

        def _toggle(doc):
            doc["hooks"]["debug"] = False

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _toggle)})

        assert resp.status_code == 200
        assert client.get("/api/hooks/debug-status").json()["enabled"] is False

    def test_put_reshaping_the_hooks_block_is_still_protected(self, client, built_project):
        """Exempting ``hooks.debug`` does not exempt ``hooks`` -- replacing the block is wiring."""

        def _flatten(doc):
            doc["hooks"] = "off"

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _flatten)})

        assert resp.status_code == 403

    def test_put_dotted_raw_key_cannot_mask_a_protected_change(self, client, built_project):
        """The adversarial case the segment-tuple view exists for.

        A top-level key literally named ``"control_system.writes_enabled"`` is
        one raw segment that happens to contain dots. It is *not* the protected
        key -- nothing reads it, config reads are nested -- but a flattened
        *dotted* view renders it to the same string as the real one. A document
        that drops the real nested key while adding this decoy would compare
        equal under that view, and the write gate would leave the file with the
        gate itself removed. Tuples cannot collide, so this is a 403.
        """

        def _decoy(doc):
            del doc["control_system"]["writes_enabled"]
            doc["control_system.writes_enabled"] = False

        before = (built_project / "config.yml").read_bytes()
        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _decoy)})

        assert resp.status_code == 403
        assert "control_system.writes_enabled" in resp.json()["detail"]
        assert (built_project / "config.yml").read_bytes() == before

    def test_put_protected_family_member_with_a_dotted_name_is_refused(self, client, built_project):
        """A dotted *name* inside a wildcard family is still one segment, still protected."""

        def _add_server(doc):
            doc.setdefault("claude_code", {}).setdefault("servers", {})["evil.srv"] = {
                "command": "/bin/sh"
            }

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _add_server)})

        assert resp.status_code == 403
        assert "claude_code.servers.evil.srv.command" in resp.json()["detail"]

    def test_put_protected_refusal_caps_the_message_but_not_the_audit(
        self, client, built_project, audit_zone
    ):
        """The cap trims what an operator reads, never what the audit keeps.

        Emptying the document changes every protected key the file carries at
        once -- far more than a refusal can name and stay readable. The message
        names the first ten and counts the rest; the ``http_config`` ledger gets
        all of them, because the trail is what a later investigation reads and a
        summarized trail is a missing one.
        """
        current = protected_view(
            "config.yml", yaml.safe_load((built_project / "config.yml").read_text())
        )
        assert len(current) > 10, "precondition: enough protected keys for the cap to fire"

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _empty_the_document)})

        assert resp.status_code == 403
        detail = resp.json()["detail"]
        head = detail.split(" are protected keys", 1)[0]
        assert re.findall(r"`([^`]+)`", head) == sorted(".".join(k) for k in current)[:10]
        assert f"and {len(current) - 10} more" in head

        records = _audit_records(audit_zone)
        assert len(records) == len(current)
        assert {r["subject"] for r in records} == {".".join(k) for k in current}
        assert all(r["surface"] == "http_config" for r in records)

    def test_put_self_referential_yaml_is_refused_before_the_protected_check(
        self, client, built_project
    ):
        """A YAML anchor that contains itself parses, then walks forever.

        The gate already fails closed on it -- the walk raises before a byte is
        written -- so what is pinned here is that the refusal is *legible*: a 422
        naming the document rather than the bare 500 an escaping RecursionError
        would produce, which is the one shape an operator could read as a gate
        that broke rather than one that held.
        """
        before = (built_project / "config.yml").read_bytes()

        resp = client.put("/api/config", json={"raw": "loop: &loop\n  self: *loop\n"})

        assert resp.status_code == 422
        assert "Invalid YAML" in resp.json()["detail"]
        assert (built_project / "config.yml").read_bytes() == before
        assert not _backup_path(built_project).exists()

    def test_put_refuses_when_the_protected_baseline_cannot_be_parsed(self, client, built_project):
        """Fail closed: no readable baseline means no check, and no check means no write."""
        (built_project / "config.yml").write_text("control_system: [unterminated\n")
        before = (built_project / "config.yml").read_bytes()

        resp = client.put("/api/config", json={"raw": "project_name: fine\n"})

        assert resp.status_code == 500
        assert (built_project / "config.yml").read_bytes() == before
        assert not _backup_path(built_project).exists()


class TestPutProtectedDocumentDiff:
    """Unit-level pins on the diff itself, for states the HTTP surface cannot reach.

    Every transition here is one a PUT would have to *arrive* at rather than
    create, so it is asserted against :func:`_changed_protected_keys` directly.
    """

    def test_put_protected_exact_depth_key_reshaped_from_a_real_value_is_caught(self):
        """The reachable half of the mapping-at-pattern-depth edge."""
        before = {"simulation": {"state_dir": "var/simulation"}}
        after = {"simulation": {"state_dir": {"evil": "data/"}}}

        assert _changed_protected_keys(before, after) == ["simulation.state_dir"]

    def test_put_protected_exact_depth_block_added_from_absent_is_the_known_residual(self):
        """The documented residual, pinned so it is a decision and not a surprise.

        ``simulation.state_dir`` is matched by a pattern with no trailing
        wildcard, so when the node at that depth is a *mapping* the flatten
        descends past it and the children come back unprotected. Adding one from
        nothing therefore compares equal.

        It is inert, and the route's own note spells out why for each of the two
        keys in this shape, because the mechanism is not the same for both. This
        one is the tidy case: ``simulation.state_dir`` is read through
        ``dotted_config_str``, which answers ``None`` for anything that is not a
        non-empty string, so a block there reads as unset and the runtime takes
        the default the absent key already gave. (The channel-finder
        ``...feedback.store_path`` is *not* read that way -- it is read raw,
        raises inside the initialization, and leaves the feedback store disabled
        rather than defaulted. Also inert for this gate, by a different route.)

        What makes the residual a dead end rather than a first step is that it
        does not ratchet: putting a real value back is caught, which is the
        second half of this test.
        """
        planted = {"simulation": {"state_dir": {"evil": "data/"}}}

        assert _changed_protected_keys({}, planted) == []
        assert _changed_protected_keys(planted, {"simulation": {"state_dir": "/evil"}}) == [
            "simulation.state_dir"
        ]

    def test_put_protected_families_are_descent_safe_or_known_inert(self):
        """Tripwire: a fourth exact-depth pattern must re-open the judgement above.

        A pattern is *descent-safe* when a hypothetical child of it is protected
        too -- true for every ``family.*`` pattern, and for ``artifacts.hooks``
        because ``artifacts.*`` sits beside it. The ones that are not are the
        ones whose subtree the flatten can lose, and both of today's name a path
        value that cannot be made to point anywhere by planting a block there --
        one because its reader treats a block as unset, the other because its
        reader chokes on it and leaves the store off. A new pattern gets neither
        guarantee for free, so if this fails the PUT gate's note needs
        re-deciding, not extending.
        """
        probe = "\x00-not-a-real-key"
        known_inert = {
            "simulation.state_dir",
            "services.channel_finder.pipelines.hierarchical.feedback.store_path",
        }

        leaky = {
            pattern
            for pattern in PROTECTED_CONFIG_KEYS["config.yml"]
            if not is_protected_key_path("config.yml", (*pattern.split("."), probe))
        }

        assert leaky == known_inert

    def test_put_protected_diff_is_exactly_view_equality(self):
        """The gate is specified as ``protected_view(new) == protected_view(old)``.

        This function returns the *names* for the refusal message; the pin is
        that it never disagrees with that equality about whether to refuse.
        """
        cases = [
            ({"control_system": {"writes_enabled": False}}, {}),
            ({}, {"control_system": {"writes_enabled": False}}),
            ({"approval": {"enabled": True}}, {"approval": {"enabled": False}}),
            ({"project_name": "a"}, {"project_name": "b"}),
            ({"hooks": {"debug": True}}, {"hooks": {"debug": False}}),
        ]
        for before, after in cases:
            equal = protected_view("config.yml", before) == protected_view("config.yml", after)
            assert bool(_changed_protected_keys(before, after)) is not equal


class TestConfigBackupLocation:
    """Both write surfaces back ``config.yml`` up into the agent-data state zone.

    Not beside the file any more. ``config.yml`` lives in the render, and the
    render becomes root-owned when the container split lands: creating a *new*
    file next to it needs write permission on the render directory, which the
    admin image will not have. Under the old sibling scheme the backup -- which
    runs before either handler writes anything -- would have raised
    ``PermissionError`` and turned every config save in that image into a 500.
    The state zone stays writable across the split, so the backup goes there.
    """

    def test_put_writes_its_backup_into_the_state_zone(self, client, built_project):
        before = (built_project / "config.yml").read_bytes()
        assert not _backup_path(built_project).exists()

        resp = client.put("/api/config", json={"raw": _put_raw(built_project, _switch_pipeline)})

        assert resp.status_code == 200
        backup = _backup_path(built_project)
        assert backup.is_file()
        # It is the *pre-write* config, which is the only thing a backup is for.
        assert backup.read_bytes() == before

    def test_patch_writes_its_backup_into_the_state_zone(self, client, built_project):
        before = (built_project / "config.yml").read_bytes()
        assert not _backup_path(built_project).exists()

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        backup = _backup_path(built_project)
        assert backup.is_file()
        assert backup.read_bytes() == before

    @pytest.mark.parametrize(
        "save",
        [
            pytest.param(
                lambda client, project: client.patch(
                    "/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}}
                ),
                id="patch",
            ),
            pytest.param(
                lambda client, project: client.put(
                    "/api/config", json={"raw": _put_raw(project, _switch_pipeline)}
                ),
                id="put",
            ),
        ],
    )
    def test_config_backup_is_never_written_beside_the_render(self, client, built_project, save):
        """The whole point: no backup file is created next to config.yml.

        Stated as "no ``.bak`` lands directly in the project directory" rather
        than "the directory is unchanged", because in the default flat layout the
        state zone is itself *inside* the project (``var/agent_data``) and gets
        created on first use. That is fine and is not the thing the split breaks:
        what breaks is creating a new file in the *render*, beside the config.
        """
        assert save(client, built_project).status_code == 200

        assert not (built_project / "config.yml.bak").exists()
        assert [p.name for p in built_project.iterdir() if p.suffix == ".bak"] == []

    def test_patch_backup_follows_a_relocated_agent_data_root(
        self, client, built_project, tmp_path
    ):
        """The zone is read from config, never assumed.

        This is the case a hard-coded ``var/agent_data`` would fail: the root is
        moved somewhere no default would guess, and the backup has to go with it.
        Relocated on disk rather than through the API because ``agent_data.*`` is
        in the protected set -- neither handler may repoint its own state zone,
        which is the other half of this being safe.
        """
        relocated = tmp_path / "elsewhere" / "state"
        config_update_fields(built_project / "config.yml", {"agent_data.base_dir": str(relocated)})
        before = (built_project / "config.yml").read_bytes()

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        assert (relocated / "config-backups" / "config.yml.bak").read_bytes() == before
        assert not (built_project / "var" / "agent_data" / "config-backups").exists()

    def test_config_backup_overwrites_a_single_slot(self, client, built_project):
        """Retention is unchanged by the move: one slot, overwritten, per filename."""
        client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})
        after_first = (built_project / "config.yml").read_bytes()

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "opus"}})

        assert resp.status_code == 200
        assert list(_backup_dir(built_project).iterdir()) == [_backup_path(built_project)]
        # The slot holds the most recent pre-write config, not the oldest.
        assert _backup_path(built_project).read_bytes() == after_first

    def test_config_backup_name_keeps_the_source_stem(self, client, built_project):
        """Two configs backed up into one zone must not land on each other.

        Both configs live in the same project, because that is what puts them in
        one zone: the anchor is the repo each config sits in, so a second file
        somewhere else would simply get its own zone and prove nothing.
        """
        other = built_project / "other.yml"
        other.write_text((built_project / "config.yml").read_text(), encoding="utf-8")
        client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})
        client.app.state.config_path = other

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "haiku"}})

        assert resp.status_code == 200
        assert {p.name for p in _backup_dir(built_project).iterdir()} == {
            "config.yml.bak",
            "other.yml.bak",
        }

    def test_config_backup_ignores_project_cwd_and_anchors_on_the_repo_root(
        self, client, built_project, tmp_path
    ):
        """The container defect, at the route: ``project_cwd`` is the RENDER.

        In the shipped image ``app.state.project_cwd`` is
        ``/app/<project>/build`` while ``agent_data.base_dir`` is rendered
        relative to ``/app/<project>``, which is where the Dockerfile creates and
        chowns ``var/agent_data/config-backups``. Anchoring the backup on
        ``project_cwd`` made the route ``mkdir`` ``<render>/var`` inside the
        root-owned render zone, so every admin ``PATCH /api/config`` failed with
        ``PermissionError`` before a byte reached ``config.yml``.

        Reproduced here by moving the config into a ``build/`` zone the way a
        render does, leaving ``project_cwd`` pointed at it, and asserting the
        backup climbs out to the repo root anyway.
        """
        render = built_project / "build"
        render.mkdir()
        moved = render / "config.yml"
        moved.write_bytes((built_project / "config.yml").read_bytes())
        before = moved.read_bytes()
        client.app.state.config_path = moved
        client.app.state.project_cwd = str(render)

        resp = client.patch("/api/config", json={"updates": {COSMETIC_KEY: "sonnet"}})

        assert resp.status_code == 200
        assert _backup_path(built_project).read_bytes() == before
        assert not (render / "var").exists()
