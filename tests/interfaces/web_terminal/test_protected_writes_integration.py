"""Every protected-write refusal leaves one durable record and one visible frame.

The individual gates are each pinned by their own suite. What is *not* pinned
anywhere else is the property that spans them: whichever writer an agent
reaches for -- the scaffold gallery, the Claude-setup panel, the Config panel's
two write paths, the ``setup_patch`` MCP tool -- a refused write produces the
same two after-effects, in the same shapes, so an operator has one place to
look and one query to run.

Three things are asserted across the whole set, and each is a proposal
requirement rather than an implementation detail:

* **FR2 -- the refusal is durable and singular.** One line per refused attempt
  on that writer's ledger (``var/audit/<identity>/<surface>.jsonl``), carrying
  the same envelope fields on every surface. Not two lines (a retry counted
  twice would misreport how hard something pushed), and not zero (a refusal
  nobody can find afterwards is an error message, not an audit trail).
* **FR3 -- the refusal is visible while it happens.** The attempt reaches
  ``GET /api/agent-activity/recent``, which is what a browser reads on connect.
  Read back through the route rather than off ``app.state``: the ring is only
  load-bearing insofar as the HTTP surface serves it.
* **The message tells the operator the truth.** Every refusal names the channel
  that *does* own the target, and states plainly that nothing happened. The
  wording is per-surface on purpose -- "NOTHING WAS CLAIMED" and "nothing was
  written" are different facts -- so each case brings its own outcome pattern
  instead of one string being forced across five writers.

**No panel token anywhere.** The HTTP surfaces publish in-process, through
``record_activity``, because the handler already holds the request. That is the
whole reason the web-terminal refusals need no ``OSPREY_PANEL_TOKEN``: an
assertion on the server process environment appears in every HTTP flow here,
because the day one of these surfaces starts posting back into itself over HTTP
is the day it needs a credential to do it, and that regression is invisible
from the response alone.

``setup_patch`` is the one surface that does *not* publish in-process: it runs
in the MCP child, which reaches the panel over HTTP with the panel token. It is
tested here at the seam its own suite uses -- the ``notify_agent_activity_async``
call -- rather than by standing up a second server, and it is included so the
cross-surface shape assertions cover it too.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import writer
from osprey.audit.envelope import (
    POSTURE_SOURCE_APP,
    POSTURE_SOURCE_SPAWN,
    AuditEnvelope,
)
from osprey.audit.protected import POSTURE_SOURCE_ENV_VAR, PROTECTED_SURFACES
from osprey.cli.profile_conventions import RESERVED_PATH_CHANNELS, is_reserved_write
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV
from osprey.interfaces.web_terminal.app import create_app
from osprey.services.build_artifacts.ownership import update_config_add_user_owned
from osprey.utils.identity import acting_identity

SETUP_MOD = "osprey.mcp_server.workspace.tools.setup"

#: Fields every protected-write record carries, whichever writer produced it.
#: The point of the shared writer is that one ``jq`` expression reads a gallery
#: refusal and a ``setup_patch`` refusal alike, which only holds while no
#: surface omits a field or invents one of its own. Derived from the envelope
#: rather than restated, so a field added there arrives here as a failure to
#: look at rather than as silence.
AUDIT_RECORD_FIELDS = {"ts", *AuditEnvelope.REQUIRED_FIELDS, "detail"}

#: The one field that is legitimately ``null`` here: these flows run outside a
#: Web Terminal session, so nothing stamped a posture-store key. Exempt from
#: the *non-empty string* loop only -- ``session`` is checked on its own terms
#: below, because "may be null" is not "may be anything".
NULLABLE_RECORD_FIELDS = {"session"}

#: The phrase the two *config-key* surfaces share in the activity feed, so one
#: search over the operator's feed finds a refused config write whether the
#: agent went through the Config panel or through ``setup_patch``.
CONFIG_FEED_PHRASE = "BLOCKED a protected config key"

#: An artifact the gallery lists and the protected set closes: instruction text
#: under ``.claude/rules/``, which the agent may read but may not author.
RESERVED_FRAMEWORK_ARTIFACT = "rules/safety"

#: A *custom* reserved-path name -- deliberately absent from the build-artifact
#: catalog, so ``unoverride`` takes its custom branch (the one that deletes a
#: file) rather than the framework restore branch.
RESERVED_CUSTOM_ARTIFACT = "rules/agent-authored"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _baked_project(tmp_path_factory) -> Path:
    """One real render, shared by the module; tests get a private copy.

    Rendering is expensive enough that one per test dominates the suite, and
    nothing here needs a pristine tree -- every case under test is refused
    before it writes, and the per-test copy carries the ownership edits.
    """
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="protected-writes",
        output_dir=tmp_path_factory.mktemp("protected-writes-bake"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    manager.regenerate_claude_code(project_dir)
    # Both reserved artifacts are recorded as user-owned so the save and the
    # release paths reach their protected-set check at all: both sit behind an
    # ownership test, and an unowned name is refused as "not user-owned" long
    # before the gate that is actually under test here.
    for name in (RESERVED_FRAMEWORK_ARTIFACT, RESERVED_CUSTOM_ARTIFACT):
        update_config_add_user_owned(project_dir, name)
    return project_dir


@pytest.fixture
def project_dir(_baked_project, tmp_path) -> Path:
    """A private copy of the render, with its absolute self-references re-anchored.

    A render records the path it was built at (``project_root`` in config.yml,
    the manifest's build args). Copying the bytes without rewriting those would
    point every resolution back at the shared bake and let one test's writes
    reach another's tree.
    """
    import shutil

    copy = tmp_path / "project"
    shutil.copytree(_baked_project, copy, symlinks=True)
    old, new = str(_baked_project).encode(), str(copy).encode()
    for path in copy.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if old in data:
            path.write_bytes(data.replace(old, new))
    return copy


@pytest.fixture
def audit_zone(tmp_path, monkeypatch) -> Path:
    """Redirect the audit zone. ``writer.audit_dir`` is the ledger's one seam."""
    zone = tmp_path / "audit-zone" / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    return zone


@pytest.fixture
def client(project_dir, tmp_path, audit_zone):
    """The real web-terminal app over the real render, lifespan and all.

    ``create_app`` rather than a bare router, because the activity ring, the
    scaffold conflict handlers and the ``app.state`` wiring the refusal paths
    read are all things the real app assembles -- a hand-built app would let a
    surface publish into a ring the shipped app does not carry.
    """
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(watch_dir)},
    ):
        app = create_app(
            config_path=project_dir / "config.yml",
            shell_command="echo",
            project_dir=str(project_dir),
        )
        with TestClient(app) as test_client:
            yield test_client


# ── Readers ──────────────────────────────────────────────────────────


def audit_records(zone: Path, surface: str | None = None) -> list[dict]:
    """Protected-write records currently on disk, oldest first *within a surface*.

    Each writer files into its own ledger now, so "every record" means reading
    the five of them. Ordering holds inside one file, which is where the
    append-per-attempt assertions live; across files the records are grouped by
    surface, because a second-resolution timestamp cannot order two refusals
    filed in the same second by different writers.
    """
    identity = zone / acting_identity()
    wanted = (surface,) if surface is not None else PROTECTED_SURFACES
    records: list[dict] = []
    for name in wanted:
        path = identity / f"{name}.jsonl"
        if not path.is_file():
            continue
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def recent_activity(client: TestClient) -> list[dict]:
    """The activity history as a *browser* sees it -- through the route, not the ring."""
    resp = client.get("/api/agent-activity/recent")
    assert resp.status_code == 200, resp.text
    return resp.json()["events"]


@contextmanager
def no_panel_token():
    """Assert the server process carries no panel token, before and after.

    The in-process publication path is the reason these surfaces need none. A
    surface that started posting back into the panel over HTTP would need one,
    and would fail here rather than silently acquiring a credential
    requirement the single-user deployment cannot satisfy.
    """
    assert PANEL_TOKEN_ENV not in os.environ, (
        f"{PANEL_TOKEN_ENV} is set before the flow -- this suite cannot tell "
        "an in-process publication from a token-authenticated one"
    )
    yield
    assert PANEL_TOKEN_ENV not in os.environ, (
        f"{PANEL_TOKEN_ENV} appeared during the flow: an HTTP surface published "
        "its refusal out-of-process instead of stamping it in-process"
    )


# ── The surfaces ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Refusal:
    """One writer's refusal, described the way the assertions need it.

    Attributes:
        surface: ``surface`` field the audit record must carry.
        tool: ``tool`` the activity frame must be filed under.
        kind: ``target.kind`` the activity frame must carry.
        key_or_path: What inside the target was protected, as the record spells it.
        target_file: The file the write was aimed at, as the record spells it.
        reason: The record's machine-ish reason.
        outcome: Pattern the 403 body must match -- each surface states its own
            outcome ("NOTHING WAS CLAIMED", "nothing was written"), and forcing
            one string across all of them would test the sameness rather than
            the honesty.
        frame_names: What the *activity* frame must carry -- ``"channel"`` for
            the two path surfaces, whose frames quote the channel outright, or
            ``"target"`` for the config surfaces, whose frames name the key and
            leave the channel to the 403 body. Stated per case rather than
            accepted as an either/or, because an either/or would go on passing
            if a surface quietly stopped carrying the half it owes.
        drive: Sends the request. Returns the response.
    """

    surface: str
    tool: str
    kind: str
    key_or_path: str
    target_file: str
    reason: str
    outcome: str
    frame_names: str
    drive: Callable[[TestClient], object]

    @property
    def channel(self) -> str:
        """The channel that owns the target, derived rather than restated.

        Read out of the same tables the writers read, so a re-worded channel
        moves the expectation with it instead of breaking every case here --
        what is under test is that the message, the record and the feed agree,
        not what the sentence happens to say this month.
        """
        return is_reserved_write(self.key_or_path) or RESERVED_PATH_CHANNELS[self.target_file]


def _put_protected_config(client: TestClient) -> object:
    """A whole-document PUT that changes exactly one protected key.

    Serialised from the document on disk so everything *except* the write gate
    is byte-for-byte what the file already holds -- the refusal then has one
    cause, and the single-record assertion means what it says.
    """
    resp = client.get("/api/config")
    doc = yaml.safe_load(resp.json()["raw"]) or {}
    control_system = doc.setdefault("control_system", {})
    # Flipped relative to what is on disk rather than set to a constant: a
    # constant that happened to match the rendered value would be no change at
    # all, and the gate would let it through for the right reason while this
    # case silently stopped testing anything.
    control_system["writes_enabled"] = not control_system.get("writes_enabled", False)
    return client.put("/api/config", json={"raw": yaml.safe_dump(doc, sort_keys=False)})


#: One case per writer entry point the protected set closes. The scaffold
#: gallery contributes six because it has six ways in, and a gate on five of
#: them is not a gate.
REFUSALS = [
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="create_artifact",
            kind="artifact",
            key_or_path=".claude/rules/agent-authored.md",
            target_file=".claude/rules/agent-authored.md",
            reason="reserved path",
            outcome="NOTHING WAS CREATED",
            frame_names="channel",
            drive=lambda c: c.post(
                "/api/scaffold/create",
                json={"category": "rules", "name": "agent-authored", "content": "# mine\n"},
            ),
        ),
        id="gallery-create",
    ),
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="claim",
            kind="artifact",
            key_or_path=".claude/rules/safety.md",
            target_file=".claude/rules/safety.md",
            reason="reserved path",
            outcome="NOTHING WAS CLAIMED",
            frame_names="channel",
            drive=lambda c: c.post(f"/api/scaffold/{RESERVED_FRAMEWORK_ARTIFACT}/claim"),
        ),
        id="gallery-claim",
    ),
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="save_override",
            kind="artifact",
            key_or_path=".claude/rules/safety.md",
            target_file=".claude/rules/safety.md",
            reason="reserved path",
            outcome="NOTHING WAS WRITTEN",
            frame_names="channel",
            drive=lambda c: c.put(
                f"/api/scaffold/{RESERVED_FRAMEWORK_ARTIFACT}/override",
                json={"content": "# rewritten by the agent\n"},
            ),
        ),
        id="gallery-save-override",
    ),
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="unoverride",
            kind="artifact",
            key_or_path=".claude/rules/agent-authored.md",
            target_file=".claude/rules/agent-authored.md",
            reason="reserved path",
            outcome="NOTHING WAS DELETED",
            frame_names="channel",
            drive=lambda c: c.delete(
                f"/api/scaffold/{RESERVED_CUSTOM_ARTIFACT}/override?delete_file=true"
            ),
        ),
        id="gallery-unoverride-delete-file",
    ),
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="register_untracked",
            kind="artifact",
            key_or_path=".claude/skills/agent-authored.md",
            target_file=".claude/skills/agent-authored.md",
            reason="reserved path",
            outcome="NOTHING WAS REGISTERED",
            frame_names="channel",
            drive=lambda c: c.post(
                "/api/scaffold/untracked/register", json={"name": "skills/agent-authored"}
            ),
        ),
        id="gallery-register-untracked",
    ),
    pytest.param(
        Refusal(
            surface="scaffold_gallery",
            tool="delete_untracked",
            kind="artifact",
            key_or_path=".claude/skills/agent-authored.md",
            target_file=".claude/skills/agent-authored.md",
            reason="reserved path",
            outcome="NOTHING WAS DELETED",
            frame_names="channel",
            drive=lambda c: c.delete("/api/scaffold/untracked/skills/agent-authored"),
        ),
        id="gallery-delete-untracked",
    ),
    pytest.param(
        Refusal(
            surface="claude_setup",
            tool="claude_setup_refused",
            kind="config",
            key_or_path=".claude/settings.json",
            target_file=".claude/settings.json",
            reason="reserved path",
            outcome="nothing was written",
            frame_names="channel",
            drive=lambda c: c.put(
                "/api/claude-setup",
                json={"path": ".claude/settings.json", "content": "{}"},
            ),
        ),
        id="claude-setup-write",
    ),
    pytest.param(
        Refusal(
            surface="claude_setup",
            tool="claude_setup_refused",
            kind="config",
            key_or_path=".claude/skills/self-authored/SKILL.md",
            target_file=".claude/skills/self-authored/SKILL.md",
            reason="reserved path",
            outcome="nothing was written",
            frame_names="channel",
            drive=lambda c: c.post(
                "/api/claude-setup",
                json={"path": ".claude/skills/self-authored/SKILL.md", "content": "# mine\n"},
            ),
        ),
        id="claude-setup-create",
    ),
    pytest.param(
        Refusal(
            surface="http_config",
            tool="config_patch_refused",
            kind="config",
            key_or_path="control_system.writes_enabled",
            target_file="config.yml",
            reason="protected_key",
            outcome="config.yml is unchanged",
            frame_names="target",
            drive=lambda c: c.patch(
                "/api/config", json={"updates": {"control_system.writes_enabled": True}}
            ),
        ),
        id="http-config-patch",
    ),
    pytest.param(
        Refusal(
            surface="http_config",
            tool="config_patch_refused",
            kind="config",
            key_or_path="control_system.writes_enabled",
            target_file="config.yml",
            reason="protected_key",
            outcome="config.yml is unchanged",
            frame_names="target",
            drive=_put_protected_config,
        ),
        id="http-config-put",
    ),
]


class TestEveryHttpSurfaceRefusesAudibly:
    """FR2 and FR3, asserted once per writer the browser can reach."""

    @pytest.mark.parametrize("case", REFUSALS)
    def test_refusal_is_a_403(self, client, case):
        with no_panel_token():
            resp = case.drive(client)
        assert resp.status_code == 403, f"{case.surface}/{case.tool}: {resp.text}"

    @pytest.mark.parametrize("case", REFUSALS)
    def test_refusal_appends_exactly_one_audit_record(self, client, audit_zone, case):
        """One line per attempt -- the log counts attempts, so neither zero nor two."""
        assert audit_records(audit_zone) == []

        with no_panel_token():
            assert case.drive(client).status_code == 403

        records = audit_records(audit_zone)
        assert len(records) == 1, f"{case.surface}/{case.tool} wrote {len(records)} records"
        record = records[0]
        assert record["surface"] == case.surface
        assert record["subject"] == case.key_or_path
        assert f"target={case.target_file}" in record["detail"]
        assert case.channel in record["detail"], (
            "the log and the refusal message must name the same way in"
        )
        assert record["decision"] == "refused"
        assert record["reason"] == case.reason

    @pytest.mark.parametrize("case", REFUSALS)
    def test_a_retried_refusal_is_a_second_line(self, client, audit_zone, case):
        """The log appends. A pushed-twice attempt must not read as pushed once."""
        with no_panel_token():
            assert case.drive(client).status_code == 403
            assert case.drive(client).status_code == 403

        records = audit_records(audit_zone)
        assert len(records) == 2, f"{case.surface}/{case.tool} collapsed a retry"
        assert records[0]["subject"] == records[1]["subject"] == case.key_or_path

    @pytest.mark.parametrize("case", REFUSALS)
    def test_refusal_reaches_the_activity_route(self, client, case):
        """FR3 read the way a browser reads it: back out through the GET route."""
        assert recent_activity(client) == []

        with no_panel_token():
            assert case.drive(client).status_code == 403

        events = recent_activity(client)
        assert len(events) == 1, f"{case.surface}/{case.tool} published {len(events)} frames"
        event = events[0]
        assert event["type"] == "agent_activity"
        assert event["tool"] == case.tool
        assert event["target"]["kind"] == case.kind
        assert isinstance(event["ts"], float)

    @pytest.mark.parametrize("case", REFUSALS)
    def test_activity_detail_identifies_what_was_refused(self, client, case):
        """A frame that says only "refused" sends the operator nowhere.

        The path surfaces quote the owning channel into the frame; the config
        surfaces quote the key and leave the channel to the 403 body, because
        a config channel sentence is longer than the frame is meant to be.
        Which of the two a surface owes is fixed per case, so neither can drop
        to the other and still pass.
        """
        with no_panel_token():
            assert case.drive(client).status_code == 403

        detail = recent_activity(client)[0]["target"]["detail"]
        expected = case.channel if case.frame_names == "channel" else case.key_or_path
        assert expected in detail, (
            f"{case.surface}/{case.tool} published a frame that does not name "
            f"the {case.frame_names}: {detail!r}"
        )

    @pytest.mark.parametrize("case", REFUSALS)
    def test_refusal_message_names_the_channel_and_says_nothing_happened(self, client, case):
        """The two halves of an honest refusal, asserted per surface.

        Naming the channel is what turns a refusal into a route forward; saying
        outright that nothing happened is what stops a reader assuming the
        write half-landed. Each surface phrases the second half in its own
        terms, so the pattern travels with the case.
        """
        with no_panel_token():
            resp = case.drive(client)

        detail = resp.json()["detail"]
        assert case.channel in detail, (
            f"{case.surface}/{case.tool} refused without naming the owning channel: {detail!r}"
        )
        assert re.search(case.outcome, detail, re.IGNORECASE), (
            f"{case.surface}/{case.tool} did not state that nothing happened: {detail!r}"
        )


class TestNothingMoved:
    """A refusal that left something behind would satisfy every assertion above."""

    @pytest.mark.parametrize("case", REFUSALS)
    def test_the_protected_target_is_untouched(self, client, project_dir, case):
        target = project_dir / case.target_file
        before = target.read_bytes() if target.is_file() else None

        with no_panel_token():
            assert case.drive(client).status_code == 403

        after = target.read_bytes() if target.is_file() else None
        assert after == before, f"{case.surface}/{case.tool} moved {case.target_file}"


class TestCrossSurfaceShape:
    """What the surfaces owe each other, rather than what each owes on its own."""

    def test_every_record_carries_the_same_fields(self, client, audit_zone):
        """One query over the log has to span every surface, so no field is optional."""
        for case in (p.values[0] for p in REFUSALS):
            with no_panel_token():
                assert case.drive(client).status_code == 403

        records = audit_records(audit_zone)
        assert len(records) == len(REFUSALS)
        for record in records:
            assert set(record) == AUDIT_RECORD_FIELDS, (
                f"{record['surface']} recorded {sorted(record)}"
            )
            for field in set(record) - NULLABLE_RECORD_FIELDS:
                assert isinstance(record[field], str) and record[field], (
                    f"{record['surface']} left {field} empty"
                )
            session = record["session"]
            assert session is None or (isinstance(session, str) and session), (
                f"{record['surface']} recorded session={session!r}: the field is "
                "nullable, not unchecked"
            )

    def test_every_http_surface_stamps_the_app_posture_source(
        self, client, audit_zone, monkeypatch
    ):
        """A web request belongs to no session, whatever the server inherited.

        ``HttpAuditMiddleware`` files ``app`` for the very request these
        refusals answer, so a surface reading the environment ladder instead
        would leave two records of one request disagreeing about where its
        posture came from -- one of them calling a web request a bare CLI
        process.

        The ladder is pointed at a *different* answer first: with
        ``OSPREY_POSTURE_SOURCE`` set, a call site that dropped its own stamp
        would quietly inherit ``spawn`` and this test would say so.
        """
        monkeypatch.setenv(POSTURE_SOURCE_ENV_VAR, POSTURE_SOURCE_SPAWN)

        for case in (p.values[0] for p in REFUSALS):
            with no_panel_token():
                assert case.drive(client).status_code == 403

        records = audit_records(audit_zone)
        assert len(records) == len(REFUSALS)
        for record in records:
            assert record["posture_source"] == POSTURE_SOURCE_APP, (
                f"{record['surface']} filed posture_source={record['posture_source']!r} "
                "for a web request"
            )

    def test_the_recorded_surfaces_are_the_documented_ones(self, client, audit_zone):
        """The ``surface`` field is what an operator filters on; it is a closed set."""
        for case in (p.values[0] for p in REFUSALS):
            with no_panel_token():
                assert case.drive(client).status_code == 403

        recorded = {r["surface"] for r in audit_records(audit_zone)}
        assert recorded == {"scaffold_gallery", "claude_setup", "http_config"}
        assert recorded <= set(PROTECTED_SURFACES), (
            "a surface an operator filters on must be one the audit package names"
        )

    def test_the_config_surfaces_spell_the_feed_phrase_identically(self, client):
        """``http_config`` and ``setup_patch`` share one searchable phrase.

        Asserted on the shared prefix rather than the whole sentence: the two
        surfaces name different things after it (a key list versus a single
        key), and only the prefix is what an operator searches for.
        """
        for case in (p.values[0] for p in REFUSALS if p.values[0].surface == "http_config"):
            with no_panel_token():
                assert case.drive(client).status_code == 403

        details = [e["target"]["detail"] for e in recent_activity(client)]
        assert details, "the config surfaces published nothing"
        for detail in details:
            assert detail.startswith(CONFIG_FEED_PHRASE), detail


class TestTheGateIsDiscriminating:
    """The negative controls, without which every assertion above is cheap.

    A gate wired to refuse *everything* would satisfy the whole suite: every
    403 would arrive, every record would be written, every frame would be
    published. What separates a protected set from a wall is that the ordinary
    writes still land -- and land *silently*, because a permitted write is not
    an incident and must not spend a line of the audit log or a frame of the
    operator's feed.
    """

    def test_an_exempt_config_key_still_patches(self, client, audit_zone, project_dir):
        """``hooks.debug`` is the one member of the ``hooks`` family left writable.

        It turns on hook tracing and gates nothing, so it fails the inclusion
        rule the rest of the family passes. Asserted here at the PATCH surface
        because that is where an over-broad ``hooks.*`` prefix would first bite
        an operator trying to debug a hook.
        """
        resp = client.patch("/api/config", json={"updates": {"hooks.debug": True}})

        assert resp.status_code == 200, resp.text
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        assert config["hooks"]["debug"] is True
        assert audit_records(audit_zone) == [], "a permitted write is not an incident"
        assert recent_activity(client) == [], "a permitted write is not an incident"

    def test_an_unreserved_artifact_is_still_creatable(self, client, audit_zone, project_dir):
        """``.claude/agents/`` is authorable material; the gallery still writes it."""
        resp = client.post(
            "/api/scaffold/create",
            json={"category": "agents", "name": "operator-authored", "content": "# mine\n"},
        )

        assert resp.status_code == 200, resp.text
        assert (project_dir / ".claude/agents/operator-authored.md").is_file()
        assert audit_records(audit_zone) == []
        assert recent_activity(client) == []

    def test_an_unreserved_claude_setup_file_is_still_writable(self, client, audit_zone):
        """The panel keeps its job: only the protected subset comes back 403."""
        created = client.post(
            "/api/claude-setup",
            json={"path": ".claude/commands/operator-note.md", "content": "# note\n"},
        )
        assert created.status_code == 200, created.text

        saved = client.put(
            "/api/claude-setup",
            json={"path": ".claude/commands/operator-note.md", "content": "# edited\n"},
        )
        assert saved.status_code == 200, saved.text
        assert audit_records(audit_zone) == []
        assert recent_activity(client) == []


# ── setup_patch: the one surface that publishes out-of-process ───────


@pytest.fixture
def mcp_render(tmp_path, monkeypatch):
    """A minimal render carrying both patchable files, audit zone redirected.

    Mirrors ``tests/mcp_server/test_setup_patch_protected.py``: the tool
    resolves its project root as the config file's parent, so pinning
    ``resolve_config_path`` is the whole setup.
    """
    (tmp_path / "config.yml").write_text(
        yaml.dump({"control_system": {"type": "mock", "writes_enabled": False}})
    )
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python"}}}, indent=2) + "\n"
    )
    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path / "var" / "audit")
    with patch(f"{SETUP_MOD}.resolve_config_path", return_value=tmp_path / "config.yml"):
        yield tmp_path


class TestSetupPatchSurface:
    """The MCP child cannot stamp the ring in-process, so it notifies instead.

    Tested at the notify seam rather than against a live server: what belongs
    to this suite is that the refusal reaches the *same* feed with the *same*
    phrase and leaves the *same* record shape, not the transport that carries
    it there -- which its own suite already pins end to end.
    """

    async def test_refusal_records_and_notifies_with_the_shared_phrase(self, mcp_render):
        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

        fn = get_tool_fn(setup_patch)
        with (
            patch(f"{SETUP_MOD}.notify_agent_activity_async") as notify,
            assert_raises_error(error_type="protected_key"),
        ):
            await fn(file="config.yml", key_path="control_system.writes_enabled", value=True)

        (record,) = audit_records(mcp_render / "var" / "audit")
        assert set(record) == AUDIT_RECORD_FIELDS
        assert record["surface"] == "setup_patch"
        assert record["subject"] == "control_system.writes_enabled"
        assert "target=config.yml" in record["detail"]
        assert RESERVED_PATH_CHANNELS["config.yml"] in record["detail"]
        assert record["reason"] == "protected_key"

        notify.assert_called_once()
        assert notify.call_args.args[:2] == ("setup_patch", "config")
        detail = notify.call_args.kwargs["detail"]
        assert detail.startswith(CONFIG_FEED_PHRASE), detail
        assert "config.yml: control_system.writes_enabled" in detail

    async def test_the_mcp_surface_keeps_the_posture_ladder(self, mcp_render, monkeypatch):
        """The counterpart to the HTTP surfaces: this tool *is* a session child.

        ``setup_patch`` runs in the MCP child a Web Terminal session spawned,
        so the marker that session stamped is the true answer -- a per-surface
        constant inside the funnel would have to overwrite it with something
        this process cannot know.
        """
        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

        monkeypatch.setenv(POSTURE_SOURCE_ENV_VAR, POSTURE_SOURCE_SPAWN)

        fn = get_tool_fn(setup_patch)
        with (
            patch(f"{SETUP_MOD}.notify_agent_activity_async"),
            assert_raises_error(error_type="protected_key"),
        ):
            await fn(file="config.yml", key_path="control_system.writes_enabled", value=True)

        (record,) = audit_records(mcp_render / "var" / "audit")
        assert record["posture_source"] == POSTURE_SOURCE_SPAWN

    async def test_a_blocked_detail_carries_no_safety_marker(self, mcp_render):
        """The honest version of a branch that reads as if it should fire here.

        ``_activity_detail`` marks *applied* ``control_system.*`` patches with a
        ``safety config —`` prefix. No refused patch can carry it: every
        ``control_system.*`` key is protected, so the blocked branch answers
        first and the marker is unreachable through this tool. Pinned as an
        absence so nobody later "fixes" the feed by prefixing refusals with a
        marker the applied path uses to mean something else.
        """
        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

        fn = get_tool_fn(setup_patch)
        with (
            patch(f"{SETUP_MOD}.notify_agent_activity_async") as notify,
            assert_raises_error(error_type="protected_key"),
        ):
            await fn(file="config.yml", key_path="control_system.limits_checking.enabled", value=0)

        detail = notify.call_args.kwargs["detail"]
        assert "safety config" not in detail, detail
        assert detail.startswith(CONFIG_FEED_PHRASE), detail

    async def test_the_refusal_message_names_the_channel_and_says_nothing_changed(self, mcp_render):
        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

        fn = get_tool_fn(setup_patch)
        with (
            patch(f"{SETUP_MOD}.notify_agent_activity_async"),
            assert_raises_error(error_type="protected_key") as ctx,
        ):
            await fn(file="config.yml", key_path="approval.mode", value="disabled")

        message = ctx["envelope"]["error_message"]
        assert RESERVED_PATH_CHANNELS["config.yml"] in message, message
        assert "config.yml is unchanged" in message, message
