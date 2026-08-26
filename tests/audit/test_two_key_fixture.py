"""Two keys, one record each, and nothing filed twice.

The P1/P2 refusal ledgers are gone: every protected-set refusal now goes
through :mod:`osprey.audit.protected` onto ``var/audit/<identity>/<surface>``,
and every executor refusal through :mod:`osprey.audit.dedup` onto the
``executor`` surface. Each surface's own suite pins what its record *says*.
What no single suite can pin is the property the retirement is judged on:

* **One record per (surface, key).** Two protected keys in one request are two
  lines, not one summarised line and not one line per request — the ledger
  counts what was attempted.
* **Zero duplicates.** No ``(surface, subject)`` pair appears twice for one
  attempt, and — the case the marker exists for — the HTTP mutation layer files
  nothing on top of a route that already recorded its own refusal. A duplicate
  is not a cosmetic problem: an operator reading the trail cannot tell a
  double-recorded refusal from an agent that pushed twice.

Two keys rather than one throughout, because a per-request recorder and a
per-key recorder are indistinguishable when only one key is refused.

The claiming half of the contract is structural, so it is pinned structurally
too (:class:`TestTheClaimingSitesRunOnTheAwaitedTask`): a marker set on a
worker thread is invisible to the layer that awaited the call, so every route
reaching a claiming recorder must be ``async def``. The one site that does not
claim — the container-start restore — is pinned in the other direction.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from osprey.audit import dedup, protected, writer
from osprey.audit.envelope import DECISION_REFUSED
from osprey.audit.protected import (
    PROTECTED_SURFACES,
    SURFACE_CLAUDE_SETUP,
    SURFACE_HTTP_CONFIG,
    SURFACE_SCAFFOLD_GALLERY,
    SURFACE_SCAFFOLD_RESTORE,
    SURFACE_SETUP_PATCH,
)
from osprey.interfaces.common_middleware import HTTP_MUTATION_SURFACE, HttpAuditMiddleware
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV, acting_identity

pytestmark = pytest.mark.unit

#: Two protected config keys, in two different blocks. Different blocks on
#: purpose: a recorder that keyed on the top-level section would collapse two
#: keys of the same block into one line and still pass a same-block fixture.
KEY_A = "control_system.writes_enabled"
KEY_B = "agent_data.base_dir"

#: Two protected paths, for the surfaces whose subject is a path.
PATH_A = ".claude/rules/safety.md"
PATH_B = ".claude/settings.json"

#: The channel string the fixtures pass through. Not under test here — each
#: surface's own suite pins that it names the real owning channel — but it must
#: reach the record, so it is distinctive.
CHANNEL = "the build profile this project was rendered from"


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    """A redirected audit zone with a known identity and no posture stamps.

    The identity is pinned rather than inherited so the ledger's directory is
    predictable, and the posture variables are cleared so a developer's shell
    cannot change what the records say.
    """
    for marker in (
        TERMINAL_USER_ENV,
        writer.AUDIT_WRITER_ENV,
        protected.POSTURE_ENV_VAR,
        protected.POSTURE_SOURCE_ENV_VAR,
        protected.POSTURE_SESSION_ENV_VAR,
    ):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

    root = tmp_path / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: root)
    dedup.clear_recorded()
    yield root
    dedup.clear_recorded()


def _records(root: Path, surface: str) -> list[dict]:
    """Every record on one surface's ledger, oldest first."""
    path = root / acting_identity() / f"{surface}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_records(root: Path) -> list[dict]:
    """Every record in the zone, whichever surface filed it."""
    if not root.exists():
        return []
    out: list[dict] = []
    for path in sorted(root.rglob("*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def _keys(records: list[dict]) -> list[tuple[str, str]]:
    """The ``(surface, subject)`` pair of each record — the identity of a refusal."""
    return [(record["surface"], record["subject"]) for record in records]


def _assert_no_duplicates(records: list[dict]) -> None:
    keys = _keys(records)
    assert len(keys) == len(set(keys)), f"a refusal was recorded twice: {keys}"


# --------------------------------------------------------------------------- #
# Per-(surface, key), through the funnel every protected surface calls
# --------------------------------------------------------------------------- #


class TestOneRecordPerSurfaceAndKey:
    """Two subjects on one surface are two records, and never more than two."""

    @pytest.mark.parametrize(
        "surface",
        [
            SURFACE_SETUP_PATCH,
            SURFACE_HTTP_CONFIG,
            SURFACE_CLAUDE_SETUP,
            SURFACE_SCAFFOLD_GALLERY,
            SURFACE_SCAFFOLD_RESTORE,
        ],
    )
    def test_two_subjects_are_two_records(self, audit_root, surface):
        subjects = (
            (KEY_A, KEY_B)
            if surface in {SURFACE_SETUP_PATCH, SURFACE_HTTP_CONFIG}
            else (
                PATH_A,
                PATH_B,
            )
        )
        for subject in subjects:
            protected.record_protected_refusal(
                surface=surface,
                target_file="config.yml",
                key_or_path=subject,
                channel=CHANNEL,
                reason="protected_key",
                claim=surface != SURFACE_SCAFFOLD_RESTORE,
            )

        records = _records(audit_root, surface)
        assert [r["subject"] for r in records] == list(subjects)
        assert _keys(records) == [(surface, subject) for subject in subjects]
        _assert_no_duplicates(_all_records(audit_root))

    def test_every_documented_surface_files_into_a_ledger_of_its_own(self, audit_root):
        """One surface, one file: the name an operator filters on is the file name."""
        for surface in PROTECTED_SURFACES:
            protected.record_protected_refusal(
                surface=surface,
                target_file="config.yml",
                key_or_path=KEY_A,
                channel=CHANNEL,
                reason="protected_key",
                claim=False,
            )

        for surface in PROTECTED_SURFACES:
            assert (audit_root / acting_identity() / f"{surface}.jsonl").is_file()
        assert len(_all_records(audit_root)) == len(PROTECTED_SURFACES)
        _assert_no_duplicates(_all_records(audit_root))

    def test_the_record_carries_what_the_retired_ledger_carried(self, audit_root):
        """Nothing the ``protected-writes`` record answered is unanswerable now."""
        protected.record_protected_refusal(
            surface=SURFACE_HTTP_CONFIG,
            target_file="config.yml",
            key_or_path=KEY_A,
            channel=CHANNEL,
            reason="protected_key",
        )

        (record,) = _records(audit_root, SURFACE_HTTP_CONFIG)
        assert record["surface"] == SURFACE_HTTP_CONFIG
        assert record["subject"] == KEY_A
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == "protected_key"
        assert "target=config.yml" in record["detail"]
        assert CHANNEL in record["detail"]
        # New with the unified envelope, and the reason the retirement happened:
        # the record now says *who*, under which posture, in which session.
        assert record["actor"] == "svc.terminal"
        assert record["posture"] == protected.POSTURE_WRITES
        assert record["posture_source"] == "process"
        assert record["session"] is None


# --------------------------------------------------------------------------- #
# The marker: what the outer layers see
# --------------------------------------------------------------------------- #


class TestTheOuterLayerDefersExactlyWhereItShould:
    def test_a_claiming_refusal_marks_the_decision(self, audit_root):
        protected.record_protected_refusal(
            surface=SURFACE_HTTP_CONFIG,
            target_file="config.yml",
            key_or_path=KEY_A,
            channel=CHANNEL,
            reason="protected_key",
        )

        marker = dedup.recorded_decision()
        assert marker is not None
        assert (marker.decision, marker.reason) == (DECISION_REFUSED, "protected_key")

    def test_the_restore_does_not_claim_the_call_it_runs_inside(self, audit_root):
        """It skips one stored body and lets the caller continue.

        Claiming would silence the outer layer's record of an operation that
        *did* happen — the request the restore ran inside was not refused.
        """
        protected.record_protected_refusal(
            surface=SURFACE_SCAFFOLD_RESTORE,
            target_file=PATH_A,
            key_or_path=PATH_A,
            channel=CHANNEL,
            reason="reserved path in ownership store",
            claim=False,
        )

        assert _records(audit_root, SURFACE_SCAFFOLD_RESTORE), "the refusal is still durable"
        assert dedup.recorded_decision() is None

    def test_the_restore_call_site_itself_asks_not_to_claim(self, audit_root, tmp_path):
        """The site, not just the funnel: ``ownership`` must pass ``claim=False``.

        The test above pins what the funnel does when asked. This one drives
        the shipped recorder — flip ``claim=False`` to ``claim=True`` in
        :func:`~osprey.interfaces.web_terminal.ownership._audit_restore_refusal`
        and nothing else in the suite notices, while the request the restore
        ran inside silently loses its own ``http_mutation`` record to a marker
        that answers ``refused`` for a call that was not refused.

        Inside a :func:`~osprey.audit.dedup.decision_scope` because that is
        where the outer layer reads the marker from: the scope is the call, and
        a marker is only ever asked about before it closes.
        """
        from osprey.interfaces.web_terminal import ownership

        with dedup.decision_scope():
            ownership._audit_restore_refusal(
                PATH_A,
                channel=CHANNEL,
                reason="reserved path in ownership store",
                because="the stored body is not this channel's to install",
            )

            assert dedup.recorded_decision() is None, (
                "the restore claimed the call it runs inside; the layer above it "
                "will now defer to a refusal that was not the answer to its request"
            )

        records = _records(audit_root, SURFACE_SCAFFOLD_RESTORE)
        assert _keys(records) == [(SURFACE_SCAFFOLD_RESTORE, PATH_A)], (
            "the refusal is still durable -- not claiming is not not recording"
        )

    def test_the_claude_setup_call_site_does_claim(self, audit_root, tmp_path):
        """The mirror: a site that *does* refuse the call leaves the marker.

        Same shape, opposite answer, so the pair says which sites claim rather
        than only that one does not.
        """
        from osprey.interfaces.web_terminal import claude_code_files

        with dedup.decision_scope():
            with pytest.raises(claude_code_files.ProtectedWriteError):
                claude_code_files._refuse_if_reserved(tmp_path, PATH_A)

            marker = dedup.recorded_decision()
            assert marker is not None, "the panel refused the request and must claim it"
            assert (marker.decision, marker.reason) == (DECISION_REFUSED, "reserved path")

        assert _keys(_records(audit_root, SURFACE_CLAUDE_SETUP)) == [(SURFACE_CLAUDE_SETUP, PATH_A)]

    def test_the_last_key_of_a_multi_key_refusal_owns_the_call(self, audit_root):
        """A per-key loop marks per key; the outer layer needs only one answer."""
        for key in (KEY_A, KEY_B):
            protected.record_protected_refusal(
                surface=SURFACE_HTTP_CONFIG,
                target_file="config.yml",
                key_or_path=key,
                channel=CHANNEL,
                reason="protected_key",
            )

        marker = dedup.recorded_decision()
        assert marker is not None and marker.decision == DECISION_REFUSED


# --------------------------------------------------------------------------- #
# End to end: the real config-route recorder under the real HTTP audit layer
# --------------------------------------------------------------------------- #


class TestTheHttpLayerFilesNothingOnTopOfTheRoute:
    """The duplicate the retirement removes, driven through both layers."""

    @pytest.fixture
    def client(self, audit_root):
        from osprey.interfaces.web_terminal.routes import config as config_routes

        app = FastAPI()

        @app.patch("/api/config")
        async def refuse_two_keys(request: Request):
            # The production recorder, not a stand-in: it is the loop that has
            # to produce one record per key, and the layer below has to defer
            # to the marker it leaves.
            raise config_routes._refuse_protected_keys(request, [KEY_A, KEY_B])

        app.add_middleware(HttpAuditMiddleware)
        return TestClient(app)

    def test_two_keys_are_two_records_and_the_layer_below_adds_none(self, client, audit_root):
        response = client.patch("/api/config", json={})

        assert response.status_code == 403
        assert _keys(_records(audit_root, SURFACE_HTTP_CONFIG)) == [
            (SURFACE_HTTP_CONFIG, KEY_A),
            (SURFACE_HTTP_CONFIG, KEY_B),
        ]
        assert _records(audit_root, HTTP_MUTATION_SURFACE) == [], (
            "the HTTP layer filed a second record for a refusal the route already owned"
        )
        _assert_no_duplicates(_all_records(audit_root))

    def test_a_retry_is_a_second_pair_of_records(self, client, audit_root):
        """Zero duplicates is not zero repeats: the ledger still counts attempts."""
        assert client.patch("/api/config", json={}).status_code == 403
        assert client.patch("/api/config", json={}).status_code == 403

        subjects = [r["subject"] for r in _records(audit_root, SURFACE_HTTP_CONFIG)]
        assert subjects == [KEY_A, KEY_B, KEY_A, KEY_B]
        assert _records(audit_root, HTTP_MUTATION_SURFACE) == []

    def test_an_admitted_mutation_is_still_recorded_by_the_layer_below(self, audit_root):
        """The negative control: with nothing claimed, the HTTP layer records.

        Without this, a marker that suppressed *everything* would pass every
        assertion above while silently emptying the mutation trail.
        """
        app = FastAPI()

        @app.patch("/api/config")
        async def allow(request: Request):
            return {"status": "ok"}

        app.add_middleware(HttpAuditMiddleware)

        assert TestClient(app).patch("/api/config", json={}).status_code == 200
        assert len(_records(audit_root, HTTP_MUTATION_SURFACE)) == 1


# --------------------------------------------------------------------------- #
# End to end: the real setup_patch tool
# --------------------------------------------------------------------------- #


class TestTheSetupPatchToolRecordsPerKey:
    @pytest.fixture
    def render(self, audit_root, tmp_path, monkeypatch):
        """The minimal render ``setup_patch`` resolves its root from."""
        from unittest.mock import patch as mock_patch

        import yaml

        (tmp_path / "config.yml").write_text(
            yaml.dump(
                {
                    "control_system": {"type": "mock", "writes_enabled": False},
                    "agent_data": {"base_dir": "./_agent_data"},
                }
            )
        )
        with mock_patch(
            "osprey.mcp_server.workspace.tools.setup.resolve_config_path",
            return_value=tmp_path / "config.yml",
        ):
            yield tmp_path

    async def test_two_refused_patches_are_two_records(self, render, audit_root):
        from unittest.mock import patch as mock_patch

        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

        fn = get_tool_fn(setup_patch)
        with mock_patch("osprey.mcp_server.workspace.tools.setup.notify_agent_activity_async"):
            for key in (KEY_A, KEY_B):
                with assert_raises_error(error_type="protected_key"):
                    await fn(file="config.yml", key_path=key, value="x")

        records = _records(audit_root, SURFACE_SETUP_PATCH)
        assert _keys(records) == [(SURFACE_SETUP_PATCH, KEY_A), (SURFACE_SETUP_PATCH, KEY_B)]
        _assert_no_duplicates(_all_records(audit_root))


# --------------------------------------------------------------------------- #
# The structural half of the contract
# --------------------------------------------------------------------------- #


class TestTheClaimingSitesRunOnTheAwaitedTask:
    """A marker is only seen by a layer that awaited the call that set it.

    Starlette hands a ``def`` route to ``run_in_threadpool``, which *copies* the
    context — the mark would die with the worker thread and the layer above
    would file ``allowed``/``route_refused`` over a decision the route already
    owned. So every route that reaches a claiming recorder is ``async def``,
    and that is a property worth failing a build over rather than a convention.
    """

    def test_the_config_write_routes_are_coroutines(self):
        from osprey.interfaces.web_terminal.routes import config as config_routes

        for route in (config_routes.put_config, config_routes.patch_config):
            assert inspect.iscoroutinefunction(route), route.__name__

    def test_the_claude_setup_write_routes_are_coroutines(self):
        from osprey.interfaces.web_terminal.routes import config as config_routes

        for route in (config_routes.save_claude_setup, config_routes.create_claude_setup):
            assert inspect.iscoroutinefunction(route), route.__name__

    def test_the_scaffold_write_routes_are_coroutines(self):
        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        for route in (
            scaffold_routes.register_untracked_scaffold,
            scaffold_routes.delete_untracked_scaffold,
            scaffold_routes.create_artifact,
            scaffold_routes.claim_scaffold,
            scaffold_routes.save_scaffold_override,
            scaffold_routes.delete_scaffold_override,
        ):
            assert inspect.iscoroutinefunction(route), route.__name__

    def test_the_setup_patch_tool_is_a_coroutine(self):
        from osprey.mcp_server.workspace.tools.setup import setup_patch
        from tests.mcp_server.conftest import get_tool_fn

        assert inspect.iscoroutinefunction(get_tool_fn(setup_patch))

    def test_the_executor_refusal_recorder_is_reached_from_coroutines(self):
        from osprey.mcp_server.python_executor.tools import _execution_gates as gates

        for fn in (
            gates.record_and_alert_refusal,
            gates.refuse_readonly_write,
            gates.enforce_path_policy,
            gates.report_runtime_refusal,
        ):
            assert inspect.iscoroutinefunction(fn), fn.__name__


# --------------------------------------------------------------------------- #
# Spellings this module cannot import
# --------------------------------------------------------------------------- #


class TestPinnedSpellings:
    """The three recorders address one posture vocabulary under three sets of names.

    All of them are re-exports of :mod:`osprey.audit.posture`; these pins hold
    the aliases to the source, so a recorder that quietly went back to spelling
    its own would describe a posture nobody set.
    """

    def test_the_posture_env_and_vocabulary_match_the_middleware(self):
        from osprey.mcp_server import audit_middleware as am

        assert protected.POSTURE_ENV_VAR == am.POSTURE_ENV
        assert protected.POSTURE_SOURCE_ENV_VAR == am.POSTURE_SOURCE_ENV
        assert protected.POSTURE_SESSION_ENV_VAR == am.POSTURE_SESSION_ENV
        assert protected.SANDBOX_MODE == am.SANDBOX_MODE
        assert protected.POSTURE_SANDBOX == am.POSTURE_SANDBOX
        assert protected.POSTURE_WRITES == am.POSTURE_WRITES

    def test_the_executor_gates_and_the_protected_funnel_agree(self):
        from osprey.mcp_server.python_executor.tools import _execution_gates as gates

        assert gates.POSTURE_ENV_VAR == protected.POSTURE_ENV_VAR
        assert gates.SANDBOX_POSTURE == protected.SANDBOX_MODE
        assert gates.POSTURE_SANDBOX == protected.POSTURE_SANDBOX
        assert gates.POSTURE_WRITES == protected.POSTURE_WRITES

    def test_the_sandbox_posture_reaches_the_record(self, audit_root, monkeypatch):
        """The posture is read per record, so a sandboxed session says so."""
        monkeypatch.setenv(protected.POSTURE_ENV_VAR, protected.SANDBOX_MODE)
        monkeypatch.setenv(protected.POSTURE_SESSION_ENV_VAR, "chat-7")

        protected.record_protected_refusal(
            surface=SURFACE_CLAUDE_SETUP,
            target_file=PATH_A,
            key_or_path=PATH_A,
            channel=CHANNEL,
            reason="reserved path",
        )

        (record,) = _records(audit_root, SURFACE_CLAUDE_SETUP)
        assert record["posture"] == protected.POSTURE_SANDBOX
        assert record["session"] == "chat-7"


# --------------------------------------------------------------------------- #
# The retirement itself
# --------------------------------------------------------------------------- #


class TestTheOldLedgersAreGone:
    def test_the_p1_ledger_module_no_longer_exists(self):
        """Retired, not deprecated: an importable copy is a second trail."""
        with pytest.raises(ImportError):
            import osprey.services.python_executor.refusal_audit  # noqa: F401

    def test_a_refusal_never_lands_in_a_deployment_wide_file(self, audit_root):
        """Every record is under an identity directory — the mount-isolated unit."""
        protected.record_protected_refusal(
            surface=SURFACE_SCAFFOLD_GALLERY,
            target_file=PATH_A,
            key_or_path=PATH_A,
            channel=CHANNEL,
            reason="reserved path",
        )

        assert not (audit_root / "protected-writes.jsonl").exists()
        assert not (audit_root / "readonly-refusals.jsonl").exists()
        assert [p.parent.name for p in audit_root.rglob("*.jsonl")] == [acting_identity()]
