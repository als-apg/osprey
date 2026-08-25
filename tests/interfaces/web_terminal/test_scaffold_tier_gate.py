"""Tests for the ``web.scaffold_gallery.write_enabled`` tier gate.

Gallery-authored ``.claude/rules``, ``.claude/skills`` and ``.claude/agents``
content is loaded by the agent at PROJECT scope — it is instruction the agent
obeys, not decoration. The gallery routes classify as ``Tier.OPERATOR``, so
without this key any authenticated session of a read-only tier can author what
the agent will run. ``web.scaffold_gallery.write_enabled: false`` therefore
withdraws the gallery's whole WRITE surface on the server:

* every write/delete verb under ``/api/scaffold`` refuses with 403, and
* ``GET /api/panels`` reports the posture as ``scaffold_write_enabled`` so the
  browser can stop painting controls for a surface that refuses them.

The route half is the load-bearing one. A client-only guard is undone by typing
the URL or by ``curl`` — the same cosmetic-gate failure ``ui_mode: simple`` was.

Three properties are asserted directly, because each can regress on its own:

* **The gate is FIRST.** It runs ahead of every service call, so a disabled
  deployment never constructs the gallery service, never touches disk, and
  never reports a protected-set refusal in place of the posture refusal.
* **Absent means enabled.** An app with no ``scaffold_write_enabled`` on
  ``app.state`` (every route unit suite, and any deployment that never mentions
  the key) behaves byte-for-byte as it did before this key existed.
* **The lifespan resolves it once.** ``create_app`` reads the key into
  ``app.state.scaffold_write_enabled``; a quoted ``"false"`` is honoured as the
  boolean a human meant, and an unreadable config fails OPEN — the shipped
  single-user posture must not be revoked by a config-read error.

Read routes are untouched throughout: seeing what the agent is running is not
a write, and a tier that may not author still has to be able to look.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    coerce_config_flag,
    create_app,
    register_scaffold_conflict_handlers,
)
from osprey.interfaces.web_terminal.routes import router as full_router
from osprey.interfaces.web_terminal.routes.scaffold import router as scaffold_router

#: The dotted key under test. Spelled once so a rename shows up as one edit.
SCAFFOLD_WRITE_KEY = "web.scaffold_gallery.write_enabled"

_SVC = "osprey.interfaces.web_terminal.routes.scaffold.ScaffoldGalleryService"


@pytest.fixture
def svc():
    """A mock gallery service, patched in at the route module's import site.

    Mocked deliberately: these tests pin the gate, and the strongest statement
    a gated route can make is that the service was never *constructed* — no
    disk read, no ownership store, nothing to undo.
    """
    service = MagicMock()
    with patch(_SVC, return_value=service) as ctor:
        service.ctor = ctor
        yield service


def _app(tmp_path, *, write_enabled):
    """A routes-only app pointed at *tmp_path*.

    ``write_enabled`` of ``None`` leaves the attribute OFF ``app.state``
    entirely — the state of every app built without the web terminal's
    lifespan, and the case that proves the routes default to enabled rather
    than to whatever a fixture happened to set.
    """
    application = FastAPI()
    application.include_router(scaffold_router)
    register_scaffold_conflict_handlers(application)
    application.state.project_cwd = str(tmp_path)
    if write_enabled is not None:
        application.state.scaffold_write_enabled = write_enabled
    return application


@pytest.fixture
def disabled_client(tmp_path):
    return TestClient(_app(tmp_path, write_enabled=False))


@pytest.fixture
def enabled_client(tmp_path):
    return TestClient(_app(tmp_path, write_enabled=True))


@pytest.fixture
def default_client(tmp_path):
    """No ``scaffold_write_enabled`` on state at all — the absent-key posture."""
    return TestClient(_app(tmp_path, write_enabled=None))


def _write_requests(client):
    """Every write/delete verb the gallery reaches, as ``(label, call)`` pairs.

    One table, because the gallery's write surface is ONE privilege: a verb
    that quietly stayed open is the whole gap. ``POST
    /api/scaffold/untracked/register`` is here with the rest — registering an
    untracked file rewrites ``config.yml`` so the file becomes managed, which
    is authoring by another name.
    """
    return [
        (
            "POST /api/scaffold/create",
            lambda: client.post(
                "/api/scaffold/create",
                json={"category": "rules", "name": "my-rule", "content": "x"},
            ),
        ),
        (
            "POST /api/scaffold/{name}/claim",
            lambda: client.post("/api/scaffold/rules/my-rule/claim"),
        ),
        (
            "PUT /api/scaffold/{name}/override",
            lambda: client.put("/api/scaffold/rules/my-rule/override", json={"content": "x"}),
        ),
        (
            "DELETE /api/scaffold/{name}/override",
            lambda: client.delete("/api/scaffold/rules/my-rule/override"),
        ),
        (
            "DELETE /api/scaffold/untracked/{name}",
            lambda: client.delete("/api/scaffold/untracked/rules/stray"),
        ),
        (
            "POST /api/scaffold/untracked/register",
            lambda: client.post("/api/scaffold/untracked/register", json={"name": "rules/stray"}),
        ),
    ]


def _read_requests(client):
    """Every read verb the gallery reaches. None of these may be gated."""
    return [
        ("GET /api/scaffold", lambda: client.get("/api/scaffold")),
        ("GET /api/scaffold/untracked", lambda: client.get("/api/scaffold/untracked")),
        ("GET /api/scaffold/{name}", lambda: client.get("/api/scaffold/rules/my-rule")),
        (
            "GET /api/scaffold/{name}/framework",
            lambda: client.get("/api/scaffold/rules/my-rule/framework"),
        ),
        (
            "GET /api/scaffold/{name}/diff",
            lambda: client.get("/api/scaffold/rules/my-rule/diff"),
        ),
    ]


def _seed_reads(service):
    """Give the mock service plausible read returns for the read-route table."""
    service.list_artifacts.return_value = [{"status": "framework"}]
    service.scan_untracked.return_value = []
    service.get_content.return_value = {"content": "x", "source": "framework"}
    service.get_framework_content.return_value = "x"
    service.compute_diff.return_value = {"diff": ""}


# ---- Disabled: the write surface is closed ----


class TestDisabledRefusesEveryWrite:
    def test_every_write_verb_refuses_with_403(self, disabled_client, svc):
        for label, call in _write_requests(disabled_client):
            resp = call()
            assert resp.status_code == 403, f"{label} answered {resp.status_code}"

    def test_refusal_names_the_key_that_produced_it(self, disabled_client, svc):
        """An operator who meets the refusal must learn which switch made it."""
        for label, call in _write_requests(disabled_client):
            detail = call().json()["detail"]
            assert "scaffold" in detail.lower(), label
            assert SCAFFOLD_WRITE_KEY in detail, label

    def test_the_service_is_never_constructed(self, disabled_client, svc):
        """The gate runs FIRST — ahead of every service call and every disk touch."""
        for _label, call in _write_requests(disabled_client):
            call()
        assert svc.ctor.call_count == 0
        assert svc.create_artifact.call_count == 0
        assert svc.scaffold_override.call_count == 0
        assert svc.save_override.call_count == 0
        assert svc.unoverride.call_count == 0
        assert svc.register_untracked.call_count == 0
        assert svc.delete_untracked.call_count == 0

    def test_reads_are_untouched(self, disabled_client, svc):
        """Looking is not authoring: every read route still answers 200."""
        _seed_reads(svc)
        for label, call in _read_requests(disabled_client):
            resp = call()
            assert resp.status_code == 200, f"{label} answered {resp.status_code}"


# ---- Enabled and absent: the shipped posture, unchanged ----


class TestEnabledAndAbsentServeWrites:
    @pytest.mark.parametrize("fixture_name", ["enabled_client", "default_client"])
    def test_writes_still_land(self, fixture_name, request, svc):
        """Every write verb reaches its service call and answers 200-shaped."""
        client = request.getfixturevalue(fixture_name)
        svc.create_artifact.return_value = {"status": "created"}
        svc.scaffold_override.return_value = {"status": "claimed"}
        svc.save_override.return_value = {"status": "saved"}
        svc.unoverride.return_value = {"status": "released"}
        svc.register_untracked.return_value = {"status": "registered"}
        svc.delete_untracked.return_value = {"status": "deleted"}
        for label, call in _write_requests(client):
            resp = call()
            assert resp.status_code == 200, f"{label} answered {resp.status_code}"

    def test_absent_state_attribute_creates(self, default_client, svc):
        """The spot check with teeth: no attribute at all is the enabled path."""
        svc.create_artifact.return_value = {"status": "created"}
        resp = default_client.post(
            "/api/scaffold/create",
            json={"category": "rules", "name": "my-rule", "content": "x"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "created"}
        svc.create_artifact.assert_called_once_with("rules", "my-rule", "x")

    def test_protected_refusals_still_reach_the_operator(self, default_client, svc):
        """An enabled gallery still reports the protected-set refusal as itself."""
        from osprey.interfaces.web_terminal.scaffold_gallery_service import (
            ProtectedArtifactError,
        )

        svc.create_artifact.side_effect = ProtectedArtifactError(
            "rules/ is written by the build",
            channel="the build",
            output_path=".claude/rules/x.md",
        )
        resp = default_client.post("/api/scaffold/create", json={"category": "rules", "name": "x"})
        assert resp.status_code == 403
        assert SCAFFOLD_WRITE_KEY not in resp.json()["detail"]


# ---- The client's half: GET /api/panels publishes the posture ----


class TestPanelsPayload:
    def _panels_client(self, tmp_path, *, write_enabled):
        application = FastAPI()
        application.include_router(full_router)
        application.state.project_cwd = str(tmp_path)
        if write_enabled is not None:
            application.state.scaffold_write_enabled = write_enabled
        return TestClient(application)

    def test_disabled_payload_says_so(self, tmp_path):
        client = self._panels_client(tmp_path, write_enabled=False)
        assert client.get("/api/panels").json()["scaffold_write_enabled"] is False

    def test_enabled_payload_says_so(self, tmp_path):
        client = self._panels_client(tmp_path, write_enabled=True)
        assert client.get("/api/panels").json()["scaffold_write_enabled"] is True

    def test_absent_state_attribute_reads_as_enabled(self, tmp_path):
        client = self._panels_client(tmp_path, write_enabled=None)
        assert client.get("/api/panels").json()["scaffold_write_enabled"] is True


# ---- coerce_config_flag on this key ----


class TestFlagCoercion:
    def test_real_booleans_pass_through(self):
        assert coerce_config_flag(SCAFFOLD_WRITE_KEY, False, True) is False
        assert coerce_config_flag(SCAFFOLD_WRITE_KEY, True, False) is True

    def test_absent_key_takes_the_default(self):
        assert coerce_config_flag(SCAFFOLD_WRITE_KEY, None, True) is True

    def test_quoted_false_is_honoured(self):
        """`bool("false")` is True — the trap this helper exists to close."""
        for spelling in ("false", "False", "FALSE", " no ", "off"):
            assert coerce_config_flag(SCAFFOLD_WRITE_KEY, spelling, True) is False

    def test_uninterpretable_value_warns_and_takes_the_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = coerce_config_flag(SCAFFOLD_WRITE_KEY, {"write_enabled": 1}, True)
        assert result is True
        assert any(
            SCAFFOLD_WRITE_KEY in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING naming the key"


# ---- Startup: the lifespan resolves the key once, onto app.state ----


@pytest.fixture
def workspace_dir(tmp_path):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    return workspace


def _started_app(workspace_dir, configured, *, raises=False):
    """Run ``create_app``'s lifespan with *configured* as the key's value.

    ``configured`` of ``None`` omits the key, exercising the absent-key path;
    ``raises=True`` makes the config read blow up, which is the unreadable-config
    path the gate must fail OPEN on. ``get_config_value`` is patched at its
    definition site because the lifespan imports it inside the function; every
    other key it reads falls through to the default the caller passed, which is
    what an absent config.yml gives them anyway.
    """

    def fake_get_config_value(key, default=None, *args, **kwargs):
        if key == SCAFFOLD_WRITE_KEY:
            if raises:
                raise OSError("config.yml is unreadable")
            if configured is not None:
                return configured
        return default

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch("osprey.utils.config.get_config_value", fake_get_config_value),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            yield client


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, True),
        (True, True),
        (False, False),
        ("false", False),
        ("true", True),
        (["not", "a", "flag"], True),
    ],
)
def test_lifespan_resolves_the_flag(workspace_dir, configured, expected):
    generator = _started_app(workspace_dir, configured)
    client = next(generator)
    try:
        assert client.app.state.scaffold_write_enabled is expected
        assert client.get("/api/panels").json()["scaffold_write_enabled"] is expected
    finally:
        next(generator, None)


def test_lifespan_fails_open_on_an_unreadable_config(workspace_dir):
    """A config-read error must not silently revoke the shipped posture."""
    generator = _started_app(workspace_dir, None, raises=True)
    client = next(generator)
    try:
        assert client.app.state.scaffold_write_enabled is True
    finally:
        next(generator, None)


def test_lifespan_disabled_refuses_the_write_routes(workspace_dir):
    """End to end: the configured key really does close the write surface."""
    generator = _started_app(workspace_dir, False)
    client = next(generator)
    try:
        assert (
            client.post("/api/scaffold/create", json={"category": "rules", "name": "x"}).status_code
            == 403
        )
        assert client.post("/api/scaffold/rules/x/claim").status_code == 403
        assert (
            client.put("/api/scaffold/rules/x/override", json={"content": "y"}).status_code == 403
        )
        assert client.delete("/api/scaffold/rules/x/override").status_code == 403
        assert client.delete("/api/scaffold/untracked/rules/x").status_code == 403
        assert (
            client.post("/api/scaffold/untracked/register", json={"name": "rules/x"}).status_code
            == 403
        )
        # And the list route still answers, because reading is not authoring.
        assert client.get("/api/scaffold").status_code == 200
    finally:
        next(generator, None)
