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
* **Absent means refused.** An app with no ``scaffold_write_enabled`` on
  ``app.state`` never ran the lifespan that decides the tier, and it refuses
  every write exactly as a disabled one does. A deployment that never mentions
  the key still gets writes: the lifespan resolves the absent key to enabled.
* **The lifespan resolves it once.** ``create_app`` reads the key into
  ``app.state.scaffold_write_enabled``; a quoted ``"false"`` is honoured as the
  boolean a human meant, and an unreadable config fails OPEN — the shipped
  single-user posture must not be revoked by a config-read error.

Read routes are untouched throughout: seeing what the agent is running is not
a write, and a tier that may not author still has to be able to look.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    create_app,
    register_scaffold_conflict_handlers,
)
from osprey.interfaces.web_terminal.routes.scaffold import router as scaffold_router

from .conftest import bare_route_app

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
    lifespan, and the case that proves the routes refuse an undecided tier
    rather than inherit whatever a fixture happened to set.
    """
    application = bare_route_app(scaffold_router, project_cwd=str(tmp_path))
    register_scaffold_conflict_handlers(application)
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
    """No ``scaffold_write_enabled`` on state at all — an app that made no tier decision."""
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


#: The two postures that refuse: configured off, and never decided at all.
REFUSING_CLIENTS = ["disabled_client", "default_client"]


class TestDisabledRefusesEveryWrite:
    @pytest.mark.usefixtures("svc")
    @pytest.mark.parametrize("fixture_name", REFUSING_CLIENTS)
    def test_refusal_names_the_key_that_produced_it(self, fixture_name, request):
        """An operator who meets the refusal must learn which switch made it."""
        for label, call in _write_requests(request.getfixturevalue(fixture_name)):
            resp = call()
            assert resp.status_code == 403, f"{label} answered {resp.status_code}"
            detail = resp.json()["detail"]
            assert "scaffold" in detail.lower(), label
            assert SCAFFOLD_WRITE_KEY in detail, label

    @pytest.mark.parametrize("fixture_name", REFUSING_CLIENTS)
    def test_the_service_is_never_constructed(self, fixture_name, request, svc):
        """The gate runs FIRST — ahead of every service call and every disk touch."""
        for _label, call in _write_requests(request.getfixturevalue(fixture_name)):
            call()
        assert svc.ctor.call_count == 0
        assert svc.create_artifact.call_count == 0
        assert svc.scaffold_override.call_count == 0
        assert svc.save_override.call_count == 0
        assert svc.unoverride.call_count == 0
        assert svc.register_untracked.call_count == 0
        assert svc.delete_untracked.call_count == 0

    @pytest.mark.parametrize("fixture_name", REFUSING_CLIENTS)
    def test_reads_are_untouched(self, fixture_name, request, svc):
        """Looking is not authoring: every read route still answers 200."""
        _seed_reads(svc)
        for label, call in _read_requests(request.getfixturevalue(fixture_name)):
            resp = call()
            assert resp.status_code == 200, f"{label} answered {resp.status_code}"


# ---- Enabled: the shipped posture ----


class TestEnabledServesWrites:
    def test_writes_still_land(self, enabled_client, svc):
        """Every write verb reaches its service call and answers 200-shaped."""
        client = enabled_client
        svc.create_artifact.return_value = {"status": "created"}
        svc.scaffold_override.return_value = {"status": "claimed"}
        svc.save_override.return_value = {"status": "saved"}
        svc.unoverride.return_value = {"status": "released"}
        svc.register_untracked.return_value = {"status": "registered"}
        svc.delete_untracked.return_value = {"status": "deleted"}
        for label, call in _write_requests(client):
            resp = call()
            assert resp.status_code == 200, f"{label} answered {resp.status_code}"


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
