"""Cross-launcher contract for the login URL and the session cookie's scope.

Seven entry points bind an OSPREY web surface and print the ``?token=`` URL that
is the operator's only way past the auth gate: ``osprey web``, ``osprey
artifacts web`` (both its direct and its ``--reload`` shape), ``osprey
channel-finder web``, ``osprey theme-lab``, ARIEL's ``run_web``, and ``osprey
chat``, which hosts every companion panel server in its own process. Each one
wires the same two things, and each one used to get at least one of them wrong
on its own.

**The settled port must be published before the app is built.** Cookies ignore
ports, so ``http://127.0.0.1:10100`` and ``http://127.0.0.1:10200`` are one origin
as far as a browser is concerned; ``session_cookie_name()`` appends the port for
exactly that reason, reading it from ``OSPREY_WEB_PORT``. A launcher that never
publishes it leaves every surface on the host sharing the bare cookie name, so
opening a second one silently logs the operator out of the first. "Before the
app is built" is the load-bearing half: the gate resolves the name per request,
but a companion app that pins one does so at construction.

``osprey chat`` is the one launcher that binds several ports from a single
process. All of its companions share one credential holder and one session map,
so one cookie name covers them all; the name it publishes is the *web
terminal's* port, because that is what ``OSPREY_WEB_PORT`` already means to its
other readers (the agent's panel tools resolve the terminal through it).

The Theme Lab's own half of this contract lives in
``tests/cli/test_theme_lab_cmd.py`` (its login URL has a second constraint — the
page it announces must not be an exempt one), and ``osprey web``'s in
``tests/cli/test_web_cmd.py``.

Nothing here binds a socket or opens a browser: ``uvicorn.run`` is patched out
and each app factory is replaced by a recorder that captures the environment as
the launcher hands it over.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner

from osprey.interfaces.common_middleware import WEB_PORT_ENV, session_cookie_name


@pytest.fixture(autouse=True)
def _clear_published_port(monkeypatch: pytest.MonkeyPatch):
    """Start each case with no published port, so a pass means the launcher set it."""
    monkeypatch.delenv(WEB_PORT_ENV, raising=False)


class _FactoryRecorder:
    """Stands in for an app factory, recording the port visible when it ran."""

    def __init__(self) -> None:
        self.port_at_construction: str | None = None
        self.calls = 0

    def __call__(self, *args, **kwargs):
        import os

        self.calls += 1
        self.port_at_construction = os.environ.get(WEB_PORT_ENV)
        return SimpleNamespace(state=SimpleNamespace())


def test_artifact_gallery_run_server_publishes_the_port_before_building(monkeypatch):
    """``osprey artifacts web`` (direct serve) settles the cookie name first."""
    from osprey.interfaces.artifacts import app as artifacts_app

    recorder = _FactoryRecorder()
    monkeypatch.setattr(artifacts_app, "create_app", recorder)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    artifacts_app.run_server(host="127.0.0.1", port=8186)

    assert recorder.calls == 1
    assert recorder.port_at_construction == "8186"
    assert session_cookie_name() == session_cookie_name(8186)


def test_ariel_run_web_publishes_the_port_before_building(monkeypatch):
    """ARIEL's launcher settles the cookie name before its factory runs."""
    from osprey.interfaces.ariel import app as ariel_app

    recorder = _FactoryRecorder()
    monkeypatch.setattr(ariel_app, "create_app", recorder)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    ariel_app.run_web(host="127.0.0.1", port=8185)

    assert recorder.calls == 1
    assert recorder.port_at_construction == "8185"


def test_ariel_reload_publishes_the_port_for_the_spawned_worker(monkeypatch):
    """Under ``--reload`` the worker inherits the port through the environment.

    The factory re-runs in a fresh process there, so the publication has to
    happen in this parent — after the spawn is too late, and inside the factory
    would give each reload a different answer.
    """
    from osprey.interfaces.ariel import app as ariel_app

    seen: dict[str, str | None] = {}

    def _fake_run(*args, **kwargs):
        import os

        seen["port"] = os.environ.get(WEB_PORT_ENV)

    monkeypatch.setattr("uvicorn.run", _fake_run)
    ariel_app.run_web(host="127.0.0.1", port=8185, reload=True)

    assert seen["port"] == "8185"


def test_artifacts_cli_reload_publishes_the_port_for_the_spawned_worker(monkeypatch):
    """``osprey artifacts web --reload`` does the same for its worker."""
    from osprey.cli.artifacts_cmd import artifacts

    seen: dict[str, str | None] = {}

    def _fake_run(*args, **kwargs):
        import os

        seen["port"] = os.environ.get(WEB_PORT_ENV)

    monkeypatch.setattr("uvicorn.run", _fake_run)
    result = CliRunner().invoke(artifacts, ["web", "--port", "8187", "--reload"])

    assert result.exit_code == 0, result.output
    assert seen["port"] == "8187"


def test_channel_finder_web_publishes_the_port_before_building(monkeypatch):
    """``osprey channel-finder web`` settles the cookie name before its factory."""
    from osprey.cli import channel_finder_cmd
    from osprey.interfaces.channel_finder import app as channel_finder_app

    recorder = _FactoryRecorder()
    monkeypatch.setattr(channel_finder_app, "create_app", recorder)
    monkeypatch.setattr(channel_finder_cmd, "_setup_config", lambda project: None)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    result = CliRunner().invoke(channel_finder_cmd.channel_finder, ["web", "--port", "8192"])

    assert result.exit_code == 0, result.output
    assert recorder.calls == 1
    assert recorder.port_at_construction == "8192"


def test_channel_finder_login_url_is_printed_unwrapped(monkeypatch):
    """The login URL survives an 80-column terminal intact.

    It is one unbroken token that has to be copy-pasteable; the plain Rich
    console wraps at the terminal width and splits it across lines, so a
    launcher printing it that way hands the operator half a secret. Every other
    launcher goes through ``output.report``, which sets ``soft_wrap``.
    """
    from osprey.cli import channel_finder_cmd
    from osprey.interfaces.channel_finder import app as channel_finder_app

    monkeypatch.setattr(channel_finder_app, "create_app", _FactoryRecorder())
    monkeypatch.setattr(channel_finder_cmd, "_setup_config", lambda project: None)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    with mock.patch.object(channel_finder_cmd, "output", wraps=channel_finder_cmd.output) as spy:
        result = CliRunner().invoke(channel_finder_cmd.channel_finder, ["web", "--port", "8193"])

    assert result.exit_code == 0, result.output
    reported = [call.args[0] for call in spy.report.call_args_list]
    login_lines = [line for line in reported if line.startswith("Open: ")]
    assert len(login_lines) == 1, reported
    # The whole URL on one reported line — no console.print, no wrapping.
    assert "?token=" in login_lines[0]


def test_chat_companions_publish_the_port_before_the_first_launch(monkeypatch, tmp_path):
    """``osprey chat`` names the session cookie before any companion is built.

    Chat is the multi-port shape of this contract: several companion servers in
    one process, sharing one credential holder and one session map. It publishes
    the deployment's web-terminal port once, before the first
    ``ensure_web_server``, so every companion derives the same cookie name — and
    a distinctive one, rather than the bare fallback that collides with any other
    OSPREY server on the host that also fell back.
    """
    from osprey.cli import chat_cmd
    from osprey.utils.workspace import reset_config_cache

    (tmp_path / "config.yml").write_text("web_terminal:\n  port: 8188\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    # An unset compose variable interpolates to the empty string; that counts as
    # absent, not as a published port.
    monkeypatch.setenv(WEB_PORT_ENV, "")

    seen: list[str | None] = []

    def _record(key: str) -> None:
        import os

        seen.append(os.environ.get(WEB_PORT_ENV))

    launcher = SimpleNamespace(_launched=False, _config_reader=lambda: ("127.0.0.1", 8199))
    monkeypatch.setattr("osprey.infrastructure.server_launcher.ensure_web_server", _record)
    monkeypatch.setattr("osprey.infrastructure.server_launcher._launchers", {"artifact": launcher})
    monkeypatch.setattr(
        "osprey.registry.web.FRAMEWORK_WEB_SERVERS",
        {"artifact": SimpleNamespace(name="Artifact gallery")},
    )

    try:
        chat_cmd._launch_companion_servers(tmp_path)

        # Published before the first companion was offered its chance to start,
        # not after the loop — the port has to be there when an app is built.
        assert seen == ["8188"]
        assert session_cookie_name() == session_cookie_name(8188)
    finally:
        reset_config_cache()


def test_chat_does_not_overwrite_a_published_port(monkeypatch, tmp_path):
    """A port already on the environment wins over this deployment's config.

    The per-user container's compose publishes the port that container really
    listens on; re-deriving one from config there would rename the cookie out
    from under the browser nginx just handed a session to.
    """
    from osprey.cli import chat_cmd
    from osprey.utils.workspace import reset_config_cache

    (tmp_path / "config.yml").write_text("web_terminal:\n  port: 8188\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    monkeypatch.setenv(WEB_PORT_ENV, "10101")

    monkeypatch.setattr("osprey.infrastructure.server_launcher.ensure_web_server", lambda key: None)
    monkeypatch.setattr("osprey.infrastructure.server_launcher._launchers", {})
    monkeypatch.setattr("osprey.registry.web.FRAMEWORK_WEB_SERVERS", {})

    try:
        chat_cmd._launch_companion_servers(tmp_path)

        assert session_cookie_name() == session_cookie_name(10101)
    finally:
        reset_config_cache()
