"""Cross-launcher contract for the login URL and the session cookie's scope.

Three entry points bind an OSPREY web surface and print the ``?token=`` URL that
is the operator's only way past the auth gate: ``osprey web``, ``osprey
theme-lab``, and ``osprey chat``, which hosts every companion panel server in its
own process. Each one wires the same two things, and each one used to get at
least one of them wrong on its own.

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

Nothing here binds a socket or opens a browser: ``ensure_web_server`` is replaced
by a recorder that captures the environment as the launcher hands it over.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osprey.interfaces.common_middleware import WEB_PORT_ENV, session_cookie_name


@pytest.fixture(autouse=True)
def _clear_published_port(monkeypatch: pytest.MonkeyPatch):
    """Start each case with no published port, so a pass means the launcher set it."""
    monkeypatch.delenv(WEB_PORT_ENV, raising=False)


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
    from osprey_connectors.workspace import reset_config_cache

    (tmp_path / "config.yml").write_text("web_terminal:\n  port: 8188\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    # An unset compose variable interpolates to the empty string; that counts as
    # absent, not as a published port.
    monkeypatch.setenv(WEB_PORT_ENV, "")

    seen: list[str | None] = []

    def _record(_key: str) -> None:
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
    from osprey_connectors.workspace import reset_config_cache

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
