"""Tests for the credential carriers ``osprey chat`` sets up (Task 2.4).

``osprey chat`` hosts its companion web servers in daemon threads of its own
process and then spawns the agent CLI as a child. Two credential facts must
hold, and this module pins both:

* :func:`~osprey.cli.chat_cmd._launch_companion_servers` announces a per-companion
  ``?token=`` login URL (the operator's only way past each companion's live auth
  middleware) and leaves neither credential carrier behind in ``os.environ``.
* :func:`~osprey.cli.chat_cmd.chat` spawns the agent with a deliberately built
  environment that STRIPS the operator secret (``OSPREY_TERMINAL_SECRET`` — the
  escalation this task closes) and re-adds only the weak panel token.

The companion tests drive ``_launch_companion_servers`` directly with a fake
launcher; the env test drives the whole verb through the same lifecycle-repo
fixtures the sibling ``test_chat_verb`` suite uses.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from osprey.cli.chat_cmd import _launch_companion_servers, chat
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo the process-global state a launch mutates.

    ``chat`` chdirs into ``build/`` and overlays env; ``_launch_companion_servers``
    and ``mint_and_announce`` publish credential vars into ``os.environ``. All are
    process-global and would otherwise reach every later test in the worker.
    """
    cwd = Path.cwd()
    environ = dict(os.environ)
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _no_managed_policy(monkeypatch: pytest.MonkeyPatch):
    """Pin the managed-policy scan to "no conflicts" so a launch does not refuse
    for a machine-local policy file none of these tests is about."""
    monkeypatch.setattr(
        "osprey.build.claude_code_resolver.detect_managed_policy_conflicts",
        lambda paths=None: {},
    )


def _fake_companion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    launched: bool,
    host: str = "127.0.0.1",
    port: int = 8199,
    name: str = "Artifact Gallery",
) -> None:
    """Make ``_launch_companion_servers`` see exactly one registered companion.

    Patches the three names it imports at call time — ``ensure_web_server`` (a
    no-op here), ``_launchers`` (holding one launcher whose ``_launched`` and
    ``_config_reader`` the caller controls), and ``FRAMEWORK_WEB_SERVERS`` (one
    definition supplying the display name).
    """
    launcher = SimpleNamespace(_launched=launched, _config_reader=lambda: (host, port))
    monkeypatch.setattr("osprey.infrastructure.server_launcher.ensure_web_server", lambda key: None)
    monkeypatch.setattr("osprey.infrastructure.server_launcher._launchers", {"artifact": launcher})
    monkeypatch.setattr(
        "osprey.registry.web.FRAMEWORK_WEB_SERVERS",
        {"artifact": SimpleNamespace(name=name)},
    )


class TestCompanionCarriers:
    """What ``_launch_companion_servers`` announces and publishes."""

    def test_started_companion_announces_token_url(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_SECRET", raising=False)
        monkeypatch.delenv("OSPREY_PANEL_TOKEN", raising=False)
        _fake_companion(monkeypatch, launched=True, host="127.0.0.1", port=8199)

        started = _launch_companion_servers(Path("/no-such-build"))

        assert len(started) == 1
        name, url = started[0]
        assert name == "Artifact Gallery"
        # A login URL carrying the operator secret, not a bare address:
        # the browser exchanges the ?token= for a session cookie.
        assert url.startswith("http://127.0.0.1:8199/?token=")

    def test_the_launch_leaves_no_credential_carrier_behind(self, monkeypatch):
        """Neither carrier is in ``os.environ`` once the launch has settled.

        ``mint_and_announce`` re-publishes the operator secret on every call, and
        the last one in the loop has no app construction behind it to close the
        carrier again — the companions' ``create_app`` calls run in daemon
        threads and close it whenever they happen to finish. The explicit close
        at the end of the launch is what makes the end state deterministic
        instead of a matter of thread timing.
        """
        monkeypatch.delenv("OSPREY_TERMINAL_SECRET", raising=False)
        monkeypatch.delenv("OSPREY_PANEL_TOKEN", raising=False)
        _fake_companion(monkeypatch, launched=True, host="127.0.0.1", port=8199)

        started = _launch_companion_servers(Path("/no-such-build"))

        from osprey.interfaces.web_auth import get_web_credentials

        assert "OSPREY_TERMINAL_SECRET" not in os.environ
        assert "OSPREY_PANEL_TOKEN" not in os.environ

        # Closing the carriers does not invalidate what was announced: the
        # credentials live in the holder, which is what the companions
        # authenticate against and what chat() reads the panel token from.
        credentials = get_web_credentials()
        _, url = started[0]
        assert url.endswith(f"?token={credentials.operator_secret}")
        assert credentials.panel_token

    def test_no_started_companion_publishes_no_panel_token(self, monkeypatch):
        monkeypatch.delenv("OSPREY_PANEL_TOKEN", raising=False)
        _fake_companion(monkeypatch, launched=False)

        started = _launch_companion_servers(Path("/no-such-build"))

        assert started == []
        # Nothing to talk to, so no credential is published.
        assert "OSPREY_PANEL_TOKEN" not in os.environ


class TestAgentChildEnvironment:
    """The environment the agent CLI child is spawned with."""

    def test_operator_secret_is_stripped_and_panel_token_is_re_added(
        self, runner, monkeypatch, lifecycle_repo
    ):
        from osprey.interfaces.web_auth import reset_web_credentials

        stub_build(lifecycle_repo)
        monkeypatch.delenv("OSPREY_TERMINAL_BIND_HOST", raising=False)
        # The holder must be unsettled so it populates from what fake_launch
        # publishes below; otherwise this test would be asserting against
        # whichever secrets an earlier test in this worker happened to mint.
        reset_web_credentials()

        # Simulate the companions having minted and published both credentials
        # into os.environ, exactly as the real _launch_companion_servers does.
        def fake_launch(project_dir):
            os.environ["OSPREY_TERMINAL_SECRET"] = "operator-secret-value"
            os.environ["OSPREY_PANEL_TOKEN"] = "panel-token-value"
            return [("Artifact Gallery", "http://127.0.0.1:8199/?token=operator-secret-value")]

        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", fake_launch)

        captured: dict[str, object] = {}

        def fake_run(argv, *args, **kwargs):
            captured["env"] = kwargs.get("env")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        env = captured["env"]
        # chat must hand the agent an EXPLICIT env, never inherit os.environ (which
        # now holds the operator secret the companions published for themselves).
        assert isinstance(env, dict)
        # The operator secret must never reach the agent — possession would let
        # agent-run code authenticate as the server that launched it.
        assert "OSPREY_TERMINAL_SECRET" not in env
        # The panel token is the weak credential the agent legitimately needs to
        # make panel-tier calls to the in-process companions; it is re-added.
        assert env.get("OSPREY_PANEL_TOKEN") == "panel-token-value"

    def test_the_panel_token_comes_from_the_holder_not_the_carrier(
        self, runner, monkeypatch, lifecycle_repo
    ):
        """A carrier that has already been closed again does not disarm the child.

        Every companion's ``create_app`` runs in its own daemon thread and calls
        ``web_auth.close_env_carriers``, which pops ``OSPREY_PANEL_TOKEN`` back
        out of ``os.environ``. A companion whose factory finishes after
        ``_launch_companion_servers`` published it would therefore un-publish it
        under this launch — so ``chat`` reads the credential holder, which is
        the same object those companions authenticate against.
        """
        from osprey.interfaces import web_auth

        stub_build(lifecycle_repo)
        holder = web_auth.WebCredentials(
            operator_secret="holder-operator-secret", panel_token="holder-panel-token"
        )
        monkeypatch.setattr(web_auth, "_CREDENTIALS", holder)

        def fake_launch(project_dir):
            # A companion started and announced — but by the time chat() runs,
            # the carrier has been closed again by an app construction.
            os.environ.pop("OSPREY_PANEL_TOKEN", None)
            return [("Artifact Gallery", "http://127.0.0.1:8199/?token=x")]

        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", fake_launch)

        captured: dict[str, object] = {}

        def fake_run(argv, *args, **kwargs):
            captured["env"] = kwargs.get("env")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        env = captured["env"]
        assert env.get("OSPREY_PANEL_TOKEN") == "holder-panel-token"
        assert "OSPREY_TERMINAL_SECRET" not in env

    def test_no_companion_means_no_panel_token_for_the_child(
        self, runner, monkeypatch, lifecycle_repo
    ):
        """With no panel host to call, the agent is handed no credential at all."""
        from osprey.interfaces import web_auth

        stub_build(lifecycle_repo)
        monkeypatch.setattr(
            web_auth,
            "_CREDENTIALS",
            web_auth.WebCredentials(operator_secret="s", panel_token="holder-panel-token"),
        )
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])

        captured: dict[str, object] = {}

        def fake_run(argv, *args, **kwargs):
            captured["env"] = kwargs.get("env")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "OSPREY_PANEL_TOKEN" not in captured["env"]

    def test_agent_child_env_is_explicit_even_with_no_companions(
        self, runner, monkeypatch, lifecycle_repo
    ):
        stub_build(lifecycle_repo)
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])
        # A stray operator secret in the ambient environment must still not reach
        # the agent, even when no companion published one.
        monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "stray-secret")

        captured: dict[str, object] = {}

        def fake_run(argv, *args, **kwargs):
            captured["env"] = kwargs.get("env")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        env = captured["env"]
        assert isinstance(env, dict)
        assert "OSPREY_TERMINAL_SECRET" not in env
