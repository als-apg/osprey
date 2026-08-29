"""Tests for the session-lifetime resolution in `osprey web`.

`modules.web_terminals.auth.session_lifetime` governs the Max-Age stamped on
every terminal session cookie, and `osprey web` is the ONE launcher for both
deployment shapes: the per-user container's CMD is `osprey web` (its
`config.yml` is baked into the image, so the multi-user compose declares the
value through `OSPREY_TERMINAL_SESSION_LIFETIME` instead), and single-user
`osprey web` reads the key out of its own render.

Two things are pinned here. First the precedence — env declaration beats
config beats the 12-hour default — which mirrors `resolve_bind_host` and
`resolve_web_port` for the same reason those exist. Second the refusal: a
PRESENT but unusable value stops the launch naming both the key and the source
that carried it, rather than silently handing out default-length sessions from
a deployment that believes it shortened them.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import (
    DECLARED_SESSION_LIFETIME_ENV,
    DECLARED_SESSION_STORE_DIR_ENV,
    PID_FILE,
    resolve_session_lifetime,
    web,
)
from tests.cli._lifecycle_build import stub_build

TWELVE_HOURS = 12 * 60 * 60


def _config(value: Any) -> dict[str, Any]:
    """A rendered config carrying *value* at the full key path."""
    return {"modules": {"web_terminals": {"auth": {"session_lifetime": value}}}}


# ---------------------------------------------------------------------------
# resolve_session_lifetime — precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    """env > config > default, with the env read as text."""

    def test_default_when_neither_source_says_anything(self):
        assert resolve_session_lifetime({}, {}) == TWELVE_HOURS

    def test_default_is_the_one_web_auth_defines(self):
        from osprey.interfaces.web_auth import DEFAULT_SESSION_LIFETIME

        assert resolve_session_lifetime({}, {}) == DEFAULT_SESSION_LIFETIME

    def test_config_when_no_env(self):
        assert resolve_session_lifetime(_config(3600), {}) == 3600

    def test_env_wins_over_config(self):
        env = {DECLARED_SESSION_LIFETIME_ENV: "60"}
        assert resolve_session_lifetime(_config(3600), env) == 60

    def test_env_is_stripped_before_it_is_read(self):
        env = {DECLARED_SESSION_LIFETIME_ENV: " 60 "}
        assert resolve_session_lifetime({}, env) == 60

    def test_blank_env_counts_as_absent_and_falls_through_to_config(self):
        env = {DECLARED_SESSION_LIFETIME_ENV: "   "}
        assert resolve_session_lifetime(_config(3600), env) == 3600

    def test_blank_env_and_no_config_takes_the_default(self):
        assert resolve_session_lifetime({}, {DECLARED_SESSION_LIFETIME_ENV: ""}) == TWELVE_HOURS


class TestConfigPathIsReadDefensively:
    """A missing or malformed level on the way down is 'nothing configured'."""

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"modules": None},
            {"modules": {}},
            {"modules": {"web_terminals": None}},
            {"modules": {"web_terminals": {}}},
            {"modules": {"web_terminals": {"auth": None}}},
            {"modules": {"web_terminals": {"auth": {}}}},
            {"modules": {"web_terminals": {"auth": {"session_lifetime": None}}}},
            {"modules": "not-a-mapping"},
            {"modules": {"web_terminals": ["not", "a", "mapping"]}},
            {"modules": {"web_terminals": {"auth": 7}}},
        ],
        ids=[
            "empty",
            "modules-none",
            "modules-empty",
            "web-terminals-none",
            "web-terminals-empty",
            "auth-none",
            "auth-empty",
            "value-none",
            "modules-scalar",
            "web-terminals-list",
            "auth-scalar",
        ],
    )
    def test_absent_anywhere_along_the_path_takes_the_default(self, config):
        assert resolve_session_lifetime(config, {}) == TWELVE_HOURS


# ---------------------------------------------------------------------------
# resolve_session_lifetime — refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    """A PRESENT but unusable value stops the launch, naming its source."""

    @pytest.mark.parametrize("raw", ["0", "-1", "12h", "1.5", "0x10"])
    def test_a_bad_env_declaration_refuses_and_names_the_env_var(self, raw):
        with pytest.raises(click.ClickException) as exc:
            resolve_session_lifetime(_config(3600), {DECLARED_SESSION_LIFETIME_ENV: raw})
        message = str(exc.value)
        assert "modules.web_terminals.auth.session_lifetime" in message
        assert DECLARED_SESSION_LIFETIME_ENV in message
        assert repr(raw) in message
        # The env declaration is what failed, so config must not be blamed.
        assert "build/config.yml" not in message

    @pytest.mark.parametrize("value", [-1, 0, True, False, "12h", "3600", 1.5, [], {}])
    def test_a_bad_config_value_refuses_and_names_the_file(self, value):
        with pytest.raises(click.ClickException) as exc:
            resolve_session_lifetime(_config(value), {})
        message = str(exc.value)
        assert "modules.web_terminals.auth.session_lifetime" in message
        assert "build/config.yml" in message
        assert repr(value) in message
        assert DECLARED_SESSION_LIFETIME_ENV not in message

    def test_a_quoted_number_in_config_is_refused_for_lint_parity(self):
        """The config source takes a real YAML int, never its quoted spelling.

        The multi-user render lint (``_check_auth_session_lifetime``) reports a
        non-int as an ERROR. Accepting ``"3600"`` here would give a deployment
        whose ``osprey build`` lint fails but whose ``osprey web`` starts
        happily — the two shapes must agree on which configs are valid.
        """
        with pytest.raises(click.ClickException) as exc:
            resolve_session_lifetime(_config("3600"), {})
        message = str(exc.value)
        assert "build/config.yml" in message
        assert repr("3600") in message

    def test_the_same_text_is_accepted_from_the_env_carrier(self):
        """Only the environment is text by nature, and only it is parsed."""
        assert resolve_session_lifetime({}, {DECLARED_SESSION_LIFETIME_ENV: "3600"}) == 3600

    def test_a_bad_config_value_is_not_reached_when_the_env_is_good(self):
        """Env authoritative means the baked-in config is never consulted."""
        env = {DECLARED_SESSION_LIFETIME_ENV: "60"}
        assert resolve_session_lifetime(_config("nonsense"), env) == 60


def test_declared_env_constant_equals_the_web_auth_spelling():
    """The launcher's literal and the reader's constant are the same carrier.

    ``web_cmd`` keeps its ``osprey`` imports function-local, so the carrier's
    name is spelled twice. Nothing but this assertion stops the two copies from
    drifting into a launcher that publishes a value the server never reads.
    """
    from osprey.interfaces.web_auth import SESSION_LIFETIME_ENV

    assert DECLARED_SESSION_LIFETIME_ENV == SESSION_LIFETIME_ENV


# ---------------------------------------------------------------------------
# osprey web — the launch
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo what an in-process launch does to the interpreter's globals.

    ``web()`` chdirs into the render and writes ``os.environ`` directly (the
    port publication, the operator secret, and now the session lifetime). All
    three outlive a ``CliRunner`` invocation and would otherwise reach every
    test that runs after this one in the same worker.
    """
    cwd = Path.cwd()
    environ = dict(os.environ)
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _isolate_web_env(monkeypatch):
    """Start each launch test from an environment that declares nothing.

    ``monkeypatch.delenv`` on a key that is already absent records NO undo, so
    a later direct ``os.environ[...] = ...`` from inside ``web()`` would never
    be rolled back. Forcing a ``setenv`` first makes monkeypatch track the key
    and restore its true pre-test state either way.
    """
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, reset_web_credentials

    for key in (
        "OSPREY_CONFIG",
        "OSPREY_PROJECT",
        "OSPREY_WEB_PORT",
        DECLARED_SESSION_LIFETIME_ENV,
        DECLARED_SESSION_STORE_DIR_ENV,
        OPERATOR_SECRET_ENV,
    ):
        monkeypatch.setenv(key, "__unset_by_test_fixture__")
        monkeypatch.delenv(key)
    reset_web_credentials()
    yield
    reset_web_credentials()


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    """``load_osprey_config`` memoizes a builder; each test renders its own."""
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture(autouse=True)
def _no_host_dependencies(monkeypatch):
    """Keep the launch off the developer's machine: no agent CLI, no repo .env."""
    monkeypatch.setattr(
        "osprey.utils.shell_resolver.resolve_shell_command", lambda command: f"/abs/{command}"
    )
    monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)


def _render(repo: Path, session_lifetime: str | None) -> None:
    """Stub a build whose config.yml carries (or omits) the lifetime key."""
    config = "claude_code:\n  provider: anthropic\n"
    if session_lifetime is not None:
        config += (
            f"modules:\n  web_terminals:\n    auth:\n      session_lifetime: {session_lifetime}\n"
        )
    stub_build(repo, config=config)


def _invoke(runner: CliRunner, repo: Path, args: list[str] | None = None):
    """Run the real ``web`` command from *repo*, with the port probe stubbed."""
    os.chdir(repo)
    with patch("socket.socket"):
        return runner.invoke(web, [*(args or []), "--skip-preflight"], catch_exceptions=False)


class TestSingleUserGate:
    """Single-user never runs lint, so the launcher IS the gate on this key."""

    def test_a_negative_lifetime_in_the_render_refuses_the_launch(
        self, runner, lifecycle_repo, monkeypatch
    ):
        _render(lifecycle_repo, "-1")
        launched: list[dict] = []
        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", lambda **kw: launched.append(kw))

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "auth.session_lifetime" in result.output
        assert not launched, "a refused launch must not reach the server"

    def test_a_declared_zero_refuses_before_the_detached_spawn(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """The resolution runs before ``if detach:``, so no child is ever forked."""
        _render(lifecycle_repo, None)
        monkeypatch.setenv(DECLARED_SESSION_LIFETIME_ENV, "0")
        spawned: list[tuple] = []
        monkeypatch.setattr(
            "osprey.cli.web_cmd._start_detached",
            lambda *a, **kw: spawned.append(a),
        )

        result = _invoke(runner, lifecycle_repo, ["--detach"])

        assert result.exit_code != 0
        assert "auth.session_lifetime" in result.output
        assert DECLARED_SESSION_LIFETIME_ENV in result.output
        assert not spawned, "a refused launch must not fork a detached server"


class TestPublication:
    """The launcher publishes a VALIDATED value for whatever serves next."""

    def test_the_configured_lifetime_reaches_the_environment_before_run_web(
        self, runner, lifecycle_repo, monkeypatch
    ):
        _render(lifecycle_repo, "3600")
        seen: dict[str, str | None] = {}

        def _record(**_kwargs):
            seen["published"] = os.environ.get(DECLARED_SESSION_LIFETIME_ENV)

        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", _record)

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert seen["published"] == "3600"

    def test_an_unconfigured_deployment_publishes_the_default(
        self, runner, lifecycle_repo, monkeypatch
    ):
        _render(lifecycle_repo, None)
        seen: dict[str, str | None] = {}

        def _record(**_kwargs):
            seen["published"] = os.environ.get(DECLARED_SESSION_LIFETIME_ENV)

        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", _record)

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert seen["published"] == str(TWELVE_HOURS)

    def test_a_declaration_survives_a_config_that_disagrees(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """The container case: the baked image config must not win over deploy time."""
        _render(lifecycle_repo, "3600")
        monkeypatch.setenv(DECLARED_SESSION_LIFETIME_ENV, "900")
        seen: dict[str, str | None] = {}

        def _record(**_kwargs):
            seen["published"] = os.environ.get(DECLARED_SESSION_LIFETIME_ENV)

        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", _record)

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert seen["published"] == "900"


# ---------------------------------------------------------------------------
# osprey web — the session-store directory publication
# ---------------------------------------------------------------------------


def _render_with_agent_data(repo: Path, base_dir: str | None) -> None:
    """Stub a build whose config.yml relocates (or omits) the agent-data root."""
    config = "claude_code:\n  provider: anthropic\n"
    if base_dir is not None:
        config += f"agent_data:\n  base_dir: {base_dir}\n"
    stub_build(repo, config=config)


class TestStoreDirectoryPublication:
    """The launcher names the DIRECTORY; the file name waits for the port.

    Only the directory is published here because the store file is named for
    the *settled* ``OSPREY_WEB_PORT``, and the busy-port auto-move happens
    later in ``web()``. The name is therefore resolved at credential
    population, in whichever process has actually bound.
    """

    def test_the_default_agent_data_root_reaches_the_environment(
        self, runner, lifecycle_repo, monkeypatch
    ):
        from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

        _render_with_agent_data(lifecycle_repo, None)
        seen: dict[str, str | None] = {}

        def _record(**_kwargs):
            seen["published"] = os.environ.get(DECLARED_SESSION_STORE_DIR_ENV)

        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", _record)

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        expected = lifecycle_repo / DEFAULT_AGENT_DATA_BASE_DIR / "web_terminal"
        assert seen["published"] == str(expected)

    def test_a_relocated_agent_data_root_moves_the_store(
        self, runner, lifecycle_repo, tmp_path, monkeypatch
    ):
        """The store follows ``agent_data.base_dir``, not a hardcoded ``var/``."""
        relocated = tmp_path / "elsewhere" / "agent_data"
        _render_with_agent_data(lifecycle_repo, str(relocated))
        seen: dict[str, str | None] = {}

        def _record(**_kwargs):
            seen["published"] = os.environ.get(DECLARED_SESSION_STORE_DIR_ENV)

        import osprey.interfaces.web_terminal as web_terminal

        monkeypatch.setattr(web_terminal, "run_web", _record)

        result = _invoke(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert seen["published"] == str(relocated / "web_terminal")

    def test_the_detach_parent_binds_no_store_but_the_child_inherits_one(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """The parent mints and prints; only the child serves, so only it persists.

        A parent holding the real store would restore the deployment's sessions
        into a process that never answers a request, and any later save from it
        would rewrite the file underneath the child that does.
        """
        from osprey.interfaces.web_auth import get_web_credentials
        from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

        _render_with_agent_data(lifecycle_repo, None)
        monkeypatch.setattr("osprey.cli.web_cmd._read_pid", lambda repo_root: None)
        monkeypatch.setattr(
            "osprey.cli.web_cmd._wait_for_server", lambda host, port, proc, **kw: True
        )

        child_env: dict[str, str] = {}

        class _FakeProc:
            pid = 4242

            def poll(self):
                return None

        def _fake_popen(cmd, **_kwargs):
            # Popen inherits os.environ, so the environment at spawn time IS
            # the child's environment.
            child_env.update(os.environ)
            return _FakeProc()

        monkeypatch.setattr("osprey.cli.web_cmd.subprocess.Popen", _fake_popen)

        result = _invoke(runner, lifecycle_repo, ["--detach"])

        assert result.exit_code == 0, result.output
        assert get_web_credentials().store is None, (
            "the detach parent must bind no store: it mints and prints, it never serves"
        )
        expected = lifecycle_repo / DEFAULT_AGENT_DATA_BASE_DIR / "web_terminal"
        assert child_env[DECLARED_SESSION_STORE_DIR_ENV] == str(expected)


def test_declared_store_dir_constant_equals_the_web_auth_spelling():
    """The launcher's literal and the reader's constant are the same carrier.

    Same drift guard as the lifetime carrier: ``web_cmd`` spells the name
    itself, so nothing but this assertion stops the launcher from publishing a
    directory the serving process never looks for.
    """
    from osprey.interfaces.web_auth import SESSION_STORE_DIR_ENV

    assert DECLARED_SESSION_STORE_DIR_ENV == SESSION_STORE_DIR_ENV


# ---------------------------------------------------------------------------
# osprey web sessions clear
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """A port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _render_on_port(repo: Path, port: int) -> None:
    """Stub a build whose config.yml pins the terminal to *port* on loopback."""
    stub_build(
        repo,
        config=(
            "claude_code:\n  provider: anthropic\n"
            f"web_terminal:\n  host: 127.0.0.1\n  port: {port}\n"
        ),
    )


def _store_dir(repo: Path) -> Path:
    """The directory the launcher publishes for this repo's session store."""
    from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

    return repo / DEFAULT_AGENT_DATA_BASE_DIR / "web_terminal"


def _write_store(repo: Path, name: str, digests: int) -> Path:
    """Write a store file holding *digests* live sessions, as the server would."""
    path = _store_dir(repo)
    path.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 3600
    payload = {
        "v": 1,
        "sessions": {f"{index:064x}": deadline for index in range(digests)},
    }
    store = path / name
    store.write_text(json.dumps(payload), encoding="utf-8")
    return store


def _write_pid_file(repo: Path) -> Path:
    """Leave the PID file a ``--detach`` launch would have left.

    Holds THIS interpreter's pid, which is certainly alive, so ``_read_pid``
    reports a running server rather than discarding the file as stale.
    """
    pid_path = repo / PID_FILE
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    return pid_path


def _clear(runner: CliRunner, repo: Path, args: list[str] | None = None):
    """Run ``osprey web sessions clear`` from *repo*."""
    os.chdir(repo)
    return runner.invoke(web, ["sessions", "clear", *(args or [])], catch_exceptions=False)


def _flat(text: str) -> str:
    """Collapse the renderer's wrapping so an assertion reads the sentence.

    ``output.fail``/``output.warn`` wrap to the console width, which is not the
    same everywhere the suite runs; a phrase split across two lines is still
    the phrase the operator reads.
    """
    return " ".join(text.split())


class TestSessionsClearRefusesALiveServer:
    """A running server would undo the delete, so the verb declines to pretend."""

    def test_a_listener_on_the_resolved_port_refuses_and_names_the_stop_verb(
        self, runner, lifecycle_repo
    ):
        """A FOREGROUND server writes no PID file, so the socket is the probe."""
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen(5)

            result = _clear(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "osprey web stop" in _flat(result.output)
        assert store.is_file(), "a refused clear must leave the store untouched"

    def test_force_clears_anyway_and_says_what_force_does_not_do(self, runner, lifecycle_repo):
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen(5)

            result = _clear(runner, lifecycle_repo, ["--force"])

        assert result.exit_code == 0, result.output
        assert not store.exists()
        assert "Dropped 2 persisted session(s)." in _flat(result.output)
        # The caveat is the whole point of allowing --force at all: the running
        # server keeps serving the sessions it already holds.
        caveat = _flat(result.output)
        assert "keeps its live sessions and warm terminals in memory" in caveat
        assert "only stops them surviving the NEXT restart" in caveat

    def test_a_detached_servers_pid_file_refuses_with_nothing_listening(
        self, runner, lifecycle_repo
    ):
        """The other half of the liveness read, on its own.

        A detached server that is mid-restart, wedged, or simply not answering
        yet still owns its sessions -- the PID file is the handle on it, and it
        has to refuse without any help from the socket probe.
        """
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)
        _write_pid_file(lifecycle_repo)

        result = _clear(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "osprey web stop" in _flat(result.output)
        assert store.is_file(), "a refused clear must leave the store untouched"

    def test_force_clears_past_a_pid_file_with_the_same_caveat(self, runner, lifecycle_repo):
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)
        _write_pid_file(lifecycle_repo)

        result = _clear(runner, lifecycle_repo, ["--force"])

        assert result.exit_code == 0, result.output
        assert not store.exists()
        assert "Dropped 2 persisted session(s)." in _flat(result.output)
        caveat = _flat(result.output)
        assert "keeps its live sessions and warm terminals in memory" in caveat
        assert "only stops them surviving the NEXT restart" in caveat


class TestSessionsClearWithNoServer:
    """With nothing serving, the delete means what it says."""

    def test_it_reports_the_digests_it_dropped_and_removes_the_file(self, runner, lifecycle_repo):
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)

        result = _clear(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert "Dropped 2 persisted session(s)." in _flat(result.output)
        assert not store.exists()

    def test_every_port_keyed_store_goes_not_only_this_one(self, runner, lifecycle_repo):
        """One deployment can have served on several ports; all of them clear."""
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        mine = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)
        other = _write_store(lifecycle_repo, "sessions-9999.json", 1)
        bare = _write_store(lifecycle_repo, "sessions.json", 3)

        result = _clear(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert "Dropped 6 persisted session(s)." in _flat(result.output)
        assert not mine.exists() and not other.exists() and not bare.exists()

    def test_an_unreadable_store_counts_nothing_but_is_still_deleted(self, runner, lifecycle_repo):
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _store_dir(lifecycle_repo) / f"sessions-{port}.json"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("{ truncated", encoding="utf-8")

        result = _clear(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert "Dropped 0 persisted session(s)." in _flat(result.output)
        assert not store.exists()

    def test_a_deployment_that_never_persisted_anything_is_not_an_error(
        self, runner, lifecycle_repo
    ):
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        assert not _store_dir(lifecycle_repo).exists()

        result = _clear(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert "Dropped 0 persisted session(s)." in _flat(result.output)

    def test_the_group_position_of_repo_reaches_the_subcommand(
        self, runner, lifecycle_repo, tmp_path
    ):
        """``osprey web --repo X sessions clear`` is the same request as the tail form."""
        port = _free_port()
        _render_on_port(lifecycle_repo, port)
        store = _write_store(lifecycle_repo, f"sessions-{port}.json", 2)

        elsewhere = tmp_path / "unrelated"
        elsewhere.mkdir()
        os.chdir(elsewhere)
        result = runner.invoke(
            web,
            ["--repo", str(lifecycle_repo), "sessions", "clear"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Dropped 2 persisted session(s)." in _flat(result.output)
        assert not store.exists()
