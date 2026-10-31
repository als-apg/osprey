"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from osprey.services.bluesky_bridge import session_dir as _session_plan_dir
from osprey_connectors_tests._harness import _guards as _harness
from osprey_connectors_tests._harness import _repo_cleanliness

# The guards the connectors suite shares, imported by name so their autouse
# fixtures take their usual places in this conftest's alphabetical order. See
# osprey_connectors_tests._harness._guards for why they are not a registered plugin.
from osprey_connectors_tests._harness._guards import (  # noqa: F401 - fixtures and hooks, used by name
    _GUARD_KNOWN_LIMIT,
    _clear_dotenv_timezone,
    _first_seen_clause,
    _is_real_deployment_item,
    _no_host_timezone_leak,
    agent_data_never_the_checkout,
    control_context_root,
    no_agent_data_in_the_repo,
    no_spool_in_the_repo,
    pytest_collection_finish,
    pytest_runtest_setup,
    pytest_xdist_make_scheduler,
    reset_second_real_block_warning,
    restore_cwd,
    restore_environ,
    restore_environ_per_module,
    restore_root_logging,
    session_posture_leak_guard,
    write_control_context,
    write_server_report,
)
from osprey_connectors_tests._harness._ports import port_reserver
from tests import _scoped_temp
from tests._agent_cli import STUB_AGENT_CLI
from tests._builds import (  # noqa: F401 - the session's shared builds
    built_control_assistant,
    offline_build_env,
)

# ===================================================================
# Provider gateway stand-in
# ===================================================================

#: The endpoint every test uses when a gateway-backed provider needs a URL.
#:
#: Providers that front a site's own gateway (``als-apg``) ship no default
#: endpoint, so a test that exercises one has to supply the URL the same way a
#: deployment does. One shared value means no test states a real deployment's
#: host, and a reader can tell "this is the fixture endpoint" at a glance.
GATEWAY_BASE_URL = "https://gateway.example.org/v1"

#: :data:`GATEWAY_BASE_URL` as Claude Code receives it — no trailing ``/v1``,
#: which the resolver strips because Claude Code appends ``/v1/messages``.
GATEWAY_ORIGIN = "https://gateway.example.org"


@pytest.fixture(autouse=True, scope="function")
def agent_cli_resolved_per_test():
    """Forget which Claude Code a project's headless runs execute, around every test.

    The executable is resolved once per process and kept, keyed by project
    directory. Tests share directories such as ``/tmp`` and replace the lookup
    in different ways, so a kept answer would make one test's result depend on
    which test ran before it in the same worker.
    """
    from osprey.agent_runner import launcher

    launcher._project_agent_cli.cache_clear()
    yield
    launcher._project_agent_cli.cache_clear()


@pytest.fixture
def real_agent_cli():
    """Opt a test out of :func:`stub_agent_cli`: it resolves the real Claude Code."""


@pytest.fixture
def reserve_port() -> Iterator[Callable[[], int]]:
    """A harness ``_ports.port_reserver`` call, its reservations held until the test ends."""
    with port_reserver() as reserve:
        yield reserve


@pytest.fixture(autouse=True, scope="function")
def stub_agent_cli(request, monkeypatch):
    """Point every project's headless runs at ``STUB_AGENT_CLI``.

    A headless run refuses to start unless an installed ``claude`` reports the
    version the project pins. Most tests only build the options of a run, or
    drive a faked one, and must pass on a host with no Claude Code at all. A
    test that needs the real lookup requests ``real_agent_cli``; the end-to-end
    tree always gets it.
    """
    if "real_agent_cli" in request.fixturenames or "e2e" in request.node.path.parts:
        return
    from osprey.agent_runner import launcher

    monkeypatch.setattr(launcher, "project_agent_cli_path", lambda _project_dir: STUB_AGENT_CLI)
    # What production resolves on the default pin, so a stubbed run never probes.
    monkeypatch.setattr(
        launcher,
        "project_thinking_display",
        lambda _project_dir: launcher.ThinkingDisplay("summarized", True),
    )


@pytest.fixture(autouse=True, scope="function")
def writable_plan_dir_per_test(tmp_path):
    """Resolve the Bluesky bridge's writable plan directory under this test's ``tmp_path``.

    Every test that reaches the bridge's plan loader or session-plan routes
    resolves ``BLUESKY_SESSION_PLAN_DIR`` to a path under its own ``tmp_path``,
    which is created only if something resolves it. ``tmp_path`` is already
    requested for every test by :func:`_isolate_spool_zone`, so this adds no
    directory. Without it, resolution falls back to the in-package default and
    the run leaves a ``plans_session/`` directory inside the checkout, which
    :func:`no_authored_plans_in_the_package` fails.

    A test that sets the variable itself wins: this fixture is set up before a
    module's own, so the test's value lands on top, and its teardown restores
    the value set here.

    It patches on its own ``MonkeyPatch.context()`` rather than the
    ``monkeypatch`` fixture, because a test's ``monkeypatch.undo()`` reverts
    every patch on that fixture. With the shared one, a test lifting its own
    patch would also drop this one and send the rest of the test to the
    in-package default.

    It is set up before the shared ``monkeypatch`` and torn down after it:
    :func:`_isolate_ahead_of_monkeypatch`, the first function-scoped fixture to request
    ``monkeypatch``, requests this one ahead of it. A test that sets the
    variable on ``monkeypatch`` records the value set here as the one to put
    back, so that undo has to run while this patch is still in place; run
    after it, the undo would write this test's ``tmp_path`` back into the
    environment for whatever resolves between tests.
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv(_PLAN_DIR_ENV, str(tmp_path / "bluesky-plans"))
        yield


#: The variable that names the Bluesky bridge's writable plan directory, read
#: from the bridge so it is spelled once.
_PLAN_DIR_ENV = _session_plan_dir._SESSION_PLAN_DIR_ENV

#: The bridge's in-package plan directory — the marker
#: :func:`no_authored_plans_in_the_package` watches.
_PLAN_DIR_MARKER = _session_plan_dir._DEFAULT_SESSION_PLAN_DIR

#: The same baseline for the in-package plan directory, taken at the same moment.
_PLAN_DIR_BASELINE = _repo_cleanliness.snapshot(_PLAN_DIR_MARKER)


#: The same bound for the bridge's in-package plan directory.
_PLAN_DIR_FIRST_SEEN: str | None = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Bound the Bluesky plan directory's leak, then run the harness's teardown.

    A conftest carries one implementation of a hook, so the harness's is
    delegated to; the plan directory is the framework's alone.
    """
    global _PLAN_DIR_FIRST_SEEN
    if (
        not _harness._REAL_DEPLOYMENT_LANE_RAN
        and _PLAN_DIR_FIRST_SEEN is None
        and _repo_cleanliness.what_this_run_did(_PLAN_DIR_BASELINE, _PLAN_DIR_MARKER)
    ):
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        _PLAN_DIR_FIRST_SEEN = f"{item.nodeid} (worker {worker})"
    yield from _harness.pytest_runtest_teardown(item)


@pytest.fixture(autouse=True, scope="session")
def no_project_venv_outlives_the_session(tmp_path_factory):
    """Fail the session if a real project venv is still in the temp tree.

    A real ``osprey build`` installs the whole dependency tree into
    ``build/.venv``, gigabytes per build. A fixture's tree is removed when its
    scope ends, but a venv inside it costs gigabytes while the scope is live,
    so a fixture that runs a real build removes the venv itself, as soon as
    nothing reads it any more.

    A real venv is told from a stub by its interpreter: ``uv venv`` links
    ``bin/python`` to the base interpreter, while the tests that fake a venv
    write a plain file there.
    """
    yield
    basetemp = tmp_path_factory.getbasetemp()
    survivors: list[str] = []
    for directory, subdirectories, _files in os.walk(basetemp):
        if ".venv" not in subdirectories:
            continue
        # Never descend into a venv: its site-packages is the bulk of the tree.
        subdirectories.remove(".venv")
        venv = Path(directory) / ".venv"
        if (venv / "bin" / "python").is_symlink():
            survivors.append(str(venv))
    if survivors:
        pytest.fail(
            "a real project venv outlived its fixture: "
            + ", ".join(sorted(survivors))
            + "\nRemove it in the fixture that ran the build, once nothing reads it."
        )


@pytest.fixture(autouse=True, scope="session")
def no_authored_plans_in_the_package():
    """Fail the session if the suite created the bridge's in-package ``plans_session/``.

    The twin of :func:`no_agent_data_in_the_repo` and
    :func:`no_spool_in_the_repo`, watching the directory
    ``resolve_session_plan_dir()`` falls back to when
    ``BLUESKY_SESSION_PLAN_DIR`` is unset. That directory sits inside the
    package tree of this checkout, so a run that resolves to it leaves an
    empty directory behind that ``git status`` never mentions.

    Keyed on what the run did, the same way as the twins: a directory already
    present from an earlier run is not a failure, but an entry this run adds to
    it is. The real-deployment lanes are exempt once such an item has run in
    this worker, for the twins' reason.
    """
    marker = _PLAN_DIR_MARKER
    yield
    if _harness._REAL_DEPLOYMENT_LANE_RAN:
        return
    clause = _repo_cleanliness.what_this_run_did(_PLAN_DIR_BASELINE, marker)
    if clause:
        raise AssertionError(
            f"the test run {clause} — something resolved the Bluesky bridge's "
            "writable plan directory to its in-package default. A test that reaches "
            "the bridge's plan loader or session-plan routes gets a per-test directory "
            "from writable_plan_dir_per_test, so a leak means something bypassed it: "
            "a thread that outlived its test, a subprocess started with a scrubbed "
            "environment, or a test that deleted the variable."
            f"{_first_seen_clause(_PLAN_DIR_FIRST_SEEN)}"
            f"{_GUARD_KNOWN_LIMIT}"
        )


@pytest.fixture(autouse=True)
def render_worker_ends_with_its_test():
    """Kill the picture render worker a test left cached, when that test ends.

    ``osprey.imaging.render`` keeps one worker subprocess per process, so a test
    that renders a picture without closing the worker hands it to whichever test
    runs next on the same xdist worker. A test that asserts no worker was
    spawned then sees a pid it never started. A test that never imported the
    module pays nothing.
    """
    yield
    render = sys.modules.get("osprey.imaging.render")
    if render is not None and render.worker_pid() is not None:
        render._forget_worker()


@pytest.fixture(autouse=True)
def _isolate_ahead_of_monkeypatch(
    writable_plan_dir_per_test,
    _isolate_spool_zone,
    monkeypatch,
):
    """Set the plan-dir and spool isolation up before the shared ``monkeypatch``.

    The first function-scoped autouse fixture to request ``monkeypatch``, so
    :func:`writable_plan_dir_per_test` and :func:`_isolate_spool_zone` wrap it
    on both sides; the plan-dir fixture's docstring gives the reason.
    """


# ===================================================================
# The record spool
# ===================================================================

#: The identity a test writes as unless it asks for another: a roster terminal.
SPOOL_TERMINAL_IDENTITY = "alice"
#: A dispatch worker, the one source class that may record a ``run``.
SPOOL_DISPATCH_IDENTITY = "dispatch-worker-1"
#: A bridge, the one source class that may record a ``bridge_turn``.
SPOOL_BRIDGE_IDENTITY = "teams-bridge"


class SpoolZone:
    """A spool root of a test's (or a module's) own, and the record it feeds.

    ``OSPREY_SPOOL_DIR`` names the writing identity's own directory under
    :attr:`root`; :meth:`spool_for` moves it to another identity's.
    :meth:`recorded_entries` consumes the whole root into an embedded backend
    kept for the zone's lifetime, so a line is reported once however often it
    is called, and fails the test on any line the consumer refused.
    """

    terminal = SPOOL_TERMINAL_IDENTITY
    dispatch = SPOOL_DISPATCH_IDENTITY
    bridge = SPOOL_BRIDGE_IDENTITY

    def __init__(self, root: Path, record_dir: Path, patched: pytest.MonkeyPatch | None):
        self.root = root
        self._record_dir = record_dir
        self._patched = patched
        self._identities: list[str] = [self.terminal, self.dispatch, self.bridge]
        self._backend = None
        self._refused: list[str] = []
        self.spool_for()

    def spool_for(self, identity: str | None = None) -> Path:
        """Write as *identity* (the terminal by default); its own directory."""
        identity = identity or self.terminal
        if identity not in self._identities:
            self._identities.append(identity)
        directory = self.root / identity
        if self._patched is not None:
            from osprey.agent_record.contract import SPOOL_DIR_ENV

            self._patched.setenv(SPOOL_DIR_ENV, str(directory))
        return directory

    def config(self) -> dict:
        """A deployment config that declares every identity this zone has written as.

        Dispatch workers and bridges are declared as deployed services; every
        other identity is a roster terminal.
        """
        from osprey.agent_record import identities

        workers = 0
        terminals: list[str] = []
        for identity in self._identities:
            match = identities._DISPATCH_WORKER_RE.match(identity)
            if match:
                workers = max(workers, int(match[1]))
            elif identity not in identities.BRIDGE_IDENTITIES.values():
                terminals.append(identity)
        return {
            "project_name": "spool-zone",
            "modules": {"web_terminals": {"enabled": True, "users": terminals}},
            "deployed_services": ["dispatch_worker", *identities.BRIDGE_IDENTITIES],
            "services": {"dispatch_worker": {"worker_count": max(workers, 1)}},
        }

    def recorded_entries(self, kind: str | None = None) -> list:
        """Every entry the root has produced so far, optionally of one *kind*.

        Fails the test on a refused line (now or in any earlier call) and on a
        backend error.
        """
        from osprey.agent_record import contract, writer
        from osprey.agent_record.backends.embedded import EmbeddedBackend
        from osprey.agent_record.ingest.spool import SpoolConsumer

        writer.seal_all()
        self.root.mkdir(parents=True, exist_ok=True)
        if self._backend is None:
            self._backend = EmbeddedBackend(self._record_dir / "record.duckdb")
        consumer = SpoolConsumer(self._backend, self.root, config=self.config())
        report = consumer.consume()
        if report.backend_error:
            pytest.fail(f"the spool could not be consumed: {report.backend_error}")
        for source in consumer.health().sources.values():
            self._refused.extend(
                f"{source.identity}/{line.segment}: {line.kind or '?'} line "
                f"{line.entry_id or '?'}: {line.reason}"
                for line in source.refused
            )
        if self._refused:
            pytest.fail("the spool consumer refused lines:\n  " + "\n  ".join(self._refused))
        kinds = frozenset({kind}) if kind else None
        return self._backend.select(contract.ScopeFilter(kinds=kinds), contract.Query())

    def close(self) -> None:
        """Seal what the writer holds open and release the record file."""
        writer = sys.modules.get("osprey.agent_record.writer")
        if writer is not None:
            writer.seal_all()
        if self._backend is not None:
            self._backend.close()
            self._backend = None


@contextmanager
def _spool_zone(node, root: Path, record_dir: Path):
    """A :class:`SpoolZone` held for the ``with`` body; real deployments are left alone."""
    if _is_real_deployment_item(node):
        zone = SpoolZone(root, record_dir, None)
        try:
            yield zone
        finally:
            zone.close()
        return
    with pytest.MonkeyPatch.context() as patched:
        zone = SpoolZone(root, record_dir, patched)
        try:
            yield zone
        finally:
            zone.close()


@pytest.fixture(autouse=True)
def _isolate_spool_zone(request, tmp_path):
    """Give every test a spool root of its own and point the writer into it.

    The record writer resolves ``OSPREY_SPOOL_DIR`` on every call and, when it
    is unset, falls back to ``<project root>/var/spool/<identity>`` — the
    checkout, which :func:`no_spool_in_the_repo` refuses. Suite-wide because
    the modules that need it are exactly the ones whose authors would not think
    to ask: any app a test drives writes audit entries and other record lines,
    and the interfaces, dispatch, MCP and integration trees all hold app
    fixtures of that shape. Set as an environment
    variable, so a subprocess the test starts writes into the same root.

    Its own ``MonkeyPatch.context()``, and requested by
    :func:`_isolate_ahead_of_monkeypatch` ahead of ``monkeypatch``, for the reasons
    :func:`writable_plan_dir_per_test` gives: a test's ``monkeypatch.undo()``
    cannot lift it, and a test that unsets the variable gets it back at
    teardown. The real-deployment lanes are left alone: they record against
    this checkout on purpose.

    Its module-scoped twin, :func:`_isolate_module_spool_zone`, holds a root
    for what a module-scoped fixture builds; this one sits on top of it for
    the test body.
    """
    base = tmp_path / "spool-zone"
    with _spool_zone(request.node, base / "spool", base / "record") as zone:
        yield zone


@pytest.fixture
def spool_zone(_isolate_spool_zone: SpoolZone) -> SpoolZone:
    """The test's :class:`SpoolZone`, by a public name."""
    return _isolate_spool_zone


@pytest.fixture
def spool_for(_isolate_spool_zone: SpoolZone):
    """``spool_for(identity=None) -> Path``: write as *identity* from now on.

    Defaults to :data:`SPOOL_TERMINAL_IDENTITY`; ``spool_zone.dispatch`` and
    ``spool_zone.bridge`` name identities whose ``run`` and ``bridge_turn``
    lines the consumer accepts.
    """
    return _isolate_spool_zone.spool_for


@pytest.fixture
def recorded_entries(_isolate_spool_zone: SpoolZone):
    """``recorded_entries(kind=None) -> list[Entry]`` over the test's spool root."""
    return _isolate_spool_zone.recorded_entries


@pytest.fixture(autouse=True, scope="module")
def _isolate_module_spool_zone(request, tmp_path_factory):
    """Hold a spool root for a whole test module.

    The twin of :func:`_isolate_spool_zone` for an app built by a
    module-scoped fixture, which pytest sets up before any per-test fixture:
    such an app enters its lifespan and serves requests before the per-test
    root exists. Module-scoped rather than session-scoped so a root is never
    shared past the module that needed it. Lines written
    while such a fixture is set up or torn down land here; lines written
    during a test body land in that test's own root.
    """
    base = tmp_path_factory.mktemp("module-spool-zone")
    with _spool_zone(request.node, base / "spool", base / "record") as zone:
        yield zone


@pytest.fixture(scope="module")
def module_spool_zone(_isolate_module_spool_zone: SpoolZone) -> SpoolZone:
    """The module's :class:`SpoolZone`, by a public name."""
    return _isolate_module_spool_zone


# ===================================================================
# Auto-reset Registry and Config Between Tests
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_state_between_tests():
    """Auto-reset registry and config before each test to ensure isolation.

    This prevents state leakage between tests by:
    - Resetting the registry
    - Clearing config caches (utils.config and utils.workspace)

    Config-related environment variables are handled by ``restore_environ``.
    """
    # Reset before test
    from osprey.registry import reset_registry
    from osprey_connectors.workspace import reset_config_cache

    reset_registry()
    reset_config_cache()

    yield

    # Reset after test
    reset_registry()
    reset_config_cache()


# ===================================================================
# Web-auth test seam
# ===================================================================
#
# ``WebAuthMiddleware`` is installed outermost on every interface app by
# ``configure_interface_app``, so every request a ``starlette.testclient``
# ``TestClient`` (re-exported as ``fastapi.testclient``; an ``httpx2.Client``
# subclass) — or an ``httpx.AsyncClient`` wrapping an ``httpx.ASGITransport`` —
# makes is 401'd unless it carries a live operator credential. Hundreds of existing tests drive interface routes through
# these clients and predate the gate; they assert the route's own behaviour, not
# the absence of authentication. Rather than touch each of them, this seam makes
# every such client present the operator credential by default.
#
# Mechanism — request-time operator-secret injection, *not* a construction-time
# cookie. Two facts about the existing suite force this:
#
#   1. Some interface tests pin their own ``app.state.web_credentials`` (a fresh
#      :class:`WebCredentials` with a known operator secret) *after* the client
#      is constructed — ``tests/interfaces/web_terminal/test_proxy.py`` does. A
#      credential minted at client-construction time lands in the holder that
#      swap discards, so it never authenticates. Reading the holder on each
#      request, instead, always sees the live one.
#   2. The proxy-boundary tests attach *decoy* inbound headers (a wrong
#      ``X-Osprey-Terminal-Secret``, a stale cookie, a bearer) to prove the
#      proxy strips them. The gate checks the operator-secret header *first* and
#      refuses a wrong one outright, so a session cookie added alongside a decoy
#      secret header would never be reached. Overriding that header with the
#      process's real secret is the only thing that authenticates such a request.
#
# So the seam force-sets ``X-Osprey-Terminal-Secret`` (``OPERATOR_SECRET_HEADER``)
# to the operator secret held by the client's target app, per request, reading
# ``app.state.web_credentials`` at send time. This is the header path the task
# explicitly allows as the alternative to the cookie: the cookie path is the one
# real browsers use, but it cannot serve these two shapes, and the Origin/CSRF
# logic it would exercise is already covered directly by the raw-ASGI tests in
# ``tests/interfaces/test_auth_middleware.py``. The operator-secret header is not
# subject to the Origin check, so mutating and websocket requests pass too.
#
# The seam acts **only** on a client aimed at a gated interface app — one whose
# ``app.state.web_credentials`` is a real :class:`WebCredentials` (the object the
# gate reads, and the tell that ``configure_interface_app`` ran). A client aimed
# at any other ASGI app, or at a real network host, is left untouched. Both
# client shapes are covered: ``TestClient`` over HTTP *and* its
# ``websocket_connect`` handshake, and ``httpx.AsyncClient`` over ``ASGITransport``.
# The two shapes come from different packages, ``httpx2`` for ``TestClient`` and
# ``httpx`` for the async client, which is why the seam patches each class by name.
#
# Install is session-scoped so it wraps client fixtures of every scope — a
# module-scoped client is built before any function-scoped fixture would run.
# Opt-out is a per-test flag (``@pytest.mark.no_auth_seam``) the injection hooks
# read at send time, for the dedicated tests that assert an *unauthenticated*
# refusal through such a client. The raw-ASGI auth tests never build one, so they
# need no marker.

#: Flipped off by :func:`_auth_seam_optout` for a ``no_auth_seam`` test. Read by
#: the injection hooks at send time, so it governs even a client built by a
#: higher-scoped fixture and shared across tests.
_AUTH_SEAM_ON = True


@pytest.fixture(autouse=True, scope="function")
def reset_web_credentials_between_tests(monkeypatch: pytest.MonkeyPatch):
    """Forget the process web credentials before and after every test.

    The env-driven population path (``web_auth._populate``) is decided once per
    process and is idempotent, so without this the first test in a worker would
    pin the operator secret for every test after it, and a test that mints a
    secret would leak it into its neighbours. Resetting both sides keeps the
    population testable in isolation. It does not clear ``app.state`` on an app
    built before the reset — such an app keeps the holder it cached. The root
    TestClient seam reads its secret from ``app.state``, so that app stays
    reachable through it; the browser and live-server seams in
    ``tests/interfaces/conftest.py`` read the PROCESS holder instead, so a
    module-level app must be re-pointed at it with
    ``use_process_web_credentials(app)`` before those seams can authenticate.

    Also clears ``SESSION_LIFETIME_ENV`` and ``SESSION_STORE_DIR_ENV`` before
    every test, for the same leak-prevention reason: ``tests/cli/test_discovery_
    rewire.py`` runs the real ``osprey web`` command in-process, and the
    launcher publishes both ``OSPREY_TERMINAL_SESSION_LIFETIME`` and
    ``OSPREY_TERMINAL_SESSION_STORE_DIR`` into ``os.environ`` for the child
    process to read. Left in place, every later test sharing that worker would
    populate a holder pointed at a real, on-disk store directory instead of the
    unconfigured one most tests assume. ``monkeypatch`` restores whichever
    value (if any) was present before the test once it tears down, rather than
    each of the two sides here reset unconditionally the way the credentials
    calls above do.
    """
    from osprey.interfaces.web_auth import (
        SESSION_LIFETIME_ENV,
        SESSION_STORE_DIR_ENV,
        reset_web_credentials,
    )
    from osprey.mcp_server.dispatch_worker.credentials import reset_worker_credentials
    from osprey.mcp_server.http import reset_panel_token_latch

    monkeypatch.delenv(SESSION_LIFETIME_ENV, raising=False)
    monkeypatch.delenv(SESSION_STORE_DIR_ENV, raising=False)

    # The MCP client latches the last panel token it saw (so an in-process
    # companion launch scrubbing the carrier does not strip its bearer); that
    # latch is process state for the same reason and is reset on both sides too.
    reset_web_credentials()
    reset_worker_credentials()
    reset_panel_token_latch()
    yield
    reset_web_credentials()
    reset_worker_credentials()
    reset_panel_token_latch()


def _gated_interface_app(client):
    """Return the client's target app if it is a gated interface app, else None.

    A ``TestClient`` records the app on ``self.app``; an ``httpx.AsyncClient`` wrapping an
    ``ASGITransport`` carries it on the transport. The tell that the app installed
    :class:`WebAuthMiddleware` is a real :class:`WebCredentials` on
    ``app.state`` — seeded by ``configure_interface_app`` and read by the gate.
    """
    from osprey.interfaces.web_auth import WebCredentials

    app = getattr(client, "app", None)
    if app is None:
        app = getattr(getattr(client, "_transport", None), "app", None)
    credentials = getattr(getattr(app, "state", None), "web_credentials", None)
    return app if isinstance(credentials, WebCredentials) else None


def _current_operator_secret(app):
    """The operator secret ``app``'s gate will accept right now, or None."""
    from osprey.interfaces.web_auth import WebCredentials

    credentials = getattr(getattr(app, "state", None), "web_credentials", None)
    if isinstance(credentials, WebCredentials):
        return credentials.operator_secret
    return None


def _install_client_auth(client) -> None:
    """Arm one test client — a ``TestClient`` (httpx2) or an ``httpx.AsyncClient`` — to
    present the operator secret to its gated app.

    A no-op for a client not aimed at a gated interface app. Otherwise it appends
    a request event hook (async for an ``AsyncClient``, sync otherwise) that
    force-sets the operator-secret header per request, and — for a client that
    speaks websockets — wraps ``websocket_connect`` to carry the same header on
    the handshake, which the request hooks never see.
    """
    import httpx

    app = _gated_interface_app(client)
    if app is None:
        return

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

    def _apply(request) -> None:
        if not _AUTH_SEAM_ON:
            return
        secret = _current_operator_secret(app)
        if secret:
            request.headers[OPERATOR_SECRET_HEADER] = secret

    async def _apply_async(request) -> None:
        _apply(request)

    hook = _apply_async if isinstance(client, httpx.AsyncClient) else _apply
    client.event_hooks.setdefault("request", []).append(hook)

    ws_connect = getattr(client, "websocket_connect", None)
    if callable(ws_connect):

        def _seamed_ws(url, *args, **kwargs):
            if _AUTH_SEAM_ON:
                secret = _current_operator_secret(app)
                if secret:
                    merged = dict(kwargs.get("headers") or {})
                    merged.setdefault(OPERATOR_SECRET_HEADER, secret)
                    kwargs["headers"] = merged
            return ws_connect(url, *args, **kwargs)

        client.websocket_connect = _seamed_ws


@pytest.fixture(autouse=True, scope="session")
def _install_auth_seam():
    """Patch the client constructors once so every client fixture is covered.

    Session-scoped on purpose: a module- or class-scoped client fixture is built
    before any function-scoped fixture runs, so a function-scoped patch would
    miss it. The per-test opt-out lives in :data:`_AUTH_SEAM_ON`, which the
    injection hooks read at send time.
    """
    import httpx
    from starlette.testclient import TestClient

    test_client_init = TestClient.__init__
    async_client_init = httpx.AsyncClient.__init__

    def _seamed_test_client_init(self, *args, **kwargs):
        test_client_init(self, *args, **kwargs)
        _install_client_auth(self)

    def _seamed_async_client_init(self, *args, **kwargs):
        async_client_init(self, *args, **kwargs)
        _install_client_auth(self)

    TestClient.__init__ = _seamed_test_client_init
    httpx.AsyncClient.__init__ = _seamed_async_client_init
    try:
        yield
    finally:
        TestClient.__init__ = test_client_init
        httpx.AsyncClient.__init__ = async_client_init


@pytest.fixture(autouse=True, scope="function")
def _auth_seam_optout(request):
    """Disable the auth seam for a ``@pytest.mark.no_auth_seam`` test."""
    global _AUTH_SEAM_ON
    previous = _AUTH_SEAM_ON
    _AUTH_SEAM_ON = request.node.get_closest_marker("no_auth_seam") is None
    try:
        yield
    finally:
        _AUTH_SEAM_ON = previous


# ===================================================================
# Health offload accounting guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_health_offload_state():
    """Zero the health runner's abandoned-thread accounting around every test.

    Leak guarded: ``osprey.health.offload`` counts abandoned worker threads
    cumulatively for the life of the process — correct for a real run, shared
    state for a test suite. ``tests/health/test_offload.py`` really abandons
    threads, and afterwards every in-process ``osprey health`` invocation in
    ``tests/cli`` takes the ``os._exit`` branch
    (``src/osprey/cli/health_cmd.py:323-326``) instead of ``sys.exit``. Under
    xdist that kills the worker and everything still queued on it.

    Reset happens only between tests, never inside one:
    ``tests/health/test_offload.py`` asserts the real counter.
    """
    from osprey.health.offload import reset_abandoned_state

    reset_abandoned_state()

    yield

    reset_abandoned_state()


# ===================================================================
# Companion server launch guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def _no_companion_server_launches(request, monkeypatch):
    """Keep ``ServerLauncher`` from starting a real uvicorn server in a test.

    Leak guarded: ``ServerLauncher._launch_in_thread`` runs uvicorn in a daemon
    thread nothing ever stops, and two runtime paths reach it unasked — the
    web-terminal lifespan launches every companion panel whose ``auto_launch``
    is on (the default), and ``ArtifactStore`` launches the gallery on every
    save. A test that boots either leaves a real, gated server listening on the
    configured port for the rest of the xdist worker's life. A later launcher
    that finds that port held probes it with the unauthenticated-then-
    credentialed ``GET /`` pair of ``_adopt_or_refuse``, and the leaked server
    files both refusals through ``osprey.audit.writer.record`` — resolved at
    call time, so they land in whichever test's spool is current. That test
    then fails on a record it never made.

    Only the thread start is replaced. Everything ``ensure_running`` decides
    before it — the auto-launch gate, the bind check, the held-port grace
    window and its verdict — still runs, so the launcher tests that patch
    those per instance are unaffected. A test of the launch itself opts out
    with ``@pytest.mark.real_server_launch``.
    """
    if request.node.get_closest_marker("real_server_launch"):
        return

    from osprey.infrastructure.server_launcher import ServerLauncher

    def _no_launch(_self, _host: str, _port: int) -> None:
        return None

    monkeypatch.setattr(ServerLauncher, "_launch_in_thread", _no_launch)


# ===================================================================
# Bounded captured runs
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def _bounded_runs_take_the_plain_spool_path(request):
    """Route ``run_captured``'s bounded runs through its plain spool path.

    A bounded run (a time limit or a watch, as every ``compose up`` now has)
    starts its child with ``Popen`` in a process group of its own, because
    stopping it has to reach that whole group. The deploy tests fake
    ``subprocess.run`` — the plain spool path's one call — to record argv and
    answer for compose. Without this, every one of them would let its
    ``compose up`` escape to the real runtime through ``Popen``.

    Only the runner is replaced: the argv, environment and spool a deploy test
    observes are the ones the plain path gives a call, which is what the bounded
    path gives it too. The runner's own tests opt out with
    ``@pytest.mark.real_bounded_run``.

    The patch is applied and undone here rather than through ``monkeypatch``:
    an autouse fixture that requests ``monkeypatch`` moves its setup ahead of
    the test's ``tmp_path``, so a test that patches ``os.open`` would still hold
    that patch when ``tmp_path`` is removed at teardown.
    """
    if request.node.get_closest_marker("real_bounded_run"):
        yield
        return

    from osprey.deployment import subprocess_capture

    def _plain(cmd, *, env, cwd, spool_name, repo_root, check, **_bounds):
        return subprocess_capture.run_captured(
            cmd, env=env, cwd=cwd, spool_name=spool_name, repo_root=repo_root, check=check
        )

    real = subprocess_capture._run_bounded
    subprocess_capture._run_bounded = _plain
    try:
        yield
    finally:
        subprocess_capture._run_bounded = real


# ===================================================================
# Marker-driven resource skip-gating
# ===================================================================
#
# Tests use @pytest.mark.requires_<resource> to declare what they need.
# This hook turns those markers into real skips when the resource is
# unavailable. Markers are registered in pyproject.toml under
# [tool.pytest.ini_options].markers; predicates live here.
#
# To add a new resource: append to _RESOURCE_CHECKS. Predicate must be
# zero-arg, side-effect-free, and fast (gets called once per item per
# resource at collection time).


def _has_als_apg_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ALS_APG_API_KEY"))


def _has_anthropic_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ANTHROPIC_API_KEY"))


def _has_any_provider_api_key() -> bool:
    """True if any supported LLM provider key is set.

    Checks ALS_APG_API_KEY, CBORG_API_KEY, AMSC_I2_API_KEY and ANTHROPIC_API_KEY.
    """
    import os as _os

    return any(
        _os.environ.get(k)
        for k in ("ALS_APG_API_KEY", "CBORG_API_KEY", "AMSC_I2_API_KEY", "ANTHROPIC_API_KEY")
    )


def _has_openai_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("OPENAI_API_KEY"))


def _has_google_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("GOOGLE_API_KEY"))


def _has_cborg_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("CBORG_API_KEY"))


def _e2e_provider_availability() -> tuple[bool, str]:
    """Whether the credential of the provider this run builds with is present.

    Two facts meet here and neither is authored: which provider the run was
    told to build with (``tests.e2e.provider``) and which variable holds that
    provider's key (``PROVIDER_API_KEYS``, whose own comment calls it the single
    source of truth). Gating on one gateway's key instead is what let a run
    naming another provider, and holding its credential, skip anyway.

    A run that named no provider reads as unavailable carrying the refusal's own
    message rather than raising: the e2e collection hook owns ending such a
    session, and keeping this predicate out of that business leaves the two free
    of any hook ordering.
    """
    import os as _os

    from osprey.models.provider_registry import PROVIDER_API_KEYS
    from tests.e2e.provider import E2E_PROVIDER_ENV, e2e_provider

    try:
        provider = e2e_provider()
    except RuntimeError as exc:
        return False, str(exc)
    if provider not in PROVIDER_API_KEYS:
        known = ", ".join(sorted(PROVIDER_API_KEYS))
        return False, (
            f"{E2E_PROVIDER_ENV} names {provider!r}, which is not a provider OSPREY "
            f"registers (known: {known})"
        )
    key_var = PROVIDER_API_KEYS[provider]
    if key_var is None or _os.environ.get(key_var):
        return True, ""
    return False, f"{key_var} not set — the provider this run builds with is {provider!r}"


def _has_e2e_provider_key() -> bool:
    return _e2e_provider_availability()[0]


def _e2e_provider_reason() -> str:
    return _e2e_provider_availability()[1]


def _is_matlab_available() -> bool:
    """True if a `matlab` executable is on PATH."""
    import shutil as _shutil

    return _shutil.which("matlab") is not None


def _is_ollama_available() -> bool:
    """True if a local Ollama server responds at localhost:11434."""
    try:
        import requests

        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def ollama_has_model(name: str, base_url: str = "http://localhost:11434") -> bool:
    """True if the Ollama server at ``base_url`` lists ``name`` in ``/api/tags``.

    A name without a tag matches its ``:latest`` listing. Never raises: an
    unreachable server or an unexpected listing reads as ``False``.
    """
    wanted = {name, name if ":" in name else f"{name}:latest"}
    try:
        import requests

        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        if response.status_code != 200:
            return False
        models = response.json().get("models") or []
        return any(
            isinstance(m, dict) and (m.get("name") in wanted or m.get("model") in wanted)
            for m in models
        )
    except Exception:
        return False


# The second element is the skip reason: a fixed string, or a zero-arg callable
# for a resource whose absence has more than one explanation to report.
_RESOURCE_CHECKS: dict[str, tuple[Callable[[], bool], str | Callable[[], str]]] = {
    "requires_e2e_provider": (_has_e2e_provider_key, _e2e_provider_reason),
    "requires_als_apg": (_has_als_apg_api_key, "ALS_APG_API_KEY not set"),
    "requires_anthropic": (_has_anthropic_api_key, "ANTHROPIC_API_KEY not set"),
    "requires_api": (
        _has_any_provider_api_key,
        "No LLM provider API key set (ALS_APG_API_KEY / CBORG_API_KEY / "
        "AMSC_I2_API_KEY / ANTHROPIC_API_KEY)",
    ),
    "requires_ollama": (_is_ollama_available, "Ollama not reachable at localhost:11434"),
    "requires_matlab": (
        _is_matlab_available,
        "matlab not on PATH — a MATLAB installation with the AT toolbox is required",
    ),
    "requires_openai": (_has_openai_api_key, "OPENAI_API_KEY not set"),
    "requires_google": (_has_google_api_key, "GOOGLE_API_KEY not set"),
    "requires_cborg": (_has_cborg_api_key, "CBORG_API_KEY not set"),
}


def pytest_collection_modifyitems(items):
    """Auto-skip items whose `requires_<resource>` marker's resource is missing.

    Each predicate is evaluated at most once per pytest run via cache. A
    missing resource adds a real `pytest.mark.skip(reason=...)`; satisfied
    markers are no-ops.
    """
    cache: dict[str, tuple[bool, str]] = {}
    for item in items:
        for marker_name, (predicate, reason) in _RESOURCE_CHECKS.items():
            if marker_name not in item.keywords:
                continue
            resolved = cache.get(marker_name)
            if resolved is None:
                available = predicate()
                text = "" if available else (reason() if callable(reason) else reason)
                resolved = (available, text)
                cache[marker_name] = resolved
            if not resolved[0]:
                item.add_marker(pytest.mark.skip(reason=resolved[1]))


def pytest_configure(config):
    """Register the harness's plugins and the markers whose seams live in this file."""
    _harness.pytest_configure(config)

    # A shared fixture's temporary directories go when its scope ends, not when
    # the session does. See tests/_scoped_temp.py.
    if not config.pluginmanager.is_registered(_scoped_temp):
        config.pluginmanager.register(_scoped_temp, _scoped_temp.PLUGIN_NAME)

    # Registered here rather than in pyproject.toml so the seam and its opt-out
    # marker live in one file; `--strict-markers` would otherwise reject it.
    config.addinivalue_line(
        "markers",
        "no_auth_seam: opt this test out of the TestClient auth seam — for a "
        "test that asserts an unauthenticated 401/403 through a TestClient.",
    )

    # Registered here for the same reason: the lane it marks lives in one file,
    # which also owns the resolution that decides whether the marked tests can
    # run at all, so the marker and its reason are read together.
    config.addinivalue_line(
        "markers",
        "requires_als_profiles: the install over a real facility's MML export, "
        "which never enters the repo — import, build and boot; needs that "
        "facility's profiles checkout, a named export and mapping, or a "
        "stand-in tree, and skips naming the unset variables anywhere else.",
    )


# ===================================================================
# The controller's half of the e2e provider refusal
# ===================================================================
#
# ``tests/e2e/conftest.py`` refuses a run that names no provider and selects an
# e2e test needing one. Under xdist that judgement runs in the workers, and a
# worker cannot end the session cleanly: pytest reports its collection finished
# even when the judging hook raised, the controller hands it tests, and the
# worker's early exit then surfaces as xdist's ``INTERNALERROR`` about a crashed
# item, with the refusal lost. So a worker hands its refusal to the controller
# instead, and the controller prints it.
#
# The controller's half lives here, not beside the refusal: the controller
# collects nothing, so it loads only the conftests on the path to each
# argument. ``tests/e2e/conftest.py`` is one of them only when an argument lies
# inside ``tests/e2e``; a whole-suite run names ``tests`` and loads this file
# alone.

#: ``workerinput`` flag the controller sets when it relays a worker's refusal.
E2E_REFUSAL_RELAY_KEY = "osprey_e2e_refusal_relay"
#: ``workeroutput`` key a worker hands its refusal to the controller under.
E2E_REFUSAL_KEY = "osprey_e2e_refusal"
#: The refusals the controller collected from its workers.
_E2E_REFUSALS = pytest.StashKey[list[str]]()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """Tell each xdist worker that this controller prints its refusal."""
    node.workerinput[E2E_REFUSAL_RELAY_KEY] = True


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node):
    """Collect the refusal a worker handed over."""
    message = getattr(node, "workeroutput", {}).get(E2E_REFUSAL_KEY)
    if message:
        node.config.stash.setdefault(_E2E_REFUSALS, []).append(message)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    """Stop git's repository discovery at the run's base temp.

    A base temp inside a checkout would otherwise make every ``tmp_path`` look
    like part of that checkout to any git a test or its subprocess runs: a
    branch appears where none should, a ``git status`` reports the host's
    changes. A repository a test creates at or below ``tmp_path`` is still
    found, because the ceiling sits above it. Set before any test starts, so
    the per-test environment snapshots take it as their baseline.
    """
    base = str(session.config._tmp_path_factory.getbasetemp().resolve())
    ceilings = os.environ.get("GIT_CEILING_DIRECTORIES")
    os.environ["GIT_CEILING_DIRECTORIES"] = os.pathsep.join((base, ceilings)) if ceilings else base


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    """Print a relayed refusal and end the session as a usage error.

    Every worker collects the same selection, so every refusal is identical and
    the first one is printed.
    """
    refusals = session.config.stash.get(_E2E_REFUSALS, [])
    if not refusals:
        return
    sys.stdout.flush()
    sys.stderr.write(f"ERROR: {refusals[0]}\n")
    sys.stderr.flush()
    session.exitstatus = pytest.ExitCode.USAGE_ERROR


# ===================================================================
# Test Configuration Helpers
# ===================================================================


@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a minimal test configuration file.

    Creates a temporary config.yml with sensible defaults for testing.
    Tests can override or extend this as needed.

    Returns:
        Path to the created config file

    Examples:
        Basic usage::

            def test_something(test_config):
                # Config file exists at test_config
                assert test_config.exists()
                # Set environment to use it
                os.environ['CONFIG_FILE'] = str(test_config)

        With custom modifications::

            def test_custom(test_config):
                # Read existing config
                import yaml
                with open(test_config) as f:
                    config = yaml.safe_load(f)
                # Modify
                config['execution']['execution_method'] = 'subprocess'
                # Write back
                with open(test_config, 'w') as f:
                    yaml.dump(config, f)
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry()
"""
    )

    # Minimal working configuration for tests
    config = {
        "project_root": str(tmp_path),
        "registry_path": str(registry_file),
        "approval": {
            "enabled": True,
            "default_policy": "selective",
        },
        "control_system": {
            "type": "virtual_accelerator",  # Use mock for tests - default patterns include write_channel/read_channel
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        },
        "execution": {
            "execution_method": "subprocess",
            "limits": {"max_retries": 3, "max_execution_time_seconds": 30},
        },
        "models": {
            "orchestrator": {"provider": "openai", "model_id": "gpt-4"},
            "python_code_generator": {"provider": "openai", "model_id": "gpt-4"},
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let each test handle registry initialization
    # to avoid state pollution between tests
    return config_file


@pytest.fixture(scope="session")
def graphdb_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding neosemantics + APOC, resolved once per session.

    Every lane that starts a real graph store mounts the same two jars, and the
    expensive half of resolving them is a release download — so this is shared
    across lanes even though the *stores* are not (two of them wipe the graph
    between corpora and therefore run their own module-scoped container).

    Skips, with the reason, on a host that could not start the container:
    see :func:`tests._graphdb_container.resolve_plugin_dir`.
    """
    from tests._graphdb_container import resolve_plugin_dir

    return resolve_plugin_dir(tmp_path_factory)


# ===================================================================
# Graph search index
# ===================================================================


@dataclass(slots=True)
class GraphIndexCache:
    """Every graph search index this session has built, keyed by corpus digest.

    Held for the whole session because the thing being cached is not a
    render's file but the *parse* behind it: two builds of the same corpus,
    in two tests, in two directories, derive byte-identical rows from
    byte-identical text.
    """

    directory: Path
    """Where the cached indexes live — one ``<digest>.duckdb`` per corpus."""

    entries: dict[str, Path] = field(default_factory=dict)
    """Digest of the corpus TEXT -> the index built from it, inside :attr:`directory`."""


@pytest.fixture(scope="session")
def graph_index_cache(tmp_path_factory: pytest.TempPathFactory) -> GraphIndexCache:
    """The session's built indexes, shared by every render the suite performs.

    Requested by the autouse stub below, and by the tests that assert on the
    caching itself: a test that needs a corpus to be built for real can drop
    its entry here first, and get a deterministic builder count whatever ran
    before it.
    """
    return GraphIndexCache(directory=tmp_path_factory.mktemp("graph-index-cache"))


@pytest.fixture(autouse=True, scope="function")
def _stub_graph_index_builds(request, monkeypatch, graph_index_cache: GraphIndexCache):
    """Build each corpus the suite renders once per session, not once per render.

    ``osprey build`` derives a graph-mode project's channel search index from
    the Turtle corpus the render staged, which is an rdflib parse of the whole
    file — about a second and a half for the corpus the control-assistant
    preset ships. The suite renders that preset over and over, in the build
    tests, the deployment tests and every exemplar render, and each of those
    renders parses the identical bytes again.

    So the parse is memoized across the whole session on the digest of the
    corpus TEXT, the same key the build's own per-build memo uses
    (:attr:`osprey.cli.build_cmd._SharedRenderInputs.graph_indexes`), and for
    the same reason: every render re-copies the profile's ``data/`` tree, so
    the staged corpus is a different file with a fresh mtime every time and a
    path or stat key would miss on every pass.

    What is stubbed is only :func:`osprey.cli.build_cmd._build_graph_index` —
    the step that turns a corpus into a file. ``_graph_index_target`` still
    resolves the corpus and the index path out of the rendered config, and the
    index that lands in the render is the real builder's output for that exact
    corpus, so a test can still read it, query it, or hand it to the roster.
    A corpus that cannot be read or parsed is handed straight to the real hook,
    which owns the warning and the decision that a failure costs the index
    rather than the build — one copy of that fact, not two.

    Opt out with ``@pytest.mark.real_graph_index`` when the derivation itself
    is what a test is about.
    """
    if request.node.get_closest_marker("real_graph_index"):
        return

    import shutil

    from osprey.cli import build_cmd

    real_hook = build_cmd._build_graph_index

    def _cached_build_graph_index(shared, target, progress):
        # Imported here, at call time, for two reasons: the fixture stays
        # import-light for the tests that never render anything, and a test
        # that counts builder calls by patching the package attribute still
        # sees this path go through it.
        from osprey.services.channel_finder.graph_index import build_graph_index
        from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

        try:
            digest = ttl_sha256(target.corpus_path.read_text(encoding="utf-8"))
            cached = graph_index_cache.entries.get(digest)
            if cached is None:
                cached = graph_index_cache.directory / f"{digest}.duckdb"
                build_graph_index(target.corpus_path, cached)
                if not cached.is_file():
                    raise FileNotFoundError(cached)
                # Recorded only once the file exists, so a build that wrote
                # nothing cannot poison every later render of this corpus for
                # the rest of the session.
                graph_index_cache.entries[digest] = cached
            target.index_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target.index_path)
        except Exception:  # the real hook decides what a failure costs
            return real_hook(shared, target, progress)

        # The build's own memo, kept consistent: a later render pass of the
        # same build reads it before asking for an index at all.
        shared.graph_indexes.setdefault(digest, target.index_path)
        progress(
            "  ✓ Reused the session-cached channel search index built from %s",
            target.corpus_path.name,
        )
        return target.index_path

    #: Marks this as the stub, so a test can assert which hook it is running
    #: against without inspecting where the function was defined.
    _cached_build_graph_index.stubbed_by_conftest = True

    monkeypatch.setattr(build_cmd, "_build_graph_index", _cached_build_graph_index)


# ===================================================================
# Entry points
# ===================================================================


@pytest.fixture(autouse=True, scope="session")
def _memoized_entry_points():
    """Read the installed distributions' entry points once per session, not once per call.

    ``importlib.metadata.entry_points()`` walks every distribution on the path
    on each call — tens of milliseconds in this environment — and a build of
    the demo facility looks its simulation engines up a few hundred times, so
    the suite spent a large share of every build re-reading the same
    ``entry_points.txt`` files. Nothing a test does installs a distribution, so
    the answer for one selection is the same for the whole session.

    A test that hands the code a fake registry still does so by patching
    ``metadata.entry_points`` itself; its patch replaces this one for that test
    and wraps it when it calls through.
    """
    import functools
    from importlib import metadata

    memoized = functools.cache(metadata.entry_points)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(metadata, "entry_points", memoized)
        yield


@contextmanager
def dispatcher_route_registry():
    """Build dispatcher servers without inheriting or leaving route registrations.

    ``osprey.dispatch.server.create_server()`` appends its dashboard and webhook
    routes to one module-level FastMCP instance. Production calls it once; a test
    calls it per module or per case, so two things go wrong when it is called
    bare: a registration an earlier caller left behind matches first and answers
    from a registry this caller never filled, and the registrations this caller
    adds outlive it.

    Inside the block, ``build()`` calls the factory and then keeps only the
    newest registration of each path and method set, so the routes the app serves
    are the ones this block registered. On the way out the singleton carries
    exactly the routes it had on the way in. A block that never calls ``build()``
    still gets the restore, which is all a test needs when the case itself is the
    one calling the factory.
    """
    from osprey.dispatch import server as dispatch_server

    baseline_routes = list(dispatch_server.mcp._additional_http_routes)

    def build():
        """Call the factory, then drop every registration it did not make."""
        built = dispatch_server.create_server()
        newest: dict[tuple, object] = {}
        for route in dispatch_server.mcp._additional_http_routes:
            methods = tuple(sorted(getattr(route, "methods", None) or ()))
            newest[getattr(route, "path", None), methods] = route
        dispatch_server.mcp._additional_http_routes = list(newest.values())
        return built

    try:
        yield build
    finally:
        dispatch_server.mcp._additional_http_routes = baseline_routes


# ===================================================================
# Packaged-template tree guard
# ===================================================================


@pytest.fixture(autouse=True, scope="session")
def no_writes_into_the_packaged_templates():
    """Fail the session if a test added or removed a path in the shipped templates.

    Tests that copy the template tree read it while other tests run beside them,
    so a test that writes into it and cleans up again surfaces as a copy error
    somewhere else. This names the directories instead, including one where an
    entry came and went before the session ended. A test that needs a changed
    tree copies it to ``tmp_path`` and points ``TemplateManager._get_template_root``
    at the copy. Under xdist each worker checks its own session.
    """
    from osprey.cli.templates.manager import TemplateManager

    root = Path(TemplateManager().template_root)
    before = _repo_cleanliness.tree_snapshot(root)
    yield
    clauses = _repo_cleanliness.what_this_run_did_to_tree(before, root)
    if clauses:
        raise AssertionError(
            f"the test run {'; '.join(clauses)}. "
            "A test that needs a changed tree writes into a tmp_path copy of it."
        )
