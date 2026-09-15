"""Real-stack e2e for the Microsoft Teams bridge: loopback fakes and a real dispatcher.

Every unit test of ``osprey.bridges.teams`` drives one seam at a time against a mock or
a transport stub, so each asserts OSPREY's half of a two-party contract against OSPREY's
own idea of the wire format. This module is the instrument for the rest of the stack: a
real ``osprey.dispatch`` dispatcher/worker pair as subprocesses, the bridge itself booted
**through its own entrypoint seams** — the same ``build_wiring``/``run`` the container's
``main`` calls — and the other party made concrete by the three fakes in
``tests/e2e/fixtures/teams_fakes.py``: an in-process queue the serve loop pulls from, a
loopback login host the token exchange authenticates against, and a loopback Bot
Connector the replies land on. What it proves is that the adapter's conversation-type
rules, its mention filter, its exactly-once claim and its settlement ordering hold when
the whole engine is running behind them.

Every test here is **deterministic**: no assertion reads model output, so they pass with
no provider key configured at all. A dispatch is observed as "the dedup entry now carries
a ``run_id``" — the engine persists that the moment the dispatcher handshake yields one,
long before the agent finishes — so a run that errors out for want of an API key proves
exactly as much as one that answers. The Teams-side strings this module compares against
are the adapter's own :func:`~osprey.bridges.teams.ops.ack_text` and
:func:`~osprey.bridges.teams.ops.quote_prefix`, called rather than re-spelled, so a
reworded ack changes one constant in the product and nothing here.

There is no agentic tier and there never will be one in this file: the lane declares no
provider secret, which is what lets it run on every pull request.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call reachable from this file)
----------------------------------------------------------------------------
**This module issues no container-runtime command at all, and needs no container.** It
creates no container, no volume and no image, and contains no call to a runtime CLI —
directly or through a fixture. Unlike its Google Chat sibling there is no broker emulator
to start: :class:`~tests.e2e.fixtures.teams_fakes.FakeQueueReceiver` stands in for the
Service Bus receiver entirely, in this interpreter. Nothing here runs ``system prune``,
``volume prune``, ``container prune``, ``image prune`` or a ``volume rm``, so the host's
unrelated stacks are untouched.

Everything the lane needs is a plain host process or a loopback HTTP server: the
dispatcher and worker are subprocesses of this interpreter, and the token and Connector
fakes are ``http.server`` instances on ephemeral loopback ports.

----------------------------------------------------------------------------
The four injected seams, and why all four are needed
----------------------------------------------------------------------------
:func:`~osprey.bridges.teams.__main__.build_wiring` exposes exactly four seams, and this
module supplies every one — which is what lets the lane run with no tenant, no bot
credentials and no reachable Microsoft endpoint while still executing the adapter's real
wiring:

``receiver_factory``
    A factory returning this test's :class:`OrderedReceiver` — the in-process queue. It
    keeps ``azure.servicebus`` and the broker out of the message path entirely, so what
    is exercised is the serve loop's own settlement contract rather than the SDK's.
``token_http``
    ``FakeTokenServer.http_client()``. The login host is resolved from a closed cloud
    table rather than from configuration, so this seam is the *only* way to point the
    token exchange at a fake; the client it builds rewrites the scheme, host and port of
    the product's own AAD URL and leaves the path and form body untouched.
``connector_http``
    A plain ``httpx.Client`` with ``trust_env=False``. The Connector needs no rewrite —
    the bridge posts to whatever ``serviceUrl`` the inbound activity carried, and the
    activity builders carry :attr:`FakeConnectorServer.base_url` — but it must not
    inherit a dev shell's proxy in front of a loopback server.
``worker_http``
    One ``httpx.Client`` for the WORKER's artifact byte route: an internal service,
    reached with the dispatch token, and not the Connector's client.

``TEAMS_SERVICEBUS_CONNECTION_STRING`` therefore names a namespace that is deliberately
**unreachable** (:data:`CONNECTION_STRING`, pointing at loopback): with the receiver seam
injected nothing may ever dial it, and a regression that started dialling would fail
loudly here instead of quietly reaching for a real broker.

----------------------------------------------------------------------------
Why the receiver is a subclass rather than the fake itself
----------------------------------------------------------------------------
:class:`OrderedReceiver` adds nothing to the message path. It records, at each
``register`` and ``complete``, how many activities the Connector had accepted at that
instant — which is what turns "register precedes and complete follows ``handle_event``"
from a claim about a call log into a claim about the work in between. The call log alone
would be satisfied by a loop that registered and completed a message it never handled;
the post window would not.

----------------------------------------------------------------------------
Host ports
----------------------------------------------------------------------------
This lane pins **none**, deliberately, and therefore adds nothing to the set its sibling
e2e modules pin (5064, 15064-15067, 15080, 18090-18110, 18191-18194, 19080-19085,
19100-19591, 19900-20201, 20700-21699, 21781, 21791, 25080-25085, 25432-25436,
27117-27118), nor to the thousand-port block a deployment claims from
``deployment.port_base`` (10000-10999 at the default base). The dispatcher and worker
take :func:`_free_port` (they are per-run subprocesses with no need to be predictable,
and a pin would collide with a developer's own running stack), and the two fakes bind
``127.0.0.1:0``.

----------------------------------------------------------------------------
Gating: there is none, and that is the point
----------------------------------------------------------------------------
No container runtime, no image, no credential and no provider key, so this module has no
``skipif`` at all and a skip here is a bug rather than an environment. What it does need
is the ``teams`` extra: :func:`~osprey.bridges.teams.receiver.make_receiver` is called
for real by the construct-only proof at the foot of this file, and ``azure.servicebus``
is imported plainly at module scope rather than behind ``importorskip`` — so a lane that
forgot ``uv sync --extra dev --extra teams`` fails at collection, loudly, instead of
skipping its way to a green that proves nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import azure.servicebus
import httpx
import pytest
import yaml
from azure.servicebus import AutoLockRenewer, ServiceBusClient, ServiceBusReceiveMode

from osprey.bridges.teams.__main__ import Wiring, build_wiring, config_from_env, run
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.events import (
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
    MS_TENANT_ID,
)
from osprey.bridges.teams.ops import ack_text, quote_prefix
from osprey.bridges.teams.receiver import (
    SETTLE_MARGIN_SEC,
    ServiceBusQueueReceiver,
    make_receiver,
)
from tests.e2e.fixtures.teams_fakes import (
    ACCESS_TOKEN,
    APP_ID,
    CHANNEL_ID,
    PERSONAL_CONVERSATION_ID,
    SENDER_ID,
    TENANT_ID,
    FakeConnectorServer,
    FakeQueueMessage,
    FakeQueueReceiver,
    FakeTokenServer,
    channel_activity,
    personal_activity,
)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]
# No ``dockerbuild`` marker and no ``skipif``: this lane builds no image, starts no
# container and reaches no network beyond loopback. See the gating note in the docstring.


# ---------------------------------------------------------------------------
# The bot's identity and the coordinates the tests speak through
# ---------------------------------------------------------------------------

APP_SECRET = "not-a-real-client-secret"
"""What the token exchange sends. The fake login host answers any secret; this one is
recorded on :attr:`FakeTokenServer.requests`, which is where it is asserted."""

QUEUE_NAME = "osprey-teams-e2e"
"""Queue the config names. Nothing ever opens it: the receiver seam is injected."""

CONNECTION_STRING = (
    "Endpoint=sb://127.0.0.1/;SharedAccessKeyName=bridge-listen;SharedAccessKey=x;"
    f"EntityPath={QUEUE_NAME}"
)
"""A well-formed Service Bus connection string pointing at loopback.

Well-formed because the construct-only proof hands it to the real SDK, which parses it;
loopback because nothing in this module may ever dial a broker, and an unreachable
address is what makes a regression that tried to dial one fail here rather than
elsewhere."""

VERSION_TAG = "e2e-teams-tag"
"""``APP_VERSION_DISPLAY`` for every bridge this module boots.

Pinned to a sentinel rather than left empty: ``TeamsBridgeConfig.from_env`` falls back to
the *installed* distribution's version when the variable is blank, so an empty pin would
make the expected ack depend on how the checkout happens to be installed. With a sentinel
the ack is ``ack_text(VERSION_TAG)`` on any machine."""

DISPATCH_TOKEN = "microsoft-teams-e2e-token"
"""Shared dispatcher<->worker bearer for this run. Both halves are local subprocesses."""

TRIGGER_NAME = "microsoft-teams-e2e"
"""Name of the deterministic trigger :func:`_write_triggers` generates."""


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

HEALTH_TIMEOUT_SEC = 45.0
"""Per-subprocess wait for the dispatcher's / worker's ``/health``."""

RUN_ID_TIMEOUT_SEC = 150.0
"""Wait for a claimed activity's dedup entry to carry a ``run_id`` — one dispatcher POST
plus the accept handshake, well short of any agent work."""

HANDLED_TIMEOUT_SEC = 300.0
"""Wait for the bridge to settle a delivered message.

Generous because "settled" for a CLAIMED activity means the whole dispatch finished: the
handler polls the run to terminal before it completes the message. Comfortably past
:data:`WORKER_RUN_CAP_SEC` plus the ack and answer posts."""

IGNORED_TIMEOUT_SEC = 90.0
"""Wait for the bridge to settle a message it will IGNORE.

The barrier the negative proofs need: once the message is completed, anything the bridge
was going to do for it, it has already done — so "nothing happened" cannot be confused
with "nothing has happened yet"."""

BRIDGE_JOIN_TIMEOUT_SEC = 90.0
"""Wait for a stopped bridge's threads.

Not politeness. The drain thread is a daemon, so an outliving one does not block the
interpreter — it shows up as a dedup store being written by a bridge the test believes is
down. Shutdown therefore waits for the threads themselves, not merely for ``run`` to
return."""

WORKER_RUN_CAP_SEC = 90
"""``DISPATCH_TIMEOUT_SEC`` for the worker AND the bridge's ``poll_budget``.

Bounds how long a wedged run can hold a handler thread. The two must agree: the bridge's
``CoreConfig`` refuses to build when ``POLL_BUDGET < DISPATCH_TIMEOUT_SEC``."""

BUILD_TIMEOUT_SEC = 600
"""Wall-clock cap on the one ``osprey build`` this module runs."""


# ---------------------------------------------------------------------------
# Wire helpers — the test's own view of an activity
# ---------------------------------------------------------------------------


def _dedup_key(activity: Mapping[str, Any]) -> str:
    """The dedup key the adapter derives for ``activity``.

    Spelled here rather than imported because it is the *test's* expectation of the
    adapter's scoping rule: a Teams activity id is unique only within its conversation,
    so the key is the pair. A change to that rule must break this line.
    """
    return f"{activity['conversation']['id']}:{activity['id']}"


# ---------------------------------------------------------------------------
# The queue seam: the fake, plus a record of what was posted while a message was held
# ---------------------------------------------------------------------------


class OrderedReceiver(FakeQueueReceiver):
    """:class:`FakeQueueReceiver` that also records the Connector's post count per call.

    The serve loop registers a message for lock renewal, hands it to ``handle_event``,
    and completes it from a ``finally``. :attr:`FakeQueueReceiver.calls` proves the first
    and last of those happened in that order; it cannot prove anything happened between
    them. So each ``register`` and ``complete`` also snapshots how many activities the
    Connector had accepted at that instant, and :meth:`posts_while_held` is the
    difference — positive for a claimed activity (its ack landed inside the window), zero
    for one the adapter ignored or de-duplicated.

    The snapshot is taken *before* delegating, which is what makes the register-side
    number a true "nothing posted yet" and the complete-side one a true "everything this
    handler posted is already in".
    """

    def __init__(self, connector: FakeConnectorServer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.connector = connector
        """The server whose accepted-post count the windows are measured in."""
        self._window_lock = threading.Lock()
        self._windows: dict[str, dict[str, int]] = {}

    def register(self, msg: Any) -> None:
        self._snapshot("register", msg)
        super().register(msg)

    def complete(self, msg: Any) -> None:
        self._snapshot("complete", msg)
        super().complete(msg)

    def posts_while_held(self, message: FakeQueueMessage | str) -> int:
        """Activities the Connector accepted between this message's register and complete.

        Raises:
            AssertionError: If the message was not both registered and completed, which
                would make the window meaningless rather than empty.
        """
        wanted = message if isinstance(message, str) else message.id
        with self._window_lock:
            window = dict(self._windows.get(wanted, {}))
        assert {"register", "complete"} <= set(window), (
            f"{wanted} was not both registered and completed; recorded {window!r} "
            f"(call log: {self.calls!r})"
        )
        return window["complete"] - window["register"]

    def _snapshot(self, action: str, msg: Any) -> None:
        posted = len(self.connector.posted)
        with self._window_lock:
            self._windows.setdefault(str(getattr(msg, "id", msg)), {})[action] = posted


# ---------------------------------------------------------------------------
# The hermetic Teams-side fakes (no runtime, no credentials — never skip)
# ---------------------------------------------------------------------------


@pytest.fixture
def connector() -> Iterator[FakeConnectorServer]:
    """The Bot Connector the bridge replies to. Function-scoped: one test's posts are
    another's noise, and the activity builders carry its ``base_url`` as ``serviceUrl``."""
    with FakeConnectorServer() as server:
        yield server


@pytest.fixture
def token() -> Iterator[FakeTokenServer]:
    """The login host the bot authenticates against."""
    with FakeTokenServer() as server:
        yield server


@pytest.fixture
def token_http(token: FakeTokenServer) -> Iterator[httpx.Client]:
    """The ``token_http`` seam: a client whose transport lands AAD's URL on the fake."""
    with token.http_client() as client:
        yield client


@pytest.fixture
def connector_http() -> Iterator[httpx.Client]:
    """The ``connector_http`` seam.

    ``trust_env=False`` for the same reason ``CoreConfig`` defaults it off: a proxy
    inherited from a dev shell must not mount itself in front of a loopback server.
    """
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        yield client


@pytest.fixture
def worker_http() -> Iterator[httpx.Client]:
    """The ``worker_http`` seam: the client the artifact byte route is read with."""
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        yield client


@pytest.fixture
def receiver(connector: FakeConnectorServer) -> OrderedReceiver:
    """The in-process queue, seeded by each test and read back for the call log."""
    return OrderedReceiver(connector)


# ---------------------------------------------------------------------------
# Real dispatcher + worker as subprocesses (harness shape shared with
# tests/e2e/test_gchat_bridge_e2e.py)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to :0, read the assigned port, release it (standard free-port trick)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drain_output(proc: subprocess.Popen) -> str:
    """Best-effort grab of a subprocess's combined output for failure messages."""
    if proc.stdout is None:
        return "(no captured output)"
    try:
        data = proc.stdout.read1(16384) if hasattr(proc.stdout, "read1") else b""
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return f"--- subprocess output (partial) ---\n{text}" if text else "(no captured output)"


def _wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    """Poll ``url`` until it returns HTTP 200, or fail with the captured output."""
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    with httpx.Client(timeout=3.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"subprocess for {url} exited early (rc={proc.returncode}).\n"
                    f"{_drain_output(proc)}"
                )
            try:
                if client.get(url).status_code == 200:
                    return
                last_err = "non-200"
            except httpx.HTTPError as exc:
                last_err = str(exc)
            time.sleep(0.5)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} (last error: {last_err}).\n"
        f"{_drain_output(proc)}"
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Ignored SIGKILL within the grace window; teardown is best-effort, so leave
            # it for the OS to reap rather than hang the test.
            pass


def _find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for the ACTIVE interpreter.

    Deliberately interpreter-relative first: a bare ``osprey`` on PATH may belong to a
    different checkout entirely, and in a worktree it usually does.
    """
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError(
        "Could not locate the 'osprey' console script. "
        f"Tried {Path(sys.executable).parent / 'osprey'} and PATH."
    )


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Init + build a real control-assistant deployment repo once per module.

    Two steps because the surface has two: ``init`` writes the repo's source zone from
    the preset, ``build`` renders ``build/`` from it. ``--skip-deps`` keeps it fast (no
    project venv); the worker and dispatcher run with this repo's interpreter. The
    provider choice never reaches an assertion here — see the module docstring on why
    these tests are model-independent.
    """
    base = tmp_path_factory.mktemp("teams_bridge_build")
    repo = base / "proj"
    osprey_bin = _find_osprey_console_script()

    def _osprey(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(osprey_bin), *argv],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SEC,
            check=False,
            env={**os.environ, "CLAUDECODE": ""},
        )

    init = _osprey(
        [
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
        ]
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    build = _osprey(["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )
    if not (repo / "build" / "config.yml").is_file():
        pytest.fail(f"build succeeded but build/config.yml missing under {repo}")
    return repo


def _write_triggers(dst: Path, worker_port: int) -> None:
    """Write the triggers document the dispatcher runs.

    :data:`TRIGGER_NAME` is deliberately not one of the shipped tutorial triggers: those
    exist to demonstrate agent behaviour, and this lane must assert nothing about what a
    model says. It asks for a fixed word and no tools, so a run either completes or fails
    for want of a provider key — and both outcomes carry a ``run_id``, which is the only
    thing these tests read.
    """
    doc = {
        "dispatcher": {
            "dispatch_target": f"http://127.0.0.1:{worker_port}",
            "max_concurrent_runs": 2,
            "max_queue_depth": 50,
        },
        "triggers": [
            {
                "name": TRIGGER_NAME,
                "source": "webhook",
                "action": {
                    "prompt": (
                        "A Microsoft Teams end-to-end test fired this event. Reply with "
                        "the single word ACKNOWLEDGED and nothing else. Do not use any "
                        "tools."
                    ),
                    "allowed_tools": [],
                },
            }
        ],
    }
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def dispatch_stack(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """A real worker + dispatcher pair as subprocesses on free ports.

    Module-scoped: the bridge is what these tests exercise, and the pair carries no state
    any assertion reads — every observation is made through the bridge's own dedup store
    or through the fakes — so one pair serves the whole module.
    """
    worker_port = _free_port()
    dispatcher_port = _free_port()
    triggers_path = tmp_path_factory.mktemp("teams_bridge_triggers") / "triggers.yml"
    _write_triggers(triggers_path, worker_port)

    worker_proc: subprocess.Popen | None = None
    dispatcher_proc: subprocess.Popen | None = None
    try:
        worker_env = {
            **os.environ,
            "DISPATCH_WORKER_PORT": str(worker_port),
            "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
            # Repo root + the render's config one level down, exactly as the
            # dispatch_worker compose template wires the deployed worker.
            "OSPREY_PROJECT_DIR": str(built_repo),
            "CONFIG_FILE": str(built_repo / "build" / "config.yml"),
            # The worker's own per-run wall-clock cap. Must match the bridge's
            # DISPATCH_TIMEOUT_SEC or its poll_budget floor is validated against the
            # wrong number; set from one constant so they cannot drift.
            "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
            "CLAUDECODE": "",
        }
        worker_proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
            cwd=str(built_repo),
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(f"http://127.0.0.1:{worker_port}/health", HEALTH_TIMEOUT_SEC, worker_proc)

        dispatcher_env = {
            **os.environ,
            "TRIGGERS_YML": str(triggers_path),
            "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
            "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
            "FASTMCP_TRANSPORT": "http",
            "FASTMCP_PORT": str(dispatcher_port),
            "FASTMCP_HOST": "127.0.0.1",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": str(dispatcher_port),
            "CLAUDECODE": "",
        }
        dispatcher_proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "osprey.dispatch"],
            cwd=str(built_repo),
            env=dispatcher_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(
            f"http://127.0.0.1:{dispatcher_port}/health", HEALTH_TIMEOUT_SEC, dispatcher_proc
        )

        yield {
            "dispatcher_url": f"http://127.0.0.1:{dispatcher_port}",
            "worker_url": f"http://127.0.0.1:{worker_port}",
            "repo": built_repo,
        }
    finally:
        _terminate(dispatcher_proc)
        _terminate(worker_proc)


# ---------------------------------------------------------------------------
# The bridge under test, booted IN-PROCESS through its own entrypoint seams
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_state(tmp_path: Path) -> Path:
    """Per-test directory for the bridge's dedup/history stores.

    Function-scoped so one test's stores are not another's noise.
    """
    state = tmp_path / "bridge-state"
    state.mkdir()
    return state


def _bridge_env(*, state_dir: Path, dispatcher_url: str, worker_url: str) -> dict[str, str]:
    """A complete bridge environment, exactly as compose would render one.

    The two store paths MUST be overridden: they default to ``/data/*.json``, which exists
    only inside the bridge container.
    """
    return {
        "TEAMS_APP_ID": APP_ID,
        "TEAMS_APP_SECRET": APP_SECRET,
        "TEAMS_TENANT_ID": TENANT_ID,
        "TEAMS_SERVICEBUS_CONNECTION_STRING": CONNECTION_STRING,
        "TEAMS_SERVICEBUS_QUEUE": QUEUE_NAME,
        "APP_VERSION_DISPLAY": VERSION_TAG,
        "DISPATCH_TRIGGER": TRIGGER_NAME,
        "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
        "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
        "DISPATCHER_URL": dispatcher_url,
        "WORKER_URL": worker_url,
        "DEDUP_PATH": str(state_dir / "dedup.json"),
        "HISTORY_PATH": str(state_dir / "history.json"),
        # poll_budget must stay >= worker_timeout or CoreConfig refuses to build.
        "POLL_BUDGET": str(WORKER_RUN_CAP_SEC),
        "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
        "POLL_INTERVAL": "1",
        "DRAIN_INTERVAL": "5",
        "BRIDGE_TRUST_ENV": "0",
    }


def _set_bridge_env(
    monkeypatch: pytest.MonkeyPatch, *, state_dir: Path, dispatch: Mapping[str, Any]
) -> None:
    """Put :func:`_bridge_env` in place for ``config_from_env``.

    Set through ``monkeypatch`` and read back by
    :func:`~osprey.bridges.teams.__main__.config_from_env`, so the test boots the bridge
    the way the container does — ``require_boot``'s startup validation included — rather
    than hand-building a config object that could satisfy the types and skip the checks.

    ``TEAMS_CLOUD`` is *removed* rather than set: the commercial default is what a
    deployment that names no cloud gets, and an ambient value from the developer's shell
    would quietly re-point the login host.
    """
    monkeypatch.delenv("TEAMS_CLOUD", raising=False)
    for name, value in _bridge_env(
        state_dir=state_dir,
        dispatcher_url=dispatch["dispatcher_url"],
        worker_url=dispatch["worker_url"],
    ).items():
        monkeypatch.setenv(name, value)


@dataclass
class RunningBridge:
    """A bridge running in a daemon thread, plus whatever killed it."""

    wiring: Wiring
    thread: threading.Thread
    state_dir: Path
    receiver: OrderedReceiver
    failure: list[BaseException] = field(default_factory=list)

    @property
    def dedup(self) -> dict[str, dict[str, Any]]:
        """The persisted dedup store, or ``{}`` before it exists.

        Safe to read while the bridge runs: every store write is a tmp-file-plus-rename,
        so a partially written file is never observable.
        """
        try:
            data = json.loads((self.state_dir / "dedup.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


def _drain_thread_alive() -> bool:
    """Whether the engine's single drain thread is still running.

    It is a daemon named ``bridge-drain``, so it never shows up as a stuck process — it
    shows up as a store being written by a bridge the test thinks has stopped.
    """
    return any(t.name == "bridge-drain" and t.is_alive() for t in threading.enumerate())


@contextlib.contextmanager
def _running_bridge(
    state_dir: Path,
    *,
    receiver: OrderedReceiver,
    token_http: httpx.Client,
    connector_http: httpx.Client,
    worker_http: httpx.Client,
) -> Iterator[RunningBridge]:
    """Boot the bridge in-process through ``build_wiring``/``run``, and stop it on the way out.

    In-process rather than ``python -m osprey.bridges.teams`` on purpose: it gives the
    test the ``Wiring`` — hence the stop event and all four seams — with no signal
    handling in the way. ``run`` re-validates the config, so this cannot boot a
    half-configured bridge that the container path would have refused.

    Shutdown is one ``set()``: the receive loop waits at most ``SLOT_TIMEOUT_SEC`` for a
    handler slot and the fake's idle pull is shorter still, so the loop observes the stop
    within about a second without the test reaching into the pull at all.
    """
    cfg = config_from_env()
    wiring = build_wiring(
        cfg,
        receiver_factory=lambda _cfg: receiver,
        token_http=token_http,
        connector_http=connector_http,
        worker_http=worker_http,
    )
    failure: list[BaseException] = []

    def _serve() -> None:
        try:
            run(wiring)
        except BaseException as exc:  # noqa: BLE001 - re-raised from the test thread
            failure.append(exc)

    thread = threading.Thread(target=_serve, name="teams-bridge-e2e", daemon=True)
    thread.start()
    bridge = RunningBridge(
        wiring=wiring, thread=thread, state_dir=state_dir, receiver=receiver, failure=failure
    )
    completed = False
    try:
        yield bridge
        completed = True
    finally:
        # One set() ends ingestion and the drain: serve_events returns, and run_forever's
        # own finally shuts the runtime down on the same event.
        wiring.stop.set()
        thread.join(BRIDGE_JOIN_TIMEOUT_SEC)
        deadline = time.monotonic() + BRIDGE_JOIN_TIMEOUT_SEC
        while _drain_thread_alive() and time.monotonic() < deadline:
            time.sleep(0.5)
        # Only when the body itself succeeded — otherwise this would mask the real
        # failure with a secondary one.
        if completed and failure:
            raise AssertionError(f"bridge thread crashed: {failure[0]!r}") from failure[0]
        if completed and thread.is_alive():
            raise AssertionError("the bridge thread did not stop on its stop event")
        if completed and _drain_thread_alive():
            raise AssertionError(
                "the drain thread outlived its stop event and is still holding the dedup store"
            )


# ---------------------------------------------------------------------------
# Barriers and shared assertions
# ---------------------------------------------------------------------------


def _await(bridge: RunningBridge, what: str, probe: Callable[[], Any], timeout: float) -> Any:
    """Poll ``probe`` until it returns non-``None``, or fail informatively.

    ``None`` and only ``None`` means "not yet", so a probe may legitimately answer a
    falsy value. Checks the bridge thread on every pass: a crashed or exited bridge is
    reported as such immediately instead of surfacing as an opaque timeout minutes later.
    """
    deadline = time.monotonic() + timeout
    while True:
        if bridge.failure:
            raise AssertionError(f"bridge thread died before {what}: {bridge.failure[0]!r}")
        if not bridge.thread.is_alive():
            raise AssertionError(f"bridge thread exited before {what}")
        value = probe()
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out after {timeout:.0f}s waiting for {what}\n"
                f"  dedup: {json.dumps(bridge.dedup, indent=2, default=str)[:2000]}"
            )
        time.sleep(0.5)


def _wait_for_dispatch(bridge: RunningBridge, key: str) -> dict[str, Any]:
    """Wait until ``key`` is claimed AND carries a ``run_id``; return the entry.

    ``run_id`` is persisted the moment the dispatcher handshake yields one, before the
    agent does any work, so this is the model-independent observation of "a dispatch was
    fired for this activity". Nothing here reads the run's outcome.
    """

    def probe() -> dict[str, Any] | None:
        entry = bridge.dedup.get(key)
        if isinstance(entry, dict) and entry.get("run_id"):
            return entry
        return None

    return _await(bridge, f"dedup entry {key} to carry a run_id", probe, RUN_ID_TIMEOUT_SEC)


def _assert_settled_once(
    receiver: OrderedReceiver,
    message: FakeQueueMessage,
    *,
    min_posts: int = 0,
    max_posts: int | None = None,
) -> None:
    """Assert the loop registered ``message``, handled it, and completed it — in that order.

    The call log gives the order and the exactly-once part (no second settlement, no
    dead-letter). The two post bounds are what make it a claim about the work in between:
    a claimed activity must have had at least its ack accepted inside the window
    (``min_posts=1``), and one the adapter ignored or de-duplicated must have had nothing
    accepted at all (``max_posts=0``).

    The upper bound is deliberately open for a claimed activity. How many activities a
    settled dispatch posts is the run's business — one ack plus an answer, an error notice,
    or several answer chunks — and pinning it here would make this ordering proof fail
    whenever the run's outcome changed, which is exactly the model dependence the lane
    exists without.
    """
    assert receiver.calls_for(message) == ["register", "complete"], (
        f"{message.id} was not registered-then-completed exactly once: "
        f"{receiver.calls_for(message)!r} (full log: {receiver.calls!r})"
    )
    assert receiver.dead_lettered == [], f"a message was dead-lettered: {receiver.dead_lettered!r}"
    held = receiver.posts_while_held(message)
    assert held >= min_posts, (
        f"{message.id} was held across {held} accepted post(s), expected at least "
        f"{min_posts}: {receiver.connector.posted_text!r}"
    )
    if max_posts is not None:
        assert held <= max_posts, (
            f"{message.id} was held across {held} accepted post(s), expected at most "
            f"{max_posts}: {receiver.connector.posted_text!r}"
        )


# ---------------------------------------------------------------------------
# The deterministic proofs
# ---------------------------------------------------------------------------


def test_a_mentioned_channel_message_is_acked_and_dispatched(
    receiver: OrderedReceiver,
    connector: FakeConnectorServer,
    token: FakeTokenServer,
    token_http: httpx.Client,
    connector_http: httpx.Client,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel message that @mentions the bot is acked, claimed and dispatched.

    The whole outbound leg is asserted from the Connector's side rather than from a mock's
    call list: the bearer really was minted against the login host, the reply really was
    addressed to the conversation and activity the question arrived on, and its text is
    the adapter's own ``ack_text`` — which in a channel carries no quote prefix, because
    the reply is already threaded under the question. The claim's persisted shape is
    asserted too, because every later step (the drain, a post-restart reconcile) works off
    that entry alone, long after the activity is gone.
    """
    question = "what is the beam current?"
    act = channel_activity(question, service_url=connector.base_url)
    key = _dedup_key(act)

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        bridge_state,
        receiver=receiver,
        token_http=token_http,
        connector_http=connector_http,
        worker_http=worker_http,
    ) as bridge:
        message = receiver.enqueue(act)
        entry = _wait_for_dispatch(bridge, key)

        assert entry[MS_SERVICE_URL] == connector.base_url
        assert entry[MS_CONVERSATION_ID] == CHANNEL_ID
        assert entry[MS_ACTIVITY_ID] == act["id"]
        assert entry[MS_CONVERSATION_TYPE] == "channel"
        assert entry[MS_TENANT_ID] == TENANT_ID
        assert entry["sender_id"] == SENDER_ID
        # The @mention is addressing, not content: it must not reach the agent.
        assert entry["text"] == question
        # A channel root post IS its own thread root, so the transcript key is the
        # channel id with that root's id appended — the same string Teams gives every
        # reply in the thread as its conversation id.
        assert entry["history_key"] == f"{CHANNEL_ID};messageid={act['id']}"

        posted = connector.wait_for_posted(1)
        assert posted[0].text == ack_text(VERSION_TAG)
        assert posted[0].conversation_id == CHANNEL_ID
        assert posted[0].reply_to == act["id"]
        assert posted[0].body["textFormat"] == "markdown"
        assert posted[0].auth == f"Bearer {ACCESS_TOKEN}"
        # The bearer came from the client-credentials exchange the product built, on the
        # tenant the config named — the token seam redirects that request, it does not
        # replace it.
        assert [r.tenant for r in token.requests] == [TENANT_ID]
        assert token.requests[0].client_id == APP_ID

        receiver.wait_for_settled(1, HANDLED_TIMEOUT_SEC)
        _assert_settled_once(receiver, message, min_posts=1)


def test_a_channel_message_without_a_mention_is_ignored(
    receiver: OrderedReceiver,
    connector: FakeConnectorServer,
    token_http: httpx.Client,
    connector_http: httpx.Client,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a channel, a message that does not @mention the bot is ignored.

    The negative half is the point — no claim, no dispatch, no post — but on its own it
    would also pass against a bridge that was simply dead, so the same channel then gets a
    message that DOES mention the bot and that one must dispatch. The pair is what makes
    this a proof that the mention filter discriminates rather than a proof that nothing
    works.
    """
    plain = channel_activity(
        "no mention here, just chatter between humans",
        service_url=connector.base_url,
        mention=False,
    )
    question = "what is the beam current?"
    mentioned = channel_activity(question, service_url=connector.base_url)

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        bridge_state,
        receiver=receiver,
        token_http=token_http,
        connector_http=connector_http,
        worker_http=worker_http,
    ) as bridge:
        ignored = receiver.enqueue(plain)
        # The barrier the negative assertions need: the message is settled, so anything
        # the bridge was going to do for it, it has already done.
        receiver.wait_for_settled(1, IGNORED_TIMEOUT_SEC)

        assert _dedup_key(plain) not in bridge.dedup, (
            "an unmentioned channel message was claimed for dispatch: "
            f"{bridge.dedup.get(_dedup_key(plain))!r}"
        )
        assert connector.posted == [], (
            f"the bridge posted into a channel it was not addressed in: {connector.posted_text!r}"
        )
        _assert_settled_once(receiver, ignored, max_posts=0)

        # Positive control, same channel, same bridge: a mention IS dispatched.
        claimed = receiver.enqueue(mentioned)
        entry = _wait_for_dispatch(bridge, _dedup_key(mentioned))

        assert entry["text"] == question
        assert connector.wait_for_posted(1)[0].text == ack_text(VERSION_TAG)
        receiver.wait_for_settled(2, HANDLED_TIMEOUT_SEC)
        _assert_settled_once(receiver, claimed, min_posts=1)


def test_a_personal_message_is_answered_without_a_mention(
    receiver: OrderedReceiver,
    connector: FakeConnectorServer,
    token_http: httpx.Client,
    connector_http: httpx.Client,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a 1:1 chat, a message with no @mention is a question and is dispatched.

    The conversation TYPE is the whole difference: the identical text is ignored in a
    channel (the proof above) and dispatched here. The ack differs too, and for a reason
    the user sees — a 1:1 chat has no threads, so the reply opens with a blockquote of the
    question it answers. That prefix is built by the product's own ``quote_prefix`` over
    the persisted entry, which is the same input the adapter used; the question itself is
    asserted separately, so a wrong ``text`` cannot make the prefix agree with itself.
    """
    question = "how many bunches are stored?"
    act = personal_activity(question, service_url=connector.base_url)
    key = _dedup_key(act)

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        bridge_state,
        receiver=receiver,
        token_http=token_http,
        connector_http=connector_http,
        worker_http=worker_http,
    ) as bridge:
        message = receiver.enqueue(act)
        entry = _wait_for_dispatch(bridge, key)

        assert entry[MS_CONVERSATION_TYPE] == "personal"
        assert entry[MS_CONVERSATION_ID] == PERSONAL_CONVERSATION_ID
        assert entry["text"] == question
        # A 1:1 chat is one continuous conversation, so it keys on the conversation
        # itself: there is no thread to scope a transcript to.
        assert entry["history_key"] == PERSONAL_CONVERSATION_ID

        posted = connector.wait_for_posted(1)
        assert posted[0].text == quote_prefix(entry) + ack_text(VERSION_TAG)
        assert quote_prefix(entry) == f"> {question}\n\n"
        assert posted[0].conversation_id == PERSONAL_CONVERSATION_ID
        assert posted[0].reply_to == act["id"]

        receiver.wait_for_settled(1, HANDLED_TIMEOUT_SEC)
        _assert_settled_once(receiver, message, min_posts=1)


def test_a_redelivered_activity_is_a_duplicate_and_is_acked_once(
    receiver: OrderedReceiver,
    connector: FakeConnectorServer,
    token_http: httpx.Client,
    connector_http: httpx.Client,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same activity delivered twice is claimed once, acked once and dispatched once.

    Service Bus is at-least-once — a lock that expires while a handler is still running is
    redelivered — so this is the failure mode the engine's dedup claim exists for. The
    redelivery carries the same activity under a new broker message id, which is exactly
    what the broker does, and the second delivery must reach ``handle_event`` and be
    reported ``duplicate`` rather than be filtered out before it: a bridge that dropped
    redeliveries at the queue would satisfy every count below and lose real messages.

    The first delivery is waited out to SETTLEMENT (the handler polls the run to terminal
    before it completes the message), which is what makes the "nothing further happened"
    assertions about a settled entry rather than a racing one.
    """
    act = personal_activity("is the shutter open?", service_url=connector.base_url)
    key = _dedup_key(act)

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    caplog.set_level(logging.DEBUG, logger="osprey.bridges.teams.receiver")
    with _running_bridge(
        bridge_state,
        receiver=receiver,
        token_http=token_http,
        connector_http=connector_http,
        worker_http=worker_http,
    ) as bridge:
        first = receiver.enqueue(act)
        entry = _wait_for_dispatch(bridge, key)
        receiver.wait_for_settled(1, HANDLED_TIMEOUT_SEC)
        settled_posts = len(connector.posted)
        assert settled_posts >= 1, "the first delivery posted nothing at all; nothing is settled"

        again = receiver.redeliver(act)
        assert again.id != first.id, "the redelivery reused the first delivery's message id"
        receiver.wait_for_settled(2, HANDLED_TIMEOUT_SEC)

        assert [k for k in bridge.dedup if k == key] == [key]
        second = bridge.dedup[key]
        assert second["run_id"] == entry["run_id"], (
            f"the redelivery started a second run: {entry['run_id']!r} -> {second['run_id']!r}"
        )
        ack = quote_prefix(second) + ack_text(VERSION_TAG)
        assert connector.posted_text.count(ack) == 1, (
            f"the ack was posted {connector.posted_text.count(ack)} time(s): "
            f"{connector.posted_text!r}"
        )
        assert len(connector.posted) == settled_posts, (
            "the redelivery posted into the conversation again: "
            f"{[a.text for a in connector.posted[settled_posts:]]!r}"
        )
        # The engine saw it and called it a duplicate — the status the loop logs.
        assert "service bus message duplicate" in caplog.text

        _assert_settled_once(receiver, first, min_posts=1)
        _assert_settled_once(receiver, again, max_posts=0)


# ---------------------------------------------------------------------------
# The production receiver, constructed for real (no broker, no pull)
# ---------------------------------------------------------------------------


def test_make_receiver_opens_a_peek_lock_receiver_with_a_renewer_covering_the_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real :func:`make_receiver` builds the objects it documents, against the real SDK.

    Everything above injects the queue seam, so nothing else in this module ever runs the
    factory the container actually uses. This does — with the ``azure.servicebus`` classes
    imported plainly, which is what makes the ``teams`` extra load-bearing for the lane.

    **Construct only.** The SDK's client, receiver and renewer are all lazy, so none of
    the three dials anything until a pull; ``receive`` is therefore never called and the
    endpoint is loopback, so a regression that started connecting at construction time
    fails here rather than reaching a broker. What is asserted is the two decisions the
    factory makes and nothing downstream can recover: the receive mode must be peek-lock
    (at-most-once delivery would settle a message before the agent had answered), and the
    renewer must keep renewing for the whole dispatch wait plus the time it takes to post
    the answer.

    Both are read from the keywords the factory passes, not from the objects it gets
    back: the SDK stores them on private attributes whose names are its own business, and
    a test that asserted on those would pin the SDK's internals rather than this
    module's decisions. The two recorders below subclass the real classes and construct
    the real objects, so the factory still runs against the SDK it ships with — the
    import inside :func:`make_receiver` is function-local, so rebinding the names on the
    ``azure.servicebus`` package is what it resolves.
    """
    receiver_keywords: list[dict[str, Any]] = []
    renewer_keywords: list[dict[str, Any]] = []

    class RecordingClient(ServiceBusClient):
        """A real client that records the keywords its queue receiver is opened with."""

        def get_queue_receiver(self, **kwargs: Any) -> Any:
            receiver_keywords.append(dict(kwargs))
            return super().get_queue_receiver(**kwargs)

    class RecordingRenewer(AutoLockRenewer):
        """A real renewer that records the keywords it was constructed with."""

        def __init__(self, **kwargs: Any) -> None:
            renewer_keywords.append(dict(kwargs))
            super().__init__(**kwargs)

    monkeypatch.setattr(azure.servicebus, "ServiceBusClient", RecordingClient)
    monkeypatch.setattr(azure.servicebus, "AutoLockRenewer", RecordingRenewer)

    cfg = TeamsBridgeConfig.from_env(
        _bridge_env(
            state_dir=tmp_path,
            # Never dialled: the factory builds no HTTP client, and this config is used
            # for nothing else.
            dispatcher_url="http://127.0.0.1:1",
            worker_url="http://127.0.0.1:1",
        )
    )

    built = make_receiver(cfg)
    try:
        assert isinstance(built, ServiceBusQueueReceiver)
        assert built.client.fully_qualified_namespace == "127.0.0.1"
        assert receiver_keywords == [
            {"queue_name": QUEUE_NAME, "receive_mode": ServiceBusReceiveMode.PEEK_LOCK}
        ]
        # Straight from the kwarg: poll_budget is how long the engine may wait for a run,
        # and SETTLE_MARGIN_SEC is the posting that follows it. A renewer that stopped
        # short would drop the lock on precisely the runs that took longest.
        assert len(renewer_keywords) == 1
        assert renewer_keywords[0]["max_lock_renewal_duration"] == (
            cfg.core.poll_budget + SETTLE_MARGIN_SEC
        )
    finally:
        built.close()
