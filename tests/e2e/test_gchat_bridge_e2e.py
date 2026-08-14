"""Real-stack e2e for the Google Chat bridge: a Pub/Sub emulator and a real dispatcher.

Every unit test of ``osprey.bridges.google_chat`` drives the adapter over a mocked
discovery service, so all of them assert OSPREY's half of a two-party contract against
OSPREY's own idea of Chat's wire format. This module is the instrument for the rest of
the stack: Google's own Pub/Sub emulator in a container, a real ``osprey.dispatch``
dispatcher/worker pair as subprocesses, and the bridge itself booted **through its own
entrypoint seams** — the same ``build_wiring``/``run`` the container's ``main`` calls.
What it proves is that the adapter's space-type rules, its mention filter, its
exactly-once claim and its restart behaviour hold when the queue is a real one.

The four tests here are **deterministic**: none of their assertions reads model output,
so they pass with no provider key configured at all. A dispatch is observed as "the
dedup entry now carries a ``run_id``" — the engine persists that the moment the
dispatcher handshake yields one, long before the agent finishes — so a run that errors
out for want of an API key proves exactly as much as one that answers. The one Chat-side
string this tier compares against is ``ACK_TEXT``, the adapter's own constant, imported
from the product rather than re-spelled. The reply half (what the agent actually said)
is the agentic lane's, and lives below the marked section at the bottom of this file.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call reachable from this file)
----------------------------------------------------------------------------
**This module issues no container-runtime command at all.** It creates no container, no
volume and no image, and contains no call to the runtime CLI. The single container the
lane needs — the Pub/Sub emulator, exact-named ``osprey-e2e-gchat-pubsub`` off
``RESOURCE_PREFIX`` — is created and removed by ``tests/e2e/fixtures/
gchat_pubsub_emulator.py``, whose own CONTAINER-OPS SAFETY block governs it: two
exact-named removal sites, both best-effort, no volumes, no prune, no ``-a``/``--all``,
no wildcard match. Nothing here ever runs ``system prune``, ``volume prune``,
``container prune``, ``image prune``, or a ``volume rm``. The host's unrelated stacks
are untouched.

Everything else in the stack is a plain host process or a loopback HTTP server: the
dispatcher and worker are subprocesses of this interpreter, and the Chat and GCS fakes
are ``http.server`` instances on ephemeral loopback ports.

----------------------------------------------------------------------------
The four injected seams, and why all four are needed
----------------------------------------------------------------------------
:func:`~osprey.bridges.google_chat.__main__.build_wiring` exposes exactly four seams,
and this module supplies every one — which is what lets the lane run with no Google
credentials, no service-account key and no reachable Google endpoint, while still
executing the adapter's real wiring:

``subscriber_factory``
    :class:`SubscriberSeam` — builds a REAL ``pubsub_v1.SubscriberClient`` against the
    emulator. See the TLS note below; it is not the spelling the library's own
    ``PUBSUB_EMULATOR_HOST`` support uses.
``chat_service``
    ``FakeChatServer.service()`` — the discovery-service OBJECT, handed straight to
    ``ChatClient``'s constructor. This seam *bypasses* rather than replaces:
    ``ChatClient`` builds a real service the moment it is given none, which imports
    ``googleapiclient`` and reads the key file, so a seam applied after the fact would
    already have failed.
``gcs_client``
    ``FakeGcsServer.client()`` — the storage-client OBJECT, which ``build_wiring`` binds
    into BOTH publishers through a ``client_factory`` that ignores its cfg argument.
``worker_http``
    One ``httpx.Client`` shared by both publishers' worker byte-route fetches AND the
    prior-artifact fetcher's worker leg — one client, one pool, for the one internal
    service both of them read.

``GCHAT_SA_KEY`` is therefore pointed at a path that deliberately **does not exist**: if
any of the four seams ever stopped being honoured, the resulting read would fail loudly
here instead of silently authenticating against something.

**The one place a real outbound request is possible**, and it is the product's, not the
harness's: the prior-artifact fetcher falls back to the descriptor's ``public_url`` when
the worker leg fails, and that URL is a ``storage.googleapis.com`` literal the publisher
builds itself. So a FAILED worker-leg re-fetch costs one anonymous GET toward Google
(``PRIOR_FETCH_TIMEOUT``, 30 s) before it gives up. It is bounded, it is unauthenticated,
and it can only ever make the agentic artifact proof SLOWER — never green: that proof
asserts the worker leg answered, which is the branch that keeps the fallback unreached.

----------------------------------------------------------------------------
Two things about the emulator that are easy to get wrong
----------------------------------------------------------------------------
**1. The library's own emulator spelling negotiates TLS and fails.** The documented path
(``PUBSUB_EMULATOR_HOST``, which makes ``pubsub_v1.SubscriberClient`` pass
``AnonymousCredentials`` plus an ``api_endpoint``) still builds a *secure* channel in
``google-api-core``, and the emulator speaks plaintext gRPC — so every pull dies in an
SSL handshake (``WRONG_VERSION_NUMBER``) and the subscription simply never delivers,
with no exception reaching the bridge. :func:`_pubsub_subscriber` therefore constructs
the transport over an explicit ``grpc.insecure_channel``, which is also why it passes no
credentials at all rather than anonymous ones.

**2. Shutting the bridge down needs the pull cancelled from the test side.**
``serve`` blocks on ``future.result(timeout=wake_interval)`` and observes the stop event
only at that boundary; ``serve_events`` does not expose ``wake_interval``, so it is the
production 60 s. Setting the stop event alone would therefore cost a minute per bridge
lifetime. :meth:`SubscriberSeam.cancel_pull` cancels the streaming-pull future — exactly
what ``serve``'s own ``finally`` does on the way out — which collapses that wake to
immediate. Nothing under test changes: every assertion is made while the bridge runs.

----------------------------------------------------------------------------
Two alarming-looking log lines that are expected
----------------------------------------------------------------------------
``messages.get failed for spaces/.../messages/...; no reply context`` — the reply-context
fallback re-reads the current message once when the event carried no quote snapshot, and
the fake only serves messages that were POSTED through it, so the read 404s. Reply
context is enrichment by contract: the warning is the guarded path working.

``pub/sub streaming pull ended; stopping ingestion`` at teardown — the consequence of
cancelling the future to collapse the supervise wake (note 2 above). ``serve`` cannot
tell a test-side cancel from a stream that ended on its own, and either way its next act
is to return, which is what shutdown wants.

----------------------------------------------------------------------------
Host ports
----------------------------------------------------------------------------
This lane pins **none**, deliberately, and therefore adds nothing to the set its sibling
e2e modules pin (5064, 15080, 18090, 18095, 18099, 18101-18107, 19081, 25080, 25432).
Every listener it needs is per-run and picked free: the emulator's published port comes
from the fixture's own free-port pick, the dispatcher and worker take :func:`_free_port`
(they are per-run subprocesses with no need to be predictable, and a pin would collide
with a developer's own running stack), and the Chat and GCS fakes bind ``127.0.0.1:0``.

Gating, and the three ways this lane can skip. It needs a container runtime whose daemon
is actually running (not just the CLI installed), the emulator image, and the ``gchat``
extra installed — see :data:`GCHAT_EXTRA`, which is the one a CI job is most likely to
forget, and the one that would otherwise turn this lane into a green that proves
nothing. Runtime is ``docker`` by default; set ``OSPREY_E2E_RUNTIME=podman`` to run
against podman instead (any other value fails at collection time with a clear error),
matching ``tests/e2e/test_nextcloud_talk_bridge_e2e.py`` and the fixture module, so one
env var drives the whole lane.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.bridges.core import PNG_MAGIC
from osprey.bridges.google_chat.__main__ import Wiring, build_wiring, config_from_env, run
from osprey.bridges.google_chat.client import MAX_CHARS, REPLY_OPTION
from osprey.bridges.google_chat.events import (
    GC_ATTACHMENTS,
    GC_IS_DM,
    GC_MESSAGE_NAME,
    GC_SPACE,
    GC_THREAD,
)
from osprey.bridges.google_chat.ops import (
    ACK_TEXT,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    SUPERSEDED_TEXT,
)
from tests.e2e.fixtures.gchat_chat_fake import FakeChatServer, PostedMessage
from tests.e2e.fixtures.gchat_gcs_fake import PUBLIC_HOST, FakeGcsServer

# Importing the emulator module is also the runtime-selection gate: it validates
# OSPREY_E2E_RUNTIME at import time and raises on an unsupported value, so this lane and
# the fixture directory cannot disagree about which runtime they are using.
from tests.e2e.fixtures.gchat_pubsub_emulator import (
    IMAGE,
    RUNTIME,
    PubSubEmulator,
    image_available,
    runtime_available,
)

GCHAT_EXTRA = "gchat"
"""The optional-dependency group this lane cannot run without.

Unlike every other test of ``osprey.bridges.google_chat`` — which run with no Google
package installed at all, because the product imports them lazily — this one drives a
REAL Pub/Sub client against the emulator and therefore needs ``google-cloud-pubsub`` and
its gRPC stack for real. Named in the skip reason so a run that installed only ``dev``
says so, instead of failing later and deeper with an ``ImportError`` raised inside the
bridge thread. **CI must install this extra**, or the lane skips its way to a green that
proves nothing."""


def _gchat_extra_installed() -> bool:
    """Whether the Pub/Sub client stack this lane needs is importable."""
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("grpc", "google.cloud.pubsub_v1")
        )
    except (ImportError, ValueError):
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which(RUNTIME) is None,
        reason=f"{RUNTIME} CLI not installed (set OSPREY_E2E_RUNTIME to choose a runtime)",
    ),
    pytest.mark.skipif(
        not _gchat_extra_installed(),
        reason=(
            f"the {GCHAT_EXTRA!r} extra is not installed; this lane drives a real "
            f"Pub/Sub client (uv sync --extra dev --extra {GCHAT_EXTRA})"
        ),
    ),
]
# No ``dockerbuild`` marker: this lane builds no image. The emulator image is public and
# is pulled once by the fixture if absent, which that marker is not about.


# ---------------------------------------------------------------------------
# The Chat app's identity and the spaces the tests speak into
# ---------------------------------------------------------------------------

APP_ID = "users/1234567890"
"""The Chat app's own user id — what ``GCHAT_APP_ID`` carries and what a room mention
has to name for the message to be a question for this app."""

MENTION_TOKEN = "@OSPREY"
"""The literal text a mention occupies in ``message.text``. Chat renders the display
name; only the annotation's offsets make it strippable, which is what the room proof's
``text`` assertion exercises."""

HUMAN_SENDER = "users/human-9001"
HUMAN_DISPLAY = "Test Operator"

DM_SPACE_TYPE = "DIRECT_MESSAGE"
ROOM_SPACE_TYPE = "SPACE"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

HEALTH_TIMEOUT_SEC = 45.0
"""Per-subprocess wait for the dispatcher's / worker's ``/health``."""

RUN_ID_TIMEOUT_SEC = 150.0
"""Wait for a claimed message's dedup entry to carry a ``run_id`` — one dispatcher POST
plus the accept handshake, well short of any agent work."""

HANDLED_TIMEOUT_SEC = 300.0
"""Wait for the bridge to finish handling one delivered payload.

Generous because "finished" for a CLAIMED message means the whole dispatch settled: the
callback polls the run to terminal before it returns. Comfortably past
:data:`WORKER_RUN_CAP_SEC` plus the ack and answer posts."""

IGNORED_TIMEOUT_SEC = 90.0
"""Wait for the bridge to finish handling a payload it will IGNORE.

The barrier the negative proof needs: once the callback has returned, anything the
bridge was going to do for that message it has already done, so "nothing happened"
cannot be confused with "nothing has happened yet"."""

BRIDGE_JOIN_TIMEOUT_SEC = 90.0
"""Wait for a stopped bridge's threads.

Not politeness. The drain thread and the Pub/Sub library's callback threads are daemons,
so an outliving one does not block the interpreter — it shows up as a dedup store being
written by a bridge the test believes is DOWN, which is precisely the vacuity the
restart proof exists to rule out. Shutdown therefore waits for the threads themselves,
not merely for ``run`` to return."""

WORKER_RUN_CAP_SEC = 90
"""``DISPATCH_TIMEOUT_SEC`` for the worker AND the bridge's ``poll_budget`` floor.

Bounds how long a wedged run can hold a Pub/Sub callback thread. The two must agree: the
bridge's ``CoreConfig`` refuses to build when ``POLL_BUDGET < DISPATCH_TIMEOUT_SEC``."""

BUILD_TIMEOUT_SEC = 600
"""Wall-clock cap on the one ``osprey build`` this module runs."""

DISPATCH_TOKEN = "google-chat-e2e-token"
"""Shared dispatcher<->worker bearer for this run. Both halves are local subprocesses."""

TRIGGER_NAME = "google-chat-e2e"
"""Name of the deterministic trigger :func:`_write_triggers` generates."""

GCS_BUCKET = "osprey-e2e-gchat-artifacts"
GCS_PROJECT = "osprey-e2e-gchat"


# ---------------------------------------------------------------------------
# Chat wire events — built here, never by the product
# ---------------------------------------------------------------------------


def _mention_annotation(text_before: str) -> dict[str, Any]:
    """A ``USER_MENTION`` annotation naming THIS app, with real offsets.

    Offsets rather than a bare annotation on purpose: the adapter prefers Chat's
    pre-stripped ``argumentText`` when it is present and falls back to slicing these
    spans, and shipping both is what a real Chat event does.
    """
    return {
        "type": "USER_MENTION",
        "startIndex": len(text_before),
        "length": len(MENTION_TOKEN),
        "userMention": {
            "type": "MENTION",
            "user": {"name": APP_ID, "displayName": "OSPREY", "type": "BOT"},
        },
    }


def _chat_event(
    *,
    space: str,
    space_type: str,
    message_id: str,
    question: str,
    thread: str,
    mention: bool,
) -> dict[str, Any]:
    """One classic-shape Chat ``MESSAGE`` event, as Pub/Sub would carry it.

    The classic envelope (``{"type": "MESSAGE", "message": {...}}``) rather than the
    Workspace add-on nesting: both are parsed, and the add-on shape's graft rules are
    pinned by the unit suite against hand-built payloads. What this lane adds is the
    trip through a real queue, which is identical for either shape.
    """
    message: dict[str, Any] = {
        "name": f"{space}/messages/{message_id}",
        "sender": {"name": HUMAN_SENDER, "displayName": HUMAN_DISPLAY, "type": "HUMAN"},
        "createTime": "2026-01-01T00:00:00Z",
        "thread": {"name": thread},
        "space": {"name": space, "spaceType": space_type},
        "text": f"{MENTION_TOKEN} {question}" if mention else question,
        "argumentText": question,
    }
    if mention:
        message["annotations"] = [_mention_annotation("")]
    return {"type": "MESSAGE", "message": message}


def _payload(event: Mapping[str, Any]) -> bytes:
    """Serialize an event to the exact bytes published.

    ``sort_keys`` so a caller that publishes the SAME event twice publishes
    byte-identical payloads — which is what makes the exactly-once proof a redelivery
    rather than two similar messages.
    """
    return json.dumps(event, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# The Pub/Sub subscriber seam
# ---------------------------------------------------------------------------


def _pubsub_subscriber(host: str) -> Any:
    """A real ``pubsub_v1.SubscriberClient`` speaking plaintext gRPC to ``host``.

    The transport is built over an explicit ``grpc.insecure_channel`` and the client is
    given no credentials at all. That is NOT an embellishment of the library's own
    emulator support — see note 1 in the module docstring: the ``PUBSUB_EMULATOR_HOST``
    path still negotiates TLS against a plaintext emulator, and the resulting handshake
    failure is invisible to the bridge (no exception; the subscription simply never
    delivers).

    Imported inside the function, like the product's own default factory, so collecting
    this module needs no Google packages installed.
    """
    import grpc
    from google.cloud import pubsub_v1
    from google.pubsub_v1.services.subscriber.transports.grpc import SubscriberGrpcTransport

    transport = SubscriberGrpcTransport(channel=grpc.insecure_channel(host))
    return pubsub_v1.SubscriberClient(transport=transport)


class SubscriberSeam:
    """The ``subscriber_factory`` seam: a real emulator client, plus two test affordances.

    One instance per bridge lifetime (the client it owns is closed on shutdown, so a
    restart proof needs a fresh one). ``build_wiring`` stores it unresolved and
    ``serve`` calls it once, which is where the client is built.

    It adds nothing to the message path except a record of **which payloads the bridge
    has finished handling**. That record is the barrier every assertion in this module
    rests on: the product callback runs ``handle_event`` synchronously, so a payload
    appearing here means the bridge has done everything it was ever going to do for that
    delivery — an ignore that returned, a duplicate that was rejected, or a claim whose
    dispatch settled. It is the seam-level analogue of the Nextcloud lane's recording
    transport: observed traffic, not inspected source.
    """

    def __init__(self, emulator: PubSubEmulator) -> None:
        self._emulator = emulator
        self._lock = threading.Lock()
        self._handled: list[bytes] = []
        self._client: Any = None
        self._future: Any = None

    # -- the seam itself ----------------------------------------------------

    def __call__(self, cfg: Any) -> SubscriberSeam:
        """Build the client. ``cfg`` is deliberately unread — the emulator's address is
        the test's, and ``cfg.sa_key`` names a file that does not exist."""
        with self._lock:
            if self._client is None:
                self._client = _pubsub_subscriber(self._emulator.host)
        return self

    def subscribe(
        self, subscription: str, *, callback: Callable[[Any], None], **kwargs: Any
    ) -> Any:
        """What ``serve`` calls. Wraps ``callback`` to record, then delegates verbatim."""

        def recording(message: Any) -> None:
            try:
                callback(message)
            finally:
                # After the product callback, never before: the record means "handled",
                # and recording it up front would make it mean "delivered" and turn
                # every barrier below into a race.
                with self._lock:
                    self._handled.append(bytes(message.data))

        future = self._client.subscribe(subscription, callback=recording, **kwargs)
        with self._lock:
            self._future = future
        return future

    # -- observation --------------------------------------------------------

    def deliveries(self, payload: bytes) -> int:
        """How many times the bridge has finished handling exactly these bytes."""
        with self._lock:
            return self._handled.count(payload)

    def wait_for_deliveries(self, payload: bytes, count: int, timeout: float) -> None:
        """Block until ``payload`` has been handled ``count`` times, or fail informatively."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.deliveries(payload) >= count:
                return
            time.sleep(0.2)
        with self._lock:
            seen = [item[:120] for item in self._handled]
        raise AssertionError(
            f"the bridge did not finish handling the payload {count}x within {timeout:.0f}s "
            f"(saw {self.deliveries(payload)}).\n  payload: {payload[:200]!r}\n"
            f"  everything handled: {seen!r}"
        )

    # -- shutdown -----------------------------------------------------------

    def cancel_pull(self) -> None:
        """Cancel the streaming pull so ``serve`` observes the stop event immediately.

        Exactly what ``serve``'s own ``finally`` does; doing it from the test side is
        what collapses the production 60 s supervise wake (see note 2 in the module
        docstring). Safe before or after ``serve`` has returned, and idempotent.
        """
        with self._lock:
            future = self._future
        if future is not None:
            future.cancel()

    def close(self) -> None:
        """Tear the channel down, so no lease survives the bridge it belonged to.

        Call only AFTER the serving thread has joined: closing the channel makes an
        in-progress ``future.result()`` raise ``Cancelled``, which would surface as a
        bridge crash rather than a clean stop.
        """
        with self._lock:
            client = self._client
        if client is not None:
            client.close()


# ---------------------------------------------------------------------------
# Emulator topic/subscription, per test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emulator() -> Iterator[PubSubEmulator]:
    """The running Pub/Sub emulator, or a clean SKIP when the runtime is absent.

    Deliberately this module's own fixture rather than the one
    ``tests/e2e/fixtures/conftest.py`` registers: that name is reachable only inside the
    fixture directory, and importing it here would bind a module-level name that every
    test's own parameter then shadows. The gate is the fixture module's public one, so
    there is still exactly one definition of "is this runtime usable".

    This is the runtime/image half of the lane's gating (the module docstring lists all
    three ways it can skip; the other is the ``gchat`` extra, checked in ``pytestmark``).
    It fires only when the container runtime or the image is genuinely unavailable —
    never to paper over a broken emulator, which ``PubSubEmulator.start`` fails with its
    container logs attached.
    """
    if not runtime_available():
        pytest.skip(
            f"{RUNTIME} is unavailable (CLI missing or daemon down); "
            f"set OSPREY_E2E_RUNTIME to choose a runtime"
        )
    if not image_available():
        pytest.skip(f"Pub/Sub emulator image {IMAGE!r} is unavailable (not present, pull failed)")

    running = PubSubEmulator.start()
    try:
        yield running
    finally:
        running.stop()


@dataclass(frozen=True)
class Queue:
    """One test's own topic and subscription on the shared emulator.

    Per test so no test can observe another's traffic, and created BEFORE any bridge
    starts: a subscription retains messages published while nothing is attached, which
    is what the restart proof depends on and what makes every other test free of a
    "publish before the bridge is listening" race.
    """

    emulator: PubSubEmulator
    topic_id: str
    subscription_id: str

    @property
    def subscription(self) -> str:
        """The fully-qualified name ``GCHAT_SUBSCRIPTION`` carries."""
        return self.emulator.subscription_path(self.subscription_id)

    def publish(self, payload: bytes) -> str:
        """Publish raw bytes onto the topic; returns the emulator's message id."""
        return self.emulator.publish(self.topic_id, payload)


_QUEUE_SEQ = itertools.count(1)
"""Per-process counter making every :func:`queue` name unique.

The test's own name is not enough on its own: the emulator answers a repeated create
with ``409 ALREADY_EXISTS`` (verified against the image this lane runs), and a
``@pytest.mark.flaky`` rerun re-enters this fixture under the SAME node name — so a
name-only id would turn any agentic rerun into a setup error instead of a second
attempt. The counter is monotonic within the process that owns the emulator, which is
the only scope that can collide."""


@pytest.fixture
def queue(request: pytest.FixtureRequest, emulator: PubSubEmulator) -> Queue:
    """A topic + pull subscription named after the requesting test, unique per attempt."""
    ident = re.sub(r"[^A-Za-z0-9]+", "-", request.node.name).strip("-").lower()
    # The counter is appended AFTER truncating, so a long test name can never cost the
    # uniqueness the rerun path depends on.
    topic_id = f"{f'gchat-{ident}'[:180]}-{next(_QUEUE_SEQ)}"
    subscription_id = f"{topic_id}-sub"
    emulator.create_topic(topic_id)
    emulator.create_subscription(subscription_id, topic_id)
    return Queue(emulator=emulator, topic_id=topic_id, subscription_id=subscription_id)


# ---------------------------------------------------------------------------
# The hermetic Google-side fakes (no runtime, no credentials — never skip)
# ---------------------------------------------------------------------------


@pytest.fixture
def chat() -> Iterator[FakeChatServer]:
    """The room the bridge posts into. Function-scoped: one test's posts are another's noise."""
    with FakeChatServer() as server:
        yield server


@pytest.fixture
def gcs() -> Iterator[FakeGcsServer]:
    """The bucket artifacts would be published to. Unused by the deterministic proofs'
    assertions, but injected all the same so the wiring under test is the real one."""
    with FakeGcsServer() as server:
        yield server


@pytest.fixture
def worker_http() -> Iterator[httpx.Client]:
    """The one HTTP client both publishers and the prior-artifact fetcher share.

    ``trust_env=False`` for the same reason ``CoreConfig`` defaults it off: a proxy
    inherited from a dev shell must not mount itself in front of a loopback worker.
    """
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Real dispatcher + worker as subprocesses (harness shape shared with
# tests/e2e/test_dispatch_tutorial.py)
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
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"subprocess for {url} exited early (rc={proc.returncode}).\n{_drain_output(proc)}"
            )
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310 - localhost only
            with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
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

    Two steps because the surface has two: ``init`` writes the repo's source zone
    from the preset, ``build`` renders ``build/`` from it. ``--skip-deps`` keeps it
    fast (no project venv); the worker and dispatcher run with this repo's
    interpreter, so the project venv is not needed. The provider choice never
    reaches an assertion here — see the module docstring on why these tests are
    model-independent.
    """
    base = tmp_path_factory.mktemp("gchat_bridge_build")
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

    :data:`TRIGGER_NAME` is the deterministic proofs' trigger, and is deliberately not
    one of the shipped tutorial triggers: those exist to demonstrate agent behaviour, and
    this lane must assert nothing about what a model says. It asks for a fixed word and
    no tools, so a run either completes or fails for want of a provider key — and both
    outcomes carry a ``run_id``, which is the only thing these tests read.

    :func:`_artifact_trigger` APPENDS the agentic lane's second trigger rather than
    changing this one, so the deterministic lane keeps its tool-free, keyless definition.
    See the handoff comment at the foot of this file.
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
                        "A Google Chat end-to-end test fired this event. Reply with the "
                        "single word ACKNOWLEDGED and nothing else. Do not use any tools."
                    ),
                    "allowed_tools": [],
                },
            },
            _artifact_trigger(dst.parent),
        ],
    }
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def dispatch_stack(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """A real worker + dispatcher pair as subprocesses on free ports.

    Module-scoped: the bridge is what these tests exercise, and the pair carries no state
    any assertion reads (every observation is made through the bridge's own dedup store),
    so one pair serves the whole module. The agentic lane should reuse it too.
    """
    worker_port = _free_port()
    dispatcher_port = _free_port()
    triggers_path = tmp_path_factory.mktemp("gchat_bridge_triggers") / "triggers.yml"
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

    Function-scoped so one test's stores are not another's noise — and the restart proof
    relies on the SAME directory surviving two bridge lifetimes within one test, which is
    exactly what a per-test path gives it.
    """
    state = tmp_path / "bridge-state"
    state.mkdir()
    return state


def _set_bridge_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subscription: str,
    state_dir: Path,
    dispatch: Mapping[str, Any],
) -> None:
    """Put a complete bridge environment in place, exactly as compose would.

    Set through ``monkeypatch`` and read back by
    :func:`~osprey.bridges.google_chat.__main__.config_from_env`, so the test boots the
    bridge the way the container does — including ``require_boot``'s startup validation —
    rather than hand-building a config object that could satisfy the types and skip the
    checks.

    ``GCHAT_SA_KEY`` names a file that does not exist, on purpose: all four Google seams
    are injected, so nothing may ever read it, and a regression that started reading it
    would fail loudly here instead of quietly reaching for ambient credentials. The two
    store paths MUST be overridden — they default to ``/data/*.json``, which exists only
    inside the bridge container. ``APP_VERSION_DISPLAY`` is pinned empty so the ack is
    exactly ``ACK_TEXT`` whatever the ambient environment carries.
    """
    env = {
        "GCHAT_SA_KEY": str(state_dir / "no-such-service-account.json"),
        "GCHAT_SUBSCRIPTION": subscription,
        "GCHAT_APP_ID": APP_ID,
        "GCS_BUCKET": GCS_BUCKET,
        "GCS_PROJECT": GCS_PROJECT,
        "APP_VERSION_DISPLAY": "",
        "DISPATCH_TRIGGER": TRIGGER_NAME,
        "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
        "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
        # Present-but-EMPTY counts as missing for these two, deliberately: an unset bare
        # ${VAR} renders as "" under compose and no code default would ever replace it.
        "DISPATCHER_URL": dispatch["dispatcher_url"],
        "WORKER_URL": dispatch["worker_url"],
        "DEDUP_PATH": str(state_dir / "dedup.json"),
        "HISTORY_PATH": str(state_dir / "history.json"),
        # poll_budget must stay >= worker_timeout or CoreConfig refuses to build.
        "POLL_BUDGET": str(WORKER_RUN_CAP_SEC),
        "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
        "POLL_INTERVAL": "1",
        "DRAIN_INTERVAL": "5",
        "BRIDGE_TRUST_ENV": "0",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@dataclass
class RunningBridge:
    """A bridge running in a daemon thread, plus whatever killed it."""

    wiring: Wiring
    thread: threading.Thread
    state_dir: Path
    subscriber: SubscriberSeam
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
    emulator: PubSubEmulator,
    *,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
) -> Iterator[RunningBridge]:
    """Boot the bridge in-process through ``build_wiring``/``run``, and stop it on the way out.

    In-process rather than ``python -m osprey.bridges.google_chat`` on purpose: it gives
    the test the ``Wiring`` (hence the stop event and the four seams) with no signal
    handling in the way. ``run`` re-validates the config, so this cannot boot a
    half-configured bridge that the container path would have refused.
    """
    cfg = config_from_env()
    seam = SubscriberSeam(emulator)
    wiring = build_wiring(
        cfg,
        subscriber_factory=seam,
        chat_service=chat.service(),
        gcs_client=gcs.client(),
        worker_http=worker_http,
    )
    failure: list[BaseException] = []

    def _serve() -> None:
        try:
            run(wiring)
        except BaseException as exc:  # noqa: BLE001 - re-raised from the test thread
            failure.append(exc)

    thread = threading.Thread(target=_serve, name="gchat-bridge-e2e", daemon=True)
    thread.start()
    bridge = RunningBridge(
        wiring=wiring, thread=thread, state_dir=state_dir, subscriber=seam, failure=failure
    )
    completed = False
    try:
        yield bridge
        completed = True
    finally:
        # One set() ends ingestion and the drain; cancelling the pull is what makes
        # ``serve`` notice it now rather than at its next 60 s wake.
        wiring.stop.set()
        seam.cancel_pull()
        thread.join(BRIDGE_JOIN_TIMEOUT_SEC)
        # Only after the serving thread is gone: closing the channel while ``serve`` is
        # still blocked on the future turns a clean stop into a crash.
        seam.close()
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
# Barriers
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


def _wait_for_dispatch(bridge: RunningBridge, message_name: str) -> dict[str, Any]:
    """Wait until ``message_name`` is claimed AND carries a ``run_id``; return the entry.

    ``run_id`` is persisted the moment the dispatcher handshake yields one, before the
    agent does any work, so this is the model-independent observation of "a dispatch was
    fired for this message". Nothing here reads the run's outcome.
    """

    def probe() -> dict[str, Any] | None:
        entry = bridge.dedup.get(message_name)
        if isinstance(entry, dict) and entry.get("run_id"):
            return entry
        return None

    return _await(
        bridge, f"dedup entry {message_name} to carry a run_id", probe, RUN_ID_TIMEOUT_SEC
    )


# ---------------------------------------------------------------------------
# The four deterministic proofs
# ---------------------------------------------------------------------------


def test_direct_message_without_a_mention_dispatches(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """In a direct message, a message with no @mention is a question and is dispatched.

    The space TYPE is the whole difference: the identical text is ignored in a room (the
    proof below) and dispatched here, and the type is read off the event's Space resource
    rather than guessed. The claim's persisted shape is asserted too, because every later
    step — the drain, a post-restart reconcile — works off that entry alone, long after
    the wire event is gone.
    """
    space = "spaces/AAAAdm"
    thread = f"{space}/threads/dm-1"
    question = "what is the beam current?"
    event = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-dm-1",
        question=question,
        thread=thread,
        mention=False,
    )
    message_name = event["message"]["name"]

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as bridge:
        queue.publish(_payload(event))
        entry = _wait_for_dispatch(bridge, message_name)

        assert entry[GC_IS_DM] is True
        assert entry[GC_SPACE] == space
        assert entry[GC_THREAD] == thread
        assert entry[GC_MESSAGE_NAME] == message_name
        assert entry[GC_ATTACHMENTS] == []
        assert entry["text"] == question
        assert entry["sender_id"] == HUMAN_SENDER
        # A DM is one continuous conversation, so it is keyed by the SPACE: keying it by
        # thread would never link two consecutive turns, since each DM message
        # technically starts its own.
        assert entry["history_key"] == space

        # The ack is the user-visible half, and it is a product constant rather than
        # anything a model chose — so it belongs in the deterministic tier.
        posted = chat.wait_for_posted(1)
        assert posted[0].text == ACK_TEXT
        assert posted[0].space == space
        assert posted[0].params["messageReplyOption"] == REPLY_OPTION
        # The thread the bridge ASKED for, read off the request body rather than off the
        # message the fake stored. The two differ on purpose: this thread was never
        # created in the fake, and under the reply option above an unknown thread name is
        # answered by minting a new one — which is exactly what a real deployment does
        # when an agent turn outlives the thread it was asked in. So the stored thread is
        # the fake's, while the requested one is the adapter's.
        assert posted[0].body["thread"]["name"] == thread


def test_room_message_without_a_mention_is_ignored(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """Outside a direct message, a message that does not @mention the app is ignored.

    The negative half is the point — no claim, no dispatch, no post — but on its own it
    would also pass against a bridge that was simply dead, so the same room then gets a
    message that DOES mention the app and that one must dispatch. The pair is what makes
    this a proof that the mention filter discriminates rather than a proof that nothing
    works.
    """
    space = "spaces/AAAAroom"
    thread = f"{space}/threads/room-1"
    plain = _chat_event(
        space=space,
        space_type=ROOM_SPACE_TYPE,
        message_id="msg-room-plain",
        question="no mention here, just chatter between humans",
        thread=thread,
        mention=False,
    )
    question = "what is the beam current?"
    mentioned = _chat_event(
        space=space,
        space_type=ROOM_SPACE_TYPE,
        message_id="msg-room-mentioned",
        question=question,
        thread=thread,
        mention=True,
    )

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as bridge:
        plain_payload = _payload(plain)
        queue.publish(plain_payload)
        # The barrier the negative assertions need: handle_event has RETURNED for this
        # payload, so anything the bridge was going to do, it has already done.
        bridge.subscriber.wait_for_deliveries(plain_payload, 1, IGNORED_TIMEOUT_SEC)

        assert plain["message"]["name"] not in bridge.dedup, (
            "an unmentioned room message was claimed for dispatch: "
            f"{bridge.dedup.get(plain['message']['name'])!r}"
        )
        assert chat.posted == [], (
            f"the bridge posted into a room it was not addressed in: {chat.posted_text!r}"
        )

        # Positive control, same room, same bridge: a mention IS dispatched.
        queue.publish(_payload(mentioned))
        entry = _wait_for_dispatch(bridge, mentioned["message"]["name"])

        assert entry[GC_IS_DM] is False
        assert entry[GC_SPACE] == space
        # A room conversation is scoped to its thread, unlike a DM.
        assert entry["history_key"] == thread
        # The @mention is addressing, not content: it must not reach the agent.
        assert entry["text"] == question
        assert MENTION_TOKEN not in entry["text"]


def test_byte_identical_republish_is_a_duplicate(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """A redelivery of the exact same bytes is claimed once and dispatched once.

    Pub/Sub is at-least-once, so this is the failure mode the dedup claim exists for, and
    the republish here is byte-identical rather than merely similar — the same payload
    object is published twice, so the subscriber cannot tell the two apart by anything
    except the claim.

    The first delivery is waited out to COMPLETION (the callback polls the run to
    terminal before it returns), which is what makes the "nothing further happened"
    assertions about a settled entry rather than a racing one.
    """
    space = "spaces/AAAAdup"
    thread = f"{space}/threads/dup-1"
    event = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-dup-1",
        question="how many bunches are stored?",
        thread=thread,
        mention=False,
    )
    message_name = event["message"]["name"]
    payload = _payload(event)

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as bridge:
        first_message_id = queue.publish(payload)
        first = _wait_for_dispatch(bridge, message_name)
        bridge.subscriber.wait_for_deliveries(payload, 1, HANDLED_TIMEOUT_SEC)
        settled_posts = len(chat.posted)
        assert settled_posts >= 1, "the first delivery posted nothing at all; nothing is settled"

        # Byte-identical republish — the same bytes, a second Pub/Sub message.
        second_message_id = queue.publish(payload)
        # The queue really did carry two messages: without this, a broker that silently
        # collapsed the republish would satisfy every assertion below, and the test would
        # be proving the emulator's behaviour rather than the bridge's dedup claim.
        assert first_message_id != second_message_id, (
            "the emulator returned one message id for both publishes, so there was no "
            f"redelivery to de-duplicate: {first_message_id!r}"
        )
        bridge.subscriber.wait_for_deliveries(payload, 2, HANDLED_TIMEOUT_SEC)

        assert [key for key in bridge.dedup if key == message_name] == [message_name]
        second = bridge.dedup[message_name]
        assert second["run_id"] == first["run_id"], (
            f"the redelivery started a second run: {first['run_id']!r} -> {second['run_id']!r}"
        )
        assert len(chat.posted) == settled_posts, (
            "the redelivery posted into the space again: "
            f"{[m.text for m in chat.posted[settled_posts:]]!r}"
        )


def test_message_published_while_stopped_is_dispatched_after_restart(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """A message published while the bridge is DOWN is handled once it re-attaches.

    Two bridge lifetimes over one state directory and one subscription. The first exists
    to establish that the subscription is genuinely this bridge's — without it, "the
    second bridge dispatched something" would not distinguish a restart from a first-ever
    start. The message is then published with nothing attached, which is exactly the
    outage shape: Pub/Sub retains it, and the assertion is that the restarted process
    re-attaches to the same subscription, finds it, and dispatches it against the dedup
    store the first lifetime left behind.

    The teardown between the two lifetimes is what keeps this honest: the pull is
    cancelled, the serving thread joined, the channel closed and the drain thread waited
    out, so nothing can still be leasing when the "while stopped" message is published.
    """
    space = "spaces/AAAArestart"
    thread = f"{space}/threads/restart-1"
    warmup = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-restart-warmup",
        question="are you listening?",
        thread=thread,
        mention=False,
    )
    question = "did you survive the restart?"
    later = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-restart-later",
        question=question,
        thread=thread,
        mention=False,
    )
    later_name = later["message"]["name"]

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as first:
        warmup_payload = _payload(warmup)
        queue.publish(warmup_payload)
        _wait_for_dispatch(first, warmup["message"]["name"])
        # Let the first lifetime finish with the warm-up entirely, so its shutdown is not
        # racing a dispatch that is still settling.
        first.subscriber.wait_for_deliveries(warmup_payload, 1, HANDLED_TIMEOUT_SEC)

    assert not first.thread.is_alive(), "the first bridge did not stop on its stop event"
    stopped_dedup = first.dedup

    queue.publish(_payload(later))
    assert later_name not in stopped_dedup, (
        "the message was claimed while the bridge was supposed to be stopped"
    )

    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as second:
        entry = _wait_for_dispatch(second, later_name)

    assert entry[GC_SPACE] == space
    assert entry[GC_MESSAGE_NAME] == later_name
    assert entry["text"] == question
    # The warm-up's entry is still there, so the second lifetime read the store the first
    # one wrote rather than starting from an empty one.
    assert warmup["message"]["name"] in second.dedup


# ---------------------------------------------------------------------------
# LLM-gated (agentic) tests — task 5.3 — go BELOW this line
# ---------------------------------------------------------------------------
#
# Everything above is model-independent and must stay that way: it runs with no provider
# key at all, and an assertion that reads model output does not belong in it.
#
# The agentic half reuses this module's fixtures as-is:
#
#   * ``emulator`` (module) — the running emulator. Take a fresh topic and
#     subscription from the ``queue`` fixture rather than sharing one: it is named after
#     the requesting test, so an agentic test gets its own by construction.
#   * ``chat`` / ``gcs`` (function) — the fake room and bucket. ``chat.posted`` is the
#     room's view of everything the bridge said, in order, with ``text``, ``thread`` and
#     ``cardsV2`` on each record; ``gcs.uploads`` is every artifact byte-string that was
#     published. Those two are where an artifact-delivery proof reads its evidence.
#   * ``dispatch_stack`` (module) — the real dispatcher/worker pair. Its trigger
#     (``TRIGGER_NAME``, from ``_write_triggers``) asks for a fixed word and no tools; a
#     test that needs the agent to do something real should APPEND a second trigger to
#     that document and point its own bridge at it by overriding ``DISPATCH_TRIGGER``
#     after ``_set_bridge_env`` — changing this one would cost the deterministic tier its
#     keyless property.
#   * ``bridge_state`` (function) + ``_running_bridge`` / ``_set_bridge_env`` — the boot
#     harness. ``_wait_for_dispatch`` gets you as far as "a run exists"; from there poll
#     ``chat.posted`` for the answer, and ``bridge.subscriber.wait_for_deliveries`` for
#     "the bridge has finished with this payload".
#
# Name them so they match ``-k agentic`` (the deterministic gate is run as
# ``-k "not agentic"``), and gate them on the provider key the same way the sibling
# agentic e2e modules do. ``ACK_TEXT`` is imported above and is a NOTICE, never an
# answer: a reply proof that accepts it would pass against a bridge that never got one.
#
# What the two proofs below assert is deliberately SHAPE only: that *an* answer exists,
# that it fits Chat's ceiling, that a published object really holds the bytes the card
# links to. Not one of them reads a word the model chose. The fixed strings they compare
# against are the adapter's OWN constants (:data:`_NOTICE_TEXTS`), imported from the
# product rather than re-spelled, and are used only to tell an answer apart from a
# notice. Both are ``flaky`` on ``AssertionError`` alone, which is narrower than it
# sounds: every barrier in this module reports a timeout AS an ``AssertionError``, so a
# slow provider, a slow emulator and a genuinely absent answer are all retried — which is
# the point, since slow-provider nondeterminism is exactly what the marker absorbs. What
# ``only_rerun`` excludes is the non-assertion crash: an ``ImportError``, a transport
# error, a bug in the harness itself, none of which a second attempt would fix.

ARTIFACT_TRIGGER_NAME = "google-chat-e2e-artifact"
"""Name of the agentic lane's own trigger — the one that makes a run produce an image.

Separate from :data:`TRIGGER_NAME` on purpose: that trigger asks for a fixed word and no
tools so the deterministic proofs need no provider key at all, and giving it a tool would
cost them that property. A test that wants this one points its bridge at it by overriding
``DISPATCH_TRIGGER`` after :func:`_set_bridge_env`."""

ARTIFACT_SOURCE_FILENAME = "osprey-e2e-gchat-artifact-source.png"
"""Basename of the PNG :func:`_artifact_trigger` writes for the agent to register.

The agent is asked to register a file that already exists rather than to *draw*
something: what this lane proves is the bridge's delivery and re-injection path, and a
plotting step would put a matplotlib/sandbox dependency between the test and the thing it
is actually asserting."""

REFETCH_TIMEOUT_SEC = 300.0
"""Wait for the prior-artifact fetcher's worker leg to be observed on the injected client.

Measured from the moment the follow-up is published, and it covers the whole run-up to
the dispatch: the claim, the ack post, the capability probe, and the re-fetch itself. A
ceiling far outside that range on purpose — the number this lane must never encode is
"how long an agent turn takes"."""

_NOTICE_TEXTS = frozenset(
    {ACK_TEXT, EMPTY_ANSWER_TEXT, ERROR_TEXT, GIVEUP_TEXT, QUEUED_TEXT, SUPERSEDED_TEXT}
)
"""The adapter's fixed space-facing notices, none of which is ever an answer.

``ACK_TEXT`` is posted before the dispatch and the rest are what land when a run fails,
is parked, is superseded, or completes silently — so a test that accepted any of them as
"the bridge replied" would pass against a bridge that never got an answer at all.
``APP_VERSION_DISPLAY`` is pinned empty by :func:`_set_bridge_env`, so the ack really is
``ACK_TEXT`` verbatim rather than a version-tagged variant of it."""


# ---------------------------------------------------------------------------
# The agentic lane's own helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A minimal, valid 8-bit greyscale PNG.

    Built here rather than committed as a binary fixture or a base64 blob: the delivery
    path guards on PNG magic bytes twice over (the publisher's own image check, and the
    engine's re-injection check on the way back in), so the test needs real PNG structure,
    and generating it keeps the bytes readable to someone auditing what this feeds the
    pipeline.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # One filter byte (0 = none) per scanline, then the row's samples.
    raw = b"".join(b"\x00" + bytes([0x80] * width) for _ in range(height))
    return (
        PNG_MAGIC
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _artifact_trigger(directory: Path) -> dict[str, Any]:
    """The agentic lane's second trigger, plus the PNG its prompt names.

    Called from :func:`_write_triggers` so both triggers land in the one document the
    module-scoped dispatcher is started with. The source PNG is written next to that
    document, in the same module-scoped temp directory, so the worker subprocess (same
    user, same host) can read the path the prompt gives it.

    Args:
        directory: Directory holding the triggers document.

    Returns:
        The trigger definition to append to the document's ``triggers`` list.
    """
    source = directory / ARTIFACT_SOURCE_FILENAME
    source.write_bytes(_png_bytes())
    return {
        "name": ARTIFACT_TRIGGER_NAME,
        "source": "webhook",
        "action": {
            "prompt": (
                "A Google Chat end-to-end test needs exactly one image artifact "
                "registered. Call the tool mcp__osprey_workspace__artifact_save exactly "
                f"once, with file_path set to {source} and title set to 'Chat bridge "
                "e2e image'. Use no other tool and register no other file. Then reply "
                "with the single word SAVED and nothing else."
            ),
            "allowed_tools": ["mcp__osprey_workspace__artifact_save"],
        },
    }


@dataclass(frozen=True)
class WorkerCall:
    """One HTTP call the bridge made over the injected worker client."""

    url: str
    status: int
    head: bytes
    """First bytes of the response body — enough to tell a PNG from an error page."""


class RecordingWorkerHttp:
    """The ``worker_http`` seam, plus a record of every call that travelled over it.

    The same shape of instrument as :class:`SubscriberSeam`: it changes nothing about the
    request path and only records what went over it, so what an assertion reads is
    observed traffic rather than inspected source. It is what makes the WORKER leg of the
    prior-artifact fetcher observable — the leg that must be the one to hit, because the
    ``public_url`` the engine stamped into history addresses ``storage.googleapis.com``
    for real (the publisher builds that literal itself, see the GCS fake's module
    docstring), so a fallback to the published URL would leave the hermetic world.

    ``trust_env=False`` and the timeout match the module's plain ``worker_http`` fixture:
    this is that client with a tap on it, not a differently configured one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[WorkerCall] = []
        self.client = httpx.Client(
            timeout=30.0, trust_env=False, event_hooks={"response": [self._record]}
        )

    def _record(self, response: httpx.Response) -> None:
        """Record one response. Reads the body, which is httpx's documented hook pattern.

        A response hook runs before the body is read, so ``read()`` here is what makes the
        bytes observable at all; it is idempotent (the caller's own read is then served
        from the cached content), and nothing this client carries is streamed.
        """
        response.read()
        with self._lock:
            self._calls.append(
                WorkerCall(
                    url=str(response.request.url),
                    status=response.status_code,
                    head=response.content[:8],
                )
            )

    def calls_to(self, path: str) -> list[WorkerCall]:
        """Every recorded call whose URL ends in ``path``, in order.

        Matched on the path rather than on a whole URL the test rebuilt, so a difference
        in how httpx normalizes the origin cannot turn a real hit into a miss.
        """
        with self._lock:
            return [call for call in self._calls if call.url.endswith(path)]

    def urls(self) -> list[str]:
        """Every recorded URL, for a failure message."""
        with self._lock:
            return [call.url for call in self._calls]

    def close(self) -> None:
        self.client.close()


@pytest.fixture
def recording_worker_http() -> Iterator[RecordingWorkerHttp]:
    """The worker HTTP seam with a tap on it. Function-scoped, like ``worker_http``."""
    recorder = RecordingWorkerHttp()
    try:
        yield recorder
    finally:
        recorder.close()


def _artifact_path(run_id: str, artifact_id: str) -> str:
    """The worker's byte route for one artifact, as
    :func:`osprey.bridges.core.artifacts.fetch_artifact` addresses it.

    Only the path, so it can be matched against a recorded URL without the test having to
    agree with httpx about the origin's spelling.
    """
    return f"/dispatch/{run_id}/artifacts/{artifact_id}"


def _answers(chat: FakeChatServer) -> list[PostedMessage]:
    """Everything the bridge posted that is an ANSWER — the notices removed."""
    return [message for message in chat.posted if message.text not in _NOTICE_TEXTS]


def _history(bridge: RunningBridge) -> dict[str, Any]:
    """The persisted conversation history, or ``{}`` before it exists.

    Read off the store file for the same reason :attr:`RunningBridge.dedup` is: every
    write is a tmp-file-plus-rename, so a partial read is not observable, and reading the
    file is reading what a RESTARTED bridge would see rather than an in-memory copy.
    """
    try:
        data = json.loads((bridge.state_dir / "history.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _image_urls(message: PostedMessage) -> list[str]:
    """Every ``imageUrl`` in a posted message's ``cardsV2``, in order.

    Walks the card structure defensively rather than indexing into it: a shape change in
    the product should fail an assertion about what is missing, not raise a ``KeyError``
    that says nothing about which card was malformed.
    """
    urls: list[str] = []
    for card in message.cards:
        sections = (card.get("card") or {}).get("sections") or []
        for section in sections:
            for widget in section.get("widgets") or []:
                url = (widget.get("image") or {}).get("imageUrl")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def _output_artifacts(turn: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The OUTPUT artifact descriptors recorded on one history turn."""
    return [
        dict(desc)
        for desc in turn.get("artifacts") or []
        if isinstance(desc, Mapping) and desc.get("origin") == "output"
    ]


# ---------------------------------------------------------------------------
# The two agentic proofs
# ---------------------------------------------------------------------------


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_answer_reaches_the_space(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    worker_http: httpx.Client,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """A question asked in a DM comes back as a real answer the space can see.

    The deterministic lane stops at "a run exists"; this is the other half of the same
    round trip — what the agent said actually reaches Chat. The assertions are shape only:
    the ack lands first, at least one later message is neither the ack nor a failure
    notice, and every message the bridge posted fits :data:`MAX_CHARS`, which is the
    ceiling Chat REJECTS a body over. Where the split points fall for a long answer is
    ``chunk_text``'s business and is pinned by the unit suite; what this adds is that
    whatever the model produced went out within that ceiling over a real Chat REST
    surface.

    This run has no tools and therefore no artifacts, which is the complement the artifact
    proof below needs: cards ride the final chunk only WHEN something was published, so a
    text-only answer must carry none at all, and nothing may reach the bucket.
    """
    space = "spaces/AAAAagentic"
    thread = f"{space}/threads/agentic-1"
    event = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-agentic-1",
        question="reply so the test can see that you answered",
        thread=thread,
        mention=False,
    )
    payload = _payload(event)

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=worker_http
    ) as bridge:
        queue.publish(payload)
        _wait_for_dispatch(bridge, event["message"]["name"])
        # The whole dispatch has settled by the time this returns — the callback polls the
        # run to terminal and posts before it hands back — so ``chat.posted`` below is the
        # complete record for this message rather than a snapshot mid-flight.
        bridge.subscriber.wait_for_deliveries(payload, 1, HANDLED_TIMEOUT_SEC)

    posted = chat.posted
    assert posted, "the bridge posted nothing at all into the space"
    assert posted[0].text == ACK_TEXT, f"the first post was not the ack: {posted[0].text[:200]!r}"

    answers = _answers(chat)
    assert answers, (
        "the bridge posted no answer — only notices: "
        f"{[message.text[:120] for message in posted]!r}"
    )
    for answer in answers:
        assert answer.space == space
        assert answer.body["thread"]["name"] == thread
        assert answer.params["messageReplyOption"] == REPLY_OPTION

    for message in posted:
        assert len(message.text) <= MAX_CHARS, (
            f"a posted message is over Chat's {MAX_CHARS}-character ceiling "
            f"({len(message.text)} chars); Chat rejects such a body outright"
        )
        assert not message.has_cards, f"a text-only answer carried cards: {message.cards!r}"

    assert gcs.uploads == [], (
        f"a text-only answer published objects to the bucket: {gcs.object_names!r}"
    )


@pytest.mark.requires_als_apg
# One rerun, not two: this test's barriers sum to a ~1050 s ceiling, and a third attempt
# could outlast the CI job's own cap and kill the run before any report is written.
@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_agentic_prior_artifact_is_refetched_for_the_next_turn(
    queue: Queue,
    chat: FakeChatServer,
    gcs: FakeGcsServer,
    recording_worker_http: RecordingWorkerHttp,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulator: PubSubEmulator,
) -> None:
    """Two turns in one DM: the first produces an artifact, the second gets it back.

    Everything the channel's artifact contract is made of, in one round trip through the
    real stack:

    * **turn 1** drives :data:`ARTIFACT_TRIGGER_NAME`, so the run really registers one
      image. The bridge publishes it to the (fake) bucket, rides a card on the FINAL chunk
      of the answer and no other, and the engine stamps the object's public URL onto that
      turn's history descriptor. The card's URL, the stamp, and the object holding the
      bytes are asserted to be the same object — a card linking somewhere the bucket knows
      nothing about is exactly the failure a mocked publisher cannot catch;
    * **turn 2** is a follow-up in the same DM, and it re-fires the SAME trigger —
      ``DISPATCH_TRIGGER`` belongs to the bridge, not to the message — so it registers and
      publishes an artifact of its OWN, under its own run id. That second object is not
      noise to be tolerated; it is what makes the proof unambiguous. The route this test
      watches carries TURN 1's run id, which turn 2's own publish can never read, so an
      increment on it has exactly one possible author: the prior-artifact fetcher. The
      re-injection itself owes nothing to the model — it is the engine's, driven by the
      prior turn's descriptor — and what this asserts is that it went through the bridge's
      own two-route fetcher and came back with the bytes.

    Why the WORKER leg is the one that must hit: the stamped ``public_url`` addresses
    ``storage.googleapis.com`` for real — the publisher builds that literal rather than
    asking the storage client — so the fallback leg would leave the hermetic world
    entirely. The worker still holds the artifact, so the first leg answers and the second
    is never reached. The fetch is observed on the injected client
    (:class:`RecordingWorkerHttp`), which is also what proves the seam is honoured: an
    engine default that built its own client would leave no trace there.
    """
    space = "spaces/AAAAagenticart"
    thread = f"{space}/threads/agentic-artifact-1"
    first = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-agentic-artifact-1",
        question="register the test image please",
        thread=thread,
        mention=False,
    )
    follow_up = _chat_event(
        space=space,
        space_type=DM_SPACE_TYPE,
        message_id="msg-agentic-artifact-2",
        question="and what about that image again?",
        thread=thread,
        mention=False,
    )
    first_payload = _payload(first)
    follow_up_payload = _payload(follow_up)

    _set_bridge_env(
        monkeypatch,
        subscription=queue.subscription,
        state_dir=bridge_state,
        dispatch=dispatch_stack,
    )
    # The one env var this test departs from the shared harness on: its first run has to
    # use a tool, and the deterministic trigger deliberately has none.
    monkeypatch.setenv("DISPATCH_TRIGGER", ARTIFACT_TRIGGER_NAME)

    with _running_bridge(
        bridge_state, emulator, chat=chat, gcs=gcs, worker_http=recording_worker_http.client
    ) as bridge:
        # --- turn 1: produce, publish, stamp -------------------------------
        queue.publish(first_payload)
        entry = _wait_for_dispatch(bridge, first["message"]["name"])
        run_id = str(entry["run_id"])
        bridge.subscriber.wait_for_deliveries(first_payload, 1, HANDLED_TIMEOUT_SEC)

        uploads = gcs.uploads
        assert len(uploads) == 1, (
            "the first turn did not publish exactly one artifact to the bucket "
            f"(published {len(uploads)}): the run either registered nothing or the "
            f"publish failed. Posted: {[m.text[:120] for m in chat.posted]!r}"
        )
        published = uploads[0]
        assert published.bucket == GCS_BUCKET
        assert published.data.startswith(PNG_MAGIC), (
            f"the published object is not a PNG (first bytes: {published.data[:8]!r})"
        )

        posted = chat.posted
        carded = [message for message in posted if message.has_cards]
        assert len(carded) == 1 and carded[0] is posted[-1], (
            "the artifact cards did not ride the FINAL posted chunk alone: "
            f"{[(m.text[:60], m.has_cards) for m in posted]!r}"
        )
        card_urls = _image_urls(posted[-1])
        assert len(card_urls) == 1, (
            f"expected exactly one image widget on the answer, got {card_urls!r}"
        )

        # The history stamp: what a LATER turn reads, and the only reason deliver_files
        # returns anything at all.
        turns = _history(bridge).get(space) or []
        assert len(turns) == 1, (
            f"the first turn was not recorded under the DM's history key {space!r}: "
            f"{_history(bridge)!r}"
        )
        descriptors = _output_artifacts(turns[0])
        assert len(descriptors) == 1, (
            f"the recorded turn carries no single output descriptor: {turns[0]!r}"
        )
        descriptor = descriptors[0]
        artifact_id = str(descriptor["entry_id"])
        public_url = descriptor.get("public_url")

        # One object, named the same way by all three: the card Chat would fetch, the
        # stamp a follow-up would fall back to, and the bytes the bucket really holds.
        assert public_url == f"{PUBLIC_HOST}/{GCS_BUCKET}/{published.name}", (
            f"the stamped public_url does not address the published object: "
            f"{public_url!r} vs {published.bucket}/{published.name}"
        )
        assert card_urls[0] == public_url, (
            f"the card links somewhere other than the stamped object: "
            f"{card_urls[0]!r} != {public_url!r}"
        )
        # Fetched anonymously, the way Chat itself would fetch it to render the card —
        # against the fake's origin, because the stamp names Google's for real.
        fetched = httpx.get(gcs.local_url(public_url), timeout=30.0, trust_env=False)
        assert fetched.status_code == httpx.codes.OK, (
            f"the linked object is not readable: HTTP {fetched.status_code}"
        )
        assert fetched.content == published.data

        # Re-injection admits only artifacts the descriptor calls an image.
        assert descriptor.get("delivered_mime") == "image/png", (
            f"the recorded descriptor is not an image, so no follow-up would ever ask "
            f"for it back: {descriptor!r}"
        )

        # --- turn 2: the follow-up gets the bytes back ---------------------
        route = _artifact_path(run_id, artifact_id)
        before = len(recording_worker_http.calls_to(route))
        # Snapshotted only now, with turn 1 fully SETTLED: its publish already read this
        # route once, and the entry is terminal, so the one other thing that re-reads a
        # run's artifacts — the drain re-attaching to re-post an undelivered answer —
        # cannot fire for it either. Every later read of this route is therefore the
        # prior-artifact fetcher's. A zero here would mean something else entirely: that
        # the route this test watches is not the route the bridge uses, a spelling drift
        # that would otherwise surface as an unexplained timeout below.
        assert before >= 1, (
            f"turn 1 never read {route!r} over the injected client, so the route this "
            f"test watches is wrong. Seen: {recording_worker_http.urls()!r}"
        )

        queue.publish(follow_up_payload)
        _wait_for_dispatch(bridge, follow_up["message"]["name"])

        def refetched() -> list[WorkerCall] | None:
            calls = recording_worker_http.calls_to(route)
            return calls if len(calls) > before else None

        calls = _await(
            bridge,
            f"the follow-up to re-fetch the prior artifact at {route}",
            refetched,
            REFETCH_TIMEOUT_SEC,
        )
        bridge.subscriber.wait_for_deliveries(follow_up_payload, 1, HANDLED_TIMEOUT_SEC)

    # The worker leg ANSWERED — a 404 here would have fallen through to the published URL,
    # which addresses Google and would never have resolved from this test.
    assert [call.status for call in calls] == [httpx.codes.OK] * len(calls), (
        f"the worker byte route did not answer every read of {route}: "
        f"{[(call.status, call.head) for call in calls]!r}"
    )
    assert calls[-1].head.startswith(PNG_MAGIC), (
        f"the re-fetched prior artifact is not a PNG: {calls[-1].head!r}"
    )

    # Both turns were answered, and the second is recorded after the first — so the
    # follow-up really was a second exchange in the same conversation rather than a
    # redelivery of the first.
    assert len(_answers(chat)) >= 2, (
        f"the follow-up produced no answer of its own: "
        f"{[message.text[:120] for message in chat.posted]!r}"
    )
    assert len(_history(bridge).get(space) or []) == 2, (
        f"the DM's history does not hold both turns: {_history(bridge)!r}"
    )
    # Two objects in the bucket, not one: turn 2 ran the same trigger and published its
    # own. Pinned because it is what keeps the route above unambiguous — a run that
    # somehow republished the PRIOR artifact instead would inflate that count for a reason
    # that has nothing to do with re-injection.
    assert len(gcs.uploads) == 2, (
        "expected one published object per turn: "
        f"{[(upload.bucket, upload.name) for upload in gcs.uploads]!r}"
    )
