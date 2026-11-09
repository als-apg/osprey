"""Real-stack e2e for the Slack bridge: loopback Slack fakes and a real dispatcher.

Every unit test of ``osprey.bridges.slack`` drives one seam at a time against a stub, so
each asserts OSPREY's half of a two-party contract against OSPREY's own idea of the wire
format. This module runs the rest of the stack: a real ``osprey.dispatch`` dispatcher and
worker as subprocesses (:mod:`tests.e2e._dispatch_harness`), the bridge booted **through
its own entrypoint seams** — the same ``build_wiring``/``run`` the container's ``main``
calls — and Slack made concrete by :mod:`tests.e2e.fixtures.slack_fakes`: a loopback Web
API and file host the bridge's HTTP client is redirected onto, and a loopback Socket Mode
server that ``apps.connections.open`` hands out. Envelopes are pushed down a real
WebSocket, acknowledged by the real ingest thread, spooled through the real inbox, and
answered through the real Web API client.

Every test here is **deterministic**: no assertion reads model output, so they pass with
no provider key at all. A dispatch is observed as "the dedup entry now carries a
``run_id``" — the engine persists it the moment the dispatcher handshake yields one. Where
a criterion is about what happens to an ANSWER (its placement, its chunks, its mentions, a
refusal, a queued or resumed request), the bridge runs with a :class:`RecordingDispatcher`:
the real :class:`~osprey.bridges.core.DispatchClient`, which fires the question at the real
dispatcher and waits for the real run id, and then reports a scripted terminal result
instead of polling the worker for a model's answer. Everything after that — settlement,
the retry queue, the drain, every Slack post — is the product's own code.

The Slack-side wording is compared against the package's exported constants
(:func:`~osprey.bridges.slack.ack_text`, :data:`~osprey.bridges.slack.QUEUED_TEXT`, …),
never re-spelled, so a reworded notice changes one constant in the product and nothing
here. The proxy criterion is not proven here: every Slack leg is injected, which bypasses
the proxy decision by construction; the wiring's construction tests cover it.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY
----------------------------------------------------------------------------
This module issues no container-runtime command and needs no container, image or volume.
The dispatcher and worker are subprocesses of this interpreter; the Slack fakes and the
worker-artifact stand-in are loopback servers on ephemeral ports. Nothing here prunes or
removes anything.

----------------------------------------------------------------------------
Host ports
----------------------------------------------------------------------------
None pinned. Every server binds ``127.0.0.1:0`` and reports the port it got.

----------------------------------------------------------------------------
Gating
----------------------------------------------------------------------------
No container runtime, credential or provider key, so nothing here skips. ``websockets``
is a core dependency of the bridge, so there is no extra to guard. Like its Teams sibling
the module is not ``model_free``: it runs under the e2e provider gate with
``OSPREY_E2E_PROVIDER`` set, and pins the build's endpoint to a loopback port nobody
answers.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import yaml

from osprey.bridges.core import DispatchClient, DispatchPipelineError, PipelineDeps
from osprey.bridges.slack import (
    ANSWER_CHUNK_CHARS,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    RESUMED_TEXT,
    SL_CHANNEL,
    SL_CHANNEL_TYPE,
    SL_THREAD_TS,
    SL_TS,
    SUPERSEDED_TEXT,
    ack_text,
    markdown_to_mrkdwn,
    quote_prefix,
)
from osprey.bridges.slack.__main__ import (
    INBOX_DIRNAME,
    Wiring,
    build_wiring,
    config_from_env,
    run,
)
from osprey.bridges.slack.ops import FOREIGN_HOST_REASON, NOT_A_FILE_REASON
from osprey.bridges.slack.receiver import POOL_SIZE, TICK_SEC
from tests.bridges.record_double import MemoryRecord, RecordHistory
from tests.e2e._dispatch_harness import (
    await_bridge,
    build_repo,
    drain_thread_alive,
    running_dispatch_stack,
)
from tests.e2e.fixtures.gchat_fake_http import FakeHttpService, _FakeHandler
from tests.e2e.fixtures.slack_fakes import (
    APP_TOKEN,
    BOT_ID,
    BOT_TOKEN,
    BOT_USER_ID,
    CHANNEL_ID,
    DM_CHANNEL_ID,
    MPIM_CHANNEL_ID,
    SENDER_ID,
    SENDER_NAME,
    ApiCall,
    FakeSlackWeb,
    FakeSocketServer,
    PostedMessage,
    bot_message,
    channel_mention,
    dm_message,
    envelope,
    file_share,
    mpim_message,
    next_ts,
    shared_channel_mention,
    thread_reply,
)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Identity and budgets
# ---------------------------------------------------------------------------

VERSION_TAG = "e2e-slack-tag"
"""``APP_VERSION_DISPLAY`` for every bridge this module boots. A sentinel rather than
empty: an empty value falls back to the installed distribution's version, which would
make the expected ack depend on how the checkout is installed."""

DISPATCH_TOKEN = "slack-e2e-dispatch-token"
TRIGGER_NAME = "slack-e2e"

DEAD_DISPATCHER_URL = "http://127.0.0.1:9"
"""A dispatcher nobody answers (the discard port): what "the dispatcher is down" is."""

HEALTH_TIMEOUT_SEC = 45.0
RUN_ID_TIMEOUT_SEC = 150.0
"""Wait for a claimed event's dedup entry to carry a ``run_id``."""

SETTLED_TIMEOUT_SEC = 300.0
"""Wait for a delivered event to leave the inbox. For a claimed event that means the
whole dispatch settled; comfortably past :data:`WORKER_RUN_CAP_SEC`."""

POST_TIMEOUT_SEC = 120.0
"""Wait for an expected Slack post that follows a scripted (model-free) result."""

ACK_TIMEOUT_SEC = 15.0
"""Wait for the ingest thread to acknowledge a pushed envelope."""

BRIDGE_JOIN_TIMEOUT_SEC = 120.0
WORKER_RUN_CAP_SEC = 90
"""``DISPATCH_TIMEOUT_SEC`` for the worker and ``POLL_BUDGET`` for the bridge (they must
agree: ``CoreConfig`` refuses ``POLL_BUDGET < DISPATCH_TIMEOUT_SEC``)."""

BUILD_TIMEOUT_SEC = 600
HOLD_TIMEOUT_SEC = 120.0
"""Longest a held :class:`RecordingDispatcher` waits for its release."""

FAST_DRAIN = {"DRAIN_INTERVAL": "1", "RETRY_MIN_AGE": "0.5"}
"""Drain settings for the retry-queue proofs: a pass every second, and an entry eligible
half a second after it parked, so the drain acts within the test's patience."""

CAROL_ID = "U0CAROL"
STRANGER_ID = "U0STRANGER"


# ---------------------------------------------------------------------------
# Real dispatcher + worker (module-scoped, shared with the sibling bridge lanes)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Init + build a demo repo once per module. The provider never reaches
    an assertion: the endpoint is pinned to a loopback port nobody answers."""
    return build_repo(
        tmp_path_factory.mktemp("slack_bridge_build"),
        provider="als-apg",
        model="claude-haiku-5-5",
        timeout=BUILD_TIMEOUT_SEC,
        env={"ALS_APG_BASE_URL": "http://127.0.0.1:9/v1"},
    )


def _write_triggers(dst: Path, worker_port: int) -> None:
    """A fixed-word, no-tools trigger: a run completes or fails for want of a provider,
    and both carry a ``run_id`` — the only thing the real-stack proofs read."""
    doc = {
        "dispatcher": {
            "dispatch_target": f"http://127.0.0.1:{worker_port}",
            "max_concurrent_runs": 4,
            "max_queue_depth": 100,
        },
        "triggers": [
            {
                "name": TRIGGER_NAME,
                "source": "webhook",
                "action": {
                    "prompt": (
                        "A Slack end-to-end test fired this event. Reply with the single "
                        "word ACKNOWLEDGED and nothing else. Do not use any tools."
                    ),
                    "allowed_tools": [],
                },
            }
        ],
    }
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def dispatch_stack(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    with running_dispatch_stack(
        built_repo,
        tmp_path_factory.mktemp("slack_bridge_triggers") / "triggers.yml",
        _write_triggers,
        token=DISPATCH_TOKEN,
        health_timeout=HEALTH_TIMEOUT_SEC,
        run_cap_sec=WORKER_RUN_CAP_SEC,
    ) as stack:
        yield stack


# ---------------------------------------------------------------------------
# The Slack fakes (function-scoped: one test's traffic is another's noise)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedCall:
    """One Web API call and the monotonic instant the fake answered it."""

    call: ApiCall
    at: float


class TimedSlackWeb(FakeSlackWeb):
    """:class:`FakeSlackWeb` that also stamps every answered call with a monotonic time.

    Adds nothing to the wire; it is what turns "the post was retried after a 429" into
    "the post was retried no sooner than Slack asked".
    """

    def __init__(self, **kwargs: Any) -> None:
        self._timed: list[TimedCall] = []
        self._timed_lock = threading.Lock()
        super().__init__(**kwargs)

    def _api(self, handler: Any, method: str, args: dict[str, Any]) -> None:
        super()._api(handler, method, args)
        last = self.calls[-1]
        with self._timed_lock:
            self._timed.append(TimedCall(last, time.monotonic()))

    def timed(self, method: str) -> list[TimedCall]:
        with self._timed_lock:
            return [t for t in self._timed if t.call.method == method]


@pytest.fixture
def socket_server() -> Iterator[FakeSocketServer]:
    with FakeSocketServer() as server:
        yield server


@pytest.fixture
def web(socket_server: FakeSocketServer) -> Iterator[TimedSlackWeb]:
    with TimedSlackWeb(socket=socket_server) as server:
        yield server


@pytest.fixture
def bridge_state(tmp_path: Path) -> Path:
    state = tmp_path / "bridge-state"
    state.mkdir()
    return state


# ---------------------------------------------------------------------------
# The dispatcher seam: the real client, recording, optionally scripted
# ---------------------------------------------------------------------------

_TERMINAL_DEFAULTS: dict[str, Any] = {
    "status": "error",
    "text_output": "",
    "error": None,
    "artifacts": [],
    "input_artifacts": [],
    "failure_class": None,
    "num_tool_calls": None,
    "error_code": None,
}


def completed(text: str, **extra: Any) -> dict[str, Any]:
    """A completed run's terminal result, as the worker reports one."""
    return {"status": "completed", "text_output": text, "num_tool_calls": 0, **extra}


def provider_failure() -> dict[str, Any]:
    """A run the provider failed before it took any action — retryable and provably
    side-effect free (a KNOWN tool-call count of zero), so the drain may re-run it."""
    return {
        "status": "error",
        "error": "provider unavailable",
        "failure_class": "provider",
        "num_tool_calls": 0,
    }


@dataclass(frozen=True)
class Fired:
    """One webhook POST the bridge made: the question and the payload around it."""

    question: str
    extra: dict[str, Any]


class RecordingDispatcher(DispatchClient):
    """The real :class:`DispatchClient`, recording what it fires and what it returns.

    Unscripted, it is the production client byte for byte. With a ``script``, each
    :meth:`run` still fires the question at the real dispatcher and waits for the real run
    id (so ``on_run_id`` and the persisted entry are exactly the live path's), then
    returns the next scripted terminal result in place of a model's; :meth:`status`
    answers that same result for its run id, as the worker would when the drain re-attaches.
    ``hold`` makes every :meth:`run` wait for its release before firing — how a test keeps
    handler threads busy. ``record`` learns each completed answer, as the worker's run
    record would, so a history read of that turn is complete.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        script: list[dict[str, Any]] | None = None,
        record: MemoryRecord | None = None,
        hold: threading.Event | None = None,
        dispatcher_url: str | None = None,
    ) -> None:
        if dispatcher_url is not None:
            cfg = dataclasses.replace(cfg, dispatcher_url=dispatcher_url)
        super().__init__(cfg)
        self._lock = threading.Lock()
        self._script = list(script or [])
        self._record = record
        self._bodies: dict[str, dict[str, Any]] = {}
        self.hold = hold
        self.fired: list[Fired] = []
        self.results: list[dict[str, Any]] = []
        self.entered = 0

    def fire(self, question: str, extra: dict[str, Any] | None = None) -> str:
        with self._lock:
            self.fired.append(Fired(question, copy.deepcopy(dict(extra or {}))))
        return super().fire(question, extra)

    def run(
        self,
        question: str,
        extra: dict[str, Any] | None = None,
        *,
        on_run_id: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.entered += 1
            scripted = self._script.pop(0) if self._script else None
        if self.hold is not None and not self.hold.wait(HOLD_TIMEOUT_SEC):
            raise AssertionError("a held dispatch was never released")
        if scripted is None:
            result = super().run(question, extra, on_run_id=on_run_id)
        else:
            result = self._scripted(question, extra, scripted, on_run_id)
        with self._lock:
            self.results.append(result)
        return result

    def _scripted(
        self,
        question: str,
        extra: dict[str, Any] | None,
        scripted: Mapping[str, Any],
        on_run_id: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        try:
            dispatch_id = self.fire(question, extra)
            run_id = self.wait_for_run_id(dispatch_id, self._monotonic() + self.cfg.poll_budget)
        except (DispatchPipelineError, httpx.HTTPError) as exc:
            raise AssertionError(f"the real dispatcher refused a scripted run: {exc}") from exc
        if on_run_id is not None:
            on_run_id(run_id)
        self._cancel(run_id)
        body = {**_TERMINAL_DEFAULTS, **scripted}
        with self._lock:
            self._bodies[run_id] = body
        if self._record is not None and body["status"] == "completed" and body["text_output"]:
            self._record.answer(run_id, body["text_output"])
        return {**body, "run_id": run_id}

    def _cancel(self, run_id: str) -> None:
        """Cancel the real run behind a scripted result, best-effort.

        Its outcome is never read — the script stands in for it — and without a provider
        it would otherwise hold a worker slot until the run cap.
        """
        with contextlib.suppress(httpx.HTTPError):
            self._client.delete(
                f"{self.cfg.worker_url}/dispatch/{run_id}",
                headers={"Authorization": f"Bearer {self.cfg.dispatch_worker_token}"},
            )

    def status(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            body = self._bodies.get(run_id)
        if body is not None:
            return dict(body)
        return super().status(run_id)

    def fired_for(self, question: str) -> list[Fired]:
        with self._lock:
            return [f for f in self.fired if f.question == question]


# ---------------------------------------------------------------------------
# The bridge under test, booted IN-PROCESS through build_wiring/run
# ---------------------------------------------------------------------------


def _bridge_env(
    *, state_dir: Path, dispatcher_url: str, worker_url: str, **overrides: str
) -> dict[str, str]:
    """A complete bridge environment, exactly as compose would render one."""
    env = {
        "SLACK_BOT_TOKEN": BOT_TOKEN,
        "SLACK_APP_TOKEN": APP_TOKEN,
        "APP_VERSION_DISPLAY": VERSION_TAG,
        "SLACK_MENTIONS": "true",
        "DISPATCH_TRIGGER": TRIGGER_NAME,
        "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
        "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
        "DISPATCHER_URL": dispatcher_url,
        "WORKER_URL": worker_url,
        "DEDUP_PATH": str(state_dir / "dedup.json"),
        "POLL_BUDGET": str(WORKER_RUN_CAP_SEC),
        "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
        "POLL_INTERVAL": "1",
        "DRAIN_INTERVAL": "5",
        "BRIDGE_TRUST_ENV": "0",
    }
    env.update(overrides)
    return env


def _set_bridge_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dir: Path,
    dispatch: Mapping[str, Any],
    worker_url: str | None = None,
    **overrides: str,
) -> None:
    """Put :func:`_bridge_env` in place for ``config_from_env`` — the container's own
    startup validation included."""
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in _bridge_env(
        state_dir=state_dir,
        dispatcher_url=dispatch["dispatcher_url"],
        worker_url=worker_url or dispatch["worker_url"],
        **overrides,
    ).items():
        monkeypatch.setenv(name, value)


@dataclass
class RunningBridge:
    """A bridge running in a daemon thread, plus whatever killed it."""

    wiring: Wiring
    thread: threading.Thread
    state_dir: Path
    socket: FakeSocketServer
    failure: list[BaseException] = field(default_factory=list)

    @property
    def dedup(self) -> dict[str, dict[str, Any]]:
        """The persisted dedup store, or ``{}`` before it exists (every write is a
        tmp-file-plus-rename, so a partial file is never observable)."""
        try:
            data = json.loads((self.state_dir / "dedup.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def pending(self) -> list[str]:
        """Event ids still in the durable inbox."""
        return [key for key, _ in self.wiring.inbox.pending()]

    def deliver(self, payload: dict[str, Any], **envelope_kwargs: Any) -> dict[str, Any]:
        """Push ``payload`` down the socket and wait for the bridge to acknowledge it —
        which it does only once the event is durable in the inbox."""
        env = envelope(payload, **envelope_kwargs)
        self.socket.push(env)
        await_bridge(
            self,
            f"the ack of envelope {env['envelope_id']}",
            lambda: True if env["envelope_id"] in self.socket.acks else None,
            ACK_TIMEOUT_SEC,
        )
        return env

    def settle(self, payload: Mapping[str, Any], timeout: float = SETTLED_TIMEOUT_SEC) -> None:
        """Wait until the delivered event has left the inbox: ``handle_event`` returned,
        so whatever the bridge was going to do for it, it has done."""
        event_id = payload["event_id"]
        await_bridge(
            self,
            f"event {event_id} to leave the inbox",
            lambda: True if event_id not in self.pending() else None,
            timeout,
        )


@contextlib.contextmanager
def _running_bridge(
    web: FakeSlackWeb,
    socket: FakeSocketServer,
    state_dir: Path,
    *,
    deps: Callable[[Wiring], PipelineDeps] | None = None,
    gate: Callable[[], bool] | None = None,
) -> Iterator[RunningBridge]:
    """Boot the bridge through ``build_wiring``/``run`` and stop it on the way out.

    ``deps`` builds a replacement engine bundle from the wiring (``run(deps=...)``, the
    entrypoint's own seam); the Slack seams are always the fakes': the Web API client is
    redirected onto ``web``, and the WebSocket ``connect`` is the production default,
    dialling whatever loopback URL ``apps.connections.open`` minted. The inbox is the
    deployment default — ``inbox/`` beside ``DEDUP_PATH`` — so a restart on the same
    state directory finds what the previous process spooled.
    """
    cfg = config_from_env()
    slack_http = web.http_client()
    wiring = build_wiring(cfg, slack_http=slack_http)
    bundle = deps(wiring) if deps is not None else None
    failure: list[BaseException] = []
    connections = socket.connection_count

    def _serve() -> None:
        try:
            run(wiring, deps=bundle, gate=gate)
        except BaseException as exc:  # re-raised from the test thread
            failure.append(exc)

    thread = threading.Thread(target=_serve, name="slack-bridge-e2e", daemon=True)
    thread.start()
    bridge = RunningBridge(
        wiring=wiring, thread=thread, state_dir=state_dir, socket=socket, failure=failure
    )
    completed_body = False
    try:
        socket.wait_for_connections(connections + 1, timeout=ACK_TIMEOUT_SEC)
        yield bridge
        completed_body = True
    finally:
        wiring.stop.set()
        thread.join(BRIDGE_JOIN_TIMEOUT_SEC)
        deadline = time.monotonic() + BRIDGE_JOIN_TIMEOUT_SEC
        while drain_thread_alive() and time.monotonic() < deadline:
            time.sleep(0.2)
        slack_http.close()
        if completed_body and failure:
            raise AssertionError(f"bridge thread crashed: {failure[0]!r}") from failure[0]
        if completed_body and thread.is_alive():
            raise AssertionError("the bridge thread did not stop on its stop event")
        if completed_body and drain_thread_alive():
            raise AssertionError("the drain thread outlived its stop event")


def _recording(
    holder: list[RecordingDispatcher], **kwargs: Any
) -> Callable[[Wiring], PipelineDeps]:
    """A ``deps`` factory: the wiring's own bundle with a :class:`RecordingDispatcher`
    (built over the wiring's proxy-free core config) in place of its dispatcher. The
    dispatcher is appended to ``holder`` so the test can read it back. ``history``
    replaces the bundle's conversation history when given."""
    history = kwargs.pop("history", None)

    def build(wiring: Wiring) -> PipelineDeps:
        dispatcher = RecordingDispatcher(wiring.deps.cfg, **kwargs)
        holder.append(dispatcher)
        bundle = dataclasses.replace(wiring.deps, dispatcher=dispatcher)
        if history is not None:
            bundle = dataclasses.replace(bundle, history=history)
        return bundle

    return build


# ---------------------------------------------------------------------------
# Barriers and small readers
# ---------------------------------------------------------------------------


def _key(payload: Mapping[str, Any]) -> str:
    """The dedup key the adapter derives: ``channel:ts``. Spelled here because it is the
    test's expectation of the adapter's scoping rule; a change to it must break this."""
    event = payload["event"]
    return f"{event['channel']}:{event['ts']}"


def _ts(payload: Mapping[str, Any]) -> str:
    return str(payload["event"]["ts"])


def _wait_for_dispatch(bridge: RunningBridge, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Wait until the event is claimed AND its entry carries a ``run_id``."""
    key = _key(payload)

    def probe() -> dict[str, Any] | None:
        entry = bridge.dedup.get(key)
        if isinstance(entry, dict) and entry.get("run_id"):
            return entry
        return None

    return await_bridge(bridge, f"dedup entry {key} to carry a run_id", probe, RUN_ID_TIMEOUT_SEC)


def _wait_for_status(
    bridge: RunningBridge, payload: Mapping[str, Any], status: str, timeout: float
) -> dict[str, Any]:
    key = _key(payload)

    def probe() -> dict[str, Any] | None:
        entry = bridge.dedup.get(key)
        return entry if isinstance(entry, dict) and entry.get("status") == status else None

    return await_bridge(bridge, f"dedup entry {key} to reach {status!r}", probe, timeout)


def _wait_for_posts(
    bridge: RunningBridge, web: FakeSlackWeb, count: int, timeout: float = POST_TIMEOUT_SEC
) -> list[PostedMessage]:
    """Wait for ``count`` accepted posts, checking the bridge thread on every pass."""
    await_bridge(
        bridge,
        f"{count} accepted post(s); got {[p.text[:60] for p in web.posted]}",
        lambda: True if len(web.posted) >= count else None,
        timeout,
    )
    return web.posted


def _texts(posts: list[PostedMessage]) -> list[str]:
    return [p.text for p in posts]


# ---------------------------------------------------------------------------
# Placement and ack
# ---------------------------------------------------------------------------


def test_a_dm_question_is_acked_at_the_top_level_with_a_quote_and_dispatched(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1:1 DM is a question without a mention, answered in the DM itself.

    Runs on the wiring's production engine bundle, untouched: the real dispatch client
    polls the real worker. The ack lands at the DM's top level (no ``thread_ts``) and,
    because a top-level DM has no thread to say which message it answers, opens with the
    product's own :func:`quote_prefix` of the question.
    """
    question = "what is the beam current?"
    payload = dm_message(question)

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(web, socket_server, bridge_state) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_dispatch(bridge, payload)

        assert entry["text"] == question
        assert entry["sender_id"] == SENDER_ID
        assert entry["history_key"] == DM_CHANNEL_ID
        assert entry[SL_CHANNEL] == DM_CHANNEL_ID
        assert entry[SL_CHANNEL_TYPE] == "im"
        assert entry[SL_THREAD_TS] == ""

        ack = _wait_for_posts(bridge, web, 1)[0]
        assert ack.text == quote_prefix(entry) + ack_text(VERSION_TAG)
        assert quote_prefix(entry) == f"> {question}\n\n"
        assert (ack.channel, ack.thread_ts) == (DM_CHANNEL_ID, "")
        # The bot token, never the app token, authorised the post.
        assert {c.token for c in web.calls_to("chat.postMessage")} == {BOT_TOKEN}
        assert [c.token for c in web.calls_to("apps.connections.open")] == [APP_TOKEN]

        bridge.settle(payload)
        assert all(p.channel == DM_CHANNEL_ID and p.thread_ts == "" for p in web.posted)


def test_a_channel_mention_is_acked_in_its_own_thread_and_dispatched(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-level channel mention is answered in the thread it roots, with no quote."""
    question = "what is the lifetime?"
    payload = channel_mention(question)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(holder, script=[completed("The lifetime is 9 hours.")]),
    ) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_dispatch(bridge, payload)

        # The @mention is addressing, not content.
        assert entry["text"] == question
        assert entry[SL_CHANNEL_TYPE] == "channel"
        assert entry[SL_THREAD_TS] == entry[SL_TS] == _ts(payload)
        assert entry["history_key"] == f"{CHANNEL_ID}:{_ts(payload)}"

        ack, answer = _wait_for_posts(bridge, web, 2)[:2]
        assert ack.text == ack_text(VERSION_TAG)
        assert answer.text == markdown_to_mrkdwn("The lifetime is 9 hours.")
        for post in (ack, answer):
            assert (post.channel, post.thread_ts) == (CHANNEL_ID, _ts(payload))
        bridge.settle(payload)
        assert bridge.dedup[_key(payload)]["status"] == "completed"


def test_a_dm_thread_question_and_a_group_dm_mention_are_answered_in_their_threads(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question asked inside a DM thread is answered in that thread (no quote, since the
    thread already says what is answered); an mpim mention is answered in its thread.

    The mpim post arrives as BOTH a ``message`` and an ``app_mention`` sharing one ``ts``;
    the second must be a duplicate of the first — one run, one ack.
    """
    parent = next_ts()
    in_thread = dm_message("and the injection efficiency?", thread_ts=parent)
    as_message, as_mention = mpim_message("what is the vacuum reading?")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(holder, script=[completed("dm answer"), completed("mpim answer")]),
    ) as bridge:
        bridge.deliver(in_thread)
        dm_entry = _wait_for_dispatch(bridge, in_thread)
        bridge.settle(in_thread)

        assert dm_entry[SL_THREAD_TS] == parent
        assert quote_prefix(dm_entry) == ""
        dm_posts = [p for p in web.posted if p.channel == DM_CHANNEL_ID]
        assert _texts(dm_posts) == [ack_text(VERSION_TAG), "dm answer"]
        assert {p.thread_ts for p in dm_posts} == {parent}

        bridge.deliver(as_message)
        bridge.deliver(as_mention)
        mpim_entry = _wait_for_dispatch(bridge, as_message)
        bridge.settle(as_message)
        bridge.settle(as_mention)

        assert mpim_entry[SL_CHANNEL_TYPE] == "mpim"
        mpim_posts = [p for p in web.posted if p.channel == MPIM_CHANNEL_ID]
        assert _texts(mpim_posts) == [ack_text(VERSION_TAG), "mpim answer"]
        assert {p.thread_ts for p in mpim_posts} == {_ts(as_message)}
        assert len(holder[0].fired_for("what is the vacuum reading?")) == 1


# ---------------------------------------------------------------------------
# What starts no run
# ---------------------------------------------------------------------------


def _unmentioned_channel_message(text: str, *, channel: str, channel_type: str) -> dict:
    """A plain ``message`` in a channel or group DM that does not mention the bot."""
    payload = dm_message(text, channel=channel)
    payload["event"]["channel_type"] = channel_type
    return payload


def test_messages_not_addressed_to_the_bot_start_no_run(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un-mentioned channel and group-DM chatter, a bare mention, this bot's own posts,
    another bot's post and a shared-channel mention are all acknowledged to Slack and
    then ignored: no claim, no run, no post.

    Each is waited out of the inbox first — the barrier that makes "nothing happened"
    mean "nothing will". Then a real mention in the same bridge must dispatch, so the
    negatives are proofs that the filter discriminates, not that the bridge is dead.
    """
    ignored = [
        _unmentioned_channel_message("chatter", channel=CHANNEL_ID, channel_type="channel"),
        _unmentioned_channel_message("group chatter", channel=MPIM_CHANNEL_ID, channel_type="mpim"),
        channel_mention(""),
        dm_message("talking to myself", user=BOT_USER_ID),
        bot_message(f"<@{BOT_USER_ID}> echo", bot_id=BOT_ID, channel=CHANNEL_ID),
        bot_message(),
        shared_channel_mention(),
    ]
    question = "is the shutter open?"
    mentioned = channel_mention(question)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("open")])
    ) as bridge:
        for payload in ignored:
            bridge.deliver(payload)
        for payload in ignored:
            bridge.settle(payload)

        assert bridge.dedup == {}, f"an unaddressed message was claimed: {bridge.dedup!r}"
        assert web.posted == [], f"the bridge posted unprompted: {_texts(web.posted)!r}"
        assert holder[0].fired == []

        bridge.deliver(mentioned)
        entry = _wait_for_dispatch(bridge, mentioned)
        assert entry["text"] == question
        assert [f.question for f in holder[0].fired] == [question]


def test_a_redelivered_event_and_the_duplicate_mention_start_exactly_one_run(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack's redelivery (same event under a new envelope, ``retry_attempt`` set) and a
    channel mention that also arrives as a channel ``message`` are each ONE run and ONE
    ack: the dedup claim on ``channel:ts`` turns the second delivery into a duplicate."""
    question = "how many bunches are stored?"
    payload = channel_mention(question)
    as_message = copy.deepcopy(payload)
    as_message["event"]["type"] = "message"
    as_message["event"]["channel_type"] = "channel"
    as_message["event_id"] = "Ev-dup-" + _ts(payload)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("320")])
    ) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_dispatch(bridge, payload)
        bridge.settle(payload)
        settled_posts = len(web.posted)

        bridge.deliver(payload, retry_attempt=1, retry_reason="timeout")
        bridge.deliver(as_message)
        bridge.settle(payload)
        bridge.settle(as_message)

        assert list(bridge.dedup) == [_key(payload)]
        assert bridge.dedup[_key(payload)]["run_id"] == entry["run_id"]
        assert len(holder[0].fired_for(question)) == 1
        assert len(web.posted) == settled_posts
        assert _texts(web.posted).count(ack_text(VERSION_TAG)) == 1


# ---------------------------------------------------------------------------
# What reaches the worker: history, reply context, files, plain text
# ---------------------------------------------------------------------------


def test_a_second_question_in_a_dm_or_thread_carries_the_first_and_a_new_thread_does_not(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation history is keyed by the adapter's ``history_key``: the DM itself, or
    one channel thread. A follow-up in the same DM or thread replays the earlier turn; a
    new top-level mention in the same channel is a new thread and replays nothing."""
    record = MemoryRecord()
    holder: list[RecordingDispatcher] = []
    dm_first = dm_message("what is the beam current?")
    dm_second = dm_message("and an hour ago?")
    root = channel_mention("what is the orbit rms?")
    follow_up = thread_reply(_ts(root), "and vertically?")
    fresh_thread = channel_mention("is the RF on?")

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(
            holder,
            record=record,
            history=RecordHistory(record),
            script=[completed(f"answer {i}") for i in range(5)],
        ),
    ) as bridge:
        for payload in (dm_first, dm_second, root, follow_up, fresh_thread):
            bridge.deliver(payload)
            bridge.settle(payload)

        fired = {f.question: f.extra for f in holder[0].fired}
        replayed = {
            q: [t["question"] for t in extra.get("conversation_so_far", [])]
            for q, extra in fired.items()
        }
        assert replayed == {
            "what is the beam current?": [],
            "and an hour ago?": ["what is the beam current?"],
            "what is the orbit rms?": [],
            "and vertically?": ["what is the orbit rms?"],
            "is the RF on?": [],
        }
        assert fired["and an hour ago?"]["conversation_so_far"][0]["answer"] == "answer 0"
        assert fired["and vertically?"]["conversation_so_far"][0]["answer"] == "answer 2"


def test_slack_markup_reaches_the_worker_as_plain_text_and_markdown_renders_as_mrkdwn(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack's ``&lt;``/``&amp;`` escapes and ``<url|label>`` links arrive at the
    dispatcher as the text a person reads; an answer's Markdown bold, code and heading
    arrive in Slack as mrkdwn."""
    raw = "compare &lt;5 mA&gt; &amp; <https://example.org/log|the log>"
    payload = dm_message(raw)
    answer = "# Summary\n\nThe current is **402 mA**; see `SR:DCCT`."
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed(answer)])
    ) as bridge:
        bridge.deliver(payload)
        bridge.settle(payload)

        (fired,) = holder[0].fired
        assert fired.question == "compare <5 mA> & the log (https://example.org/log)"
        entry = bridge.dedup[_key(payload)]
        posted = _texts(web.posted)
        assert posted[-1] == quote_prefix(entry) + markdown_to_mrkdwn(answer)
        rendered = posted[-1]
        assert "*402 mA*" in rendered and "**" not in rendered
        assert "`SR:DCCT`" in rendered
        assert "# Summary" not in rendered and "*Summary*" in rendered


def test_a_thread_reply_carries_the_parents_text_and_files_as_reply_context(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent of a thread reply is looked up with ``conversations.replies``; its text
    and its attached file travel with the question — the file's bytes in ``input_files``,
    its name inside ``reply_to`` so the agent knows where it came from."""
    csv = b"time,current\n10:00,402\n"
    ref = web.add_file("current.csv", csv, mimetype="text/csv")
    parent = channel_mention("here is today's current trace", files=[ref])["event"]
    web.add_message(CHANNEL_ID, parent)
    reply = thread_reply(parent["ts"], "does it dip anywhere?")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("no")])
    ) as bridge:
        bridge.deliver(reply)
        _wait_for_dispatch(bridge, reply)
        bridge.settle(reply)

        (fired,) = holder[0].fired
        assert fired.question == "does it dip anywhere?"
        reply_to = fired.extra["reply_to"]
        assert reply_to["sender"] == SENDER_NAME
        assert reply_to["text"] == "here is today's current trace"
        assert reply_to["attachments"] == ["current.csv"]
        (shipped,) = fired.extra["input_files"]
        assert shipped["filename"] == "current.csv"
        assert base64.b64decode(shipped["content_b64"]) == csv
        replies = web.calls_to("conversations.replies")
        assert [(c.args["channel"], c.args["ts"]) for c in replies] == [(CHANNEL_ID, parent["ts"])]


def test_an_attached_file_reaches_the_worker_and_unfetchable_ones_are_skipped_with_a_note(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three files on one DM: a Slack-hosted CSV is downloaded with the bot token and
    reaches the real worker as an input file; one whose download answers an HTML page is
    skipped as not-a-file; one hosted anywhere but ``files.slack.com`` is skipped before
    any request, so that host never sees a credential. Neither token appears in the
    bridge's log of the run."""
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.DEBUG, logger="osprey")
    csv = b"time,current\n10:00,402\n"
    good = web.add_file("current.csv", csv, mimetype="text/csv")
    page = web.add_file("notes.txt", b"<!DOCTYPE html><html>Sign in</html>", mimetype="text/html")
    foreign = {**web.add_file("plot.png", b"\x89PNG"), "id": "F0FOREIGN"}
    foreign["url_private"] = foreign["url_private_download"] = "https://evil.example.org/plot.png"
    payload = file_share([good, page, foreign], "please look at these")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("thanks")])
    ) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_dispatch(bridge, payload)
        bridge.settle(payload)

        (fired,) = holder[0].fired
        (shipped,) = fired.extra["input_files"]
        assert (shipped["filename"], shipped["mime"]) == ("current.csv", "text/csv")
        assert base64.b64decode(shipped["content_b64"]) == csv
        assert fired.extra["skipped_attachments"] == [
            {"filename": "notes.txt", "reason": NOT_A_FILE_REASON},
            {"filename": "plot.png", "reason": FOREIGN_HOST_REASON},
        ]
        # The real worker accepted the run carrying the file: it validates an
        # input-file batch (type allowlist, decodable bytes, size caps) at request
        # time, and a rejected batch yields no run id.
        assert entry["run_id"] == holder[0].results[0]["run_id"]

        assert not [a for a in web.attempts if "evil.example.org" in a.url]
        downloads = [a for a in web.attempts if "/files-pri/" in a.url]
        assert {a.authorization for a in downloads} == {f"Bearer {BOT_TOKEN}"}
        assert sorted(fid for fid, _ in web.downloads) == sorted([good["id"], page["id"]])

    assert caplog.records, "the bridge logged nothing; the token check would be vacuous"
    assert BOT_TOKEN not in caplog.text
    assert APP_TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# What reaches Slack: artifacts, long answers, mentions
# ---------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
ARTIFACT_ID = "a1"


class _ArtifactHandler(_FakeHandler):
    fake: ClassVar[Any]

    def do_GET(self) -> None:  # http.server API
        path, _ = self.split_path()
        if path == "/health":
            self.respond_json(200, {"status": "ok"})
            return
        if path.endswith(f"/artifacts/{ARTIFACT_ID}") and path.startswith("/dispatch/"):
            self.fake.requests.append((path, self.headers.get("Authorization")))
            self.respond(200, PNG, content_type="image/png")
            return
        self.respond(404, b"not found", content_type="text/plain")


class FakeArtifactRoute(FakeHttpService):
    """The worker's artifact byte route for a scripted run, on a loopback port."""

    handler_class: ClassVar[type[_FakeHandler]] = _ArtifactHandler

    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None]] = []
        super().__init__()


def test_a_run_artifact_is_uploaded_into_the_answers_thread(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's artifact is fetched from the worker's byte route with the dispatch token
    and uploaded through Slack's three-step external upload into the answer's thread —
    the pre-signed byte POST carrying no bearer."""
    payload = channel_mention("plot the current")
    holder: list[RecordingDispatcher] = []
    artifact = {
        "artifact_id": ARTIFACT_ID,
        "filename": "current.png",
        "delivered_mime": "image/png",
    }

    with FakeArtifactRoute() as route:
        _set_bridge_env(
            monkeypatch,
            state_dir=bridge_state,
            dispatch=dispatch_stack,
            worker_url=route.base_url,
        )
        with _running_bridge(
            web,
            socket_server,
            bridge_state,
            deps=_recording(holder, script=[completed("Here it is.", artifacts=[artifact])]),
            gate=lambda: True,
        ) as bridge:
            bridge.deliver(payload)
            entry = _wait_for_dispatch(bridge, payload)
            bridge.settle(payload)

            (upload,) = web.uploads
            assert upload.content == PNG
            assert (upload.channel_id, upload.thread_ts) == (CHANNEL_ID, _ts(payload))
            assert upload.filename.startswith("current") and upload.filename.endswith(".png")
            assert upload.upload_had_authorization is False
            assert route.requests == [
                (f"/dispatch/{entry['run_id']}/artifacts/{ARTIFACT_ID}", f"Bearer {DISPATCH_TOKEN}")
            ]
            assert _texts(web.posted) == [ack_text(VERSION_TAG), "Here it is."]


def test_an_over_limit_answer_arrives_complete_and_in_order(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer several times :data:`ANSWER_CHUNK_CHARS` long lands as several messages
    in the question's thread, each within the limit, which read back as the whole answer
    in its original order."""
    lines = [f"line {i:04d}: the orbit is within tolerance at this bpm" for i in range(400)]
    answer = "\n".join(lines)
    assert len(answer) > 3 * ANSWER_CHUNK_CHARS
    payload = channel_mention("list every bpm")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed(answer)])
    ) as bridge:
        bridge.deliver(payload)
        bridge.settle(payload)

        ack, *chunks = web.posted
        assert ack.text == ack_text(VERSION_TAG)
        assert len(chunks) >= 4
        assert all(len(c.text) <= ANSWER_CHUNK_CHARS for c in chunks)
        assert {(c.channel, c.thread_ts) for c in chunks} == {(CHANNEL_ID, _ts(payload))}
        received = [line for c in chunks for line in c.text.splitlines() if line.strip()]
        assert received == lines


def test_answer_mentions_ping_only_listed_roster_members(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``<@ID>`` of a member of the channel stays a live mention; a non-member's
    becomes plain text, and ``<!channel>``/``<!here>``/``<!everyone>``/user-group
    broadcasts are escaped so they ping nobody. The question went out with the room."""
    web.add_user(CAROL_ID, real_name="Carol", display_name="Carol")
    web.set_members(CHANNEL_ID, [SENDER_ID, CAROL_ID, BOT_USER_ID])
    answer = (
        f"Ask <@{CAROL_ID}>; <@{STRANGER_ID}> is not here. "
        "<!channel> <!here> <!everyone> <!subteam^S0OPS|@ops>"
    )
    payload = channel_mention("who knows the RF?")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed(answer)])
    ) as bridge:
        bridge.deliver(payload)
        bridge.settle(payload)

        posted = web.posted[-1].text
        assert f"<@{CAROL_ID}>" in posted
        assert f"<@{STRANGER_ID}>" not in posted and f"@{STRANGER_ID}" in posted
        assert "<!" not in posted, posted
        assert "&lt;!channel&gt;" in posted and "&lt;!here&gt;" in posted

        (fired,) = holder[0].fired
        room_ids = {m.get("id") for m in fired.extra["room"]["members"]}
        assert {SENDER_ID, CAROL_ID} <= room_ids and BOT_USER_ID not in room_ids


# ---------------------------------------------------------------------------
# The retry queue: queued, resumed, given up, superseded
# ---------------------------------------------------------------------------


def _resumed_line(text: str, prefix: str) -> bool:
    head, tail = RESUMED_TEXT.split("{since}")
    return text.startswith(prefix + head) and text.endswith(tail)


def test_a_failed_run_is_queued_then_resumed_with_its_answer(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run the provider failed before acting is parked: the user is told once that it
    is queued. The drain re-attaches, sees the run failed safely, re-dispatches, and the
    delayed answer arrives behind the resumed line — all in the same DM."""
    question = "what is the beam current?"
    payload = dm_message(question)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack, **FAST_DRAIN)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(holder, script=[provider_failure(), completed("402 mA")]),
    ) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_status(bridge, payload, "completed", POST_TIMEOUT_SEC)

        prefix = quote_prefix(entry)
        ack, queued, resumed, answer = _wait_for_posts(bridge, web, 4)
        assert ack.text == prefix + ack_text(VERSION_TAG)
        assert queued.text == prefix + QUEUED_TEXT
        assert _resumed_line(resumed.text, prefix), resumed.text
        assert answer.text == prefix + "402 mA"
        assert {p.channel for p in web.posted} == {DM_CHANNEL_ID}
        assert len(holder[0].fired_for(question)) == 2
        assert entry.get("retried") is True


def test_a_question_the_dispatcher_never_took_is_queued_and_then_given_up(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatcher down: the question is queued with the notice. Nothing proves the failed
    attempt took no action (no run ever reported a tool count), so the drain will not
    re-run it and gives up honestly instead, in the same thread."""
    payload = channel_mention("what is the beam current?")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack, **FAST_DRAIN)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(holder, dispatcher_url=DEAD_DISPATCHER_URL),
    ) as bridge:
        bridge.deliver(payload)
        entry = _wait_for_status(bridge, payload, "error", POST_TIMEOUT_SEC)

        ack, queued, gave_up = _wait_for_posts(bridge, web, 3)
        assert _texts([ack, queued, gave_up]) == [ack_text(VERSION_TAG), QUEUED_TEXT, GIVEUP_TEXT]
        assert {(p.channel, p.thread_ts) for p in web.posted} == {(CHANNEL_ID, _ts(payload))}
        assert entry.get("give_up_reason")
        assert entry.get("run_id") is None
        time.sleep(3 * float(FAST_DRAIN["DRAIN_INTERVAL"]))
        assert len(web.posted) == 3, _texts(web.posted)


def test_a_newer_queued_question_supersedes_the_older_with_a_note_on_the_older(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two questions from one person in one DM, both queued while the dispatcher is down:
    the newer takes the older's place, and the older gets the superseded note — quoting
    the OLDER question, so it reads as being about that one."""
    first = dm_message("what was the current at noon?")
    second = dm_message("what is the current now?")
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(holder, dispatcher_url=DEAD_DISPATCHER_URL),
    ) as bridge:
        bridge.deliver(first)
        bridge.settle(first)
        bridge.deliver(second)
        bridge.settle(second)

        older = bridge.dedup[_key(first)]
        newer = bridge.dedup[_key(second)]
        assert (older["status"], newer["status"]) == ("superseded", "queued")
        assert _texts(web.posted) == [
            quote_prefix(older) + ack_text(VERSION_TAG),
            quote_prefix(older) + QUEUED_TEXT,
            quote_prefix(newer) + ack_text(VERSION_TAG),
            quote_prefix(older) + SUPERSEDED_TEXT,
            quote_prefix(newer) + QUEUED_TEXT,
        ]


# ---------------------------------------------------------------------------
# Slack refusing or pacing the bridge
# ---------------------------------------------------------------------------


def test_an_archived_channel_refusal_settles_terminal_with_no_further_posts(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_archived`` on the answer is a refusal no retry can clear: the entry settles
    terminal with the refusal on record, and neither the drain nor anything else tries
    to post there again."""
    payload = channel_mention("what is the beam current?")
    web.fail_method("chat.postMessage", "is_archived", times=10)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack, **FAST_DRAIN)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("402 mA")])
    ) as bridge:
        bridge.deliver(payload)
        bridge.settle(payload)

        entry = bridge.dedup[_key(payload)]
        assert entry["status"] == "error"
        assert "undeliverable" in entry["give_up_reason"]
        assert "is_archived" in entry["give_up_reason"]
        # The ack (refused, swallowed) and the answer (refused, settled): two attempts.
        assert [c.error for c in web.calls_to("chat.postMessage")] == ["is_archived"] * 2
        time.sleep(3 * float(FAST_DRAIN["DRAIN_INTERVAL"]))
        assert len(web.calls_to("chat.postMessage")) == 2
        assert web.posted == []


def test_a_rate_limited_post_waits_at_least_the_retry_after_slack_asked_for(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 with ``Retry-After: 2`` on the ack delays the retried post by at least two
    seconds; the ack and the answer both still land, once each."""
    retry_after = 2
    payload = dm_message("what is the beam current?")
    web.rate_limit("chat.postMessage", retry_after=retry_after)
    holder: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web, socket_server, bridge_state, deps=_recording(holder, script=[completed("402 mA")])
    ) as bridge:
        bridge.deliver(payload)
        bridge.settle(payload)

        limited, retried, *_ = web.timed("chat.postMessage")
        assert (limited.call.status, retried.call.status) == (429, 200)
        assert retried.at - limited.at >= retry_after
        entry = bridge.dedup[_key(payload)]
        assert _texts(web.posted) == [
            quote_prefix(entry) + ack_text(VERSION_TAG),
            quote_prefix(entry) + "402 mA",
        ]


# ---------------------------------------------------------------------------
# A crash with an acknowledged question still in the inbox
# ---------------------------------------------------------------------------


def test_an_acknowledged_question_not_yet_started_is_answered_after_a_restart(
    web: TimedSlackWeb,
    socket_server: FakeSocketServer,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack was told the question was received; the bridge stopped before it started
    on it. The next process on the same state volume answers it, unprompted by Slack.

    The first bridge's handler pool is held busy (:data:`POOL_SIZE` dispatches waiting
    on a release), so one more question is acknowledged and spooled but never handed to
    the engine. The bridge is stopped, and only once its serve loop has seen the stop are
    the held dispatches released — so it cannot pick the waiting question up on the way
    out. A second bridge, booted on the same state directory and so the same default
    inbox, must answer that question without Slack redelivering anything.
    """
    busy = [dm_message(f"busy question {i}") for i in range(POOL_SIZE)]
    waiting = dm_message("the question still in the inbox")
    release = threading.Event()
    first: list[RecordingDispatcher] = []

    _set_bridge_env(monkeypatch, state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(
            first, hold=release, script=[completed(f"busy answer {i}") for i in range(POOL_SIZE)]
        ),
    ) as bridge:
        for payload in busy:
            bridge.deliver(payload)
        await_bridge(
            bridge,
            f"{POOL_SIZE} held dispatches",
            lambda: True if first[0].entered == POOL_SIZE else None,
            POST_TIMEOUT_SEC,
        )
        bridge.deliver(waiting)
        assert waiting["event_id"] in bridge.pending()
        assert _key(waiting) not in bridge.dedup

        bridge.wiring.stop.set()
        time.sleep(3 * TICK_SEC)  # the serve loop observes the stop and stops feeding
        release.set()
        bridge.thread.join(BRIDGE_JOIN_TIMEOUT_SEC)
        assert not bridge.thread.is_alive()

        assert waiting["event_id"] in bridge.pending()
        assert _key(waiting) not in bridge.dedup
        assert (bridge_state / INBOX_DIRNAME).is_dir()
        assert first[0].entered == POOL_SIZE

    posted_before = len(web.posted)
    second: list[RecordingDispatcher] = []
    with _running_bridge(
        web,
        socket_server,
        bridge_state,
        deps=_recording(second, script=[completed("answered after the restart")]),
    ) as bridge:
        entry = _wait_for_dispatch(bridge, waiting)
        bridge.settle(waiting)

        assert [f.question for f in second[0].fired] == ["the question still in the inbox"]
        assert _texts(web.posted[posted_before:]) == [
            quote_prefix(entry) + ack_text(VERSION_TAG),
            quote_prefix(entry) + "answered after the restart",
        ]
        assert bridge.pending() == []
