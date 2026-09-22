"""The bridge process's wiring: the boot refusal, the collaborators, the seams, the stop.

``__main__`` decides nothing about messages, so what is worth testing is the shape of
what it builds — and every property here is one whose absence produces a bridge that
still starts, still logs, and still looks healthy:

*  **an incomplete environment aborts, naming the variables.** Including the pair with
   code defaults (``DISPATCHER_URL``/``WORKER_URL``), which under compose arrive as
   empty strings rather than absent keys, so no default ever fires and the bridge would
   POST every question to a protocol-less URL forever.
*  **one token source behind one connector client**, shared by the receive threads and
   the drain thread. Two token sources are two AAD exchanges on every expiry; two
   connector clients are two HTTP locks where the point of the lock was that there is
   one.
*  **all four injected seams are honored** — queue receiver, token leg, connector leg,
   worker byte route — because the end-to-end tier boots this exact wiring with fakes
   in all four, and a seam that were quietly ignored would send the test at Microsoft.
*  **the receiver is closed when the pull ends, and the close is guarded.** The
   production receiver holds an AMQP link and renewal threads; ``close`` is not in the
   :class:`~osprey.bridges.teams.receiver.QueueReceiver` Protocol and the end-to-end
   lane's in-process fake has none, so an unguarded call would break the lane that
   proves the wiring.
*  **one :class:`threading.Event`**, from the signal handler through ``run_forever`` to
   the ingestion loop. A second one is a container that logs a tidy shutdown and then
   keeps leasing messages until it is killed.

No Service Bus and no Pillow are imported, at collection or at run time: the receiver
seam is exercised with hand-written stand-ins and both HTTP legs over
:class:`httpx.MockTransport`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import osprey.bridges.teams.__main__ as main_module
from osprey.bridges.core import CoreConfig, PipelineDeps
from osprey.bridges.teams.__main__ import (
    CANNOT_START,
    QUIET_LIBRARIES,
    Wiring,
    build_wiring,
    config_from_env,
    configure_logging,
    install_signal_handlers,
    main,
    run,
    serve_events,
)
from osprey.bridges.teams.client import token_url
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.ops import ack_text
from osprey.bridges.teams.receiver import QueueReceiver, make_receiver
from tests.bridges.teams.test_posting import (
    APP_ID,
    TENANT,
    VERSION,
    make_entry,
)

QUEUE = "teams-activities"
CONNECTION_STRING = (
    "Endpoint=sb://ns.servicebus.windows.net/;SharedAccessKeyName=bridge-listen;"
    "SharedAccessKey=secret;EntityPath=teams-activities"
)
DISPATCHER_URL = "http://disp:9180"
WORKER_URL = "http://work:9190"
ACCESS_TOKEN = "a-bearer"

COMPLETE_ENV = {
    "TEAMS_APP_ID": APP_ID,
    "TEAMS_APP_SECRET": "a-client-secret",
    "TEAMS_TENANT_ID": TENANT,
    "TEAMS_SERVICEBUS_CONNECTION_STRING": CONNECTION_STRING,
    "TEAMS_SERVICEBUS_QUEUE": QUEUE,
    "DISPATCH_TRIGGER": "teams-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
    "DISPATCHER_URL": DISPATCHER_URL,
    "WORKER_URL": WORKER_URL,
    "APP_VERSION_DISPLAY": VERSION,
}
"""Everything a bridge needs, as a deployment sets it. Tests that check the refusal
start from this and take one thing away, so a passing refusal is always about the thing
that was removed."""

REQUIRED_NAMES = (
    "TEAMS_APP_ID",
    "TEAMS_APP_SECRET",
    "TEAMS_TENANT_ID",
    "TEAMS_SERVICEBUS_CONNECTION_STRING",
    "TEAMS_SERVICEBUS_QUEUE",
    "DISPATCH_TRIGGER",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
)

EXPORTED_NAMES = (
    "ANSWER_CHUNK_CHARS",
    "ConnectorClient",
    "EMPTY_ANSWER_TEXT",
    "ERROR_TEXT",
    "GIVEUP_TEXT",
    "MS_ACTIVITY_ID",
    "MS_CONVERSATION_ID",
    "MS_CONVERSATION_TYPE",
    "MS_SERVICE_URL",
    "MS_TENANT_ID",
    "QUEUED_TEXT",
    "QueueReceiver",
    "ReceiverFactory",
    "SUPERSEDED_TEXT",
    "ServiceBusQueueReceiver",
    "TeamsBridgeConfig",
    "TeamsOps",
    "TokenSource",
    "ack_text",
    "make_receiver",
    "markdown_to_teams",
    "parse_event",
    "quote_prefix",
    "require_boot",
    "resolve_reply_context",
    "serve",
    "skipped_images_note",
)
"""What the package root must hand out. The wording constants are here because the
end-to-end lane asserts posted text by equality against them; the types because the
lane wires the bridge from the root alone."""


@pytest.fixture
def cfg(tmp_path: Any) -> TeamsBridgeConfig:
    """A complete config whose stores live under ``tmp_path``.

    The paths matter: :func:`build_wiring` constructs the dedup and history stores, and
    a test must not read (or later write) a real deployment's ``/data``.
    """
    return TeamsBridgeConfig(
        app_id=APP_ID,
        app_secret="a-client-secret",
        tenant_id=TENANT,
        servicebus_connection_string=CONNECTION_STRING,
        servicebus_queue=QUEUE,
        version_tag=VERSION,
        core=CoreConfig(
            trigger="teams-question",
            dispatcher_url=DISPATCHER_URL,
            worker_url=WORKER_URL,
            event_dispatcher_token="disp-token",
            dispatch_worker_token="work-token",
            dedup_path=str(tmp_path / "dedup.json"),
            history_path=str(tmp_path / "history.json"),
        ),
    )


# --- stand-ins ---------------------------------------------------------------


class RecordingReceiver:
    """A :class:`~osprey.bridges.teams.receiver.QueueReceiver` that records and closes.

    The production receiver's shape rather than the end-to-end lane's: it has the
    ``close`` the real one has, so a test can see that the process shuts it down.
    """

    def __init__(self) -> None:
        self.pulls: list[tuple[int, float]] = []
        self.closed = 0

    def receive(self, max_messages: int, max_wait: float) -> list[Any]:
        self.pulls.append((max_messages, max_wait))
        return []

    def complete(self, _msg: Any) -> None:
        raise AssertionError("nothing was delivered to complete")

    def dead_letter(self, _msg: Any, _reason: str) -> None:
        raise AssertionError("nothing was delivered to dead-letter")

    def register(self, _msg: Any) -> None:
        raise AssertionError("nothing was delivered to register")

    def close(self) -> None:
        self.closed += 1


class CloselessReceiver(RecordingReceiver):
    """A receiver with no ``close`` at all, as the end-to-end lane's fake queue is.

    ``close`` is deliberately outside the Protocol, so this is a *conforming*
    receiver — which is exactly why the shutdown path has to look before it calls.
    """

    close = None  # type: ignore[assignment]


class FailingCloseReceiver(RecordingReceiver):
    """A receiver whose shutdown raises, as a broken AMQP link's would."""

    def close(self) -> None:
        self.closed += 1
        raise RuntimeError("the link was already gone")


class FakeRuntime:
    """Enough :class:`~osprey.bridges.core.BridgeRuntime` for the ingestion loop."""

    def __init__(self) -> None:
        self.supervised = 0

    def supervise(self) -> None:
        self.supervised += 1

    def handle_event(self, _event: Any) -> str:
        return "ignored"


def recording_client(
    handler: Any,
) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client over ``handler``, plus the requests it was asked to make.

    The request list is the assertion that matters: an injected client that the wiring
    dropped on the floor records nothing, whatever the call happens to return.
    """
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record)), seen


def token_leg() -> tuple[httpx.Client, list[httpx.Request]]:
    """A client answering the AAD client-credentials exchange."""
    return recording_client(
        lambda _request: httpx.Response(
            200, json={"access_token": ACCESS_TOKEN, "expires_in": 3600}
        )
    )


def connector_leg() -> tuple[httpx.Client, list[httpx.Request]]:
    """A client answering the Bot Connector's reply route."""
    return recording_client(lambda _request: httpx.Response(201, json={"id": "posted"}))


def factory_for(receiver: Any) -> tuple[Any, list[TeamsBridgeConfig]]:
    """A ``receiver_factory`` returning ``receiver``, plus what it was called with."""
    asked: list[TeamsBridgeConfig] = []

    def factory(config: TeamsBridgeConfig) -> Any:
        asked.append(config)
        return receiver

    return factory, asked


# --- refusing an incomplete environment --------------------------------------


def test_main_exits_naming_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment is replaced wholesale rather than pruned by name: the process
    that runs these tests may itself carry a ``DISPATCH_*`` value, and a refusal test
    that depends on the ambient environment proves nothing."""
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert CANNOT_START.format(reason="") in message
    for name in REQUIRED_NAMES:
        assert name in message, f"{name} missing from the abort: {message}"


def test_main_exits_when_the_dispatch_urls_render_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compose trap: a bare unset ``${VAR}`` arrives as ``""``, not as an absent
    key, so ``CoreConfig``'s localhost defaults never fire. Without the second check the
    bridge boots and re-POSTs every question to ``""`` forever."""
    monkeypatch.setattr(os, "environ", COMPLETE_ENV | {"DISPATCHER_URL": "", "WORKER_URL": ""})

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert "DISPATCHER_URL" in message
    assert "WORKER_URL" in message


def test_the_module_is_runnable_as_the_containers_command() -> None:
    """``python -m osprey.bridges.teams`` is the compose ``command``, so the one thing
    no in-process test can prove is proven here: the module runs as a script at all, and
    a deployment missing a variable dies with a non-zero status naming it rather than
    sitting there looking healthy.

    The child's environment is built rather than inherited — an ambient ``DISPATCH_*``
    from the developer's shell would make this pass for the wrong reason."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}

    finished = subprocess.run(
        [sys.executable, "-m", "osprey.bridges.teams"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert finished.returncode != 0
    for name in REQUIRED_NAMES:
        assert name in finished.stderr, f"{name} missing from the abort: {finished.stderr}"


def test_main_aborts_before_building_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation precedes construction, so a misconfigured deployment fails on its
    environment rather than on a Service Bus import or an unreachable namespace."""
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(
        main_module, "build_wiring", lambda *a, **k: pytest.fail("built a wiring anyway")
    )

    with pytest.raises(SystemExit):
        main()


def test_config_from_env_reads_a_complete_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", dict(COMPLETE_ENV))

    parsed = config_from_env()

    assert parsed.app_id == APP_ID
    assert parsed.tenant_id == TENANT
    assert parsed.servicebus_queue == QUEUE
    assert parsed.version_tag == VERSION
    assert parsed.core.trigger == "teams-question"
    assert parsed.core.worker_url == WORKER_URL


def test_run_refuses_an_incomplete_config(cfg: TeamsBridgeConfig) -> None:
    """``run`` is reachable in-process without ``config_from_env``, so it validates
    again rather than trusting its caller."""
    wiring = build_wiring(TeamsBridgeConfig(core=cfg.core))

    with pytest.raises(ValueError, match="TEAMS_APP_ID"):
        run(wiring)


# --- what build_wiring builds ------------------------------------------------


def test_the_ops_object_posts_through_the_wirings_connector_client(
    cfg: TeamsBridgeConfig,
) -> None:
    """One client, because it serializes its HTTP leg behind its own lock — the drain
    thread posts alongside live ingestion, and a second instance is a second lock."""
    wiring = build_wiring(cfg)

    assert wiring.ops._client is wiring.client


def test_the_connector_client_asks_the_wirings_one_token_source(cfg: TeamsBridgeConfig) -> None:
    """A second token source would mint a second bearer on every expiry, from whichever
    thread noticed first, for the same bot."""
    wiring = build_wiring(cfg)

    assert wiring.client._tokens is wiring.tokens


def test_the_deps_bundle_is_built_over_this_wirings_ops(cfg: TeamsBridgeConfig) -> None:
    wiring = build_wiring(cfg)

    assert wiring.deps.ops is wiring.ops
    assert wiring.deps.cfg is cfg.core


def test_the_engines_own_prior_artifact_fetcher_is_kept(cfg: TeamsBridgeConfig) -> None:
    """Deliberate, and the opposite of the Google Chat wiring's choice: Teams posts
    image bytes inline and stamps no ``public_url``, so the worker's copy is the only
    copy a follow-up question could be given and the engine's default already reads
    it. An override here would be a second implementation of one fetch."""
    wiring = build_wiring(cfg)

    assert (
        wiring.deps.fetch_prior_artifact
        is PipelineDeps.__dataclass_fields__["fetch_prior_artifact"].default
    )


def test_the_wiring_is_frozen(cfg: TeamsBridgeConfig) -> None:
    """Every field is read concurrently by the receive threads and the drain thread, and
    the stop event in particular must not be swappable under a thread already waiting on
    it."""
    wiring = build_wiring(cfg)

    assert isinstance(wiring, Wiring)
    with pytest.raises(dataclasses.FrozenInstanceError):
        wiring.stop = threading.Event()  # type: ignore[misc]


# --- the four seams ----------------------------------------------------------


def test_the_injected_token_and_connector_clients_are_what_a_post_travels_over(
    cfg: TeamsBridgeConfig,
) -> None:
    """Behavioural, not identity: one real ``post_ack`` over both injected legs, which
    is what the end-to-end lane does and what a dropped seam would fail."""
    token_http, token_requests = token_leg()
    connector_http, connector_requests = connector_leg()
    wiring = build_wiring(cfg, token_http=token_http, connector_http=connector_http)

    wiring.ops.post_ack(make_entry())

    assert [str(request.url) for request in token_requests] == [token_url(cfg)]
    assert len(connector_requests) == 1
    assert connector_requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert json.loads(connector_requests[0].content)["text"] == ack_text(VERSION)


def test_the_injected_worker_client_is_the_one_the_artifact_route_uses(
    cfg: TeamsBridgeConfig,
) -> None:
    """Identity rather than a fetch: the byte route is reached only from the delivery
    path, which needs Pillow and a real PNG, and what this module owns is whether the
    injected client reached ops at all."""
    worker_http, _ = recording_client(lambda _request: httpx.Response(404))
    wiring = build_wiring(cfg, worker_http=worker_http)

    assert wiring.ops._http is worker_http


def test_the_injected_receiver_factory_is_asked_for_the_receiver_the_loop_pulls(
    cfg: TeamsBridgeConfig,
) -> None:
    """The whole ingestion loop over a stand-in receiver: the factory is called with the
    config, and a stop already set ends the loop before it pulls."""
    receiver = RecordingReceiver()
    factory, asked = factory_for(receiver)
    wiring = build_wiring(cfg, receiver_factory=factory)
    wiring.stop.set()

    serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]

    assert asked == [cfg]
    assert receiver.pulls == []


def test_the_receiver_is_closed_once_the_pull_has_returned(cfg: TeamsBridgeConfig) -> None:
    """The AMQP link and its renewal threads are this process's to release, and
    ``serve`` has already waited for every in-flight handler by the time it returns."""
    receiver = RecordingReceiver()
    factory, _ = factory_for(receiver)
    wiring = build_wiring(cfg, receiver_factory=factory)
    wiring.stop.set()

    serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]

    assert receiver.closed == 1


def test_a_receiver_without_a_close_shuts_down_cleanly(cfg: TeamsBridgeConfig) -> None:
    """``close`` is outside the Protocol, so a conforming receiver need not have one —
    and the end-to-end lane's in-process queue does not. An unguarded call here would
    fail the lane that proves this wiring."""
    receiver = CloselessReceiver()
    assert isinstance(receiver, QueueReceiver)
    factory, _ = factory_for(receiver)
    wiring = build_wiring(cfg, receiver_factory=factory)
    wiring.stop.set()

    serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]


def test_a_failed_close_is_logged_rather_than_raised(
    cfg: TeamsBridgeConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """This runs on the way out of the process: a raise here would replace the reason
    the bridge is stopping with the reason it could not tidy up."""
    receiver = FailingCloseReceiver()
    factory, _ = factory_for(receiver)
    wiring = build_wiring(cfg, receiver_factory=factory)
    wiring.stop.set()

    with caplog.at_level(logging.WARNING, logger=main_module.__name__):
        serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]

    assert receiver.closed == 1
    assert "closing the service bus receiver failed" in caplog.text


def test_a_raise_from_the_pull_still_closes_the_receiver(cfg: TeamsBridgeConfig) -> None:
    """A receiver that has died must fail loudly *and* be released: ``serve`` re-raises,
    and the close belongs in a ``finally`` rather than after the call."""
    receiver = RecordingReceiver()
    factory, _ = factory_for(receiver)
    wiring = build_wiring(cfg, receiver_factory=factory)

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("the receiver is gone")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main_module, "serve", explode)
        with pytest.raises(RuntimeError, match="the receiver is gone"):
            serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]

    assert receiver.closed == 1


@pytest.mark.usefixtures("no_servicebus")
def test_the_default_factory_is_the_packages_own_and_opens_nothing(cfg: TeamsBridgeConfig) -> None:
    """Wiring a bridge imports no Service Bus: the factory is named here and called
    only when the engine is ready to pull, which is what lets an ``osprey build`` host
    and every non-extra test import this module."""
    wiring = build_wiring(cfg)

    assert wiring.receiver_factory is make_receiver


# --- one stop event ----------------------------------------------------------


@pytest.fixture
def _restore_signal_handlers() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.usefixtures("_restore_signal_handlers")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_both_signals_set_the_wirings_stop_event(
    cfg: TeamsBridgeConfig, sig: signal.Signals
) -> None:
    wiring = build_wiring(cfg)
    install_signal_handlers(wiring.stop)

    handler = signal.getsignal(sig)
    assert callable(handler)
    handler(sig, None)

    assert wiring.stop.is_set()


def test_run_threads_one_event_through_the_engine_and_the_ingestion_loop(
    cfg: TeamsBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single load-bearing invariant of shutdown: what ``run_forever`` stops on and
    what the pull stops on are the same object, so one ``set()`` ends both."""
    forever: dict[str, Any] = {}
    served: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    receiver = RecordingReceiver()
    factory, _ = factory_for(receiver)

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:
        forever.update(core=core, ops=ops, serve=serve, **kwargs)

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)
    monkeypatch.setattr(main_module, "serve", lambda *a, **k: served.append((a, k)))

    wiring = build_wiring(cfg, receiver_factory=factory)
    run(wiring)

    assert forever["core"] is cfg.core
    assert forever["ops"] is wiring.ops
    assert forever["deps"] is wiring.deps
    assert forever["stop"] is wiring.stop

    forever["serve"](FakeRuntime())
    assert served and served[0][1]["stop"] is wiring.stop
    assert served[0][0][0] is receiver


def test_run_lets_a_caller_replace_the_deps_bundle(
    cfg: TeamsBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An end-to-end tier boots this ``run`` with a bundle over its own transports; the
    ``gate`` seam travels the same way."""
    forever: dict[str, Any] = {}

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:  # noqa: ARG001 - stands in for run_forever, which collects its remaining arguments as keywords
        forever.update(kwargs)

    def always_healthy() -> bool:
        return True

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)

    wiring = build_wiring(cfg)
    replacement = PipelineDeps(
        cfg=cfg.core, ops=wiring.ops, dedup=wiring.deps.dedup, dispatcher=wiring.deps.dispatcher
    )

    run(wiring, deps=replacement, gate=always_healthy)

    assert forever["deps"] is replacement
    assert forever["gate"] is always_healthy


# --- logging -----------------------------------------------------------------


def test_the_chatty_libraries_are_demoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of these logs per request or per AMQP frame, and this process makes one
    request per posted chunk and per token refresh and renews a message lock every few
    seconds."""
    for name in QUIET_LIBRARIES:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)

    configure_logging()

    assert all(logging.getLogger(name).level == logging.WARNING for name in QUIET_LIBRARIES)


def test_the_startup_line_names_no_credential(
    cfg: TeamsBridgeConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A startup line is the most-copied line in any incident, and either the client
    secret or the listen connection string is enough to act as the bot."""
    monkeypatch.setattr(main_module, "run_forever", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        run(build_wiring(cfg))

    assert QUEUE in caplog.text
    assert cfg.app_secret not in caplog.text
    assert "SharedAccessKey=secret" not in caplog.text


# --- the package's import surface --------------------------------------------


def test_the_package_root_hands_out_every_name_the_bridge_is_wired_from() -> None:
    """The end-to-end lane imports the wiring types and the posted wording from the
    root alone, and compares what a conversation received against these constants by
    equality."""
    package = importlib.import_module("osprey.bridges.teams")

    assert set(EXPORTED_NAMES) <= set(package.__all__)
    for name in package.__all__:
        assert getattr(package, name, None) is not None, f"{name} is exported but unresolvable"


def test_the_exported_names_are_sorted() -> None:
    """One order, so a new export lands in one obvious place rather than at the end."""
    package = importlib.import_module("osprey.bridges.teams")

    assert package.__all__ == sorted(package.__all__)


@pytest.mark.usefixtures("no_servicebus", "no_pillow")
def test_the_package_root_imports_without_the_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standing promise of the package, proven the only way it can be: with both
    optional distributions blocked AND the package purged from the module cache, so the
    import runs for real rather than being served from a cache some earlier test
    filled."""
    for name in list(sys.modules):
        if name == "osprey.bridges.teams" or name.startswith("osprey.bridges.teams."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    package = importlib.import_module("osprey.bridges.teams")

    for name in EXPORTED_NAMES:
        assert getattr(package, name, None) is not None, f"{name} vanished without the extra"
