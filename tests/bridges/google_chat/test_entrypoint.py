"""The bridge process's wiring: the boot refusal, the collaborators, the seams, the stop.

``__main__`` decides nothing about messages, so what is worth testing is the shape of
what it builds — and every property here is one whose absence produces a bridge that
still starts, still logs, and still looks healthy:

*  **an incomplete environment aborts, naming the variables.** Including the pair with
   code defaults (``DISPATCHER_URL``/``WORKER_URL``), which under compose arrive as
   empty strings rather than absent keys, so no default ever fires and the bridge would
   POST every question to a protocol-less URL forever.
*  **one Chat client and one deps bundle**, the latter carrying this channel's
   two-route prior-artifact fetcher rather than the engine's worker-only default.
*  **all four injected seams are honored** — Pub/Sub subscriber, Chat service, storage
   client, worker HTTP — because the end-to-end tier boots this exact wiring with fakes
   in all four, and a seam that were quietly ignored would send the test at Google.
*  **one :class:`threading.Event`**, from the signal handler through ``run_forever`` to
   the ingestion loop. A second one is a container that logs a tidy shutdown and then
   keeps leasing messages until it is killed.

No Google library is imported at collection or at run time: the Chat seam is exercised
with a hand-written discovery stand-in, the storage seam with a fake client, and both
HTTP legs over :class:`httpx.MockTransport`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

import osprey.bridges.google_chat.__main__ as main_module
from osprey.bridges.core import PNG_MAGIC, CoreConfig, PipelineDeps
from osprey.bridges.google_chat import client as client_module
from osprey.bridges.google_chat.__main__ import (
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
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import GC_SPACE, GC_THREAD
from osprey.bridges.google_chat.gcs import publish_artifacts, publish_documents

SA_KEY = "/nonexistent/sa.json"
SUBSCRIPTION = "projects/proj/subscriptions/gchat-events"
APP_ID = "users/999"
BUCKET = "test-bucket"
SPACE = "spaces/AAA"
THREAD = "spaces/AAA/threads/TTT"
WORKER_URL = "http://work:9190"
WORKER_TOKEN = "work-token"
RUN_ID = "R1"
ARTIFACT_ID = "a1"

PNG_BYTES = PNG_MAGIC + b"fake-png-body"
BUDGET = 1_000_000

COMPLETE_ENV = {
    "GCHAT_SA_KEY": SA_KEY,
    "GCHAT_SUBSCRIPTION": SUBSCRIPTION,
    "GCHAT_APP_ID": APP_ID,
    "GCS_BUCKET": BUCKET,
    "DISPATCH_TRIGGER": "gchat-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": WORKER_TOKEN,
    "DISPATCHER_URL": "http://disp:9180",
    "WORKER_URL": WORKER_URL,
}
"""Everything a bridge needs, as a deployment sets it. Tests that check the refusal
start from this and take one thing away, so a passing refusal is always about the thing
that was removed."""

REQUIRED_NAMES = (
    "GCHAT_SA_KEY",
    "GCHAT_SUBSCRIPTION",
    "GCHAT_APP_ID",
    "DISPATCH_TRIGGER",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
)


@pytest.fixture
def cfg(tmp_path: Any) -> GoogleChatBridgeConfig:
    """A complete config whose stores live under ``tmp_path``.

    The paths matter: :func:`build_wiring` constructs the dedup and history stores, and
    a test must not read (or later write) a real deployment's ``/data``.
    """
    return GoogleChatBridgeConfig(
        sa_key=SA_KEY,
        subscription=SUBSCRIPTION,
        app_id=APP_ID,
        gcs_bucket=BUCKET,
        gcs_project="my-project",
        core=CoreConfig(
            trigger="gchat-question",
            dispatcher_url="http://disp:9180",
            worker_url=WORKER_URL,
            event_dispatcher_token="disp-token",
            dispatch_worker_token=WORKER_TOKEN,
            dedup_path=str(tmp_path / "dedup.json"),
            history_path=str(tmp_path / "history.json"),
        ),
    )


@pytest.fixture(autouse=True)
def _no_real_chat_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything reaches the real discovery build.

    ``ChatClient`` builds one the moment it is handed no service, which imports
    ``googleapiclient`` and reads the key file — so an un-honored ``chat_service`` seam
    would otherwise surface as a confusing ``ImportError`` (or, on a machine that has
    the extra, as a file error) instead of as the wiring bug it is. Tests that mean to
    exercise the default path override this with their own stub.
    """
    monkeypatch.setattr(
        client_module,
        "build_chat_service",
        lambda cfg: pytest.fail("the chat_service seam was ignored: a real service was built"),
    )


# --- stand-ins ---------------------------------------------------------------


class FakeChatService:
    """The ``spaces().messages().create(...)`` surface, recording what was posted."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def spaces(self) -> FakeChatService:
        return self

    def messages(self) -> FakeChatService:
        return self

    def create(self, **kwargs: Any) -> FakeRequest:
        return FakeRequest(self.created, kwargs)


class FakeRequest:
    def __init__(self, log: list[dict[str, Any]], call: dict[str, Any]) -> None:
        self._log = log
        self._call = call

    def execute(self) -> dict[str, Any]:
        self._log.append(self._call)
        return {"name": f"{SPACE}/messages/posted"}


class FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: bytes | None = None
        self.content_type: str | None = None

    def upload_from_string(self, data: bytes, content_type: str) -> None:
        self.uploaded = data
        self.content_type = content_type


class FakeBucket:
    def __init__(self, client: FakeGcsClient, name: str) -> None:
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        blob = FakeBlob(name)
        self.client.blobs.append(blob)
        return blob


class FakeGcsClient:
    """A storage client that keeps every blob it was asked to write."""

    def __init__(self) -> None:
        self.blobs: list[FakeBlob] = []
        self.buckets: list[str] = []

    def bucket(self, name: str) -> FakeBucket:
        self.buckets.append(name)
        return FakeBucket(self, name)


class FakeFuture:
    """A streaming-pull future that never completes and records its cancellation."""

    def __init__(self) -> None:
        self.cancelled = False

    def result(self, timeout: float | None = None) -> Any:
        raise TimeoutError

    def cancel(self) -> None:
        self.cancelled = True


class FakeSubscriber:
    def __init__(self) -> None:
        self.subscriptions: list[str] = []
        self.future = FakeFuture()

    def subscribe(self, subscription: str, callback: Callable[[Any], None]) -> FakeFuture:
        self.subscriptions.append(subscription)
        return self.future


class FakeRuntime:
    """Enough :class:`~osprey.bridges.core.BridgeRuntime` for the ingestion loop."""

    def __init__(self) -> None:
        self.supervised = 0

    def supervise(self) -> None:
        self.supervised += 1

    def handle_event(self, event: Any) -> str:
        return "ignored"


def worker_route(body: bytes, content_type: str) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client serving the worker's byte route, plus the requests it was asked to make.

    The request list is the assertion that matters: an injected client that the wiring
    dropped on the floor records nothing, whatever the call happens to return.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


# --- refusing an incomplete environment --------------------------------------


def test_main_exits_naming_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment is replaced wholesale rather than pruned by name: the process
    that runs these tests may itself carry a ``DISPATCH_*`` value, and a refusal test
    that depends on the ambient environment proves nothing."""
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert "google chat bridge cannot start" in message
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


def test_main_aborts_before_building_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation precedes construction, so a misconfigured deployment fails on its
    environment rather than on a Google import or an unreadable key file."""
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(
        main_module, "build_wiring", lambda *a, **k: pytest.fail("built a wiring anyway")
    )

    with pytest.raises(SystemExit):
        main()


def test_config_from_env_reads_a_complete_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", dict(COMPLETE_ENV))

    cfg = config_from_env()

    assert cfg.subscription == SUBSCRIPTION
    assert cfg.app_id == APP_ID
    assert cfg.gcs_bucket == BUCKET
    assert cfg.core.trigger == "gchat-question"
    assert cfg.core.worker_url == WORKER_URL


def test_run_refuses_an_incomplete_config(cfg: GoogleChatBridgeConfig) -> None:
    """``run`` is reachable in-process without ``config_from_env``, so it validates
    again rather than trusting its caller."""
    wiring = build_wiring(
        GoogleChatBridgeConfig(sa_key=SA_KEY, core=cfg.core), chat_service=FakeChatService()
    )

    with pytest.raises(ValueError, match="GCHAT_SUBSCRIPTION"):
        run(wiring)


# --- what build_wiring builds ------------------------------------------------


def test_the_ops_object_and_the_wiring_share_one_chat_client(cfg: GoogleChatBridgeConfig) -> None:
    """One client, because it serializes its HTTP leg behind its own lock — the drain
    thread posts alongside live ingestion, and a second instance is a second lock."""
    wiring = build_wiring(cfg, chat_service=FakeChatService())

    assert wiring.ops._client is wiring.client


def test_the_deps_bundle_carries_this_channels_prior_artifact_fetcher(
    cfg: GoogleChatBridgeConfig,
) -> None:
    wiring = build_wiring(cfg, chat_service=FakeChatService())

    assert wiring.deps.ops is wiring.ops
    assert wiring.deps.cfg is cfg.core
    assert (
        wiring.deps.fetch_prior_artifact
        is not PipelineDeps.__dataclass_fields__["fetch_prior_artifact"].default
    ), "the engine's worker-only default survived: the published GCS copy is unreachable"


def test_the_wiring_is_frozen(cfg: GoogleChatBridgeConfig) -> None:
    """Every field is read concurrently by the subscriber and the drain thread, and the
    stop event in particular must not be swappable under a thread already waiting on
    it."""
    wiring = build_wiring(cfg, chat_service=FakeChatService())

    assert isinstance(wiring, Wiring)
    with pytest.raises(dataclasses.FrozenInstanceError):
        wiring.stop = threading.Event()  # type: ignore[misc]


# --- the four seams ----------------------------------------------------------


def test_the_injected_chat_service_is_what_the_ops_object_posts_through(
    cfg: GoogleChatBridgeConfig,
) -> None:
    """Behavioural, not identity: the autouse fixture already fails a service *build*,
    so what is left to prove is that a post lands on the injected one."""
    service = FakeChatService()
    wiring = build_wiring(cfg, chat_service=service)

    wiring.ops.post_ack({GC_SPACE: SPACE, GC_THREAD: THREAD})

    assert len(service.created) == 1
    assert service.created[0]["parent"] == SPACE


def test_a_chat_service_is_built_from_the_config_when_none_is_injected(
    cfg: GoogleChatBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[GoogleChatBridgeConfig] = []
    service = FakeChatService()

    def fake_build(config: GoogleChatBridgeConfig) -> FakeChatService:
        built.append(config)
        return service

    monkeypatch.setattr(client_module, "build_chat_service", fake_build)

    wiring = build_wiring(cfg)

    assert built == [cfg]
    assert wiring.client._service is service


def test_the_injected_subscriber_factory_and_stop_event_reach_the_pull(
    cfg: GoogleChatBridgeConfig,
) -> None:
    """The whole ingestion loop, over a fake subscriber: the factory is called with the
    config, the pull is opened on the configured subscription, and a stop already set
    tears it down instead of blocking."""
    subscriber = FakeSubscriber()
    asked: list[GoogleChatBridgeConfig] = []

    def factory(config: GoogleChatBridgeConfig) -> FakeSubscriber:
        asked.append(config)
        return subscriber

    wiring = build_wiring(cfg, chat_service=FakeChatService(), subscriber_factory=factory)
    wiring.stop.set()

    serve_events(wiring, FakeRuntime())  # type: ignore[arg-type]

    assert asked == [cfg]
    assert subscriber.subscriptions == [SUBSCRIPTION]
    assert subscriber.future.cancelled


def test_no_subscriber_factory_is_named_when_none_was_injected(
    cfg: GoogleChatBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default lives in the subscriber module's own signature and is left there —
    naming it here would be a second place to keep correct."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main_module, "serve", lambda *a, **k: calls.append(k))

    serve_events(build_wiring(cfg, chat_service=FakeChatService()), FakeRuntime())  # type: ignore[arg-type]

    assert calls and "subscriber_factory" not in calls[0]


def test_the_publishers_carry_the_injected_storage_and_worker_clients(
    cfg: GoogleChatBridgeConfig,
) -> None:
    """One publish, end to end over both injected clients: the bytes come off the
    injected worker transport and land in the injected storage client."""
    http, seen = worker_route(PNG_BYTES, "image/png")
    gcs = FakeGcsClient()
    wiring = build_wiring(cfg, chat_service=FakeChatService(), gcs_client=gcs, worker_http=http)

    urls = wiring.ops._publisher(cfg, RUN_ID, [ARTIFACT_ID])

    assert len(urls) == 1
    assert urls[0].startswith(f"https://storage.googleapis.com/{BUCKET}/plots/")
    assert [blob.uploaded for blob in gcs.blobs] == [PNG_BYTES]
    assert gcs.buckets == [BUCKET]
    assert [str(request.url) for request in seen] == [
        f"{WORKER_URL}/dispatch/{RUN_ID}/artifacts/{ARTIFACT_ID}"
    ]


def test_the_publishers_are_the_gcs_modules_own_when_nothing_is_injected(
    cfg: GoogleChatBridgeConfig,
) -> None:
    wiring = build_wiring(cfg, chat_service=FakeChatService())

    assert wiring.ops._publisher is publish_artifacts
    assert wiring.ops._doc_publisher is publish_documents


def test_the_prior_artifact_fetcher_reads_the_worker_over_the_injected_client(
    cfg: GoogleChatBridgeConfig,
) -> None:
    http, seen = worker_route(PNG_BYTES, "image/png")
    wiring = build_wiring(cfg, chat_service=FakeChatService(), worker_http=http)

    data, served = wiring.deps.fetch_prior_artifact(
        cfg.core, RUN_ID, {"entry_id": ARTIFACT_ID}, BUDGET
    )

    assert (data, served) == (PNG_BYTES, "image/png")
    assert [str(request.url) for request in seen] == [
        f"{WORKER_URL}/dispatch/{RUN_ID}/artifacts/{ARTIFACT_ID}"
    ]


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
    cfg: GoogleChatBridgeConfig, sig: signal.Signals
) -> None:
    wiring = build_wiring(cfg, chat_service=FakeChatService())
    install_signal_handlers(wiring.stop)

    handler = signal.getsignal(sig)
    assert callable(handler)
    handler(sig, None)

    assert wiring.stop.is_set()


def test_run_threads_one_event_through_the_engine_and_the_ingestion_loop(
    cfg: GoogleChatBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single load-bearing invariant of shutdown: what ``run_forever`` stops on and
    what the pull stops on are the same object, so one ``set()`` ends both."""
    forever: dict[str, Any] = {}
    served: list[dict[str, Any]] = []

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:
        forever.update(core=core, ops=ops, serve=serve, **kwargs)

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)
    monkeypatch.setattr(main_module, "serve", lambda *a, **k: served.append(k))

    wiring = build_wiring(cfg, chat_service=FakeChatService())
    run(wiring)

    assert forever["core"] is cfg.core
    assert forever["ops"] is wiring.ops
    assert forever["deps"] is wiring.deps
    assert forever["stop"] is wiring.stop

    forever["serve"](FakeRuntime())
    assert served and served[0]["stop"] is wiring.stop


def test_run_lets_a_caller_replace_the_deps_bundle(
    cfg: GoogleChatBridgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An end-to-end tier boots this ``run`` with a bundle over its own transports; the
    ``gate`` seam travels the same way."""
    forever: dict[str, Any] = {}

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:
        forever.update(kwargs)

    def always_healthy() -> bool:
        return True

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)

    wiring = build_wiring(cfg, chat_service=FakeChatService())
    replacement = PipelineDeps(
        cfg=cfg.core, ops=wiring.ops, dedup=wiring.deps.dedup, dispatcher=wiring.deps.dispatcher
    )

    run(wiring, deps=replacement, gate=always_healthy)

    assert forever["deps"] is replacement
    assert forever["gate"] is always_healthy


# --- logging -----------------------------------------------------------------


def test_the_chatty_libraries_are_demoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of these logs per request, and this process makes one per posted chunk, per
    published artifact and per token refresh."""
    for name in QUIET_LIBRARIES:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)

    configure_logging()

    assert all(logging.getLogger(name).level == logging.WARNING for name in QUIET_LIBRARIES)
