"""The bridge process's wiring: the boot refusal, the collaborators, the seams, the stop.

``__main__`` decides nothing about messages, so what is worth testing is the shape of
what it builds — every property here is one whose absence produces a bridge that still
starts, still logs, and still looks healthy:

*  **an incomplete environment aborts, naming the variables**, including the
   dispatcher/worker URLs that compose renders as empty strings.
*  **a credential Slack refuses at ``auth.test`` aborts with the cause**, as the
   cannot-start exit rather than a traceback.
*  **one Web API client and one fatal event**, shared by ops, socket and ingest.
*  **the proxy split** (Decision 2): the Slack legs follow ``BRIDGE_TRUST_ENV``; the
   engine's deps and the worker byte route never use a proxy.
*  **ingest starts before the engine** (Decision 4) and is stopped whatever ends it.
*  **one stop event**, from the signal handler through ``run_forever`` to serve.

No network: the Web API leg runs over :class:`httpx.MockTransport` and the socket is
never iterated except through stand-ins.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

import osprey.bridges.slack.__main__ as main_module
from osprey.bridges.core import CoreConfig, PipelineDeps
from osprey.bridges.slack.__main__ import (
    CANNOT_START,
    QUIET_LIBRARIES,
    Wiring,
    build_wiring,
    config_from_env,
    configure_logging,
    default_inbox_dir,
    install_signal_handlers,
    main,
    run,
)
from osprey.bridges.slack.client import SlackAuthError, SlackWebClient
from osprey.bridges.slack.config import SlackBridgeConfig
from osprey.bridges.slack.events import BotIdentity
from osprey.bridges.slack.inbox import Inbox
from tests.bridges.conftest import recording_client
from tests.bridges.slack._posting import BOT_ID, BOT_USER, TEAM, VERSION

BOT_TOKEN = "xoxb-1111-2222-bot-secret"
APP_TOKEN = "xapp-1-A111-3333-app-secret"  # gitleaks:allow - a fake test token
DISPATCHER_URL = "http://disp:9180"
WORKER_URL = "http://work:9190"

COMPLETE_ENV = {
    "SLACK_BOT_TOKEN": BOT_TOKEN,
    "SLACK_APP_TOKEN": APP_TOKEN,
    "DISPATCH_TRIGGER": "slack-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
    "DISPATCHER_URL": DISPATCHER_URL,
    "WORKER_URL": WORKER_URL,
    "APP_VERSION_DISPLAY": VERSION,
}
"""Everything a bridge needs, as a deployment sets it."""

REQUIRED_NAMES = (
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "DISPATCH_TRIGGER",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
)

EXPORTED_NAMES = (
    "ACK_TEXT",
    "ANSWER_CHUNK_CHARS",
    "BotIdentity",
    "EMPTY_ANSWER_TEXT",
    "ERROR_TEXT",
    "GIVEUP_TEXT",
    "Inbox",
    "Ingest",
    "QUEUED_TEXT",
    "RESUMED_TEXT",
    "SKIPPED_FILES_NOTE",
    "SL_CHANNEL",
    "SL_THREAD_TS",
    "SL_TS",
    "SUPERSEDED_TEXT",
    "SlackApiError",
    "SlackAuthError",
    "SlackBridgeConfig",
    "SlackOps",
    "SlackWebClient",
    "SocketModeConnection",
    "VERSION_TAG_SUFFIX",
    "ack_text",
    "markdown_to_mrkdwn",
    "parse_event",
    "quote_prefix",
    "require_boot",
    "serve",
    "skipped_files_note",
)
"""What the package root must hand out: the posted wording (the end-to-end lane asserts
by equality against it) and the types the bridge is wired from."""

AUTH_ANSWER = {"ok": True, "user_id": BOT_USER, "bot_id": BOT_ID, "team_id": TEAM}


@pytest.fixture
def cfg(tmp_path: Path) -> SlackBridgeConfig:
    """A complete config whose stores live under ``tmp_path``."""
    return SlackBridgeConfig(
        bot_token=BOT_TOKEN,
        app_token=APP_TOKEN,
        version_tag=VERSION,
        core=CoreConfig(
            trigger="slack-question",
            dispatcher_url=DISPATCHER_URL,
            worker_url=WORKER_URL,
            event_dispatcher_token="disp-token",
            dispatch_worker_token="work-token",
            dedup_path=str(tmp_path / "dedup.json"),
        ),
    )


def slack_leg(
    answer: dict[str, Any] | None = None,
) -> tuple[httpx.Client, list[httpx.Request]]:
    """A Web API client answering ``auth.test`` (and anything else) with ``answer``."""
    body = AUTH_ANSWER if answer is None else answer
    return recording_client(lambda _request: httpx.Response(200, json=body))


def wired(cfg: SlackBridgeConfig, tmp_path: Path, **seams: Any) -> Wiring:
    """``build_wiring`` over a Web API leg that answers ``auth.test``."""
    seams.setdefault("slack_http", slack_leg()[0])
    seams.setdefault("inbox_dir", tmp_path / "inbox")
    return build_wiring(cfg, **seams)


class FakeIngest:
    """Stands in for :class:`~osprey.bridges.slack.receiver.Ingest`; records its life."""

    instances: list[FakeIngest] = []

    def __init__(self, connection: Any, inbox: Any, *, stop: Any, fatal: Any) -> None:
        self.connection, self.inbox, self.stop, self.fatal = connection, inbox, stop, fatal
        self.started = False
        self.joined = False
        FakeIngest.instances.append(self)

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002 - Thread.join shape
        self.joined = True


@pytest.fixture
def fake_ingest(monkeypatch: pytest.MonkeyPatch) -> type[FakeIngest]:
    FakeIngest.instances = []
    monkeypatch.setattr(main_module, "Ingest", FakeIngest)
    return FakeIngest


# --- refusing an incomplete environment --------------------------------------


def test_main_exits_naming_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment is replaced wholesale, never pruned: an ambient ``DISPATCH_*``
    in the test process would make a pruned refusal prove nothing."""
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert CANNOT_START.format(reason="") in message
    for name in REQUIRED_NAMES:
        assert name in message, f"{name} missing from the abort: {message}"


def test_main_exits_when_the_dispatch_urls_render_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", COMPLETE_ENV | {"DISPATCHER_URL": "", "WORKER_URL": ""})

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert "DISPATCHER_URL" in message
    assert "WORKER_URL" in message


def test_the_module_is_runnable_as_the_containers_command() -> None:
    """``python -m osprey.bridges.slack`` is the compose ``command``: it runs as a
    script at all, and a missing variable is a non-zero exit naming it."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}

    finished = subprocess.run(
        [sys.executable, "-m", "osprey.bridges.slack"],
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
    """Validation precedes construction, so a misconfigured deployment never calls
    ``auth.test`` with half a config."""
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(
        main_module, "build_wiring", lambda *a, **k: pytest.fail("built a wiring anyway")
    )

    with pytest.raises(SystemExit):
        main()


def test_config_from_env_reads_a_complete_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", dict(COMPLETE_ENV))

    parsed = config_from_env()

    assert parsed.bot_token == BOT_TOKEN
    assert parsed.app_token == APP_TOKEN
    assert parsed.version_tag == VERSION
    assert parsed.core.trigger == "slack-question"
    assert parsed.core.worker_url == WORKER_URL


def test_run_refuses_an_incomplete_config(
    cfg: SlackBridgeConfig, tmp_path: Path, fake_ingest: type[FakeIngest]
) -> None:
    """``run`` is reachable in-process without ``config_from_env``, so it validates
    again — before any thread starts."""
    wiring = wired(dataclasses.replace(cfg, app_token=""), tmp_path)

    with pytest.raises(ValueError, match="SLACK_APP_TOKEN"):
        run(wiring)

    assert fake_ingest.instances == []


# --- a refused credential ----------------------------------------------------


def test_build_wiring_raises_when_slack_refuses_the_token(
    cfg: SlackBridgeConfig, tmp_path: Path
) -> None:
    http, requests = slack_leg({"ok": False, "error": "invalid_auth"})

    with pytest.raises(SlackAuthError, match="invalid_auth"):
        wired(cfg, tmp_path, slack_http=http)

    assert [request.url.path for request in requests] == ["/api/auth.test"]


def test_main_maps_a_refused_token_to_the_cannot_start_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator reads the cause, not a traceback — and never the token."""
    monkeypatch.setattr(os, "environ", dict(COMPLETE_ENV))
    refusal = SlackAuthError("slack auth.test refused the credential: token_revoked")

    def refuse(*_a: Any, **_k: Any) -> Wiring:
        raise refusal

    monkeypatch.setattr(main_module, "build_wiring", refuse)

    with pytest.raises(SystemExit) as exc:
        main()

    assert str(exc.value) == CANNOT_START.format(reason=refusal)
    assert exc.value.__cause__ is refusal
    assert BOT_TOKEN not in str(exc.value)


def test_main_maps_a_credential_dying_while_running_to_the_cannot_start_exit(
    cfg: SlackBridgeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", dict(COMPLETE_ENV))
    wiring = wired(cfg, tmp_path)
    monkeypatch.setattr(main_module, "build_wiring", lambda *_a, **_k: wiring)
    monkeypatch.setattr(main_module, "install_signal_handlers", lambda _stop: None)
    dead = SlackAuthError("slack credential is dead (fatal flag set)")

    def run_dies(_wiring: Wiring) -> None:
        raise dead

    monkeypatch.setattr(main_module, "run", run_dies)

    with pytest.raises(SystemExit) as exc:
        main()

    assert str(exc.value) == CANNOT_START.format(reason=dead)
    assert exc.value.__cause__ is dead


# --- what build_wiring builds ------------------------------------------------


def test_the_bot_identity_comes_from_auth_test(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    http, requests = slack_leg()
    wiring = wired(cfg, tmp_path, slack_http=http)

    expected = BotIdentity(bot_user_id=BOT_USER, bot_id=BOT_ID, team_id=TEAM)
    assert wiring.ctx == expected
    assert wiring.ops._ctx == expected
    assert requests[0].headers["authorization"] == f"Bearer {BOT_TOKEN}"


def test_one_client_and_one_fatal_event_are_shared(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    """A second client or a second fatal flag is a dead token one half of the bridge
    notices and the other keeps using."""
    wiring = wired(cfg, tmp_path)

    assert isinstance(wiring.client, SlackWebClient)
    assert wiring.ops._client is wiring.client
    assert wiring.connection._web is wiring.client
    assert wiring.client.fatal is wiring.fatal
    assert wiring.fatal is not wiring.stop


def test_the_deps_bundle_is_built_over_this_wirings_ops(
    cfg: SlackBridgeConfig, tmp_path: Path
) -> None:
    wiring = wired(cfg, tmp_path)

    assert wiring.deps.ops is wiring.ops
    assert wiring.deps.cfg == dataclasses.replace(cfg.core, trust_env=False)


def test_the_injected_slack_client_is_what_a_post_travels_over(
    cfg: SlackBridgeConfig, tmp_path: Path
) -> None:
    http, requests = slack_leg(AUTH_ANSWER | {"ts": "1.2"})
    wiring = wired(cfg, tmp_path, slack_http=http)

    wiring.client.post_message("C0CHAN", "hello")

    assert [request.url.path for request in requests] == ["/api/auth.test", "/api/chat.postMessage"]


def test_the_injected_connect_is_the_sockets(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    def connect(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("not dialled at wiring time")

    wiring = wired(cfg, tmp_path, connect=connect)

    assert wiring.connection._connect is connect


def test_the_default_connect_is_websockets(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    from websockets.sync.client import connect as ws_connect

    wiring = wired(cfg, tmp_path)

    assert wiring.connection._connect is ws_connect


def test_the_inbox_lives_where_it_was_told(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    wiring = wired(cfg, tmp_path, inbox_dir=tmp_path / "spool")

    assert isinstance(wiring.inbox, Inbox)
    assert wiring.inbox.directory == tmp_path / "spool"
    assert (tmp_path / "spool").is_dir()


def test_the_default_inbox_sits_beside_the_dedup_store(
    cfg: SlackBridgeConfig, tmp_path: Path
) -> None:
    """Same state volume as ``DEDUP_PATH``, so a spooled event survives a restart."""
    wiring = build_wiring(cfg, slack_http=slack_leg()[0])

    assert wiring.inbox.directory == default_inbox_dir(cfg) == tmp_path / "inbox"


def test_the_wiring_is_frozen(cfg: SlackBridgeConfig, tmp_path: Path) -> None:
    wiring = wired(cfg, tmp_path)

    assert isinstance(wiring, Wiring)
    with pytest.raises(dataclasses.FrozenInstanceError):
        wiring.stop = threading.Event()  # type: ignore[misc]


# --- the proxy split (Decision 2) --------------------------------------------


@pytest.mark.parametrize(("flag", "trust"), [("true", True), ("false", False)])
def test_without_seams_the_slack_legs_follow_bridge_trust_env_and_the_engine_never_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, trust: bool
) -> None:
    """Built from the environment with no injected client: the Web API client and the
    socket honour the proxy exactly when ``BRIDGE_TRUST_ENV`` says so; the engine's
    deps and the adapter's worker byte route are proxy-free either way."""
    monkeypatch.setattr(
        os,
        "environ",
        COMPLETE_ENV | {"BRIDGE_TRUST_ENV": flag, "DEDUP_PATH": str(tmp_path / "dedup.json")},
    )
    monkeypatch.setattr(SlackWebClient, "auth_test", lambda _self: dict(AUTH_ANSWER))
    parsed = config_from_env()
    assert parsed.core.trust_env is trust

    wiring = build_wiring(parsed)

    assert wiring.client._http.trust_env is trust
    assert wiring.connection._proxy is (True if trust else None)
    assert wiring.deps.cfg.trust_env is False
    assert wiring.ops._cfg.core.trust_env is False
    assert wiring.cfg.core.trust_env is trust


# --- run: ingest before the engine, one stop event (Decision 4) -------------


def test_run_starts_ingest_before_the_engine_and_threads_one_event(
    cfg: SlackBridgeConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_ingest: type[FakeIngest],
) -> None:
    forever: dict[str, Any] = {}
    served: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:
        assert fake_ingest.instances and fake_ingest.instances[0].started, (
            "the engine booted before ingest was listening"
        )
        forever.update(core=core, ops=ops, serve=serve, **kwargs)

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)
    monkeypatch.setattr(main_module, "serve", lambda *a, **k: served.append((a, k)))

    wiring = wired(cfg, tmp_path)
    run(wiring)

    (ingest,) = fake_ingest.instances
    assert ingest.connection is wiring.connection
    assert ingest.inbox is wiring.inbox
    assert ingest.stop is wiring.stop
    assert ingest.fatal is wiring.fatal
    assert forever["core"].trust_env is False
    assert forever["ops"] is wiring.ops
    assert forever["deps"] is wiring.deps
    assert forever["stop"] is wiring.stop

    forever["serve"]("the-runtime")
    (args, kwargs) = served[0]
    assert args == (wiring.inbox, "the-runtime")
    assert kwargs["stop"] is wiring.stop
    assert kwargs["fatal"] is wiring.fatal
    assert kwargs["ingest"] is ingest


def test_run_stops_ingest_however_the_engine_ends(
    cfg: SlackBridgeConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_ingest: type[FakeIngest],
) -> None:
    """No daemon thread outlives ``run``: a raise from the engine still stops, closes
    and joins."""

    def explode(*_a: Any, **_k: Any) -> None:
        raise SlackAuthError("slack credential is dead (fatal flag set)")

    monkeypatch.setattr(main_module, "run_forever", explode)
    wiring = wired(cfg, tmp_path)

    with pytest.raises(SlackAuthError):
        run(wiring)

    assert wiring.stop.is_set()
    assert wiring.connection._closed is True
    assert fake_ingest.instances[0].joined


def test_run_lets_a_caller_replace_the_deps_bundle(
    cfg: SlackBridgeConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_ingest: type[FakeIngest],  # noqa: ARG001 - keeps the socket closed
) -> None:
    forever: dict[str, Any] = {}

    def fake_run_forever(core: CoreConfig, ops: Any, serve: Any, **kwargs: Any) -> None:  # noqa: ARG001 - stands in for run_forever
        forever.update(kwargs)

    def always_healthy() -> bool:
        return True

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)
    wiring = wired(cfg, tmp_path)
    replacement = PipelineDeps(
        cfg=wiring.deps.cfg,
        ops=wiring.ops,
        dedup=wiring.deps.dedup,
        dispatcher=wiring.deps.dispatcher,
    )

    run(wiring, deps=replacement, gate=always_healthy)

    assert forever["deps"] is replacement
    assert forever["gate"] is always_healthy


@pytest.mark.usefixtures("_restore_signal_handlers")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_both_signals_set_the_wirings_stop_event(
    cfg: SlackBridgeConfig, tmp_path: Path, sig: signal.Signals
) -> None:
    wiring = wired(cfg, tmp_path)
    install_signal_handlers(wiring.stop)

    handler = signal.getsignal(sig)
    assert callable(handler)
    handler(sig, None)

    assert wiring.stop.is_set()


# --- logging -----------------------------------------------------------------


def test_the_chatty_libraries_are_demoted(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in QUIET_LIBRARIES:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)

    configure_logging()

    assert "websockets" in QUIET_LIBRARIES
    assert all(logging.getLogger(name).level == logging.WARNING for name in QUIET_LIBRARIES)


def test_the_startup_line_names_no_credential(
    cfg: SlackBridgeConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_ingest: type[FakeIngest],  # noqa: ARG001 - keeps the socket closed
) -> None:
    """A startup line is the most-copied line in any incident, and either token is
    enough to act as the app."""
    monkeypatch.setattr(main_module, "run_forever", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        run(wired(cfg, tmp_path))

    assert "slack bridge starting" in caplog.text
    assert TEAM in caplog.text
    assert BOT_USER in caplog.text
    assert BOT_TOKEN not in caplog.text
    assert APP_TOKEN not in caplog.text
    assert "xoxb-" not in caplog.text
    assert "xapp-" not in caplog.text


# --- the package's import surface --------------------------------------------


def test_the_package_root_hands_out_every_name_the_bridge_is_wired_from() -> None:
    package = importlib.import_module("osprey.bridges.slack")

    assert set(EXPORTED_NAMES) <= set(package.__all__)
    for name in package.__all__:
        assert getattr(package, name, None) is not None, f"{name} is exported but unresolvable"


def test_the_exported_names_are_sorted() -> None:
    package = importlib.import_module("osprey.bridges.slack")

    assert package.__all__ == sorted(package.__all__)


def test_the_package_root_does_not_re_export_the_entrypoint() -> None:
    """Importing ``__main__`` from the root would run the process module twice under
    ``python -m``."""
    package = importlib.import_module("osprey.bridges.slack")

    assert "main" not in package.__all__
    assert "build_wiring" not in package.__all__


def test_the_package_root_imports_from_a_clean_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "osprey.bridges.slack" or name.startswith("osprey.bridges.slack."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    package = importlib.import_module("osprey.bridges.slack")

    for name in EXPORTED_NAMES:
        assert getattr(package, name, None) is not None, f"{name} vanished on a clean import"
