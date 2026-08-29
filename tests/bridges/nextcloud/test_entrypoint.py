"""The process entrypoint: what boot wires together, and what it refuses to boot on.

The failures guarded here are the ones a running container would not report. Every
one of them leaves a bridge that logs nothing unusual and answers nobody:

* a **shared stop event** missed at one of its three injection points — the process
  then ignores SIGTERM for as long as a backoff wait already in progress (up to
  ``BACKOFF_CAP``);
* a **second room directory** built for the ops object — only the poller-side
  ``resolve_once`` populates one, so a private one stays permanently empty, every
  ``parse_event`` raises ``RoomTypeUnresolved``, and every message is deferred forever;
* **reconcile after ingestion** rather than before it — the previous process's
  in-flight answers then compete with new questions for the worker;
* an **empty ``DISPATCHER_URL``**, which is what compose renders for an unset bare
  ``${VAR}``, booting a bridge that re-fetches the same batch forever instead of
  refusing to start.

The wiring assertions are on *identity*, never on timing: "the same event reached the
directory" is exactly the property, and a test that waited out a backoff to infer it
would be slow, flaky, and weaker. The one test that does run threads synchronises on an
:class:`threading.Event` the fake Nextcloud sets and joins every thread before it
returns.
"""

import threading

import httpx
import pytest

from osprey.bridges.core import CoreConfig, DedupStore, DispatchClient, build_deps
from osprey.bridges.nextcloud_talk import (
    NextcloudBridgeConfig,
    OffsetStore,
    RoomDirectory,
    RoomTypeUnresolved,
)
from osprey.bridges.nextcloud_talk import __main__ as entry
from osprey.bridges.nextcloud_talk.client import LAST_GIVEN_HEADER, ROOM_TYPE_ONE_TO_ONE
from osprey.bridges.nextcloud_talk.ops import ACK_TEXT, NextcloudTalkOps
from osprey.bridges.nextcloud_talk.poller import RoomPoller

ROOM = "roomA"
OTHER_ROOM = "roomB"
BOT = "osprey-bot"
TRIGGER = "nextcloud-question"

DISPATCHER = "http://dispatcher.test:10010"
WORKER = "http://worker.test:9190"

TIMEOUT = 10.0
"""Upper bound on waiting for a thread to reach a milestone. Never used as a delay —
only as the point where a hung thread fails the test instead of hanging the suite."""

NEUTRAL_ENV = (
    "DISPATCHER_URL",
    "WORKER_URL",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
    "DISPATCH_TRIGGER",
    "POLL_INTERVAL",
    "POLL_BUDGET",
    "DISPATCH_TIMEOUT_SEC",
    "BRIDGE_TRUST_ENV",
    "DRAIN_INTERVAL",
    "RETRY_MIN_AGE",
    "RETRY_GIVE_UP",
    "RETRY_LIFETIME_CAP",
    "GITLAB_URL",
    "GITLAB_PROJECT",
    "GITLAB_ISSUES_TOKEN",
    "DEDUP_PATH",
    "HISTORY_PATH",
)
"""Every name ``CoreConfig.from_env`` reads. Cleared before each env-driven test: the
entrypoint reads the real :data:`os.environ`, and one leaked value (a
``DISPATCH_TIMEOUT_SEC`` above the default poll budget, say) would fail the config for
a reason the test is not about."""

NEXTCLOUD_ENV = (
    "NEXTCLOUD_BASE_URL",
    "NEXTCLOUD_BOT_ACCOUNT",
    "NEXTCLOUD_APP_PASSWORD",
    "NEXTCLOUD_ROOMS",
    "OFFSETS_PATH",
)


def _core(tmp_path, **overrides):
    """A complete neutral config.

    The dispatcher and the worker get **different hosts** deliberately: both expose
    ``GET /dispatch/{id}`` and the fake pair below tells them apart by host, exactly as
    a deployment does.
    """
    settings = {
        "dispatcher_url": DISPATCHER,
        "worker_url": WORKER,
        "event_dispatcher_token": "dispatcher-token",
        "dispatch_worker_token": "worker-token",
        "trigger": TRIGGER,
        "dedup_path": str(tmp_path / "dedup.json"),
        "history_path": str(tmp_path / "history.json"),
    }
    settings.update(overrides)
    return CoreConfig(**settings)


def _cfg(tmp_path, **overrides):
    """A complete config whose three stores live under ``tmp_path``."""
    core = _core(tmp_path)
    settings = {
        "base_url": "https://cloud.example.org",
        "bot_account": BOT,
        "app_password": "app-pw",
        "rooms": (ROOM,),
        "offsets_path": str(tmp_path / "offsets.json"),
        "core": core,
    }
    settings.update(overrides)
    return NextcloudBridgeConfig(**settings)


def _env(monkeypatch, tmp_path, **overrides):
    """Install a complete bridge environment, then apply ``overrides``.

    An override of ``None`` *unsets* the variable (an absent key), and ``""`` leaves it
    set-but-empty — the two are different failures and the entrypoint has to abort on
    both.
    """
    for name in NEUTRAL_ENV + NEXTCLOUD_ENV:
        monkeypatch.delenv(name, raising=False)
    env = {
        "NEXTCLOUD_BASE_URL": "https://cloud.example.org",
        "NEXTCLOUD_BOT_ACCOUNT": BOT,
        "NEXTCLOUD_APP_PASSWORD": "app-pw",
        "NEXTCLOUD_ROOMS": ROOM,
        "OFFSETS_PATH": str(tmp_path / "offsets.json"),
        "DEDUP_PATH": str(tmp_path / "dedup.json"),
        "HISTORY_PATH": str(tmp_path / "history.json"),
        "DISPATCHER_URL": DISPATCHER,
        "WORKER_URL": WORKER,
        "EVENT_DISPATCHER_TOKEN": "dispatcher-token",
        "DISPATCH_WORKER_TOKEN": "worker-token",
        "DISPATCH_TRIGGER": TRIGGER,
    }
    env.update(overrides)
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _ocs(data, *, last_given=None):
    headers = {} if last_given is None else {LAST_GIVEN_HEADER: str(last_given)}
    return httpx.Response(
        200,
        json={"ocs": {"meta": {"status": "ok", "statuscode": 200}, "data": data}},
        headers=headers,
    )


def _message(message_id, text):
    """A minimal Talk chat message in a one-to-one room (so no mention is needed)."""
    return {
        "id": message_id,
        "token": ROOM,
        "actorId": "alice",
        "actorType": "users",
        "actorDisplayName": "Alice",
        "messageType": "comment",
        "message": text,
    }


class FakeNextcloud:
    """A scripted Nextcloud, recording every interaction in ONE ordered timeline.

    The timeline is what makes the boot ordering assertable: ``room:``/``poll:`` entries
    are ingestion and ``post:`` entries are replies, so "reconcile ran before the first
    poll" is a statement about their order and not about elapsed time.
    """

    def __init__(self, *, last_message_id, chat=(), room_type=ROOM_TYPE_ONE_TO_ONE):
        self.last_message_id = last_message_id
        self.room_type = room_type
        self.chat = list(chat)
        self.timeline = []
        self.reached = threading.Event()
        """Set once :attr:`posts_expected` replies have been posted."""
        self.posts_expected = None

    @property
    def posted(self):
        """Just the reply texts, in order."""
        return [item[len("post:") :] for item in self.timeline if item.startswith("post:")]

    def handler(self, request):
        if "/api/v4/room/" in request.url.path:
            self.timeline.append("room:" + request.url.path.rsplit("/", 1)[-1])
            payload = {"type": self.room_type, "displayName": "Ops"}
            if self.last_message_id is not None:
                payload["lastMessage"] = {"id": self.last_message_id}
            return _ocs(payload)

        if request.method == "POST":
            text = httpx.QueryParams(request.content.decode())["message"]
            self.timeline.append(f"post:{text}")
            if self.posts_expected is not None and len(self.posted) >= self.posts_expected:
                self.reached.set()
            return _ocs({"id": 900 + len(self.posted)})

        self.timeline.append("poll:" + request.url.params["lastKnownMessageId"])
        batch = self.chat.pop(0) if self.chat else None
        if batch is None:
            # Talk's 304: the ordinary idle outcome. Answered for every further poll, so
            # the loop under test idles instead of replaying the script.
            return httpx.Response(httpx.codes.NOT_MODIFIED)
        return _ocs(batch, last_given=max(item["id"] for item in batch))


class FakePair:
    """A scripted dispatcher + worker: one webhook, one handshake, one terminal status.

    ``answers`` maps run id to the answer text the worker reports. The dispatcher hands
    back the run id for the *next* unanswered run, so a re-attach of an existing run id
    (startup reconcile) and a fresh dispatch can both be scripted in one fake.
    """

    def __init__(self, *, run_id, answers):
        self.run_id = run_id
        self.answers = dict(answers)
        self.fired = []

    def handler(self, request):
        host = request.url.host
        if request.method == "POST":
            assert host == httpx.URL(DISPATCHER).host, f"webhook went to {host}"
            self.fired.append(request.url.path)
            return httpx.Response(202, json={"dispatch_id": "d1"})
        run_id = request.url.path.rsplit("/", 1)[-1]
        if host == httpx.URL(DISPATCHER).host:
            return httpx.Response(
                200, json={"status": "completed", "result": {"run_id": self.run_id}}
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "text_output": self.answers.get(run_id, ""),
                "artifacts": [],
            },
        )


def _seed_inflight(cfg, message_id, *, run_id, room=ROOM, nc_message_id=41):
    """Leave behind what a crash mid-poll leaves: a claimed, non-terminal entry with a
    run id. Startup reconcile must re-attach to it, and it is written through the real
    store API so the entry shape cannot drift from a live claim's."""
    dedup = DedupStore(cfg.core.dedup_path)
    dedup.claim(
        message_id,
        text="what did the previous process ask?",
        sender_id="alice",
        history_key=f"nextcloud:{room}",
        nc_room=room,
        nc_message_id=nc_message_id,
        nc_actor_id="alice",
        nc_actor_type="users",
        nc_files=[],
    )
    dedup.record(message_id, run_id=run_id, status="in_flight")


def _talk(nextcloud):
    return httpx.Client(transport=httpx.MockTransport(nextcloud.handler))


def _deps(cfg, ops, pair):
    """Engine collaborators over the fake pair. ``sleep`` is neutralised so a scripted
    non-terminal status could never cost the suite ``poll_interval`` seconds."""
    return build_deps(
        cfg.core,
        ops,
        dispatcher=DispatchClient(
            cfg.core,
            httpx.Client(transport=httpx.MockTransport(pair.handler)),
            sleep=lambda _: None,
        ),
    )


# ==========================================================================
# Boot, end to end
# ==========================================================================


def test_boot_reconciles_before_it_polls_then_dispatches_and_answers(tmp_path):
    """One in-process boot: recovery, ingestion, dispatch and reply, in that order."""
    cfg = _cfg(tmp_path)
    nextcloud = FakeNextcloud(
        last_message_id=100, chat=[[_message(101, "what is the current orbit?")]]
    )
    # The reconciled answer, the ack for the polled message, and its answer.
    nextcloud.posts_expected = 3
    pair = FakePair(
        run_id="run-live",
        answers={"run-recovered": "the recovered answer", "run-live": "the live answer"},
    )
    _seed_inflight(cfg, f"{ROOM}:41", run_id="run-recovered")

    wiring = entry.build_wiring(cfg, talk=_talk(nextcloud))
    # gate=False keeps the drain thread inert: it owns only the retry queue, which is
    # empty here, and its real gate would probe the pair's /health routes.
    thread = threading.Thread(
        target=entry.run,
        args=(wiring,),
        kwargs={"deps": _deps(cfg, wiring.ops, pair), "gate": lambda: False},
        daemon=True,
    )
    thread.start()
    try:
        assert nextcloud.reached.wait(TIMEOUT), f"never answered; timeline={nextcloud.timeline}"
    finally:
        wiring.stop.set()
        thread.join(TIMEOUT)
    assert not thread.is_alive(), "setting the stop event did not end the process"

    # ORDERING, not just occurrence: the previous process's in-flight answer is
    # delivered before the first room is even asked what type it is, so a user still
    # waiting never queues behind questions that arrived after the restart.
    assert nextcloud.timeline[0] == "post:the recovered answer"
    assert nextcloud.timeline[1:4] == [f"room:{ROOM}", "poll:100", f"post:{ACK_TEXT}"]
    assert nextcloud.posted == ["the recovered answer", ACK_TEXT, "the live answer"]
    assert pair.fired == [f"/webhook/{TRIGGER}"]

    # Both entries settled, and the offset advanced only after the batch was handled.
    dedup = DedupStore(cfg.core.dedup_path)
    assert dedup.get(f"{ROOM}:41")["status"] == "completed"
    assert dedup.get(f"{ROOM}:101")["status"] == "completed"
    assert OffsetStore(cfg.offsets_path).get(ROOM) == 101


def test_the_engine_shares_the_shutdown_event_with_the_rooms(monkeypatch, tmp_path):
    """``run_forever`` gets the same event, so ``shutdown()`` and the room stop are one
    ``set()`` — the drain cannot outlive ingestion, in either direction.

    The ingestion loop is replaced by one that returns at once, which is also what makes
    this a thread-free test: ``run_forever`` reconciles (nothing to recover), starts the
    drain, calls ``serve``, and unwinds.
    """
    cfg = _cfg(tmp_path)
    seen = []
    monkeypatch.setattr(entry, "serve_rooms", lambda wiring, runtime: seen.append(runtime))

    wiring = entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1)))
    entry.run(wiring, gate=lambda: False)

    assert len(seen) == 1
    assert seen[0].stop is wiring.stop


# ==========================================================================
# Refusing to boot
# ==========================================================================


def test_a_missing_credential_aborts_naming_the_variable(monkeypatch, tmp_path, caplog):
    _env(monkeypatch, tmp_path, NEXTCLOUD_APP_PASSWORD=None, NEXTCLOUD_ROOMS=None)

    with pytest.raises(SystemExit) as excinfo:
        entry.main()

    # Every missing name in one abort: a half-configured deployment is fixed in one pass.
    assert "NEXTCLOUD_APP_PASSWORD" in str(excinfo.value)
    assert "NEXTCLOUD_ROOMS" in str(excinfo.value)
    assert "NEXTCLOUD_APP_PASSWORD" in caplog.text


def test_empty_dispatch_urls_abort_naming_them(monkeypatch, tmp_path, caplog):
    """The compose case: an unset bare ``${VAR}`` arrives as ``""``, not as an absent
    key, so ``CoreConfig``'s own localhost defaults never fire. Unguarded, the bridge
    boots and re-fetches the same batch forever instead of aborting."""
    _env(monkeypatch, tmp_path, DISPATCHER_URL="", WORKER_URL="")

    with pytest.raises(SystemExit) as excinfo:
        entry.main()

    assert "DISPATCHER_URL" in str(excinfo.value)
    assert "WORKER_URL" in str(excinfo.value)
    assert "DISPATCHER_URL" in caplog.text


def test_the_dispatch_url_check_is_reached_by_run_too(tmp_path):
    """``run`` validates as well as ``main``: an in-process caller must not be able to
    boot a half-configured bridge by building the config itself."""
    cfg = _cfg(tmp_path, core=_core(tmp_path, dispatcher_url="", worker_url=""))

    with pytest.raises(ValueError, match="DISPATCHER_URL"):
        entry.run(entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1))))


def test_a_complete_environment_boots(monkeypatch, tmp_path):
    """The negatives above are only meaningful if the positive passes."""
    _env(monkeypatch, tmp_path)

    cfg = entry.config_from_env()

    assert cfg.rooms == (ROOM,)
    assert cfg.core.dispatcher_url == DISPATCHER


def test_a_non_https_instance_warns_and_still_boots(monkeypatch, tmp_path, caplog):
    """A plain-http Nextcloud is a legitimate internal deployment, so this is a visible
    choice and never an abort."""
    _env(monkeypatch, tmp_path, NEXTCLOUD_BASE_URL="http://cloud.internal")

    with caplog.at_level("WARNING"):
        cfg = entry.config_from_env()

    assert cfg.base_url == "http://cloud.internal"
    assert "not https" in caplog.text
    # And the wiring still builds — nothing downstream treats it as fatal either.
    assert entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1))) is not None


# ==========================================================================
# The two wirings that cannot be inferred from the types
# ==========================================================================


class RecordingDirectory(RoomDirectory):
    """A directory that keeps the ``sleep`` it was constructed with visible.

    The real one stores it privately, and the property under test is *which callable*
    was injected at construction — a poller cannot supply it later, so this is the only
    place the shutdown signal can reach a room-type-fetch backoff.
    """

    def __init__(self, client, **kwargs):
        super().__init__(client, **kwargs)
        self.sleep_arg = kwargs.get("sleep")


class RecordingOps(NextcloudTalkOps):
    """An ops object that keeps the directory it was *given* visible, so the assertion
    is about the argument rather than about whatever it fell back to."""

    def __init__(self, cfg, client=None, http=None, rooms=None):
        super().__init__(cfg, client=client, http=http, rooms=rooms)
        self.rooms_arg = rooms


class RecordingPoller(RoomPoller):
    """A poller that keeps its directory argument visible."""

    def __init__(self, room, **kwargs):
        super().__init__(room, **kwargs)
        self.rooms_arg = kwargs["rooms"]


@pytest.fixture
def recorded(monkeypatch):
    """``build_wiring``/``build_pollers`` with the three collaborators recording their
    constructor arguments."""
    monkeypatch.setattr(entry, "RoomDirectory", RecordingDirectory)
    monkeypatch.setattr(entry, "NextcloudTalkOps", RecordingOps)
    monkeypatch.setattr(entry, "RoomPoller", RecordingPoller)


def test_one_poller_per_configured_room(tmp_path):
    cfg = _cfg(tmp_path, rooms=(ROOM, OTHER_ROOM))
    wiring = entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1)))

    pollers = entry.build_pollers(wiring, runtime=object())

    assert [poller.room for poller in pollers] == [ROOM, OTHER_ROOM]


def test_one_stop_event_reaches_every_poller_and_the_directory(recorded, tmp_path):
    """Asserted by identity, not by timing. A missed injection costs up to
    ``BACKOFF_CAP`` seconds of a container ignoring SIGTERM, and a test that waited that
    out to find out would be both slow and weaker than ``is``."""
    cfg = _cfg(tmp_path, rooms=(ROOM, OTHER_ROOM))
    wiring = entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1)))

    pollers = entry.build_pollers(wiring, runtime=object())

    assert len(pollers) == 2
    assert all(poller.stop_event is wiring.stop for poller in pollers)
    # The directory's backoff sleep is the SAME event's waiter — the one shutdown path
    # the pollers' stop event cannot reach on its own.
    assert wiring.rooms.sleep_arg.__self__ is wiring.stop
    assert wiring.rooms.sleep_arg == wiring.stop.wait


def test_the_ops_object_and_the_pollers_hold_the_same_directory(recorded, tmp_path):
    cfg = _cfg(tmp_path, rooms=(ROOM, OTHER_ROOM))
    wiring = entry.build_wiring(cfg, talk=_talk(FakeNextcloud(last_message_id=1)))

    pollers = entry.build_pollers(wiring, runtime=object())

    assert wiring.ops.rooms_arg is wiring.rooms
    assert all(poller.rooms_arg is wiring.rooms for poller in pollers)


def test_a_poller_side_resolve_unblocks_the_shared_ops_object(tmp_path):
    """The failure the shared directory prevents, stated as behaviour.

    Only ``resolve_once`` (poller-side) populates a directory. With a private one the
    ops object would raise ``RoomTypeUnresolved`` for every message forever, deferring
    each one while the bridge looked healthy — so resolving through the directory the
    pollers hold must be what makes ``ops.parse_event`` answer.
    """
    cfg = _cfg(tmp_path)
    nextcloud = FakeNextcloud(last_message_id=100)
    wiring = entry.build_wiring(cfg, talk=_talk(nextcloud))
    message = _message(101, "what is the current orbit?")

    with pytest.raises(RoomTypeUnresolved):
        wiring.ops.parse_event(message)

    assert wiring.rooms.resolve_once(ROOM) == ROOM_TYPE_ONE_TO_ONE
    parsed = wiring.ops.parse_event(message)

    assert parsed is not None
    assert parsed.message_id == f"{ROOM}:101"
