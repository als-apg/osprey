"""Hermetic fakes for the Microsoft Teams bridge: the other party, made concrete.

Every unit test of ``osprey.bridges.teams`` drives one seam at a time against a
mock or a transport stub, so each asserts OSPREY's half of a two-party contract
against OSPREY's own idea of the wire format. These fakes are the other half: a
real Service Bus queue stand-in the serve loop pulls from, a real HTTP server the
token exchange authenticates against, and a real HTTP server the Bot Connector
posts land on — so the e2e lane can boot the whole bridge (``build_wiring``, the
runtime, the dispatcher and worker) and assert what *arrived*, not what a mock
was called with.

Three fakes, one rule each
--------------------------
:class:`FakeQueueReceiver`
    Implements the :class:`~osprey.bridges.teams.receiver.QueueReceiver`
    Protocol in-process over a :class:`queue.Queue`. It records the **order** of
    ``register`` / ``complete`` / ``dead_letter`` per message, because that
    ordering is the bridge's crash-safety contract (renew before the handler,
    settle after it, exactly once) and nothing else in the lane can observe it.

:class:`FakeTokenServer`
    Answers AAD's ``/{tenant}/oauth2/v2.0/token`` route with a fixed bearer.
    The login host is *not* configurable — it is resolved from the closed
    ``TEAMS_CLOUD`` table on
    :class:`~osprey.bridges.teams.config.TeamsBridgeConfig` — so the only way to
    point the bridge at this fake is the ``token_http`` seam, and
    :meth:`FakeTokenServer.http_client` builds exactly that: an
    :class:`httpx.Client` whose transport rewrites the real AAD URL onto this
    server's loopback address, path and body untouched. That keeps the product
    code aiming at the URL it would use in production.

:class:`FakeConnectorServer`
    Answers the Bot Connector's reply route and records every posted activity.
    Its ``serviceUrl`` *is* configurable — it is whatever the inbound activity
    carried — so a test points the bridge here simply by building activities
    with ``service_url=connector.base_url``. It can be told to answer **413**
    for chosen posts, which is the one Connector status the ops layer routes on
    (``MessageSizeTooBig`` → re-split the answer), and it keeps rejected posts on
    :attr:`FakeConnectorServer.attempts` so a test can see the oversized attempt
    *and* the pieces that replaced it.

No skips, no Azure
------------------
Nothing here needs a container runtime, a credential or a network: "cannot boot"
is a failure, never a skip. And nothing here imports ``azure`` — the queue fake
stands in for the SDK receiver entirely, so the lane runs on a machine with no
Azure packages installed and proves the serve loop's own seam rather than the
SDK's.

Activity builders
-----------------
The four shapes at the bottom of this module produce activities
:func:`~osprey.bridges.teams.events.parse_event` accepts (or deliberately
ignores): a channel root post with an ``@mention``, a channel post without one, a
1:1 message, and a group chat with a mention. They exist here rather than in the
test module because the mention markup, the ``28:`` actor prefix and the
``;messageid=`` thread suffix are wire details a test should not have to restate.
"""

from __future__ import annotations

import itertools
import json
import queue
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from osprey.bridges.teams.receiver import QueueReceiver

from .gchat_fake_http import FakeHttpService, _FakeHandler

__all__ = [
    "ACCESS_TOKEN",
    "APP_ID",
    "BOT_NAME",
    "CHANNEL_ID",
    "GROUP_CHAT_ID",
    "PERSONAL_CONVERSATION_ID",
    "SENDER_ID",
    "SENDER_NAME",
    "TENANT_ID",
    "TOKEN_EXPIRES_IN",
    "FakeConnectorServer",
    "FakeQueueMessage",
    "FakeQueueReceiver",
    "FakeTokenServer",
    "PostedActivity",
    "TokenRequest",
    "activity",
    "channel_activity",
    "group_chat_activity",
    "personal_activity",
]

# --- what the fakes and the builders agree on --------------------------------

APP_ID = "11111111-2222-3333-4444-555555555555"
"""The bot's app-registration id. Mentions target ``28:{APP_ID}``."""

BOT_NAME = "Osprey"
"""Display name inside the ``<at>…</at>`` span of a mention."""

TENANT_ID = "99999999-8888-7777-6666-555555555555"
"""Directory id, carried on ``channelData.tenant.id``."""

CHANNEL_ID = "19:channel@thread.tacv2"
"""A team channel's conversation id, as Teams shapes it."""

GROUP_CHAT_ID = "19:groupchat@thread.v2"
"""A group chat's conversation id."""

PERSONAL_CONVERSATION_ID = "a:1personalchatwithalice"
"""A 1:1 chat's conversation id."""

SENDER_ID = "29:alice"
"""The human asking. Teams' ``29:`` prefix is a user actor."""

SENDER_NAME = "Alice"

ACCESS_TOKEN = "fake-bot-connector-token"
"""The bearer :class:`FakeTokenServer` hands out, and the value a test asserts
arrived on the Connector's ``Authorization`` header."""

TOKEN_EXPIRES_IN = 3600
"""Lifetime the token response states, in seconds. Comfortably past the client's
expiry margin, so one exchange serves a whole test."""

HTTP_TIMEOUT = 10.0
"""Client-side timeout for the helper clients these fakes build. Loopback."""

_TOKEN_ROUTE = re.compile(r"^/(?P<tenant>[^/]+)/oauth2/v2\.0/token$")
_ACTIVITY_ROUTE = re.compile(
    r"^/v3/conversations/(?P<conversation>[^/]+)/activities(?:/(?P<reply_to>[^/]*))?$"
)


# ---------------------------------------------------------------------------
# The fake AAD token endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenRequest:
    """One client-credentials exchange, as the login host received it."""

    tenant: str
    """The ``{tenant}`` path segment — proof the config's tenant id was used."""

    grant_type: str
    client_id: str
    client_secret: str
    scope: str
    form: dict[str, str]
    """The form body verbatim, for assertions these fields do not name."""


class _TokenHandler(_FakeHandler):
    fake: ClassVar[FakeTokenServer]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path, _ = self.split_path()
        match = _TOKEN_ROUTE.match(path)
        if match is None:
            self._aad_error(404, "invalid_request", f"no such route: POST {path}")
            return
        form = {
            k: v[-1]
            for k, v in urllib.parse.parse_qs(self.read_body().decode("utf-8", "replace")).items()
        }
        self.fake._record(match.group("tenant"), form)
        self.respond_json(
            200,
            {
                "token_type": "Bearer",
                "expires_in": self.fake.expires_in,
                "ext_expires_in": self.fake.expires_in,
                "access_token": self.fake.access_token,
            },
        )

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path, _ = self.split_path()
        self._aad_error(405, "invalid_request", f"the token route is POST-only: GET {path}")

    def _aad_error(self, status: int, code: str, description: str) -> None:
        """AAD's error envelope, not Google's — what ``TokenError`` quotes."""
        self.respond_json(status, {"error": code, "error_description": description})


class FakeTokenServer(FakeHttpService):
    """The login host: answers the client-credentials exchange with a fixed bearer.

    Point the bridge at it with :meth:`http_client`, which is the ``token_http``
    seam; there is no other route, because the real login host is resolved from a
    closed cloud table rather than from configuration.
    """

    handler_class: ClassVar[type[_FakeHandler]] = _TokenHandler

    def __init__(
        self, *, access_token: str = ACCESS_TOKEN, expires_in: int = TOKEN_EXPIRES_IN
    ) -> None:
        self.access_token = access_token
        self.expires_in = expires_in
        self._lock = threading.Lock()
        self._requests: list[TokenRequest] = []
        self._attempted: list[str] = []
        super().__init__()

    # -- recording surface --------------------------------------------------

    @property
    def requests(self) -> list[TokenRequest]:
        """Every exchange this host answered, in order."""
        with self._lock:
            return list(self._requests)

    @property
    def attempted_urls(self) -> list[str]:
        """The URLs the client *aimed* at, before the transport rewrote them.

        A test asserts the real AAD host and path here, which is what proves the
        rewrite is a redirection of the product's own URL rather than the product
        being handed a loopback address it would never use in production.
        """
        with self._lock:
            return list(self._attempted)

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._attempted.clear()

    # -- the seam -----------------------------------------------------------

    def http_client(self, *, timeout: float = HTTP_TIMEOUT) -> httpx.Client:
        """An :class:`httpx.Client` for the ``token_http`` seam, aimed here.

        Its transport rewrites only the scheme, host and port of each request,
        leaving the path (``/{tenant}/oauth2/v2.0/token``), the form body and every
        header exactly as the product built them. Close it with the client, or let
        the test's fixture do it.
        """
        return httpx.Client(transport=_RedirectTransport(self.base_url, self), timeout=timeout)

    # -- internals ----------------------------------------------------------

    def _record(self, tenant: str, form: dict[str, str]) -> None:
        with self._lock:
            self._requests.append(
                TokenRequest(
                    tenant=tenant,
                    grant_type=form.get("grant_type", ""),
                    client_id=form.get("client_id", ""),
                    client_secret=form.get("client_secret", ""),
                    scope=form.get("scope", ""),
                    form=dict(form),
                )
            )

    def _note_attempt(self, url: str) -> None:
        with self._lock:
            self._attempted.append(url)


class _RedirectTransport(httpx.BaseTransport):
    """Sends every request to one loopback origin, keeping path, body and headers.

    The alternative — an :class:`httpx.MockTransport` answering in-process — would
    never exercise a socket, a ``Content-Length`` or a form encoding, and the
    token leg is exactly where those have gone wrong before. This keeps the real
    HTTP round trip and changes only where it lands.
    """

    def __init__(self, base_url: str, recorder: FakeTokenServer) -> None:
        self._target = httpx.URL(base_url)
        self._recorder = recorder
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._recorder._note_attempt(str(request.url))
        request.url = request.url.copy_with(
            scheme=self._target.scheme, host=self._target.host, port=self._target.port
        )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


# ---------------------------------------------------------------------------
# The fake Bot Connector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostedActivity:
    """One activity the bridge posted, as the Connector received it."""

    number: int
    """Which post this was, 1-based over every post the server saw — the same
    numbering :meth:`FakeConnectorServer.reject_post` takes."""

    conversation_id: str
    """The conversation segment of the path, percent-decoding undone."""

    reply_to: str
    """The activity being replied to. Empty when the post named no reply target."""

    text: str
    attachments: list[dict[str, Any]]
    status: int
    """What this fake answered. ``200`` unless the post was rejected."""

    auth: str
    """The ``Authorization`` header verbatim, so a test can assert the bearer."""

    body: dict[str, Any]
    """The posted activity verbatim, for assertions this dataclass does not name."""

    @property
    def accepted(self) -> bool:
        return self.status < 400

    @property
    def has_attachments(self) -> bool:
        return bool(self.attachments)


class _ConnectorHandler(_FakeHandler):
    fake: ClassVar[FakeConnectorServer]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path, _ = self.split_path()
        match = _ACTIVITY_ROUTE.match(path)
        if match is None:
            self._connector_error(404, "ResourceNotFound", f"no such route: POST {path}")
            return
        posted = self.fake._record(
            conversation_id=urllib.parse.unquote(match.group("conversation")),
            reply_to=urllib.parse.unquote(match.group("reply_to") or ""),
            body=self.read_json(),
            auth=self.headers.get("Authorization", ""),
        )
        if posted.accepted:
            self.respond_json(200, {"id": f"posted-{posted.number}"})
            return
        self._connector_error(
            posted.status,
            "MessageSizeTooBig" if posted.status == 413 else "ServiceError",
            self.fake.reject_message,
        )

    def _connector_error(self, status: int, code: str, message: str) -> None:
        """The Bot Connector's error envelope: ``{"error": {"code", "message"}}``."""
        self.respond_json(status, {"error": {"code": code, "message": message}})


class FakeConnectorServer(FakeHttpService):
    """The Bot Connector: records every reply the bridge posts, and can refuse one.

    A test points the bridge here by building inbound activities with
    ``service_url=<this server's base_url>`` — the bridge posts to whatever
    ``serviceUrl`` the activity carried, which is the whole reason this needs no
    injected seam of its own.
    """

    handler_class: ClassVar[type[_FakeHandler]] = _ConnectorHandler

    reject_message = "the activity exceeded the size limit"
    """Message the refusal carries. Bounded and boring: the product quotes it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: list[PostedActivity] = []
        self._reject: dict[int, int] = {}
        self._counter = 0
        super().__init__()

    # -- recording surface (what tests assert on) ---------------------------

    @property
    def attempts(self) -> list[PostedActivity]:
        """Every post that reached this server, refused ones included, in order."""
        with self._lock:
            return list(self._attempts)

    @property
    def posted(self) -> list[PostedActivity]:
        """Only the posts this server accepted — what actually landed in Teams."""
        return [a for a in self.attempts if a.accepted]

    @property
    def posted_text(self) -> list[str]:
        return [a.text for a in self.posted]

    def wait_for_posted(self, count: int, timeout: float = 30.0) -> list[PostedActivity]:
        """Block until at least ``count`` activities have been *accepted*.

        Raises ``AssertionError`` naming what did arrive — never returns short, so
        a caller cannot mistake a timeout for a quiet conversation.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            posted = self.posted
            if len(posted) >= count:
                return posted
            time.sleep(0.05)
        posted = self.posted
        raise AssertionError(
            f"expected >= {count} accepted activit(y/ies) within {timeout:.0f}s, "
            f"got {len(posted)}: {[a.text[:80] for a in posted]}"
        )

    def reset(self) -> None:
        """Drop recorded posts and pending refusals; the port stays bound."""
        with self._lock:
            self._attempts.clear()
            self._reject.clear()
            self._counter = 0

    # -- failure injection --------------------------------------------------

    def reject_post(self, number: int, status: int = 413) -> None:
        """Answer post ``number`` (1-based, counted over *all* posts) with ``status``.

        413 is the interesting one: it is what the Connector says about an
        oversized activity, and the ops layer answers it by re-splitting the text
        and posting again. The refused attempt stays on :attr:`attempts`, so a test
        can assert both the attempt that was too big and the pieces that replaced
        it.
        """
        if number < 1:
            raise ValueError(f"post numbers are 1-based; got {number}")
        with self._lock:
            self._reject[number] = status

    # -- internals ----------------------------------------------------------

    def _record(
        self, *, conversation_id: str, reply_to: str, body: dict[str, Any], auth: str
    ) -> PostedActivity:
        with self._lock:
            self._counter += 1
            status = self._reject.pop(self._counter, 200)
            attachments = body.get("attachments") or []
            posted = PostedActivity(
                number=self._counter,
                conversation_id=conversation_id,
                reply_to=reply_to,
                text=str(body.get("text", "")),
                attachments=list(attachments) if isinstance(attachments, list) else [],
                status=status,
                auth=auth,
                body=dict(body),
            )
            self._attempts.append(posted)
            return posted


# ---------------------------------------------------------------------------
# The fake Service Bus receiver
# ---------------------------------------------------------------------------


class FakeQueueMessage:
    """A queued message: an id for the call log, and a body shaped like the SDK's.

    The SDK exposes a data body as an **iterator of byte sections**, which is the
    shape :attr:`body` takes by default, so the lane proves
    :func:`~osprey.bridges.teams.receiver.decode_body` against the least
    convenient form it meets in production. A fresh iterator is built per read:
    a redelivered message is read twice, and an exhausted iterator would look
    like an empty body rather than a redelivery.
    """

    def __init__(self, message_id: str, payload: Any, *, sections: bool = True) -> None:
        self.id = message_id
        self.activity: dict[str, Any] | None = payload if isinstance(payload, dict) else None
        if isinstance(payload, dict | list):
            self._raw: Any = json.dumps(payload).encode("utf-8")
        else:
            self._raw = payload
        self._sections = sections

    @property
    def body(self) -> Any:
        if self._sections and isinstance(self._raw, bytes) and len(self._raw) > 1:
            half = len(self._raw) // 2
            return iter([self._raw[:half], self._raw[half:]])
        return self._raw

    def __repr__(self) -> str:
        return f"<FakeQueueMessage {self.id}>"


class FakeQueueReceiver:
    """An in-process :class:`~osprey.bridges.teams.receiver.QueueReceiver`.

    A :class:`queue.Queue` of messages plus an ordered log of every settlement
    call. Thread-safe on purpose and not merely by inheritance from ``Queue``:
    the serve loop pulls on its own thread while handler threads complete, and
    although the loop takes its own lock around every call here, a fake that
    depended on that would stop being a test of the loop.

    ``receive`` **blocks**, it does not spin — an empty pull waits on the queue
    and returns ``[]``. It waits only :attr:`idle_wait` rather than the full
    ``max_wait`` the loop asks for (5 s in production): the loop holds its
    receiver lock for the whole pull, so an honest 5-second idle wait would delay
    every handler's settlement behind it and make a two-turn test take half a
    minute. The requested ``max_wait`` is recorded on :attr:`receive_calls`
    verbatim, so a test can still assert the loop asked for the documented wait.
    """

    idle_wait: float = 0.05
    """Seconds an empty pull blocks on the queue before returning ``[]``."""

    def __init__(self, *, idle_wait: float | None = None) -> None:
        if idle_wait is not None:
            self.idle_wait = idle_wait
        self._queue: queue.Queue[FakeQueueMessage] = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._calls: list[tuple[str, str]] = []
        self._dead_lettered: list[tuple[str, str]] = []
        self.receive_calls: list[tuple[int, float]] = []
        self.pulled = threading.Event()
        """Set on the first pull — how a test waits for the loop to be listening."""

    # -- seeding surface ----------------------------------------------------

    def enqueue(self, payload: Any, *, sections: bool = True) -> FakeQueueMessage:
        """Queue one activity for delivery. Mappings are JSON-encoded like the relay's.

        Args:
            payload: The activity, as a mapping (encoded here), or raw ``bytes``
                or ``str`` for the poison-body cases.
            sections: Deliver the body as byte sections, as the SDK does. Pass
                ``False`` to hand the body over whole.

        Returns:
            The queued message, so a test can name it on the call log.
        """
        message = FakeQueueMessage(f"msg-{next(self._ids)}", payload, sections=sections)
        self._queue.put(message)
        return message

    def redeliver(self, payload: Any, *, sections: bool = True) -> FakeQueueMessage:
        """Hand the same activity back, as the broker does after a lost lock.

        The message id differs from the original's — it names the delivery on the
        call log, and nothing downstream reads it. What makes this a *redelivery*
        rather than a second question is the activity: the engine's dedup claim is
        keyed on the conversation and activity ids inside the body, so the same
        payload arriving twice is exactly what a redelivery looks like to the
        bridge.
        """
        message = FakeQueueMessage(f"msg-{next(self._ids)}-redelivery", payload, sections=sections)
        self._queue.put(message)
        return message

    # -- the QueueReceiver Protocol ----------------------------------------

    def receive(self, max_messages: int, max_wait: float) -> list[Any]:
        """Pull up to ``max_messages``, blocking briefly when the queue is empty."""
        with self._lock:
            self.receive_calls.append((max_messages, max_wait))
        self.pulled.set()
        batch: list[Any] = []
        try:
            batch.append(self._queue.get(timeout=min(max_wait, self.idle_wait)))
        except queue.Empty:
            return []
        while len(batch) < max_messages:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def register(self, msg: Any) -> None:
        self._note("register", msg)

    def complete(self, msg: Any) -> None:
        self._note("complete", msg)

    def dead_letter(self, msg: Any, reason: str) -> None:
        self._note("dead_letter", msg)
        with self._lock:
            self._dead_lettered.append((_message_id(msg), reason))

    # -- recording surface (what tests assert on) ---------------------------

    @property
    def calls(self) -> list[tuple[str, str]]:
        """``(action, message id)`` for every settlement call, in order."""
        with self._lock:
            return list(self._calls)

    def calls_for(self, message: FakeQueueMessage | str) -> list[str]:
        """The actions taken on one message, in order — e.g. ``["register", "complete"]``."""
        wanted = message if isinstance(message, str) else message.id
        return [action for action, mid in self.calls if mid == wanted]

    @property
    def registered(self) -> list[str]:
        return [mid for action, mid in self.calls if action == "register"]

    @property
    def completed(self) -> list[str]:
        return [mid for action, mid in self.calls if action == "complete"]

    @property
    def dead_lettered(self) -> list[tuple[str, str]]:
        """``(message id, reason)`` for every dead-lettered message, in order."""
        with self._lock:
            return list(self._dead_lettered)

    @property
    def pending(self) -> int:
        """Messages queued and not yet pulled."""
        return self._queue.qsize()

    def wait_for_settled(self, count: int, timeout: float = 30.0) -> list[tuple[str, str]]:
        """Block until ``count`` messages have been completed or dead-lettered.

        Raises ``AssertionError`` naming the call log — never returns short.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            calls = self.calls
            settled = [c for c in calls if c[0] in ("complete", "dead_letter")]
            if len(settled) >= count:
                return settled
            time.sleep(0.02)
        raise AssertionError(
            f"expected >= {count} settled message(s) within {timeout:.0f}s; "
            f"call log was {self.calls}"
        )

    def _note(self, action: str, msg: Any) -> None:
        with self._lock:
            self._calls.append((action, _message_id(msg)))


_static: QueueReceiver = FakeQueueReceiver()
"""Type-level proof that the fake satisfies the Protocol; ``mypy`` checks it."""


def _message_id(msg: Any) -> str:
    """The id of a received message; anything without one is named by repr."""
    return str(getattr(msg, "id", repr(msg)))


# ---------------------------------------------------------------------------
# Activity builders
# ---------------------------------------------------------------------------

_activity_ids = itertools.count(1)
_activity_id_lock = threading.Lock()


def next_activity_id() -> str:
    """A fresh activity id, shaped like the epoch-milliseconds ids Teams sends."""
    with _activity_id_lock:
        return f"17000000{next(_activity_ids):05d}"


def activity(
    *,
    text: str,
    service_url: str,
    conversation_id: str,
    conversation_type: str,
    activity_id: str | None = None,
    mention: bool = False,
    app_id: str = APP_ID,
    bot_name: str = BOT_NAME,
    sender_id: str = SENDER_ID,
    sender_name: str = SENDER_NAME,
    tenant_id: str = TENANT_ID,
) -> dict[str, Any]:
    """One Bot Framework activity, as the relay enqueues it.

    Args:
        text: The question, without any mention markup — the mention is prepended
            here so the ``<at>…</at>`` span and its entity cannot drift apart.
        service_url: Where the reply will be posted. Point it at a
            :class:`FakeConnectorServer`'s ``base_url``.
        conversation_id: The conversation as Teams sends it, thread suffix and all.
        conversation_type: ``personal``, ``channel`` or ``groupChat``. Only
            ``personal`` waives the mention requirement.
        activity_id: The message id; generated when omitted.
        mention: Prepend an ``@mention`` of this bot and describe it in
            ``entities``, which is what makes a channel message a question.
        app_id: The bot's app registration id; the mention targets ``28:{app_id}``.
        bot_name: Display name inside the mention span.
        sender_id: The asker's actor id. A ``28:`` prefix here would be the bot
            itself and the parser would ignore the activity.
        sender_name: The asker's display name.
        tenant_id: Carried on ``channelData.tenant.id``.

    Returns:
        The activity mapping, ready to enqueue on a :class:`FakeQueueReceiver`.
    """
    span = f"<at>{bot_name}</at>"
    built: dict[str, Any] = {
        "type": "message",
        "id": activity_id or next_activity_id(),
        "serviceUrl": service_url,
        "channelId": "msteams",
        "from": {"id": sender_id, "name": sender_name},
        "conversation": {"id": conversation_id, "conversationType": conversation_type},
        "recipient": {"id": f"28:{app_id}", "name": bot_name},
        "text": f"{span} {text}" if mention else text,
        "channelData": {"tenant": {"id": tenant_id}},
    }
    if mention:
        built["entities"] = [
            {"type": "mention", "text": span, "mentioned": {"id": f"28:{app_id}", "name": bot_name}}
        ]
    return built


def channel_activity(
    text: str,
    *,
    service_url: str,
    root_id: str | None = None,
    mention: bool = True,
    channel_id: str = CHANNEL_ID,
    **kwargs: Any,
) -> dict[str, Any]:
    """A message in a team channel — mentioned by default, since that is the question.

    Args:
        text: The question.
        service_url: The Connector to reply to.
        root_id: Activity id of the thread's root post. ``None`` makes this the
            root post itself, which Teams gives the bare channel id; a value
            appends the ``;messageid=<root>`` suffix a reply carries.
        mention: Pass ``False`` for a message that is not addressed to the bot —
            the parser ignores it, which is the access control being asserted.
        channel_id: The channel's conversation id.
        **kwargs: Passed through to :func:`activity`.
    """
    conversation_id = channel_id if root_id is None else f"{channel_id};messageid={root_id}"
    return activity(
        text=text,
        service_url=service_url,
        conversation_id=conversation_id,
        conversation_type="channel",
        mention=mention,
        **kwargs,
    )


def personal_activity(
    text: str,
    *,
    service_url: str,
    conversation_id: str = PERSONAL_CONVERSATION_ID,
    mention: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """A 1:1 message. No mention: a chat with the bot has nobody else to address."""
    return activity(
        text=text,
        service_url=service_url,
        conversation_id=conversation_id,
        conversation_type="personal",
        mention=mention,
        **kwargs,
    )


def group_chat_activity(
    text: str,
    *,
    service_url: str,
    conversation_id: str = GROUP_CHAT_ID,
    mention: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """A group chat message — mention-gated exactly like a channel."""
    return activity(
        text=text,
        service_url=service_url,
        conversation_id=conversation_id,
        conversation_type="groupChat",
        mention=mention,
        **kwargs,
    )
