"""Hermetic fakes for the Slack bridge: the other party, made concrete.

Every unit test of ``osprey.bridges.slack`` drives one seam at a time against a
mock or a transport stub, so each asserts OSPREY's half of a two-party contract
against OSPREY's own idea of the wire format. These fakes are the other half: a
real HTTP server standing in for Slack's Web API and file host, and a real
WebSocket server standing in for Socket Mode — so the e2e lane can boot the whole
bridge and assert what *arrived*, not what a mock was called with.

Two fakes and a transport
-------------------------
:class:`FakeSlackWeb`
    Answers the Web API methods the bridge calls (``POST /api/<method>``, JSON
    or form bodies, query arguments also honoured): ``auth.test``,
    ``apps.connections.open`` (which mints a loopback ``ws://`` URL on a
    :class:`FakeSocketServer`), ``chat.postMessage`` (recorded, and appended to
    the channel so ``conversations.replies`` sees the bot's own answers),
    ``conversations.replies``, ``conversations.members`` (cursor-paged),
    ``users.info`` and the upload trio (``files.getUploadURLExternal`` → ``POST``
    the bytes to the returned ``upload_url`` → ``files.completeUploadExternal``).
    It also serves seeded files on a ``files.slack.com``-style route
    (``GET /files-pri/<team>-<id>/download/<name>``); like the real host, a
    missing or wrong bearer gets a **200 HTML login page**, not a 401.

    Slack reports API errors as HTTP 200 with ``{"ok": false, "error": code}``;
    :meth:`FakeSlackWeb.fail_method` scripts exactly that (``not_in_channel``,
    ``is_archived``, ``invalid_auth``, …) for the next N calls of a method, and
    :meth:`FakeSlackWeb.rate_limit` scripts the one non-200 Slack sends: **429**
    with a ``Retry-After`` header. The app-level token is the only bearer
    ``apps.connections.open`` accepts and the bot token the only bearer
    everything else accepts, so a test sees a token mix-up as ``invalid_auth``.

:class:`FakeSocketServer`
    A loopback :mod:`websockets` server. Every connection is greeted with
    Slack's ``hello``; :meth:`FakeSocketServer.push` sends an envelope to the
    newest connection, every inbound frame is recorded, and ``{"envelope_id"}``
    frames are recorded as acks. :meth:`FakeSocketServer.send_disconnect` sends
    ``{"type": "disconnect", "reason": …}`` (``warning``, ``refresh_requested``,
    ``link_disabled``), and :meth:`FakeSocketServer.drop_connections` closes the
    sockets abnormally — the two ways a real connection ends.

:func:`http_client` / :class:`_RedirectTransport`
    The bridge's Web API host is not configurable (Decision 14): production code
    aims at ``https://slack.com/api/…`` and ``https://files.slack.com/…``. The
    transport rewrites exactly those two hosts onto :class:`FakeSlackWeb`'s
    loopback origin, path, body and headers untouched, and records every attempt
    (URL as aimed, and its ``Authorization`` header). A request to **any other
    host** is recorded and refused with :class:`httpx.ConnectError` — the lane
    never touches a network, and "a foreign host got no credentials" is an
    assertion on :attr:`FakeSlackWeb.attempts`.

No skips, no network
--------------------
Nothing here needs a container runtime, a credential or a network: "cannot boot"
is a failure, never a skip. Nothing here imports ``slack_sdk``.

Event builders
--------------
The functions at the bottom produce Events API ``event_callback`` payloads (the
object an inbox record holds and ``parse_event`` takes) and Socket Mode envelopes
around them: a DM, a channel mention (``app_mention``, which carries **no**
``channel_type``), an mpim post (which arrives as **both** ``message`` and
``app_mention``), a thread reply, a file share, a bot message and a shared-channel
mention. They live here because the ``<@U…>`` markup, the ``ts`` shape and the
channel-id prefixes are wire details a test should not have to restate.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import Server, ServerConnection, serve

from .gchat_fake_http import FakeHttpService, _FakeHandler

__all__ = [
    "APP_ID",
    "APP_TOKEN",
    "BOT_ID",
    "BOT_NAME",
    "BOT_TOKEN",
    "BOT_USER_ID",
    "CHANNEL_ID",
    "DM_CHANNEL_ID",
    "FILES_HOST",
    "FOREIGN_TEAM_ID",
    "MPIM_CHANNEL_ID",
    "SENDER_ID",
    "SENDER_NAME",
    "SLACK_API_HOST",
    "TEAM_ID",
    "ApiCall",
    "AttemptedRequest",
    "CompletedUpload",
    "FakeSlackWeb",
    "FakeSocketServer",
    "PostedMessage",
    "bot_message",
    "channel_mention",
    "dm_message",
    "envelope",
    "event_callback",
    "file_share",
    "http_client",
    "mpim_message",
    "next_event_id",
    "next_ts",
    "shared_channel_mention",
    "thread_reply",
]

# --- what the fakes and the builders agree on --------------------------------

TEAM_ID = "T0FAKETEAM"
"""The installing workspace."""

FOREIGN_TEAM_ID = "T0FOREIGN"
"""Another workspace, as a shared (Slack Connect) channel's members carry it."""

APP_ID = "A0FAKEAPP"
BOT_USER_ID = "U0FAKEBOT"
"""The bot's user id: mentions target ``<@U0FAKEBOT>``."""

BOT_ID = "B0FAKEBOT"
"""The bot's ``bot_id``, carried on the bot's own posts."""

BOT_NAME = "osprey"

BOT_TOKEN = "xoxb-fake-bot-token"
"""The only bearer every Web API method but ``apps.connections.open`` accepts."""

APP_TOKEN = "xapp-fake-app-token"
"""The only bearer ``apps.connections.open`` accepts."""

CHANNEL_ID = "C0FAKECHAN"
"""A public channel. Channel ids start with ``C``; private ones with ``G``."""

DM_CHANNEL_ID = "D0FAKEDM"
"""A 1:1 DM with the bot. DM ids start with ``D``."""

MPIM_CHANNEL_ID = "G0FAKEMPIM"
"""A group DM. Its id alone does not say mpim; ``channel_type`` does."""

SENDER_ID = "U0ALICE"
SENDER_NAME = "Alice"

SLACK_API_HOST = "slack.com"
FILES_HOST = "files.slack.com"
_REDIRECTED_HOSTS = frozenset({SLACK_API_HOST, FILES_HOST})

HTTP_TIMEOUT = 10.0
"""Client-side timeout for :func:`http_client`. Loopback."""

LOGIN_PAGE = b"<!DOCTYPE html><html><head><title>Slack</title></head><body>Sign in</body></html>"
"""What the file host answers an unauthenticated download with — status 200."""

_API_ROUTE = re.compile(r"^/api/(?P<method>[A-Za-z0-9_.]+)$")
_FILE_ROUTE = re.compile(
    r"^/files-pri/(?P<team>[^-/]+)-(?P<file>[^/]+)/(?:download/)?(?P<name>.+)$"
)
_UPLOAD_ROUTE = re.compile(r"^/upload/v1/(?P<ticket>[^/]+)$")

_APP_TOKEN_METHODS = frozenset({"apps.connections.open"})

_counter_lock = threading.Lock()
_ts_counter = itertools.count(1)
_event_counter = itertools.count(1)
_envelope_counter = itertools.count(1)


def next_ts() -> str:
    """A fresh, strictly increasing Slack ``ts`` (``"<seconds>.<micros>"``)."""
    with _counter_lock:
        n = next(_ts_counter)
    return f"{1700000000 + n // 1000}.{n % 1000:03d}100"


def next_event_id() -> str:
    """A fresh Events API ``event_id``."""
    with _counter_lock:
        n = next(_event_counter)
    return f"Ev{n:010d}"


def _next_envelope_id() -> str:
    with _counter_lock:
        n = next(_envelope_counter)
    return f"00000000-0000-4000-8000-{n:012d}"


# ---------------------------------------------------------------------------
# Recorded shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiCall:
    """One Web API call, as :class:`FakeSlackWeb` received it."""

    method: str
    """The Slack method name (``chat.postMessage``)."""

    args: dict[str, Any]
    """Query, form and JSON arguments merged (body wins)."""

    token: str
    """The bearer it carried, ``""`` when none."""

    status: int
    """The HTTP status it was answered with."""

    ok: bool
    """The ``ok`` field it was answered with."""

    error: str = ""


@dataclass(frozen=True)
class PostedMessage:
    """One accepted ``chat.postMessage``."""

    channel: str
    ts: str
    """The ``ts`` the fake minted for it."""

    text: str
    thread_ts: str
    args: dict[str, Any]
    """Every argument verbatim (``blocks``, ``mrkdwn``, ``unfurl_links``, …)."""


@dataclass(frozen=True)
class CompletedUpload:
    """One file made visible by ``files.completeUploadExternal``."""

    file_id: str
    filename: str
    title: str
    content: bytes
    """What was POSTed to the ``upload_url``."""

    channel_id: str
    thread_ts: str
    initial_comment: str
    upload_had_authorization: bool
    """Whether the byte POST carried an ``Authorization`` header (it should not)."""


@dataclass(frozen=True)
class AttemptedRequest:
    """One request the redirect transport saw, before any rewrite."""

    method: str
    url: str
    """The URL the product aimed at."""

    authorization: str | None
    """Its ``Authorization`` header, ``None`` when absent."""

    redirected: bool
    """False for a foreign host, which was refused without a network call."""


@dataclass
class _Upload:
    file_id: str
    filename: str
    length: int
    content: bytes | None = None
    had_authorization: bool = False


@dataclass
class _Script:
    error: str
    remaining: int
    status: int = 200
    retry_after: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The fake Web API + file host
# ---------------------------------------------------------------------------


class _SlackHandler(_FakeHandler):
    fake: ClassVar[FakeSlackWeb]

    def do_GET(self) -> None:  # http.server API
        path, query = self.split_path()
        match = _FILE_ROUTE.match(path)
        if match is not None:
            self.fake._serve_file(self, match.group("file"))
            return
        match = _API_ROUTE.match(path)
        if match is not None:
            self.fake._api(self, match.group("method"), dict(query))
            return
        self.respond(404, b"not found", content_type="text/plain")

    def do_POST(self) -> None:  # http.server API
        path, query = self.split_path()
        match = _UPLOAD_ROUTE.match(path)
        if match is not None:
            self.fake._receive_upload(self, match.group("ticket"))
            return
        match = _API_ROUTE.match(path)
        if match is None:
            self.respond(404, b"not found", content_type="text/plain")
            return
        args: dict[str, Any] = dict(query)
        args.update(self._read_args())
        self.fake._api(self, match.group("method"), args)

    def _read_args(self) -> dict[str, Any]:
        raw = self.read_body()
        if not raw:
            return {}
        ctype = (self.headers.get("Content-Type") or "").lower()
        if "application/json" in ctype:
            try:
                parsed = json.loads(raw)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {k: v[-1] for k, v in urllib.parse.parse_qs(raw.decode("utf-8", "replace")).items()}

    def bearer(self) -> str:
        header = self.headers.get("Authorization") or ""
        return header[len("Bearer ") :] if header.startswith("Bearer ") else ""

    def slack_json(self, payload: dict[str, Any], *, status: int = 200, headers=None) -> None:
        self.respond(status, json.dumps(payload).encode("utf-8"), headers=headers)


class FakeSlackWeb(FakeHttpService):
    """``slack.com/api`` and ``files.slack.com`` on one loopback port.

    Reach it from product code with :meth:`http_client` (the redirect seam), or
    directly at :attr:`base_url` from a test. ``socket`` is the server
    ``apps.connections.open`` hands out URLs for; without one that method answers
    ``{"ok": false, "error": "no_socket_configured"}``.
    """

    handler_class: ClassVar[type[_FakeHandler]] = _SlackHandler

    def __init__(
        self,
        *,
        socket: FakeSocketServer | None = None,
        bot_token: str = BOT_TOKEN,
        app_token: str = APP_TOKEN,
        team_id: str = TEAM_ID,
    ) -> None:
        self.socket = socket
        self.bot_token = bot_token
        self.app_token = app_token
        self.team_id = team_id
        self._lock = threading.Lock()
        self._calls: list[ApiCall] = []
        self._posted: list[PostedMessage] = []
        self._attempts: list[AttemptedRequest] = []
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._members: dict[str, list[str]] = {}
        self._users: dict[str, dict[str, Any]] = {}
        self._files: dict[str, tuple[bytes, str]] = {}
        self._downloads: list[tuple[str, bool]] = []
        self._uploads: dict[str, _Upload] = {}
        self._completed: list[CompletedUpload] = []
        self._scripts: dict[str, list[_Script]] = {}
        self._ids = itertools.count(1)
        super().__init__()
        self.add_user(SENDER_ID, name="alice", real_name=SENDER_NAME, display_name=SENDER_NAME)
        self.add_user(BOT_USER_ID, name=BOT_NAME, real_name=BOT_NAME, is_bot=True)

    # -- seeding --------------------------------------------------------------

    def add_user(
        self,
        user_id: str,
        *,
        name: str = "",
        real_name: str = "",
        display_name: str = "",
        email: str = "",
        is_bot: bool = False,
        team_id: str | None = None,
    ) -> dict[str, Any]:
        """Make ``users.info`` know a user; returns the ``user`` object served."""
        user = {
            "id": user_id,
            "team_id": team_id or self.team_id,
            "name": name or user_id.lower(),
            "real_name": real_name,
            "is_bot": is_bot,
            "deleted": False,
            "profile": {
                "real_name": real_name,
                "display_name": display_name,
                **({"email": email} if email else {}),
            },
        }
        with self._lock:
            self._users[user_id] = user
        return dict(user)

    def set_members(self, channel: str, members: list[str]) -> None:
        """Seed ``conversations.members`` for ``channel`` (served cursor-paged)."""
        with self._lock:
            self._members[channel] = list(members)

    def add_message(self, channel: str, message: dict[str, Any]) -> None:
        """Seed one message (an event-shaped dict with at least ``ts``) in ``channel``.

        ``conversations.replies`` serves a thread as every message whose ``ts``
        or ``thread_ts`` equals the requested ``ts``, parent first.
        """
        with self._lock:
            self._messages.setdefault(channel, []).append(dict(message))

    def add_file(
        self, name: str, content: bytes, *, mimetype: str = "text/plain", file_id: str = ""
    ) -> dict[str, Any]:
        """Host a file; returns the Slack ``file`` object for an event's ``files``.

        Its ``url_private`` / ``url_private_download`` aim at
        ``https://files.slack.com/files-pri/…``, which :func:`http_client`
        redirects here.
        """
        fid = file_id or f"F{next(self._ids):09d}"
        with self._lock:
            self._files[fid] = (content, mimetype)
        quoted = urllib.parse.quote(name)
        url = f"https://{FILES_HOST}/files-pri/{self.team_id}-{fid}/{quoted}"
        return {
            "id": fid,
            "name": name,
            "title": name,
            "mimetype": mimetype,
            "filetype": name.rsplit(".", 1)[-1] if "." in name else "text",
            "size": len(content),
            "url_private": url,
            "url_private_download": f"https://{FILES_HOST}/files-pri/{self.team_id}-{fid}/download/{quoted}",
        }

    # -- scripted failures ----------------------------------------------------

    def fail_method(self, method: str, error: str, *, times: int = 1, **extra: Any) -> None:
        """The next ``times`` calls of ``method`` answer ``200 {"ok": false, "error"}``.

        ``extra`` is merged into the error body (e.g. ``needed="files:write"``).
        """
        with self._lock:
            self._scripts.setdefault(method, []).append(_Script(error, times, extra=dict(extra)))

    def rate_limit(self, method: str, *, retry_after: int = 1, times: int = 1) -> None:
        """The next ``times`` calls of ``method`` answer **429** with ``Retry-After``."""
        with self._lock:
            self._scripts.setdefault(method, []).append(
                _Script("ratelimited", times, status=429, retry_after=retry_after)
            )

    # -- recording surface ----------------------------------------------------

    @property
    def calls(self) -> list[ApiCall]:
        """Every Web API call answered, in order, including scripted failures."""
        with self._lock:
            return list(self._calls)

    def calls_to(self, method: str) -> list[ApiCall]:
        return [c for c in self.calls if c.method == method]

    @property
    def posted(self) -> list[PostedMessage]:
        """Every accepted ``chat.postMessage``, in order."""
        with self._lock:
            return list(self._posted)

    @property
    def uploads(self) -> list[CompletedUpload]:
        """Every file ``files.completeUploadExternal`` made visible, in order."""
        with self._lock:
            return list(self._completed)

    @property
    def downloads(self) -> list[tuple[str, bool]]:
        """``(file_id, authorized)`` per file-route GET, in order."""
        with self._lock:
            return list(self._downloads)

    @property
    def attempts(self) -> list[AttemptedRequest]:
        """Every request :func:`http_client` saw, as the product aimed it."""
        with self._lock:
            return list(self._attempts)

    def wait_for_posted(self, count: int, timeout: float = 30.0) -> list[PostedMessage]:
        """Block until ``count`` messages were posted; AssertionError names what came."""
        deadline = time.monotonic() + timeout
        while True:
            posted = self.posted
            if len(posted) >= count:
                return posted
            if time.monotonic() >= deadline:
                texts = [p.text[:80] for p in posted]
                raise AssertionError(
                    f"expected {count} posted message(s) within {timeout}s, got "
                    f"{len(posted)}: {texts}"
                )
            time.sleep(0.05)

    def reset(self) -> None:
        """Forget recorded traffic and scripts; seeded users, members and files stay."""
        with self._lock:
            self._calls.clear()
            self._posted.clear()
            self._attempts.clear()
            self._downloads.clear()
            self._completed.clear()
            self._scripts.clear()

    # -- the seam -------------------------------------------------------------

    def http_client(self, *, timeout: float = HTTP_TIMEOUT) -> httpx.Client:
        """An :class:`httpx.Client` whose Slack hosts land here (see module docstring)."""
        return http_client(self, timeout=timeout)

    # -- internals: Web API ---------------------------------------------------

    def _note_attempt(self, attempt: AttemptedRequest) -> None:
        with self._lock:
            self._attempts.append(attempt)

    def _take_script(self, method: str) -> _Script | None:
        with self._lock:
            scripts = self._scripts.get(method) or []
            if not scripts:
                return None
            script = scripts[0]
            script.remaining -= 1
            if script.remaining <= 0:
                scripts.pop(0)
            return script

    def _api(self, handler: _SlackHandler, method: str, args: dict[str, Any]) -> None:
        token = handler.bearer()
        status, body, headers = self._answer(method, args, token)
        with self._lock:
            self._calls.append(
                ApiCall(
                    method=method,
                    args=dict(args),
                    token=token,
                    status=status,
                    ok=bool(body.get("ok")),
                    error=str(body.get("error", "")),
                )
            )
        handler.slack_json(body, status=status, headers=headers)

    def _answer(
        self, method: str, args: dict[str, Any], token: str
    ) -> tuple[int, dict[str, Any], dict[str, str] | None]:
        expected = self.app_token if method in _APP_TOKEN_METHODS else self.bot_token
        if not token:
            return 200, {"ok": False, "error": "not_authed"}, None
        if token != expected:
            return 200, {"ok": False, "error": "invalid_auth"}, None
        script = self._take_script(method)
        if script is not None:
            headers = None
            if script.retry_after is not None:
                headers = {"Retry-After": str(script.retry_after)}
            return script.status, {"ok": False, "error": script.error, **script.extra}, headers
        route = getattr(self, "_m_" + method.replace(".", "_"), None)
        if route is None:
            return 200, {"ok": False, "error": "unknown_method"}, None
        return 200, route(args), None

    def _m_auth_test(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "url": "https://fake-workspace.slack.com/",
            "team": "Fake Workspace",
            "user": BOT_NAME,
            "team_id": self.team_id,
            "user_id": BOT_USER_ID,
            "bot_id": BOT_ID,
            "is_enterprise_install": False,
        }

    def _m_apps_connections_open(self, _args: dict[str, Any]) -> dict[str, Any]:
        if self.socket is None:
            return {"ok": False, "error": "no_socket_configured"}
        return {"ok": True, "url": self.socket.connect_url()}

    def _m_chat_postMessage(self, args: dict[str, Any]) -> dict[str, Any]:
        channel = str(args.get("channel", ""))
        if not channel:
            return {"ok": False, "error": "channel_not_found"}
        text = str(args.get("text", "") or args.get("markdown_text", ""))
        if not text and not args.get("blocks"):
            return {"ok": False, "error": "no_text"}
        ts = next_ts()
        thread_ts = str(args.get("thread_ts", "") or "")
        message = {
            "type": "message",
            "user": BOT_USER_ID,
            "bot_id": BOT_ID,
            "app_id": APP_ID,
            "text": text,
            "ts": ts,
            **({"thread_ts": thread_ts} if thread_ts else {}),
        }
        with self._lock:
            self._posted.append(
                PostedMessage(
                    channel=channel, ts=ts, text=text, thread_ts=thread_ts, args=dict(args)
                )
            )
            self._messages.setdefault(channel, []).append(message)
        return {"ok": True, "channel": channel, "ts": ts, "message": message}

    def _m_conversations_replies(self, args: dict[str, Any]) -> dict[str, Any]:
        channel = str(args.get("channel", ""))
        ts = str(args.get("ts", ""))
        with self._lock:
            if channel not in self._messages and channel not in self._members:
                return {"ok": False, "error": "channel_not_found"}
            thread = [
                dict(m)
                for m in self._messages.get(channel, [])
                if m.get("ts") == ts or m.get("thread_ts") == ts
            ]
        if not any(m.get("ts") == ts for m in thread):
            return {"ok": False, "error": "thread_not_found"}
        thread.sort(key=lambda m: (m.get("ts") != ts, float(m.get("ts", "0"))))
        return {"ok": True, "messages": thread, "has_more": False}

    def _m_conversations_members(self, args: dict[str, Any]) -> dict[str, Any]:
        channel = str(args.get("channel", ""))
        with self._lock:
            members = self._members.get(channel)
        if members is None:
            return {"ok": False, "error": "channel_not_found"}
        limit = int(args.get("limit", 100) or 100)
        start = int(args.get("cursor") or 0)
        page = members[start : start + limit]
        nxt = start + limit
        cursor = str(nxt) if nxt < len(members) else ""
        return {"ok": True, "members": page, "response_metadata": {"next_cursor": cursor}}

    def _m_users_info(self, args: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            user = self._users.get(str(args.get("user", "")))
        if user is None:
            return {"ok": False, "error": "user_not_found"}
        return {"ok": True, "user": json.loads(json.dumps(user))}

    def _m_files_getUploadURLExternal(self, args: dict[str, Any]) -> dict[str, Any]:
        filename = str(args.get("filename", ""))
        try:
            length = int(args.get("length", ""))
        except ValueError:
            return {"ok": False, "error": "invalid_arguments"}
        if not filename or length <= 0:
            return {"ok": False, "error": "invalid_arguments"}
        fid = f"F{next(self._ids):09d}"
        ticket = f"ticket{fid}"
        with self._lock:
            self._uploads[ticket] = _Upload(file_id=fid, filename=filename, length=length)
        return {
            "ok": True,
            "file_id": fid,
            "upload_url": f"https://{FILES_HOST}/upload/v1/{ticket}",
        }

    def _m_files_completeUploadExternal(self, args: dict[str, Any]) -> dict[str, Any]:
        raw = args.get("files", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return {"ok": False, "error": "invalid_arguments"}
        if not isinstance(raw, list) or not raw:
            return {"ok": False, "error": "invalid_arguments"}
        with self._lock:
            by_id = {u.file_id: u for u in self._uploads.values()}
            done: list[CompletedUpload] = []
            for entry in raw:
                upload = by_id.get(str(entry.get("id", "")))
                if upload is None or upload.content is None:
                    return {"ok": False, "error": "file_not_found"}
                done.append(
                    CompletedUpload(
                        file_id=upload.file_id,
                        filename=upload.filename,
                        title=str(entry.get("title", "") or upload.filename),
                        content=upload.content,
                        channel_id=str(args.get("channel_id", "") or ""),
                        thread_ts=str(args.get("thread_ts", "") or ""),
                        initial_comment=str(args.get("initial_comment", "") or ""),
                        upload_had_authorization=upload.had_authorization,
                    )
                )
            self._completed.extend(done)
        return {"ok": True, "files": [{"id": d.file_id, "title": d.title} for d in done]}

    # -- internals: file host -------------------------------------------------

    def _receive_upload(self, handler: _SlackHandler, ticket: str) -> None:
        body = handler.read_body()
        with self._lock:
            upload = self._uploads.get(ticket)
            if upload is not None:
                upload.content = body
                upload.had_authorization = bool(handler.headers.get("Authorization"))
        if upload is None:
            handler.respond(404, b"unknown upload", content_type="text/plain")
            return
        handler.respond(200, f"OK - {len(body)}".encode(), content_type="text/plain")

    def _serve_file(self, handler: _SlackHandler, file_id: str) -> None:
        authorized = handler.bearer() == self.bot_token
        with self._lock:
            self._downloads.append((file_id, authorized))
            hosted = self._files.get(file_id)
        if not authorized:
            handler.respond(200, LOGIN_PAGE, content_type="text/html; charset=utf-8")
            return
        if hosted is None:
            handler.respond(404, b"file not found", content_type="text/plain")
            return
        content, mimetype = hosted
        handler.respond(200, content, content_type=mimetype)


# ---------------------------------------------------------------------------
# The redirect transport (Decision 14)
# ---------------------------------------------------------------------------


class _RedirectTransport(httpx.BaseTransport):
    """Lands ``slack.com`` / ``files.slack.com`` on the fake; refuses every other host.

    A copy of the Teams fixture's transport, not an import of it: the real HTTP
    round trip (socket, ``Content-Length``, form encoding) is kept and only the
    origin changes. Foreign hosts are recorded and refused with
    :class:`httpx.ConnectError`, so the lane never reaches a network and a test
    can still see what (and with which credentials) was attempted.
    """

    def __init__(self, fake: FakeSlackWeb) -> None:
        self._fake = fake
        self._target = httpx.URL(fake.base_url)
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        redirected = request.url.host in _REDIRECTED_HOSTS
        self._fake._note_attempt(
            AttemptedRequest(
                method=request.method,
                url=str(request.url),
                authorization=request.headers.get("Authorization"),
                redirected=redirected,
            )
        )
        if not redirected:
            raise httpx.ConnectError(
                f"slack fake: refusing foreign host {request.url.host!r}", request=request
            )
        request.url = request.url.copy_with(
            scheme=self._target.scheme, host=self._target.host, port=self._target.port
        )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def http_client(fake: FakeSlackWeb, *, timeout: float = HTTP_TIMEOUT) -> httpx.Client:
    """An :class:`httpx.Client` for the bridge's Web API seam, aimed at ``fake``."""
    return httpx.Client(transport=_RedirectTransport(fake), timeout=timeout)


# ---------------------------------------------------------------------------
# The fake Socket Mode server
# ---------------------------------------------------------------------------


class FakeSocketServer:
    """A loopback Socket Mode endpoint on :mod:`websockets`' threaded server.

    Every connection gets ``hello`` first. Frames the client sends are recorded
    in :attr:`received`; those shaped ``{"envelope_id": …}`` also land in
    :attr:`acks`. Use as a context manager, or call :meth:`close`.
    """

    def __init__(self, *, app_id: str = APP_ID) -> None:
        self.app_id = app_id
        self._lock = threading.Lock()
        self._connections: list[ServerConnection] = []
        self._paths: list[str] = []
        self._received: list[dict[str, Any]] = []
        self._acks: list[str] = []
        self._tickets = itertools.count(1)
        self._server: Server = serve(self._handle, "127.0.0.1", 0, compression=None)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="FakeSocketServer-serve", daemon=True
        )
        self._thread.start()

    # -- addressing -----------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._server.socket.getsockname()[1])

    def connect_url(self) -> str:
        """A fresh ``ws://`` URL, as ``apps.connections.open`` would mint it."""
        with self._lock:
            ticket = next(self._tickets)
        return f"ws://127.0.0.1:{self.port}/link/?ticket=fake-{ticket}&app_id={self.app_id}"

    # -- driving --------------------------------------------------------------

    def push(self, message: dict[str, Any], *, connection: int = -1) -> None:
        """Send one frame (normally an :func:`envelope`) to a connection (default newest)."""
        self._open(connection).send(json.dumps(message))

    def send_disconnect(self, reason: str = "refresh_requested", *, connection: int = -1) -> None:
        """Slack's ``disconnect`` notice: ``warning``, ``refresh_requested``, ``link_disabled``."""
        self.push(
            {
                "type": "disconnect",
                "reason": reason,
                "debug_info": {"host": "fake-socket"},
            },
            connection=connection,
        )

    def drop_connections(self) -> None:
        """Close every open connection abnormally (1011), as a dying link would."""
        with self._lock:
            conns = list(self._connections)
        for conn in conns:
            # A connection already closing raises; that is the state we want.
            with contextlib.suppress(Exception):
                conn.close(code=1011, reason="fake socket dropped")

    # -- recording surface ----------------------------------------------------

    @property
    def connection_count(self) -> int:
        """Connections ever accepted (open or closed)."""
        with self._lock:
            return len(self._connections)

    @property
    def open_connections(self) -> int:
        with self._lock:
            return sum(1 for c in self._connections if c.state.name == "OPEN")

    @property
    def paths(self) -> list[str]:
        """The request path (with ticket) of each connection, in order."""
        with self._lock:
            return list(self._paths)

    @property
    def received(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._received)

    @property
    def acks(self) -> list[str]:
        """``envelope_id`` of every ack, in arrival order."""
        with self._lock:
            return list(self._acks)

    def wait_for_connections(self, count: int, timeout: float = 30.0) -> None:
        self._wait(lambda: self.connection_count >= count, timeout, f"{count} connection(s)")

    def wait_for_acks(self, count: int, timeout: float = 30.0) -> list[str]:
        self._wait(lambda: len(self.acks) >= count, timeout, f"{count} ack(s)")
        return self.acks

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeSocketServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ------------------------------------------------------------

    def _wait(self, done, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not done():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"expected {what} within {timeout}s; connections={self.connection_count} "
                    f"acks={self.acks}"
                )
            time.sleep(0.02)

    def _open(self, index: int) -> ServerConnection:
        deadline = time.monotonic() + 5.0
        while True:
            with self._lock:
                if self._connections:
                    return self._connections[index]
            if time.monotonic() >= deadline:
                raise AssertionError("FakeSocketServer: no client has connected")
            time.sleep(0.02)

    def _handle(self, conn: ServerConnection) -> None:
        conn.send(
            json.dumps(
                {
                    "type": "hello",
                    "num_connections": self.open_connections + 1,
                    "connection_info": {"app_id": self.app_id},
                    "debug_info": {"host": "fake-socket", "approximate_connection_time": 3600},
                }
            )
        )
        with self._lock:
            self._connections.append(conn)
            self._paths.append(conn.request.path if conn.request is not None else "")
        try:
            for raw in conn:
                try:
                    frame = json.loads(raw)
                except ValueError:
                    frame = {"_raw": raw}
                if not isinstance(frame, dict):
                    frame = {"_raw": frame}
                with self._lock:
                    self._received.append(frame)
                    if "envelope_id" in frame:
                        self._acks.append(str(frame["envelope_id"]))
        except ConnectionClosed:
            pass


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def _profile(name: str) -> dict[str, Any]:
    return {"display_name": name, "real_name": name, "name": name.lower(), "team": TEAM_ID}


def _base_message(
    *,
    type_: str,
    text: str,
    user: str,
    channel: str,
    ts: str | None,
    thread_ts: str | None,
    user_name: str,
    team: str,
    files: list[dict[str, Any]] | None,
    channel_type: str | None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": type_,
        "user": user,
        "text": text,
        "ts": ts or next_ts(),
        "channel": channel,
        "team": team,
        "user_team": team,
        "source_team": team,
        "event_ts": "",
        "user_profile": {**_profile(user_name), "team": team},
    }
    event["event_ts"] = event["ts"]
    if channel_type is not None:
        event["channel_type"] = channel_type
    if thread_ts:
        event["thread_ts"] = thread_ts
    if files:
        event["files"] = list(files)
    return event


def event_callback(
    event: dict[str, Any],
    *,
    team_id: str = TEAM_ID,
    is_ext_shared_channel: bool = False,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Wrap an inner event in the Events API ``event_callback`` payload."""
    return {
        "token": "deprecated-verification-token",
        "team_id": team_id,
        "context_team_id": team_id,
        "context_enterprise_id": None,
        "api_app_id": APP_ID,
        "event": event,
        "type": "event_callback",
        "event_id": event_id or next_event_id(),
        "event_time": int(float(event.get("ts", "0") or 0)),
        "authorizations": [
            {
                "enterprise_id": None,
                "team_id": team_id,
                "user_id": BOT_USER_ID,
                "is_bot": True,
                "is_enterprise_install": False,
            }
        ],
        "is_ext_shared_channel": is_ext_shared_channel,
    }


def envelope(
    payload: dict[str, Any],
    *,
    envelope_id: str | None = None,
    retry_attempt: int | None = None,
    retry_reason: str = "",
) -> dict[str, Any]:
    """Wrap an ``event_callback`` payload in a Socket Mode ``events_api`` envelope."""
    env: dict[str, Any] = {
        "envelope_id": envelope_id or _next_envelope_id(),
        "type": "events_api",
        "payload": payload,
        "accepts_response_payload": False,
    }
    if retry_attempt is not None:
        env["retry_attempt"] = retry_attempt
        env["retry_reason"] = retry_reason or "timeout"
    return env


def dm_message(
    text: str = "what is the beam current?",
    *,
    user: str = SENDER_ID,
    user_name: str = SENDER_NAME,
    channel: str = DM_CHANNEL_ID,
    ts: str | None = None,
    thread_ts: str | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A 1:1 DM to the bot (``message``, ``channel_type: im``), no mention needed."""
    return event_callback(
        _base_message(
            type_="message",
            text=text,
            user=user,
            channel=channel,
            ts=ts,
            thread_ts=thread_ts,
            user_name=user_name,
            team=TEAM_ID,
            files=files,
            channel_type="im",
        )
    )


def channel_mention(
    question: str = "what is the beam current?",
    *,
    user: str = SENDER_ID,
    user_name: str = SENDER_NAME,
    channel: str = CHANNEL_ID,
    ts: str | None = None,
    thread_ts: str | None = None,
    files: list[dict[str, Any]] | None = None,
    mention: bool = True,
) -> dict[str, Any]:
    """A channel question as Slack delivers it: ``app_mention`` with **no** ``channel_type``.

    ``question=""`` gives a bare mention. ``mention=False`` drops the ``<@bot>``
    token (a malformed ``app_mention``, for negative tests).
    """
    text = f"<@{BOT_USER_ID}> {question}".strip() if mention else question
    return event_callback(
        _base_message(
            type_="app_mention",
            text=text,
            user=user,
            channel=channel,
            ts=ts,
            thread_ts=thread_ts,
            user_name=user_name,
            team=TEAM_ID,
            files=files,
            channel_type=None,
        )
    )


def mpim_message(
    question: str = "what is the beam current?",
    *,
    user: str = SENDER_ID,
    user_name: str = SENDER_NAME,
    channel: str = MPIM_CHANNEL_ID,
    ts: str | None = None,
    thread_ts: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One mpim post as its **two** deliveries: ``(message, app_mention)``.

    Same ``channel`` and ``ts``, different ``event_id`` — the bridge's dedup on
    ``channel:ts`` must turn the second into a duplicate.
    """
    ts = ts or next_ts()
    text = f"<@{BOT_USER_ID}> {question}".strip()
    common: dict[str, Any] = {
        "text": text,
        "user": user,
        "channel": channel,
        "ts": ts,
        "thread_ts": thread_ts,
        "user_name": user_name,
        "team": TEAM_ID,
        "files": None,
    }
    as_message = _base_message(type_="message", channel_type="mpim", **common)
    as_mention = _base_message(type_="app_mention", channel_type=None, **common)
    return event_callback(as_message), event_callback(as_mention)


def thread_reply(
    parent_ts: str,
    question: str = "and the lifetime?",
    *,
    channel: str = CHANNEL_ID,
    user: str = SENDER_ID,
    user_name: str = SENDER_NAME,
    ts: str | None = None,
) -> dict[str, Any]:
    """A reply inside a thread: an ``app_mention`` in a channel, a ``message`` in a DM."""
    if channel.startswith("D"):
        return dm_message(
            question, user=user, user_name=user_name, channel=channel, ts=ts, thread_ts=parent_ts
        )
    return channel_mention(
        question, user=user, user_name=user_name, channel=channel, ts=ts, thread_ts=parent_ts
    )


def file_share(
    files: list[dict[str, Any]],
    text: str = "please look at this file",
    *,
    channel: str = DM_CHANNEL_ID,
    user: str = SENDER_ID,
    user_name: str = SENDER_NAME,
    ts: str | None = None,
) -> dict[str, Any]:
    """A DM ``message`` with ``subtype: file_share`` carrying ``files`` (see ``add_file``)."""
    payload = dm_message(text, user=user, user_name=user_name, channel=channel, ts=ts, files=files)
    payload["event"]["subtype"] = "file_share"
    payload["event"]["upload"] = False
    return payload


def bot_message(
    text: str = f"<@{BOT_USER_ID}> automated status ping",
    *,
    channel: str = DM_CHANNEL_ID,
    bot_id: str = "B0OTHERBOT",
    ts: str | None = None,
) -> dict[str, Any]:
    """Another bot's post (``subtype: bot_message``, ``bot_id``, no ``user``)."""
    channel_type = "im" if channel.startswith("D") else "channel"
    event = {
        "type": "message",
        "subtype": "bot_message",
        "bot_id": bot_id,
        "username": "otherbot",
        "text": text,
        "ts": ts or next_ts(),
        "channel": channel,
        "channel_type": channel_type,
    }
    event["event_ts"] = event["ts"]
    return event_callback(event)


def shared_channel_mention(
    question: str = "what is the beam current?",
    *,
    channel: str = "C0SHARED",
    user: str = "U0BOB",
    user_name: str = "Bob",
    user_team: str = FOREIGN_TEAM_ID,
    ts: str | None = None,
) -> dict[str, Any]:
    """An ``app_mention`` in a Slack Connect channel, from a foreign workspace's user."""
    event = _base_message(
        type_="app_mention",
        text=f"<@{BOT_USER_ID}> {question}",
        user=user,
        channel=channel,
        ts=ts,
        thread_ts=None,
        user_name=user_name,
        team=user_team,
        files=None,
        channel_type=None,
    )
    return event_callback(event, is_ext_shared_channel=True)
