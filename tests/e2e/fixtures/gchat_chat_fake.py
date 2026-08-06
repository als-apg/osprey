"""A fake Google Chat REST server: the room the bridge posts into.

What it is for
--------------
Every unit test of ``osprey.bridges.google_chat`` drives the adapter over a
mocked service object, so all of them assert OSPREY's half of a two-party
contract against OSPREY's own idea of Chat's wire format. This fake is the other
half made concrete: a real HTTP server on loopback that speaks the subset of the
Chat v1 REST surface the bridge actually uses, and **records every posted
message** so a test can assert what landed in the room — text, chunking, thread,
and the ``cardsV2`` payload — instead of asserting against a mock's call log.

Surface implemented
-------------------
``POST   /v1/{space}/messages``       ``spaces.messages.create``
``GET    /v1/{space}/messages/{id}``  ``spaces.messages.get``
``GET    /v1/{space}/messages``       ``spaces.messages.list``
``GET    /v1/media/{resourceName}``   media download (``?alt=media``), the route
                                      the inbound attachment path fetches with

Anything else is a 404 in Google's JSON error envelope. Message and thread names
are readable stand-ins (``spaces/AAA/messages/msg-1``), not Google's opaque ids;
nothing in the bridge parses them.

Threading fidelity worth knowing
--------------------------------
The bridge posts with ``messageReplyOption='REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD'``
and a ``thread.name`` taken from the inbound event. This fake honors that
literally: an unknown thread name creates a new thread **only** under that
option, and 404s without it — so a test that drops the option sees the same
failure a real deployment would, rather than a silently forgiving fake.
``thread.threadKey`` is not implemented (the bridge does not use it).

Two ways in
-----------
:meth:`FakeChatServer.service` returns a stand-in for the
``googleapiclient.discovery`` service, so ``ChatClient``'s injected-service seam
can be pointed at this fake with no Google libraries involved; the calls it makes
travel over real HTTP to this server. Code that speaks raw HTTP (the media
download path uses an ``AuthorizedSession``) can address :attr:`base_url`
directly.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from .gchat_fake_http import FakeHttpService, _FakeHandler

__all__ = ["FakeChatServer", "PostedMessage", "FakeChatApiError"]

_MESSAGES_COLLECTION = re.compile(r"^/v1/(?P<space>spaces/[^/]+)/messages$")
_MESSAGE_NAME = re.compile(r"^/v1/(?P<name>spaces/[^/]+/messages/[^/]+)$")
_MEDIA = re.compile(r"^/v1/media/(?P<resource>.+)$")

REPLY_FALLBACK = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
"""The ``messageReplyOption`` under which an unknown thread name is not an error."""

HTTP_TIMEOUT = 10.0
"""Client-side timeout for the discovery stand-in. Loopback: generous already."""


class FakeChatApiError(RuntimeError):
    """Raised by the discovery stand-in when ``googleapiclient`` is not installed.

    With the ``gchat`` extra present the stand-in raises the real
    ``googleapiclient.errors.HttpError`` instead, so product code that catches it
    behaves exactly as it does in production. This class is the google-free
    fallback, and carries the same ``status_code`` the response had.
    """

    def __init__(self, status_code: int, content: bytes) -> None:
        super().__init__(f"HTTP {status_code}: {content[:400]!r}")
        self.status_code = status_code
        self.content = content


@dataclass(frozen=True)
class PostedMessage:
    """One message the bridge posted, as the room received it."""

    name: str
    space: str
    thread: str
    text: str
    cards: list[dict[str, Any]]
    body: dict[str, Any]
    """The create body verbatim — for assertions this dataclass does not name."""
    params: dict[str, str]
    """Query parameters of the create call (``messageReplyOption``, ...)."""

    @property
    def has_cards(self) -> bool:
        return bool(self.cards)


@dataclass
class _MediaObject:
    data: bytes
    content_type: str


class _ChatHandler(_FakeHandler):
    fake: ClassVar[FakeChatServer]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path, params = self.split_path()
        match = _MESSAGES_COLLECTION.match(path)
        if match is None:
            self.respond_error(404, f"no such route: POST {path}")
            return
        body = self.read_json()
        try:
            message = self.fake._create_message(match.group("space"), body, params)
        except _UnknownThread as exc:
            self.respond_error(404, str(exc))
            return
        self.respond_json(200, message)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path, params = self.split_path()

        media = _MEDIA.match(path)
        if media is not None:
            self._serve_media(media.group("resource"), params)
            return

        collection = _MESSAGES_COLLECTION.match(path)
        if collection is not None:
            self.respond_json(200, self.fake._list_messages(collection.group("space"), params))
            return

        single = _MESSAGE_NAME.match(path)
        if single is not None:
            message = self.fake._get_message(single.group("name"))
            if message is None:
                self.respond_error(404, f"message not found: {single.group('name')}")
                return
            self.respond_json(200, message)
            return

        self.respond_error(404, f"no such route: GET {path}")

    def _serve_media(self, resource: str, params: dict[str, str]) -> None:
        if params.get("alt") != "media":
            self.respond_error(
                400, "media downloads require alt=media", api_status="INVALID_ARGUMENT"
            )
            return
        obj = self.fake._take_media(resource)
        if obj is None:
            self.respond_error(404, f"media not found: {resource}")
            return
        self.respond(200, obj.data, content_type=obj.content_type)


class FakeChatServer(FakeHttpService):
    """The fake room. Start it, point the bridge at it, assert on what arrived."""

    handler_class: ClassVar[type[_FakeHandler]] = _ChatHandler

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._posted: list[PostedMessage] = []
        self._threads: set[str] = set()
        self._media: dict[str, _MediaObject] = {}
        self._media_requests: list[str] = []
        self._counter = 0
        super().__init__()

    # -- recording surface (what tests assert on) ---------------------------

    @property
    def posted(self) -> list[PostedMessage]:
        """Every message posted through ``create``, in order."""
        with self._lock:
            return list(self._posted)

    @property
    def posted_text(self) -> list[str]:
        return [m.text for m in self.posted]

    @property
    def media_requests(self) -> list[str]:
        """Resource names that were requested from the media route, in order."""
        with self._lock:
            return list(self._media_requests)

    def wait_for_posted(self, count: int, timeout: float = 30.0) -> list[PostedMessage]:
        """Block until at least ``count`` messages have been posted.

        Raises ``AssertionError`` naming what did arrive — never returns short,
        so a caller cannot mistake a timeout for an empty room.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            posted = self.posted
            if len(posted) >= count:
                return posted
            time.sleep(0.05)
        posted = self.posted
        raise AssertionError(
            f"expected >= {count} posted message(s) within {timeout:.0f}s, "
            f"got {len(posted)}: {[m.text[:80] for m in posted]}"
        )

    def reset(self) -> None:
        """Drop recorded messages and threads. Registered media survives."""
        with self._lock:
            self._messages.clear()
            self._order.clear()
            self._posted.clear()
            self._threads.clear()
            self._media_requests.clear()

    # -- seeding surface ----------------------------------------------------

    def add_media(self, resource_name: str, data: bytes, content_type: str = "image/png") -> str:
        """Register bytes the media route will serve; returns ``resource_name``.

        The resource name is what an inbound attachment's
        ``attachmentDataRef.resourceName`` carries and what the download URL
        interpolates, slashes and all.
        """
        with self._lock:
            self._media[resource_name] = _MediaObject(data, content_type)
        return resource_name

    def media_url(self, resource_name: str) -> str:
        """The download URL for ``resource_name`` on THIS fake."""
        quoted = urllib.parse.quote(resource_name, safe="/")
        return f"{self.base_url}/v1/media/{quoted}?alt=media"

    # -- route implementations ---------------------------------------------

    def _create_message(
        self, space: str, body: dict[str, Any], params: dict[str, str]
    ) -> dict[str, Any]:
        requested_thread = ((body.get("thread") or {}).get("name") or "").strip()
        with self._lock:
            self._counter += 1
            n = self._counter
            if requested_thread and requested_thread not in self._threads:
                if params.get("messageReplyOption") != REPLY_FALLBACK:
                    raise _UnknownThread(
                        f"thread not found: {requested_thread} (and messageReplyOption is "
                        f"{params.get('messageReplyOption')!r}, not {REPLY_FALLBACK})"
                    )
                requested_thread = ""
            thread = requested_thread or f"{space}/threads/thread-{n}"
            self._threads.add(thread)

            name = f"{space}/messages/msg-{n}"
            cards = list(body.get("cardsV2") or [])
            message: dict[str, Any] = {
                "name": name,
                "text": body.get("text", ""),
                "thread": {"name": thread},
                "sender": {"name": "users/app", "type": "BOT"},
                "createTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if cards:
                message["cardsV2"] = cards
            self._messages[name] = message
            self._order.append(name)
            self._posted.append(
                PostedMessage(
                    name=name,
                    space=space,
                    thread=thread,
                    text=str(body.get("text", "")),
                    cards=cards,
                    body=dict(body),
                    params=dict(params),
                )
            )
            return dict(message)

    def _get_message(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            message = self._messages.get(name)
            return dict(message) if message is not None else None

    def _list_messages(self, space: str, params: dict[str, str]) -> dict[str, Any]:
        """Creation-ordered list for one space. Single page — no page token."""
        with self._lock:
            messages = [
                dict(self._messages[n]) for n in self._order if n.startswith(f"{space}/messages/")
            ]
        try:
            page_size = int(params.get("pageSize", "0"))
        except ValueError:
            page_size = 0
        if page_size > 0:
            messages = messages[:page_size]
        return {"messages": messages}

    def _take_media(self, resource: str) -> _MediaObject | None:
        unquoted = urllib.parse.unquote(resource)
        with self._lock:
            self._media_requests.append(unquoted)
            return self._media.get(unquoted) or self._media.get(resource)

    # -- the discovery stand-in --------------------------------------------

    def service(self) -> Any:
        """A ``googleapiclient.discovery``-shaped stand-in bound to this fake.

        Supports exactly what ``ChatClient`` uses::

            service.spaces().messages().create(parent=..., body=..., **params).execute()
            service.spaces().messages().get(name=...).execute()
            service.spaces().messages().list(parent=..., **params).execute()

        Every call is a real HTTP request against this server, so assertions read
        the recorded state rather than a mock's call log.
        """
        return _ChatService(self.base_url)


class _UnknownThread(Exception):
    """Internal: an unknown ``thread.name`` without the fallback reply option."""


def _api_error(response: httpx.Response) -> Exception:
    """The real ``HttpError`` when googleapiclient is installed, else our own.

    Product code catches ``googleapiclient.errors.HttpError``; raising that exact
    class when it is importable keeps this stand-in honest about failure paths.
    The response shim is a ``dict`` subclass with ``.status``/``.reason`` because
    that is what ``httplib2.Response`` is, and ``HttpError`` reads it both ways.
    """
    try:
        from googleapiclient.errors import HttpError  # type: ignore[import-not-found]
    except ImportError:
        return FakeChatApiError(response.status_code, response.content)

    shim = _Httplib2Response(
        {"status": str(response.status_code), "content-type": "application/json"}
    )
    shim.status = response.status_code
    shim.reason = response.reason_phrase
    error: Exception = HttpError(shim, response.content, uri=str(response.url))
    return error


class _Httplib2Response(dict):
    """``httplib2.Response`` stand-in: a dict that also has ``.status``/``.reason``."""

    status: int = 0
    reason: str = ""


@dataclass(frozen=True)
class _ChatService:
    base_url: str

    def spaces(self) -> _Spaces:
        return _Spaces(self.base_url)


@dataclass(frozen=True)
class _Spaces:
    base_url: str

    def messages(self) -> _Messages:
        return _Messages(self.base_url)


@dataclass(frozen=True)
class _Messages:
    base_url: str

    def create(self, *, parent: str, body: dict[str, Any] | None = None, **params: Any) -> _Call:
        return _Call(
            "POST", f"{self.base_url}/v1/{parent}/messages", _str_params(params), body or {}
        )

    def get(self, *, name: str, **params: Any) -> _Call:
        return _Call("GET", f"{self.base_url}/v1/{name}", _str_params(params), None)

    def list(self, *, parent: str, **params: Any) -> _Call:
        return _Call("GET", f"{self.base_url}/v1/{parent}/messages", _str_params(params), None)


@dataclass(frozen=True)
class _Call:
    method: str
    url: str
    params: dict[str, str]
    body: dict[str, Any] | None = field(default=None)

    def execute(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Perform the call. ``num_retries=`` and friends are accepted and ignored."""
        response = httpx.request(
            self.method,
            self.url,
            params=self.params,
            json=self.body,
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code >= 400:
            raise _api_error(response)
        payload: dict[str, Any] = response.json()
        return payload


def _str_params(params: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in params.items() if v is not None}
