"""Google Chat REST calls the bridge makes, over one discovery service object.

:class:`ChatClient` is a thin mapper onto the Chat v1 ``spaces.messages`` surface
and nothing more: it assembles the message body (text, threading, ``cardsV2``),
issues the call, and hands the parsed resource back. It carries **no retry, no
backoff, and no swallow policy** for posting — a failed create raises, on the
first failure, having made exactly one request.

That is deliberate rather than unfinished. The engine's crash-safety ordering is
built on the outbound members' *failure* contracts (see
:mod:`osprey.bridges.core.ports`): ``post_answer`` and ``post_giveup`` MUST raise
so a lost message is re-delivered next cycle, while ``post_ack`` and
``deliver_files`` must swallow so a cosmetic failure never un-delivers an answer.
Those are four different policies over the *same* REST call, so the call site —
``GoogleChatOps`` — is the only place that can choose correctly. A retry or an
``except`` hidden down here would fight all four.

:meth:`ChatClient.get_message` is the one exception, and only because its single
caller is reply-context enrichment: a Chat hiccup there degrades to answering
without the quoted context, so it returns ``None`` instead of raising.

Threading
---------
Every message is created with
``messageReplyOption='REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD'`` plus the inbound
thread's name, so a reply lands in the thread the question was asked in — and
still lands, in a new thread, if that thread is gone by the time the answer is
ready. The option is not optional: without it an unknown thread name is a hard
error, which for a long-running agent turn is a dropped answer.

Chunking
--------
Chat rejects a message body over :data:`MAX_CHARS` characters, and an agent
answer routinely exceeds it, so one answer travels as several threaded messages.
The splitting itself is channel-agnostic and lives in
:mod:`osprey.bridges.core.text`; what stays here is Chat's ceiling — :func:`chunk_text`
is a wrapper that supplies :data:`MAX_CHARS` as the default the core function
deliberately refuses to pick. It is a pure function the ops layer calls; this
class never chunks on its own, because which chunk carries the ``cardsV2``
payload is the ops layer's decision.

Google imports
--------------
``googleapiclient`` and ``google-auth`` ship in the optional ``gchat`` extra, so
they are imported **inside** :func:`build_chat_service` and nowhere else. Importing
this module — and driving :class:`ChatClient` over an injected service — therefore
needs no Google library installed at all, which is what lets the adapter's whole
unit suite run in an environment without the extra.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

# Re-exported, not used here: the helper moved to the core with the chunker, and
# this name stays importable from the module it has always lived in.
from ..core.text import _fence_spans as _fence_spans
from ..core.text import chunk_text as _core_chunk_text
from .config import GoogleChatBridgeConfig

logger = logging.getLogger(__name__)

SA_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/chat.bot",)
"""OAuth scope the service account authenticates with.

``chat.bot`` is "act as the app": messages appear as sent by the Osprey Chat app
rather than by a user, and it is also the credential ``spaces.messages.get``
answers a full message snapshot to. The bridge holds no user OAuth token, so this
is the only identity it ever posts or reads as."""

MAX_CHARS = 4096
"""Chat's hard per-message character limit, and therefore :func:`chunk_text`'s
default ceiling. A body over it is rejected outright — not truncated — so the
splitting is what makes a long answer deliverable at all."""

REPLY_OPTION = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
"""``messageReplyOption`` for every create: reply into the named thread, and fall
back to opening a new thread if that thread no longer exists. The fallback is the
load-bearing half — an agent turn can outlive the thread it was asked in, and the
alternative to a new thread is an error and a silently dropped answer."""


def build_chat_service(cfg: GoogleChatBridgeConfig) -> Any:
    """Build an authenticated Chat v1 service from the configured service-account key.

    The default factory :class:`ChatClient` falls back to when no service is
    injected. Both Google imports are function-local: see the module docstring —
    they live in the optional ``gchat`` extra, and nothing else in this module may
    depend on them being installed.

    Args:
        cfg: Bridge config supplying ``sa_key``, the path to the service-account
            JSON key. Presence of that path is :meth:`GoogleChatBridgeConfig.require_startup`'s
            check, not this function's — an unset key surfaces here as the file
            error it is.

    Returns:
        The ``googleapiclient`` discovery service, whose ``spaces().messages()``
        surface is all this module uses.

    Raises:
        ImportError: If the ``gchat`` extra is not installed.
        OSError: If the key file cannot be read.
        ValueError: If the key file is not a usable service-account key.
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        cfg.sa_key, scopes=list(SA_SCOPES)
    )
    # cache_discovery=False: the on-disk discovery cache wants a writable home
    # directory the container does not promise, and its absence is a warning on
    # every build otherwise. The document is fetched per process instead.
    return build("chat", "v1", credentials=credentials, cache_discovery=False)


def chunk_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split ``text`` for Chat, defaulting to :data:`MAX_CHARS` per chunk.

    A wrapper over :func:`osprey.bridges.core.text.chunk_text`, which owns the
    splitting rules — the limit, the preference for newline boundaries, and keeping
    a ``` fenced block whole. The core function requires a ``limit`` because the
    ceiling is a property of the channel; supplying Chat's is all this adds, so the
    ops layer can call it bare.

    Args:
        text: The message text, already transformed for Chat by the caller.
        limit: Maximum characters per chunk. Defaults to :data:`MAX_CHARS`.

    Returns:
        The chunks in order, none of them empty; empty for input that is empty or
        nothing but newlines. See the core function for the full contract.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    return _core_chunk_text(text, limit)


class ChatClient:
    """The Chat v1 ``spaces.messages`` calls the bridge makes, over one service.

    Thread-safe: the engine may invoke the ops members concurrently (the retry
    drain runs alongside live ingestion), and ``googleapiclient``'s transport is
    not documented as thread-safe, so the HTTP leg of every call is serialized
    behind one lock. Serializing costs nothing here — an answer is a handful of
    messages, posted in sequence anyway — and the alternative is two threads
    sharing one connection.
    """

    def __init__(self, cfg: GoogleChatBridgeConfig, service: Any | None = None) -> None:
        """Wire the client to a Chat service.

        Args:
            cfg: The bridge config. Used only to build the default service; nothing
                is read from it afterwards, so an injected service makes the config
                irrelevant to every call.
            service: Service to use instead of building one. Tests pass a mock (or
                the e2e fake's discovery stand-in) and thereby need no Google
                library at all; the wiring passes whatever its injected factory
                returned. ``None`` builds the real one via
                :func:`build_chat_service`.
        """
        self._service: Any = build_chat_service(cfg) if service is None else service
        self._lock = threading.Lock()

    def _execute(self, request: Any) -> Any:
        """Run one built request's HTTP leg under the shared lock.

        Args:
            request: A ``googleapiclient`` request object, already built off the
                service (building is local model lookup and needs no lock).

        Returns:
            Whatever the API answered, undecoded by us.

        Raises:
            Exception: Whatever the transport raises, unchanged — callers own the
                policy.
        """
        with self._lock:
            return request.execute()

    def create_message(
        self,
        space: str,
        thread: str | None,
        text: str,
        *,
        cards: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create one message in ``space``, threaded under ``thread``.

        Posts with :data:`REPLY_OPTION`, so the message joins the named thread or
        opens a new one if that thread is gone. Nothing is chunked here: ``text``
        is sent as one body, and a body over :data:`MAX_CHARS` is rejected by Chat
        — run it through :func:`chunk_text` first.

        Args:
            space: Space resource name (``spaces/AAAA``) to post into.
            thread: Thread resource name to reply under. ``None`` (or empty) omits
                the thread from the body, which starts a new thread.
            text: The message body, sent verbatim. Any Markdown-to-Chat transform
                is the caller's, applied to its own copy.
            cards: ``cardsV2`` entries to attach (images, document buttons). Omitted
                from the body when ``None`` or empty, which is byte-identical to a
                text-only post.

        Returns:
            The created ``Message`` resource. An empty dict if the service answered
            something that is not a message object — the message still landed, there
            is simply nothing to report about it.

        Raises:
            Exception: Whatever the Chat transport raises (typically
                ``googleapiclient.errors.HttpError``). Never swallowed: the ops
                layer decides, per outcome, whether a lost message must raise.
        """
        body: dict[str, Any] = {"text": text}
        if cards:
            body["cardsV2"] = list(cards)
        if thread:
            body["thread"] = {"name": thread}
        response = self._execute(
            self._service.spaces()
            .messages()
            .create(parent=space, messageReplyOption=REPLY_OPTION, body=body)
        )
        return response if isinstance(response, dict) else {}

    def get_message(self, name: str) -> dict[str, Any] | None:
        """Fetch one ``Message`` resource by name, or ``None`` on any failure.

        The quote-reply fallback: when a Chat event carries no usable quoted-message
        snapshot, the reply-context path re-reads the quoted message here and parses
        the metadata off this fuller view instead.

        The only guarded call on this class, and only because reply context is pure
        enrichment — a Chat hiccup degrades to answering the question without the
        quoted text, which is far better than failing the answer over it.

        Args:
            name: Message resource name (``spaces/AAAA/messages/BBBB``).

        Returns:
            The message resource, or ``None`` if the call failed or answered
            something that is not a message object.
        """
        try:
            body = self._execute(self._service.spaces().messages().get(name=name))
        except Exception:
            logger.warning("messages.get failed for %s; no reply context", name, exc_info=True)
            return None
        return body if isinstance(body, dict) else None
