"""HTTP client for the Nextcloud surfaces the Talk bridge speaks to.

:class:`TalkClient` is a thin mapper onto Nextcloud's OCS API and nothing more:
it builds the URLs, carries the bot's Basic auth plus the two headers OCS
insists on, unwraps the ``{"ocs": {"data": ...}}`` envelope, and hands the
parsed payload back. It holds **no retry or backoff policy** — every failure
raises to the caller. That is deliberate: the room-type cache and each room's
poll loop own *different* capped backoffs over these calls, and a retry hidden
in here would fight both and stretch a single logical attempt past the budget
its caller thinks it has.

One shared :class:`httpx.Client` serves every room thread and the drain thread
(httpx clients are thread-safe), so connections and TLS sessions are pooled
across rooms instead of re-handshaked per poll. Tests inject their own
:class:`httpx.MockTransport` client; auth and headers are applied per request,
so an injected transport-only client is authenticated exactly like the real one.

Timeouts — the detail that decides whether long-polling works at all:
    Talk's chat endpoint is a *long poll*. It holds the request open for up to
    ~30 s waiting for a new message and answers ``304 Not Modified`` when none
    arrives. A read timeout at or below that hold would abort every idle poll
    locally as a :class:`httpx.ReadTimeout`, so an idle room would look exactly
    like a broken network — the single most common way to get this wrong. The
    two profiles are therefore explicit and separate:

    * ordinary calls (post, room info) use the client-wide timeout built from
      :data:`CONNECT_TIMEOUT` / :data:`REQUEST_TIMEOUT`;
    * a long poll passes a **per-request** override whose read timeout is the
      requested server hold plus :data:`POLL_READ_MARGIN`, so the local timeout
      can never be the thing that ends an idle poll.

The two OCS API versions below are not a typo: Talk's chat routes are ``v1``
while the room routes moved to ``v4``. Both are the current spelling for the
Talk versions this bridge supports.

The WebDAV upload/share half of the bot's I/O lives on this same class; the
envelope helper, header set, and URL builders here are shared by both halves.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from osprey.bridges.core import MAX_DELIVERED_DOC_BYTES

from .config import NextcloudBridgeConfig

logger = logging.getLogger(__name__)

# --- OCS wire constants ----------------------------------------------------

CHAT_API = "apps/spreed/api/v1/chat"
"""Talk's chat routes, still served under OCS API ``v1``."""

ROOM_API = "apps/spreed/api/v4/room"
"""Talk's room routes, served under OCS API ``v4`` (the chat routes are not)."""

LAST_GIVEN_HEADER = "X-Chat-Last-Given"
"""Response header carrying the cursor to send as the next ``lastKnownMessageId``."""

ROOM_TYPE_ONE_TO_ONE = 1
"""``RoomInfo.type`` value for a one-to-one conversation, where every message is
addressed to the bot and the group-room mention filter does not apply."""

SHARES_API = "apps/files_sharing/api/v1/shares"
"""OCS route that mints a share. Not a Talk route — this is core Files sharing."""

SHARE_TYPE_ROOM = 10
"""``shareType`` for a Talk-room share: readable by the room's members only, with
no world-readable link. The one share type this bridge ever creates."""

_OCS_HEADERS: Mapping[str, str] = {
    # Nextcloud rejects OCS calls without this header (it is the CSRF guard for
    # the API surface), and asks for JSON explicitly rather than OCS's XML default.
    "OCS-APIRequest": "true",
    "Accept": "application/json",
}

# --- file surface: WebDAV routes and the two tolerated "already there" answers ---

DAV_FILES_ROOT = "remote.php/dav/files"
"""WebDAV entry point. A user's tree is ``{DAV_FILES_ROOT}/{user}/...`` — the bot
reads and writes only its **own** tree, so the user is always ``cfg.bot_account``."""

UPLOAD_ROOT = "OSPREY artifacts"
"""Top collection artifacts are uploaded into, inside the bot's DAV tree.

Deliberately human-readable, space included: an operator browsing the bot account
in the Nextcloud web UI should see what this folder is without decoding it. The
space is why every DAV path here is assembled from percent-encoded *segments*
rather than interpolated into a URL string."""

MAX_DOWNLOAD_BYTES = MAX_DELIVERED_DOC_BYTES
"""Cap on a single :meth:`TalkClient.dav_download`, taken from the core budget.

Two reasons to inherit it rather than pick a number: the engine's own fresh-file
ceiling is the same 10 MiB, so bytes above this are ones the pipeline would
decline anyway, and a single knob keeps the two from drifting apart. Unlike the
core constant — a post-hoc rendering bound applied after a body is fully read —
this one is a **memory** bound, enforced while the response streams, so an
unexpectedly huge attachment can never be buffered whole in a bridge process."""

_MKCOL_EXISTS_STATUS = httpx.codes.METHOD_NOT_ALLOWED
"""What a WebDAV server answers to ``MKCOL`` on a collection that already exists.

Tolerated, not an error: every upload re-creates its whole directory chain, so
the *normal* steady state is "already there" for all but the first run in a room."""

_SHARE_EXISTS_STATUSES = frozenset(
    {
        httpx.codes.BAD_REQUEST,
        httpx.codes.FORBIDDEN,
        httpx.codes.CONFLICT,
    }
)
"""Statuses a duplicate share can arrive as, across Nextcloud versions."""

_SHARE_EXISTS_MARKERS = ("already shared", "already exists")
"""Substrings in an OCS failure message that mean "this share is already there".

Status alone cannot be trusted here — a bare ``403`` is also how a genuine
permission failure arrives, and tolerating those would turn "the bot cannot share
into this room" into silence. Both a matching status *and* a matching message are
required. The message is the server's, so a Nextcloud answering in another
language falls through to raising: a caller that treats delivery as best-effort
logs a harmless failure, which is the safe side of that trade."""

# --- timeout profile -------------------------------------------------------

CONNECT_TIMEOUT = 10.0
"""Seconds allowed for the TCP + TLS handshake, for every call."""

REQUEST_TIMEOUT = 30.0
"""Read/write timeout for every ordinary (non-long-poll) call."""

DEFAULT_POLL_HOLD = 30.0
"""Seconds a long poll asks the server to hold an idle request open. 30 s is
Talk's own maximum; a larger value is simply held for less than requested."""

POLL_READ_MARGIN = 15.0
"""Slack added to the requested hold to derive a long poll's read timeout.

Must stay well above zero. If the read timeout were at or below the hold, the
local socket would time out *before* the server's ``304`` arrived and every idle
poll would surface as a transport error instead of "nothing new"."""


class TalkApiError(Exception):
    """Raised when a response is not the OCS payload the endpoint promises.

    Transport failures and HTTP error statuses surface as their own
    :class:`httpx.HTTPError` subclasses; this covers the other half — a 2xx whose
    body is not JSON, is not an OCS envelope, or whose ``data`` has the wrong
    shape for the endpoint that was called.
    """


class DavTooLargeError(TalkApiError):
    """Raised when a DAV download exceeds its byte cap.

    A :class:`TalkApiError` subclass so a caller that treats every non-transport
    surprise the same way needs only one ``except``, while a caller that wants to
    tell "too big to ship" apart from "that was not a file" can catch this alone.

    The bytes read so far are discarded — this is a memory guard, so a truncated
    prefix is deliberately not offered as a partial result.
    """


def upload_dir(room: str, run_id: str) -> str:
    """DAV directory (relative to the bot's tree) that a run's artifacts go into.

    ``{UPLOAD_ROOT}/{room}/{run_id}``: room first so an operator can see one
    conversation's artifacts together, run id second so two runs in a room cannot
    collide on a generic filename like ``plot.png``.

    Args:
        room: Talk room token.
        run_id: Dispatch run id the artifacts came from.

    Returns:
        The unencoded relative path, with no trailing ``/``. Pass it straight to
        :meth:`TalkClient.dav_mkcol_put`, which creates each level and encodes the
        segments (:data:`UPLOAD_ROOT` contains a space).

    Raises:
        ValueError: If either component is empty or is not a single path segment
            — a ``/`` or ``..`` in a room token or run id would otherwise address
            a directory outside the artifact tree.
    """
    for label, value in (("room", room), ("run_id", run_id)):
        if not value or "/" in value or value in (".", ".."):
            raise ValueError(f"{label} must be a single path segment; got {value!r}")
    return f"{UPLOAD_ROOT}/{room}/{run_id}"


@dataclass(frozen=True)
class RoomInfo:
    """The parsed room payload the bridge needs from Talk's v4 room route."""

    token: str
    """The room token this describes (echoed from the request, not the payload)."""

    type: int
    """Talk's conversation type. :data:`ROOM_TYPE_ONE_TO_ONE` means one-to-one;
    anything else is a group room, where the mention filter applies."""

    display_name: str
    """Human-readable room name, for logs and operator-facing notices. Empty
    string when Talk sends none — never ``None``, so log lines stay printable."""

    last_message_id: int | None
    """Id of the room's most recent message, or ``None`` when the room has none.

    This is the anchor a room's first-ever start uses to skip history: polling
    from here dispatches only what arrives *after* the bridge came up."""


def _last_given(resp: httpx.Response) -> int | None:
    """Read the ``X-Chat-Last-Given`` cursor from a chat response.

    The cursor is advisory rather than load-bearing — a caller can always fall
    back to the highest message id in the batch — so a missing or unparseable
    header yields ``None`` instead of failing a poll that returned real messages.

    Args:
        resp: The chat response to read.

    Returns:
        The cursor, or ``None`` if the header is absent or not an integer.
    """
    raw = resp.headers.get(LAST_GIVEN_HEADER)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring unparseable %s header: %r", LAST_GIVEN_HEADER, raw)
        return None


def _last_message_id(data: Mapping[str, Any]) -> int | None:
    """Extract ``lastMessage.id`` from a room payload.

    Talk serializes an *absent* ``lastMessage`` as an empty JSON list (PHP's
    empty array), not as ``null`` or an empty object, so the type is checked
    rather than assumed.

    Args:
        data: The room payload's ``ocs.data`` object.

    Returns:
        The last message's id, or ``None`` when the room has no messages.
    """
    last = data.get("lastMessage")
    if isinstance(last, dict):
        message_id = last.get("id")
        if isinstance(message_id, int):
            return message_id
    return None


def _already_shared(resp: httpx.Response) -> bool:
    """Whether a failed share response means "this share already exists".

    Both halves must agree — a status from :data:`_SHARE_EXISTS_STATUSES` *and* an
    OCS message matching :data:`_SHARE_EXISTS_MARKERS`. A body that is not an OCS
    envelope, or a message in another language, reads as "not a duplicate" so the
    caller raises rather than assuming the file arrived.

    Args:
        resp: The share response, of any status.

    Returns:
        ``True`` if the response should be tolerated as a duplicate.
    """
    if resp.status_code not in _SHARE_EXISTS_STATUSES:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    message = ""
    if isinstance(body, dict):
        ocs = body.get("ocs")
        meta = ocs.get("meta") if isinstance(ocs, dict) else None
        raw = meta.get("message") if isinstance(meta, dict) else None
        if isinstance(raw, str):
            message = raw.lower()
    return any(marker in message for marker in _SHARE_EXISTS_MARKERS)


class TalkClient:
    """Nextcloud OCS calls the Talk bridge makes, over one shared HTTP client.

    Thread-safe: the instance is stateless apart from the pooled
    :class:`httpx.Client`, so all room threads share one.
    """

    def __init__(
        self,
        cfg: NextcloudBridgeConfig,
        client: httpx.Client | None = None,
    ):
        """Wire the client to a config.

        Args:
            cfg: The bridge config supplying the base URL, the bot credentials,
                and (via ``cfg.core.trust_env``) whether ambient proxy
                environment variables are honored.
            client: Client to use instead of the default one. Tests pass a
                :class:`httpx.MockTransport`-backed client; auth and headers are
                still applied per request, so it sees exactly what a real
                Nextcloud would.
        """
        self._cfg = cfg
        self._auth = httpx.BasicAuth(cfg.bot_account, cfg.app_password)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=REQUEST_TIMEOUT,
                write=REQUEST_TIMEOUT,
                pool=CONNECT_TIMEOUT,
            ),
            # Default False: the Nextcloud host is reached directly, so a proxy
            # inherited from a dev shell or CI runner cannot silently mount
            # itself in front of it. Same rule the core's clients follow.
            trust_env=cfg.core.trust_env,
        )

    # --- shared plumbing (chat, room, and the file surface all use these) ---

    def _url(self, path: str) -> str:
        """Absolute URL for ``path`` under the Nextcloud instance root."""
        return f"{self._cfg.base_url}/{path.lstrip('/')}"

    def _ocs_url(self, path: str) -> str:
        """Absolute URL for ``path`` under the OCS v2 entry point."""
        return self._url(f"ocs/v2.php/{path.lstrip('/')}")

    @staticmethod
    def _token(room: str) -> str:
        """Percent-encode a room token for use as a single URL path segment.

        Room tokens come from operator-set config; encoding them means a stray
        ``/`` yields a 404 from the intended route rather than silently
        addressing a different endpoint.
        """
        return quote(room, safe="")

    @staticmethod
    def _ocs_data(resp: httpx.Response) -> Any:
        """Unwrap ``ocs.data`` from an OCS response body.

        Args:
            resp: A response whose status has already been checked.

        Returns:
            The ``data`` member, whatever shape the endpoint returns.

        Raises:
            TalkApiError: If the body is not JSON or not an OCS envelope.
        """
        try:
            body = resp.json()
        except ValueError as exc:
            raise TalkApiError(
                f"{resp.request.method} {resp.request.url} returned a non-JSON body: "
                f"{resp.text[:200]!r}"
            ) from exc
        if not isinstance(body, dict) or not isinstance(body.get("ocs"), dict):
            raise TalkApiError(
                f"{resp.request.method} {resp.request.url} returned no OCS envelope: "
                f"{resp.text[:200]!r}"
            )
        ocs = body["ocs"]
        if "data" not in ocs:
            raise TalkApiError(
                f"{resp.request.method} {resp.request.url} returned an OCS envelope "
                f"without data: {resp.text[:200]!r}"
            )
        return ocs["data"]

    @staticmethod
    def _poll_timeout(hold: float) -> httpx.Timeout:
        """Timeout profile for a long poll asking the server to hold ``hold`` seconds.

        The read leg is ``hold + POLL_READ_MARGIN`` so the server's own answer —
        messages or a ``304`` — always arrives before the local socket gives up.
        """
        return httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=hold + POLL_READ_MARGIN,
            write=REQUEST_TIMEOUT,
            pool=CONNECT_TIMEOUT,
        )

    # --- chat surface -------------------------------------------------------

    def poll_messages(
        self,
        room: str,
        last_known: int,
        timeout: float = DEFAULT_POLL_HOLD,
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Long-poll a room for messages newer than ``last_known``.

        Blocks until Talk has something to return or its hold expires. An empty
        result is the *normal* idle outcome, signalled by Talk as ``304`` and
        reported here as ``([], None)`` — never an error.

        Args:
            room: Talk room token.
            last_known: Cursor to read past — the previous
                ``X-Chat-Last-Given``, or a first-ever start's anchor (use
                :attr:`RoomInfo.last_message_id` to skip history, or ``0`` to
                read a room from its beginning). The anchor message itself is
                excluded from the batch.
            timeout: Seconds to ask the server to hold an idle request open.
                Sent truncated to whole seconds, so a value below 1 means "do
                not hold". Talk caps its own hold at 30 s.

        Returns:
            ``(messages, last_given)``. ``messages`` are Talk's raw message
            objects, passed through unmodified in the order Talk returned them.
            ``last_given`` is the cursor for the next call, or ``None`` when
            nothing was returned or Talk sent no usable cursor header.

        Raises:
            ValueError: If ``timeout`` is negative.
            TalkApiError: If a 2xx body is not an OCS envelope holding a list of
                message objects.
            httpx.HTTPError: On any transport failure or non-304 error status.
                Backoff is the caller's; this raises on the first failure.
        """
        if timeout < 0:
            raise ValueError(f"long-poll hold must be >= 0; got {timeout}")
        resp = self._client.get(
            self._ocs_url(f"{CHAT_API}/{self._token(room)}"),
            params={
                "lookIntoFuture": 1,
                "lastKnownMessageId": last_known,
                # Explicit rather than defaulted: re-delivering the anchor
                # message on every poll would hand the engine a duplicate to
                # absorb on each pass.
                "includeLastKnown": 0,
                "timeout": int(timeout),
            },
            headers=_OCS_HEADERS,
            auth=self._auth,
            timeout=self._poll_timeout(timeout),
        )
        # Talk's "nothing new within the hold window" — an idle room, not a
        # failure. Must not raise: it is the most common outcome by far.
        if resp.status_code == httpx.codes.NOT_MODIFIED:
            return [], None
        resp.raise_for_status()
        data = self._ocs_data(resp)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TalkApiError(f"chat payload for room {room!r} is not a list of messages")
        return list(data), _last_given(resp)

    def post_message(self, room: str, text: str, reply_to: int | None = None) -> int | None:
        """Post one chat message into a room.

        Args:
            room: Talk room token.
            text: Message body, posted verbatim (Talk renders a markdown
                subset). Not split here — Talk's per-message length limit is the
                caller's to respect.
            reply_to: Id of the message this replies to. Omitted when ``None``,
                giving a plain room message rather than a quoted reply.

        Returns:
            The new message's id, or ``None`` if the response carried none.

        Raises:
            TalkApiError: If a 2xx body is not an OCS envelope.
            httpx.HTTPError: On any transport failure or error status.
        """
        payload: dict[str, Any] = {"message": text}
        if reply_to is not None:
            payload["replyTo"] = reply_to
        resp = self._client.post(
            self._ocs_url(f"{CHAT_API}/{self._token(room)}"),
            # OCS takes form-encoded parameters, not a JSON body.
            data=payload,
            headers=_OCS_HEADERS,
            auth=self._auth,
        )
        resp.raise_for_status()
        data = self._ocs_data(resp)
        if isinstance(data, dict):
            message_id = data.get("id")
            if isinstance(message_id, int):
                return message_id
        return None

    # --- room surface -------------------------------------------------------

    def room_info(self, room: str) -> RoomInfo:
        """Fetch a room's type, name, and newest message id.

        Args:
            room: Talk room token.

        Returns:
            The parsed room payload. Compare :attr:`RoomInfo.type` against
            :data:`ROOM_TYPE_ONE_TO_ONE` to decide whether the mention filter
            applies.

        Raises:
            TalkApiError: If the payload carries no integer ``type`` — the
                bridge must not guess a room's type, since guessing wrong either
                drops every group message or answers every unaddressed one.
            httpx.HTTPError: On any transport failure or error status. Backoff is
                the caller's.
        """
        resp = self._client.get(
            self._ocs_url(f"{ROOM_API}/{self._token(room)}"),
            headers=_OCS_HEADERS,
            auth=self._auth,
        )
        resp.raise_for_status()
        data = self._ocs_data(resp)
        if not isinstance(data, dict):
            raise TalkApiError(f"room payload for {room!r} is not an object")
        room_type = data.get("type")
        if not isinstance(room_type, int):
            raise TalkApiError(
                f"room payload for {room!r} has no integer type: {data.get('type')!r}"
            )
        display_name = data.get("displayName")
        return RoomInfo(
            token=room,
            type=room_type,
            display_name=display_name if isinstance(display_name, str) else "",
            last_message_id=_last_message_id(data),
        )

    # --- file surface -------------------------------------------------------

    @staticmethod
    def _segments(path: str) -> list[str]:
        """Split a DAV-relative path into validated segments.

        Empty segments are dropped, so a leading ``/`` or a doubled ``//`` is
        harmless. ``.`` and ``..`` are rejected rather than normalized: paths come
        from Talk message payloads and from run ids, and a traversal segment there
        would let a crafted value address a collection outside the intended tree.

        Args:
            path: Path relative to the bot's DAV root.

        Returns:
            The path's segments, unencoded, in order.

        Raises:
            ValueError: If the path is empty or holds a ``.``/``..`` segment.
        """
        segments = [part for part in path.split("/") if part]
        if not segments:
            raise ValueError(f"DAV path must name a file or collection; got {path!r}")
        for part in segments:
            if part in (".", ".."):
                raise ValueError(f"DAV path may not contain {part!r}: {path!r}")
        return segments

    def _dav_url(self, path: str) -> str:
        """Absolute WebDAV URL for ``path`` inside the bot's own file tree.

        Every segment is percent-encoded individually — the account name included,
        the ``/`` separators excluded. That is what keeps :data:`UPLOAD_ROOT`'s
        space (and any space in an artifact filename) from ending the path early
        or being read as a query separator.

        Args:
            path: Path relative to the bot's DAV root.

        Returns:
            The absolute URL.

        Raises:
            ValueError: If ``path`` is empty or holds a traversal segment.
        """
        parts = [self._cfg.bot_account, *self._segments(path)]
        encoded = "/".join(quote(part, safe="") for part in parts)
        return self._url(f"{DAV_FILES_ROOT}/{encoded}")

    def dav_download(self, path: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
        """Download one file from the bot's own DAV tree.

        The response is **streamed** and abandoned the moment it passes
        ``max_bytes``, so a file far larger than the cap costs neither the memory
        nor the time to read it whole. A declared ``Content-Length`` short-circuits
        that check, but an absent or understated one cannot get past it.

        Only the bot's tree is reachable: the URL is built from
        ``cfg.bot_account``, so a path from a message payload selects a file inside
        that tree and can never name another user's.

        Args:
            path: Path relative to the bot's DAV root — for an inbound attachment,
                the ``path`` Talk reports for the shared file.
            max_bytes: Cap on the download. Defaults to
                :data:`MAX_DOWNLOAD_BYTES`.

        Returns:
            The file's bytes.

        Raises:
            ValueError: If ``path`` is empty or holds a traversal segment, or
                ``max_bytes`` is not positive.
            DavTooLargeError: If the file is larger than ``max_bytes``.
            httpx.HTTPError: On any transport failure or error status (a deleted
                file is a plain ``404``). Nothing is swallowed here — a caller for
                which a missing attachment is survivable decides that itself.
        """
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0; got {max_bytes}")
        url = self._dav_url(path)
        chunks: list[bytes] = []
        total = 0
        with self._client.stream("GET", url, auth=self._auth) as resp:
            resp.raise_for_status()
            declared = resp.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise DavTooLargeError(
                    f"{path!r} declares {declared} bytes, over the {max_bytes}-byte cap"
                )
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise DavTooLargeError(
                        f"{path!r} exceeds the {max_bytes}-byte cap; download abandoned"
                    )
                chunks.append(chunk)
        return b"".join(chunks)

    def dav_mkcol_put(self, directory: str, name: str, data: bytes) -> str:
        """Create ``directory`` if needed and upload ``data`` into it as ``name``.

        WebDAV's ``MKCOL`` is **not recursive** — it fails with ``409`` when the
        parent is missing — so each level of ``directory`` is created in turn,
        outermost first. Every one of those calls tolerates
        :data:`_MKCOL_EXISTS_STATUS`, which is the answer for all but the first
        upload into a given tree; a fresh deployment and the thousandth run take
        the same path.

        Args:
            directory: Directory relative to the bot's DAV root, ``/``-separated
                and as deep as needed. :func:`upload_dir` builds the layout this
                bridge uses.
            name: Filename to write, a single path segment.
            data: File contents, uploaded in one ``PUT``.

        Returns:
            The uploaded file's path relative to the bot's DAV root, unencoded —
            hand it straight to :meth:`share_to_room`.

        Raises:
            ValueError: If ``directory`` or ``name`` is empty or holds a traversal
                segment, or ``name`` is more than one segment.
            httpx.HTTPError: On any transport failure or error status other than a
                tolerated "collection exists". Raises on the first failure — a
                caller for which delivery is best-effort decides that itself.
        """
        parents = self._segments(directory)
        leaf = self._segments(name)
        if len(leaf) != 1:
            raise ValueError(f"upload name must be a single path segment; got {name!r}")
        for depth in range(1, len(parents) + 1):
            self._mkcol("/".join(parents[:depth]))
        path = "/".join([*parents, leaf[0]])
        resp = self._client.request("PUT", self._dav_url(path), content=data, auth=self._auth)
        resp.raise_for_status()
        return path

    def _mkcol(self, path: str) -> None:
        """Create one collection, tolerating "it already exists".

        Args:
            path: Collection path relative to the bot's DAV root. Its parent must
                already exist — see :meth:`dav_mkcol_put`.

        Raises:
            httpx.HTTPError: On any transport failure or error status other than
                :data:`_MKCOL_EXISTS_STATUS`.
        """
        resp = self._client.request("MKCOL", self._dav_url(path), auth=self._auth)
        if resp.status_code == _MKCOL_EXISTS_STATUS:
            logger.debug("DAV collection %s already exists", path)
            return
        resp.raise_for_status()

    def share_to_room(self, path: str, room: str) -> None:
        """Share a file from the bot's tree into a Talk room.

        Creates a :data:`SHARE_TYPE_ROOM` share, which is visible to the room's
        members and to nobody else — no world-readable link is ever minted, so
        there is no URL for this to return. A share that already exists is
        tolerated (see :data:`_SHARE_EXISTS_MARKERS`), which is what makes a
        redelivery of the same artifact idempotent.

        Args:
            path: File path relative to the bot's DAV root — the return value of
                :meth:`dav_mkcol_put`.
            room: Talk room token to share into.

        Raises:
            ValueError: If ``path`` is empty or holds a traversal segment.
            httpx.HTTPError: On any transport failure or error status that is not a
                recognized duplicate — including a genuine ``403``, which is how
                "the bot may not share into that room" arrives and must not pass
                silently.
        """
        resp = self._client.post(
            self._ocs_url(SHARES_API),
            # Form-encoded like every OCS write; the encoder handles the space in
            # UPLOAD_ROOT, so this path is sent unencoded and absolute-in-tree.
            data={
                "path": "/" + "/".join(self._segments(path)),
                "shareType": SHARE_TYPE_ROOM,
                "shareWith": room,
            },
            headers=_OCS_HEADERS,
            auth=self._auth,
        )
        if _already_shared(resp):
            logger.debug("%s is already shared with room %s", path, room)
            return
        resp.raise_for_status()

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the shared HTTP client. Idempotent."""
        self._client.close()
