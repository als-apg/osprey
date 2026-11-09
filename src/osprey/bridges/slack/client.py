"""Slack Web API calls the bridge makes, over one httpx client.

:class:`SlackWebClient` is a thin mapper onto ``https://slack.com/api/<method>``:
one :meth:`~SlackWebClient.call` that posts form-encoded parameters with a bearer
token and hands back the JSON body when Slack says ``ok``, plus typed helpers for
the handful of methods the bridge uses. It carries **no swallow policy**: the
engine's crash-safety ordering rests on each outbound member's own failure
contract (see :mod:`osprey.bridges.core.ports`), and those differ over the same
call, so only the ops layer can choose. What *is* decided here is how a failure
is typed, because that is a fact about the failure rather than a policy.

Error typing
------------
Slack answers almost every API error as HTTP 200 with ``{"ok": false, "error":
"<code>"}``. The code is routed three ways:

* **Refusals** (:data:`REFUSAL_CODES`) say the destination itself is closed to the
  app — removed from the channel, channel archived or gone. Every retry would be
  refused the same way, so they raise
  :class:`~osprey.bridges.core.errors.UndeliverableError` and the engine settles
  the entry terminal instead of failing it once per drain pass.
* **Auth-fatal** codes (:data:`FATAL_CODES`) say the credential is dead. Nothing
  this process does will work again, so the shared ``fatal`` event is set — the
  ingest and serve loops check it and stop the bridge — and
  :class:`SlackAuthError` is raised.
* Everything else — ``missing_scope`` included, which degrades one feature but is
  never fatal — raises :class:`SlackApiError` carrying the code.

A non-2xx status, a body that is not a JSON object, and a transport failure are
:class:`SlackApiError` too.

Rate limits
-----------
A ``429`` (or an ``ok: false`` ``ratelimited``) is waited out for its
``Retry-After`` seconds and the call retried, up to :data:`RATE_LIMIT_RETRIES`
times, before it raises :class:`SlackApiError` with code ``ratelimited``. Slack
refuses a throttled call before acting on it, so the retry cannot double-post. The
wait is capped at :data:`RETRY_AFTER_MAX_SEC` so a hostile or confused header
cannot park a worker thread for a day. No lock is held while waiting:
``httpx.Client`` is thread-safe, and Slack's limits are per method, so a throttled
``chat.postMessage`` must not stall a ``users.info`` on another thread.

Files
-----
:meth:`~SlackWebClient.download` attaches the bot token only to
``https://files.slack.com`` (:data:`FILES_HOST`, default port, no userinfo) and
checks that **before** any request, so a file shared from another host raises
:class:`ForeignHostError` with nothing sent. Redirects are not followed. Slack
answers a bad or missing token with a ``200`` HTML login page rather than a
``401``, so a ``text/html`` response is :class:`NotAFileError`, as is a body
over the caller's cap (:class:`FileTooLargeError`, streamed and abandoned the
moment it passes). :meth:`~SlackWebClient.upload` runs the
``files.getUploadURLExternal`` → POST bytes → ``files.completeUploadExternal``
sequence; the upload URL is pre-signed and gets no bearer.

Credentials
-----------
Every message this module builds is capped at :data:`_ERROR_TEXT_CHARS` and has
both configured tokens (and any per-call override) scrubbed out, so no exception
can carry a credential into a log line. The bearer is built per call and never
stored on an exception.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn

import httpx

from osprey.bridges.core.errors import UndeliverableError

from .config import SlackBridgeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "API_BASE",
    "API_TIMEOUT_SEC",
    "FATAL_CODES",
    "FILES_HOST",
    "FileTooLargeError",
    "ForeignHostError",
    "MEMBERS_PAGE_MAX",
    "NotAFileError",
    "RATE_LIMIT_RETRIES",
    "REFUSAL_CODES",
    "RETRY_AFTER_DEFAULT_SEC",
    "RETRY_AFTER_MAX_SEC",
    "SlackApiError",
    "SlackAuthError",
    "SlackWebClient",
]

API_BASE = "https://slack.com/api"
"""Root of the Web API; a method is addressed as ``{API_BASE}/{method}``."""

API_TIMEOUT_SEC = 30.0
"""Timeout for one request when this module builds its own client. An injected
client carries its own; this is only the default."""

RATE_LIMIT_RETRIES = 3
"""How many times a rate-limited call is waited out and retried before raising."""

RETRY_AFTER_DEFAULT_SEC = 1.0
"""The wait used when a 429 carries no usable ``Retry-After``."""

RETRY_AFTER_MAX_SEC = 60.0
"""The longest single wait a ``Retry-After`` may impose."""

MEMBERS_PAGE_MAX = 200
"""The largest ``limit`` asked of ``conversations.members`` per page — Slack's
recommended ceiling for cursor-paginated reads."""

REFUSAL_CODES = frozenset(
    {
        "channel_not_found",
        "not_in_channel",
        "is_archived",
        "restricted_action",
        "team_access_not_granted",
    }
)
"""Codes that permanently close a destination to the app → ``UndeliverableError``."""

FATAL_CODES = frozenset({"invalid_auth", "not_authed", "token_revoked", "account_inactive"})
"""Codes that mean the credential is dead → set ``fatal``, raise ``SlackAuthError``.
``missing_scope`` is deliberately absent: it degrades one feature, not the bridge."""

FILES_HOST = "files.slack.com"
"""The only host a download may address; the bot token is never sent elsewhere."""

_RATE_LIMITED = "ratelimited"
_ERROR_TEXT_CHARS = 200
"""Cap on any remote text (a code, a body, a transport error) quoted into a message."""

_REDACTED = "[redacted]"


class SlackApiError(RuntimeError):
    """A Web API call failed for a reason a later attempt or a fix may clear.

    Covers an ``ok: false`` answer whose code is neither a refusal nor auth-fatal
    (``missing_scope``, ``user_not_found``, an exhausted rate limit as
    ``ratelimited``), a non-2xx status, a body that is not a JSON object, and a
    transport failure. ``error`` carries the Slack code when there is one (else
    ``None``) so callers can branch on it without parsing the message; ``method``
    names the call.
    """

    def __init__(self, message: str, *, method: str, error: str | None = None) -> None:
        super().__init__(message)
        self.method = method
        self.error = error


class SlackAuthError(RuntimeError):
    """The bridge's credential is dead (Decision 6); the process cannot continue.

    Raised after the shared ``fatal`` event is set. Deliberately *not* a
    :class:`SlackApiError`: an ops member that catches ``SlackApiError`` to degrade
    one feature must not swallow a revoked token along with it. The socket and
    receiver raise it too (``link_disabled``, a set fatal flag), and ``main`` maps
    it to the bridges' cannot-start exit code.
    """

    def __init__(self, message: str, *, error: str | None = None) -> None:
        super().__init__(message)
        self.error = error


class ForeignHostError(ValueError):
    """A download URL does not address ``https://`` :data:`FILES_HOST`.

    Raised before any request, so the host never sees the bot token (Decision 9).
    """


class NotAFileError(RuntimeError):
    """A download answered something other than the file's bytes.

    Slack's ``200`` HTML login page (what a bad or unscoped token gets), or a
    body over the caller's cap (:class:`FileTooLargeError`).
    """


class FileTooLargeError(NotAFileError):
    """A download declared or streamed more bytes than the caller's cap."""


class SlackWebClient:
    """One Slack workspace's Web API, authorised by the bot token."""

    def __init__(
        self,
        cfg: SlackBridgeConfig,
        http: httpx.Client | None = None,
        fatal: threading.Event | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wire the client.

        Args:
            cfg: Bridge config; supplies the bot token (default bearer), the app
                token (scrubbed from messages) and ``core.trust_env`` for the
                default HTTP client.
            http: Injected client (tests, the e2e fake). ``None`` builds one with
                :data:`API_TIMEOUT_SEC` honouring ``cfg.core.trust_env``.
            fatal: The process-wide fatal flag, set on an auth-fatal code. ``None``
                creates a private one, still readable as :attr:`fatal`.
            sleep: How a ``Retry-After`` is waited out (tests inject a recorder).
        """
        self._bot_token = cfg.bot_token
        self._secrets = tuple(t for t in (cfg.bot_token, cfg.app_token) if t)
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=API_TIMEOUT_SEC, trust_env=cfg.core.trust_env)
        )
        self._fatal = fatal if fatal is not None else threading.Event()
        self._sleep = sleep

    @property
    def fatal(self) -> threading.Event:
        """The fatal flag this client sets on an auth-fatal answer."""
        return self._fatal

    # -- the one call ------------------------------------------------------------

    def call(self, method: str, *, token: str | None = None, **params: Any) -> dict[str, Any]:
        """Call ``method`` and return its JSON body when Slack answers ``ok``.

        Parameters are sent form-encoded: ``None`` values are omitted, booleans
        become ``true``/``false``, lists and dicts are JSON-encoded (as Slack
        expects for e.g. ``files``), everything else is ``str()``-ed.

        Args:
            method: The Web API method, e.g. ``chat.postMessage``.
            token: Bearer override for this call only — the app token for
                ``apps.connections.open``. Defaults to the bot token.
            **params: The method's arguments.

        Raises:
            UndeliverableError: On a refusal code (:data:`REFUSAL_CODES`).
            SlackAuthError: On an auth-fatal code (:data:`FATAL_CODES`), after
                setting the fatal flag.
            SlackApiError: On any other failure, including an exhausted rate limit.
        """
        bearer = token if token is not None else self._bot_token
        secrets = self._secrets + ((token,) if token and token not in self._secrets else ())
        url = f"{API_BASE}/{method}"
        data = _encode(params)
        headers = {"Authorization": f"Bearer {bearer}"}

        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = self._http.post(url, data=data, headers=headers)
            except httpx.HTTPError as exc:
                detail = _clean(f"{type(exc).__name__}: {exc}", secrets)
                raise SlackApiError(
                    f"slack {method} request failed: {detail}", method=method
                ) from exc

            rate_limited = response.status_code == 429
            body: dict[str, Any] = {}
            if not rate_limited:
                if not response.is_success:
                    raise SlackApiError(
                        f"slack {method} answered HTTP {response.status_code}: "
                        f"{_clean(response.text, secrets)}",
                        method=method,
                    )
                try:
                    parsed: Any = response.json()
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    raise SlackApiError(
                        f"slack {method} answered a body that is not a JSON object",
                        method=method,
                    )
                body = parsed
                rate_limited = body.get("ok") is not True and body.get("error") == _RATE_LIMITED

            if rate_limited:
                if attempt < RATE_LIMIT_RETRIES:
                    wait = _retry_after(response)
                    logger.warning(
                        "slack %s rate limited; retrying in %.1fs (%d/%d)",
                        method,
                        wait,
                        attempt + 1,
                        RATE_LIMIT_RETRIES,
                    )
                    self._sleep(wait)
                    continue
                raise SlackApiError(
                    f"slack {method} still rate limited after {RATE_LIMIT_RETRIES} retries",
                    method=method,
                    error=_RATE_LIMITED,
                )

            if body.get("ok") is True:
                return body
            self._raise_for(method, body, secrets)

        raise AssertionError("unreachable")  # pragma: no cover - loop always returns/raises

    def _raise_for(self, method: str, body: dict[str, Any], secrets: tuple[str, ...]) -> NoReturn:
        raw = body.get("error")
        code = raw if isinstance(raw, str) and raw else "unknown_error"
        shown = _clean(code, secrets)
        if code in REFUSAL_CODES:
            raise UndeliverableError(code)
        if code in FATAL_CODES:
            self._fatal.set()
            raise SlackAuthError(f"slack {method} refused the credential: {shown}", error=code)
        raise SlackApiError(f"slack {method} failed: {shown}", method=method, error=shown)

    # -- typed helpers -------------------------------------------------------------

    def auth_test(self) -> dict[str, Any]:
        """``auth.test`` — the bot's ``user_id``, ``bot_id`` and ``team_id``."""
        return self.call("auth.test")

    def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> dict[str, Any]:
        """``chat.postMessage`` — top level, or in ``thread_ts``'s thread when given.

        Returns the body (its ``ts`` names the posted message).
        """
        return self.call("chat.postMessage", channel=channel, text=text, thread_ts=thread_ts)

    def conversations_replies(
        self, channel: str, ts: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """One page of ``conversations.replies`` for the thread rooted at ``ts``.

        Slack lists the parent first. Only dict items are kept; a body without a
        ``messages`` list is an empty thread.
        """
        body = self.call("conversations.replies", channel=channel, ts=ts, limit=limit)
        messages = body.get("messages")
        if not isinstance(messages, list):
            return []
        return [m for m in messages if isinstance(m, dict)]

    def conversations_members(self, channel: str, *, limit: int | None) -> tuple[list[str], bool]:
        """Read up to ``limit`` member ids of ``channel``, page by page.

        Each page asks ``limit`` clamped to ``[1, MEMBERS_PAGE_MAX]`` (the maximum
        without a limit) and, from the second page on, the previous page's
        ``response_metadata.next_cursor``. Stops when the cursor is empty, when it
        repeats (a looping server), or when ``limit`` ids are collected.

        Returns:
            ``(member_ids, more)``: at most ``limit`` ids, and whether the listing
            stopped on ``limit`` with more left — always ``False`` without a limit.
        """
        page_size = MEMBERS_PAGE_MAX if limit is None else max(1, min(limit, MEMBERS_PAGE_MAX))
        members: list[str] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            body = self.call(
                "conversations.members", channel=channel, limit=page_size, cursor=cursor
            )
            items = body.get("members")
            if isinstance(items, list):
                members.extend(item for item in items if isinstance(item, str))
            meta = body.get("response_metadata")
            nxt = meta.get("next_cursor") if isinstance(meta, dict) else None
            cursor = nxt if isinstance(nxt, str) and nxt else None

            if limit is not None and len(members) >= limit:
                return members[:limit], len(members) > limit or cursor is not None
            if cursor is None or cursor in seen:
                return members, False
            seen.add(cursor)

    def users_info(self, user: str) -> dict[str, Any]:
        """``users.info`` — the ``user`` object.

        Raises:
            SlackApiError: Also when the answer carries no ``user`` object.
        """
        body = self.call("users.info", user=user)
        info = body.get("user")
        if not isinstance(info, dict):
            raise SlackApiError("slack users.info answered without a user", method="users.info")
        return info

    # -- files ---------------------------------------------------------------------

    def download(self, url: str, max_bytes: int) -> bytes:
        """Fetch a Slack-hosted file (``url_private_download``) with the bot token.

        The response is streamed and abandoned the moment it passes ``max_bytes``;
        a declared ``Content-Length`` over the cap short-circuits that.

        Raises:
            ValueError: If ``max_bytes`` is not positive (no request made).
            ForeignHostError: If ``url`` is not ``https://files.slack.com/...``
                (no request made).
            NotAFileError: On an HTML (login page) response.
            FileTooLargeError: On a body over ``max_bytes``.
            SlackApiError: On a transport failure or a non-2xx status (a redirect
                included — none is followed).
        """
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0; got {max_bytes}")
        _require_files_host(url, self._secrets)
        method = "files.download"
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        chunks: list[bytes] = []
        total = 0
        try:
            with self._http.stream("GET", url, headers=headers, follow_redirects=False) as response:
                if not response.is_success:
                    raise SlackApiError(
                        f"slack file download answered HTTP {response.status_code}",
                        method=method,
                    )
                ctype = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if ctype == "text/html":
                    raise NotAFileError("slack file download answered an HTML page, not a file")
                declared = response.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise FileTooLargeError(
                        f"slack file declares {declared} bytes, over the {max_bytes}-byte cap"
                    )
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise FileTooLargeError(
                            f"slack file exceeds the {max_bytes}-byte cap; download abandoned"
                        )
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            detail = _clean(f"{type(exc).__name__}: {exc}", self._secrets)
            raise SlackApiError(f"slack file download failed: {detail}", method=method) from exc
        return b"".join(chunks)

    def upload(
        self, channel: str, thread_ts: str | None, name: str, data: bytes, mime: str
    ) -> dict[str, Any]:
        """Share ``data`` as file ``name`` into ``channel`` (in ``thread_ts``'s thread).

        ``files.getUploadURLExternal`` → POST the raw bytes to the returned
        pre-signed URL (no bearer) → ``files.completeUploadExternal``, without
        which Slack aborts the upload. An empty ``thread_ts`` posts top level.

        Returns:
            The ``files.completeUploadExternal`` body.

        Raises:
            UndeliverableError, SlackAuthError, SlackApiError: As :meth:`call`; a
                failed byte POST or an answer without ``upload_url``/``file_id``
                is :class:`SlackApiError`.
        """
        ticket = self.call("files.getUploadURLExternal", filename=name, length=len(data))
        upload_url = ticket.get("upload_url")
        file_id = ticket.get("file_id")
        if not (
            isinstance(upload_url, str) and upload_url and isinstance(file_id, str) and file_id
        ):
            raise SlackApiError(
                "slack files.getUploadURLExternal answered without upload_url/file_id",
                method="files.getUploadURLExternal",
            )
        method = "files.upload"
        try:
            response = self._http.post(upload_url, content=data, headers={"Content-Type": mime})
        except httpx.HTTPError as exc:
            detail = _clean(f"{type(exc).__name__}: {exc}", self._secrets)
            raise SlackApiError(f"slack file upload failed: {detail}", method=method) from exc
        if not response.is_success:
            raise SlackApiError(
                f"slack file upload answered HTTP {response.status_code}: "
                f"{_clean(response.text, self._secrets)}",
                method=method,
            )
        return self.call(
            "files.completeUploadExternal",
            files=[{"id": file_id, "title": name}],
            channel_id=channel,
            thread_ts=thread_ts or None,
        )


def _require_files_host(url: str, secrets: tuple[str, ...]) -> None:
    """Raise :class:`ForeignHostError` unless ``url`` is ``https://`` :data:`FILES_HOST`."""
    try:
        parsed = httpx.URL(url)
        good = (
            parsed.scheme == "https"
            and parsed.host == FILES_HOST
            and parsed.port is None
            and not parsed.userinfo
        )
    except (httpx.InvalidURL, TypeError, ValueError):
        good = False
    if not good:
        raise ForeignHostError(
            f"refusing to download from a non-Slack URL: {_clean(url, secrets)!r}"
        )


def _encode(params: dict[str, Any]) -> dict[str, str]:
    """Form-encode ``params`` the way Slack reads them."""
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            out[key] = json.dumps(value)
        else:
            out[key] = str(value)
    return out


def _retry_after(response: httpx.Response) -> float:
    """The ``Retry-After`` seconds, defaulted when absent or unparsable and capped."""
    try:
        wait = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return RETRY_AFTER_DEFAULT_SEC
    if wait != wait or wait <= 0:  # NaN or non-positive
        return RETRY_AFTER_DEFAULT_SEC
    return min(wait, RETRY_AFTER_MAX_SEC)


def _clean(text: str, secrets: tuple[str, ...]) -> str:
    """``text`` with every secret scrubbed, capped at :data:`_ERROR_TEXT_CHARS`."""
    for secret in secrets:
        text = text.replace(secret, _REDACTED)
    if len(text) > _ERROR_TEXT_CHARS:
        text = text[:_ERROR_TEXT_CHARS] + "…"
    return text
