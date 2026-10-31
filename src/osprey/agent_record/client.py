"""The client of the agent-record read API.

:class:`RecordClient` calls ``GET <url>/v1/read/<method>`` on the agent-record
service, one method per :data:`~osprey.agent_record.service.READ_METHODS`
name, with the keyword arguments of the matching
:class:`~osprey.agent_record.reader.RecordReader` method plus three every read
takes:

``wait``
    A :class:`~osprey.agent_record.contract.Drain` (or a whole
    :class:`~osprey.agent_record.contract.WaitMark`): the service waits until
    ingest has committed what the caller's scope had pending. In a host
    launch's children (:data:`~osprey.agent_record.host.FORWARD_CONFIG_ENV`)
    a drained read first forwards the host's transcripts while the stack owns
    the record (:func:`~osprey.agent_record.host.forward_for_drain`), so what
    the host session wrote is pending when the drain is asked for.
``narrow_session`` / ``narrow_runs``
    A ``Source`` login's narrowing to one session or to a set of run ids.
    The service refuses either on any other login.

Two calls are not reads: :meth:`RecordClient.health_sources` answers the
service's ``GET /health/sources`` report as ``data`` (any login may ask), and
:meth:`RecordClient.export` runs ``POST /jobs/export`` and answers the days
exported, ISO dates oldest first, as ``data`` (the host and operator logins
only; any other is refused).

It authenticates with HTTP Basic ``<login>:<password>`` and, for the history
login only, names its viewer in
:data:`~osprey.agent_record.identities.VIEWER_HEADER`.

Every call answers a :class:`ClientAnswer` and never raises on a transport,
timeout, authentication or decode failure, nor on a refusal: each of those is
an answer with :attr:`ClientAnswer.unavailable` set to the reason. An answer
that is not :attr:`ClientAnswer.complete` may be missing entries; callers
render it as incomplete (or, when unavailable, as "record unavailable"), never
an empty answer as a complete one. The client retries nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import httpx

from osprey.agent_record.contract import EXPORT_PATH, HEALTH_SOURCES_PATH, Drain, WaitMark
from osprey.agent_record.credentials import LOGIN_ENV
from osprey.agent_record.identities import VIEWER_HEADER
from osprey.agent_record.reader import READ_SECONDS, WAIT_SECONDS

__all__ = [
    "DEFAULT_TIMEOUT",
    "READ_NAMES",
    "URL_ENV",
    "ClientAnswer",
    "RecordClient",
]

#: The variable naming the agent-record service's base URL.
URL_ENV = "OSPREY_AGENT_RECORD_URL"

#: Seconds a call waits for its answer. A drained read may wait
#: :data:`~osprey.agent_record.reader.WAIT_SECONDS` before it starts reading
#: for up to :data:`~osprey.agent_record.reader.READ_SECONDS`, so the default
#: outlasts both.
DEFAULT_TIMEOUT = WAIT_SECONDS + READ_SECONDS + 5.0

#: The read methods this client calls, named as the service's routes.
READ_NAMES = (
    "list_sessions",
    "session_users",
    "timeline",
    "agent_timeline",
    "full_entry",
    "run",
    "run_timeline",
    "recent_audit",
    "bridge_history",
    "entries",
)

_READ_PATH = "/v1/read/"


@dataclass(frozen=True)
class ClientAnswer:
    """What every :class:`RecordClient` call answers.

    ``unavailable`` is ``None`` when the service answered the read, and
    otherwise the reason it did not (``complete`` is then false and ``data``
    ``None``). ``complete`` says nothing the answer depends on is still
    missing. ``more`` says another page exists; a timeline page carries
    ``next``, the cursor of the following page (``None`` on the last), and
    ``newer``, true when entries arrived after its snapshot. ``reason`` names
    why a served answer is incomplete. ``data`` is the decoded JSON.
    """

    data: Any = field(default=None, hash=False)
    complete: bool = False
    unavailable: str | None = None
    next: str | None = None
    newer: bool = False
    more: bool = False
    reason: str = ""

    @classmethod
    def failed(cls, reason: str) -> ClientAnswer:
        """An unavailable answer naming *reason*."""
        return cls(unavailable=reason)

    @property
    def available(self) -> bool:
        return self.unavailable is None


def _json_object(response: httpx.Response) -> Mapping[str, Any] | ClientAnswer:
    """The JSON object *response* carries, or the failed answer naming why it has none."""
    try:
        body = response.json()
    except ValueError:
        return ClientAnswer.failed(f"agent-record answered {response.status_code} without JSON")
    if not isinstance(body, Mapping):
        return ClientAnswer.failed(f"agent-record answered {response.status_code} with no object")
    return body


def _decode(response: httpx.Response) -> ClientAnswer:
    body = _json_object(response)
    if isinstance(body, ClientAnswer):
        return body
    status = body.get("status")
    reason = body.get("reason")
    reason = reason if isinstance(reason, str) else ""
    if response.status_code != 200 or status != "ok":
        if status == "unavailable":
            return ClientAnswer.failed(reason or "the record is unavailable")
        detail = f": {reason}" if reason else ""
        return ClientAnswer.failed(
            f"agent-record refused the read ({response.status_code}){detail}"
        )
    complete, more, newer, cursor = (
        body.get("complete"),
        body.get("more"),
        body.get("newer", False),
        body.get("next"),
    )
    if (
        not isinstance(complete, bool)
        or not isinstance(more, bool)
        or not isinstance(newer, bool)
        or not (cursor is None or isinstance(cursor, str))
        or "data" not in body
    ):
        return ClientAnswer.failed("agent-record answered a body this client cannot read")
    return ClientAnswer(
        data=body["data"],
        complete=complete,
        next=cursor,
        newer=newer,
        more=more,
        reason=reason,
    )


def _decode_report(response: httpx.Response, key: str | None) -> ClientAnswer:
    """A ``/health/sources`` body (*key* ``None``: the whole object) or ``body[key]``."""
    body = _json_object(response)
    if isinstance(body, ClientAnswer):
        return body
    if response.status_code != 200 or (key is not None and body.get("status") != "ok"):
        reason = body.get("reason")
        detail = f": {reason}" if isinstance(reason, str) and reason else ""
        if body.get("status") == "unavailable":
            return ClientAnswer.failed(f"the record is unavailable{detail}")
        return ClientAnswer.failed(
            f"agent-record refused the call ({response.status_code}){detail}"
        )
    if key is None:
        return ClientAnswer(data=dict(body), complete=True)
    if key not in body:
        return ClientAnswer.failed("agent-record answered a body this client cannot read")
    return ClientAnswer(data=body[key], complete=True)


def _forward_host_transcripts() -> None:
    """In a host launch's children, forward the host's transcripts before a drain."""
    from osprey.agent_record.host import FORWARD_CONFIG_ENV, forward_for_drain

    config_path = os.environ.get(FORWARD_CONFIG_ENV)
    if config_path:
        forward_for_drain(config_path)


def _param(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class RecordClient:
    """A login's view of the agent record, over the read API.

    Args:
        url: The service's base URL, e.g. ``http://agent-record:4318``.
        login: The Basic login.
        password: The login's token.
        timeout: Seconds each call waits for its answer.
        viewer: The viewer the history login reads for (``all``,
            ``own:<person>`` or ``reader:<name>``), sent in
            :data:`~osprey.agent_record.identities.VIEWER_HEADER`. Any other
            login naming one is refused.
    """

    def __init__(
        self,
        url: str,
        login: str,
        password: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        viewer: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.login = login
        self.timeout = timeout
        self.viewer = viewer
        self._password = password
        self._unconfigured = ""

    def __repr__(self) -> str:
        return f"RecordClient({self.url!r}, login={self.login!r}, viewer={self.viewer!r})"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str],
        *,
        timeout: float = DEFAULT_TIMEOUT,
        viewer: str | None = None,
    ) -> RecordClient:
        """The client *env* configures: :data:`URL_ENV` and the login file's
        :data:`~osprey.agent_record.credentials.LOGIN_ENV` (``<login>:<token>``).

        Never raises: when either is missing or malformed, every call of the
        returned client answers unavailable, naming what is missing.
        """
        url = (env.get(URL_ENV) or "").strip()
        pair = (env.get(LOGIN_ENV) or "").strip()
        login, sep, token = pair.partition(":")
        problem = ""
        if not url:
            problem = f"{URL_ENV} is not set"
        elif not pair:
            problem = f"{LOGIN_ENV} is not set"
        elif not sep or not login or not token:
            problem = f"{LOGIN_ENV} is not <login>:<token>"
        if problem:
            login = token = ""
        client = cls(url, login, token, timeout=timeout, viewer=viewer)
        client._unconfigured = f"the record client is not configured: {problem}" if problem else ""
        return client

    # -- transport -----------------------------------------------------------

    def _read(
        self,
        name: str,
        args: Mapping[str, Any],
        wait: Drain | WaitMark | None,
        narrow_session: str | None,
        narrow_runs: Collection[str],
    ) -> ClientAnswer:
        params: list[tuple[str, str]] = [
            (key, _param(value)) for key, value in args.items() if value is not None
        ]
        if wait is not None:
            mark = wait if isinstance(wait, WaitMark) else WaitMark(drain=wait)
            if mark.drain is not None:
                _forward_host_transcripts()
            params.append(("wait", json.dumps(mark.to_json())))
        if narrow_session is not None:
            params.append(("narrow_session", narrow_session))
        params.extend(("narrow_run", run_id) for run_id in sorted(narrow_runs))
        response = self._send("GET", f"{_READ_PATH}{name}", params, self.timeout)
        return response if isinstance(response, ClientAnswer) else _decode(response)

    def _send(
        self, method: str, path: str, params: list[tuple[str, str]], timeout: float
    ) -> httpx.Response | ClientAnswer:
        """The service's response, or the unavailable answer naming why there is none."""
        if self._unconfigured:
            return ClientAnswer.failed(self._unconfigured)
        headers = {VIEWER_HEADER: self.viewer} if self.viewer is not None else {}
        try:
            return httpx.request(
                method,
                f"{self.url}{path}",
                params=tuple(params),
                headers=headers,
                auth=(self.login, self._password),
                timeout=timeout,
            )
        except httpx.TimeoutException:
            return ClientAnswer.failed(f"agent-record did not answer within {timeout:g} s")
        except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError) as exc:
            return ClientAnswer.failed(f"agent-record is unreachable: {type(exc).__name__}: {exc}")

    # -- health and the export -----------------------------------------------

    def health_sources(self) -> ClientAnswer:
        """The service's per-source, backend and export report, as ``data``."""
        response = self._send("GET", HEALTH_SOURCES_PATH, [], self.timeout)
        if isinstance(response, ClientAnswer):
            return response
        return _decode_report(response, None)

    def export(self, *, timeout: float | None = None) -> ClientAnswer:
        """Run the catch-up export now; ``data`` is the days exported, ISO dates
        oldest first. *timeout* overrides the client's for this call."""
        wait = self.timeout if timeout is None else timeout
        response = self._send("POST", EXPORT_PATH, [], wait)
        if isinstance(response, ClientAnswer):
            return response
        return _decode_report(response, "exported")

    # -- the read API --------------------------------------------------------

    def list_sessions(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """Every session the login sees, newest activity first."""
        return self._read(
            "list_sessions", {"since": since, "limit": limit}, wait, narrow_session, narrow_runs
        )

    def session_users(
        self,
        *,
        since: datetime | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """Every person with a session the login sees."""
        return self._read("session_users", {"since": since}, wait, narrow_session, narrow_runs)

    def timeline(
        self,
        session_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """One session's stream, a page at a time; *after* is the ``next`` of the page before."""
        args = {"session_id": session_id, "after": after, "limit": limit}
        return self._read("timeline", args, wait, narrow_session, narrow_runs)

    def agent_timeline(
        self,
        session_id: str,
        agent_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """One subagent's stream within *session_id*, paged like :meth:`timeline`."""
        args = {"session_id": session_id, "agent_id": agent_id, "after": after, "limit": limit}
        return self._read("agent_timeline", args, wait, narrow_session, narrow_runs)

    def full_entry(
        self,
        key: str,
        *,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """One entry with its whole body; ``data`` is ``None`` when the login sees none."""
        return self._read("full_entry", {"key": key}, wait, narrow_session, narrow_runs)

    def run(
        self,
        run_id: str,
        *,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """One run's summary."""
        return self._read("run", {"run_id": run_id}, wait, narrow_session, narrow_runs)

    def run_timeline(
        self,
        run_id: str,
        *,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """The timeline of the run's session."""
        return self._read("run_timeline", {"run_id": run_id}, wait, narrow_session, narrow_runs)

    def recent_audit(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        surface: str | None = None,
        written_by: Literal["self"] | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """The newest audit entries; ``written_by="self"`` keeps only the login's own."""
        args = {"since": since, "limit": limit, "surface": surface, "written_by": written_by}
        return self._read("recent_audit", args, wait, narrow_session, narrow_runs)

    def bridge_history(
        self,
        *,
        conversation: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """The newest bridge turns, each with its run."""
        args = {"conversation": conversation, "since": since, "limit": limit}
        return self._read("bridge_history", args, wait, narrow_session, narrow_runs)

    def entries(
        self,
        kind: str,
        since: datetime | None = None,
        limit: int | None = None,
        *,
        source: str | None = None,
        wait: Drain | WaitMark | None = None,
        narrow_session: str | None = None,
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        """The newest entries of one *kind*; *source* matches the body's ``source``
        before *limit* counts."""
        args = {"kind": kind, "since": since, "limit": limit, "source": source}
        return self._read("entries", args, wait, narrow_session, narrow_runs)
