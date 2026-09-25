"""HTTP the Teams bridge speaks: the bot's own credentials, and the Bot Connector.

The bridge makes two Bot Connector calls: it posts every reply, and it reads a
conversation's paged member list for the room roster. Both go to the same
per-conversation ``serviceUrl`` and both carry a bearer token the bot mints for itself — an OAuth2 *client-credentials*
exchange against Azure AD with the app registration's id and secret. There is no
user in that flow and no refresh token: the bridge simply asks again when the
token it holds is about to stop working, which is what :class:`TokenSource` is.

Two HTTP clients, not one
-------------------------
The token exchange and the Connector calls take **separate** injected
:class:`httpx.Client` seams (``token_http`` and ``connector_http``, wired in
``__main__``). They talk to different hosts under different failure policies: the
login host is one well-known endpoint per cloud, while the Connector host is
whatever ``serviceUrl`` the inbound activity named — an attacker-influenced value
in the general case, and one that varies per conversation. Folding them into a
single client would put the bot's credential exchange on the same connection
pool, timeouts and proxy settings as calls to a host the bridge was *told* about,
and would make it impossible to give the two legs different transports in a test
or a deployment. Keeping them apart costs one constructor argument.

Where the cloud comes from
--------------------------
The login host and the token scope are read off
:class:`~osprey.bridges.teams.config.TeamsBridgeConfig` — its ``login_host`` and
``token_scope`` properties resolve the ``TEAMS_CLOUD`` row. This module holds no
cloud table of its own: two tables would eventually disagree, and the failure
would look like an authentication problem rather than a configuration one.

Failure policy
--------------
Nothing here retries and nothing here swallows. The engine's crash-safety
ordering is built on the *ops* layer's per-member failure contracts (see
:mod:`osprey.bridges.core.ports`) — ``post_ack`` must swallow, ``post_answer``
must raise — and those are different policies over the same call, so only the
call site can choose. What this module guarantees instead is that every failure
arrives as one of its own exceptions, carrying the status code or the transport
error in its message, so the ops layer can route on the type and an operator can
diagnose from the container log alone.

Third-party imports
-------------------
``httpx`` only, and at module level: it is a core dependency, not part of the
optional ``teams`` extra. Nothing from ``azure`` or ``PIL`` may be imported here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from .config import TeamsBridgeConfig

logger = logging.getLogger(__name__)

TOKEN_URL_TEMPLATE = "https://{login_host}/{tenant}/oauth2/v2.0/token"
"""AAD's v2.0 client-credentials endpoint. The host varies by cloud and the tenant
segment by deployment; the path is the same everywhere."""

GRANT_TYPE = "client_credentials"
"""The only grant the bridge uses. There is no user in the flow — the bot
authenticates as itself — so there is no authorization code and no refresh token
to store."""

EXPIRY_MARGIN_SEC = 60.0
"""How far before a token's stated expiry it is treated as already expired.

A token handed out a second before the service stops honouring it fails *in
flight*, mid-answer, and the ops layer would have to tell that apart from a real
authorization problem. A minute is comfortably more than one Connector round trip
plus any clock skew between here and Azure, and costs one extra token fetch an
hour."""

TOKEN_TIMEOUT_SEC = 30.0
"""Timeout for the token exchange when this module builds its own client. The
injected ``token_http`` carries its own timeout; this is only the default."""

ACTIVITY_URL_TEMPLATE = "{service_url}/v3/conversations/{conversation_id}/activities/{activity_id}"
"""Where a reply goes. The host is never chosen here: ``service_url`` is the
``serviceUrl`` the inbound activity carried, and the two ids are the conversation
and the message being replied to — all three are used exactly as they arrived."""

MEMBERS_URL_TEMPLATE = "{service_url}/v3/conversations/{conversation_id}/pagedmembers"
"""Where a conversation's members are read, page by page. The host and the id are
used exactly as they arrived, as for :data:`ACTIVITY_URL_TEMPLATE`. The un-paged
``/members`` route is never used: Teams says not to in teams and channels, and a
chat answers its whole roster in one page of this one."""

MEMBERS_PAGE_MIN = 50
"""The smallest ``pageSize`` Teams documents for the paged member listing. A value
outside the documented bounds is refused or silently clamped by the service, so
the client clamps first."""

MEMBERS_PAGE_MAX = 500
"""The largest ``pageSize`` Teams documents for the paged member listing; see
:data:`MEMBERS_PAGE_MIN`."""

CONNECTOR_TIMEOUT_SEC = 30.0
"""Timeout for a Connector post when this module builds its own client. The
injected ``connector_http`` carries its own timeout; this is only the default.

Long enough to cover a slow round trip to a regional Connector host, short enough
that a wedged host cannot hold a worker thread — and therefore an answer's
remaining chunks — indefinitely."""

SIZE_REJECTED_STATUS = 413
"""The status the Connector answers for an activity over its size cap.

Named rather than inlined because it is the one status this module routes on: it
is the difference between "this answer is too long, split it" and "this answer
did not land", and the ops layer's re-split path keys on the exception it
produces."""

_ERROR_BODY_CHARS = 200
"""How much of a failed token response is quoted into the exception.

Enough for AAD's ``AADSTS`` code and its first sentence, which is what actually
names the misconfiguration; bounded because the body is attacker-influenced only
in the sense of being remote, and an unbounded string ends up in every log line.
The *request* body — which carries the client secret — is never quoted.

The Connector's error bodies are quoted to the same bound, for the same reason:
its ``error.code`` (``MessageSizeTooBig``, ``BotNotInConversationRoster``) leads
the body, and the rest is not worth a log line."""


class TokenError(RuntimeError):
    """The bot could not obtain a Bot Connector token.

    Covers every way the exchange can fail: a transport error reaching the login
    host, a non-2xx answer from it, and an answer that is not a usable token.
    They are one exception because the caller's options are the same in all three
    cases — it cannot post, and the message is what says why.

    Deliberately *not* a :class:`ConnectorError`. A missing secret or a wrong
    tenant is a deployment problem that every conversation shares, while a
    Connector failure is about one activity in one conversation; an ops member
    that retried or re-split on the former would be answering a configuration
    error with more traffic. Callers that genuinely treat the two alike catch
    both by name.
    """


class ConnectorError(RuntimeError):
    """A reply could not be posted to the Bot Connector.

    Covers a transport failure reaching the ``serviceUrl`` the activity named and
    any non-2xx answer from it. One exception for both because the caller's
    options are the same: the activity did not land, and the message — which
    carries the status or the transport error, and the URL that was attempted —
    is what says why.

    The one distinguishable case is :class:`MessageSizeTooBig`, which subclasses
    this so a caller that only cares whether the post landed needs one ``except``.
    A caller that re-splits on size must therefore catch the subclass *first*.
    """


class MessageSizeTooBig(ConnectorError):
    """The Connector refused one activity as too large (HTTP 413).

    Separate from its parent because it is the one failure the ops layer can do
    something about: the answer exceeded the activity size cap, so the same text
    posted in smaller pieces still lands. Nothing is re-split here — this class
    only makes the outcome routable, since a client that split its own payload
    would be guessing at a chunking policy that belongs to the layer holding the
    whole answer.
    """


def token_url(cfg: TeamsBridgeConfig) -> str:
    """The AAD token endpoint for this deployment's cloud and tenant.

    Args:
        cfg: Bridge config. Only ``login_host`` (resolved from ``TEAMS_CLOUD``)
            and ``tenant_id`` are read.

    Returns:
        The absolute URL to POST the client-credentials grant to.
    """
    return TOKEN_URL_TEMPLATE.format(login_host=cfg.login_host, tenant=cfg.tenant_id)


class TokenSource:
    """A Bot Connector bearer token, fetched on demand and cached until it expires.

    Thread-safe, and deliberately *coarsely* so: the whole of :meth:`token` runs
    under one lock, so a refresh happens exactly once no matter how many threads
    want a token at the moment the old one lapses. The engine drives the ops
    members from the retry-drain thread and the receive loop at the same time, and
    the alternative — a lock held only around the cache write — lets every one of
    those threads fire its own AAD exchange on the same expiry. Holding the lock
    across the request costs waiting callers one round trip they would otherwise
    have each made themselves.

    The cache is a token and the monotonic time it stops being usable. There is no
    background refresh and no timer thread: a bridge that is not posting does not
    need a token, and one that is will mint the next one on its next post.
    """

    def __init__(
        self,
        cfg: TeamsBridgeConfig,
        http: httpx.Client | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the token source to a config and an HTTP client.

        Args:
            cfg: Bridge config supplying the app registration and, through it, the
                cloud's login host and scope. Read on every fetch rather than
                copied, so a ``dataclasses.replace`` variant cannot go stale here.
            http: Client for the token leg — the injected ``token_http`` seam, or a
                test's :class:`httpx.MockTransport` client. ``None`` builds a
                default one honouring the core config's ``trust_env``.
            clock: Monotonic time source, injected so expiry is testable without
                sleeping. Monotonic rather than wall-clock on purpose: ``expires_in``
                is a duration, and a wall-clock step would otherwise expire a good
                token early or keep a dead one alive.
        """
        self._cfg = cfg
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=TOKEN_TIMEOUT_SEC, trust_env=cfg.core.trust_env)
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at = 0.0

    def token(self) -> str:
        """The current bearer token, fetching a new one if the cached one is stale.

        Returns:
            The raw token value, to be sent as ``Authorization: Bearer <token>``.

        Raises:
            TokenError: If the exchange failed or produced no usable token. The
                cache is left untouched, so the next call retries rather than
                serving a token that was never obtained.
        """
        with self._lock:
            now = self._clock()
            if self._token and now < self._expires_at:
                return self._token
            token, ttl = self._fetch()
            # Expiry is measured from before the request went out, so time spent
            # waiting for AAD counts against the token rather than being handed
            # back as usable life it does not have.
            self._token = token
            self._expires_at = now + ttl
            return token

    def _fetch(self) -> tuple[str, float]:
        """Run one client-credentials exchange.

        Returns:
            The token and how long it may be cached, in seconds — the stated
            ``expires_in`` less :data:`EXPIRY_MARGIN_SEC`, never negative. Zero
            means "usable now, cache it for nothing", which is what an answer with
            no stated expiry or one shorter than the margin gets.

        Raises:
            TokenError: On a transport failure, a non-2xx answer, a body that is
                not JSON, or a body with no ``access_token``.
        """
        url = token_url(self._cfg)
        try:
            response = self._http.post(
                url,
                data={
                    "grant_type": GRANT_TYPE,
                    "client_id": self._cfg.app_id,
                    "client_secret": self._cfg.app_secret,
                    "scope": self._cfg.token_scope,
                },
            )
        except httpx.HTTPError as exc:
            # Deliberately not re-raised as-is: the ops layer routes on this
            # module's exception types, and "could not reach the login host" and
            # "the login host said no" are the same outcome to it.
            raise TokenError(f"token request to {url} failed: {exc}") from exc

        if not response.is_success:
            raise TokenError(
                f"token endpoint answered HTTP {response.status_code}: "
                f"{response.text[:_ERROR_BODY_CHARS]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise TokenError(f"token endpoint answered a non-JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise TokenError("token endpoint answered a JSON value that is not an object")

        token = body.get("access_token")
        if not isinstance(token, str) or not token:
            raise TokenError("token response carried no access_token")

        return token, _cacheable_seconds(body.get("expires_in"))


def _cacheable_seconds(expires_in: object) -> float:
    """How long a token stating ``expires_in`` may be held, margin already taken off.

    Args:
        expires_in: The response field, which AAD sends as a number of seconds but
            which is only ever read back out of JSON — so a string, a missing key
            or something unparseable are all shapes this has to survive.

    Returns:
        Seconds to cache for, never negative. Zero for an absent or unusable
        value, and for a token that would already be inside the margin: a token
        with no known lifetime is still a usable token, and refetching on every
        post is slower than caching but never wrong.
    """
    if expires_in is None:
        logger.warning("token response carried no expires_in; not caching the token")
        return 0.0
    try:
        lifetime = float(expires_in)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("token response carried an unusable expires_in=%r; not caching", expires_in)
        return 0.0
    return max(0.0, lifetime - EXPIRY_MARGIN_SEC)


def activity_url(service_url: str, conversation_id: str, activity_id: str) -> str:
    """Where a reply to ``activity_id`` in ``conversation_id`` is posted.

    The ids are interpolated **as they arrived**. A channel reply's conversation
    id carries a ``;messageid=<root>`` suffix that roots the answer under the
    question, and the thread id itself contains ``:`` and ``@`` — all of which are
    legal in a URL path and all of which identify the conversation only while they
    are byte-for-byte what Teams sent. Normalising or re-encoding any of it posts
    the answer somewhere else, or nowhere.

    The only thing adjusted is ``service_url``'s trailing slash, which Teams does
    send and which would otherwise produce an empty path segment.

    Args:
        service_url: The ``serviceUrl`` the inbound activity carried. Regional and
            per-conversation, which is why it is an argument rather than config.
        conversation_id: The activity's ``conversation.id``.
        activity_id: The activity being replied to.

    Returns:
        The absolute URL to POST the reply activity to.
    """
    return ACTIVITY_URL_TEMPLATE.format(
        service_url=service_url.rstrip("/"),
        conversation_id=conversation_id,
        activity_id=activity_id,
    )


def members_url(service_url: str, conversation_id: str) -> str:
    """Where ``conversation_id``'s paged member list is read.

    The id is interpolated **as it arrived**, for the same reason as
    :func:`activity_url`; only ``service_url``'s trailing slash is adjusted.

    Args:
        service_url: The ``serviceUrl`` the inbound activity carried.
        conversation_id: The conversation whose members are listed.

    Returns:
        The absolute URL to GET the member pages from.
    """
    return MEMBERS_URL_TEMPLATE.format(
        service_url=service_url.rstrip("/"), conversation_id=conversation_id
    )


class ConnectorClient:
    """The two Bot Connector calls the bridge makes: post a reply activity, and read
    a conversation's members.

    Thread-safe, and coarsely so for the same reason as
    :class:`~osprey.bridges.google_chat.client.ChatClient`: the engine drives the
    ops members from the retry-drain thread and the receive loop at once, and both
    post through this one client. The HTTP leg of every call is serialized behind
    one lock, which costs nothing — an answer is a handful of activities posted in
    sequence anyway — and the alternative is several threads sharing one
    connection pool's sockets under a library whose per-connection state is not
    the bridge's to reason about.

    The bearer is fetched *before* that lock is taken. :class:`TokenSource` holds
    its own lock across its exchange, and nesting the two would mean one AAD round
    trip blocking every post rather than only the posts that needed the refresh.

    Nothing here retries, chunks, or swallows: the ops layer owns all three, and
    only it knows whether a given lost activity must raise (see
    :mod:`osprey.bridges.core.ports`).
    """

    def __init__(
        self,
        cfg: TeamsBridgeConfig,
        tokens: TokenSource,
        http: httpx.Client | None = None,
    ) -> None:
        """Wire the client to a credential source and an HTTP client.

        Args:
            cfg: Bridge config. Read only to build the default client, so an
                injected one makes it irrelevant to every call — the Connector host
                comes from the activity, never from config.
            tokens: The shared :class:`TokenSource`. Asked on every post rather
                than read once: it caches, so the cost is a lock and a comparison,
                and a client holding its own copy would post with a dead credential
                an hour into the process.
            http: Client for the Connector leg — the injected ``connector_http``
                seam, or a test's :class:`httpx.MockTransport` client. ``None``
                builds a default one honouring the core config's ``trust_env``.
                Separate from the token leg's client on purpose; see the module
                docstring.
        """
        self._tokens = tokens
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=CONNECTOR_TIMEOUT_SEC, trust_env=cfg.core.trust_env)
        )
        self._lock = threading.Lock()

    def reply(
        self,
        service_url: str,
        conversation_id: str,
        activity_id: str,
        activity: dict[str, Any],
    ) -> None:
        """Post one activity as a reply, or raise saying why it did not land.

        The activity is sent verbatim as the JSON body: what it says, and whether
        it carries attachments, is the ops layer's business. The Connector's answer
        is read only for its status — the created activity's id is of no use to a
        bridge that never edits or deletes what it posted.

        Args:
            service_url: The ``serviceUrl`` the inbound activity carried.
            conversation_id: The conversation to post into, exactly as sent.
            activity_id: The activity being replied to, exactly as sent.
            activity: The reply activity — a JSON-serialisable body.

        Raises:
            MessageSizeTooBig: If the Connector answered HTTP 413. The same text
                in smaller activities can still land, so the caller may re-split
                and post again.
            ConnectorError: On a transport failure or any other non-2xx answer.
            TokenError: If no bearer could be obtained. Nothing is posted in that
                case, so a caller that catches it has lost the activity and
                nothing else.
        """
        url = activity_url(service_url, conversation_id, activity_id)
        bearer = self._tokens.token()
        try:
            with self._lock:
                response = self._http.post(
                    url,
                    json=activity,
                    headers={"Authorization": f"Bearer {bearer}"},
                )
        except httpx.HTTPError as exc:
            # Deliberately not re-raised as-is: "could not reach the Connector"
            # and "the Connector said no" are the same outcome to the ops layer,
            # which routes on this module's exception types.
            raise ConnectorError(f"connector request to {url} failed: {exc}") from exc

        if response.status_code == SIZE_REJECTED_STATUS:
            raise MessageSizeTooBig(
                f"connector rejected the activity for {url} as too large "
                f"(HTTP {SIZE_REJECTED_STATUS}): {response.text[:_ERROR_BODY_CHARS]}"
            )
        if not response.is_success:
            raise ConnectorError(
                f"connector answered HTTP {response.status_code} for {url}: "
                f"{response.text[:_ERROR_BODY_CHARS]}"
            )

    def list_members(
        self, service_url: str, conversation_id: str, *, limit: int
    ) -> tuple[list[dict[str, Any]], bool]:
        """Read up to ``limit`` members of ``conversation_id``, page by page.

        Each page asks ``pageSize`` clamped to the documented bounds and, from the
        second page on, the ``continuationToken`` the previous page returned. A
        dict body gives ``members`` (only dict items are kept) and
        ``continuationToken``; a list body is one page of members with no token
        (the shape a chat may answer); any other JSON is an empty page. Stops when
        the token is absent, when a token repeats (a looping server), or when
        ``limit`` members are collected.

        Args:
            service_url: The ``serviceUrl`` the inbound activity carried.
            conversation_id: The conversation to list, exactly as it will be
                addressed.
            limit: The most members to return.

        Returns:
            ``(members, more)``: at most ``limit`` ``ChannelAccount`` objects, and
            whether the listing stopped on ``limit`` with more left to read.

        Raises:
            ConnectorError: On a transport failure, a non-2xx answer or a body
                that is not JSON.
            TokenError: If no bearer could be obtained.
        """
        url = members_url(service_url, conversation_id)
        page_size = max(MEMBERS_PAGE_MIN, min(limit, MEMBERS_PAGE_MAX))
        members: list[dict[str, Any]] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"pageSize": page_size}
            if token:
                params["continuationToken"] = token
            bearer = self._tokens.token()
            try:
                with self._lock:
                    response = self._http.get(
                        url, params=params, headers={"Authorization": f"Bearer {bearer}"}
                    )
            except httpx.HTTPError as exc:
                raise ConnectorError(f"connector request to {url} failed: {exc}") from exc
            if not response.is_success:
                raise ConnectorError(
                    f"connector answered HTTP {response.status_code} for {url}: "
                    f"{response.text[:_ERROR_BODY_CHARS]}"
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise ConnectorError(f"connector answered a non-JSON body for {url}") from exc

            items: Any = []
            next_token: Any = None
            if isinstance(body, dict):
                items = body.get("members")
                next_token = body.get("continuationToken")
            elif isinstance(body, list):
                items = body
            if isinstance(items, list):
                members.extend(item for item in items if isinstance(item, dict))
            token = next_token if isinstance(next_token, str) and next_token else None

            if len(members) >= limit:
                return members[:limit], len(members) > limit or token is not None
            if token is None or token in seen_tokens:
                return members, False
            seen_tokens.add(token)
