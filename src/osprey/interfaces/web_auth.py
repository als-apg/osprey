"""Process-wide credentials for the OSPREY web surfaces.

Every OSPREY web interface — the web terminal, the panel hosts, the companion
apps that share its process — is reachable over HTTP, and until this module
existed none of them asked the caller for anything. That made the agent's own
sandbox a foothold: code the agent ran could reach the very server that spawned
it and drive the operator's UI. This module holds the three credentials that
close that path and the one place they are decided, so that no surface invents
its own answer:

* the **operator secret**, which authorises everything an operator can do;
* the **panel token**, a deliberately weaker credential for the narrow set of
  panel-arrangement calls in-process companions legitimately make;
* a **browser-session map**, ``{digest of a session id: expiry}``, holding the
  sessions handed out when a browser exchanges a one-time URL token for a
  cookie — served from memory, and kept across a restart by
  :class:`SessionStore` where a deployment configured a store directory.

**Population happens once per process and is idempotent.** The first caller
builds the credentials under :data:`_POPULATION_LOCK`; every later caller — on
any thread, from any app in the process — gets the same object back. That is
what lets an in-process companion app inherit the parent's credentials instead
of minting a second set that the browser holding the first one would fail
against.

**The environment read is a ``pop``, not a ``get``, and that is load-bearing.**
The agent SDK layers its own environment *over* ``os.environ`` when it spawns a
child — it builds the CLI's environment as ``{**os.environ, **options.env}`` —
so a secret left in ``os.environ`` when the server starts serving is handed
straight to the sandboxed process this module exists to keep out, and a launch
helper that merely declines to *copy* the name changes nothing there. Popping
it means the value lives in this module's memory and is no longer inherited by
a child this process spawns. An empty or whitespace-only value counts as absent: an unset compose
variable interpolates to the empty string, and treating that as a credential
would authorise every caller who also sends nothing.

**Population runs once, but the carriers are closed on every launch.** A
launcher that is about to spawn or become the server re-publishes the operator
secret (:func:`mint_and_announce`) precisely so the child inherits it; in the
*direct-serve* shape — the default ``osprey web``, and the per-user container's
``CMD`` — that "child" is this same, already-populated process, where no second
:func:`_populate` and therefore no second pop would ever run.
:func:`close_env_carriers` is what closes the window again: every interface app
calls it at construction (see
:func:`osprey.interfaces._app_setup.configure_interface_app`), so whichever way
the process was started, neither carrier is in ``os.environ`` by the time it is
answering requests and spawning agents.

**What the pop does not reach.** A process that received the secret through
``execve`` — the ``--reload`` worker, the ``--detach`` child, the per-user
container — also keeps the exec-time environment block the kernel recorded for
it. On Linux that block is readable at ``/proc/<pid>/environ`` by any process
running as the same UID, which includes the agent this module is keeping the
secret from; ``os.environ.pop`` edits the interpreter's mapping and the
``environ(7)`` array, not that snapshot. So the pop closes the *inheritance*
path — the one an SDK overlay travels — and does not make the value unreadable
to a same-UID reader on those shapes. Closing that residual needs either a
privilege split (a container user the agent cannot become) or a handoff that
never touches the environment at all (an inherited file descriptor, a 0600
file); both are deliberately out of scope here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BIND_HOST_ENV",
    "DEFAULT_SESSION_LIFETIME",
    "OPERATOR_SECRET_ENV",
    "ROSTER_ACCEPT_ENV",
    "ROSTER_SECRET_ENV_PREFIX",
    "PANEL_REGISTER_ROUTE",
    "PANEL_TIER_ROUTES",
    "PANEL_TOKEN_ENV",
    "SESSION_LIFETIME_ENV",
    "SESSION_STORE_DIR_ENV",
    "SessionStore",
    "Tier",
    "WebCredentials",
    "classify",
    "close_env_carriers",
    "get_web_credentials",
    "mint_and_announce",
    "mint_secret",
    "peek_web_credentials",
    "reset_web_credentials",
]

#: The operator secret's environment carrier. In the container shape the deploy
#: ``.env`` supplies it per user and nginx forwards the same value as a header;
#: in the single-user shape nothing sets it and this module mints one.
OPERATOR_SECRET_ENV = "OSPREY_TERMINAL_SECRET"

#: Prefix of the per-user operator secrets a multi-user deployment mints
#: (``OSPREY_TERMINAL_SECRET_<USER>``). Each per-user container receives its
#: own under the fixed name above; a shared sidecar that per-user terminals
#: proxy into — the bluesky-web panel — is handed every entitled user's
#: variable under its own name — with :data:`ROSTER_ACCEPT_ENV` beside them —
#: and accepts any of them, so a persona presenting its own secret is let in
#: without the deployment-wide one ever leaving the deploy ``.env``. The one
#: definition of the prefix: the mint
#: (:data:`osprey.deployment.web_terminals.auth_credentials.TERMINAL_SECRET_VAR_PREFIX`)
#: and nginx's ``NGINX_ENVSUBST_FILTER`` spell it from here.
ROSTER_SECRET_ENV_PREFIX = f"{OPERATOR_SECRET_ENV}_"

#: The switch that lets a process ACCEPT the roster secrets it finds. Set to
#: ``"1"`` in exactly one place: the bluesky-web sidecar's compose environment
#: (``services/bluesky_web/docker-compose.yml``), beside the roster variables
#: it lists. Every roster variable is popped from the environment wherever it
#: is found — the deploy ``.env`` carries them all, and ``osprey web`` on the
#: host loads that file — but only a process whose compose declared this flag
#: treats them as operators. Without it the host's own terminal, loading the
#: same ``.env``, would let any roster user in as its operator.
ROSTER_ACCEPT_ENV = "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"

#: The panel token's environment carrier.
PANEL_TOKEN_ENV = "OSPREY_PANEL_TOKEN"

#: The multi-user shape's tell, read (never popped — ``osprey web`` reads it
#: too) exactly as :func:`osprey.cli.web_cmd.resolve_bind_host` reads it: any
#: truthy value means "a deployment declared this, nginx owns the perimeter".
BIND_HOST_ENV = "OSPREY_TERMINAL_BIND_HOST"

#: How long a browser session stays valid when nothing configures it — the
#: default behind ``modules.web_terminals.auth.session_lifetime``, which is
#: honoured in both the single-user and the multi-user shape. Twelve hours
#: outlives a working day at the console without leaving a forgotten tab
#: authorised indefinitely. The cookie carrying the session is persistent: it
#: stamps this lifetime as ``Max-Age``, so it survives a browser restart and
#: expires on its own schedule rather than on the browser's. Defined exactly
#: once — :mod:`osprey.deployment.web_terminals.render` and the auth sidecar
#: import this name rather than repeating the number.
DEFAULT_SESSION_LIFETIME = 12 * 60 * 60

#: The environment carrier for ``modules.web_terminals.auth.session_lifetime``,
#: read (never popped) for the reason :data:`BIND_HOST_ENV` is: a duration is
#: not a credential, and more than one reader needs it. ``osprey web`` resolves
#: env > config > default and publishes the answer here for the server it is
#: about to spawn or become, and the multi-user compose sets the same value on
#: every terminal container. Absent or blank means nothing configured the
#: lifetime and :data:`DEFAULT_SESSION_LIFETIME` applies; anything else must be
#: a positive whole number of seconds or the process refuses to start.
SESSION_LIFETIME_ENV = "OSPREY_TERMINAL_SESSION_LIFETIME"

#: Where the browser-session store keeps its file, read (never popped) for the
#: reason :data:`SESSION_LIFETIME_ENV` is: a directory path is not a credential.
#: Absent or blank means no store at all, which is the deliberate default for
#: any process that has no agent-data directory to write into — the sessions
#: then live and die with the process, exactly as they did before the store
#: existed. Set by ``osprey web`` to ``<agent_data.base_dir>/web_terminal``,
#: whose contents the deployment already treats as the terminal's own state.
#: The per-user compose does not carry this variable and does not need to: a
#: terminal container runs ``osprey web`` as its command, so the same launcher
#: resolves the same path there from the image's baked-in ``agent_data``.
SESSION_STORE_DIR_ENV = "OSPREY_TERMINAL_SESSION_STORE_DIR"

#: Compared against when a session lookup misses, purely so a miss costs the
#: same comparison work as a hit — it is a cost-equalizer, never a credential,
#: and :meth:`WebCredentials.verify_session` gates its answer on the map lookup
#: so that a candidate whose digest equalled this value would still refuse.
#: Shaped like a digest — 64 lowercase hex characters — because a digest is
#: what the comparison now weighs: sized or spelled like anything else, the
#: miss path would cost visibly less than the hit path, which is the one thing
#: the decoy exists to prevent. Nothing has to *be* a preimage of it; only its
#: shape matters.
_SESSION_DECOY = "0" * 64


def mint_secret() -> str:
    """Mint a fresh credential.

    The same recipe as :func:`osprey.deployment.service_tokens._default_token`,
    deliberately: 256 bits drawn from :mod:`secrets`, URL-safe so the value
    survives being pasted into a query string by ``mint_and_announce`` and into
    a ``.env`` by the deploy path without escaping.
    """
    return secrets.token_urlsafe(32)


def _digest(session_id: str) -> str:
    """Return the session map's key for *session_id*: its sha256, lowercase hex.

    The one place the mapping is spelled, so the minting side, the verifying
    side and the on-disk store cannot drift into keying by different things.
    This is a plain digest with no salt and no stretching deliberately: the
    input is 256 bits from :mod:`secrets`, so there is no dictionary to defend
    against, and :meth:`WebCredentials.verify_session` runs this on every
    cookie-bearing request.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


class SessionStore:
    """The on-disk half of the browser-session map.

    A browser session lives in :class:`WebCredentials.sessions` for the life of
    one process, which means every restart of the terminal — a config reload, a
    container roll, a crash — logs out every operator holding a valid cookie.
    This class is what survives that: it keeps the map in a small JSON file
    under the deployment's agent-data directory, one file per port so two
    terminals on one host never share a store.

    **What is persisted authenticates nobody.** The file holds the sha256
    *digest* of each session id against its wall-clock expiry —
    ``{"v": 1, "sessions": {digest: epoch}}`` — never the id a browser sends.
    A leaked store is therefore a list of deadlines: an attacker who reads it
    still has to produce a 256-bit id whose digest is in it. Deadlines are
    wall-clock epoch seconds — the same clock and the same numbers
    :attr:`WebCredentials.sessions` holds, so neither side converts anything: a
    monotonic reading would be meaningless to the process that loads it.

    **Reading never raises, and a failed write is a warning, not an error.**
    Sessions are a convenience layered over credentials that live in memory, so
    a store that is missing, truncated, owned by another user or on a full disk
    must cost an operator nothing more than a re-login. A login refused because
    a disk filled up would be a worse failure than the one being avoided. A
    missing file is not even that much: a fresh deployment has no store, and
    that is the ordinary first start, so it is silent. Every other failure logs
    exactly one ``WARNING`` per process — naming the path and the reason — and
    is then silenced, because the caller goes on calling :meth:`save` on every
    login and logout and an unwritable directory would otherwise fill the log
    with the same line. Silenced does not mean given up on: each later call
    still tries, so a disk that frees up starts persisting again on its own.

    **Why ``save`` takes a sequence number.** The credentials snapshot their
    session map under their own lock and then write it *outside* that lock, so
    that a slow disk never blocks a request holding the map. Two threads can
    therefore reach this class in the opposite order to the one they snapshotted
    in, and the older snapshot would land last and resurrect a session that was
    just revoked. The sequence number is stamped at snapshot time and strictly
    increases, so :meth:`save` can drop anything that is not newer than what it
    last accepted.
    """

    def __init__(self, store_dir: Path, port: str) -> None:
        """Resolve the store's path and make sure its directory exists.

        Args:
            store_dir: The directory the store file lives in — in serving use
                ``<agent_data.base_dir>/web_terminal``.
            port: The port this terminal serves on, which names the file
                (``sessions-8080.json``) so two terminals sharing a host do not
                overwrite each other's sessions. Empty — a caller with no port
                to distinguish — gives the bare ``sessions.json``.

        A directory that cannot be created is the first write failure rather
        than a construction error: the process must still serve, memory-only.
        """
        self._dir = Path(store_dir)
        self._path = self._dir / (f"sessions-{port}.json" if port else "sessions.json")
        self._lock = threading.Lock()
        self._warned = False
        self._last_seq: int | None = None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._warn(f"cannot create its directory: {exc}")

    @property
    def path(self) -> Path:
        """The store file this instance reads and writes."""
        return self._path

    def load(self) -> dict[str, float]:
        """Return the persisted ``{digest: epoch}`` map, or ``{}``.

        Returns ``{}`` — never raises — for every reason the file might not be
        usable: it does not exist yet, it cannot be read, it does not decode as
        UTF-8, it is not JSON, it carries a version this code does not know, or
        its shape is wrong anywhere (a non-object payload, a non-object
        ``sessions``, a key that is not a string, a deadline that is not a
        number). A partially valid file is treated as no file rather than
        salvaged: a store that has been corrupted has no claim to the entries
        that happen to still parse, and the cost of discarding them is one
        re-login.

        The missing-file case is silent. Every other case warns once — see the
        class docstring.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # A fresh deployment has no store. That is the ordinary first
            # start, not a failure, so it must not warn.
            return {}
        except (OSError, ValueError) as exc:
            # ``ValueError`` is here for ``UnicodeDecodeError``, which is one
            # and is NOT an ``OSError``: a store file holding bytes that are not
            # UTF-8 — a truncated write, a file another tool clobbered — would
            # otherwise raise out of the read and, through _populate, refuse
            # every request the process would ever serve. A store is never
            # allowed to cost more than a re-login.
            self._warn(f"cannot be read: {exc}")
            return {}

        try:
            payload = json.loads(raw)
        except ValueError as exc:
            self._warn(f"is not valid JSON: {exc}")
            return {}

        if not isinstance(payload, dict):
            self._warn(f"holds {type(payload).__name__}, not an object")
            return {}
        if payload.get("v") != 1:
            self._warn(f"has unsupported version {payload.get('v')!r}, expected 1")
            return {}

        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            self._warn(f"has a {type(sessions).__name__} 'sessions' field, not an object")
            return {}

        loaded: dict[str, float] = {}
        for digest, deadline in sessions.items():
            # ``bool`` is an ``int`` subclass; ``True`` is not a deadline.
            if (
                not isinstance(digest, str)
                or isinstance(deadline, bool)
                or not isinstance(deadline, int | float)
            ):
                self._warn("holds a malformed session entry")
                return {}
            loaded[digest] = float(deadline)
        return loaded

    def save(self, snapshot: dict[str, float], seq: int) -> None:
        """Persist *snapshot* if *seq* is newer than the last accepted write.

        Args:
            snapshot: The ``{digest: epoch}`` map to write, already copied by
                the caller — this method does not lock the caller's map and
                must not be handed one that is still being mutated.
            seq: A strictly increasing stamp taken when the snapshot was made.
                A value that is not above the last one accepted is dropped
                silently: it describes a state older than what is already on
                disk, and writing it would undo a newer save (see the class
                docstring).

        Never raises. The write is a temporary file in the store's own
        directory, fsynced, chmodded ``0600`` and then :func:`os.replace`d over
        the target, so a reader never sees a half-written store and a crash
        mid-write leaves the previous one intact. A failure warns once and
        leaves the sessions valid in memory.
        """
        with self._lock:
            if self._last_seq is not None and seq <= self._last_seq:
                return
            # Recorded before the attempt, not after it: a failed write must
            # still shut out the older snapshots it was racing, or a retry
            # would let a stale one through.
            self._last_seq = seq
            payload = {"v": 1, "sessions": dict(snapshot)}
            try:
                self._write_atomic(payload)
            except OSError as exc:
                self._warn(f"cannot be written: {exc}")

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        """Write *payload* over :attr:`path`, atomically and at mode ``0600``.

        The same idiom as
        :func:`osprey.interfaces.web_terminal.feedback_store._atomic_write`. The
        explicit ``chmod`` is redundant with :func:`tempfile.mkstemp`, which
        already creates at ``0600``; it is here so the mode the store must have
        is stated where the file is created rather than inherited from another
        module's default.
        """
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                with suppress(OSError):
                    os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_name)
            raise

    def _warn(self, reason: str) -> None:
        """Log the first store failure of this process and silence the rest."""
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "Browser sessions will not persist across restarts: the session store %s "
            "%s. Sessions remain valid in memory; operators keep their cookies until "
            "this process stops.",
            self._path,
            reason,
        )


@dataclass(eq=False)
class WebCredentials:
    """The credentials one OSPREY process serves its web surfaces with.

    Ordinarily obtained through :func:`get_web_credentials` rather than
    constructed: the module-level holder is what makes the credentials the
    *process's*, shared by every app in it. Constructing one directly is for
    tests that want a known secret without touching the environment.

    The verification methods are what request handling calls, so none of them
    touch the disk or the network. The two credential comparisons are
    constant-time; the session lookup is not, for the reason
    :meth:`verify_session` gives.

    **The session map holds nothing that authenticates on its own.** Its keys
    are digests, not ids, and :meth:`verify_session` digests the cookie before
    it looks — so a key lifted out of this map (or out of the file
    :class:`SessionStore` writes it to) and presented as a cookie is refused
    like any other guess.
    """

    operator_secret: str
    panel_token: str

    #: Other secrets that authorise as the operator: the per-user secrets of a
    #: multi-user roster, handed to a shared sidecar whose environment also
    #: sets :data:`ROSTER_ACCEPT_ENV` (see :data:`ROSTER_SECRET_ENV_PREFIX`).
    #: Empty everywhere else — including a process that merely SEES roster
    #: variables (the host's ``osprey web`` loading the deploy ``.env``).
    roster_secrets: tuple[str, ...] = ()

    #: How long a session this holder mints stays valid, in seconds — what
    #: :meth:`create_session` uses when it is not told otherwise, and what the
    #: exchange cookie's ``Max-Age`` carries, so a browser's copy of the session
    #: and the process's copy expire together. Comes from
    #: ``modules.web_terminals.auth.session_lifetime`` by way of
    #: :data:`SESSION_LIFETIME_ENV`, and is :data:`DEFAULT_SESSION_LIFETIME`
    #: when nothing configured it.
    session_ttl_seconds: int = DEFAULT_SESSION_LIFETIME

    #: Live browser sessions, ``{sha256 digest of the id: wall-clock deadline}``.
    #: Keyed by digest so that neither this map nor the file
    #: :class:`SessionStore` writes it to ever holds a value that would
    #: authenticate: :meth:`verify_session` digests the cookie again before it
    #: looks, so a leaked key is a deadline and nothing more.
    #:
    #: Deadlines are :func:`time.time` rather than :func:`time.monotonic`
    #: because they are meant to outlive the process that set them — the store
    #: persists them across a restart, and a monotonic reading taken by one
    #: process says nothing to the next. The cost is that a clock
    #: adjustment — an NTP step, a laptop resuming — moves every live deadline
    #: with it. That is accepted rather than solved: it is the same exposure the
    #: auth sidecar's own signed session cookies already carry, and while the
    #: process runs it is bounded by :attr:`session_ttl_seconds`, since every
    #: session a serving caller mints is one lifetime out.
    sessions: dict[str, float] = field(default_factory=dict)

    #: Digests of the sessions minted with ``persist=False`` — the ones that
    #: must stay in this process and never reach the store. The browser-context
    #: session an interface mints for its own page loads is the case this exists
    #: for — it is scoped to a running server, so persisting it would outlive the
    #: thing it belongs to for no gain. Holding digests rather than ids keeps the
    #: same property the map has — this set authenticates nobody either.
    _ephemeral: set[str] = field(default_factory=set, repr=False)

    #: Where this holder's sessions outlive the process, or ``None`` when the
    #: deployment configured no store directory (see
    #: :data:`SESSION_STORE_DIR_ENV`). Optional rather than always present
    #: because persistence is a convenience, not part of the credential
    #: contract: every other path here has to behave identically whether or not
    #: a disk is involved, and holding ``None`` is what makes that literal —
    #: the no-store process runs the same code with the writes skipped.
    store: SessionStore | None = None

    #: The stamp taken with each snapshot, so the store can tell a newer one
    #: from an older. Snapshots are written *outside* the lock that made them,
    #: so two threads can reach :meth:`SessionStore.save` in the order opposite
    #: to the one they snapshotted in, and a stale write would resurrect a
    #: session that was just revoked. Counting here rather than timing the write
    #: is what makes the ordering total: two saves in the same clock tick would
    #: be indistinguishable by time.
    _store_seq: int = field(default=0, repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __repr__(self) -> str:
        """Describe the holder without printing either secret.

        The default dataclass ``__repr__`` would put both credentials into
        every log line, traceback frame and debugger view that touches this
        object — which is the one place a credential popped out of the
        environment must not end up.
        """
        return f"WebCredentials(sessions={len(self.sessions)})"

    def verify_operator(self, candidate: str | None) -> bool:
        """Whether ``candidate`` is the operator secret — or one of the roster's
        — each compared in constant time.

        Every accepted secret is compared, never just until the first match,
        so the answer's timing does not say which one matched.
        """
        matched = _same_secret(candidate, self.operator_secret)
        for secret in self.roster_secrets:
            matched = _same_secret(candidate, secret) or matched
        return matched

    def verify_panel(self, candidate: str | None) -> bool:
        """Whether ``candidate`` is the panel token, compared in constant time.

        A true answer authorises only the panel-arrangement tier, never an
        operator action; the route table decides which tier a request needs and
        this method never widens it.
        """
        return _same_secret(candidate, self.panel_token)

    def create_session(self, ttl_seconds: float | None = None, *, persist: bool = True) -> str:
        """Mint a browser session id, record its expiry, and return it.

        Expired entries are dropped here rather than on a timer: sessions are
        only ever created by a browser exchanging a token, so this is the one
        call whose frequency tracks the map's growth, and it means the process
        carries no reaper thread for a dictionary that holds a handful of keys.

        Args:
            ttl_seconds: An explicit lifetime for this one session. Left
                ``None`` — which is how every serving caller calls it — the
                session lasts :attr:`session_ttl_seconds`, so the deployment's
                one configured value governs every session the process hands
                out. Passing a value is for tests that need a deadline they
                control, notably one already in the past.
            persist: Whether this session may be written to the on-disk store.
                ``False`` records its digest in :attr:`_ephemeral`; the snapshot
                handed to :class:`SessionStore` skips those digests, so such a
                session never reaches the disk. The returned id is otherwise an
                ordinary one — it verifies and revokes like any other.

        The id is returned; only its digest is kept, so this is the last moment
        the process holds the value a browser will send back.
        """
        ttl = self.session_ttl_seconds if ttl_seconds is None else ttl_seconds
        session_id = mint_secret()
        digest = _digest(session_id)
        store = self.store
        snapshot: dict[str, float] | None = None
        seq = 0
        with self._lock:
            now = time.time()
            self._purge_expired(now)
            self.sessions[digest] = now + ttl
            if not persist:
                self._ephemeral.add(digest)
            if store is not None:
                snapshot, seq = self._snapshot()
        if snapshot is not None and store is not None:
            # Outside the lock: the write is a disk round-trip, and a request
            # verifying a cookie must not queue behind it.
            store.save(snapshot, seq)
        return session_id

    def revoke_session(self, session_id: str) -> bool:
        """Drop one browser session, returning whether it had been live.

        Logging out has to invalidate the cookie server-side as well as clear
        it in the browser: a cookie value that has already left the process
        cannot be un-sent, so the only thing that can refuse it afterwards is
        the map here no longer holding it. The store is rewritten for the same
        reason: a revocation that lived only in memory would be undone by the
        next restart, which is precisely when the operator has stopped watching.
        A revocation that matched nothing writes nothing: unlike
        :meth:`create_session` this method purges no expiries, so there is no
        other change to propagate, and a logout that has to try each of the
        browser's cookie candidates would otherwise cost one full store rewrite
        per candidate to record a map that never moved.
        """
        digest = _digest(session_id)
        store = self.store
        snapshot: dict[str, float] | None = None
        seq = 0
        with self._lock:
            self._ephemeral.discard(digest)
            was_live = self.sessions.pop(digest, None) is not None
            if store is not None and was_live:
                snapshot, seq = self._snapshot()
        # ``is not None`` rather than truthiness: revoking the last session
        # leaves an EMPTY snapshot, and that is precisely the one that has to
        # reach the disk — it is what clears the file.
        if snapshot is not None and store is not None:
            store.save(snapshot, seq)
        return was_live

    def verify_session(self, session_id: str | None) -> bool:
        """Whether ``session_id`` names a live browser session.

        One sha256, one dictionary lookup and one
        :func:`secrets.compare_digest` per call, with no I/O — this runs on
        every cookie-bearing request, so anything that read a file or opened a
        socket would land on the hot path of every page load and every
        websocket frame. The candidate is digested exactly once, before the
        lookup, and the digest is what both the lookup and the comparison see.

        **The map lookup is what decides the answer.** The comparison below
        cannot: on a hit it weighs the candidate against itself, so it is true
        by construction. Its job is the other half — it runs on the miss path
        too, against a fixed-length decoy, so that a rejected id costs the same
        visible work as an accepted one and the timing of a refusal says
        nothing about how much of a guess was right. The decoy is never
        allowed to *be* the answer: ``hit`` gates the return, because a
        candidate that happens to equal the decoy would otherwise compare equal
        to it and authenticate against an empty map.

        **Digesting first is also what keeps a hostile cookie a refusal rather
        than a 500.** ``compare_digest`` raises ``TypeError`` on a non-ASCII
        ``str``, and a cookie is unauthenticated input; what reaches it here is
        the output of :func:`_digest`, which is lowercase hex whatever the
        caller sent, so the candidate's encoding can no longer decide whether
        the call raises.

        **This is not constant time, and that is accepted rather than
        overlooked.** ``digest in self.sessions`` hashes the digest and, on a
        bucket hit, falls back to ``str.__eq__``, which stops at the first
        differing character; the ``compare_digest`` above equalises the cost of
        the *comparison* but cannot equalise the cost of the *lookup*. What
        that timing could reveal is that the candidate's digest shares a prefix
        with the digest of a live session — a fact about a value the attacker
        computed themselves from a guess they had already made, not about any
        id this process holds. It does not narrow the 256-bit id behind that
        digest and does not carry to the next guess, and it is strictly less
        than a map keyed by raw ids leaked, where the same timing spoke about a
        prefix of the live id itself. The credentials an attacker would
        actually target — the operator secret and the panel token — are
        compared by :func:`_same_secret`, which is constant time.
        """
        if not session_id:
            return False
        digest = _digest(session_id)
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            hit = digest in self.sessions
            canonical = digest if hit else _SESSION_DECOY
            matched = secrets.compare_digest(digest, canonical)
        return hit and matched

    def _snapshot(self) -> tuple[dict[str, float], int]:
        """Copy the persistable sessions and stamp the copy. Caller holds ``_lock``.

        A copy rather than the map itself, because what comes back is written
        after the lock is released and :class:`SessionStore` must never be
        handed a dictionary another thread is still mutating. The comprehension
        is also where :attr:`_ephemeral` is applied: a session minted with
        ``persist=False`` is scoped to this process and has no business
        outliving it.
        """
        snapshot = {
            digest: deadline
            for digest, deadline in self.sessions.items()
            if digest not in self._ephemeral
        }
        self._store_seq += 1
        return snapshot, self._store_seq

    def _purge_expired(self, now: float) -> None:
        """Drop every session whose deadline has passed. Caller holds ``_lock``.

        A purged digest leaves :attr:`_ephemeral` with it, so the set tracks the
        map rather than accumulating the digests of sessions that are long gone.
        """
        expired = [digest for digest, deadline in self.sessions.items() if deadline <= now]
        for digest in expired:
            del self.sessions[digest]
            self._ephemeral.discard(digest)


def _same_secret(candidate: str | None, expected: str) -> bool:
    """Compare a caller-supplied credential against the held one in constant time.

    UTF-8 bytes rather than ``str``, because the candidate arrives in a header
    an attacker writes and ``secrets.compare_digest`` raises ``TypeError`` on a
    non-ASCII ``str``: one accented character would turn a refusal into a 500.
    (:meth:`WebCredentials.verify_session` reaches the same end differently — it
    compares digests, which are ASCII whatever the cookie held.)
    """
    if not candidate:
        return False
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


_POPULATION_LOCK = threading.Lock()
_CREDENTIALS: WebCredentials | None = None


def _populate() -> WebCredentials:
    """Build this process's credentials from the environment, minting what is absent.

    Raises:
        RuntimeError: in the container shape, when no operator secret was
            supplied. Minting one there would be worse than failing: nginx
            forwards the value the deploy ``.env`` pinned, so a locally minted
            secret would never match the header the reverse proxy sends and
            every request would be refused with no indication why. Also when
            the configured session lifetime is unreadable — see
            :func:`_session_ttl_from_env`.
    """
    # Both carriers are popped BEFORE the check below, because the check can
    # raise and the process that catches it goes on serving and spawning
    # children. A panel token left in ``os.environ`` at that point is exactly
    # the inheritance this module exists to prevent, so the fatal path must not
    # be the one path that leaks it.
    operator_secret = os.environ.pop(OPERATOR_SECRET_ENV, "").strip()
    supplied_panel_token = os.environ.pop(PANEL_TOKEN_ENV, "").strip()
    # Popped unconditionally (they must not reach any child process); kept
    # only where the compose environment said so — see ROSTER_ACCEPT_ENV.
    harvested = _pop_roster_secrets()
    roster_secrets = harvested if _roster_accepted() else ()

    if not operator_secret:
        if os.environ.get(BIND_HOST_ENV):
            raise RuntimeError(
                f"{OPERATOR_SECRET_ENV} is empty or unset, but {BIND_HOST_ENV} is declared, "
                "so this process is running behind the multi-user reverse proxy. The secret "
                f"must be supplied by the deployment: set {OPERATOR_SECRET_ENV} for this "
                "service in the compose environment (the deploy .env holds the per-user "
                "value). Refusing to mint one here, because nginx forwards the deployment's "
                "value and a locally minted secret would refuse every request."
            )
        operator_secret = mint_secret()

    # Read after the pops above, deliberately: this call can raise too, and the
    # carriers must already be out of the environment when it does.
    ttl_seconds = _session_ttl_from_env()
    store = _session_store_from_env()

    return WebCredentials(
        operator_secret=operator_secret,
        panel_token=supplied_panel_token or mint_secret(),
        roster_secrets=roster_secrets,
        session_ttl_seconds=ttl_seconds,
        sessions=_restore_sessions(store, ttl_seconds),
        store=store,
    )


def _session_store_from_env() -> SessionStore | None:
    """Build this process's session store, or ``None`` if none was configured.

    A blank carrier is no store rather than a store in the current directory:
    an unset compose variable interpolates to the empty string, and writing the
    deployment's session file into whatever directory the process happened to
    start in would be worse than not persisting at all.
    """
    store_dir = os.environ.get(SESSION_STORE_DIR_ENV, "").strip()
    if not store_dir:
        return None
    return SessionStore(Path(store_dir), _web_port())


def _web_port() -> str:
    """Return the settled port the store file is named for, or ``""``.

    The same derivation as
    :func:`osprey.interfaces.common_middleware.session_cookie_name` — a
    non-numeric value names nothing and falls back to the bare file — because
    the two must agree: a browser holding a cookie named for one port has to
    find its session in the store named for the same one.

    The variable's name is spelled out here rather than imported from
    :mod:`osprey.interfaces.common_middleware`, which is where it is defined:
    that module imports this one at its top, so importing back would close a
    cycle. A duplicated literal is the cheaper of the two.
    """
    text = os.environ.get("OSPREY_WEB_PORT", "").strip()
    return text if text.isdigit() else ""


def _restore_sessions(store: SessionStore | None, ttl_seconds: int) -> dict[str, float]:
    """Read the persisted sessions back, clamped to the configured lifetime.

    Nothing is written here. Population happens before the process serves
    anything, and a restart that failed to reach the point of serving must not
    have already replaced the store it read — a crash loop would otherwise
    empty it. The first write is the first login or logout.

    Every restored deadline is clamped to ``now + ttl_seconds`` because the
    restart may be the one that *shortened* the lifetime: an operator who cuts
    ``modules.web_terminals.auth.session_lifetime`` and restarts has said what
    the longest session may now be, and a deadline written under the old value
    must not outlive it. Sessions already past their deadline are dropped
    rather than restored — the map only ever holds live entries, and
    :meth:`WebCredentials._purge_expired` would drop them on the first call
    anyway.

    **The read is guarded here as well as inside the store, deliberately.**
    :meth:`SessionStore.load` already promises never to raise, and this is the
    call site that turns a broken promise into a process that answers nothing:
    the exception would leave :func:`_populate` and, since a failed population
    is not cached, be raised again on every request for the life of the
    process. Two layers cost one ``try`` and mean no future change to ``load``
    can lock an operator out of the console over a file that only holds
    convenience.
    """
    if store is None:
        return {}
    try:
        persisted = store.load()
    except Exception as exc:
        store._warn(f"could not be read: {exc}")
        return {}
    now = time.time()
    ceiling = now + ttl_seconds
    restored: dict[str, float] = {}
    for digest, deadline in persisted.items():
        clamped = min(deadline, ceiling)
        if clamped > now:
            restored[digest] = clamped
    return restored


def _session_ttl_from_env() -> int:
    """Read the configured session lifetime in seconds, or refuse to start.

    The only environment reader here that neither pops nor falls back on a bad
    value. It does not pop because the value is a duration, not a credential,
    and later readers need it (see :data:`SESSION_LIFETIME_ENV`). It does not
    fall back because a silent default would hide a typo on a shared console:
    the deployment would believe it had shortened the lifetime while every
    terminal went on handing out twelve-hour sessions. An unset or blank
    carrier is the one case that is not a typo — nothing configured the
    lifetime — and takes :data:`DEFAULT_SESSION_LIFETIME`.

    Raises:
        RuntimeError: when the carrier holds anything but a whole number of
            seconds greater than zero. The message names both spellings, the
            config key and the environment variable, because the operator who
            has to fix it may be looking at either.
    """
    raw = os.environ.get(SESSION_LIFETIME_ENV, "")
    text = raw.strip()
    if not text:
        return DEFAULT_SESSION_LIFETIME

    refusal = (
        f"modules.web_terminals.auth.session_lifetime (carried as "
        f"{SESSION_LIFETIME_ENV}) must be a whole number of seconds greater "
        f"than zero, got {raw!r}"
    )
    try:
        seconds = int(text, 10)
    except ValueError as exc:
        raise RuntimeError(refusal) from exc
    if seconds <= 0:
        raise RuntimeError(refusal)
    return seconds


def _roster_accepted() -> bool:
    """Whether this process's environment opted in to the roster secrets.

    Popped like the secrets themselves, so the flag is read once at
    construction and inherited by nothing.
    """
    return os.environ.pop(ROSTER_ACCEPT_ENV, "").strip() == "1"


def _pop_roster_secrets() -> tuple[str, ...]:
    """Take every ``OSPREY_TERMINAL_SECRET_<USER>`` out of the environment.

    Popped for the reason the operator secret is: a value left in
    ``os.environ`` is inherited by every child this process spawns. Blank
    values (an unset variable compose interpolated as ``${VAR:-}``) count as
    absent. Sorted by name so the tuple is deterministic.
    """
    names = sorted(name for name in os.environ if name.startswith(ROSTER_SECRET_ENV_PREFIX))
    values = (os.environ.pop(name, "").strip() for name in names)
    return tuple(value for value in values if value)


def get_web_credentials(app: Any = None) -> WebCredentials:
    """Return this process's credentials, populating them on first use.

    Pass the Starlette/FastAPI ``app`` a request is being served by and the
    result is cached on ``app.state.web_credentials``, which is where the auth
    middleware reads it from. Every app in the process resolves to the *same*
    object: a companion app mounted alongside the terminal must accept the
    cookie the terminal handed the browser, and it can only do that by sharing
    the credentials rather than minting its own.

    Called with no ``app`` it returns the same process-wide holder, for callers
    that need the credentials outside a request — the CLI printing the one-time
    URL, or a test.

    Population is guarded by a lock and runs at most once, so two threads
    racing to serve the first two requests cannot end up with different
    secrets. A failure to populate is not cached: the container-shape
    ``RuntimeError`` re-raises on every call, because the condition that caused
    it — a declared bind host with no supplied secret — is still true, and so
    does a refused session lifetime, whose carrier is read rather than popped
    and is therefore still just as unreadable on the next call.
    """
    if app is not None:
        state = getattr(app, "state", None)
        existing = getattr(state, "web_credentials", None)
        if isinstance(existing, WebCredentials):
            return existing

    global _CREDENTIALS
    with _POPULATION_LOCK:
        if _CREDENTIALS is None:
            _CREDENTIALS = _populate()
        credentials = _CREDENTIALS

    if app is not None:
        state = getattr(app, "state", None)
        if state is not None:
            state.web_credentials = credentials
    return credentials


def peek_web_credentials() -> WebCredentials | None:
    """Return this process's credentials if it already holds them, else ``None``.

    The side-effect-free companion to :func:`get_web_credentials`, and the only
    safe way for a caller that merely wants to *know* whether this process has
    an identity to ask. It never calls :func:`_populate`, so it never pops
    :data:`OPERATOR_SECRET_ENV` or :data:`PANEL_TOKEN_ENV` out of
    ``os.environ``, never harvests the roster carriers, never mints a secret,
    and never raises the container-shape ``RuntimeError``.

    That distinction is load-bearing, not hygiene. ``get_web_credentials`` in a
    process that never held a carrier does two damaging things: it MINTS an
    operator secret and panel token that no other process in the deployment
    recognises — a fabricated identity, which is worse than none — and it
    removes the panel token from the environment on the way, so children spawned
    afterwards (the MCP server the agent runs, and whatever it spawns) lose the
    carrier they were deliberately given. A caller asking "do I have an
    identity?" must therefore never be the caller that creates one.

    Reads the holder under :data:`_POPULATION_LOCK` so it cannot observe a
    half-built value while another thread is populating.
    """
    with _POPULATION_LOCK:
        return _CREDENTIALS


def close_env_carriers() -> None:
    """Remove both credential carriers from ``os.environ`` if either is still there.

    :func:`_populate` pops them, but it runs at most once per process, while
    :func:`mint_and_announce` re-publishes the operator secret on *every*
    launch — deliberately, so a ``--reload`` worker or ``--detach`` child
    inherits it. On the direct-serve path the launcher becomes the server
    without a new process, so that re-published value would otherwise sit in
    ``os.environ`` for the server's whole life and be handed to every agent the
    SDK spawns, which builds its child environment as
    ``{**os.environ, **options.env}``.

    Called at every interface app's construction, from
    :func:`osprey.interfaces._app_setup.configure_interface_app`, immediately
    after the credentials have been resolved — so the carriers are closed
    whichever launch shape this process took, including the per-user
    container's ``osprey web``. Idempotent, and cheap enough to be
    unconditional: the credentials live in this module's memory and nothing
    here reads the environment back.

    **Call :func:`get_web_credentials` first.** This does not settle anything;
    running it before population would discard a deployment-supplied secret and
    leave the holder minting a local one that nginx would never match.
    """
    os.environ.pop(OPERATOR_SECRET_ENV, None)
    os.environ.pop(PANEL_TOKEN_ENV, None)
    os.environ.pop(ROSTER_ACCEPT_ENV, None)
    _pop_roster_secrets()


def reset_web_credentials() -> None:
    """Forget the process credentials so the next call re-populates them.

    For tests. A test that exercises the environment-driven population path has
    to start from an unpopulated holder, and without this the first test in a
    worker would decide the secrets for every test after it. Note that it does
    not restore the environment variables population popped, and does not clear
    ``app.state.web_credentials`` on an app built before the reset — an app
    outliving the reset keeps the credentials it cached.
    """
    global _CREDENTIALS
    with _POPULATION_LOCK:
        _CREDENTIALS = None


def mint_and_announce(host: str, port: int, *, path: str = "/") -> str:
    """Settle this process's operator secret and return the URL that carries it.

    Every OSPREY serving entry point calls this once, just before it binds, and
    prints what comes back. The returned URL is the operator's only way in:
    nothing else prints the secret, and a browser that has not been handed this
    URL cannot reach the surface at all.

    Args:
        host: The address being bound, used verbatim — a caller that binds
            ``0.0.0.0`` announces ``0.0.0.0``, because only the caller knows
            which name its operator should reach it by. An IPv6 literal is
            wrapped in brackets, since ``http://::1:8080/`` has no unambiguous
            parse.
        port: The **settled** port, not the requested one. Entry points that
            fall back to a free port must call this after that fallback, or
            they announce a token for a socket nothing is listening on.
        path: The page the token is exchanged at, defaulting to the app root.
            A bare path, with no query string of its own — the ``?token=`` is
            appended, so a path already carrying a query would produce two.

    Returns:
        ``http://<host>:<port><path>?token=<operator secret>``.

    **The token in the URL is the operator secret itself**, not a second
    credential the holder tracks alongside it. That is forced by how the
    serving process is started: under ``--reload`` and ``--detach`` the process
    that prints this URL is not the process that answers it, and the only thing
    that crosses that boundary is the environment. A distinct URL token would
    have to be carried over in an environment variable of its own, at which
    point it is one more secret in the child's environment buying nothing — it
    is exchangeable for a full operator session either way. Nothing is written
    to disk, and nothing is registered: the child recognises the token because
    :func:`get_web_credentials` there pops the same secret out of the
    environment, and :meth:`WebCredentials.verify_operator` compares it in
    constant time.

    The token is exchanged **once** by the browser — the exchange answers with
    a session cookie and a redirect to the clean URL, so the secret leaves the
    address bar — but it is not burned by that exchange. It stays valid for the
    life of the process, because the operator who reloads the tab, opens a
    second window, or comes back after the cookie expires has no other way to
    re-enter, and re-minting would invalidate the session already handed out.

    **Setting the environment variable here contradicts the ``pop`` in
    :func:`_populate` on purpose, and the window it opens is closed again
    before the process serves anything.** The pop closes the agent-visible
    window in the *serving* process: by the time that process is answering
    requests and spawning the agent's sandbox, the secret exists only in this
    module's memory. This function runs in the *CLI parent*, before any serving
    app is constructed and before any agent exists, and re-publishes the value
    for exactly one purpose — the process it is about to spawn or become. Under
    ``--reload`` and ``--detach`` that is a fresh process, which pops the
    carrier at its own first :func:`get_web_credentials`. In *direct-serve*
    mode the launcher **becomes** the server in the same, already-populated
    process, so no fresh :func:`_populate` runs and nothing there would pop the
    value this function just re-published; :func:`close_env_carriers`, called
    from every interface app's construction, is what removes it on that path.
    Relying instead on :mod:`osprey.agent_runner.clean_env` to omit the name
    would not work: the agent SDK overlays its ``env`` onto ``os.environ``
    rather than replacing it, so an omitted key is simply inherited. The rule
    that keeps all of this consistent is: **call this from a launcher,
    immediately before serving or spawning, never from inside a process that is
    already serving.** A serving process that calls it puts its own secret back
    into the environment its agent inherits — until the next app construction,
    if any — which is the escalation this module exists to prevent.

    Idempotent per process: the secret is settled by the process-wide holder,
    so a second call — a second entry point in the same process, a retry after
    a port fallback — announces the same token and the same URL rather than
    invalidating the first.

    Raises:
        RuntimeError: in the container shape with no supplied secret, from
            :func:`get_web_credentials`. The guard is untouched here: a caller
            in that shape must be given the deployment's secret, and this
            function then announces *that* value, which is the one nginx
            forwards. Host-side entry points normally leave
            :data:`BIND_HOST_ENV` unset and never see it.
    """
    credentials = get_web_credentials()

    # Deliberate re-publication for the child/worker about to be started; see
    # the contradiction paragraph above for why this is not the leak the pop
    # exists to prevent.
    os.environ[OPERATOR_SECRET_ENV] = credentials.operator_secret

    if not path.startswith("/"):
        path = f"/{path}"
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    # A minted secret is already URL-safe, so this is a no-op for one. A
    # deployment-supplied secret is not: it comes out of a ``.env`` and may
    # hold anything, and an unescaped ``&`` or ``#`` there would truncate the
    # token the browser sends back into something that authenticates nobody.
    token = urllib.parse.quote(credentials.operator_secret, safe="")
    return f"http://{authority}:{port}{path}?token={token}"


class Tier(Enum):
    """Which credential a request needs before it is allowed to reach a route.

    Two members and no ordering: the middleware branches on identity, and a
    numeric or comparable tier would invite a ``>=`` test that quietly treats a
    new member as sufficient. :data:`Tier.OPERATOR` is the default answer for
    anything :func:`classify` does not recognise, so adding a route to the tree
    without touching this module leaves it protected rather than exposed.
    """

    #: The narrow, low-privilege tier: panel arrangement and activity reporting.
    #: Reachable with the panel token *or* with any operator credential.
    PANEL = "panel"

    #: Everything else. Reachable only with the operator secret or a live
    #: browser session — never with the panel token.
    OPERATOR = "operator"


#: The exact ``(method, path)`` pairs an in-process companion may drive with the
#: panel token alone.
#:
#: Every entry is a real route in this tree: the first six live in
#: :mod:`osprey.interfaces.web_terminal.routes.panels` and
#: :mod:`osprey.interfaces.web_terminal.routes.agent_activity`, and
#: ``POST /api/focus`` is the artifacts app's focus setter
#: (:mod:`osprey.interfaces.artifacts.app`). Because one process serves several
#: apps and the table is shared by all of them, ``POST /api/focus`` is
#: panel-tier everywhere — on an app that has no such route the request clears
#: the gate and is then answered with a 404 by routing, which grants nothing.
#:
#: What is deliberately *absent* matters as much as what is present:
#: ``GET /api/panel-focus`` (a real route) and ``POST /api/panel-layout`` (which
#: writes the persisted layout) are operator-only, as is every config, scaffold,
#: memory, chat, feedback, file, session and proxy route, and both websockets.
PANEL_TIER_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/panels"),
        ("POST", "/api/panel-focus"),
        ("POST", "/api/panel-visibility"),
        ("POST", "/api/panel-close"),
        ("POST", "/api/panel-arrange"),
        ("POST", "/api/agent-activity"),
        ("POST", "/api/focus"),
    }
)

#: The one route whose tier depends on its body, kept out of
#: :data:`PANEL_TIER_ROUTES` so the flat table stays a pure lookup. Registering
#: a panel *with* a ``url`` decides where the terminal's own proxy forwards
#: operator traffic, which is an operator act; registering without one cannot
#: repoint anything.
PANEL_REGISTER_ROUTE: tuple[str, str] = ("POST", "/api/panels/register")


def classify(method: str | None, path: str | None, body_has_url: bool) -> Tier:
    """Return the credential tier a request must satisfy.

    Args:
        method: The HTTP method from the ASGI scope. Websocket scopes carry no
            method, so the middleware supplies one for them; either way the
            value is upper-cased before the lookup. A missing or empty method
            yields :data:`Tier.OPERATOR`.
        path: The request path from the ASGI scope, **bare**. In the
            multi-user shape nginx strips the ``/u/<user>`` prefix before the
            app sees the request, so a prefixed path is not something this
            function accepts — it is something it refuses, by not matching.
            The path must also carry no query string; ``scope["path"]`` never
            does.
        body_has_url: Whether the JSON body of a
            ``POST /api/panels/register`` carries a ``url`` key. Ignored for
            every other route. **Pass ``True`` whenever the answer is not
            known** — an unreadable or unparseable body must be treated as the
            dangerous case, not the harmless one. Required rather than
            defaulted: a ``False`` default would let a caller that forgot the
            argument grant URL-backed registration to the weak credential, and
            a ``True`` default would be a parameter whose name contradicts its
            value. Making the caller decide is the only spelling with no wrong
            way to hold it.

    Returns:
        :data:`Tier.PANEL` for the handful of routes in
        :data:`PANEL_TIER_ROUTES`, and for a ``url``-free panel registration.
        :data:`Tier.OPERATOR` for everything else.

    The match is **exact** on both fields, and that is the security property
    rather than an implementation shortcut. A prefix match on ``/api/panel``
    would hand the weak credential ``/api/panel-layout``; a prefix match on
    ``/api/panels`` would hand it every registration, ``url`` and all. So a
    trailing slash, a different case, an unexpected extra segment or a
    surviving ``/u/<user>`` prefix all fall through to
    :data:`Tier.OPERATOR`. That direction of failure costs a companion at most
    a refusal on a spelling no client uses; the other direction would be a
    privilege escalation.
    """
    if not method or not path:
        return Tier.OPERATOR

    route = (method.upper(), path)
    if route in PANEL_TIER_ROUTES:
        return Tier.PANEL
    if route == PANEL_REGISTER_ROUTE and not body_has_url:
        return Tier.PANEL
    return Tier.OPERATOR
