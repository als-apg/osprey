"""Who a record source is, whose ``on_behalf_of`` is trusted, and which credential reads what.

Three tables, all pure functions of the rendered deployment config:

**The source-identity map.** Every entry's ``identity`` comes from where it
was found — the spool directory, the OTLP login, or the transcript root — and
never from anything a sender wrote. :func:`spool_source_identity`,
:func:`login_source_identity` and :func:`transcript_source_identity` turn each
of those into the one trusted identity. Dispatch workers in the shared
workspace mode all map to :data:`SHARED_DISPATCH_IDENTITY`, because they share
one workspace and one transcript volume; a run's accept entry and the
session's transcript lines therefore carry the same identity in both modes.

**The trusted on-behalf list.** ``on_behalf_of`` is honoured for the sidecar
on every audit surface and for the rendered lane queueservers on surface
``preflight`` only. Every other sender's value is dropped at ingest
(:func:`trusted_on_behalf`, :func:`keep_on_behalf_of`).

**The credential → scope table.** :func:`credential_table` names every
login agent-record accepts, the identity it stands for and the scope it reads.
A login is the identity's name, except a named reader's: the identity
``reader:<name>`` logs in as ``reader.<name>``, because a Basic user-id cannot
contain ``:``. Only the ``history`` login may assert a viewer
(:class:`ViewerAsserted`); the reader maps that viewer to ``Own``, ``All`` or
``Reader`` the way the history page's own sign-in mapping does.

The module holds no I/O. Config helpers from :mod:`osprey.deployment` are
imported inside the functions that use them, so importing this module stays
cheap and cycle-free.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from osprey.agent_record.contract import (
    KINDS,
    SOURCE_BRIDGE,
    SOURCE_DISPATCH,
    SOURCE_HISTORY,
    SOURCE_HOST,
    SOURCE_OPERATOR,
    SOURCE_READER,
    SOURCE_SERVICE,
    SOURCE_SIDECAR,
    SOURCE_TERMINAL,
    All,
    Own,
    Reader,
    Scope,
    Source,
    check_identity,
    check_kind,
)

#: The auth sidecar's identity (the same name its audit identity carries).
SIDECAR_IDENTITY = "sidecar"
#: The history front's identity; the one login that may assert a viewer.
HISTORY_IDENTITY = "history"
#: The operator host credential (loopback only), which reads everything.
OPERATOR_IDENTITY = "operator"
#: The surface on which queueservers' ``on_behalf_of`` is trusted.
PREFLIGHT_SURFACE = "preflight"

#: Each bridge's ``deployed_services`` name and its identity.
BRIDGE_IDENTITIES: Mapping[str, str] = {
    "teams_bridge": "teams-bridge",
    "gchat_bridge": "gchat-bridge",
    "nextcloud_bridge": "nextcloud-bridge",
}

#: The one identity every dispatch worker maps to in the shared workspace mode.
SHARED_DISPATCH_IDENTITY = "dispatch-worker-shared"
_DISPATCH_WORKER_RE = re.compile(r"^dispatch-worker-([1-9][0-9]*)$")

#: A named reader's identity is ``reader:<name>``; its login is ``reader.<name>``.
READER_IDENTITY_PREFIX = "reader:"
READER_LOGIN_PREFIX = "reader."

#: Transcript roots: a web terminal's ``<user>-claude-projects`` volume, the
#: dispatch workers' ``dispatch_claude_projects`` (shared) or
#: ``dispatch_claude_projects_<i>`` volumes, and the host build's projects
#: directory, which belongs to ``deployment.host_identity``.
TERMINAL_TRANSCRIPT_VOLUME_SUFFIX = "-claude-projects"
DISPATCH_TRANSCRIPT_VOLUME = "dispatch_claude_projects"
HOST_TRANSCRIPT_ROOT = "host"
_DISPATCH_VOLUME_RE = re.compile(r"^dispatch_claude_projects_([1-9][0-9]*)$")


# ---------------------------------------------------------------------------
# Config reads
# ---------------------------------------------------------------------------


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _deployed(config: Mapping[str, Any]) -> set[str]:
    return {str(name) for name in config.get("deployed_services") or []}


def _web_terminals_enabled(config: Mapping[str, Any]) -> bool:
    block = (config.get("modules") or {}).get("web_terminals") or {}
    return isinstance(block, Mapping) and bool(block.get("enabled"))


def dispatch_shared(config: Mapping[str, Any] | None) -> bool:
    """Whether the dispatch workers run in the shared workspace mode."""
    worker = (_config(config).get("services") or {}).get("dispatch_worker")
    return isinstance(worker, Mapping) and worker.get("workspace_mode", "isolated") == "shared"


def terminal_identities(config: Mapping[str, Any] | None) -> list[str]:
    """The roster terminals (people and shared cards alike) the config renders.

    Empty unless web terminals are on.
    """
    config = _config(config)
    if not _web_terminals_enabled(config):
        return []
    from osprey.deployment.web_terminals.roster import resolve_cards

    web_terminals = dict((config.get("modules") or {}).get("web_terminals") or {})
    registry = config.get("registry") or {}
    return [entry["name"] for entry in resolve_cards(web_terminals, registry, strict=False)]


def dispatch_identities(config: Mapping[str, Any] | None) -> list[str]:
    """The source identity of every dispatch worker; one in the shared mode."""
    from osprey.deployment.compose_generator import dispatch_worker_audit_identities

    workers = dispatch_worker_audit_identities(_config(config))
    if workers and dispatch_shared(config):
        return [SHARED_DISPATCH_IDENTITY]
    return list(workers)


def bridge_identities(config: Mapping[str, Any] | None) -> list[str]:
    """The identity of every deployed bridge, in :data:`BRIDGE_IDENTITIES` order."""
    deployed = _deployed(_config(config))
    return [identity for service, identity in BRIDGE_IDENTITIES.items() if service in deployed]


def queueserver_identities(config: Mapping[str, Any] | None) -> list[str]:
    """The rendered lane queueservers' identities."""
    from osprey.deployment.compose_generator import lane_queueserver_audit_identities

    return list(lane_queueserver_audit_identities(_config(config)))


def host_identity(config: Mapping[str, Any] | None) -> str | None:
    """The identity host launches send and record under (``deployment.host_identity``).

    ``None`` when the config records none, which is a config no build has
    stamped yet.
    """
    deployment = _config(config).get("deployment")
    value = deployment.get("host_identity") if isinstance(deployment, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def reader_entries(config: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The ``agent_record.readers`` entries that name a reader."""
    block = _config(config).get("agent_record") or {}
    stated = block.get("readers") if isinstance(block, Mapping) else None
    if not isinstance(stated, list):
        return []
    return [e for e in stated if isinstance(e, Mapping) and isinstance(e.get("name"), str)]


# ---------------------------------------------------------------------------
# The source-identity map
# ---------------------------------------------------------------------------


def _collapse_dispatch(config: Mapping[str, Any] | None, name: str) -> str:
    if _DISPATCH_WORKER_RE.match(name) and dispatch_shared(config):
        return SHARED_DISPATCH_IDENTITY
    return name


def spool_source_identity(config: Mapping[str, Any] | None, directory: str) -> str:
    """The trusted identity of the lines found in spool directory *directory*.

    The directory name is the writer's identity; a dispatch worker's maps to
    the shared identity in the shared mode.

    Raises:
        ValueError: If *directory* cannot be an identity.
    """
    return _collapse_dispatch(config, check_identity(directory))


def login_source_identity(config: Mapping[str, Any] | None, login: str) -> str:
    """The trusted identity of an OTLP sender (and of ``.otlp/<login>/``).

    Raises:
        ValueError: If *login* is a reader's login (readers send nothing) or
            cannot be an identity.
    """
    if login.startswith(READER_LOGIN_PREFIX):
        raise ValueError(f"reader login {login!r} is not a record source")
    return _collapse_dispatch(config, check_identity(login))


def transcript_source_identity(config: Mapping[str, Any] | None, root: str) -> str:
    """The trusted identity of the transcripts under transcript root *root*.

    Raises:
        ValueError: If *root* is not a transcript root this module knows, or
            names the host root while no host identity is configured.
    """
    if root == HOST_TRANSCRIPT_ROOT:
        host = host_identity(config)
        if host is None:
            raise ValueError("the host transcript root needs deployment.host_identity")
        return check_identity(host)
    if root == DISPATCH_TRANSCRIPT_VOLUME:
        return SHARED_DISPATCH_IDENTITY
    match = _DISPATCH_VOLUME_RE.match(root)
    if match:
        return _collapse_dispatch(config, f"dispatch-worker-{match[1]}")
    if root.endswith(TERMINAL_TRANSCRIPT_VOLUME_SUFFIX):
        user = root[: -len(TERMINAL_TRANSCRIPT_VOLUME_SUFFIX)]
        if user:
            return check_identity(user)
    raise ValueError(f"{root!r} is not a transcript root")


def transcript_roots(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Every transcript root this deployment renders, mapped to its identity."""
    roots = {
        f"{user}{TERMINAL_TRANSCRIPT_VOLUME_SUFFIX}": user for user in terminal_identities(config)
    }
    for worker in dispatch_identities(config):
        if worker == SHARED_DISPATCH_IDENTITY:
            roots[DISPATCH_TRANSCRIPT_VOLUME] = worker
        elif index := _DISPATCH_WORKER_RE.match(worker):
            roots[f"{DISPATCH_TRANSCRIPT_VOLUME}_{index[1]}"] = worker
    host = host_identity(config)
    if host is not None:
        roots[HOST_TRANSCRIPT_ROOT] = host
    return roots


def identity_class(config: Mapping[str, Any] | None, identity: str) -> str:
    """The source class of *identity*, for the per-source kind allow-list."""
    if identity == SIDECAR_IDENTITY:
        return SOURCE_SIDECAR
    if identity == HISTORY_IDENTITY:
        return SOURCE_HISTORY
    if identity == OPERATOR_IDENTITY:
        return SOURCE_OPERATOR
    if identity.startswith(READER_IDENTITY_PREFIX):
        return SOURCE_READER
    if identity in BRIDGE_IDENTITIES.values():
        return SOURCE_BRIDGE
    if identity == SHARED_DISPATCH_IDENTITY or _DISPATCH_WORKER_RE.match(identity):
        return SOURCE_DISPATCH
    if identity == host_identity(config):
        return SOURCE_HOST
    if identity in terminal_identities(config):
        return SOURCE_TERMINAL
    return SOURCE_SERVICE


class IdentityClasses:
    """:func:`identity_class` over one deployment config, each identity worked out once."""

    def __init__(self, config: Mapping[str, Any] | None) -> None:
        self._config = config
        self._known: dict[str, str] = {}

    def __call__(self, identity: str) -> str:
        known = self._known.get(identity)
        if known is None:
            known = self._known[identity] = identity_class(self._config, identity)
        return known


# ---------------------------------------------------------------------------
# The trusted on-behalf list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedOnBehalf:
    """Whose ``on_behalf_of`` ingest keeps.

    ``sidecar`` is trusted on every surface; ``queueservers`` on
    :data:`PREFLIGHT_SURFACE` only.
    """

    sidecar: str
    queueservers: tuple[str, ...]


def trusted_on_behalf(config: Mapping[str, Any] | None) -> TrustedOnBehalf:
    """The trusted on-behalf list of the deployment *config* renders."""
    return TrustedOnBehalf(SIDECAR_IDENTITY, tuple(queueserver_identities(config)))


def keep_on_behalf_of(trusted: TrustedOnBehalf, identity: str, surface: str | None) -> bool:
    """Whether an entry from *identity* on audit *surface* keeps its ``on_behalf_of``."""
    if identity == trusted.sidecar:
        return True
    return surface == PREFLIGHT_SURFACE and identity in trusted.queueservers


# ---------------------------------------------------------------------------
# The credential → scope table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewerAsserted:
    """The history front's grant: it names a viewer in ``X-Osprey-Record-Viewer``.

    The reader maps the viewer to ``Own``, ``All`` or ``Reader`` from the
    roster. A request from this login that names no viewer reads nothing.
    """


#: The header the history front names its viewer in.
VIEWER_HEADER = "X-Osprey-Record-Viewer"

Grant = Scope | ViewerAsserted


@dataclass(frozen=True)
class CredentialGrant:
    """One login agent-record accepts: the identity it stands for and what it reads.

    A ``Source`` grant held by a dispatch worker may be narrowed per request
    (its own session, or the allow-listed run ids); the grant itself is the
    widest view the login ever gets.
    """

    login: str
    identity: str
    source_class: str
    scope: Grant

    @property
    def may_assert_viewer(self) -> bool:
        return isinstance(self.scope, ViewerAsserted)


def reader_identity(name: str) -> str:
    """``reader:<name>``."""
    return f"{READER_IDENTITY_PREFIX}{_reader_name(name)}"


def _reader_name(name: str) -> str:
    check_identity(name)
    return name


def login_for(identity: str) -> str:
    """The Basic login of *identity*: its name, or ``reader.<name>`` for a reader."""
    if identity.startswith(READER_IDENTITY_PREFIX):
        return f"{READER_LOGIN_PREFIX}{_reader_name(identity[len(READER_IDENTITY_PREFIX) :])}"
    return check_identity(identity)


def _reader_scope(entry: Mapping[str, Any]) -> Reader:
    name = entry["name"]
    kinds = entry.get("kinds")
    if kinds is None:
        stated = frozenset(KINDS)
    elif isinstance(kinds, list) and all(isinstance(k, str) for k in kinds):
        stated = frozenset(check_kind(k) for k in kinds)
    else:
        raise ValueError(f"agent_record.readers {name!r}: kinds must be a list of kinds")
    return Reader(name, stated, entry.get("days_back"))


def credential_table(config: Mapping[str, Any] | None) -> dict[str, CredentialGrant]:
    """Every login agent-record accepts, keyed by login, in a stable order.

    Terminals and the host identity read ``Own`` of their person; dispatch
    workers, bridges and the sidecar read ``Source`` of their identity; the
    history front asserts a viewer; the operator reads ``All``; each named
    reader reads its ``Reader`` scope.

    Raises:
        ValueError: If two grants would share a login, naming both, or an
            identity cannot be a login.
    """
    grants: list[CredentialGrant] = []

    def add(identity: str, source_class: str, scope: Grant, what: str) -> None:
        try:
            login = login_for(identity)
        except ValueError as exc:
            raise ValueError(f"{what} {identity!r} cannot be a record login: {exc}") from None
        grants.append(CredentialGrant(login, identity, source_class, scope))

    for user in terminal_identities(config):
        add(user, SOURCE_TERMINAL, Own(user), "roster name")
    for worker in dispatch_identities(config):
        add(worker, SOURCE_DISPATCH, Source(worker), "dispatch worker")
    for bridge in bridge_identities(config):
        add(bridge, SOURCE_BRIDGE, Source(bridge), "bridge")
    if _web_terminals_enabled(_config(config)):
        add(SIDECAR_IDENTITY, SOURCE_SIDECAR, Source(SIDECAR_IDENTITY), "sidecar")
    host = host_identity(config)
    if host is not None:
        add(host, SOURCE_HOST, Own(host), "deployment.host_identity")
    add(HISTORY_IDENTITY, SOURCE_HISTORY, ViewerAsserted(), "history")
    add(OPERATOR_IDENTITY, SOURCE_OPERATOR, All(), "operator")
    for entry in reader_entries(config):
        add(reader_identity(entry["name"]), SOURCE_READER, _reader_scope(entry), "reader")

    table: dict[str, CredentialGrant] = {}
    for grant in grants:
        prior = table.get(grant.login)
        if prior is not None:
            raise ValueError(
                f"login {grant.login!r} is claimed twice: by the {prior.source_class} "
                f"{prior.identity!r} and by the {grant.source_class} {grant.identity!r}"
            )
        table[grant.login] = grant
    return table
