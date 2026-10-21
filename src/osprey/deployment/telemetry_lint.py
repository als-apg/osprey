"""Load-time checks on the telemetry store, its gateway and its readers.

Every rule here answers from the config alone, before anything is rendered or
started, and names the key to change. They run wherever the web-terminal lint
runs — ``osprey build``, ``osprey validate``,
``osprey scaffold web-terminals lint`` — and
``osprey up`` refuses on the ones in :func:`preflight_problems`.

Three severities, and what each one means to the command reporting it:

``error``
    The config must be rejected.
``warn``
    Worth an operator's attention; the command goes on.
``info``
    A fact about what this config deploys, said once so that it is a choice
    rather than a surprise. Never a problem: a store nobody is named to read
    is a complete, working deployment.

The rules about a roster *name* — the reserved audit identities, the lane
queueserver names — are the web-terminal lint's, because the name is a
directory in every posture; this module holds the rules about the store.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from osprey.deployment import telemetry_config as tc
from osprey.deployment.qmd_service import is_loopback_bind, resolve_bind_address
from osprey.services.auth_sidecar.roster_env import env_var_suffix_collisions

__all__ = [
    "READER_NAME_RE",
    "TelemetryFinding",
    "host_identity_advisories",
    "lint_telemetry",
    "preflight_problems",
    "reader_suffix_collision_problems",
    "telemetry_errors",
    "telemetry_warnings",
]

#: What a reader may be called: the roster's charset, because a reader is a
#: store user, a credential file suffix and a sidecar carrier under that name.
READER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_STORE = "services.telemetry_store"
_GATEWAY = "services.telemetry_gateway"
_WEB = "modules.web_terminals"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class TelemetryFinding:
    """One lint result; see the module docstring for the three severities."""

    severity: Literal["error", "warn", "info"]
    code: str
    message: str


def _error(code: str, message: str) -> TelemetryFinding:
    return TelemetryFinding("error", f"telemetry.{code}", message)


def _warn(code: str, message: str) -> TelemetryFinding:
    return TelemetryFinding("warn", f"telemetry.{code}", message)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mib(size: float) -> str:
    return f"{size / (1024 * 1024):,.0f} MiB"


def lint_telemetry(
    config: Mapping[str, Any] | None,
    *,
    receivers: int | None = None,
    runtime_user: str | None = None,
) -> list[TelemetryFinding]:
    """Check everything the config says about the telemetry store.

    Args:
        config: The rendered config, as nested mappings.
        receivers: The number of ledger receivers the gateway is rendered
            with. Only a caller holding the render knows it; without it the
            gateway memory rule is not asked.
        runtime_user: The account running the command, for the host-identity
            rule. ``None`` skips that rule, which is right wherever nothing is
            about to be launched.

    Returns:
        Every finding, empty when the config says nothing wrong. A deployment
        that does not deploy the store gets the rules about ``history`` values
        only: the other keys are read by nothing there.
    """
    config = _mapping(config)
    web_terminals = _mapping(_mapping(config.get("modules")).get("web_terminals"))
    findings = _check_history_values(web_terminals)
    if not tc.store_deployed(config):
        return findings
    findings.extend(_check_values(config))
    findings.extend(_check_host_network(config))
    findings.extend(_check_readers(config, web_terminals))
    findings.extend(_check_store_memory(config))
    if receivers is not None:
        findings.extend(_check_gateway_memory(config, receivers))
    if web_terminals.get("enabled"):
        findings.extend(_check_reader_origin(web_terminals))
        findings.extend(_check_reader_transport(config, web_terminals))
        findings.extend(_check_query_page(config))
        findings.extend(_check_unbound_history_roles(web_terminals))
    if runtime_user is not None:
        findings.extend(_check_host_identity(config, runtime_user))
    return findings


def telemetry_errors(config: Mapping[str, Any] | None, **kwargs: Any) -> list[str]:
    """The :func:`lint_telemetry` messages that must fail a command."""
    return [f.message for f in lint_telemetry(config, **kwargs) if f.severity == "error"]


def telemetry_warnings(config: Mapping[str, Any] | None, **kwargs: Any) -> list[str]:
    """The advisory :func:`lint_telemetry` messages."""
    return [f.message for f in lint_telemetry(config, **kwargs) if f.severity == "warn"]


def reader_suffix_collision_problems(
    config: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    """Reader names that would share one set of sidecar carriers.

    ``jean-luc`` and ``jean_luc`` both key ``..._JEAN_LUC``, so one reader's
    subject and tables would be read as the other's.

    Returns:
        One ``(problem, remedy)`` pair per shared suffix, naming both readers.
    """
    collisions = env_var_suffix_collisions(tc.reader_names(config))
    return [
        (
            f"{_STORE}.readers {' and '.join(repr(name) for name in names)} share the "
            f"environment-variable suffix {suffix} (a name is upper-cased and '-' "
            "becomes '_'), so the sign-on settings of one would be read as the other's",
            "Rename one of them",
        )
        for suffix, names in collisions.items()
    ]


def preflight_problems(config: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """What ``osprey up`` refuses a start over, before anything is touched.

    The two rules whose violation the running stack cannot survive or contain:
    reader names sharing a carrier suffix, and a store whose query limits do
    not fit its memory limit. Pure: reads the config and nothing else.

    Returns:
        ``(problem, remedy)`` pairs, empty when the start may go on.
    """
    if not tc.store_deployed(config):
        return []
    return [*reader_suffix_collision_problems(config), *_store_memory_problems(_mapping(config))]


def host_identity_advisories(
    config: Mapping[str, Any] | None, runtime_user: str | None
) -> list[str]:
    """What a command about to start the deployment says about who is running it.

    The one telemetry rule that is about the invoking account rather than the
    config, so it is asked where something is launched and nowhere else.
    Empty when the store is not deployed, when no account name is known, and
    when the account is the one the build recorded.
    """
    if runtime_user is None or not tc.store_deployed(config):
        return []
    return [f.message for f in _check_host_identity(_mapping(config), runtime_user)]


# --- values ------------------------------------------------------------------


def _check_values(config: Mapping[str, Any]) -> list[TelemetryFinding]:
    """Every scalar the store and gateway blocks state must be usable as stated."""
    findings: list[TelemetryFinding] = []
    for read in (
        tc.store_port,
        tc.gateway_port,
        tc.reader_port,
        tc.retention_days,
        tc.store_limits,
        tc.store_memory_bytes,
        lambda cfg: tc.gateway_memory_bytes(cfg, 0),
    ):
        try:
            read(config)
        except ValueError as exc:
            findings.append(_error("invalid_value", str(exc)))
    for service in (tc.STORE_SERVICE, tc.GATEWAY_SERVICE):
        image = tc.service_block(config, service).get("image")
        if image is not None and not (isinstance(image, str) and image.strip()):
            findings.append(
                _error(
                    "invalid_value",
                    f"services.{service}.image must be an image reference, got {image!r}",
                )
            )
    return findings


def _check_host_network(config: Mapping[str, Any]) -> list[TelemetryFinding]:
    """The store and the gateway always sit on the compose network.

    The store is published on loopback so that the readers' origin is the only
    remote path to it, and the gateway reaches it by service name; on the host
    network neither holds.
    """
    findings = []
    for service in (tc.STORE_SERVICE, tc.GATEWAY_SERVICE):
        network = tc.service_block(config, service).get("network")
        if network == "host":
            findings.append(
                _error(
                    "host_network",
                    f"services.{service}.network is 'host', which this service does not "
                    "support: the store is reached by the gateway by its service name and "
                    "publishes on loopback only, so both stay on the compose network. "
                    "Remove the key",
                )
            )
    return findings


# --- readers -----------------------------------------------------------------


def _check_readers(
    config: Mapping[str, Any], web_terminals: Mapping[str, Any]
) -> list[TelemetryFinding]:
    """Reader entries: shape, names, ``days_back`` and ``oidc_subject``."""
    entries = tc.readers(config)
    stated = tc.service_block(config, tc.STORE_SERVICE).get("readers")
    if stated is not None and not isinstance(stated, list):
        return [_error("invalid_readers", f"{_STORE}.readers must be a list, got {stated!r}")]
    if not entries:
        return [
            TelemetryFinding(
                "info",
                "telemetry.no_reader",
                f"{_STORE}.readers is empty: the store records everything and nobody is "
                "named to query it. Add an entry (`- name: <who>`) to give someone a "
                "read-only key",
            )
        ]

    findings: list[TelemetryFinding] = []
    try:
        retention: int | None = tc.retention_days(config)
    except ValueError:
        retention = None  # reported by _check_values
    users = web_terminals.get("users")
    roster = {
        name
        for user in (users if isinstance(users, list) else [])
        if isinstance(name := user.get("name") if isinstance(user, Mapping) else user, str)
    }
    auth_method = _mapping(web_terminals.get("auth")).get("method")
    seen: set[str] = set()
    duplicates: list[str] = []
    for position, entry in enumerate(entries):
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not READER_NAME_RE.fullmatch(name):
            findings.append(
                _error(
                    "invalid_reader_name",
                    f"{_STORE}.readers[{position}] must be a mapping whose 'name' is "
                    "lowercase letters, digits, '_' and '-', starting with a letter or "
                    f"digit; got {entry!r}",
                )
            )
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
        if tc.is_reserved_account_name(name):
            findings.append(
                _error(
                    "reserved_reader_name",
                    f"{_STORE}.readers entry {name!r} takes a name the store keeps for "
                    f"its own accounts ({tc.RESERVED_ACCOUNT_PREFIX}*, "
                    f"{', '.join(sorted(tc.RESERVED_ACCOUNT_NAMES))}). Rename the reader",
                )
            )
        findings.extend(_check_days_back(name, entry, retention))
        findings.extend(_check_reader_subject(name, entry, name in roster, auth_method))
    if duplicates:
        findings.append(
            _error(
                "duplicate_reader",
                f"{_STORE}.readers names {', '.join(repr(name) for name in duplicates)} "
                "more than once. A list in a profile is appended to the list it "
                "extends, so an entry repeated to change it becomes a second entry; "
                "state each reader once",
            )
        )
    findings.extend(
        _error("reader_suffix_collision", f"{problem}. {remedy}")
        for problem, remedy in reader_suffix_collision_problems(config)
    )
    return findings


def _check_days_back(
    name: str, entry: Mapping[str, Any], retention: int | None
) -> list[TelemetryFinding]:
    """``days_back`` is a whole number of days, at most what the store keeps."""
    if "days_back" not in entry:
        return []
    days = entry.get("days_back")
    key = f"{_STORE}.readers entry {name!r} days_back"
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        return [
            _error(
                "invalid_days_back",
                f"{key} must be a whole number of days, 1 or more, got {days!r}. Remove "
                "the key to let this reader see everything the store keeps",
            )
        ]
    if retention is not None and days > retention:
        return [
            _error(
                "invalid_days_back",
                f"{key} is {days}, more than {_STORE}.retention_days ({retention}): the "
                f"store holds nothing older than {retention} days, so this window "
                "promises rows that do not exist. Lower it or raise retention_days",
            )
        ]
    return []


def _check_reader_subject(
    name: str, entry: Mapping[str, Any], on_roster: bool, auth_method: Any
) -> list[TelemetryFinding]:
    """``oidc_subject`` recognises a reader who is NOT on the roster, under OIDC."""
    if "oidc_subject" not in entry:
        return []
    subject = entry.get("oidc_subject")
    key = f"{_STORE}.readers entry {name!r} oidc_subject"
    if not isinstance(subject, str) or not subject.strip():
        return [_error("invalid_reader_subject", f"{key} must be a non-empty string")]
    if on_roster:
        return [
            _error(
                "roster_reader_subject",
                f"{key} is set, and {name!r} is also a {_WEB}.users entry. A roster "
                "user is recognised in the browser by their own sign-in; a second "
                "subject on the reader entry would let a different sign-in read as "
                "them. Remove oidc_subject from the reader entry",
            )
        ]
    if auth_method != "oidc":
        return [
            _warn(
                "reader_subject_without_oidc",
                f"{key} is set but {_WEB}.auth.method is {auth_method!r}: nothing "
                "asserts a subject without OIDC sign-on, so this reader has a key for "
                "scripts and no way into the history page",
            )
        ]
    return []


# --- memory ------------------------------------------------------------------


def _check_store_memory(config: Mapping[str, Any]) -> list[TelemetryFinding]:
    """Every query that can run at once, at its memory cap, must fit the store."""
    return [
        _error("store_memory", f"{problem}. {remedy}")
        for problem, remedy in _store_memory_problems(config)
    ]


def _store_memory_problems(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The store memory rule as a ``(problem, remedy)`` pair, or nothing."""
    try:
        limits = tc.store_limits(config)
        memory = tc.store_memory_bytes(config)
    except ValueError:
        return []  # reported by _check_values
    demand = tc.store_memory_demand(config)
    ceiling = tc.STORE_MEMORY_RATIO * memory
    if demand <= ceiling:
        return []
    slots = limits["reader_slots"] + tc.HISTORY_CONCURRENCY + tc.STORE_SPARE_QUERY_SLOTS
    return [
        (
            f"The telemetry store's query limits do not fit its memory limit: "
            f"({limits['reader_slots']} reader slots [{_STORE}.limits.reader_slots] + "
            f"{tc.HISTORY_CONCURRENCY} history-page queries + "
            f"{tc.STORE_SPARE_QUERY_SLOTS} for the archive and the schema check) = "
            f"{slots} queries × {_mib(limits['query_memory'])} "
            f"[{_STORE}.limits.query_memory] + {_mib(tc.STORE_IDLE_BYTES)} idle = "
            f"{_mib(demand)}, more than {tc.STORE_MEMORY_RATIO} × {_mib(memory)} "
            f"[{_STORE}.memory] = {_mib(ceiling)}",
            f"Raise {_STORE}.memory, or lower limits.query_memory or limits.reader_slots",
        )
    ]


def _check_gateway_memory(config: Mapping[str, Any], receivers: int) -> list[TelemetryFinding]:
    """A stated gateway memory limit must cover a worst-case ledger catch-up."""
    try:
        memory = tc.gateway_memory_bytes(config, receivers)
    except ValueError:
        return []  # reported by _check_values
    floor = tc.gateway_memory_floor(config, receivers)
    if memory >= floor:
        return []
    body = tc.gateway_max_request_body_size(config)
    line = tc.gateway_max_log_size(config)
    return [
        _error(
            "gateway_memory",
            f"{_GATEWAY}.memory is {_mib(memory)}, below what the gateway needs: "
            f"256 MiB + 8 × {_mib(body)} (the largest request body) + 2 × 100 × "
            f"{_mib(line)} (the longest ledger line) × R, where R = {receivers} is the "
            f"number of audit directories it reads, = {_mib(floor)}. Raise the limit, "
            "or remove the key to have it derived",
        )
    ]


# --- the readers' origin -----------------------------------------------------


def _origin_host(value: Any) -> str | None:
    """The host of a ``scheme://host[:port]`` string, or ``None`` if it is not one."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        parts.port  # noqa: B018 - raises on a port that is not one
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        return None
    return parts.hostname.lower()


def _check_reader_origin(web_terminals: Mapping[str, Any]) -> list[TelemetryFinding]:
    """``reader_origin`` mirrors ``external_origin``, on the same host.

    When a load balancer is what browsers reach, nothing here can derive the
    readers' origin either, so it has to be stated beside ``external_origin``.
    And the sign-in cookie crosses ports on one host only: a readers' origin on
    another host would never see it.
    """
    external = web_terminals.get("external_origin")
    reader = web_terminals.get("reader_origin")
    has_external = isinstance(external, str) and bool(external.strip())
    if reader is None:
        if not has_external:
            return []
        return [
            _error(
                "reader_origin_required",
                f"{_WEB}.external_origin is set and {_WEB}.reader_origin is not. With "
                "an external origin the address browsers reach the readers' pages on "
                "cannot be derived either; set reader_origin to it (scheme://host[:port])",
            )
        ]
    reader_host = _origin_host(reader)
    if reader_host is None:
        return [
            _error(
                "invalid_reader_origin",
                f"{_WEB}.reader_origin must be scheme://host[:port] and nothing else "
                f"(no path, no trailing slash), got {reader!r}",
            )
        ]
    external_host = _origin_host(external) if has_external else None
    if external_host is not None and external_host != reader_host:
        return [
            _error(
                "reader_origin_host",
                f"{_WEB}.reader_origin is on host {reader_host!r} and "
                f"{_WEB}.external_origin on {external_host!r}. The sign-in cookie "
                "crosses ports on one host only, so the readers' pages on another host "
                "would never see a signed-in person. Serve both from one host, on two "
                "ports",
            )
        ]
    return []


def _check_reader_transport(
    config: Mapping[str, Any], web_terminals: Mapping[str, Any]
) -> list[TelemetryFinding]:
    """Reader keys over cleartext HTTP, on a port reachable from elsewhere.

    Two things expose the readers' port: the interface it is published on, and
    the name clients are told to reach it by. Either one alone is enough, so a
    loopback name over a wildcard bind is exposed, and so is a config whose
    origin cannot be derived at all. With no reader there is no key to carry.
    """
    # Imported in-function: render pulls in the whole web-terminals package,
    # and this module is read by commands that deploy none.
    from osprey.deployment.web_terminals.render import _tls_enabled, deployment_origin

    if _tls_enabled(dict(web_terminals)) or not tc.reader_names(config):
        return []
    exposed = not is_loopback_bind(resolve_bind_address(config))
    reader = web_terminals.get("reader_origin")
    if isinstance(reader, str) and reader.strip():
        host = _origin_host(reader)
        if host is None or urlsplit(reader.strip()).scheme == "https":
            # An invalid origin is the origin rule's finding; an https one is
            # served by something that terminates TLS in front of the port.
            return []
        exposed = exposed or host not in _LOOPBACK_HOSTS
    else:
        # A config no origin can be derived from is refused by the render in
        # its own words; the bind alone answers here.
        with contextlib.suppress(ValueError):
            origin = deployment_origin(dict(config))
            if origin.scheme == "https":
                return []
            exposed = exposed or not origin.is_loopback
    if not exposed:
        return []
    return [
        _warn(
            "reader_cleartext",
            f"{_WEB}.tls.enabled is false and the readers' origin is reachable from "
            "other machines: a reader's key travels in every query, in cleartext, "
            "where anything on the network path can read and reuse it. Enable tls for "
            "any deployment reachable beyond a trusted host",
        )
    ]


def _check_query_page(config: Mapping[str, Any]) -> list[TelemetryFinding]:
    """The store image's query page is one whose script this code has read.

    The readers' origin lets exactly one script run, named by its hash, and the
    hash is known per image tag. An image outside that table still stores and
    answers queries; only the page is withheld.
    """
    try:
        if tc.play_script_sha256(config) is not None:
            return []
        image = tc.store_image(config)
    except ValueError:
        # An image that is not a reference is the value rule's finding.
        return []
    known = ", ".join(tc.PLAY_SCRIPT_SHA256)
    return [
        _warn(
            "unknown_query_page",
            f"{_STORE}.image is {image!r}, and the query page of that image has no "
            f"entry in PLAY_SCRIPT_SHA256 (known tags: {known}). The readers' origin "
            "answers /telemetry/play with 404 for it; queries are not affected. Add "
            "the tag's script hash as osprey.deployment.telemetry_config describes, "
            "or keep a known tag",
        )
    ]


# --- history -----------------------------------------------------------------


def _roles(web_terminals: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(web_terminals.get("authorization")).get("roles"))


def _check_history_values(web_terminals: Mapping[str, Any]) -> list[TelemetryFinding]:
    """``history`` on a roster entry or a role is ``own`` or ``all``."""
    findings = []
    users = web_terminals.get("users")
    stated = [
        (f"{_WEB}.users entry {user.get('name')!r}", user["history"])
        for user in (users if isinstance(users, list) else [])
        if isinstance(user, Mapping) and "history" in user
    ]
    stated += [
        (f"{_WEB}.authorization.roles.{role}", entry["history"])
        for role, entry in _roles(web_terminals).items()
        if isinstance(entry, Mapping) and "history" in entry
    ]
    for where, value in stated:
        if value not in tc.HISTORY_VALUES:
            findings.append(
                _error(
                    "invalid_history",
                    f"{where} has history {value!r}; it is 'own' (this person's "
                    "sessions, the default) or 'all' (everyone's)",
                )
            )
    return findings


def _check_unbound_history_roles(web_terminals: Mapping[str, Any]) -> list[TelemetryFinding]:
    """``history: all`` on a role nobody can hold widens nobody's view."""
    users = web_terminals.get("users")
    # Strings only: a wrong-typed role or claim target is the web-terminal
    # lint's finding, and binds nothing here.
    bound = {
        role
        for user in (users if isinstance(users, list) else [])
        if isinstance(user, Mapping) and isinstance(role := user.get("role"), str)
    }
    claims = _mapping(_mapping(web_terminals.get("authorization")).get("claims"))
    claimed = {role for role in _mapping(claims.get("map")).values() if isinstance(role, str)}
    return [
        _warn(
            "unbound_history_role",
            f"{_WEB}.authorization.roles.{role} has history 'all', but no roster entry "
            "carries that role and no authorization.claims.map entry grants it, so it "
            "widens nobody's view. Bind the role, or remove the key",
        )
        for role, entry in _roles(web_terminals).items()
        if isinstance(entry, Mapping)
        and entry.get("history") == "all"
        and role not in bound
        and role not in claimed
    ]


# --- host identity -----------------------------------------------------------


def _check_host_identity(config: Mapping[str, Any], runtime_user: str) -> list[TelemetryFinding]:
    """Host launches send and record as the account that BUILT the deployment."""
    identity = tc.host_identity(config)
    if identity is None or identity == runtime_user:
        return []
    return [
        _warn(
            "host_identity_mismatch",
            f"deployment.host_identity is {identity!r} and this command runs as "
            f"{runtime_user!r}. Sessions started from this account are recorded and "
            f"sent as {identity!r}, the account that built the deployment. Run "
            "`osprey build` as the account that will launch them, or set "
            "deployment.host_identity",
        )
    ]
