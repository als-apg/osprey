"""Static validation for the ``modules.web_terminals`` stanza of a facility config.

Encodes validation rules 11 (port range overlap), 12 (reserved service names), and
13 (empty ``users[]`` warning/error interaction with ``modules.benchmarks``) from
``references/facility-config-schema.md``. The renderer and the build-profile
interview both derive everything from ``users[]``, so "consistency" here means: no
duplicate user names, and every user can actually be allocated a full four-family
port set via :func:`allocate_ports`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from osprey.deployment.web_terminals.ports import (
    FAMILY_BASE_FIELDS,
    allocate_ports,
    base_ports_from_config,
)

# Rule 12's closed set of reserved compose service keys. "dispatch-sidecar-*" is a
# prefix pattern (one service per sidecar), not a single literal.
_RESERVED_SERVICE_NAMES = frozenset(
    {
        "nginx",
        "ariel-postgres",
        "typesense",
        "event-dispatcher",
        "integration-tests",
        "ariel-sync",
    }
)
_RESERVED_SERVICE_PREFIX = "dispatch-sidecar-"

# Usernames become nginx `location` keys and URL path segments (`/<user>/...`), so
# they're held to a stricter charset than a bare "no reserved collision" check.
_USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The TLS seam's listener port (`listen 443 ssl` in the gated nginx block, see
# Task 1.3). `auth.method` has no other value than "none" in this schema revision
# (Task 1.4), so there's no dedicated auth-service port to reserve yet.
_TLS_LISTEN_PORT = 443


@dataclass(frozen=True)
class Finding:
    """A single lint result for a facility config."""

    severity: Literal["error", "warn"]
    code: str
    message: str


def lint_web_terminals(config: Any) -> list[Finding]:
    """Validate the ``modules.web_terminals`` stanza of a facility config.

    Args:
        config: The parsed facility config, read defensively as nested dicts (no
            assumption that ``config`` is a particular schema/dataclass type).

    Returns:
        A list of :class:`Finding` objects, empty if nothing is wrong. Findings
        with ``severity="warn"`` do not indicate a config that must be rejected.
    """
    root = _as_dict(config)
    modules = _as_dict(root.get("modules"))
    web_terminals = _as_dict(modules.get("web_terminals"))

    if not web_terminals.get("enabled"):
        return []

    users_raw = web_terminals.get("users")
    users = list(users_raw) if isinstance(users_raw, list) else []

    findings: list[Finding] = []
    findings.extend(_check_empty_users(web_terminals, modules, users))
    findings.extend(_check_duplicate_users(users))
    findings.extend(_check_reserved_names(users))
    findings.extend(_check_username_charset(users))
    findings.extend(_check_port_families_allocatable(web_terminals, users))
    findings.extend(_check_port_overlap(root, web_terminals, users))
    return findings


def _check_empty_users(
    web_terminals: dict[str, Any], modules: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Rule 13: enabled + empty users[] is a warning, unless benchmarks needs one."""
    if users:
        return []
    benchmarks_enabled = bool(_as_dict(modules.get("benchmarks")).get("enabled"))
    if benchmarks_enabled:
        return [
            Finding(
                severity="error",
                code="web_terminals.empty_users_with_benchmarks",
                message=(
                    "modules.web_terminals.users is empty but modules.benchmarks.enabled "
                    "is true; benchmarks.runs_in_container has no first user to resolve."
                ),
            )
        ]
    return [
        Finding(
            severity="warn",
            code="web_terminals.empty_users",
            message=(
                "modules.web_terminals is enabled with an empty users[]; nginx and the "
                "landing page will render with zero per-user terminal services."
            ),
        )
    ]


def _check_duplicate_users(users: list[Any]) -> list[Finding]:
    """Consistency rule: user names must be unique (each names one compose service)."""
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for user in users:
        if user in seen:
            duplicates.add(user)
        seen.add(user)
    if not duplicates:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.duplicate_user",
            message=f"modules.web_terminals.users contains duplicate name(s): {sorted(duplicates)}",
        )
    ]


def _check_reserved_names(users: list[Any]) -> list[Finding]:
    """Rule 12: a user name may not collide with a reserved compose service key."""
    findings: list[Finding] = []
    for user in users:
        is_reserved = isinstance(user, str) and (
            user in _RESERVED_SERVICE_NAMES or user.startswith(_RESERVED_SERVICE_PREFIX)
        )
        if is_reserved:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.reserved_name",
                    message=(
                        f"modules.web_terminals.users entry {user!r} collides with a "
                        "reserved service name"
                    ),
                )
            )
    return findings


def _check_username_charset(users: list[Any]) -> list[Finding]:
    """A user name becomes an nginx `location` key and a URL path segment
    (``/<user>/...``); it must match ``^[a-z0-9][a-z0-9_-]*$``."""
    findings: list[Finding] = []
    for user in users:
        if isinstance(user, str) and not _USERNAME_CHARSET_RE.match(user):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_username_charset",
                    message=(
                        f"modules.web_terminals.users entry {user!r} does not match "
                        f"{_USERNAME_CHARSET_RE.pattern!r} (usernames become nginx "
                        "location keys and URL path segments)"
                    ),
                )
            )
    return findings


def _check_port_families_allocatable(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Consistency rule: every user must resolve a full four-family port set."""
    if not users:
        return []
    base_ports = base_ports_from_config(web_terminals)
    try:
        allocate_ports(base_ports, index=0)
    except ValueError as exc:
        return [
            Finding(
                severity="error",
                code="web_terminals.incomplete_port_families",
                message=f"modules.web_terminals cannot allocate per-user ports: {exc}",
            )
        ]
    return []


def _check_port_overlap(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Rule 11: the closed set S1 (web_terminals families) ∪ S2 (event_dispatcher
    sidecars) ∪ S3 (ports.* literals) ∪ S4 (test_ioc's unmirrored ports) must be
    pairwise disjoint.

    ``nginx_port``/``event_dispatcher.port``/``custom_mcp_servers[].port`` are
    deliberately NOT added a second time here: they are each required to equal an
    existing ``ports.*`` entry, which S3 already covers. Re-adding them would make
    every valid config falsely look like it collides with its own mirror.
    """
    modules = _as_dict(root.get("modules"))
    entries: list[tuple[int, str]] = []

    # S1: web_terminals family ranges, one per family, over the N configured users.
    base_ports = base_ports_from_config(web_terminals)
    for base_field, family in FAMILY_BASE_FIELDS.items():
        base = base_ports.get(family)
        if base is None:
            continue
        for index in range(len(users)):
            entries.append((base + index, f"web_terminals.{base_field}[index={index}]"))

    # S2: event_dispatcher sidecar range.
    event_dispatcher = modules.get("event_dispatcher")
    if isinstance(event_dispatcher, dict):
        sidecar_base = event_dispatcher.get("sidecar_port_base")
        sidecar_count = event_dispatcher.get("sidecar_count")
        if isinstance(sidecar_base, int) and isinstance(sidecar_count, int):
            for index in range(sidecar_count):
                entries.append(
                    (sidecar_base + index, f"event_dispatcher.sidecar_port_base[index={index}]")
                )

    # S3: every ports.* literal.
    ports = root.get("ports")
    if isinstance(ports, dict):
        for key, value in ports.items():
            if isinstance(value, int):
                entries.append((value, f"ports.{key}"))

    # S4: test_ioc's two ports with no ports.* mirror.
    test_ioc = modules.get("test_ioc")
    if isinstance(test_ioc, dict):
        for field in ("cas_server_port", "cas_beacon_port"):
            value = test_ioc.get(field)
            if isinstance(value, int):
                entries.append((value, f"test_ioc.{field}"))

    # S5: the gated auth/TLS seam's port(s) (Task 1.3/1.4) — only join the
    # collision set when the seam is actually enabled by config; the default
    # (tls disabled, auth "none") must not reserve 443 against ordinary configs.
    tls = _as_dict(web_terminals.get("tls"))
    if bool(tls.get("enabled", False)):
        entries.append((_TLS_LISTEN_PORT, "web_terminals.tls (listen 443 ssl)"))
    # `auth.method` has no value other than "none" in this schema revision, so
    # there's no dedicated auth-service port to add yet (see Task 1.4's schema).

    by_port: dict[int, list[str]] = {}
    for port, source in entries:
        by_port.setdefault(port, []).append(source)

    findings: list[Finding] = []
    for port in sorted(by_port):
        distinct_sources = sorted(set(by_port[port]))
        if len(distinct_sources) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.port_overlap",
                    message=f"Port {port} is allocated by more than one source: "
                    f"{', '.join(distinct_sources)}",
                )
            )
    return findings


def _as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}
