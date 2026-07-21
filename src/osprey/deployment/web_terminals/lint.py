"""Static validation for the ``modules.web_terminals`` stanza of a facility config.

Encodes validation rules 11 (port range overlap), 12 (reserved service names), and
13 (empty ``users[]`` warning/error interaction with ``modules.benchmarks``) from
``references/facility-config-schema.md``. The renderer and the build-profile
interview both derive everything from ``users[]``, so "consistency" here means: no
duplicate user names, and every user can actually be allocated a full port-family
set via :func:`allocate_ports`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from osprey.deployment.web_terminals.personas import (
    SUPPORTED_MCP_TOPOLOGY,
    as_dict,
    effective_image_source,
    resolve_image_tag,
    resolve_personas,
)
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


def _is_reserved_service_name(name: str) -> bool:
    """Rule 12's predicate: collides with a reserved compose service key."""
    return name in _RESERVED_SERVICE_NAMES or name.startswith(_RESERVED_SERVICE_PREFIX)


# Usernames become nginx `location` keys and URL path segments (`/<user>/...`), so
# they're held to a stricter charset than a bare "no reserved collision" check.
_USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The TLS seam's listener port (`listen 443 ssl` in the gated nginx block, see
# Task 1.3). `auth.method` has no other value than "none" in this schema revision
# (Task 1.4), so there's no dedicated auth-service port to reserve yet.
_TLS_LISTEN_PORT = 443


@dataclass(frozen=True)
class Finding:
    """A single lint result for a facility config.

    ``severity`` is one of ``"error"`` (a config that must be rejected),
    ``"warn"`` (worth flagging, does not fail the check), or ``"info"`` (a
    non-blocking note — e.g. a persona whose ``project_path`` does not exist
    yet but is auto-renderable at deploy time).
    """

    severity: Literal["error", "warn", "info"]
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
    root = as_dict(config)
    modules = as_dict(root.get("modules"))
    web_terminals = as_dict(modules.get("web_terminals"))

    if not web_terminals.get("enabled"):
        return []

    users_raw = web_terminals.get("users")
    users = list(users_raw) if isinstance(users_raw, list) else []

    findings: list[Finding] = []
    findings.extend(_check_empty_users(web_terminals, modules, users))
    findings.extend(_check_duplicate_users(users))
    findings.extend(_check_reserved_names(users))
    findings.extend(_check_username_charset(users))
    findings.extend(_check_invalid_index(users))
    findings.extend(_check_duplicate_index(users))
    findings.extend(_check_bare_list_port_drift_risk(users))
    findings.extend(_check_port_families_allocatable(web_terminals, users))
    findings.extend(_check_port_overlap(root, web_terminals, users))
    findings.extend(_check_persona_charset(web_terminals))
    findings.extend(_check_persona_reserved_names(web_terminals))
    findings.extend(_check_default_persona_exists(web_terminals))
    findings.extend(_check_unknown_persona_reference(root, web_terminals, users))
    findings.extend(_check_empty_facility_prefix(root, web_terminals, users))
    findings.extend(_check_unknown_image_source(web_terminals))
    findings.extend(_check_image_tag_empty(web_terminals))
    findings.extend(_check_registry_url_coherence(root, web_terminals))
    findings.extend(_check_local_mode_requires_catalog(web_terminals))
    findings.extend(_check_persona_project_paths(web_terminals, users))
    findings.extend(_check_registry_mode_build_profile(web_terminals, users))
    findings.extend(_check_persona_extra_mounts(web_terminals))
    findings.extend(_check_unknown_mcp_topology(web_terminals))
    return findings


def _check_empty_users(
    web_terminals: dict[str, Any], modules: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Rule 13: enabled + empty users[] is a warning, unless benchmarks needs one."""
    if users:
        return []
    benchmarks_enabled = bool(as_dict(modules.get("benchmarks")).get("enabled"))
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
    """Consistency rule: user names must be unique (each names one compose service).

    Users may be bare strings or object-form ``{"name": ..., "index": ...}`` dicts
    (dicts aren't hashable, so identity is compared on the resolved name, not the
    raw entry — this only changes how object-form entries are compared, bare
    strings behave exactly as before).
    """
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for user in users:
        key = user.get("name") if isinstance(user, dict) else user
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if not duplicates:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.duplicate_user",
            message=f"modules.web_terminals.users contains duplicate name(s): {sorted(duplicates)}",
        )
    ]


def _user_name(user: Any) -> str | None:
    """Return a user entry's name, uniformly for either roster form.

    A bare string is its own name. An object-form entry contributes its
    ``name`` field, but only when it's actually a string — a malformed dict
    without a str name is simply skipped by the checks that use this (name
    well-formedness isn't their concern; only well-formed names are validated
    for reserved/charset collisions here).
    """
    if isinstance(user, str):
        return user
    if isinstance(user, dict):
        name = user.get("name")
        if isinstance(name, str):
            return name
    return None


def _check_reserved_names(users: list[Any]) -> list[Finding]:
    """Rule 12: a user name may not collide with a reserved compose service key."""
    findings: list[Finding] = []
    for user in users:
        name = _user_name(user)
        if name is None:
            continue
        if _is_reserved_service_name(name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.reserved_name",
                    message=(
                        f"modules.web_terminals.users entry {name!r} collides with a "
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
        name = _user_name(user)
        if name is not None and not _USERNAME_CHARSET_RE.match(name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_username_charset",
                    message=(
                        f"modules.web_terminals.users entry {name!r} does not match "
                        f"{_USERNAME_CHARSET_RE.pattern!r} (usernames become nginx "
                        "location keys and URL path segments)"
                    ),
                )
            )
    return findings


def _valid_index(user: Any) -> int | None:
    """Return an object-form user's ``index`` if it's a non-negative int, else None.

    Bool is deliberately rejected even though ``bool`` is an ``int`` subclass in
    Python — ``index: true``/``false`` is not a meaningful port offset.
    """
    if not isinstance(user, dict):
        return None
    index = user.get("index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index
    return None


def _check_invalid_index(users: list[Any]) -> list[Finding]:
    """Every object-form user's index must resolve to a real, distinct port offset."""
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict):
            continue  # bare-string entries have no index to validate
        if _valid_index(user) is not None:
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_index",
                message=(
                    f"modules.web_terminals.users entry {name!r} has an invalid "
                    f"index {user.get('index')!r}; index must be a non-negative integer"
                ),
            )
        )
    return findings


def _check_duplicate_index(users: list[Any]) -> list[Finding]:
    """Two object-form users sharing an index would collide on every per-user
    port family. Entries with an already-invalid index are skipped here — that's
    reported by :func:`_check_invalid_index` instead."""
    by_index: dict[int, list[Any]] = {}
    for user in users:
        index = _valid_index(user)
        if index is None:
            continue
        by_index.setdefault(index, []).append(user.get("name", user))

    findings: list[Finding] = []
    for index in sorted(by_index):
        names = by_index[index]
        if len(names) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.duplicate_index",
                    message=(
                        f"modules.web_terminals.users entries {names} share index "
                        f"{index}, which would collide on every per-user port family"
                    ),
                )
            )
    return findings


def _check_bare_list_port_drift_risk(users: list[Any]) -> list[Finding]:
    """A legacy bare-string roster has no explicit index — each user's ports are
    derived from list position. Removing a mid-list user shifts every survivor
    after it onto a different port. A single-user list has no such drift risk,
    and a roster that already uses (even partially) explicit indices is exempt."""
    if len(users) <= 1:
        return []
    if any(isinstance(user, dict) for user in users):
        return []
    return [
        Finding(
            severity="warn",
            code="web_terminals.bare_list_port_drift_risk",
            message=(
                "modules.web_terminals.users is a legacy bare-string list with more "
                "than one user; removing a mid-list user will shift every "
                "subsequent user's ports. Migrate to explicit {name, index} entries "
                "before decommissioning a user."
            ),
        )
    ]


def _check_port_families_allocatable(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Consistency rule: every user must resolve a full port-family set (the
    ``web`` family plus one family per registry companion server — see
    ``ports.FAMILY_BASE_FIELDS``). Companion families carry registry defaults,
    so in practice only a missing ``web_base_port`` can fail this."""
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
    modules = as_dict(root.get("modules"))
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
    tls = as_dict(web_terminals.get("tls"))
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


# --- Task 2.3: persona catalog identity/reference checks --------------------
#
# Note on duplicate catalog keys: `modules.web_terminals.personas` arrives here
# already parsed by `yaml.safe_load`, which silently collapses a YAML mapping's
# duplicate keys down to the last-declared value — by the time this module
# sees a Python dict, a duplicate `personas:` key has already vanished without
# a trace (no exception, no marker to detect). There is nothing observable
# post-load, so no duplicate-catalog-key check exists here.
#
# Mode-coherence checks (image_source/registry.url agreement, project_path /
# Dockerfile / config.yml existence, build_profile requirements) are below,
# layered on top of `_persona_catalog` and the checks above (Task 2.4).


def _persona_catalog(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """Read ``modules.web_terminals.personas``, defensively as a dict."""
    catalog = web_terminals.get("personas")
    return catalog if isinstance(catalog, dict) else {}


def _check_persona_charset(web_terminals: dict[str, Any]) -> list[Finding]:
    """A persona catalog key becomes an image-tag suffix
    (``web-terminal-<persona>:latest``) and a path component (``/app/<project>``,
    local-mode image tags); it's held to the same charset as usernames,
    ``^[a-z0-9][a-z0-9_-]*$`` (see :func:`_check_username_charset`)."""
    findings: list[Finding] = []
    for persona_name in _persona_catalog(web_terminals):
        if isinstance(persona_name, str) and not _USERNAME_CHARSET_RE.match(persona_name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_persona_charset",
                    message=(
                        f"modules.web_terminals.personas key {persona_name!r} does not "
                        f"match {_USERNAME_CHARSET_RE.pattern!r} (persona names become "
                        "image-tag suffixes and path components)"
                    ),
                )
            )
    return findings


def _check_persona_reserved_names(web_terminals: dict[str, Any]) -> list[Finding]:
    """A persona catalog key may not collide with a reserved compose service
    name, the same closed set held over usernames (see
    :func:`_check_reserved_names`)."""
    findings: list[Finding] = []
    for persona_name in _persona_catalog(web_terminals):
        if not isinstance(persona_name, str):
            continue
        if _is_reserved_service_name(persona_name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.persona_reserved_name",
                    message=(
                        f"modules.web_terminals.personas key {persona_name!r} collides "
                        "with a reserved service name"
                    ),
                )
            )
    return findings


def _check_default_persona_exists(web_terminals: dict[str, Any]) -> list[Finding]:
    """``default_persona``, when set, must name an entry in the persona catalog
    — the entry every roster user with no ``persona:`` of its own inherits."""
    default_persona = web_terminals.get("default_persona")
    if not isinstance(default_persona, str) or not default_persona:
        return []
    if default_persona in _persona_catalog(web_terminals):
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_default_persona",
            message=(
                f"modules.web_terminals.default_persona {default_persona!r} has no "
                "entry in modules.web_terminals.personas"
            ),
        )
    ]


def _check_unknown_persona_reference(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Every roster entry's effective persona reference (its own ``persona:``
    key, else the inherited ``default_persona``) must name a catalog entry.

    Resolves via :func:`resolve_personas`'s lenient path (``strict=False``) —
    the same function the render path calls with ``strict=True`` — so an
    unresolvable reference degrades to a reportable :class:`Finding` here
    instead of raising ``ValueError``.
    """
    if not users:
        return []
    personas_catalog = _persona_catalog(web_terminals)
    facility_prefix = as_dict(root.get("facility")).get("prefix") or ""
    registry_cfg = as_dict(root.get("registry"))
    resolved = resolve_personas(web_terminals, registry_cfg, facility_prefix, strict=False)

    findings: list[Finding] = []
    for entry in resolved:
        persona_ref = entry["persona"]
        if persona_ref is not None and persona_ref not in personas_catalog:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.unknown_persona_reference",
                    message=(
                        f"modules.web_terminals user {entry['name']!r} references "
                        f"persona {persona_ref!r}, which has no entry in "
                        "modules.web_terminals.personas"
                    ),
                )
            )
    return findings


def _check_empty_facility_prefix(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Every web container name is derived from ``facility.prefix``:
    ``<prefix>-nginx`` and ``<prefix>-web-<user>`` (see the compose template /
    :mod:`osprey.deployment.web_terminals.seeding`). An empty prefix renders
    leading-dash names like ``-nginx``, which Docker rejects — and only at
    ``deploy up``, which never runs this lint pass. This check pulls that
    failure forward to lint/build time.

    The effective prefix is derived exactly as ``render.py`` derives it
    (``facility.get("prefix") or ""``). Scoped to a configured roster — an
    empty ``users[]`` renders no per-user services and is handled by
    :func:`_check_empty_users` instead.
    """
    if not users:
        return []
    facility_prefix = as_dict(root.get("facility")).get("prefix") or ""
    if facility_prefix:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.empty_facility_prefix",
            message=(
                "modules.web_terminals has users configured but the effective "
                "facility.prefix is empty; web container names render as "
                "'-nginx'/'-web-<user>', which Docker rejects at deploy up"
            ),
        )
    ]


# --- Task 2.4: mode-coherence checks ----------------------------------------

# The two recognized `modules.web_terminals.image_source` values (schema rule
# 14). Anything else is `_check_unknown_image_source`'s ERROR.
_VALID_IMAGE_SOURCES = frozenset({"registry", "local"})


def _check_unknown_image_source(web_terminals: dict[str, Any]) -> list[Finding]:
    """``image_source``, when set, must be one of the two recognized modes."""
    value = web_terminals.get("image_source")
    if value is None or value in _VALID_IMAGE_SOURCES:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_image_source",
            message=(
                f"modules.web_terminals.image_source {value!r} is not a recognized "
                f"value; expected one of {sorted(_VALID_IMAGE_SOURCES)}"
            ),
        )
    ]


def _check_image_tag_empty(web_terminals: dict[str, Any]) -> list[Finding]:
    """Registry mode bakes ``modules.web_terminals.image_tag`` literally into
    every pulled image ref (``web-terminal:<tag>``). When the field references a
    ``${VAR}`` that is unset at lint/render time (or is otherwise empty), it
    resolves to an empty string and the ref degrades to a tagless
    ``web-terminal:`` no registry can pull. Warn so the misconfiguration
    surfaces here rather than at ``deploy up``. Scoped to registry mode — local
    mode builds ``:local`` images and never reads ``image_tag``."""
    if effective_image_source(web_terminals) != "registry":
        return []
    if resolve_image_tag(web_terminals):
        return []
    return [
        Finding(
            severity="warn",
            code="web_terminals.empty_image_tag",
            message=(
                "modules.web_terminals.image_tag resolves to an empty string "
                "(likely a ${VAR} whose variable is unset at render time); the "
                "rendered image ref would be a tagless 'web-terminal:' that no "
                "registry can pull"
            ),
        )
    ]


def _check_registry_url_coherence(
    root: dict[str, Any], web_terminals: dict[str, Any]
) -> list[Finding]:
    """Rule 14: ``image_source``/``registry.url`` agreement.

    Only evaluated once a persona catalog is actually configured. A config
    with no ``personas:`` block at all resolves every user through
    :func:`~osprey.deployment.web_terminals.personas.resolve_personas`'s
    zero-migration path exactly as it did before catalogs/mode-coherence
    existed — this check does not retroactively demand a ``registry.url`` from
    deployments that never opted into the persona system.
    """
    if not _persona_catalog(web_terminals):
        return []
    registry_url = as_dict(root.get("registry")).get("url")
    has_url = isinstance(registry_url, str) and bool(registry_url)
    image_source = effective_image_source(web_terminals)
    if image_source == "registry" and not has_url:
        return [
            Finding(
                severity="error",
                code="web_terminals.registry_mode_missing_url",
                message=(
                    "modules.web_terminals.image_source is 'registry' (the "
                    "default) but registry.url is not set; registry mode needs "
                    "it to pull every persona's image"
                ),
            )
        ]
    if image_source == "local" and has_url:
        return [
            Finding(
                severity="warn",
                code="web_terminals.local_mode_unused_registry_url",
                message=(
                    "modules.web_terminals.image_source is 'local' but "
                    f"registry.url is set to {registry_url!r}; local mode builds "
                    "every persona's image and never reads registry.url"
                ),
            )
        ]
    return []


def _check_local_mode_requires_catalog(web_terminals: dict[str, Any]) -> list[Finding]:
    """Rule 14: the lint-side mirror of
    :func:`~osprey.deployment.web_terminals.personas.resolve_personas`'s
    ``strict=True`` ``ValueError`` guard. ``deploy up`` never runs the lint
    pass, so both guards must independently fail closed on ``image_source:
    local`` without a catalog + ``default_persona``."""
    if effective_image_source(web_terminals) != "local":
        return []
    default_persona = web_terminals.get("default_persona")
    has_default = isinstance(default_persona, str) and bool(default_persona)
    if _persona_catalog(web_terminals) and has_default:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.local_mode_requires_catalog",
            message=(
                "modules.web_terminals.image_source is 'local', which requires "
                "both a non-empty modules.web_terminals.personas catalog and "
                "default_persona to be configured"
            ),
        )
    ]


def _referenced_persona_names(web_terminals: dict[str, Any], users: list[Any]) -> set[str]:
    """Every persona name actually in play for this roster: ``default_persona``
    plus each roster entry's own explicit ``persona:`` override. A catalog
    entry nobody references sits outside every check below — an unused draft
    entry with a broken ``project_path`` never blocks a deploy, since only
    referenced personas are ever built or pulled."""
    names: set[str] = set()
    default_persona = web_terminals.get("default_persona")
    if isinstance(default_persona, str) and default_persona:
        names.add(default_persona)
    for user in users:
        if isinstance(user, dict):
            persona = user.get("persona")
            if isinstance(persona, str) and persona:
                names.add(persona)
    return names


def _read_project_name(config_yml_path: Path) -> str | None:
    """Best-effort read of a persona project's own ``config.yml``
    ``project_name``. Any failure to open or parse degrades to ``None`` rather
    than raising — an unreadable ``config.yml`` is already its own ERROR (see
    :func:`_check_persona_project_paths`, which only calls this once the file
    is confirmed to exist)."""
    try:
        with config_yml_path.open("r", encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("project_name")
    return name if isinstance(name, str) and name else None


def _check_persona_project_paths(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """Local mode: validate every referenced persona's ``project_path``.

    ``project_path`` names the directory ``osprey deploy up`` builds a persona's
    image from. Two invariants are enforced here:

    * **Name invariant.** When ``project_path`` is set, its basename must equal
      the catalog entry's ``project``. Persona auto-render derives its output
      directory as ``<output_dir>/<project>`` (``build_cmd`` resolves
      ``output_path / project_name``), so a basename that disagrees with
      ``project`` would render into one directory while the catalog builds/mounts
      another — a dead path at runtime. A mismatch is an ERROR regardless of
      whether the directory exists yet.
    * **Existence.** The directory must exist and hold a ``Dockerfile`` and a
      ``config.yml`` whose own ``project_name`` equals the catalog ``project``
      (a mismatch silently produces a dead mount, since the per-svc
      ``container_project_dir`` derivation is keyed on the catalog's ``project``,
      not on anything read from the persona's own ``config.yml``).

    Existence is relaxed for auto-render: a ``project_path`` that does not exist
    yet but whose entry carries a usable ``build_profile`` is only an
    informational finding ("missing but auto-renderable"), since ``deploy up``
    renders it from that profile before building. A *partially* rendered
    directory that exists but is missing its ``Dockerfile``/``config.yml`` stays
    an ERROR — auto-render never overwrites an existing directory.
    """
    if effective_image_source(web_terminals) != "local":
        return []
    catalog = _persona_catalog(web_terminals)
    findings: list[Finding] = []
    for persona_name in sorted(_referenced_persona_names(web_terminals, users)):
        entry = catalog.get(persona_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — _check_unknown_persona_reference /
            # _check_default_persona_exists already report this
        findings.extend(_check_one_persona_project_path(persona_name, entry))
    return findings


def _check_one_persona_project_path(persona_name: str, entry: dict[str, Any]) -> list[Finding]:
    """The per-persona body of :func:`_check_persona_project_paths`: validate one
    catalog entry's ``project_path``. Early returns short-circuit the later
    checks exactly where a failed prerequisite makes them meaningless (see the
    parent's docstring for the invariants)."""
    catalog_project = entry.get("project")
    has_catalog_project = isinstance(catalog_project, str) and bool(catalog_project)
    build_profile = entry.get("build_profile")
    has_build_profile = isinstance(build_profile, str) and bool(build_profile)

    project_path_raw = entry.get("project_path")
    if not isinstance(project_path_raw, str) or not project_path_raw:
        return [
            Finding(
                severity="error",
                code="web_terminals.persona_missing_project_path",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}] has no "
                    "project_path set; image_source: local requires one to "
                    "build this persona's image from"
                ),
            )
        ]

    project_path = Path(project_path_raw)

    # Name invariant: auto-render writes into <output_dir>/<project>, so
    # project_path's basename must equal the catalog `project`. A
    # disagreement is a hard config error regardless of whether the
    # directory exists yet, and supersedes every existence check below —
    # there is nothing else about this persona worth reporting on top of it.
    if has_catalog_project and project_path.name != catalog_project:
        return [
            Finding(
                severity="error",
                code="web_terminals.persona_project_path_name_mismatch",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}].project_path "
                    f"{project_path_raw!r} has basename {project_path.name!r}, which "
                    f"does not match its project {catalog_project!r}; auto-render "
                    "derives the output directory from project, so the two must agree"
                ),
            )
        ]

    if not project_path.is_dir():
        # Missing directory: only auto-renderable (info) when a build_profile
        # can render it, otherwise the pre-existing hard error.
        if has_build_profile:
            return [
                Finding(
                    severity="info",
                    code="web_terminals.persona_project_path_auto_renderable",
                    message=(
                        f"modules.web_terminals.personas[{persona_name!r}].project_path "
                        f"{project_path_raw!r} does not exist yet, but the entry has a "
                        f"build_profile {build_profile!r}; deploy up will render it "
                        "before building"
                    ),
                )
            ]
        return [
            Finding(
                severity="error",
                code="web_terminals.persona_project_path_not_dir",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}]."
                    f"project_path {project_path_raw!r} does not exist or is "
                    "not a directory"
                ),
            )
        ]

    findings: list[Finding] = []
    if not (project_path / "Dockerfile").is_file():
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.persona_missing_dockerfile",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}]."
                    f"project_path {project_path_raw!r} has no Dockerfile; "
                    "local mode builds each persona's image from its own "
                    "project directory"
                ),
            )
        )

    config_yml_path = project_path / "config.yml"
    if not config_yml_path.is_file():
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.persona_missing_config_yml",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}]."
                    f"project_path {project_path_raw!r} has no config.yml"
                ),
            )
        )
        return findings  # nothing to compare `project` against

    if not has_catalog_project:
        return findings  # entry.project itself unset — not this check's concern
    rendered_project_name = _read_project_name(config_yml_path)
    if rendered_project_name is not None and rendered_project_name != catalog_project:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.persona_project_mismatch",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}].project "
                    f"{catalog_project!r} does not match its project_path's "
                    f"config.yml project_name {rendered_project_name!r}"
                ),
            )
        )
    return findings


def _check_registry_mode_build_profile(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Registry mode only: every referenced non-default persona must set
    ``build_profile`` — the committed profile YAML that feeds its one
    ``.gitlab-ci.yml`` build job. The default persona is exempt: its image
    stays the un-suffixed ``web-terminal:latest``, built by the pre-existing
    core CI job, not a per-persona one."""
    if effective_image_source(web_terminals) != "registry":
        return []
    catalog = _persona_catalog(web_terminals)
    default_persona = web_terminals.get("default_persona")
    findings: list[Finding] = []
    for persona_name in sorted(_referenced_persona_names(web_terminals, users)):
        if persona_name == default_persona:
            continue
        entry = catalog.get(persona_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        build_profile = entry.get("build_profile")
        if isinstance(build_profile, str) and build_profile:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.persona_missing_build_profile",
                message=(
                    f"modules.web_terminals.personas[{persona_name!r}] has no "
                    "build_profile set; image_source: registry needs one to "
                    "generate this non-default persona's CI build job"
                ),
            )
        )
    return findings


def _is_valid_mount_string(value: Any) -> bool:
    """A compose bind/volume mount string: 2 or 3 non-empty ``:``-separated parts
    (``source:target`` or ``source:target:mode``, e.g. ``/opt/data:/app/data:ro``)."""
    if not isinstance(value, str):
        return False
    parts = value.split(":")
    return len(parts) in (2, 3) and all(parts)


def _check_persona_extra_mounts(web_terminals: dict[str, Any]) -> list[Finding]:
    """Every ``modules.web_terminals.personas.<name>.extra_mounts`` entry must be a
    compose volume string (2 or 3 non-empty colon-separated parts). These generic
    host-path mounts are applied to every user of that persona, so a malformed
    entry would render a broken per-user ``volumes:`` line — reject it here. The
    ``extra_mounts`` key is optional; an entry that omits it is never flagged."""
    findings: list[Finding] = []
    for persona_name, entry in _persona_catalog(web_terminals).items():
        if not isinstance(entry, dict):
            continue
        raw = entry.get("extra_mounts")
        if raw is None:
            continue
        if not isinstance(raw, list):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.persona_extra_mounts_not_list",
                    message=(
                        f"modules.web_terminals.personas[{persona_name!r}].extra_mounts "
                        f"must be a list of compose volume strings, got {type(raw).__name__}"
                    ),
                )
            )
            continue
        for mount in raw:
            if not _is_valid_mount_string(mount):
                findings.append(
                    Finding(
                        severity="error",
                        code="web_terminals.persona_invalid_extra_mount",
                        message=(
                            f"modules.web_terminals.personas[{persona_name!r}]."
                            f"extra_mounts entry {mount!r} is not a valid compose volume "
                            "string; expected 'source:target' or 'source:target:mode' "
                            "with non-empty colon-separated parts"
                        ),
                    )
                )
    return findings


def _check_unknown_mcp_topology(web_terminals: dict[str, Any]) -> list[Finding]:
    """Lint-side mirror of render.py's ``_check_mcp_topology`` fail-closed
    ``ValueError`` (Task 2.5) — ``shared_http`` and any other unrecognized
    value are an ERROR here too, so a bad topology value is caught before a
    render/deploy attempt rather than only at render time."""
    mcp_cfg = as_dict(web_terminals.get("mcp"))
    topology = mcp_cfg.get("topology") or SUPPORTED_MCP_TOPOLOGY
    if topology == SUPPORTED_MCP_TOPOLOGY:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_mcp_topology",
            message=(
                f"modules.web_terminals.mcp.topology {topology!r} is not wired "
                "yet for the shared framework-MCP tier; per_container_stdio is "
                "the only supported topology (a facility's own "
                "claude_code.servers custom `url` entries are a separate, "
                "already-supported path and are unaffected)"
            ),
        )
    ]
