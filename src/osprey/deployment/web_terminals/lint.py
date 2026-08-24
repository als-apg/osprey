"""Static validation for the ``modules.web_terminals`` multi-user stanza.

The multi-user web stack is derived entirely from ``users[]`` and the persona
catalog beside it, so "consistent" here means: every user resolves to exactly one
container with its own port family, every persona reference resolves to a catalog
entry, and nothing in the stack contends with another service for a host port.

Two surfaces call in, at different altitudes:

* :func:`lint_web_terminals` reads a **rendered project ``config.yml``** — the
  deploy-time view, where the ``services:`` block names every published host port
  and every persona project either exists on disk or is one ``osprey build``
  away. ``osprey scaffold web-terminals lint|render`` runs this one.
* :func:`lint_profile_config` reads a **build profile's ``config:`` block** — the
  authoring-time view, before anything is materialized. It runs the same checks
  minus the ones that can only be answered against a rendered project (see that
  function's docstring), so a profile can be validated without building it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from osprey.deployment.web_terminals.persona_images import persona_build_profile_shape_problem
from osprey.deployment.web_terminals.personas import (
    SUPPORTED_MCP_TOPOLOGY,
    USERNAME_CHARSET_RE,
    as_dict,
    effective_image_source,
    env_var_suffix_collisions,
    resolve_image_tag,
    resolve_personas,
)
from osprey.deployment.web_terminals.ports import (
    FAMILY_BASE_FIELDS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.deployment.web_terminals.render import (
    SUPPORTED_AUTH_METHODS,
    _auth_tls_context,
    _configured_external_origin,
    _external_origin,
)

# The TLS seam's listener port (`listen 443 ssl` in the gated nginx block). The
# auth sidecar's listener has no constant here: it is config-driven
# (`auth.port`), and its effective value is read from render's parsed auth
# context rather than restated.
_TLS_LISTEN_PORT = 443

# The credential env-var stem a roster username is keyed into
# (`OSPREY_AUTH_PW_HASH_<SUFFIX>`), quoted only inside this module's collision
# message. It is deliberately NOT imported from `auth_credentials`, which owns
# the constant: this module is pure static validation of a config file, and
# importing the credential provisioner to quote one string in a message would
# pull the whole deploy-time secret-minting path in behind it.
_PW_HASH_VAR_PREFIX = "OSPREY_AUTH_PW_HASH_"


@dataclass(frozen=True)
class Finding:
    """A single lint result.

    ``severity`` is ``"error"`` (a config that must be rejected) or ``"warn"``
    (worth flagging, does not fail the check — e.g. a persona whose
    ``project_path`` has not been rendered yet, which no start will run past but
    an ``osprey build`` clears).

    There is deliberately no third, purely informational level. Every check here
    answers "will this config deploy", so a finding either blocks or names work
    the operator has to do; a level that meant neither would be a way to report
    a problem without owning whether it is one.
    """

    severity: Literal["error", "warn"]
    code: str
    message: str


def lint_web_terminals(config: Any, *, rendered_project: bool = True) -> list[Finding]:
    """Validate a rendered project config's ``modules.web_terminals`` stanza.

    Args:
        config: The parsed project config, read defensively as nested dicts (no
            assumption that ``config`` is a particular schema/dataclass type).
        rendered_project: Whether *config* describes a project that has actually
            been rendered. False drops the two checks that can only be answered
            once it has — see :func:`lint_profile_config`, the only caller that
            passes it.

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
    findings.extend(_check_empty_users(users))
    findings.extend(_check_duplicate_users(users))
    findings.extend(_check_username_charset(users))
    findings.extend(_check_display_name(users))
    findings.extend(_check_user_theme(users))
    findings.extend(_check_user_login(web_terminals, users))
    findings.extend(_check_invalid_index(users))
    findings.extend(_check_duplicate_index(users))
    findings.extend(_check_bare_list_port_drift_risk(users))
    findings.extend(_check_port_families_allocatable(web_terminals, users))
    findings.extend(_check_port_overlap(root, web_terminals, users))
    findings.extend(_check_persona_charset(web_terminals))
    findings.extend(_check_persona_seed_base(web_terminals))
    findings.extend(_check_default_persona_exists(web_terminals))
    findings.extend(_check_unknown_persona_reference(root, web_terminals, users))
    findings.extend(_check_empty_facility_prefix(root, web_terminals, users))
    findings.extend(_check_unknown_image_source(web_terminals))
    findings.extend(_check_image_tag_empty(web_terminals))
    findings.extend(_check_registry_url_coherence(root, web_terminals))
    findings.extend(_check_local_mode_requires_catalog(web_terminals))
    if rendered_project:
        findings.extend(_check_persona_project_paths(web_terminals, users))
        # Reads each persona's rendered config.yml, so it rides on the same
        # gate: a profile has no rendered project to compare against yet.
        findings.extend(_check_persona_bundle_path_agreement(root, web_terminals, users))
        # Resolves notice paths against the project directory, so it rides the
        # same gate: a profile has no rendered project to look in yet.
        findings.extend(_check_notice_docs(root, web_terminals))
    findings.extend(_check_registry_mode_build_profile(web_terminals, users))
    findings.extend(_check_persona_extra_mounts(web_terminals))
    findings.extend(_check_unknown_mcp_topology(web_terminals))
    findings.extend(_check_nginx_image(web_terminals))
    findings.extend(_check_external_origin(web_terminals))
    findings.extend(_check_auth_method(web_terminals))
    findings.extend(_check_auth_transport(root, web_terminals))
    findings.extend(_check_auth_oidc(root, web_terminals))
    findings.extend(_check_auth_credential_collisions(web_terminals, users))
    return findings


def lint_profile_config(config: Mapping[str, Any]) -> list[Finding]:
    """Validate the ``modules.web_terminals`` a build profile's ``config:`` sets.

    A ``config:`` block is a flat bag of dotted keys (``modules.web_terminals``,
    ``facility.prefix``, ``services.openobserve.port``, …) applied over the
    rendered template, so it is nested into the shape the checks read before
    linting. Shallowest key first, so a deeper key refines the subtree a
    shallower one set rather than being overwritten by it — the same order
    ``osprey build`` applies them in.

    Two checks are skipped, because a profile cannot answer them yet:

    * **Persona project paths.** ``project_path`` resolves against the rendered
      project's directory, which does not exist at authoring time.
    * **``build_profile`` shape**, which rides on the same check.
      ``osprey init`` rewrites each catalog entry's preset name into the
      ``personas/<name>.yml`` delta it materializes beside the profile, so the
      value is only in its final form once that has happened.

    Both are enforced in full by :func:`lint_web_terminals` at deploy time.
    Cross-service port overlap is likewise checked only against the host ports
    the ``config:`` block itself declares; a service declared at profile top
    level joins the collision set once the project is rendered.

    Call this from a COMMAND, never from ``BuildProfile.validate()``. That
    method also runs during profile *resolution*, which ``osprey init``
    goes through, and these findings would then pre-empt that command's own
    persona validator — which reports every unusable catalog entry at once and
    names the file-name rule a persona name really has to meet. The engine
    would replace a better error with a worse one.
    """
    return lint_web_terminals(_nest_dotted(config), rendered_project=False)


def profile_config_errors(config: Mapping[str, Any]) -> list[str]:
    """The messages from :func:`lint_profile_config` that must fail a command.

    One home for "which severity blocks", so the surfaces that gate on it
    cannot drift apart: warnings and informational findings are advisory and
    never stop a build.
    """
    return [
        finding.message for finding in lint_profile_config(config) if finding.severity == "error"
    ]


def _merge_nested(into: dict[str, Any], value: Mapping[str, Any]) -> None:
    """Deep-merge ``value`` into ``into`` in place; ``value`` wins on conflict."""
    for key, item in value.items():
        current = into.get(key)
        if isinstance(current, dict) and isinstance(item, Mapping):
            _merge_nested(current, item)
        else:
            into[key] = copy.deepcopy(item)


def _nest_dotted(config: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a flat bag of dotted keys into the nested config it addresses."""
    root: dict[str, Any] = {}
    dotted = [key for key in config if isinstance(key, str)]
    for key in sorted(dotted, key=lambda k: k.count(".")):
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        value = config[key]
        current = node.get(parts[-1])
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_nested(current, value)
        else:
            node[parts[-1]] = copy.deepcopy(value)
    return root


def _check_empty_users(users: list[Any]) -> list[Finding]:
    """An enabled module with an empty roster renders no terminals — worth flagging."""
    if users:
        return []
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
    raw entry, so a bare string and an object entry naming the same user
    collide, and an all-strings roster compares unchanged).
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
    well-formedness isn't their concern; only well-formed names are held to the
    charset).
    """
    if isinstance(user, str):
        return user
    if isinstance(user, dict):
        name = user.get("name")
        if isinstance(name, str):
            return name
    return None


def _check_username_charset(users: list[Any]) -> list[Finding]:
    """A user name becomes an nginx `location` key and a URL path segment
    (``/u/<user>/...``); it must match ``^[a-z0-9][a-z0-9_-]*$``."""
    findings: list[Finding] = []
    for user in users:
        name = _user_name(user)
        if name is not None and not USERNAME_CHARSET_RE.fullmatch(name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_username_charset",
                    message=(
                        f"modules.web_terminals.users entry {name!r} does not match "
                        f"{USERNAME_CHARSET_RE.pattern!r} (usernames become nginx "
                        "location keys and URL path segments)"
                    ),
                )
            )
    return findings


def _check_display_name(users: list[Any]) -> list[Finding]:
    """An object-form entry's optional ``display_name`` (the per-user window/tab
    title emitted as ``OSPREY_WEB_APP_NAME``) must be a string when present.

    The renderer reads it defensively (:func:`resolve_personas` drops a non-string
    one rather than emitting a broken env line), so a bad value degrades silently
    at render time; this check pulls that config typo forward to lint/build time
    as an ERROR. An empty string is a well-formed (if inert) value — the template
    guards on truthiness and simply emits no env line — so it is not flagged here.
    """
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict) or "display_name" not in user:
            continue
        display_name = user.get("display_name")
        if isinstance(display_name, str):
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_display_name",
                message=(
                    f"modules.web_terminals.users entry {name!r} has a non-string "
                    f"display_name {display_name!r}; display_name must be a string"
                ),
            )
        )
    return findings


def _check_user_theme(users: list[Any]) -> list[Finding]:
    """An object-form entry's optional ``theme`` (the per-user default web UI
    theme emitted as ``OSPREY_WEB_THEME``) must be a string when present.

    Same shape and rationale as :func:`_check_display_name`: the renderer drops
    a non-string one rather than emitting a broken env line, so a config typo
    would otherwise degrade silently at render time.

    Only the *type* is checked here. Whether the string names a real theme
    family or id is deliberately not lint's business: the theme registry lives
    in the design system, is versioned with the image rather than with this
    config, and the web terminal already warns and falls back on an unknown
    value at startup. Failing a build over a name this module cannot
    authoritatively resolve would be worse than that warning.
    """
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict) or "theme" not in user:
            continue
        theme = user.get("theme")
        if isinstance(theme, str):
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_user_theme",
                message=(
                    f"modules.web_terminals.users entry {name!r} has a non-string "
                    f"theme {theme!r}; theme must be a string (a theme family such "
                    f"as 'desy', or a concrete id such as 'desy-light')"
                ),
            )
        )
    return findings


def _check_user_login(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """An object-form entry's optional ``login`` must be a boolean, and only
    does anything while authentication is on.

    Two findings, for the two ways the key can lie:

    * A non-boolean ``login`` is an ERROR. The normalizer reads anything but
      the literal ``false`` as "login required" — deliberately fail-closed —
      so the typo can never open an entry to the world, but the author who
      wrote ``login: no-thanks`` believes the opposite of what deploys.
    * ``login: false`` with ``auth.method: none`` is a WARN. The key is inert
      (there is no login wall to be exempt from), so the config claims a
      distinction the deployment does not have.
    """
    findings: list[Finding] = []
    login_declared = False
    for user in users:
        if not isinstance(user, dict) or "login" not in user:
            continue
        login = user.get("login")
        if isinstance(login, bool):
            login_declared = login_declared or login is False
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_user_login",
                message=(
                    f"modules.web_terminals.users entry {name!r} has a non-boolean "
                    f"login {login!r}; login must be true (default: this entry "
                    f"requires a login) or false (served without authentication). "
                    f"Anything else deploys as 'login required'"
                ),
            )
        )

    context = _auth_context(web_terminals)
    auth_off = context is None or context["auth_method"] == "none"
    if login_declared and auth_off:
        findings.append(
            Finding(
                severity="warn",
                code="web_terminals.user_login_inert",
                message=(
                    "a modules.web_terminals.users entry sets login: false, but "
                    "auth.method is 'none' so no entry has a login to be exempt "
                    "from; the key changes nothing until authentication is enabled"
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


def _is_host_port_key(key: Any) -> bool:
    """Whether a ``services.<name>`` key names a port published on the host.

    ``port`` and ``port_host`` are the two spellings the service templates use
    for a published port, plus per-service qualified ones (``tiled_port``). A
    container-internal listener is spelled differently on purpose (the dispatch
    worker's ``worker_port_base`` reaches the dispatcher over the compose
    network and binds nothing on the host), and must stay out of this set: a
    port nobody publishes cannot collide with one that is.
    """
    return isinstance(key, str) and (key in ("port", "port_host") or key.endswith("_port"))


def _check_port_overlap(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Every host port the deployment binds must be bound by exactly one thing.

    The web stack runs under ``network_mode: host`` — each per-user container
    and nginx bind their ports on the host directly — so they contend not only
    with each other but with every port the ``services:`` block publishes. The
    collision set is therefore: the per-user port families over the N configured
    users, nginx's own listener, every published service port, and the TLS
    listener when that seam is enabled.
    """
    entries: list[tuple[int, str]] = []

    # Per-user families: one range per family, over the N configured users.
    base_ports = base_ports_from_config(web_terminals)
    for base_field, family in FAMILY_BASE_FIELDS.items():
        base = base_ports.get(family)
        if base is None:
            continue
        for index in range(len(users)):
            entries.append((base + index, f"web_terminals.{base_field}[index={index}]"))

    # nginx's own listener, the one port the stack is reached on from off-host.
    nginx_port = web_terminals.get("nginx_port")
    if isinstance(nginx_port, int) and not isinstance(nginx_port, bool):
        entries.append((nginx_port, "web_terminals.nginx_port"))

    # Every host port the deployed services publish.
    services = root.get("services")
    if isinstance(services, dict):
        for name, spec in services.items():
            if not isinstance(spec, dict):
                continue
            for key, value in spec.items():
                if (
                    _is_host_port_key(key)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    entries.append((value, f"services.{name}.{key}"))

    # S4: test_ioc's two ports with no ports.* mirror.
    test_ioc = as_dict(root.get("modules")).get("test_ioc")
    if isinstance(test_ioc, dict):
        for field in ("cas_server_port", "cas_beacon_port"):
            value = test_ioc.get(field)
            if isinstance(value, int):
                entries.append((value, f"test_ioc.{field}"))

    # S5: the gated auth/TLS seam's port(s) — only join the collision set when
    # the seam is actually enabled by config; the default (tls disabled, auth
    # "none") must not reserve 443 or the sidecar port against ordinary configs.
    tls = as_dict(web_terminals.get("tls"))
    if bool(tls.get("enabled", False)):
        entries.append((_TLS_LISTEN_PORT, "web_terminals.tls (listen 443 ssl)"))
    auth_context = _auth_context(web_terminals)
    if auth_context is not None and auth_context["auth_method"] != "none":
        # The sidecar's own listener, published on the host beside every other
        # service in the stack. Unlike `nginx_port` it has no `ports.*` mirror
        # to be covered by S3 — `auth.port` is where it is declared — so it is
        # added here directly, exactly like the TLS listener above.
        entries.append((auth_context["auth_port"], "web_terminals.auth.port"))

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


# --- persona catalog identity/reference checks ------------------------------
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
# layered on top of `_persona_catalog` and the checks above.


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
        if isinstance(persona_name, str) and not USERNAME_CHARSET_RE.fullmatch(persona_name):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_persona_charset",
                    message=(
                        f"modules.web_terminals.personas key {persona_name!r} does not "
                        f"match {USERNAME_CHARSET_RE.pattern!r} (persona names become "
                        "image-tag suffixes and path components)"
                    ),
                )
            )
    return findings


def _check_persona_seed_base(web_terminals: dict[str, Any]) -> list[Finding]:
    """A persona catalog entry's ``seed_base``, when present, must be a boolean.

    ``seed_base`` opts a persona's users out of the mandatory shared base-context
    prepend at seed time (default ``true``). A non-boolean value (e.g. a YAML
    string ``"false"``) is a config typo — ``resolve_personas`` defensively
    coerces it back to ``True``, so without this check the opt-out would silently
    not take effect. Reported here so the mistake fails the config instead."""
    findings: list[Finding] = []
    for persona_name, entry in _persona_catalog(web_terminals).items():
        if not isinstance(entry, dict) or "seed_base" not in entry:
            continue
        value = entry["seed_base"]
        if not isinstance(value, bool):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.persona_invalid_seed_base",
                    message=(
                        f"modules.web_terminals.personas[{persona_name!r}].seed_base "
                        f"{value!r} is not a boolean; seed_base must be true or false"
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
    ``osprey up``, which never runs this lint pass. This check pulls that
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
                "'-nginx'/'-web-<user>', which Docker rejects at `osprey up`"
            ),
        )
    ]


# --- mode-coherence checks --------------------------------------------------

# The two recognized `modules.web_terminals.image_source` values. Anything else
# is `_check_unknown_image_source`'s ERROR.
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
    surfaces here rather than at ``osprey up``. Scoped to registry mode — local
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
    """``image_source`` and ``registry.url`` must agree.

    Only evaluated once a persona catalog is actually configured. A config
    with no ``personas:`` block at all resolves every user through
    :func:`~osprey.deployment.web_terminals.personas.resolve_personas`'s
    zero-migration path — this check never demands a ``registry.url`` from a
    deployment that has not opted into the persona system.
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
    """The lint-side mirror of
    :func:`~osprey.deployment.web_terminals.personas.resolve_personas`'s
    ``strict=True`` ``ValueError`` guard. ``osprey up`` never runs the lint
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


def _read_bundle_path(config_yml_path: Path) -> str | None:
    """Best-effort read of a config's ``facility_knowledge.bundle_path``.

    Degrades to ``None`` on any read/parse failure, exactly like
    :func:`_read_project_name`, whose caller already reports an unreadable
    ``config.yml`` as its own error.
    """
    try:
        with config_yml_path.open("r", encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    raw = as_dict(as_dict(parsed).get("facility_knowledge")).get("bundle_path")
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _check_persona_bundle_path_agreement(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Every entitled persona must put its knowledge bundle where the deploy does.

    The split this catches: **entitlement** is decided by each persona's own
    rendered ``config.yml`` (does it name a ``facility_knowledge.bundle_path``?),
    while the in-container **mount target** is derived from the DEPLOY config's
    ``bundle_path`` anchored on that persona's project directory. Where the two
    values disagree, a persona is entitled and gets the deployment's bundle bound
    at the deployment's path — while its OKF panel and ``facility_knowledge`` MCP
    server both read the persona's path, which nothing mounted.

    The result is an empty bundle with no error anywhere: the panel lists no
    concepts, the agent's knowledge tools return nothing, and every layer reports
    success. Nothing at runtime can tell that apart from a facility that simply
    has not written any concepts yet, which is why it is caught here.

    An ERROR rather than a warning: unlike a persona that is merely unrendered,
    nothing an operator does later resolves this. The two keys have to be made to
    agree.

    Read from disk on the same terms as :func:`_check_persona_project_paths` —
    ``project_path`` resolved against the working directory, an unreadable or
    unrendered project contributing nothing, since a persona that has not been
    built yet is already reported by that check.
    """
    deploy_bundle = as_dict(root.get("facility_knowledge")).get("bundle_path")
    if not isinstance(deploy_bundle, str) or not deploy_bundle.strip():
        return []  # nothing is mounted at all — no target to disagree with
    deploy_bundle = deploy_bundle.strip()

    catalog = _persona_catalog(web_terminals)
    findings: list[Finding] = []
    for persona_name in sorted(_referenced_persona_names(web_terminals, users)):
        entry = catalog.get(persona_name)
        if not isinstance(entry, dict):
            continue
        project_path_raw = entry.get("project_path")
        if not isinstance(project_path_raw, str) or not project_path_raw:
            continue
        config_yml = Path(project_path_raw) / "config.yml"
        if not config_yml.is_file():
            continue
        persona_bundle = _read_bundle_path(config_yml)
        if persona_bundle is None or persona_bundle == deploy_bundle:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.persona_bundle_path_divergence",
                message=(
                    f"persona {persona_name!r} sets facility_knowledge.bundle_path to "
                    f"{persona_bundle!r}, but this deployment sets {deploy_bundle!r}. "
                    "The bundle is bind-mounted at the DEPLOYMENT's path while the "
                    "persona's knowledge tools read its own, so that container would "
                    "see an empty bundle and report no error. Make the two agree"
                ),
            )
        )
    return findings


def _check_persona_project_paths(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """Local mode: validate every referenced persona's ``project_path``.

    ``project_path`` names the directory ``osprey up`` builds a persona's
    image from. Two invariants are enforced here:

    * **Name invariant.** When ``project_path`` is set, its basename must equal
      the catalog entry's ``project``. ``osprey build`` names each persona render
      ``<repo>-<delta stem>`` and writes it into ``build/``, so a basename that
      disagrees with ``project`` would leave the render in one directory while
      the catalog builds/mounts
      another — a dead path at runtime. A mismatch is an ERROR regardless of
      whether the directory exists yet.
    * **Existence.** The directory must exist and hold a ``Dockerfile`` and a
      ``config.yml`` whose own ``project_name`` equals the catalog ``project``
      (a mismatch silently produces a dead mount, since the per-svc
      ``container_project_dir`` derivation is keyed on the catalog's ``project``,
      not on anything read from the persona's own ``config.yml``).

    Existence is relaxed to a WARNING — never waived — for a ``project_path``
    that does not exist yet but whose entry carries a usable ``build_profile``.
    That is the ordinary state of a persona added since the last build, and the
    ordinary command clears it: ``osprey build`` renders one project per delta.
    It is not an error because nothing is misconfigured; it is not merely
    informational because ``osprey up`` refuses to start until the render is
    there. A *partially* rendered directory that exists but is missing its
    ``Dockerfile``/``config.yml`` stays an ERROR — a half-written render is a
    broken build rather than an absent one.
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

    # Name invariant: the build writes each persona render at a path whose
    # basename is the catalog `project`, so project_path's basename must equal
    # it. A
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
                    f"does not match its project {catalog_project!r}; the render "
                    "`osprey build` writes is found by project, so the two must agree"
                ),
            )
        ]

    # Shape of build_profile, enforced through the SAME predicate the deploy-time
    # resolver uses, so this gate cannot bless a value `osprey up` will
    # reject — the failure mode that matters here, since `osprey up` never runs
    # lint and an operator who lints clean would otherwise meet a hard deploy
    # error the gate promised away.
    #
    # Checked regardless of whether project_path exists: a rendered directory
    # makes an unusable value harmless only until someone removes it, and a
    # verdict that depended on local filesystem state would not be a gate. Like
    # the name mismatch above it supersedes the existence findings — an entry no
    # build could ever render has nothing to add about being missing.
    if has_build_profile:
        problem = persona_build_profile_shape_problem(cast(str, build_profile))
        if problem is not None:
            return [
                Finding(
                    severity="error",
                    code="web_terminals.persona_build_profile_not_a_delta",
                    message=(
                        f"modules.web_terminals.personas[{persona_name!r}].build_profile "
                        f"{problem} Set it to {f'personas/{persona_name}.yml'!r} — the "
                        "delta `osprey init` writes in this repo's personas/ directory, "
                        "which is what `osprey build` renders the persona project from. "
                        "A variant build that predates the delta layout has no such file "
                        "to point at yet; run /osprey-build-interview to convert it into one"
                    ),
                )
            ]

    if not project_path.is_dir():
        # Missing directory: a warning rather than the hard error when a
        # build_profile names the delta a build would render it from, because
        # that is the ordinary state of a persona added since the last build and
        # the ordinary command clears it. It is not merely informational: no
        # start will run until it is cleared.
        if has_build_profile:
            return [
                Finding(
                    severity="warn",
                    code="web_terminals.persona_project_path_not_rendered_yet",
                    message=(
                        f"modules.web_terminals.personas[{persona_name!r}].project_path "
                        f"{project_path_raw!r} does not exist. `osprey build` renders it "
                        f"from the delta its build_profile names ({build_profile!r}); "
                        "`osprey up` REFUSES to start until it is there. Run `osprey build`"
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
    stays the un-suffixed ``web-terminal:latest``, built by the core CI job,
    not a per-persona one."""
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
    ``ValueError`` — ``shared_http`` and any other unrecognized
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


def _check_nginx_image(web_terminals: dict[str, Any]) -> list[Finding]:
    """``nginx_image``, when set, overrides the nginx service's image reference —
    the one image in the web stack not built from the facility's own project, so
    a facility whose hosts pull only from a private registry mirror points it
    there. Absent is fine (the render-time default applies); a non-string is an
    ERROR, and an empty/whitespace-only string is a WARN (it would fall back to
    the default via the template's ``| default(...)``, so it is inert but almost
    certainly a mistake)."""
    if "nginx_image" not in web_terminals:
        return []
    value = web_terminals.get("nginx_image")
    if not isinstance(value, str):
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_nginx_image",
                message=(
                    f"modules.web_terminals.nginx_image {value!r} is not a string; "
                    "it must be an image reference, e.g. "
                    "'registry.example.com:5050/mirrors/nginx:1.27-alpine'"
                ),
            )
        ]
    if not value.strip():
        return [
            Finding(
                severity="warn",
                code="web_terminals.empty_nginx_image",
                message=(
                    "modules.web_terminals.nginx_image is set but empty; the default "
                    "nginx image applies. Remove the key or set a real image reference"
                ),
            )
        ]
    return []


def _check_external_origin(web_terminals: dict[str, Any]) -> list[Finding]:
    """``external_origin``, when set, replaces the derived browser origin —
    the value every terminal checks a mutating request's ``Origin`` against, the
    landing link, and the OIDC ``redirect_uri``. It exists for the topology where
    a load balancer terminating TLS in front of this nginx is what browsers
    actually reach, so nothing in this config can name that address.

    Absent is fine (``deploy.fqdn`` and the published port are used instead). A
    non-string, or a string that is not ``scheme://host[:port]`` and nothing
    else, is an ERROR: render refuses it too, and this is the same rejection at
    scaffold time. An empty string is a WARN — it falls back to the derivation,
    so it is inert but almost certainly a mistake.

    The failure this catches is quiet, which is why it is worth catching early:
    a value with a trailing slash or a stray path segment renders a deployment
    whose landing page and terminals all load, and whose every write, approval
    and chat message answers 403 with an origin message inside a container."""
    if "external_origin" not in web_terminals:
        return []
    value = web_terminals.get("external_origin")
    if isinstance(value, str) and not value.strip():
        return [
            Finding(
                severity="warn",
                code="web_terminals.empty_external_origin",
                message=(
                    "modules.web_terminals.external_origin is set but empty; the origin "
                    "derived from deploy.fqdn applies. Remove the key or set the address "
                    "browsers actually reach this deployment on"
                ),
            )
        ]
    try:
        _configured_external_origin({"modules": {"web_terminals": web_terminals}})
    except ValueError as exc:
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_external_origin",
                message=str(exc),
            )
        ]
    return []


# --- auth seam checks --------------------------------------------------------
#
# These are scaffold-time feedback only. The authoritative deploy-path gates
# live elsewhere and fail closed on their own: render.py raises on an unknown
# `auth.method` and on auth-without-TLS, and `auth_credentials.py` raises on a
# roster it cannot key credentials for. `osprey up` never runs this module,
# so nothing here may be the only thing standing between a bad config and a
# deployment — every check below mirrors a gate that also exists downstream,
# except where the downstream path *cannot* see the mistake (see
# :func:`_check_auth_method`).


def _auth_context(web_terminals: dict[str, Any]) -> dict[str, Any] | None:
    """render.py's parsed view of the ``auth``/``tls`` stanzas, or ``None``.

    Every check below reads the derived values the nginx template and the
    compose overlay consume rather than re-reading ``auth.*`` itself, so lint
    and render can't disagree about what a stanza means (see
    :func:`~osprey.deployment.web_terminals.render._auth_tls_context`, which is
    the single definition).

    That function raises on exactly one input — an ``auth.method`` string
    naming a method that does not exist — which :func:`_check_auth_method`
    reports on its own. Every check keyed on a parsed method is meaningless for
    such a config, so this degrades to ``None`` and they skip themselves rather
    than reporting confused follow-on findings.
    """
    try:
        return _auth_tls_context(web_terminals)
    except ValueError:
        return None


def _check_auth_method(web_terminals: dict[str, Any]) -> list[Finding]:
    """``modules.web_terminals.auth.method`` must name a supported method.

    Two distinct mistakes, both ERRORs:

    * An **unknown method string** (``"basic"``). render raises on this too —
      this check is the scaffold-time mirror, with the same message.
    * A **wrong-typed** ``auth`` stanza or ``method`` value (a mapping, an int,
      a bare ``auth: password`` string where a mapping belongs). render reads
      every value defensively, so a wrong-typed one falls back to its default
      and the deployment renders with authentication silently *off* — nothing
      downstream can catch it. This module is the only surface that sees it.

    An absent key, and an ``auth:``/``method:`` written with no value at all
    (both ``None`` after YAML load), are the documented defaults and are not
    flagged.
    """
    auth_raw = web_terminals.get("auth")
    if auth_raw is not None and not isinstance(auth_raw, dict):
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_auth_stanza",
                message=(
                    f"modules.web_terminals.auth {auth_raw!r} is not a mapping; it must "
                    "be a block with a 'method' key (e.g. 'auth:\\n  method: password'). "
                    "A non-mapping stanza is read as no auth stanza at all, which would "
                    "render the deployment with authentication silently disabled"
                ),
            )
        ]

    auth = as_dict(auth_raw)
    if "method" not in auth:
        return []
    method = auth.get("method")
    if method is None:
        return []
    if not isinstance(method, str):
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_auth_method_type",
                message=(
                    f"modules.web_terminals.auth.method {method!r} is not a string; "
                    f"expected one of {', '.join(SUPPORTED_AUTH_METHODS)}. A non-string "
                    "value falls back to 'none' at render time, which would render the "
                    "deployment with authentication silently disabled"
                ),
            )
        ]
    if method in SUPPORTED_AUTH_METHODS:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_auth_method",
            message=(
                f"modules.web_terminals.auth.method {method!r} is not a supported "
                f"authentication method; expected one of {', '.join(SUPPORTED_AUTH_METHODS)}"
            ),
        )
    ]


def _check_auth_transport(root: dict[str, Any], web_terminals: dict[str, Any]) -> list[Finding]:
    """Authentication over cleartext HTTP: an ERROR, or a WARN once accepted.

    A session cookie is a bearer credential. Served over plain HTTP it is
    readable — and replayable — by anything on the path, so ``auth.method`` set
    without ``tls.enabled`` is refused at render time unless the deployment
    explicitly accepts that risk with ``auth.allow_insecure_http: true``. This
    is the scaffold-time mirror of that gate: the ERROR for the refusal, and a
    WARN when the escape hatch is what's keeping the config renderable, so the
    risk is restated at every lint rather than only in the commit that took it.

    With TLS on, ``allow_insecure_http`` is inert and nothing is reported. The
    WARN is also withheld when ``deploy.fqdn`` names loopback: the deployment
    advertises itself as same-host-only, so its cookies cross no network path —
    the exact case the escape hatch exists for, and the posture the
    control-assistant preset ships in. Pointing ``fqdn`` at a real host brings
    the WARN back with the config change that creates the exposure.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] == "none":
        return []
    if context["tls_enabled"]:
        return []
    if context["auth_allow_insecure_http"]:
        fqdn = str(as_dict(root.get("deploy")).get("fqdn") or "").strip()
        if fqdn in ("127.0.0.1", "localhost", "::1"):
            return []
        return [
            Finding(
                severity="warn",
                code="web_terminals.auth_insecure_http",
                message=(
                    f"modules.web_terminals.auth.method is {context['auth_method']!r} "
                    "with tls.enabled false and allow_insecure_http true; session "
                    "cookies will travel over cleartext HTTP, where anything on the "
                    "network path can read and replay them. Enable tls for any "
                    "deployment reachable beyond a trusted host"
                ),
            )
        ]
    return [
        Finding(
            severity="error",
            code="web_terminals.auth_requires_tls",
            message=(
                f"modules.web_terminals.auth.method is {context['auth_method']!r} but "
                "tls.enabled is false; session cookies would travel over cleartext "
                "HTTP. Enable modules.web_terminals.tls, or set "
                "auth.allow_insecure_http: true to accept that risk (only sensible on "
                "a trusted network)"
            ),
        )
    ]


def _check_auth_oidc(root: dict[str, Any], web_terminals: dict[str, Any]) -> list[Finding]:
    """``method: oidc`` needs an issuer, usable client env-var names, safe
    subjects, and an origin.

    Four ERRORs, all config-visible and all fatal at *request* time rather
    than deploy time if they slip through — a sidecar that cannot complete a
    login flow locks the whole roster out (or, for a ``$``-bearing subject,
    one named user out):

    * **Issuer.** ``auth.oidc.issuer`` has no default: without it there is no
      discovery document to fetch and no IdP to redirect to.
    * **Client env-var names.** ``client_id_env``/``client_secret_env`` name the
      variables the sidecar reads its client credentials from (never the
      credentials themselves), and both default to a documented
      ``OSPREY_AUTH_OIDC_*`` name. Omitting them is therefore fine; setting one
      to something unusable (empty, wrong type) is not — render silently
      restores the default, so the sidecar would read a variable the operator
      never set.
    * **External origin.** The OIDC ``redirect_uri`` is built from the
      deployment's one external origin, which needs ``deploy.fqdn``. An IdP
      rejects a callback whose ``redirect_uri`` isn't character-for-character
      the registered one, so an underivable origin means no login can complete.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "oidc":
        return []

    findings: list[Finding] = []
    oidc = as_dict(as_dict(web_terminals.get("auth")).get("oidc"))

    if not context["auth_oidc_issuer"]:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.auth_oidc_missing_issuer",
                message=(
                    "modules.web_terminals.auth.method is 'oidc' but "
                    "auth.oidc.issuer is not set; the sidecar needs the issuer URL to "
                    "discover the IdP's endpoints"
                ),
            )
        )

    for field in ("client_id_env", "client_secret_env"):
        if field not in oidc:
            continue  # unset is fine — the documented OSPREY_AUTH_OIDC_* default applies
        value = oidc.get(field)
        if isinstance(value, str) and value.strip():
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.auth_oidc_invalid_client_env",
                message=(
                    f"modules.web_terminals.auth.oidc.{field} {value!r} is not a "
                    "non-empty string; it must name the environment variable holding "
                    "that OIDC client credential (the name, never the credential). An "
                    "unusable value silently restores the default variable name, which "
                    "the deployment has not set"
                ),
            )
        )

    # A roster entry's `oidc_subject` is the one OIDC value that travels through
    # the compose *document* (an `environment:` entry on the sidecar), not an
    # env_file — so the deploy-time `$` scan over `.env`/`.env.auth`/
    # `.env.users` never sees it, and compose-document interpolation
    # rewrites any `$` sequence on the way through. The sidecar would then match
    # logins against an identity the IdP never issues: that one user can never
    # log in, silently. Subjects are near-universally UUIDs or emails, so a `$`
    # is far more likely a typo than a real identity — refusing at lint is the
    # honest failure. The message names the user, never the subject: not
    # because the subject is secret (it is published by the IdP), but because
    # echoing a value compose would mangle invites pasting the mangled form.
    for entry in web_terminals.get("users") or []:
        if not isinstance(entry, dict):
            continue
        subject = entry.get("oidc_subject")
        if isinstance(subject, str) and "$" in subject:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.auth_oidc_subject_unsafe",
                    message=(
                        f"modules.web_terminals.users entry {entry.get('name')!r} has an "
                        "oidc_subject containing '$'. The subject is rendered into the "
                        "compose document, where '$' sequences are interpolated — the "
                        "sidecar would match against a rewritten identity and this user "
                        "could never log in. If the IdP truly issues a '$'-bearing "
                        "subject, map a different claim via auth.oidc.claim instead"
                    ),
                )
            )

    # Only `deploy.fqdn` can make the origin underivable; the published port
    # merely fills the ':port' suffix when TLS is off. A malformed `nginx_port`
    # is render's own error, reported there — substituting one here keeps this
    # check to the one thing it is about.
    nginx_port = web_terminals.get("nginx_port")
    try:
        _external_origin(
            root,
            nginx_port if isinstance(nginx_port, int) else 0,
            tls_enabled=bool(context["tls_enabled"]),
        )
    except ValueError as exc:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.auth_oidc_unresolvable_origin",
                message=(
                    "modules.web_terminals.auth.method is 'oidc' but this deployment's "
                    f"external origin cannot be derived ({exc}); the OIDC redirect_uri "
                    "is built from it and must match the URI registered with the IdP"
                ),
            )
        )
    return findings


def _check_auth_credential_collisions(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Password mode: two roster users may not key the same credential variable.

    A username is normalized (uppercased, ``-`` to ``_``) into the env-var
    suffix its stored password hash lives under, so ``alice-b`` and ``alice_b``
    would share one ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry — one operator's
    password opening the other's terminal, the exact isolation failure this
    feature exists to establish. ``auth_credentials`` refuses to provision such
    a roster with a hard raise on the deploy path; this is the same rejection
    at scaffold time, where renaming a user is still cheap.

    Scoped to ``method: password``: the per-user credential variable is what
    collides, and OIDC deployments have none (identity comes from the IdP).
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "password":
        return []
    names = [name for name in (_user_name(user) for user in users) if name is not None]
    return [
        Finding(
            severity="error",
            code="web_terminals.auth_credential_collision",
            message=(
                f"modules.web_terminals.users entries {colliding} all map onto the "
                f"credential variable {_PW_HASH_VAR_PREFIX}{suffix}; they would share a "
                "single password, so one user's credentials would open another's "
                "terminal. Rename one of them"
            ),
        )
        for suffix, colliding in env_var_suffix_collisions(names).items()
    ]


def _check_notice_docs(root: dict[str, Any], web_terminals: dict[str, Any]) -> list[Finding]:
    """Every path in ``landing.notices`` must name a file that exists.

    A missing notice is a ``warn``, not an ``error``: the page still renders and
    every other section still appears. But it is reported rather than swallowed,
    because :func:`osprey.deployment.web_terminals.render._build_notices`
    deliberately does NOT fall back to the packaged default here — a facility
    that mistyped ``local-procedures.md`` would otherwise get OSPREY's safety
    text in its place and no indication that their own document never loaded.
    """
    from osprey.deployment.compose_generator import resolve_repo_root

    landing = as_dict(web_terminals.get("landing"))
    raw = landing.get("notices")
    if not isinstance(raw, list):
        return []

    repo_root = resolve_repo_root(root)
    findings: list[Finding] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            findings.append(
                Finding(
                    severity="warn",
                    code="notice-invalid",
                    message=(
                        f"modules.web_terminals.landing.notices contains {entry!r}, "
                        "which is not a path. Each entry is a path to a markdown "
                        "file, relative to the project directory."
                    ),
                )
            )
            continue
        if not (repo_root / entry).is_file():
            findings.append(
                Finding(
                    severity="warn",
                    code="notice-missing",
                    message=(
                        f"modules.web_terminals.landing.notices lists {entry!r}, "
                        "which does not exist. That section will be missing from "
                        "the landing page."
                    ),
                )
            )
    return findings
