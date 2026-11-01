"""Static validation for the ``modules.web_terminals`` multi-user stanza.

The multi-user web stack is derived entirely from ``users[]`` and the render
catalog beside it, so "consistent" here means: every user resolves to exactly one
container with its own port family, every render reference resolves to a catalog
entry, and nothing in the stack contends with another service for a host port.

Two surfaces call in, at different altitudes:

* :func:`lint_web_terminals` reads a **rendered project ``config.yml``** — the
  deploy-time view, where the ``services:`` block names every published host port
  and every render project either exists on disk or is one ``osprey build``
  away. ``osprey scaffold web-terminals lint|render`` runs this one.
* :func:`lint_profile_config` reads a **build profile's ``config:`` block** — the
  authoring-time view, before anything is materialized. It runs the same checks
  minus the ones that can only be answered against a rendered project (see that
  function's docstring), so a profile can be validated without building it.
"""

from __future__ import annotations

import contextlib
import copy
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import yaml

from osprey.agent_record.config import HISTORY_SERVICE, history_origin_port, history_page_deployed
from osprey.agent_record.identities import (
    BRIDGE_IDENTITIES,
    OPERATOR_IDENTITY,
    host_identity,
    reader_entries,
)
from osprey.cli.build_profile_document import RETIRED_WRITES_LEAF
from osprey.config_guards import is_positive_int
from osprey.deployment.compose_generator import (
    DISPATCH_WORKER_SERVICE_PREFIX,
    deployed_app_audit_identities,
    resolve_project_name,
)
from osprey.deployment.web_terminals.ports import (
    FAMILY_BASE_FIELDS,
    allocate_ports,
    base_ports_from_config,
    resolve_nginx_port,
)
from osprey.deployment.web_terminals.render import (
    _AUTH_PORT_SLOT,
    _RESERVED_AUDIT_IDENTITY_RE,
    AUTH_SIDECAR_AUDIT_IDENTITY,
    HISTORY_ORIGIN_REQUIRED,
    SUPPORTED_AUTH_METHODS,
    TLS_LISTEN_PORT,
    _auth_throttle_problems,
    _auth_tls_context,
    _authorization_context,
    _blank_scope_index,
    _configured_external_origin,
    _external_origin,
    _port_int,
    _scope_list,
    _tls_enabled,
    deployment_origin,
    invalid_history_origin,
)
from osprey.deployment.web_terminals.render_images import (
    PREDATES_DELTA_REMEDY,
    _canonical_delta_reference,
    render_build_profile_shape_problem,
)
from osprey.deployment.web_terminals.render_naming import RENDER_KIND_TIER
from osprey.deployment.web_terminals.roster import (
    ALL_PRIVILEGES,
    REGISTRY_MODE_MISSING_URL,
    RENDER_CATALOG_REQUIRED,
    SUPPORTED_MCP_TOPOLOGY,
    USERNAME_CHARSET_RE,
    _is_shared_entry,
    _roster_entry_name,
    as_dict,
    auth_is_enforced,
    authored_people,
    configured_registry_url,
    control_identity_collision_warnings,
    control_identity_problems,
    deployment_wide_privileged_exposure_problems,
    dispatch_tier_name,
    effective_image_source,
    effective_tier,
    entry_is_shared,
    is_app_entry,
    privilege_phrase,
    privileged_default_tier_problem,
    privileges_beyond_baseline,
    rendered_config_file,
    rendered_configs,
    resolve_access_principals,
    resolve_cards,
    resolve_image_tag,
    roster_entries,
    roster_people,
    served_render_names,
    shared_card_privileged_problems,
    tier_privileges,
)
from osprey.interfaces.web_auth import DEFAULT_SESSION_LIFETIME
from osprey.port_layout import _MAX_PORT, INDEX_MAX, default_port, resolve_port_base
from osprey.services.auth_sidecar.identity_headers import CASE_INSENSITIVE_CLAIMS
from osprey.services.auth_sidecar.passwords import stored_hash_problem
from osprey.services.auth_sidecar.roster_env import (
    PW_HASH_VAR_PREFIX,
    env_var_suffix,
    env_var_suffix_collisions,
)
from osprey.utils.workspace import LAYER_DIR_KINDS
from osprey_connectors.dotenv import parse_dotenv_file
from osprey_connectors.types import (
    TARGET_LIVE,
    WRITES_KEY,
    baseline_target,
    configured_targets,
    resolve_target,
    speaks_channel_access,
    tier_allows,
)

# Both listeners in the gated auth/TLS seam are config-driven: nginx's TLS
# listener is `tls.port` and the auth sidecar's is `auth.port`. Neither default
# is restated here. `render.TLS_LISTEN_PORT` is imported because the render
# stamps that same fallback into both `listen` lines and the perimeter
# deny-list, and a second literal here could come to reserve a port the template
# does not listen on; the sidecar's default is the port layout's `auth` slot,
# read off render's parsed auth context or derived from the same layout. So is
# `render._port_int`, the one reader that says which values reach a `listen`
# directive at all — the checks below resolve a configured port exactly as
# render resolves it, which is what makes a finding that says "render falls
# back" true across the whole invalid domain. `render._scope_list` is imported
# for the same reason: the shapes it reads as unset are exactly the ones whose
# scopes never reach the sidecar.

# The deployment-repo file the per-user password hashes live in, quoted here
# rather than imported from `auth_credentials`, which owns it: this module is
# pure static validation, and importing the credential provisioner to quote one
# string would pull the whole deploy-time secret-minting path in behind it.
_AUTH_ENV_FILENAME = ".env.auth"


@dataclass(frozen=True)
class Finding:
    """A single lint result.

    ``severity`` is ``"error"`` (a config that must be rejected) or ``"warn"``
    (worth flagging, does not fail the check — e.g. a render whose
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


def lint_web_terminals(
    config: Any,
    *,
    rendered_project: bool = True,
    project_root: Path | None = None,
    profile_root: Path | None = None,
) -> list[Finding]:
    """Validate a rendered project config's ``modules.web_terminals`` stanza.

    Args:
        config: The parsed project config, read defensively as nested dicts (no
            assumption that ``config`` is a particular schema/dataclass type).
        rendered_project: Whether *config* describes a project that has actually
            been rendered. False drops the two checks that can only be answered
            once it has — see :func:`lint_profile_config`, the only caller that
            passes it.
        project_root: The directory a rendered ``project_path`` resolves
            against — the deployment's build zone. Only the caller knows it:
            this module is handed a config, never a location. ``None`` falls
            back to the working directory, which is the answer for a command
            run from the repo root and the wrong one everywhere else, so every
            command surface passes it explicitly.
        profile_root: The same fact at profile altitude — the directory holding
            the ``profile.yml`` being linted, which a catalog entry's
            ``build_profile: tiers/<name>.yml`` resolves against. ``None``
            likewise falls back to the working directory.

    Returns:
        A list of :class:`Finding` objects, empty if nothing is wrong. Findings
        with ``severity="warn"`` do not indicate a config that must be rejected.
    """
    root = as_dict(config)
    modules = as_dict(root.get("modules"))
    web_terminals = as_dict(modules.get("web_terminals"))

    if not web_terminals.get("enabled"):
        return []

    if not rendered_project:
        root, web_terminals = _with_profile_renders(root, web_terminals, profile_root)

    # The people as authored: every check below reads the keys a person's entry
    # carries, and `_check_bare_list_port_drift_risk` in particular is about the
    # people's positional indices alone.
    users = authored_people(web_terminals)

    findings: list[Finding] = []
    findings.extend(_check_empty_users(users))
    findings.extend(_check_duplicate_users(users))
    findings.extend(_check_username_charset(users))
    findings.extend(_check_reserved_audit_identities(users, deployed_app_audit_identities(root)))
    findings.extend(_check_display_name(users))
    findings.extend(_check_user_theme(users))
    findings.extend(_check_user_tour(users))
    findings.extend(_check_user_access(users))
    findings.extend(_check_control_identities(web_terminals, users))
    findings.extend(_check_invalid_index(users))
    findings.extend(_check_duplicate_index(users, _app_slots(web_terminals)))
    findings.extend(_check_bare_list_port_drift_risk(users))
    findings.extend(_check_port_families_allocatable(root, web_terminals, users))
    findings.extend(_check_port_overlap(root, web_terminals, users))
    findings.extend(
        _check_shared_apps(
            web_terminals,
            users,
            rendered_project=rendered_project,
            profile_root=profile_root,
        )
    )
    findings.extend(_check_render_charset(web_terminals))
    findings.extend(_check_render_seed_base(web_terminals))
    findings.extend(_check_default_tier_exists(web_terminals))
    findings.extend(_check_unknown_tier_reference(root, web_terminals, users))
    # Reads each referenced render's config — a rendered `config.yml` here, a
    # `tiers/<name>.yml` delta or a bundled preset at profile time — so it
    # takes the gate rather than riding one, and runs at both altitudes.
    findings.extend(
        _check_privileged_tier_exposure(
            root,
            web_terminals,
            users,
            rendered_project=rendered_project,
            project_root=project_root,
            profile_root=profile_root,
        )
    )
    # Reads the same tier layers, and asks the one question that needs them
    # unmerged: whether a read-only-looking render inherits an armed connector
    # block. Profile altitude only — see the check.
    findings.extend(
        _check_readonly_tier_inherits_writes(
            root,
            web_terminals,
            rendered_project=rendered_project,
            profile_root=profile_root,
        )
    )
    # The same render walk once more, asking whether a card writes to the real
    # machine without saying who is writing. Both altitudes.
    findings.extend(
        _check_live_writer_without_control_identity(
            root,
            web_terminals,
            users,
            rendered_project=rendered_project,
            project_root=project_root,
            profile_root=profile_root,
        )
    )
    findings.extend(_check_unknown_image_source(web_terminals))
    findings.extend(_check_image_tag_empty(web_terminals))
    findings.extend(_check_registry_url_coherence(root, web_terminals))
    findings.extend(_check_requires_catalog(web_terminals))
    findings.extend(_check_render_project_collisions(root, web_terminals))
    if rendered_project:
        findings.extend(_check_render_project_paths(root, web_terminals, project_root=project_root))
        # Reads each render's rendered config.yml, so it rides on the same
        # gate: a profile has no rendered project to compare against yet.
        findings.extend(
            _check_render_bundle_path_agreement(root, web_terminals, project_root=project_root)
        )
        # Same split, for the ARIEL mirror: entitlement from the render's
        # config, the bind's source and target from the deploy's.
        findings.extend(
            _check_render_mirror_agreement(root, web_terminals, project_root=project_root)
        )
        # Resolves notice paths against the project directory, so it rides the
        # same gate: a profile has no rendered project to look in yet.
        findings.extend(_check_notice_docs(root, web_terminals))
        # Reads each render's rendered `.claude/settings.json`, so it rides the
        # same gate for the same reason — and rides it honestly: at profile
        # altitude there is no shipped artifact to read, and a rule that
        # answered anyway would be guessing at the one file the deploy gate
        # refuses on.
        findings.extend(_check_open_mode_egress(root, project_root=project_root))
        # Both read the deployment's `.env` or `.env.auth`, so they ride the
        # same gate: at profile altitude neither file exists yet.
        findings.extend(_check_seeded_passwords(root, web_terminals, project_root=project_root))
        findings.extend(_check_auth_stored_hashes(web_terminals, users, project_root=project_root))
    findings.extend(_check_registry_mode_build_profile(root, web_terminals))
    findings.extend(_check_render_missing_project(root, web_terminals))
    findings.extend(_check_render_extra_mounts(web_terminals))
    findings.extend(_check_unknown_mcp_topology(web_terminals))
    findings.extend(_check_nginx_image(web_terminals))
    findings.extend(_check_external_origin(web_terminals))
    findings.extend(_check_auth_method(web_terminals))
    findings.extend(_check_auth_session_lifetime(web_terminals))
    findings.extend(_check_auth_throttle(web_terminals))
    findings.extend(_check_listener_ports(root, web_terminals))
    findings.extend(_check_auth_transport(root, web_terminals))
    findings.extend(_check_auth_oidc(root, web_terminals))
    findings.extend(_check_landing_names(web_terminals))
    findings.extend(_check_landing_group_types(web_terminals))
    findings.extend(_check_auth_credential_collisions(web_terminals, users))
    findings.extend(_check_shared_card_duplicate_subject(web_terminals, users))
    findings.extend(_check_shared_card_subject(web_terminals, users))
    findings.extend(_check_access_principal_without_idp(web_terminals, users))
    findings.extend(_check_access_domain_without_email_claim(web_terminals, users))
    findings.extend(_check_duplicate_access_principals(web_terminals, users))
    findings.extend(_check_authorization(web_terminals))
    findings.extend(_check_claims_without_oidc(web_terminals))
    findings.extend(_check_role_charset(web_terminals))
    findings.extend(_check_authorization_unsafe_values(web_terminals))
    findings.extend(_check_user_role(web_terminals, users))
    findings.extend(_check_history_values(web_terminals, users))
    if history_page_deployed(root):
        findings.extend(_check_history_port(root))
        findings.extend(_check_history_origin(web_terminals))
        findings.extend(
            Finding(severity="error", code=HISTORY_CARRIER_COLLISION, message=f"{p}. {r}")
            for p, r in history_carrier_collision_problems(root)
        )
        findings.extend(_check_unbound_history_roles(web_terminals, users))
    return findings


def lint_profile_config(
    config: Mapping[str, Any], *, profile_root: Path | None = None
) -> list[Finding]:
    """Validate the ``modules.web_terminals`` a build profile's ``config:`` sets.

    A ``config:`` block is a flat bag of dotted keys (``modules.web_terminals``,
    ``deploy.fqdn``, ``services.postgresql.port_host``, …) applied over the
    rendered template, so it is nested into the shape the checks read before
    linting. Shallowest key first, so a deeper key refines the subtree a
    shallower one set rather than being overwritten by it — the same order
    ``osprey build`` applies them in.

    Two checks are skipped, because a profile cannot answer them yet:

    * **Render project paths.** ``project_path`` resolves against the rendered
      project's directory, which does not exist at authoring time.
    * **``build_profile`` shape**, which rides on the same check.
      ``osprey init`` rewrites each catalog entry's preset name into the
      ``tiers/<name>.yml`` delta it materializes beside the profile, so the
      value is only in its final form once that has happened.

    Both are enforced in full by :func:`lint_web_terminals` at deploy time.
    Cross-service port overlap is likewise checked only against the host ports
    the ``config:`` block itself declares; a service declared at profile top
    level joins the collision set once the project is rendered.

    ``profile_root`` is the directory the profile being linted lives in, and it
    is not optional in practice: a catalog entry's
    ``build_profile: tiers/<name>.yml`` is relative to the profile, so
    without it this falls back to the working directory and a validate run from
    a subdirectory, or through ``--repo`` from outside the repo entirely, reads
    no layer at all and reports every render as unprivileged. Pass it.

    Call this from a COMMAND, never from ``BuildProfile.validate()``. That
    method also runs during profile *resolution*, which ``osprey init``
    goes through, and these findings would then pre-empt that command's own
    render validator — which reports every unusable catalog entry at once and
    names the file-name rule a render name really has to meet. The engine
    would replace a better error with a worse one.
    """
    return lint_web_terminals(
        _nest_dotted(config), rendered_project=False, profile_root=profile_root
    )


def profile_config_errors(
    config: Mapping[str, Any], *, profile_root: Path | None = None
) -> list[str]:
    """The messages from :func:`lint_profile_config` that must fail a command.

    One home for "which severity blocks", so the surfaces that gate on it
    cannot drift apart: warnings and informational findings are advisory and
    never stop a build.
    """
    return [
        finding.message
        for finding in lint_profile_config(config, profile_root=profile_root)
        if finding.severity == "error"
    ]


def profile_config_warnings(
    config: Mapping[str, Any], *, profile_root: Path | None = None
) -> list[str]:
    """The advisory messages from :func:`lint_profile_config`.

    The other half of :func:`profile_config_errors`, and it exists for the same
    reason: a finding nobody prints is a finding nobody has. Warnings do not
    fail a command — an unauthenticated deployment is a legitimate loopback
    posture, and a build that refused one would reject deployments nobody
    exposed — but they name exposures an operator would want to know about, so
    every surface that gates on the errors also reports these above its success
    line.
    """
    return [
        finding.message
        for finding in lint_profile_config(config, profile_root=profile_root)
        if finding.severity == "warn"
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


def _check_username_charset(users: list[Any]) -> list[Finding]:
    """A user name becomes an nginx `location` key and a URL path segment
    (``/u/<user>/...``); it must match ``^[a-z0-9][a-z0-9_-]*$``."""
    findings: list[Finding] = []
    for user in users:
        name = _roster_entry_name(user)
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

    The renderer reads it defensively (:func:`resolve_cards` drops a non-string
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


def _check_user_tour(users: list[Any]) -> list[Finding]:
    """An object-form entry's optional ``tour`` (the per-user onboarding-tour
    invite policy emitted as ``OSPREY_WEB_TOUR``) must be a string when present.

    Same shape and rationale as :func:`_check_user_theme`: only the *type* is
    checked. The policy vocabulary (``once``/``always``/``never``) is owned by
    the web terminal, which warns and falls back to its default on an unknown
    value at startup — failing a build over a word this module does not own
    would be worse than that warning.
    """
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict) or "tour" not in user:
            continue
        tour = user.get("tour")
        if isinstance(tour, str):
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_user_tour",
                message=(
                    f"modules.web_terminals.users entry {name!r} has a non-string "
                    f"tour {tour!r}; tour must be a string "
                    f"('once', 'always' or 'never')"
                ),
            )
        )
    return findings


def _check_control_identities(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """Each card's optional ``control_identity`` must be a name the container accepts.

    ``control_identity`` is the account name the control system sees a card's
    writes arrive under. Every refusal the builders name
    (:func:`~osprey.deployment.web_terminals.roster.control_identity_problems`)
    is an ERROR — an unusable value, a value on a shared card, or one value
    carried for two different people — so lint accepts exactly what the
    container accepts. A collision the roster cannot prove wrong
    (:func:`~osprey.deployment.web_terminals.roster.control_identity_collision_warnings`)
    is a WARN.

    The rules read only the roster, so they run in every auth posture. The
    configured OIDC claim decides whether subjects compare case-folded; with no
    ``oidc`` login, or an ``auth.method`` naming no method (reported on its
    own), the sidecar default applies and subjects compare exactly.
    """
    context = _auth_context(web_terminals)
    claim = (context.get("auth_oidc_claim") if context is not None else None) or ""
    findings = [
        Finding(severity="error", code=code, message=message)
        for code, message in control_identity_problems(users, claim=claim)
    ]
    findings.extend(
        Finding(severity="warn", code=code, message=message)
        for code, message in control_identity_collision_warnings(users, claim=claim)
    )
    return findings


def _check_user_access(users: list[Any]) -> list[Finding]:
    """An object-form entry's optional ``access`` must resolve to a principal set.

    ``access`` says who may open an entry's card, authored either as one of the
    two shorthands (``own``, ``any``) or as a list of principals (``self``,
    ``roster``, ``user:<id>``, ``domain:<domain>``).
    :func:`~osprey.deployment.web_terminals.roster.resolve_access_principals`
    is the single parser of that vocabulary, so this rule resolves the entry and
    reports the resolver's refusal verbatim rather than re-stating the grammar.
    Lint and the render preflight therefore cannot disagree about which
    spellings a deployment accepts, and an operator reads the same sentence from
    both. The resolver's messages already name the entry, quote the offending
    value and say what to write instead — ``group:``, for one, is refused by
    name as reserved and not supported yet, which is a different instruction
    than "unknown prefix".

    Every refusal is an ERROR, because a value no consumer recognises deploys
    fail-closed as ``own``: ``access: ANY`` (or ``true``, or a number) never
    widens an entry's reach, but the author who wrote it believes the opposite
    of what deploys. ``access: own`` explicit, an absent key, and an authored
    key with no value (YAML null) are silent — all three are valid spellings of
    the default.

    One deliberate silence remains: ``access: any`` is not reported where it
    cannot change anything, i.e. where the deployment stands no wall at all
    (``sidecar_active`` is false — under ``token`` and ``none`` alike no login
    gates the roster). That silence is not an oversight: profile overlays
    concatenate the roster list rather than merging entries by name, so a
    passive base that a host variant arms with ``password``/``oidc`` has to
    carry its ``access`` key on the base entry — the same position
    ``oidc_subject`` already sits in silently (:func:`_check_auth_oidc` reads it
    only under ``oidc``). A finding there would recur on every validate/build of
    the base and train operators to skip the WARN block; the key arms the moment
    the variant is selected, and the merged view is what both commands lint.
    """
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict) or "access" not in user:
            continue
        try:
            resolve_access_principals(user)
        except ValueError as refusal:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_user_access",
                    message=str(refusal),
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


def _check_duplicate_index(
    users: list[Any], app_slots: Sequence[tuple[str, int]] = ()
) -> list[Finding]:
    """Two cards sharing an index would collide on every per-user port family.

    The cards are the object-form users and the derived shared-app slots. Entries
    with an already-invalid index are skipped here — that's reported by
    :func:`_check_invalid_index` instead."""
    by_index: dict[int, list[Any]] = {}
    apps_at: set[int] = set()
    for user in users:
        index = _valid_index(user)
        if index is None:
            continue
        by_index.setdefault(index, []).append(user.get("name", user))
    for name, index in app_slots:
        by_index.setdefault(index, []).append(name)
        apps_at.add(index)

    findings: list[Finding] = []
    for index in sorted(by_index):
        names = by_index[index]
        if len(names) > 1:
            where = (
                "modules.web_terminals users and shared apps"
                if index in apps_at
                else "modules.web_terminals.users entries"
            )
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.duplicate_index",
                    message=(
                        f"{where} {names} share index "
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


def _people_slots(users: list[Any]) -> list[tuple[Any, int]]:
    """Each person's name and the per-user port index it allocates on.

    Mirrors ``renders.normalize_users``: an object entry carries its own
    ``index``, a legacy bare string takes its position in the list. An object
    entry with an unusable index is skipped — that is
    :func:`_check_invalid_index`'s finding, and guessing a number for it here
    would report a second, wrong one.

    Args:
        users: The raw ``modules.web_terminals.users`` list.

    Returns:
        ``(name, index)`` pairs, in roster order.
    """
    slots: list[tuple[Any, int]] = []
    for position, user in enumerate(users):
        if isinstance(user, dict):
            index = _valid_index(user)
            if index is not None:
                slots.append((user.get("name"), index))
        else:
            slots.append((user, position))
    return slots


def _app_slots(web_terminals: dict[str, Any]) -> list[tuple[str, int]]:
    """Each derived shared-app card's name and port index, in ``shared:`` order.

    Read through :func:`~osprey.deployment.web_terminals.roster.roster_entries`,
    the one reader that derives them, so lint judges the slots the deployment
    will actually run.
    """
    return [
        (entry["name"], entry["index"])
        for entry in roster_entries(web_terminals, strict=False)
        if is_app_entry(entry)
    ]


def _check_port_families_allocatable(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Consistency rule: every user must resolve a full port-family set (the
    ``web`` family plus one family per registry companion server — see
    ``ports.FAMILY_BASE_FIELDS``) at an index its family band holds.

    Every family now has a layout default, so no port key can be *missing*.
    What can still fail is the roster outgrowing the block: a family band holds
    one hundred users, so a user index past
    :data:`osprey.port_layout.INDEX_MAX` would take a port belonging to the
    next family. The highest index in the roster — a person's or a derived
    shared-app slot's — is the one that decides it."""
    indices = [index for _, index in (*_people_slots(users), *_app_slots(web_terminals))]
    base_ports = base_ports_from_config(web_terminals, base=resolve_port_base(root))
    if not indices:
        return []
    try:
        allocate_ports(base_ports, index=max(indices))
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
    listener (``tls.port``, defaulting to :data:`TLS_LISTEN_PORT`) when that
    seam is enabled.
    """
    entries: list[tuple[int, str]] = []

    # Per-user families: one range per family, over the N configured users and
    # the derived shared-app slots at the top of the band. Derived from the base
    # this deployment resolved, so the overlap set is the ports it will actually
    # bind rather than the default block's.
    app_indices = [index for _, index in _app_slots(web_terminals)]
    base_ports = base_ports_from_config(web_terminals, base=resolve_port_base(root))
    for base_field, family in FAMILY_BASE_FIELDS.items():
        base = base_ports.get(family)
        if base is None:
            continue
        entries.extend(
            (base + index, f"web_terminals.{base_field}[index={index}]")
            for index in [*range(len(users)), *app_indices]
        )

    # nginx's own listener, the one port the stack is reached on from off-host.
    # Resolved rather than read off the key: an unset `nginx_port` is not an
    # absent listener, it is the gateway slot of this deployment's block, and a
    # port the stack will bind belongs in the collision set however it was
    # spelled. A value that is not a port binds nothing and joins nothing —
    # render refuses that config outright, and this rule is about collisions.
    with contextlib.suppress(ValueError):
        entries.append((resolve_nginx_port(root), "web_terminals.nginx_port"))

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
    # the seam is actually enabled by config; the default (tls disabled, no
    # sidecar) must not reserve the TLS listener or the sidecar port against
    # ordinary configs.
    #
    # `tls.enabled` is read through render's `_tls_enabled`, the same reader
    # `_auth_tls_context` uses, rather than through `_auth_context`: that context
    # is None for a config whose `auth.method` names no method, and a port this
    # deployment's nginx will bind belongs in the collision set whether or not
    # the auth stanza parses. `_port_int` resolves it the way render does, so an
    # unusable `tls.port` reserves the default nginx will actually listen on
    # rather than a port nothing binds.
    if _tls_enabled(web_terminals):
        tls = as_dict(web_terminals.get("tls"))
        entries.append((_port_int(tls.get("port"), TLS_LISTEN_PORT), "web_terminals.tls.port"))
    # The deployment's own base, the same one the port families above were
    # allocated at: a sidecar port resolved at the layout default would land in
    # a different block and make this collision set mixed-base — missing a real
    # collision and inventing a false one.
    auth_context = _auth_context(web_terminals, base=resolve_port_base(root))
    if auth_context is not None and auth_context["sidecar_active"]:
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


# --- shared login cards ------------------------------------------------------
#
# `modules.web_terminals.shared` names the logins every roster login may open.
# Each name derives one card (see `renders.roster_entries`) at port slot
# `INDEX_MAX - position`, counting down from the top of the family band while the
# people count up from 0. The checks below are the ones the derivation leaves to
# lint: what a name must be, what it must not collide with, and which login
# files exist to back it.


def _with_profile_renders(
    root: dict[str, Any], web_terminals: dict[str, Any], profile_root: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """*root* and *web_terminals* with the renders a profile repo holds on disk.

    A profile carries no ``renders`` projection — the build writes that — so at
    profile altitude the logins are the ``tiers/<name>.yml`` files beside the
    profile. They are projected here into
    the same ``renders`` shape the rendered config carries, so every check reads
    one catalog at both altitudes. A config that already states ``renders`` or a
    ``renders`` catalog is returned unchanged; the caller's dicts are never
    mutated.
    """
    if isinstance(web_terminals.get("renders"), Mapping):
        return root, web_terminals
    base = profile_root or Path(".")
    renders: dict[str, Any] = {}
    for dirname, kind in LAYER_DIR_KINDS.items():
        directory = base / dirname
        if directory.is_dir():
            for path in sorted(directory.glob("*.yml")):
                renders.setdefault(path.stem, {"kind": kind})
    if not renders:
        return root, web_terminals
    projected = {**web_terminals, "renders": renders}
    modules = {**as_dict(root.get("modules")), "web_terminals": projected}
    return {**root, "modules": modules}, projected


def _known_logins(
    web_terminals: dict[str, Any], *, rendered_project: bool, profile_root: Path | None
) -> frozenset[str] | None:
    """The logins a ``shared:`` name may name, or ``None`` when this altitude cannot tell.

    At profile altitude they are the login files beside the profile
    (``tiers/<name>.yml``). At rendered altitude they are the
    ``modules.web_terminals.renders`` entries the build projected; a rendered
    config without that projection names no logins either way, so nothing is
    judged against it.
    """
    if not rendered_project:
        base = profile_root or Path(".")
        return frozenset(
            path.stem
            for dirname in LAYER_DIR_KINDS
            if (base / dirname).is_dir()
            for path in (base / dirname).glob("*.yml")
        )
    renders = web_terminals.get("renders")
    if not isinstance(renders, Mapping):
        return None
    return frozenset(str(name) for name, entry in renders.items() if isinstance(entry, Mapping))


def _check_shared_apps(
    web_terminals: dict[str, Any],
    users: list[Any],
    *,
    rendered_project: bool,
    profile_root: Path | None,
) -> list[Finding]:
    """Every ``shared:`` name must derive one card that collides with nothing.

    Errors, each its own code:

    * ``invalid_shared_entry`` — ``shared`` is not a list, or an item is not a
      non-empty string (it derives no card but keeps its slot).
    * ``duplicate_shared_app`` — one login named twice would run two cards under
      one compose service name.
    * ``unknown_shared_app`` — no login file backs the name, so there is nothing
      to render for it.
    * ``user_named_like_app`` — a person whose name is a shared login's would
      share that card's service name, URL and secrets.
    * ``user_index_in_app_band`` — a person whose index reaches
      ``INDEX_MAX - len(shared) + 1`` sits in the slots the shared logins take.
    """
    if "shared" not in web_terminals:
        return []
    raw = web_terminals.get("shared")
    if not isinstance(raw, list):
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_shared_entry",
                message=(
                    "modules.web_terminals.shared must be a list of login names, got "
                    f"{type(raw).__name__}"
                ),
            )
        ]
    findings: list[Finding] = []
    names: list[str] = []
    for position, item in enumerate(raw):
        if isinstance(item, str) and item:
            names.append(item)
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_shared_entry",
                message=(
                    f"modules.web_terminals.shared[{position}] is {item!r}; each entry "
                    "must be the name of a login in this repo's tiers/ directory"
                ),
            )
        )

    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.duplicate_shared_app",
                message=(
                    f"modules.web_terminals.shared names {sorted(duplicates)} more than "
                    "once; each login is one shared card, so list it once"
                ),
            )
        )

    known = _known_logins(
        web_terminals, rendered_project=rendered_project, profile_root=profile_root
    )
    if known is not None:
        unknown = sorted({name for name in names if name not in known})
        if unknown:
            where = "modules.web_terminals.renders" if rendered_project else "this repo's tiers/"
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.unknown_shared_app",
                    message=(
                        f"modules.web_terminals.shared names {unknown}, which "
                        f"{'have' if len(unknown) > 1 else 'has'} no login in {where}. "
                        "Add the login with `osprey scaffold tiers`, or remove the name "
                        "from modules.web_terminals.shared"
                    ),
                )
            )

    app_names = set(names)
    clashing = sorted(
        {name for name, _ in _people_slots(users) if isinstance(name, str) and name in app_names}
    )
    if clashing:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.user_named_like_app",
                message=(
                    f"modules.web_terminals.users names {clashing}, which "
                    f"{'are shared logins' if len(clashing) > 1 else 'is a shared login'}; "
                    "a person and a shared card would share one service name, URL and "
                    "secret. Rename the person"
                ),
            )
        )

    if raw:
        floor = INDEX_MAX - len(raw) + 1
        crowded = [(name, index) for name, index in _people_slots(users) if index >= floor]
        for name, index in crowded:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.user_index_in_app_band",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has index {index}, "
                        f"but indices {floor}..{INDEX_MAX} are the {len(raw)} shared "
                        "login slots (modules.web_terminals.shared counts down from "
                        f"{INDEX_MAX}). Give the person an index below {floor}"
                    ),
                )
            )
    return findings


# --- render catalog identity/reference checks ------------------------------
#
# Note on duplicate catalog keys: `modules.web_terminals.renders` arrives here
# already parsed by `yaml.safe_load`, which silently collapses a YAML mapping's
# duplicate keys down to the last-declared value — by the time this module
# sees a Python dict, a duplicate render key has already vanished without a trace (no exception, no marker to detect). There is nothing observable
# post-load, so no duplicate-catalog-key check exists here.
#
# Mode-coherence checks (image_source/registry.url agreement, project_path /
# Dockerfile / config.yml existence, build_profile requirements) are below,
# layered on top of `_render_catalog` and the checks above.


def _render_catalog(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """Every render the projection lists, as catalog entries keyed by name.

    Read from the render projection ``modules.web_terminals.renders``. Each
    entry keeps its ``kind``, ``project``, ``project_path`` and file knobs and
    states the file the build renders it from as ``build_profile``
    (``tiers/<name>.yml``) unless the entry states one of its own, so the checks below judge the layer file exactly as
    they judge any rendered-from-a-file entry. A rule whose question is about
    the tiers a person can be given reads :func:`_tier_catalog` instead."""
    renders = web_terminals.get("renders")
    if not isinstance(renders, dict):
        return {}
    catalog: dict[str, Any] = {}
    for name, render in renders.items():
        if not isinstance(render, dict) or render.get("kind") != RENDER_KIND_TIER:
            continue
        entry = dict(render)
        entry.setdefault("build_profile", _canonical_delta_reference(name))
        catalog[name] = entry
    return catalog


def _tier_catalog(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """The tiers a person can be given: :func:`_render_catalog` minus the shared logins.

    A shared login is sealed read-only for everyone on the roster, so its name
    never stands in for a person's tier — not as ``default_tier``, and not as
    the catalog a roster needs."""
    shared = web_terminals.get("shared")
    shared_names = {n for n in shared if isinstance(n, str)} if isinstance(shared, list) else set()
    return {
        name: entry
        for name, entry in _render_catalog(web_terminals).items()
        if name not in shared_names
    }


def _served_names(root: dict[str, Any], web_terminals: dict[str, Any]) -> dict[str, str]:
    """Every render this deployment serves, by name and kind, read leniently.

    :func:`~osprey.deployment.web_terminals.roster.served_render_names` over
    this stanza and the dispatch tier the config carries
    (``services.dispatch_worker.tier``, at both altitudes): every card's
    render, ``default_tier``, every role's tier and the dispatch tier. A render
    the projection lists that nobody serves sits outside every per-render check
    — an unused draft entry with a broken ``project_path`` never blocks a
    deploy, since only served renders are ever built or pulled.

    Lenient: this is a reporting surface, and an entry whose binding does not
    resolve already has its own ERROR (``web_terminals.unknown_role_reference``
    / ``web_terminals.conflicting_user_tier_and_role``)."""
    return served_render_names(web_terminals, dispatch_tier_name(root), strict=False)


def _default_tier(web_terminals: dict[str, Any]) -> Any:
    """``modules.web_terminals.default_tier`` — the tier of every person who names none."""
    return web_terminals.get("default_tier")


def _check_render_charset(web_terminals: dict[str, Any]) -> list[Finding]:
    """A render catalog key becomes an image-tag suffix
    (``web-terminal-<render>:latest``) and a path component (``/app/<project>``,
    local-mode image tags); it's held to the same charset as usernames,
    ``^[a-z0-9][a-z0-9_-]*$`` (see :func:`_check_username_charset`)."""
    return [
        Finding(
            severity="error",
            code="web_terminals.invalid_render_charset",
            message=(
                f"modules.web_terminals.renders key {render_name!r} does not "
                f"match {USERNAME_CHARSET_RE.pattern!r} (tier and app names become "
                "image-tag suffixes and path components)"
            ),
        )
        for render_name in _render_catalog(web_terminals)
        if isinstance(render_name, str) and not USERNAME_CHARSET_RE.fullmatch(render_name)
    ]


def _check_render_seed_base(web_terminals: dict[str, Any]) -> list[Finding]:
    """A render catalog entry's ``seed_base``, when present, must be a boolean.

    ``seed_base`` opts a render's users out of the mandatory shared base-context
    prepend at seed time (default ``true``). A non-boolean value (e.g. a YAML
    string ``"false"``) is a config typo — ``resolve_cards`` defensively
    coerces it back to ``True``, so without this check the opt-out would silently
    not take effect. Reported here so the mistake fails the config instead."""
    findings: list[Finding] = []
    for render_name, entry in _render_catalog(web_terminals).items():
        if not isinstance(entry, dict) or "seed_base" not in entry:
            continue
        value = entry["seed_base"]
        if not isinstance(value, bool):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.render_invalid_seed_base",
                    message=(
                        f"modules.web_terminals.renders[{render_name!r}].seed_base "
                        f"{value!r} is not a boolean; seed_base must be true or false"
                    ),
                )
            )
    return findings


def _check_default_tier_exists(web_terminals: dict[str, Any]) -> list[Finding]:
    """``default_tier``, when set, must name an entry in the render catalog
    — the entry every roster user with no ``tier:`` of its own inherits."""
    default_tier = _default_tier(web_terminals)
    if not isinstance(default_tier, str) or not default_tier:
        return []
    if default_tier in _tier_catalog(web_terminals):
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_default_tier",
            message=(
                f"modules.web_terminals.default_tier {default_tier!r} names no "
                f"tier: there is no {_canonical_delta_reference(default_tier)}"
            ),
        )
    ]


@dataclass(frozen=True)
class _UnreadableRender:
    """Why one referenced render's privileges could not be determined.

    The alternative to a ``None`` that means both "no privilege" and "no idea".
    At NEITHER altitude may those two answers collapse: a render whose document
    cannot be read may hold everything the base tier floors, and reading that as
    "holds nothing" is how a catalog entry pointed at a file outside
    ``tiers/`` walks an unauthenticated admin terminal past the gate — and,
    at rendered altitude, how a render whose project was never rendered (a
    ``project_path`` typo, a partial build) walks one past ``osprey up``.
    """

    #: The catalog entry's ``build_profile`` value, verbatim, so the message can
    #: quote what the operator actually wrote — including a non-string one.
    #: ``None`` at rendered altitude, where the delta is not what was read.
    build_profile: Any
    #: The path this tried to read, when it got as far as naming one.
    path_tried: str | None
    #: :func:`render_build_profile_shape_problem`'s clause, when the value's
    #: shape is itself the reason nothing resolved.
    shape_problem: str | None
    #: The catalog entry's ``project_path``, whose ``config.yml`` is not there:
    #: at rendered altitude for any entry, at profile altitude for an entry
    #: with no ``build_profile`` (a render the build does not render is read
    #: from its own render at both). Quoted verbatim rather than resolved — the
    #: join that decides where it resolves lives in
    #: :func:`~osprey.deployment.web_terminals.roster.rendered_config_file`
    #: and repeating it here is exactly the second convention that walk exists
    #: to prevent. ``None`` means "not this shape of failure", including when
    #: the entry declares no usable ``project_path`` at all.
    project_path: str | None = None
    #: The composer's own refusal, verbatim, when an app file read cleanly but
    #: the build would refuse to compose it: the reason the build gives is the
    #: one worth repeating, not that the file "cannot be read".
    refusal: str | None = None


#: Render name → the privileges it holds. Two of these travel together out of
#: :func:`_privileges_by_tier` — one absolute, one baseline-relative — and
#: naming the shape once keeps the pair readable at the call sites.
_PrivilegeMap = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _TierLayers:
    """One catalog render's config chain, and the one layer it wrote itself.

    Two answers rather than one, because two checks here ask different
    questions of the same render. What it *holds* is the whole chain merged —
    that is what gets deployed. What its author *said* is one layer of it, and
    the difference between the two is exactly an inherited key: see
    :func:`_check_readonly_tier_inherits_writes`, the check that cannot be
    written against the merged document at all.
    """

    #: The chain, base first, in the order a merge applies them — the argument
    #: list :func:`~osprey.deployment.web_terminals.roster.tier_privileges`
    #: and friends take.
    layers: tuple[Any, ...]
    #: The render's OWN ``config:`` block: the delta file's, or the preset's
    #: before ``extends`` folds its parents in. Always the last word in
    #: ``layers`` too, never a layer that is missing from it.
    authored: dict[str, Any]
    #: The catalog entry's ``build_profile`` value, verbatim — the file or
    #: preset name a message points the operator's edit at.
    source: str
    #: Whether *source* is a layer beside the profile (rather than a
    #: bundled preset name). The two inherit from different documents, which is
    #: the half a message has to phrase differently.
    is_delta: bool


def _profile_tier_layers(
    root: dict[str, Any],
    entry: dict[str, Any],
    *,
    profile_root: Path | None,
) -> _TierLayers | _UnreadableRender:
    """The config layers that decide one catalog render's privileges, at PROFILE altitude.

    Nothing is rendered yet and the catalog's ``build_profile`` names one of two
    things. A bundled PRESET name resolves on its own and carries its whole
    ``extends`` chain, so it is the only layer. A render DELTA
    (``tiers/<name>.yml``, which is what ``osprey init`` writes) is by
    construction *only* its own layer — it is merged over the host profile
    beside it — so the host's ``config:`` block, which is the document being
    linted, is layered underneath it. An entry with NO ``build_profile`` is a
    render the build does not render, and the one document that says what it
    holds is the render its ``project_path`` points at — see
    :func:`_rendered_elsewhere_layers`.

    The delta is resolved against ``profile_root``, the directory holding the
    profile being linted, and NOT against the working directory. Anchoring it on
    the cwd made the whole check a no-op from anywhere but the repo root: ``cd
    data && osprey build``, ``osprey profile validate <repo>`` from the
    directory above, and ``osprey build --repo <path>`` each read no delta at
    all and therefore reported every render as unprivileged — a guard that
    passes when it cannot see is worse than no guard, because the build says the
    word "valid".

    Returns:
        A :class:`_TierLayers`, or an :class:`_UnreadableRender` saying why
        there are none. "Cannot tell" is deliberately NOT "no privilege" here —
        see :func:`_check_unreadable_tier_privileges` for what is done with
        it.
    """
    build_profile = entry.get("build_profile")
    if not isinstance(build_profile, str) or not build_profile:
        return _rendered_elsewhere_layers(entry, profile_root=profile_root)

    shape_problem = render_build_profile_shape_problem(build_profile)
    if shape_problem is None and _is_composed_login(root, build_profile, profile_root):
        return _profile_composed_layers(root, build_profile, profile_root=profile_root)
    if shape_problem is None:
        # A delta beside the host profile: its own layer over the linted config.
        delta_file = (profile_root or Path(".")) / build_profile
        if not delta_file.is_file():
            return _UnreadableRender(build_profile, str(delta_file), None)
        try:
            with delta_file.open("r", encoding="utf-8") as fh:
                delta = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            return _UnreadableRender(build_profile, str(delta_file), None)
        authored = as_dict(as_dict(delta).get("config"))
        return _TierLayers((root, authored), authored, build_profile, is_delta=True)

    # Not a delta reference, so the only other thing it can be is a bundled
    # preset name — the shape the presets themselves ship with, before
    # `osprey init` rewrites the catalog. Resolving one needs the profile
    # engine, imported here rather than at module scope: this module is pure
    # static validation and is imported by the deploy path, which must not pull
    # the whole CLI resolution stack in behind it. `staleness.py` reaches the
    # same module the same way.
    try:
        from osprey.cli.build_profile import resolve_build_profile
        from osprey.cli.build_profile_resolve import preset_authored_config

        resolved, _preset_dir = resolve_build_profile(None, build_profile)
        # The same file, read one step earlier: what this preset says before its
        # `extends` parents are folded in. Taken here rather than recovered
        # afterwards, because the merged document no longer records which layer
        # any of its keys came from.
        authored = preset_authored_config(build_profile)
    except Exception:
        # The shape problem travels with the failure: a value like
        # `../admin.yml` is neither a delta this may read nor a preset name that
        # resolves, and naming both halves is the difference between "could not
        # resolve" and an instruction.
        return _UnreadableRender(build_profile, None, shape_problem)
    return _TierLayers((resolved.config,), authored, build_profile, is_delta=False)


def _is_composed_login(root: dict[str, Any], build_profile: str, profile_root: Path | None) -> bool:
    """Whether the login at *build_profile* renders composed rather than merged.

    A ``lists: own`` login is composed from the host and its file, and a login
    ``shared:`` names is sealed read-only; either way the layer chain a plain
    tier delta has is not the document the deployment runs.
    """
    shared = as_dict(as_dict(root.get("modules")).get("web_terminals")).get("shared")
    if isinstance(shared, list) and PurePosixPath(build_profile).stem in shared:
        return True
    try:
        with ((profile_root or Path(".")) / build_profile).open("r", encoding="utf-8") as fh:
            delta = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return False
    return as_dict(delta).get("lists") == "own"


def _profile_composed_layers(
    root: dict[str, Any], build_profile: str, *, profile_root: Path | None
) -> _TierLayers | _UnreadableRender:
    """The one config a narrow or shared login is deployed from, at PROFILE altitude.

    Such a login is never a delta merged over the host: the build composes it
    (:func:`~osprey.cli.build_profile_merge.render_document`) from the host
    profile and the login's file, sealing a shared one read-only. So the layer
    chain a plain tier has does not exist for it, and judging the file as a
    tier delta would read a document no deployment runs. The composed config
    is the single layer, and the login's own ``config:`` the authored one. The
    host is ``profile.yml`` beside the login's directory, resolved the way the
    build resolves it (its ``extends`` chain folded in). Where there is none —
    a bundled preset, whose ``tiers/`` sit beside the preset file rather than a
    ``profile.yml`` — the host is the document being linted, the same host a
    tier delta is layered over here.

    Returns:
        A :class:`_TierLayers`, or an :class:`_UnreadableRender` when either
        file cannot be read or the composition refuses the login.
    """
    base = profile_root or Path(".")
    app_file = base / build_profile
    if not app_file.is_file():
        return _UnreadableRender(build_profile, str(app_file), None)
    # The profile engine, imported here for the reason the preset branch of
    # `_profile_tier_layers` gives: this module is imported by the deploy path.
    try:
        from osprey.cli.build_profile_document import _read_profile_document
        from osprey.cli.build_profile_merge import _resolve_extends, render_document
        from osprey.utils.workspace import PROFILE_FILENAME
        from osprey_connectors.errors import BuildProfileError

        app_raw = _read_profile_document(app_file, build_profile)
        if not isinstance(app_raw, dict):
            return _UnreadableRender(build_profile, str(app_file), None)
        host_path = base / PROFILE_FILENAME
        if host_path.is_file():
            host_raw = _read_profile_document(host_path)
            if not isinstance(host_raw, dict):
                return _UnreadableRender(build_profile, str(host_path), None)
            host = _resolve_extends(dict(host_raw), host_path)
        else:
            host = {"config": copy.deepcopy(root)}
    except Exception:
        return _UnreadableRender(build_profile, str(app_file), None)
    try:
        composed = render_document(host, dict(app_raw), build_profile, layer_path=app_file)
    except BuildProfileError as refusal:
        # The files read; the composition refused the login, and says why. That
        # reason is the instruction, so it travels instead of "cannot be read".
        return _UnreadableRender(build_profile, str(app_file), None, refusal=str(refusal))
    except Exception:
        return _UnreadableRender(build_profile, str(app_file), None)
    config = as_dict(composed.get("config"))
    authored = as_dict(app_raw.get("config"))
    return _TierLayers((config,), authored, build_profile, is_delta=True)


def _rendered_elsewhere_layers(
    entry: dict[str, Any], *, profile_root: Path | None
) -> _TierLayers | _UnreadableRender:
    """The config a render the build does not render is judged by, at PROFILE altitude.

    A catalog entry with no ``build_profile`` points at a project rendered
    somewhere else — a render supplied ready-made rather than materialized from
    a delta beside the profile — and names where with its ``project_path``. The
    render's ``config.yml`` is the one document that says what that render
    holds, and it is the same file ``osprey up`` reads at rendered altitude
    (:func:`~osprey.deployment.web_terminals.roster.rendered_configs`),
    so a hand-written render proves its tier the same way at both altitudes
    and may be the ``default_tier`` or sit on a shared card like any other.

    One composed document, so it is both the only layer and the authored one:
    there is no delta or preset for an inherited key to have come from, which
    is why :func:`_check_readonly_tier_inherits_writes` finds nothing to ask
    of it.

    ``project_path`` is spelled against the repo root, which at this altitude is
    ``profile_root`` — the directory the profile being linted lives in — read
    through the one join every other reader of a render uses.

    Returns:
        A :class:`_TierLayers` holding the render, or an
        :class:`_UnreadableRender` naming the path tried. An entry naming no
        ``project_path`` at all is unreadable too: nothing points at a document.
    """
    project_path = entry.get("project_path")
    if not isinstance(project_path, str) or not project_path:
        return _UnreadableRender(
            build_profile=None, path_tried=None, shape_problem=None, project_path=""
        )
    config_yml = rendered_config_file(profile_root or Path("."), project_path)
    if not config_yml.is_file():
        return _UnreadableRender(
            build_profile=None,
            path_tried=str(config_yml),
            shape_problem=None,
            project_path=project_path,
        )
    try:
        with config_yml.open("r", encoding="utf-8") as fh:
            rendered = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return _UnreadableRender(
            build_profile=None,
            path_tried=str(config_yml),
            shape_problem=None,
            project_path=project_path,
        )
    document = as_dict(rendered)
    return _TierLayers((document,), document, str(config_yml), is_delta=False)


def _render_documents(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    *,
    rendered_project: bool,
    project_root: Path | None,
    profile_root: Path | None,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, _UnreadableRender]]:
    """The config layers each REFERENCED render is deployed from, at either altitude.

    The one render walk behind every rule here that asks what a render holds,
    so those rules cannot disagree about which renders they could read:

    * **Rendered project.** Each render's ``build/<project_path>/config.yml``
      is already the fully composed answer, so it is the only layer. Read
      through
      :func:`~osprey.deployment.web_terminals.roster.rendered_configs`
      — the same walk the credential grants use.
    * **Profile.** There is no merged document yet, so the chain
      :func:`_profile_tier_layers` returns is handed back as it stands, base
      first; a rule folds it the way it needs (see :func:`_fold_layers`).

    Returns:
        ``(layers, unreadable)``. A referenced render whose catalog entry does
        not resolve is in neither — that reference is reported elsewhere. One
        whose document could not be read is in ``unreadable`` only: "cannot
        tell" is never "holds nothing".
    """
    layers_by_render: dict[str, tuple[Any, ...]] = {}
    unreadable: dict[str, _UnreadableRender] = {}

    if rendered_project:
        documents = rendered_configs(root, project_root or Path("."))
        catalog = _render_catalog(web_terminals)
        for render_name in sorted(_served_names(root, web_terminals)):
            if render_name in documents:
                layers_by_render[render_name] = (documents[render_name],)
                continue
            if not isinstance(catalog.get(render_name), dict):
                continue  # unresolvable reference — reported elsewhere
            # No rendered config.yml where this render's project_path says one
            # is. "Cannot tell" is not "holds nothing" here either: once `osprey
            # up` gates on this belt, reading an absent render as unprivileged
            # is fail-open on the deploy path itself, and the only other signal
            # is `render_project_path_not_rendered_yet` — a WARN about a
            # different question that no error-filtering surface sees.
            project_path = as_dict(catalog.get(render_name)).get("project_path")
            unreadable[render_name] = _UnreadableRender(
                build_profile=None,
                path_tried=None,
                shape_problem=None,
                # `""` is the "declares none at all" case, which reads as its
                # own sentence rather than as a quoted empty path.
                project_path=project_path if isinstance(project_path, str) else "",
            )
        return layers_by_render, unreadable

    catalog = _render_catalog(web_terminals)
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        layers = _profile_tier_layers(root, entry, profile_root=profile_root)
        if isinstance(layers, _UnreadableRender):
            unreadable[render_name] = layers
            continue
        layers_by_render[render_name] = layers.layers
    return layers_by_render, unreadable


def _privileges_by_tier(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    *,
    rendered_project: bool,
    project_root: Path | None,
    profile_root: Path | None,
) -> tuple[_PrivilegeMap, _PrivilegeMap, dict[str, _UnreadableRender]]:
    """What each REFERENCED render can edit — absolutely, and beyond its baseline.

    Only referenced renders are read: a catalog entry nobody runs deploys
    nothing, and resolving it would make an unused entry able to fail a build.

    BOTH answers are returned because the two rules built on them disagree, on
    purpose. The shared-card rule reads the absolute set — a terminal every
    login may open hands out what its render holds, not what it holds more
    than its neighbours — and the ``default_tier`` rule reads the set beyond
    the baseline, because that key is inherited by profiles written before the
    floor existed. See
    :func:`~osprey.deployment.web_terminals.roster.privileges_beyond_baseline`
    for the full statement of the asymmetry.

    Both altitudes are read through :func:`_render_documents`, the render
    walk every rule here that asks what a render holds shares.

    Returns:
        ``(absolute, lifted, unreadable)``. A render absent from all three is
        readable and holds nothing at all. ``absolute`` and ``lifted`` each omit
        the renders whose entry would be empty, so a missing key reads as "no
        privilege" to
        :func:`~osprey.deployment.web_terminals.roster._privileged_entries`
        either way.
    """
    baseline = tier_privileges(root)
    absolute: _PrivilegeMap = {}
    lifted: _PrivilegeMap = {}

    def record(render_name: str, held: tuple[str, ...]) -> None:
        if held:
            absolute[render_name] = held
        beyond = privileges_beyond_baseline(held, baseline)
        if beyond:
            lifted[render_name] = beyond

    documents, unreadable = _render_documents(
        root,
        web_terminals,
        rendered_project=rendered_project,
        project_root=project_root,
        profile_root=profile_root,
    )
    for render_name, layers in documents.items():
        record(render_name, tier_privileges(*layers))
    return absolute, lifted, unreadable


def _deployment_declares_a_privilege_split(root: dict[str, Any]) -> bool:
    """Whether this deployment floors anything a render tier could lift.

    The precondition for the MIGRATION-SENSITIVE findings in this belt, asked
    without holding a render. A baseline that already grants the setup tool and
    the Config panel leaves nothing for any render to rise above, so there is
    no split to protect and nothing to report that would not simply be "you have
    not adopted tiers yet" — see
    :func:`~osprey.deployment.web_terminals.roster.privileges_beyond_baseline`.

    :func:`_privileges_by_tier` reaches the same verdict per render, by
    comparison. This is the version for the render whose document could not be
    read at all.

    NOT a precondition for the shared-card finding, nor for the advisory
    ``auth.method: none`` arm of the unreadable-render rule. The first is
    absolute (see :func:`_check_privileged_tier_exposure`); the second names
    an exposure the deployment has whatever it floors.
    """
    return bool(privileges_beyond_baseline(ALL_PRIVILEGES, tier_privileges(root)))


def _check_unreadable_tier_privileges(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    resolved_entries: list[dict[str, Any]],
    unreadable: Mapping[str, _UnreadableRender],
    *,
    rendered_project: bool,
) -> list[Finding]:
    """Refuse a render whose privileges cannot be read where it would matter.

    ``None`` used to mean "cannot tell" and was treated as "holds nothing",
    which is the fail-OPEN direction on the one question this belt exists to
    answer. A catalog entry whose ``build_profile`` points at ``../admin.yml``
    is neither a delta lint may read nor a preset name that resolves, so it read
    as unprivileged — and a roster that handed it out validated clean, built
    clean, and rendered a terminal holding the deployment-editing setup tool.

    Reported only where the answer would decide something, which is exactly
    the two rules this belt owns: the ``default_tier`` every unlabelled
    entry inherits, and a shared card (``access: any``) every login may open —
    plus, advisorily, a deployment that stands no login wall at all. A
    privileged tier behind one person's own login is not an exposure, so an
    unreadable one there is not one either, and refusing it would fail builds
    over catalog entries whose real problem — an absent delta, a preset that
    does not resolve — is already reported, in better words, by the check that
    owns it.

    Three narrowings, all deliberate:

    * The ``default_tier`` half is reported only where the deployment
      declares a privilege split (:func:`_deployment_declares_a_privilege_split`)
      — the same migration argument that makes the KNOWN version of that rule
      baseline-relative: an inherited default on a floorless deployment has no
      unprivileged tier to be re-pointed at. The shared-card half is NOT so
      narrowed, for the same reason its known version is absolute: on a
      floorless deployment a render nobody can read is one nobody can show
      holds less than everything, and that card is opened by every login.
    * At rendered altitude the ``default_tier`` half is further gated on
      ``image_source: local``. A registry-mode host has no local render
      by design, so an unreadable default is that mode's normal state rather
      than drift, and no remedy the operator can execute exists there — the
      profile altitude, where the deltas live, owns that question. Again the
      shared-card half is not gated: its subject is an entry that typed itself
      shared, and its remedy is that entry's own key.
    * With no login wall (``auth.method`` of ``token`` or ``none``) a shared
      card is inert (there is no authenticated identity for it to widen), so
      no entry singled itself out; the
      deployment-wide exposure is advisory (see
      :func:`~osprey.deployment.web_terminals.roster.deployment_wide_privileged_exposure_problems`)
      and an ERROR on a render that *might* be privileged would be a harder
      refusal than the one for a render known to be. So the unreadable render
      is reported as a WARN instead — not dropped. Under that posture a READABLE
      privileged tier still draws its own warning, and "cannot tell" must
      never be quieter than "known privileged"; skipping the advisory rung
      entirely was how one cell of the matrix (registry mode, no
      authentication) came to report nothing at all.

    **The remedy always includes a way out the operator holds.** The
    altitude-specific halves ("write the delta", "re-run ``osprey build``") both
    presuppose a document that can be made to resolve, which a deployment
    building its images in CI (``image_source: registry``, the default) cannot
    produce on the host it starts on. Wherever an entry's own ``access: any``
    is what put the render at stake, ``access: own`` for that card is offered
    too — the same remedy the KNOWN version of this rule
    (:func:`~osprey.deployment.web_terminals.roster.shared_card_privileged_problems`)
    leads with, so the two halves of one rule do not disagree about their own
    remedy set.
    """
    if not unreadable:
        return []
    declares_a_split = _deployment_declares_a_privilege_split(root)

    # Render name → every reason an unknown answer for it is not survivable.
    # A render can be at stake for BOTH halves at once — it is the
    # `default_tier` AND a shared card resolves to it — and the two reasons
    # have different remedies, so the finding says both rather than letting
    # whichever ran first win.
    at_stake: dict[str, list[str]] = {}
    # Render name → EVERY shared card that put it at stake, in roster order.
    # All of them, not just the first: the known half of this rule emits one
    # message per entry and names them all, and an operator who sets
    # `access: own` for the one name they were given only to meet the same
    # refusal for the next has paid for a round trip this message could have
    # saved them.
    shared_by: dict[str, list[Any]] = {}
    # Render name → the users a wall-less deployment serves it to. Kept apart
    # from `shared_by` because it is the ADVISORY arm: with no wall standing no
    # entry singled itself out, and the deployment-wide exposure this belongs
    # to is deliberately not build-failing.
    open_to_everyone: dict[str, list[Any]] = {}

    # The default_tier half is a RENDERED-altitude question only where a
    # render was supposed to happen here. With `image_source: registry` — the
    # default, and every value but the literal `local` — the images and their
    # deltas are built in CI and the deploy host holds no render project by
    # design, which is why `verify_renders` is skipped there too. So
    # "cannot tell" is that mode's normal state rather than drift, and no
    # remedy exists that an operator on a pull-only host can carry out. The
    # profile altitude, where the deltas live, owns that question.
    reads_a_local_render = not rendered_project or effective_image_source(web_terminals) == "local"
    default_tier = _default_tier(web_terminals)
    if (
        declares_a_split
        and reads_a_local_render
        and isinstance(default_tier, str)
        and default_tier in unreadable
    ):
        at_stake.setdefault(default_tier, []).append(
            "it is this deployment's default_tier, which every roster entry with no "
            "tier: key of its own inherits"
        )

    for entry in resolved_entries:
        render = entry.get("render")
        if not isinstance(render, str) or render not in unreadable:
            continue
        if not auth_is_enforced(web_terminals):
            open_to_everyone.setdefault(render, []).append(entry.get("name"))
            continue
        if not entry_is_shared(dict(entry)):
            continue
        shared_by.setdefault(render, []).append(entry.get("name"))
    for render_name, names in shared_by.items():
        many = len(names) > 1
        at_stake.setdefault(render_name, []).append(
            f"{'users' if many else 'user'} {_named_users(names)} "
            f"{'are' if many else 'is'} shared with somebody beyond "
            f"{'their' if many else 'its'} own user and "
            f"{'resolve' if many else 'resolves'} to it"
        )

    # Why an unknown answer is not assumed harmless, said the way the
    # deployment's own posture makes true — built from the same baseline reading
    # every remedy in this family uses, so one message cannot claim a floor in
    # one sentence and name it as missing in the next.
    stance = _unknown_privilege_stance(tier_privileges(root))
    findings: list[Finding] = []

    def report(
        render_name: str, reasons: list[str], remedy: str, severity: Literal["error", "warn"]
    ) -> None:
        findings.append(
            Finding(
                severity=severity,
                code="web_terminals.render_privileges_unknown",
                message=(
                    f"modules.web_terminals tier or app {render_name!r} could not be read, so "
                    f"this build cannot tell whether it holds "
                    f"{privilege_phrase(ALL_PRIVILEGES)} — and {', and '.join(reasons)}. "
                    # The shape clause is a whole sentence and brings its own
                    # full stop; the others do not.
                    f"{_unreadable_render_clause(unreadable[render_name]).rstrip('.')}. "
                    f"{stance}. {remedy}"
                ),
            )
        )

    def altitude_remedy(render_name: str, re_point: str) -> str:
        # The remedy is about the document that was not there, which is a
        # different document at each altitude: a delta the operator writes, or a
        # render `osprey build` produces. `re_point` names the roster edit that
        # sidesteps the document — the one remedy every host holds.
        if rendered_project:
            return (
                f"Re-run `osprey build` to render it, or point {re_point} at a tier "
                f"whose project is rendered"
            )
        return (
            f"Point build_profile at the delta beside this profile "
            f"(tiers/{render_name}.yml), or point {re_point} at a tier that does "
            f"resolve"
        )

    for render_name in sorted(at_stake):
        re_point = (
            "default_tier and the shared card"
            if render_name in shared_by and render_name == default_tier
            else "the shared card"
            if render_name in shared_by
            else "default_tier"
        )
        remedy = altitude_remedy(render_name, re_point)
        # The one remedy an operator can always carry out, and the one that
        # actually closes the door this half of the rule is about. Both remedies
        # above presuppose a document that resolves — a delta the operator has
        # to write, a render this deployment may build in CI rather than here —
        # so without this clause a registry-mode or delta-less deployment reads
        # the refusal as a dead end. Said at BOTH altitudes, and only where an
        # entry named itself, which is what `shared_by` records.
        if render_name in shared_by:
            remedy += f", or set access: own for {_named_users(shared_by[render_name])}"
        report(render_name, at_stake[render_name], remedy, "error")

    # The advisory rung. With no wall standing no entry singled itself out, so
    # nothing here is refused — but a READABLE privileged tier under that
    # same posture still draws
    # `unauthenticated_privileged_deployment` as a WARN, and this belt's whole
    # premise is that "cannot tell" is never quieter than "known privileged".
    # Reported for the renders the error arm did not already name, so one
    # unreadable render is one finding.
    for render_name in sorted(open_to_everyone):
        if render_name in at_stake:
            continue
        names = open_to_everyone[render_name]
        many = len(names) > 1
        report(
            render_name,
            [
                f"{'users' if many else 'user'} {_named_users(names)} "
                f"{'are' if many else 'is'} served with no login wall at all "
                f"(modules.web_terminals.auth.method is not password or oidc) and "
                f"{'resolve' if many else 'resolves'} to it"
            ],
            f"{altitude_remedy(render_name, 'the exposed terminal')}, or turn "
            f"authentication on (auth.method: password or oidc)",
            "warn",
        )
    return findings


def _named_users(names: Sequence[Any]) -> str:
    """``'carol'`` / ``'carol' and 'dave'`` — every user a clause points at.

    Names every entry rather than the first: an operator fixing a roster should
    see the whole list, not discover the second one after fixing the first. The
    subject and its verb are left to the caller, which has two of them to
    conjugate and one bare list to paste into a remedy.

    No square brackets: these strings reach ``osprey build``, which renders its
    refusals through rich and reads ``[...]`` as a style tag.
    """
    return privilege_phrase([repr(name) for name in names])


def _unknown_privilege_stance(baseline: Sequence[str]) -> str:
    """Why an unreadable render is not read as holding nothing, per posture.

    Built from the deployment's own baseline rather than from a "does it declare
    a split at all" boolean, so the sentence stays true of the deployment it is
    printed for. A PARTIALLY floored deployment used to be told it "floors
    those surfaces" while one of them was open to every render it has.
    """
    unfloored = tuple(privilege for privilege in ALL_PRIVILEGES if privilege in baseline)
    if not unfloored:
        return (
            "This deployment floors every door for its base tier, so a render nobody "
            "can read is not assumed harmless"
        )
    if len(unfloored) < len(ALL_PRIVILEGES):
        floored = tuple(privilege for privilege in ALL_PRIVILEGES if privilege not in unfloored)
        return (
            f"This deployment floors {privilege_phrase(floored)} for its base tier but "
            f"leaves {privilege_phrase(unfloored)} open to every render, so a render "
            f"nobody can read is not assumed harmless"
        )
    return (
        "This deployment floors no door, so a render nobody can read cannot "
        "be shown to hold anything less than every one of them — and this one is "
        "served to whoever opens its page"
    )


def _unreadable_render_clause(record: _UnreadableRender) -> str:
    """One sentence saying what was tried and why nothing came back."""
    if record.project_path is not None:
        # The render is what was read, not a delta: at rendered altitude for
        # every entry, at profile altitude for one with no build_profile.
        if not record.project_path:
            return (
                "Its catalog entry names neither a build_profile nor a project_path, so "
                "no document says what it holds"
            )
        if record.path_tried is not None:
            return (
                f"Its catalog entry names no build_profile, so it is read from its own "
                f"render — and its project_path {record.project_path!r} holds no readable "
                f"config.yml at {record.path_tried}"
            )
        return (
            f"Its project_path {record.project_path!r} holds no rendered config.yml, so "
            f"nothing was built there for this deployment to read"
        )
    if record.refusal is not None:
        return (
            f"Its build_profile {record.build_profile!r} is refused by the build: {record.refusal}"
        )
    if record.shape_problem is not None:
        # Phrased by `render_build_profile_shape_problem` to follow exactly
        # this subject, so the two halves are written once between them.
        return f"Its build_profile {record.shape_problem}"
    if not isinstance(record.build_profile, str) or not record.build_profile:
        return f"Its build_profile is {record.build_profile!r}, which names nothing"
    if record.path_tried is not None:
        return (
            f"Its build_profile {record.build_profile!r} resolves to {record.path_tried}, "
            f"which is not a readable profile"
        )
    return f"Its build_profile {record.build_profile!r} did not resolve to a profile"


def _check_privileged_tier_exposure(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    users: list[Any],
    *,
    rendered_project: bool,
    project_root: Path | None,
    profile_root: Path | None,
) -> list[Finding]:
    """Deployment-editing privilege must not be inherited by accident or handed to the roster.

    The belt behind the one build-time guard —
    :func:`osprey.cli.profile_cmd._privileged_tier_problems`, which
    :func:`~osprey.cli.profile_cmd._layer_profile_texts` raises at ``osprey
    init`` — catching the same mistakes on the paths that guard does not sit
    on: a profile edited after ``osprey init`` (``osprey profile validate``,
    ``osprey build``) and a project already rendered (``osprey up``). Every
    rule and every message comes from
    :mod:`~osprey.deployment.web_terminals.roster`, so the belt and the brace
    cannot disagree about what counts as privileged or about what to do next.

    **The rules read two different answers**, which is the one thing to know
    before reading the code below. The shared-card rule is judged on what a
    render ABSOLUTELY holds; the ``default_tier`` rule (and the advisory
    no-wall arm) on what it holds beyond its deployment's baseline. Exempting a
    floorless deployment from the first would be a hole big enough to drive the
    whole belt through: delete the base tier's deny floor from a profile and
    every render is at baseline, so a shared admin card would validate clean,
    build clean and lint clean while rendering a settings.json with nothing
    denied. See
    :func:`~osprey.deployment.web_terminals.roster.privileges_beyond_baseline`
    for why the other rules keep the relative reading.

    **A shared privileged card is an ERROR at BOTH altitudes.** The exposure is
    the same one either way — a card on the landing page that every login
    opens into a terminal that can edit the deployment — and the rendered
    altitude is the only place a hand-edited ``config.yml`` is ever read, so a
    warning there is a finding the surfaces that gate on errors discard.

    A privileged ``default_tier`` stays advisory against a RENDERED project.
    It is an authoring mistake, not an open door — the entries that inherit it
    are still behind whatever wall the deployment has. Refusing the start of a
    running stack over the shape of its roster would stop a shift to fix a
    profile. ``osprey up`` prints warnings as advisories, so it is said out
    loud either way, and rebuilding turns it into the error it is at profile
    altitude.
    """
    absolute, lifted, unreadable = _privileges_by_tier(
        root,
        web_terminals,
        rendered_project=rendered_project,
        project_root=project_root,
        profile_root=profile_root,
    )

    if not absolute and not lifted and not unreadable:
        return []

    # Resolved only once there is something to say about an entry: this walks
    # the whole roster, and a clean config must not pay for a report nobody is
    # going to make.
    registry_cfg = as_dict(root.get("registry"))
    resolved = list(resolve_cards(web_terminals, registry_cfg, strict=False)) if users else []

    findings: list[Finding] = _check_unreadable_tier_privileges(
        root, web_terminals, resolved, unreadable, rendered_project=rendered_project
    )
    if not absolute and not lifted:
        return findings

    # Baseline-relative, deliberately: an inherited default on a deployment with
    # no floor has no unprivileged tier to be pointed at. See
    # `privileges_beyond_baseline`.
    default_problem = privileged_default_tier_problem(
        _default_tier(web_terminals),
        lifted.get(str(_default_tier(web_terminals)), ()),
    )
    if default_problem is not None:
        findings.append(
            Finding(
                # See the docstring: an authored default blocks, a rendered one
                # is advisory.
                severity="warn" if rendered_project else "error",
                code="web_terminals.privileged_default_tier",
                message=default_problem,
            )
        )

    if not users:
        return findings

    if not auth_is_enforced(web_terminals):
        # No wall stands, so the exposure is the deployment's, not one entry's.
        # WARN, not error: see `deployment_wide_privileged_exposure_problems`
        # for why failing a build over the shipped default would reject
        # deployments nobody exposed.
        #
        # Baseline-RELATIVE, unlike the shared-card rule below. Not because the
        # absolute exposure is smaller — it is the same door — but because this
        # arm names every roster entry at once and never blocks: read
        # absolutely, every legacy deployment (no floor, and `auth.method: none`
        # is the default) would print one advisory per user on every `osprey up`
        # for a posture it has always had. The narrower claim, `access: any`,
        # is the one that gets the absolute reading and the refusal.
        return findings + [
            Finding(
                severity="warn",
                code="web_terminals.unauthenticated_privileged_deployment",
                message=problem,
            )
            for problem in deployment_wide_privileged_exposure_problems(resolved, lifted)
        ]

    # A shared card is judged on the ABSOLUTE answer, and only reported where a
    # wall stands: with no sidecar there is no authenticated identity for
    # `access: any` to widen (see `_check_user_access` on why that is a carried
    # key rather than a finding), and the deployment-wide arm above already
    # names every privileged entry once.
    findings.extend(
        Finding(
            severity="error",
            code="web_terminals.shared_card_privileged",
            message=problem,
        )
        for problem in shared_card_privileged_problems(resolved, absolute)
    )
    return findings


#: The two fixed ends of ``control_system.connector.<type>.writes_enabled``,
#: which this module matches one path segment at a time. The section comes from
#: the tier key the runtime reads and the retired leaf from the build's
#: migration code, the one place that still reads that spelling; the
#: ``connector`` hop in the middle is spelled here because neither keeps a
#: constant for it.
_CONTROL_SYSTEM_SECTION = WRITES_KEY.split(".")[0]
_CONNECTOR_TABLE_KEY = "connector"

#: The retired flat posture key, ``control_system.writes_enabled``.
_RETIRED_FLAT_KEY = f"{_CONTROL_SYSTEM_SECTION}.{RETIRED_WRITES_LEAF}"


def _config_leaves(*layers: Any) -> dict[str, Any]:
    """Every value the layers declare, one dotted path per leaf, later wins.

    A config layer may spell the same place two ways — a profile's ``config:``
    block is a flat bag of dotted keys, a preset's own block nests where it
    likes, and a rendered ``config.yml`` is fully nested — so flattening first
    is what keeps a check from depending on which spelling its author chose.
    Nested and dotted spellings of one path collapse onto the same entry here,
    which is also what lets two layers written in different styles be compared
    key for key.

    Layers fold base first, so a later layer's leaf replaces an earlier one's —
    the merge order the build itself applies.
    """
    spelled: list[tuple[str, Any]] = []
    for layer in layers:
        _collect_every_leaf(layer, (), spelled)
    return dict(spelled)


def _collect_every_leaf(node: Any, prefix: tuple[str, ...], into: list[tuple[str, Any]]) -> None:
    """Every leaf of one config layer as ``(dotted path, value)``, repeats kept.

    Unlike :func:`_config_leaves`, two spellings of one path both appear, so a
    check that every spelling holds cannot be satisfied by whichever came last.
    """
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        path = prefix + tuple(str(key).split("."))
        if isinstance(value, Mapping) and value:
            _collect_every_leaf(value, path, into)
        else:
            into.append((".".join(path), value))


def _connector_writes_type(path: str) -> str | None:
    """The connector type ``control_system.connector.<type>.writes_enabled`` names.

    ``None`` for any other path. The type is everything between the two fixed
    ends rather than one segment, because a custom connector's type key is its
    dotted module path (``mypackage.MoatConnector``) and rejoining it is what
    keeps the message naming the key the operator would have to write.
    """
    parts = path.split(".")
    if len(parts) < 4:
        return None
    if parts[0] != _CONTROL_SYSTEM_SECTION or parts[1] != _CONNECTOR_TABLE_KEY:
        return None
    if parts[-1] != RETIRED_WRITES_LEAF:
        return None
    return ".".join(parts[2:-1])


#: The hook that refuses a write tool call while the target's posture is off.
#: Every shared login's render selects it, so a write switch is never its only guard.
_WRITES_CHECK_HOOK = "writes-check"


#: The one write posture a shared login's render may hold.
_SHARED_WRITES = "never"


def lint_shared_render(render: Mapping[str, Any]) -> list[Finding]:
    """Check that one sealed shared login's render can neither write nor edit the deployment.

    A login ``modules.web_terminals.shared`` names is opened by everyone on
    the roster, never as the deployment's writer: its
    ``control_system.writes`` is pinned ``never``, no retired
    ``writes_enabled`` leaf (the flat one or a connector type's) holds
    anything but ``false``, ``writes-check`` is among its hooks, and every
    door in :data:`~osprey.deployment.web_terminals.roster.DEPLOYMENT_EDIT_DOORS`
    is closed — the setup tool denied (it rewrites the render's own config,
    the tier key included), the Config panel and gallery writes off. Every
    spelling of a write switch or a door switch counts — dotted, partly
    nested, nested, or under a custom connector's dotted type — each one is
    checked; the tier key must be present, and only the literal ``never``
    passes.

    Args:
        render: The resolved shared render (its ``hooks`` and ``config``).

    Returns:
        One ``web_terminals.shared_render_holds_privilege`` error naming every
        problem, or ``[]`` for a render that holds nothing.
    """
    from osprey.cli.profile_conventions import SETUP_PATCH_TOOL, is_setup_patch_capable
    from osprey.deployment.web_terminals.roster import (
        DEPLOYMENT_EDIT_DOORS,
        render_capability_document,
    )

    config = render.get("config")
    config = config if isinstance(config, Mapping) else {}
    spelled: list[tuple[str, Any]] = []
    _collect_every_leaf(config, (), spelled)
    problems: list[str] = []
    # A scalar written where the section belongs (`control_system: x`) is one
    # more spelling, and never a pinned one.
    tier_key = [
        (path, value)
        for path, value in spelled
        if path == WRITES_KEY
        or (path == WRITES_KEY.split(".")[0] and not isinstance(value, Mapping))
    ]
    if not any(path == WRITES_KEY for path, _ in tier_key) or any(
        path != WRITES_KEY or value != _SHARED_WRITES for path, value in tier_key
    ):
        problems.append(f"{WRITES_KEY} is not pinned to {_SHARED_WRITES}")
    for path, value in spelled:
        if path != _RETIRED_FLAT_KEY and _connector_writes_type(path) is None:
            continue
        if value is not False:
            problems.append(f"{path} is {value!r}")
    from osprey.cli.profile_conventions import library_selection

    hooks = render.get("hooks")
    # A path to the deployment's copy of the kill switch selects it.
    if not isinstance(hooks, list) or _WRITES_CHECK_HOOK not in library_selection("hooks", hooks):
        problems.append(f"{_WRITES_CHECK_HOOK!r} is not among its hooks")
    if is_setup_patch_capable(render_capability_document(dict(config))):
        problems.append(f"{SETUP_PATCH_TOOL!r} is not denied, so it could rewrite its own config")
    for door in DEPLOYMENT_EDIT_DOORS:
        if not door.is_switch:
            continue
        # A scalar written where the door's section belongs
        # (`web.config_panel: true`) is one more spelling, and never a closed one.
        declared = [
            (path, value)
            for path, value in spelled
            if path == door.key or (path.count(".") >= 1 and door.key.startswith(path + "."))
        ]
        if not any(path == door.key for path, _ in declared):
            problems.append(f"{door.key} is not pinned to false")
        for path, value in declared:
            if path != door.key or value is not door.closed:
                problems.append(
                    f"{path} is {value!r}, so {door.privilege} could edit the deployment"
                )
    if not problems:
        return []
    return [
        Finding(
            "error",
            "web_terminals.shared_render_holds_privilege",
            "A shared login's render must not be able to write or edit the deployment, "
            "and this one can: "
            + "; ".join(problems)
            + f". A login named by modules.web_terminals.shared is sealed with {WRITES_KEY} "
            f"pinned to {_SHARED_WRITES}, no retired writes_enabled key, every "
            "deployment-editing door closed and 'writes-check' selected; rebuild with "
            "`osprey build` rather than editing the render, and give write or editing "
            "access through a person's login.",
        )
    ]


def _layer_writes(layer: Any) -> str | None:
    """The ``control_system.writes`` one config layer states, read as the build reads it.

    ``never``, ``asked`` or ``allowed``, the words the parse admits: the tier
    key's value when the layer spells it (any spelling, later wins);
    else what the layer's retired ``writes_enabled`` leaves amount to — the
    rewrite the normalizer applies to every parsed document, applied here too
    because a layer this module reads may be a raw mapping; else ``None``, a
    layer that says nothing about the posture.
    """
    from osprey.cli.build_profile_document import retired_write_answer

    leaves = _config_leaves(layer)
    if WRITES_KEY in leaves:
        return cast(str | None, leaves[WRITES_KEY])
    return retired_write_answer(as_dict(layer))


def _check_readonly_tier_inherits_writes(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    *,
    rendered_project: bool,
    profile_root: Path | None,
) -> list[Finding]:
    """A render whose own layer says ``control_system.writes: never`` must not render armed.

    ``control_system.writes`` is one deployment-wide key, so a tier layer that
    writes ``never`` — the line an operator scans for — overrides whatever the
    document underneath it says, and the retired per-type leaves that could
    arm a machine behind the flat key are rewritten to the tier key per
    document before any merge. The read-only-looking tier that inherited an
    armed connector block is therefore closed by construction. This check
    stays as the guard on that claim: it reads the posture the author wrote in
    the AUTHORED layer and the posture the chain resolves to — each layer read
    as the build reads it (:func:`_layer_writes`), a later layer winning — and
    reports the render where the two disagree, which only a spelling the merge
    does not fold the way the author expected can produce.

    Keyed on the authored file as well as the merged one, which is why the
    layers carry both: the chain alone cannot say what the author meant.
    Profile altitude only. A rendered ``config.yml`` is one composed document
    with no authored layer left in it, so there is no question to ask there.
    """
    if rendered_project:
        return []

    catalog = _render_catalog(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        layers = _profile_tier_layers(root, entry, profile_root=profile_root)
        if isinstance(layers, _UnreadableRender):
            continue  # reported by `_check_unreadable_tier_privileges`

        # Only the literal `never` is a read-only claim: `asked` over an `allowed`
        # root is an arming author, and silence claims nothing. An invalid word
        # is refused at parse.
        if _layer_writes(layers.authored) != "never":
            continue
        if not _layers_allow_writes(*layers.layers):
            continue

        layer_word = "delta" if layers.is_delta else "preset"
        inherited_from = (
            "the profile it is merged over"
            if layers.is_delta
            else f"the preset {layers.source!r} extends"
        )
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.tier_inherits_armed_connector",
                message=(
                    f"modules.web_terminals tier {render_name!r} sets "
                    f"{WRITES_KEY}: never in {layers.source!r}, but its render resolves "
                    f"{WRITES_KEY}: {_resolved_writes(*layers.layers)} from "
                    f"{inherited_from}, so this tier "
                    f"would be served as a read-only tier while writes stay armed for it. "
                    f"Spell {WRITES_KEY} the same way (one dotted key) in this {layer_word} "
                    f"and in {inherited_from}"
                ),
            )
        )
    return findings


def _fold_layers(*layers: Any) -> dict[str, Any]:
    """One nested document out of a render's config layers, later layers winning.

    The merged document a rendered project already is, assembled from the chain
    :func:`_render_documents` returns at profile altitude, where the layers mix
    dotted and nested spellings of the same keys (see :func:`_config_leaves`).
    """
    return _nest_dotted(_config_leaves(*layers))


def _resolved_writes(*layers: Any) -> str | None:
    """The ``control_system.writes`` a render's config layers resolve to, ``None`` if silent.

    Each layer read as the build reads it (:func:`_layer_writes`), a later
    layer winning.
    """
    resolved: str | None = None
    for layer in layers:
        stated = _layer_writes(layer)
        if stated is not None:
            resolved = stated
    return resolved


def _layers_allow_writes(*layers: Any) -> bool:
    """Whether a render's config layers resolve to a tier that arms: the word is not ``never``.

    Read through :func:`~osprey_connectors.types.tier_allows`, so silence and any
    value a raw layer carries that is not an arming word read ``never``.
    """
    return tier_allows({"writes": _resolved_writes(*layers)})


def _live_writes_type(section: Any, *, tier_allows_writes: bool) -> str | None:
    """The connector type a render writes to for ``live``, when its tier allows writes.

    ``None`` unless all of it holds: the render's login tier allows writes
    (*tier_allows_writes*), ``live`` is one of the section's
    :func:`~osprey_connectors.types.configured_targets`, and it resolves to a
    real connector type, and a session can be on it. The tier is target-blind,
    so a tier that allows writes allows them on every machine a session can
    reach; what this adds is the machine. A ``live`` that does not resolve — a
    mock or other simulated deployment that never named its real machine — is
    not live here: a simulator has no control system to attribute writes on.

    A session is on ``live`` either because it is the deployment's baseline, or
    because the target switch moved it there. The switch only moves a session
    to a Channel Access block that names its ``gateways`` and a
    ``probe_channel`` (the config-only half of the switch's eligibility check),
    so a ``live`` block left unauthored — the shipped preset's ``epics:`` block
    holds only a timeout — is a machine nobody's writes reach. The operator's
    acknowledgment is not required here: it is the last step before the first
    switch, and the card should be named before that.
    """
    if not tier_allows_writes or TARGET_LIVE not in configured_targets(section):
        return None
    try:
        live_type = resolve_target(section, TARGET_LIVE)
    except ValueError:
        return None
    if baseline_target(section) == TARGET_LIVE:
        return live_type
    connector = section.get("connector") if isinstance(section, dict) else None
    block = connector.get(live_type) if isinstance(connector, dict) else None
    if not isinstance(block, dict) or not speaks_channel_access(section, live_type):
        return None
    probe = block.get("probe_channel")
    if not block.get("gateways") or probe is None or not str(probe).strip():
        return None
    return live_type


def _check_live_writer_without_control_identity(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    users: list[Any],
    *,
    rendered_project: bool,
    project_root: Path | None,
    profile_root: Path | None,
) -> list[Finding]:
    """An owner-only card that writes to the real machine should say who is writing.

    A WARN, per roster entry: behind a login wall, an owner-only card whose
    render's tier allows writes and which reaches the ``live`` target, and which carries no
    ``control_identity``. Its writes then reach the control system under the
    container's shared account, so the machine's own records cannot tell which
    person made them — the one attribution ``control_identity`` exists to give.

    Advisory rather than an error: the deployment works, and a facility whose
    control system does not attribute by account loses nothing. Silent where
    the question has no answer — with no wall there is no person behind a card,
    a shared card must not carry one person's identity (that is an error of its
    own), and a render that cannot write to ``live`` writes nothing to
    attribute. Renders whose documents cannot be read are reported by the
    privilege belt, not here.
    """
    if not users or not auth_is_enforced(web_terminals):
        return []

    documents, _unreadable = _render_documents(
        root,
        web_terminals,
        rendered_project=rendered_project,
        project_root=project_root,
        profile_root=profile_root,
    )
    live_writers: dict[str, tuple[str, str | None]] = {}
    for render_name, layers in documents.items():
        live_type = _live_writes_type(
            as_dict(_fold_layers(*layers).get("control_system")),
            tier_allows_writes=_layers_allow_writes(*layers),
        )
        if live_type is not None:
            live_writers[render_name] = (live_type, _resolved_writes(*layers))
    if not live_writers:
        return []

    registry_cfg = as_dict(root.get("registry"))
    findings: list[Finding] = []
    for entry in resolve_cards(web_terminals, registry_cfg, strict=False):
        render = entry.get("render")
        if not isinstance(render, str) or render not in live_writers:
            continue
        if entry.get("control_identity") or entry_is_shared(dict(entry)):
            continue
        name = entry.get("name")
        live_type, word = live_writers[render]
        findings.append(
            Finding(
                severity="warn",
                code="web_terminals.live_writer_without_control_identity",
                message=(
                    f"modules.web_terminals user {name!r} resolves to render "
                    f"{render!r}, whose {WRITES_KEY}: {word} reaches the live machine "
                    f"({live_type!r}), but sets no control_identity; its "
                    f"writes reach the control system under the container's shared "
                    f"account, so the machine cannot tell who made them. Set "
                    f"control_identity on {name!r} to the account name that person "
                    f"holds on the control system"
                ),
            )
        )
    return findings


def _check_unknown_tier_reference(
    root: dict[str, Any], web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Every tier the deployment serves must be a tier the projection lists.

    Walks the served set (:func:`_served_names`: every card's render,
    ``default_tier``, every role's tier, the dispatch tier) and reports each
    tier name with no ``kind: tier`` entry in ``modules.web_terminals.renders``
    — an app's name does not stand in for a tier — once per key that names it:

    * ``user <name>`` — the person's own ``tier:`` pin;
    * ``default_tier`` — what every person who names none inherits (also
      ``web_terminals.unknown_default_tier``, which judges the key itself);
    * ``authorization.roles.<role>.tier`` — a role binds the tier of every
      person who carries it, so this one rule covers a role naming a tier
      nobody has heard of, and a person carrying that role is not reported a
      second time;
    * ``dispatch.tier`` — the tier the dispatch workers run at.

    App names stay with :func:`_check_shared_apps`. Lenient throughout: an
    entry whose binding does not resolve has its own finding.
    """
    tiers = _tier_catalog(web_terminals)
    unknown = sorted(
        name
        for name, kind in _served_names(root, web_terminals).items()
        if kind == RENDER_KIND_TIER and name not in tiers
    )
    if not unknown:
        return []

    context = _authorization_ctx(web_terminals)
    roles: dict[str, str] = (context or {}).get("authorization_roles") or {}
    sources: dict[str, list[str]] = {name: [] for name in unknown}
    for user in users:
        # The entry's OWN reference (default_tier is reported on its own key).
        own = effective_tier(user, roles, None, strict=False)
        role = user.get("role") if isinstance(user, dict) else None
        if isinstance(role, str) and role in roles and own == roles[role]:
            continue  # bound by its role, which is reported below
        if own in sources:
            sources[own].append(f"modules.web_terminals user {as_dict(user).get('name')!r}")
    default_tier = _default_tier(web_terminals)
    if isinstance(default_tier, str) and default_tier in sources:
        sources[default_tier].append("modules.web_terminals.default_tier")
    for role, tier in sorted(roles.items()):
        if tier in sources:
            sources[tier].append(f"modules.web_terminals.authorization.roles.{role}.tier")
    dispatch_tier = dispatch_tier_name(root)
    if dispatch_tier in sources:
        sources[dispatch_tier].append("dispatch.tier")

    return [
        Finding(
            severity="error",
            code="web_terminals.unknown_tier_reference",
            message=(
                f"{source} references tier {name!r}, which names no tier: there "
                f"is no {_canonical_delta_reference(name)}"
            ),
        )
        for name in unknown
        for source in sources[name]
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

    Registry mode names every web-terminal image under ``registry.url``, the
    default tier's image included. Local mode builds its images, so a URL set
    there only draws a warning.
    """
    registry_url = configured_registry_url(root.get("registry"))
    has_url = bool(registry_url)
    image_source = effective_image_source(web_terminals)
    if image_source == "registry" and not has_url:
        return [
            Finding(
                severity="error",
                code="web_terminals.registry_mode_missing_url",
                message=REGISTRY_MODE_MISSING_URL,
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
                    "every render's image and never reads registry.url"
                ),
            )
        ]
    return []


def _check_requires_catalog(web_terminals: dict[str, Any]) -> list[Finding]:
    """The lint-side mirror of the strict guard in
    :func:`~osprey.deployment.web_terminals.roster.resolve_cards`: web
    terminals need a render catalog and a ``default_tier``, in either image
    source. ``osprey up`` never runs the full lint, so both fail closed."""
    default_tier = _default_tier(web_terminals)
    has_default = isinstance(default_tier, str) and bool(default_tier)
    if _tier_catalog(web_terminals) and has_default:
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.requires_catalog",
            message=RENDER_CATALOG_REQUIRED,
        )
    ]


def _read_project_name(config_yml_path: Path) -> str | None:
    """Best-effort read of a render project's own ``config.yml``
    ``project_name``. Any failure to open or parse degrades to ``None`` rather
    than raising — an unreadable ``config.yml`` is already its own ERROR (see
    :func:`_check_render_project_paths`, which only calls this once the file
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


def _check_render_bundle_path_agreement(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> list[Finding]:
    """Every entitled render must put its knowledge bundle where the deploy does.

    The split this catches: **entitlement** is decided by each render's own
    rendered ``config.yml`` (does it name a ``facility_knowledge.bundle_path``?),
    while the in-container **mount target** is derived from the DEPLOY config's
    ``bundle_path`` anchored on that render's project directory. Where the two
    values disagree, a render is entitled and gets the deployment's bundle bound
    at the deployment's path — while its OKF panel and ``facility_knowledge`` MCP
    server both read the render's path, which nothing mounted.

    The result is an empty bundle with no error anywhere: the panel lists no
    concepts, the agent's knowledge tools return nothing, and every layer reports
    success. Nothing at runtime can tell that apart from a facility that simply
    has not written any concepts yet, which is why it is caught here.

    An ERROR rather than a warning: unlike a render that is merely unrendered,
    nothing an operator does later resolves this. The two keys have to be made to
    agree.

    Read from disk on the same terms as :func:`_check_render_project_paths` —
    ``project_path`` resolved against ``project_root``, the deployment repo the
    caller names, an unreadable or unrendered project contributing nothing,
    since a render that has not been built yet is already reported by that
    check.
    """
    deploy_bundle = as_dict(root.get("facility_knowledge")).get("bundle_path")
    if not isinstance(deploy_bundle, str) or not deploy_bundle.strip():
        return []  # nothing is mounted at all — no target to disagree with
    deploy_bundle = deploy_bundle.strip()

    catalog = _render_catalog(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue
        project_path_raw = entry.get("project_path")
        if not isinstance(project_path_raw, str) or not project_path_raw:
            continue
        config_yml = (project_root or Path(".")) / project_path_raw / "config.yml"
        if not config_yml.is_file():
            continue
        render_bundle = _read_bundle_path(config_yml)
        if render_bundle is None or render_bundle == deploy_bundle:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.render_bundle_path_divergence",
                message=(
                    f"tier or app {render_name!r} sets facility_knowledge.bundle_path to "
                    f"{render_bundle!r}, but this deployment sets {deploy_bundle!r}. "
                    "The bundle is bind-mounted at the DEPLOYMENT's path while the "
                    "card's knowledge tools read its own, so that container would "
                    "see an empty bundle and report no error. Make the two agree"
                ),
            )
        )
    return findings


def _read_config(config_yml_path: Path) -> dict[str, Any]:
    """Best-effort parse of a rendered ``config.yml``; ``{}`` on any failure,
    like :func:`_read_bundle_path` (an unreadable project is already its own
    finding)."""
    try:
        with config_yml_path.open("r", encoding="utf-8") as fh:
            return as_dict(yaml.safe_load(fh))
    except (OSError, yaml.YAMLError):
        return {}


def _check_render_mirror_agreement(
    root: dict[str, Any],
    web_terminals: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> list[Finding]:
    """Every render that writes the ARIEL mirror must write the deployment's.

    The mirror counterpart of :func:`_check_render_bundle_path_agreement`.
    **Entitlement** to the mirror bind is decided by each render's own
    rendered ``config.yml`` (does it run a qmd export with a ``mirror_path``?),
    while the bind's SOURCE and its in-container TARGET both come from the
    deploy config's ``ariel.enhancement_modules.qmd_export.mirror_path``. Two
    ways for those to disagree, both silent:

    * The deployment writes no mirror at all. The render is entitled but the
      overlay emits no bind (there is no source), so its exporter writes the
      mirror into the container's writable layer — indexed by nothing,
      discarded at the next recreate — while every layer reports success.
    * The deployment writes one somewhere else. The deployment's directory is
      bound at the deployment's path; the render's exporter writes its own
      path, which nothing mounted.

    An ERROR: nothing at run time resolves either, and a search that returns
    nothing looks exactly like a logbook with nothing in it.
    """
    from osprey.deployment.compose_generator import configured_ariel_mirror_path

    deploy_mirror = configured_ariel_mirror_path(root)

    catalog = _render_catalog(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue
        project_path_raw = entry.get("project_path")
        if not isinstance(project_path_raw, str) or not project_path_raw:
            continue
        config_yml = (project_root or Path(".")) / project_path_raw / "config.yml"
        if not config_yml.is_file():
            continue
        render_mirror = configured_ariel_mirror_path(_read_config(config_yml))
        if render_mirror is None or render_mirror == deploy_mirror:
            continue
        if deploy_mirror is None:
            message = (
                f"tier or app {render_name!r} runs an ARIEL qmd export writing "
                f"{render_mirror!r}, but this deployment runs none, so no mirror "
                "directory is bound into that container: its exporter would write into "
                "the container's writable layer, which the qmd sidecar never indexes and "
                "the next recreate discards. Enable the export on the hosting profile "
                "(one shared mirror, indexed by the sidecar) or switch it off in the "
                "tier or app"
            )
        else:
            message = (
                f"tier or app {render_name!r} sets ariel.enhancement_modules.qmd_export."
                f"mirror_path to {render_mirror!r}, but this deployment sets "
                f"{deploy_mirror!r}. The mirror is bind-mounted at the DEPLOYMENT's path "
                "while the card's exporter writes its own, so its entries would land "
                "in the writable layer and never reach the sidecar. Make the two agree"
            )
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.render_mirror_path_divergence",
                message=message,
            )
        )
    return findings


def _check_render_project_collisions(
    root: dict[str, Any], web_terminals: dict[str, Any]
) -> list[Finding]:
    """Local mode: every render image tag must name exactly one image.

    A render image is tagged by its render alone
    (``<project>:local`` — see
    :func:`osprey.deployment.web_terminals.roster.resolve_cards`), so tag
    uniqueness is exactly ``project`` uniqueness, and that is enforced here
    rather than by a tag suffix. Config-only, so it runs at both altitudes —
    no filesystem is consulted. Two invariants:

    * **Render vs render.** No two referenced tiers or apps may share a
      ``project``. Each one is its own render, so two of them on one
      ``project`` race their builds onto one tag: the last build wins, and
      every card of the losing render silently runs the winning render's
      image — an entitlement bleed, not just a naming wart. Pointing both at
      one ``project_path`` does not make that safe: a tier and an app are
      separate definitions, and one render serving two names is a second
      name for one agent rather than a second agent.
    * **Render vs project.** No tier or app ``project`` may equal the
      deployment's own project name: the project image already owns that
      ``<project>:local`` tag (the dispatch worker and the deployment's own
      services run it), and a render build would overwrite it (or be
      overwritten) with entirely different content. Checked only when the
      config declares an explicit ``project_name`` — the resolver's
      path-derived fallbacks are unknowable at profile altitude, and every
      rendered project carries the key.

    A catalog entry with no ``project`` is skipped: it names no tag, and
    :func:`_check_render_missing_project` reports it.
    """
    if effective_image_source(web_terminals) != "local":
        return []
    catalog = _render_catalog(web_terminals)
    by_project: dict[str, list[tuple[str, str]]] = {}
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        project = entry.get("project")
        if not isinstance(project, str) or not project:
            continue  # no project: reported by _check_render_missing_project
        project_path = entry.get("project_path")
        by_project.setdefault(project, []).append(
            (render_name, project_path if isinstance(project_path, str) else "")
        )

    deployment_project = root.get("project_name")
    if isinstance(deployment_project, str) and deployment_project:
        deployment_project = resolve_project_name(root)

    findings: list[Finding] = []
    for project, members in sorted(by_project.items()):
        if len(members) > 1:
            names = ", ".join(repr(name) for name, _ in members)
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.render_project_collision",
                    message=(
                        f"tiers or apps {names} share project {project!r}; their builds "
                        f"would race onto the one image tag '{project}:local', and the "
                        "cards of the losing one would silently run the winning one's "
                        "image. Give each tier and app its own project (matching its "
                        "render)"
                    ),
                )
            )
        if project == deployment_project:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.render_shadows_project_image",
                    message=(
                        f"tiers or apps {', '.join(repr(name) for name, _ in members)} "
                        f"use project {project!r}, which is this deployment's own project "
                        f"name; their image tag '{project}:local' would collide with the "
                        "project image. Rename the render (osprey build names each one "
                        "<repo>-<name>)"
                    ),
                )
            )
    return findings


def _check_render_project_paths(
    root: dict[str, Any], web_terminals: dict[str, Any], *, project_root: Path | None = None
) -> list[Finding]:
    """Local mode: validate every referenced render's ``project_path``.

    ``project_path`` names the directory ``osprey up`` builds a render's
    image from. Two invariants are enforced here:

    * **Name invariant.** When ``project_path`` is set, its basename must equal
      the catalog entry's ``project``. ``osprey build`` names each render
      ``<repo>-<delta stem>`` and writes it into ``build/``, so a basename that
      disagrees with ``project`` would leave the render in one directory while
      the catalog builds/mounts
      another — a dead path at runtime. A mismatch is an ERROR regardless of
      whether the directory exists yet.
    * **Existence.** The directory must exist and hold a ``Dockerfile`` and a
      ``config.yml`` whose own ``project_name`` equals the catalog ``project``
      (a mismatch silently produces a dead mount, since the per-svc
      ``container_project_dir`` derivation is keyed on the catalog's ``project``,
      not on anything read from the render's own ``config.yml``).

    Existence is relaxed to a WARNING — never waived — for a ``project_path``
    that does not exist yet but whose entry carries a usable ``build_profile``.
    That is the ordinary state of a render added since the last build, and the
    ordinary command clears it: ``osprey build`` renders one project per delta.
    It is not an error because nothing is misconfigured; it is not merely
    informational because ``osprey up`` refuses to start until the render is
    there. A *partially* rendered directory that exists but is missing its
    ``Dockerfile``/``config.yml`` stays an ERROR — a half-written render is a
    broken build rather than an absent one.
    """
    if effective_image_source(web_terminals) != "local":
        return []
    catalog = _render_catalog(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — _check_unknown_tier_reference /
            # _check_default_tier_exists already report this
        findings.extend(
            _check_one_render_project_path(render_name, entry, project_root=project_root)
        )
    return findings


def _check_one_render_project_path(
    render_name: str, entry: dict[str, Any], *, project_root: Path | None = None
) -> list[Finding]:
    """The per-render body of :func:`_check_render_project_paths`: validate one
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
                code="web_terminals.render_missing_project_path",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}] has no "
                    "project_path set; image_source: local requires one to "
                    "build this tier or app's image from"
                ),
            )
        ]

    # Resolved against the deployment repo the caller names, not the working
    # directory: `osprey scaffold web-terminals lint` run from a subdirectory
    # otherwise reports every rendered render as missing.
    project_path = (project_root or Path(".")) / project_path_raw

    # Name invariant: the build writes each render at a path whose
    # basename is the catalog `project`, so project_path's basename must equal
    # it. A
    # disagreement is a hard config error regardless of whether the
    # directory exists yet, and supersedes every existence check below —
    # there is nothing else about this render worth reporting on top of it.
    if has_catalog_project and project_path.name != catalog_project:
        return [
            Finding(
                severity="error",
                code="web_terminals.render_project_path_name_mismatch",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}].project_path "
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
        problem = render_build_profile_shape_problem(cast(str, build_profile))
        if problem is not None:
            reference = _canonical_delta_reference(render_name)
            return [
                Finding(
                    severity="error",
                    code="web_terminals.render_build_profile_not_a_delta",
                    message=(
                        f"modules.web_terminals.renders[{render_name!r}].build_profile "
                        f"{problem} Set it to {reference!r} — the delta `osprey init` "
                        f"writes in this repo's {posixpath.dirname(reference)}/ directory, "
                        "which is what `osprey build` renders this project from. "
                        + PREDATES_DELTA_REMEDY
                    ),
                )
            ]

    if not project_path.is_dir():
        # Missing directory: a warning rather than the hard error when a
        # build_profile names the delta a build would render it from, because
        # that is the ordinary state of a render added since the last build and
        # the ordinary command clears it. It is not merely informational: no
        # start will run until it is cleared.
        if has_build_profile:
            return [
                Finding(
                    severity="warn",
                    code="web_terminals.render_project_path_not_rendered_yet",
                    message=(
                        f"modules.web_terminals.renders[{render_name!r}].project_path "
                        f"{project_path_raw!r} does not exist. `osprey build` renders it "
                        f"from the delta its build_profile names ({build_profile!r}); "
                        "`osprey up` REFUSES to start until it is there. Run `osprey build`"
                    ),
                )
            ]
        return [
            Finding(
                severity="error",
                code="web_terminals.render_project_path_not_dir",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}]."
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
                code="web_terminals.render_missing_dockerfile",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}]."
                    f"project_path {project_path_raw!r} has no Dockerfile; "
                    "local mode builds each tier or app's image from its own "
                    "project directory"
                ),
            )
        )

    config_yml_path = project_path / "config.yml"
    if not config_yml_path.is_file():
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.render_missing_config_yml",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}]."
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
                code="web_terminals.render_project_mismatch",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}].project "
                    f"{catalog_project!r} does not match its project_path's "
                    f"config.yml project_name {rendered_project_name!r}"
                ),
            )
        )
    return findings


def _check_registry_mode_build_profile(
    root: dict[str, Any], web_terminals: dict[str, Any]
) -> list[Finding]:
    """Registry mode only: every referenced non-default tier must set
    ``build_profile`` — the committed profile YAML that feeds its one
    ``.gitlab-ci.yml`` build job. The default tier is exempt: its image
    stays the un-suffixed ``web-terminal:latest``, built by the core CI job,
    not a per-render one."""
    if effective_image_source(web_terminals) != "registry":
        return []
    catalog = _render_catalog(web_terminals)
    default_tier = _default_tier(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        if render_name == default_tier:
            continue
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        build_profile = entry.get("build_profile")
        if isinstance(build_profile, str) and build_profile:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.render_missing_build_profile",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}] has no "
                    "build_profile set; image_source: registry needs one to "
                    "generate this non-default tier's CI build job"
                ),
            )
        )
    return findings


def _check_render_missing_project(
    root: dict[str, Any], web_terminals: dict[str, Any]
) -> list[Finding]:
    """Every referenced render must name the project its terminal runs.

    A terminal runs in ``/app/<project>``, so a render with no ``project``
    names no directory. An entry with a ``build_profile`` is skipped: the build
    derives its ``project``
    (:func:`~osprey.deployment.web_terminals.render_naming.derived_renders`),
    and at profile altitude there is nothing to derive yet. A ready-made entry —
    an image built elsewhere — states the project that image was built from.
    Config-only, so it runs at both altitudes, in either image source."""
    catalog = _render_catalog(web_terminals)
    findings: list[Finding] = []
    for render_name in sorted(_served_names(root, web_terminals)):
        entry = catalog.get(render_name)
        if not isinstance(entry, dict):
            continue  # unresolvable reference — reported elsewhere
        build_profile = entry.get("build_profile")
        if isinstance(build_profile, str) and build_profile:
            continue
        project = entry.get("project")
        if isinstance(project, str) and project:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.render_missing_project",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}] has no project; "
                    "a tier or app whose image is built elsewhere states the project that "
                    "image was built from (its directory is /app/<project>)"
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


def _check_render_extra_mounts(web_terminals: dict[str, Any]) -> list[Finding]:
    """Every ``modules.web_terminals.renders.<name>.extra_mounts`` entry must be a
    compose volume string (2 or 3 non-empty colon-separated parts). These generic
    host-path mounts are applied to every user of that render, so a malformed
    entry would render a broken per-user ``volumes:`` line — reject it here. The
    ``extra_mounts`` key is optional; an entry that omits it is never flagged."""
    findings: list[Finding] = []
    for render_name, entry in _render_catalog(web_terminals).items():
        if not isinstance(entry, dict):
            continue
        raw = entry.get("extra_mounts")
        if raw is None:
            continue
        if not isinstance(raw, list):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.render_extra_mounts_not_list",
                    message=(
                        f"modules.web_terminals.renders[{render_name!r}].extra_mounts "
                        f"must be a list of compose volume strings, got {type(raw).__name__}"
                    ),
                )
            )
            continue
        findings.extend(
            Finding(
                severity="error",
                code="web_terminals.render_invalid_extra_mount",
                message=(
                    f"modules.web_terminals.renders[{render_name!r}]."
                    f"extra_mounts entry {mount!r} is not a valid compose volume "
                    "string; expected 'source:target' or 'source:target:mode' "
                    "with non-empty colon-separated parts"
                ),
            )
            for mount in raw
            if not _is_valid_mount_string(mount)
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
                    "'registry.example.com:5050/mirrors/nginx:1.31-alpine'"
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
# The authoritative deploy-path gates live elsewhere and fail closed on their
# own: render.py raises on an unknown `auth.method` and on auth-without-TLS, and
# `auth_credentials.py` raises on a roster it cannot key credentials for.
# `osprey up` runs this module but refuses only on the codes in
# `provision._UP_BLOCKING_LINT_CODES`, so every other check below mirrors a gate
# that also exists downstream, except where the downstream path *cannot* see the
# mistake (see :func:`_check_auth_method` and :func:`_check_auth_session_lifetime`).


def _auth_context(
    web_terminals: dict[str, Any], *, base: int | None = None
) -> dict[str, Any] | None:
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

    The few checks that do read the raw stanza (:func:`_check_auth_method`,
    :func:`_check_auth_session_lifetime`) read it for well-formedness only — a
    value render would silently normalise away — never for its meaning.

    Args:
        web_terminals: The ``modules.web_terminals`` block being linted.
        base: The port base this deployment resolved. Only the caller that reads
            ``auth_port`` needs it — :func:`_check_port_overlap`, whose collision
            set is built at that base, so an un-based sidecar port would put two
            halves of one set on two different bases. Every other caller reads
            only the derived booleans and leaves it ``None``.

    Returns:
        The parsed context, or ``None`` when ``auth.method`` names no method.
    """
    try:
        return _auth_tls_context(web_terminals, base=base)
    except ValueError:
        return None


def _check_open_mode_egress(root: dict[str, Any], *, project_root: Path | None) -> list[Finding]:
    """An OPEN deployment whose renders can still reach the host network.

    The authoring-time voice of the deploy gate
    :func:`~osprey.deployment.web_terminals.artifacts.check_open_mode_requirements`,
    which refuses this deployment at ``osprey up``, at ``decommission``, and at
    the render seam. Without this rule the whole posture is discovered only when
    a start is attempted: ``osprey build`` and ``osprey profile validate`` would
    render and bless a deployment that cannot come up, and the operator would
    meet the refusal one step later than the edit that caused it.

    Driven by
    :func:`~osprey.deployment.web_terminals.artifacts.open_mode_missing_by_render`
    — the SAME predicate the gate raises on, not a second reading of the same
    idea. A lint rule that re-derived "which renders may reach the network"
    would be free to clear a deployment the gate refuses, which is the worst of
    both surfaces: a green authoring run and a start nobody can perform.

    ERROR rather than WARN, and for the gate's reason rather than lint's usual
    one: under ``auth.method: none`` nginx vouches for every terminal it
    proxies, so a render that keeps one of these entries is one prompt away
    from a neighbour's session.

    Imported at call time. This module is static validation of a config file,
    and ``artifacts`` pulls the deploy-time artifact writer — and the credential
    provisioner behind it — in with it; the same reason ``_AUTH_ENV_FILENAME``
    is quoted here rather than imported.

    Args:
        root: The whole parsed config — the gate reads the ``auth`` posture off
            ``modules.web_terminals`` itself, so it takes the config rather than
            that stanza.
        project_root: The deployment repo whose ``build/`` holds the renders a
            ``project_path`` resolves against. ``None`` falls back to the
            working directory, as everywhere else in this module.

    Returns:
        One finding naming every offender and what each is missing, or none.
    """
    from osprey.agent_runner.tool_names import OPEN_MODE_EGRESS_TOOLS
    from osprey.deployment.web_terminals.artifacts import open_mode_missing_by_render

    missing = open_mode_missing_by_render(root, project_root or Path("."))
    if not missing:
        return []
    detail = "; ".join(
        (
            f"{render!r} does not deny {', '.join(repr(tool) for tool in tools)}"
            if tools
            # The empty tuple is the gate's "there is nothing rendered here to
            # read", whose remedy is a render rather than a deny entry.
            else f"{render!r} has no rendered .claude/settings.json on this host"
        )
        for render, tools in sorted(missing.items())
    )
    return [
        Finding(
            severity="error",
            code="web_terminals.open_mode_egress",
            message=(
                f"modules.web_terminals.auth.method is 'none' (open), so nginx vouches "
                f"for every terminal it proxies — but {detail}. An agent in one terminal "
                f"reaches nginx over loopback and is served a neighbour's session, and "
                f"the python executor's socket guard covers only executed code. Every "
                f"render's shipped .claude/settings.json must deny all of "
                f"{', '.join(repr(tool) for tool in OPEN_MODE_EGRESS_TOOLS)} (a missing "
                f"or unparseable settings.json counts the same). Set auth.method to "
                f"'token' to keep the magic-link wall, or restore those deny entries, "
                f"render with `osprey build` and rebuild the images this deployment "
                "runs"
            ),
        )
    ]


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
                    "value falls back to 'token' at render time, which would render the "
                    "deployment with the login wall silently disabled"
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


def _check_auth_throttle(web_terminals: dict[str, Any]) -> list[Finding]:
    """``modules.web_terminals.auth.throttle`` must build the login throttle.

    One error per problem :func:`_auth_throttle_problems` names: a non-mapping
    block, a key outside the four, or a value the throttle's own predicate
    refuses. render refuses the same config, so this rule is the early report
    at scaffold time rather than the only gate. A non-mapping ``auth`` stanza is
    :func:`_check_auth_method`'s finding and is passed over here.
    """
    return [
        Finding(severity="error", code="web_terminals.invalid_auth_throttle", message=problem)
        for problem in _auth_throttle_problems(web_terminals)
    ]


def _check_auth_session_lifetime(web_terminals: dict[str, Any]) -> list[Finding]:
    """``modules.web_terminals.auth.session_lifetime`` must be a positive int.

    How long a terminal session cookie stays valid, in whole seconds. render
    reads it defensively and substitutes the default for anything that is not a
    whole number greater than zero, so ``0``, ``-1``, ``true`` and ``'12h'``
    all render as the default lifetime without a word — a deployment that meant
    to shorten its sessions silently keeps the long ones. Nothing downstream
    sees the mistake, so like :func:`_check_auth_method` this module is its only
    surface.

    An absent key, and a ``session_lifetime:`` written with no value at all
    (``None`` after YAML load), are the documented default and are not flagged.
    A non-mapping ``auth`` stanza is :func:`_check_auth_method`'s finding rather
    than this one's, so it is passed over here.
    """
    auth_raw = web_terminals.get("auth")
    if not isinstance(auth_raw, dict):
        return []
    if "session_lifetime" not in auth_raw:
        return []
    lifetime = auth_raw["session_lifetime"]
    if lifetime is None:
        return []
    # The same predicate the render reads, so this rule and the fall-back it
    # describes cannot drift: `bool` is excluded there because it passes
    # `isinstance(..., int)`, and `session_lifetime: true` becoming a
    # one-second session would be a baffling deployment.
    if is_positive_int(lifetime):
        return []
    return [
        Finding(
            severity="error",
            code="web_terminals.invalid_session_lifetime",
            message=(
                f"modules.web_terminals.auth.session_lifetime {lifetime!r} is not a whole "
                f"number of seconds greater than zero; render would silently fall back to "
                f"the default ({DEFAULT_SESSION_LIFETIME} s)"
            ),
        )
    ]


def _check_listener_ports(root: dict[str, Any], web_terminals: dict[str, Any]) -> list[Finding]:
    """``tls.port`` and ``auth.port`` must name a TCP port that exists.

    The two listeners the gated auth/TLS seam binds are both config-driven, and
    render reads both defensively (see
    :func:`~osprey.deployment.web_terminals.render._port_int`): anything that is
    not a whole number in ``1..65535`` — ``0``, a negative, ``true``, ``'8443'``
    as a string, ``70000`` — is substituted with the default. A deployment that
    meant to move nginx off 443, or the sidecar off its layout slot, therefore
    comes up on the port it was trying to leave, and nothing downstream sees
    the mistake: the render is well-formed, nginx starts, and the only symptom
    is a listener in the wrong place. Like :func:`_check_auth_method` and
    :func:`_check_auth_session_lifetime` this module is the only surface that
    can report it, so the rule reads the raw stanzas rather than render's
    parsed context — by the time that context exists the bad value is gone.

    An absent key, and a ``port:`` written with no value at all (``None`` after
    YAML load), are the documented default and are not flagged. A non-mapping
    ``tls`` or ``auth`` stanza has no keys to read and is passed over; the
    ``auth`` case is :func:`_check_auth_method`'s finding rather than this
    one's.

    Both ports are checked whether or not their seam is enabled. An unusable
    value is an authoring mistake in either case, and the config that turns the
    seam on is often the next edit.

    Args:
        root: The whole config, for the port base the sidecar's layout default
            is derived at — quoting the default block's port for a deployment
            that resolved another base would name a port it never binds.
        web_terminals: The ``modules.web_terminals`` block being linted.

    Returns:
        One finding per unusable port value.
    """
    base = resolve_port_base(root)
    stanzas = (
        ("tls", "tls.port", TLS_LISTEN_PORT),
        ("auth", "auth.port", default_port(_AUTH_PORT_SLOT, base=base)),
    )
    findings: list[Finding] = []
    for stanza_key, dotted, default in stanzas:
        stanza = web_terminals.get(stanza_key)
        if not isinstance(stanza, dict) or "port" not in stanza:
            continue
        port = stanza["port"]
        if port is None:
            continue
        # The same predicate the render reads (see
        # :func:`osprey.config_guards.is_positive_int`), bounded above by the
        # highest port there is: `bool` is excluded because it passes
        # `isinstance(..., int)`, and `tls.port: true` becoming a listener on
        # port 1 would be a baffling deployment.
        if is_positive_int(port) and port <= _MAX_PORT:
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_listener_port",
                message=(
                    f"modules.web_terminals.{dotted} {port!r} is not a whole number "
                    f"between 1 and {_MAX_PORT}; render would silently fall back to the "
                    f"default ({default})"
                ),
            )
        )
    return findings


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
    WARN is withheld when the origin browsers use (``external_origin`` when set,
    else the one derived from ``deploy.fqdn``) is on a loopback host, because no
    cookie crosses a network path. It is also withheld when that origin is
    ``https``, because a terminator in front carries the browser's leg over TLS,
    which is the topology the escape hatch is documented for. An origin that
    cannot be derived keeps the WARN.
    """
    context = _auth_context(web_terminals)
    if context is None or not context["sidecar_active"]:
        return []
    if context["tls_enabled"]:
        return []
    if context["auth_allow_insecure_http"]:
        try:
            origin = deployment_origin(root)
        except ValueError:
            pass
        else:
            if origin.is_loopback or origin.scheme == "https":
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


def _check_seeded_passwords(
    root: dict[str, Any], web_terminals: dict[str, Any], *, project_root: Path | None
) -> list[Finding]:
    """A login that still accepts a published password, on an origin off this machine.

    ``profile.yml`` publishes demo passwords under ``env.defaults``, and anyone
    who can read that file can sign in with them. That is harmless while the
    origin browsers use is on a loopback host, and an open login once it is not.
    HTTPS does not change this: it keeps a password from being sniffed, not a
    published one secret. An origin that cannot be derived is not read as
    loopback.

    Keyed on what the login wall accepts
    (:func:`~osprey.deployment.web_terminals.auth_credentials.seeded_password_users`),
    so a stored hash of the published value counts after its ``.env`` line is
    gone. A shared card's password cannot be changed while it is shared, so its
    remedy is deleting the card's stored hash from ``.env.auth`` and running
    ``osprey up``, which mints no password for a shared card.

    Imported at call time, for the reason :func:`_check_open_mode_egress` gives.

    Args:
        root: The whole parsed config, which the origin is derived from.
        web_terminals: The ``modules.web_terminals`` block being linted.
        project_root: The deployment repo holding ``profile.yml``, ``.env`` and
            ``.env.auth``. ``None`` falls back to the working directory.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "password":
        return []
    # The lenient normalizer drops an entry whose `access` is unreadable, so
    # `entry_is_shared` never refuses one of these.
    roster = roster_people(web_terminals, strict=False)
    shared = frozenset(entry["name"] for entry in roster if entry_is_shared(entry))

    from osprey.deployment.web_terminals.auth_credentials import seeded_password_users

    seeded = seeded_password_users(
        project_root or Path("."),
        [entry["name"] for entry in roster],
        shared=shared,
    )
    if not seeded:
        return []
    try:
        origin = deployment_origin(root)
    except ValueError as exc:
        where = f"this deployment's origin cannot be derived ({exc})"
    else:
        if origin.is_loopback:
            return []
        where = f"browsers reach this deployment at {origin.origin}"
    names = ", ".join(repr(name) for name in seeded)
    steps: list[str] = []
    own = [name for name in seeded if name not in shared]
    cards = [name for name in seeded if name in shared]
    if own:
        steps.append("run " + ", ".join(f"`osprey users passwd {name}`" for name in own))
    if cards:
        hashes = ", ".join(f"`{PW_HASH_VAR_PREFIX}{env_var_suffix(name)}`" for name in cards)
        steps.append(f"delete {hashes} from `{_AUTH_ENV_FILENAME}`, then run `osprey up`")
    remedy = " and ".join(steps)
    return [
        Finding(
            severity="error",
            code="web_terminals.auth_seeded_password",
            message=(
                f"{where}, and these logins still accept the password profile.yml "
                "publishes under env.defaults, so anyone who can read that file can sign "
                f"in: {names}. {remedy[0].upper()}{remedy[1:]}"
            ),
        )
    ]


def _check_auth_oidc(root: dict[str, Any], web_terminals: dict[str, Any]) -> list[Finding]:
    """``method: oidc`` needs an issuer, usable client env-var names, readable
    scopes, safe subjects, and an origin.

    Five ERRORs, all config-visible and all fatal at *request* time rather
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
    * **Scopes.** ``auth.oidc.scopes`` is a list of scope strings (or one
      space-separated string). Any other shape — including a list carrying an
      entry that is blank — renders no scopes line at all, so the sidecar's
      default applies and the authored scopes are gone without a trace.
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

    # `scopes` is read the same way and fails the same way: render joins a list
    # of strings (or an already-joined string) into the env line and treats
    # every other shape — including a list with a blank entry — as unset, so an
    # unusable value renders nothing and the sidecar's own default applies — the
    # authored scopes are gone with no trace. A blank entry gets its own finding
    # because the index is the whole repair, and reporting it as an unreadable
    # list would name everything except the one entry to fix. Only the SHAPE is
    # checked here; the sidecar refuses a list without `openid` at startup,
    # where it also knows what the IdP was asked for.
    if "scopes" in oidc:
        scopes = oidc.get("scopes")
        blank_index = _blank_scope_index(scopes)
        if blank_index is not None:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.auth_oidc_blank_scope",
                    message=(
                        f"modules.web_terminals.auth.oidc.scopes[{blank_index}] is blank "
                        "where a scope was meant to go; render would emit no scopes line "
                        "and the sidecar would fall back to its own default, silently "
                        "dropping what was authored"
                    ),
                )
            )
        elif _scope_list(scopes) is None:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.auth_oidc_invalid_scopes",
                    message=(
                        f"modules.web_terminals.auth.oidc.scopes {scopes!r} is neither a "
                        "non-empty list of scope strings nor a space-separated string; "
                        "render would emit no scopes line and the sidecar would fall "
                        "back to its own default, silently dropping what was authored"
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
    for entry in authored_people(web_terminals):
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
    # merely fills the ':port' suffix when TLS is off. The port is resolved the
    # way render resolves it, because an unset `nginx_port` is a legal config
    # whose origin is perfectly derivable — reading the key raw would have this
    # check report a placeholder origin for every deployment that takes the
    # layout's gateway slot. Only a value that is not a port has no port to
    # substitute, and that is render's own error, reported there; standing a
    # zero in for it keeps this check to the one thing it is about.
    try:
        nginx_port = resolve_nginx_port(root)
    except ValueError:
        nginx_port = 0
    try:
        _external_origin(
            root,
            nginx_port,
            tls_enabled=bool(context["tls_enabled"]),
            tls_port=int(context["tls_port"]),
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
    names = [name for name in (_roster_entry_name(user) for user in users) if name is not None]
    return [
        Finding(
            severity="error",
            code="web_terminals.auth_credential_collision",
            message=(
                f"modules.web_terminals.users entries {colliding} all map onto the "
                f"credential variable {PW_HASH_VAR_PREFIX}{suffix}; they would share a "
                "single password, so one user's credentials would open another's "
                "terminal. Rename one of them"
            ),
        )
        for suffix, colliding in env_var_suffix_collisions(names).items()
    ]


def _check_auth_stored_hashes(
    web_terminals: dict[str, Any], users: list[Any], *, project_root: Path | None
) -> list[Finding]:
    """Password mode: every stored hash in ``.env.auth`` must be one the login service can evaluate.

    A truncated paste or another tool's hash format leaves a user whom no
    password will log in, and the login page cannot say so. The shape test is
    :func:`~osprey.services.auth_sidecar.passwords.stored_hash_problem`, the same
    parse the sidecar runs, so this finding predicts the sidecar without deriving
    a key. The message names the user and the variable, never the value.

    WARN, not ERROR: the deployment still serves every other user, so this is
    work the operator has to do rather than a config to reject (see
    :class:`Finding`). Rendered-project only: the file lives in the deployment
    repo, which a profile does not have. Shared entries are included, because
    the sidecar loads a hash for every roster name. A blank entry is skipped,
    since ``osprey up`` provisions it; a missing or unreadable file is the deploy
    path's to report, and a lint that crashed on it would hide every other
    finding.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "password":
        return []
    env_auth = (project_root or Path(".")) / _AUTH_ENV_FILENAME
    if not env_auth.is_file():
        return []
    try:
        parsed = parse_dotenv_file(env_auth)
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    for name in (_roster_entry_name(user) for user in users):
        if name is None:
            continue
        suffix = env_var_suffix(name)
        value = parsed.get(f"{PW_HASH_VAR_PREFIX}{suffix}", "").strip()
        if not value:
            continue
        problem = stored_hash_problem(value)
        if problem is None:
            continue
        findings.append(
            Finding(
                severity="warn",
                code="web_terminals.auth_credential_unevaluable",
                message=(
                    f"{_AUTH_ENV_FILENAME} holds a {PW_HASH_VAR_PREFIX}{suffix} entry for "
                    f"{name!r} that the login service cannot evaluate ({problem}), so no "
                    f"password will log {name!r} in. Run `osprey users passwd {name}` to "
                    "replace it"
                ),
            )
        )
    return findings


def _check_shared_card_duplicate_subject(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """OIDC with a shared card on the roster: one subject may not map to two entries.

    Without a shared card a duplicated ``oidc_subject`` is legitimate — one
    person holding two cards, where every login arrives through the card that
    was clicked and the sidecar only confirms the asserted identity matches
    that one entry. A shared card changes the question: the sidecar must
    resolve the identity to a single roster entry on its own to know who
    opened it, and a subject carried by two entries has no one answer. The
    sidecar refuses such a login outright (``ambiguous_identity`` in the audit
    log) rather than minting a session whose roster identity nobody decided
    on; this is the same refusal at scaffold time, where dropping one
    ``oidc_subject:`` is still cheap.

    Scoped to ``method: oidc`` — no other method reads the subject mapping —
    and grouped like :func:`_check_auth_credential_collisions`: exact strings
    (stripped of surrounding whitespace) across the raw roster, one finding
    per colliding subject. Under an ``email`` claim the grouping is
    case-insensitive as well, because that is how the sidecar matches such a
    claim (:data:`~osprey.services.auth_sidecar.identity_headers.CASE_INSENSITIVE_CLAIMS`):
    two entries spelling one mailbox in two cases ARE the ambiguity it would
    refuse, and lint has to see the collision the callback would. The message
    names the entries, never the subject — same reasoning as the ``$`` scan
    above: echoing a value invites pasting it, and the names are what the
    operator edits.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "oidc":
        return []
    # `None` is render's "unset" — the sidecar's own default (`sub`) applies,
    # which is not on the case-insensitive list, so the exact grouping holds.
    claim = context.get("auth_oidc_claim") or ""
    fold = claim in CASE_INSENSITIVE_CLAIMS
    # The raw entry and the normalized one carry the same authored `access`
    # value — `normalize_users` copies it through verbatim — so the shared
    # predicate answers the same here as at the surfaces that deploy. The one
    # place the two readings part is a value the resolver refuses, which the
    # normalizer drops the whole entry over and `_is_shared_entry` reads as
    # not shared.
    if not any(_is_shared_entry(user) for user in users):
        return []
    by_subject: dict[str, list[str]] = {}
    for user in users:
        if not isinstance(user, dict):
            continue
        name = _roster_entry_name(user)
        subject = user.get("oidc_subject")
        if name is None or not isinstance(subject, str) or not subject.strip():
            continue
        key = subject.strip()
        by_subject.setdefault(key.casefold() if fold else key, []).append(name)
    return [
        Finding(
            severity="error",
            code="web_terminals.shared_card_duplicate_subject",
            message=(
                f"modules.web_terminals.users entries {colliding} all carry the same "
                "oidc_subject. With a shared card on the roster the auth sidecar must "
                "resolve every login to a single roster entry, so a login by that "
                "identity is ambiguous and refused (ambiguous_identity). A person "
                "holding two cards must carry oidc_subject: on one of them only"
            ),
        )
        for colliding in by_subject.values()
        if len(colliding) > 1
    ]


def _check_shared_card_subject(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """OIDC: a shared card that also carries a non-empty ``oidc_subject`` — a WARN.

    A shared card is opened with the OPENER's identity: the sidecar's callback
    reverse-matches the asserted subject over every roster entry's mapping,
    and the card's own subject participates like any other entry's. So a
    subject on the shared card itself maps one more identity that may open
    the card as itself — a documented shape (a service identity on its own
    shared card), which is why this is a WARN and not a refusal.

    The hazard is the other reading of the same two keys: an ``access`` that
    admits somebody else typo'd onto a *personal* card — the admin login, say
    — which hands that terminal to whoever the value names the moment ``oidc``
    is in force, with a green build. Nothing else in this module says so, and
    the entry alone cannot tell the two readings apart, so the message names
    the entry and both ways out.

    The message quotes the ``access`` value as authored rather than naming one
    spelling of it. Every shared form reaches this rule — ``any`` and any list
    that admits more than ``self`` alike — so a remedy phrased at ``access:
    any`` would name a key half these findings do not have.

    Scoped to ``method: oidc`` like the two subject checks above: no other
    method reads the mapping, and on a passive ``none``/``token`` base that an
    ``oidc`` variant arms both keys are carried base-entry state
    (:func:`_check_user_access`); the finding belongs to the merged render
    where the wall stands. The message names the user, never the subject —
    same reasoning as the ``$`` scan.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "oidc":
        return []
    findings: list[Finding] = []
    for user in users:
        if not _is_shared_entry(user):
            continue
        subject = user.get("oidc_subject")
        if not isinstance(subject, str) or not subject.strip():
            continue
        name = user.get("name", user)
        findings.append(
            Finding(
                severity="warn",
                code="web_terminals.shared_card_subject",
                message=(
                    f"modules.web_terminals.users entry {name!r} sets access "
                    f"{user.get('access')!r} and also carries an oidc_subject. A shared "
                    "card is opened with the opener's identity, so this subject only lets "
                    "that one identity open the card as itself — it is not the card's own "
                    f"login. If {name!r} is a personal card, narrow it to access: own (or "
                    "drop the key): as written, the card admits more than its own user. If "
                    "it is a shared card, its own oidc_subject is usually unneeded"
                ),
            )
        )
    return findings


# The ID-token claim a ``domain:`` principal can be read from. A domain is the
# part of an email address after the ``@``, so ``email`` is the one claim in the
# OIDC vocabulary that carries one; every other claim — the sidecar's own
# default included — holds an opaque identifier with no domain in it. Named
# here rather than imported: the sidecar knows the claim only as a configured
# string, and the module holding its default pulls the whole Flask app in with
# it.
_DOMAIN_CLAIM = "email"


def _access_members(user: Any) -> tuple[str, list[str]] | None:
    """An object-form entry's name and the ``access`` members it authored.

    The three rules below all ask something about the *members* an entry lists,
    so they share one reading of them — and one set of silences:

    * a bare-string entry, or an object-form entry without a string name, lists
      nothing and is skipped (as everywhere else in this module);
    * the two shorthands (``own``, ``any``), an absent key and a YAML-null one
      name no ``user:``/``domain:`` principal at all, so there is nothing here
      to ask about;
    * an ``access`` value
      :func:`~osprey.deployment.web_terminals.roster.resolve_access_principals`
      refuses is skipped, because :func:`_check_user_access` already reports
      that refusal verbatim. A second finding over the same value would bury the
      one sentence that says how to fix it.

    Members are returned as authored, not as resolved: the resolver
    ASCII-lower-cases a ``domain:`` value and de-duplicates the set, and both
    the rules and the operator need the line that was actually written.

    Args:
        user: One roster entry, of either form.

    Returns:
        The entry's name and its authored members, or ``None`` when the entry is
        one of the silences above.
    """
    name = _roster_entry_name(user)
    if not isinstance(user, dict) or name is None:
        return None
    authored = user.get("access")
    if not isinstance(authored, list):
        return None
    try:
        resolve_access_principals(user)
    except ValueError:
        return None
    # Every member is a string by now — the resolver refuses any that is not —
    # so this narrows the type rather than filtering anything out.
    return name, [member for member in authored if isinstance(member, str)]


def _check_access_principal_without_idp(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """Password mode: a ``user:`` or ``domain:`` principal names nobody. ERROR.

    Both kinds name an identity by the value of an ID-token claim, and a
    password login asserts no claims: the sidecar checks the credential it
    provisioned for a roster name, and no identity provider is in the exchange
    to say who the person is. So a member of either kind admits none of the
    people it names — on a card whose author wrote it precisely to admit them.

    ERROR rather than the WARN :func:`_check_claims_without_oidc` gives the
    other block that is inert under the wrong method. A ``claims`` map that
    never fires grants nothing, which is the safe direction to fail; an access
    member that never fires is the only thing standing between a shared card and
    the people it was opened for, and the operator reading the config file sees
    a grant that is not there.

    Scoped to ``method: password`` exactly, like
    :func:`_check_auth_credential_collisions`, and left silent under ``token``
    and ``none`` for the reason :func:`_check_user_access` gives at length:
    profile overlays concatenate the roster rather than merging entries by name,
    so a passive base that a host variant arms with ``oidc`` has to carry its
    ``access`` key on the base entry. A finding there would fire on every build
    of a base whose key is correct for the variant that uses it.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "password":
        return []
    findings: list[Finding] = []
    for user in users:
        listed = _access_members(user)
        if listed is None:
            continue
        name, members = listed
        for member in members:
            if not member.startswith(("user:", "domain:")):
                continue
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.access_principal_without_idp",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has an access member "
                        f"{member!r}, but auth.method is 'password': that member names an "
                        "identity by an ID-token claim, and a password login asserts no "
                        "claims, so nobody it names can open the card. Write 'access: any' "
                        "to open the card to the whole roster, or set auth.method: oidc to "
                        "admit the identities named here"
                    ),
                )
            )
    return findings


def _check_access_domain_without_email_claim(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """OIDC: a ``domain:`` principal needs the identity claim to be ``email``. ERROR.

    Lint accepts a ``domain:`` member only under ``auth.oidc.claim: email``,
    the one claim the provider guarantees to carry an address. An unset
    ``auth.oidc.claim`` resolves to ``sub``, which OpenID Connect defines as an
    opaque identifier. A claim set to something else may well hold an address
    — a UPN in the username claim, say — but nothing guarantees it, and a
    grant that depends on a provider's habit is a grant this rule refuses to
    write. The refusal is policy, not mechanics: the comparator the sidecar
    admits with reads a domain off whatever the configured claim asserts.

    ERROR, and reported at authoring time rather than left to the deployment,
    because of the shape of the failure: nothing is malformed, nothing raises,
    and the login page comes up. The first person to learn that the card admits
    nobody is the operator who needed it, at whatever hour they needed it.

    Scoped to ``method: oidc``. Under ``password`` the member is already
    reported by :func:`_check_access_principal_without_idp` — a second finding
    over the same line would say the same thing twice — and under ``token`` and
    ``none`` this module stays silent about carried ``access`` keys for the
    reason :func:`_check_user_access` gives.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "oidc":
        return []
    # `None` is render's "unset": the sidecar then applies its own default
    # claim, which is not `email` — so an unset claim is one of the cases here.
    claim = context.get("auth_oidc_claim") or ""
    if claim == _DOMAIN_CLAIM:
        return []
    configured = (
        f"auth.oidc.claim is {claim!r}"
        if claim
        else "auth.oidc.claim is not set, so it resolves to the opaque 'sub'"
    )
    findings: list[Finding] = []
    for user in users:
        listed = _access_members(user)
        if listed is None:
            continue
        name, members = listed
        for member in members:
            if not member.startswith("domain:"):
                continue
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.access_domain_without_email_claim",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has an access member "
                        f"{member!r}, but {configured}: a domain is matched only under "
                        "auth.oidc.claim: email, the one claim guaranteed to carry an "
                        "address. Set auth.oidc.claim: email, or name the people "
                        "individually with 'user:<id>'"
                    ),
                )
            )
    return findings


def _resolved_principal(entry_name: str, member: str) -> str:
    """The one principal an authored ``access`` member resolves to.

    The resolver is asked rather than imitated. It ASCII-lower-cases a
    ``domain:`` value and leaves a ``user:`` value byte-exact, and those two
    rules are the whole of what makes two written members one grant — so a rule
    that re-implemented the fold would be free to drift out of step with the
    admission check it exists to anticipate, silently, in the direction of
    reporting nothing.

    Args:
        entry_name: The entry's ``name``, which the resolver quotes in refusals.
        member: One member of that entry's authored ``access`` list.

    Returns:
        The resolved principal. Never raises for a member that came from
        :func:`_access_members`, whose whole list has already resolved.
    """
    return next(iter(resolve_access_principals({"name": entry_name, "access": [member]})))


def _check_duplicate_access_principals(
    web_terminals: dict[str, Any], users: list[Any]
) -> list[Finding]:
    """OIDC: one card may not name the same principal twice. ERROR.

    Two members that resolve to one principal are not additive — the set is a
    union, so the second grants nothing. The hazard is what happens when the
    grant is withdrawn: the operator deletes the line they recognise, the build
    stays green, and the card is still open through the duplicate they did not
    see. A grant that survives its own revocation is the one failure this
    vocabulary must not have.

    Both kinds are grouped, on the fold each one is compared by:

    * ``domain:`` under every claim, because the resolver ASCII-lower-cases the
      value and the login side folds the asserted domain the same way — so
      ``domain:EXAMPLE.org`` and ``domain:example.org`` are one principal
      wherever they are read, and the pair is invisible in the resolved set;
    * ``user:`` byte-exact, and additionally without regard to case under an
      ``email`` claim — the fold
      :data:`~osprey.services.auth_sidecar.identity_headers.CASE_INSENSITIVE_CLAIMS`
      names and :func:`_check_shared_card_duplicate_subject` already groups the
      subject mapping by. Under every other claim the value is opaque and
      compared byte-exact, so only a literal repeat is a duplicate; the resolver
      keeps ``user:`` byte-exact for exactly that reason, which leaves this the
      only place the pair can be caught.

    Scoped to ``method: oidc``, on the same split as the two rules above: under
    ``password`` every one of these members is already reported once, and under
    ``token``/``none`` carried ``access`` keys stay silent.
    """
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] != "oidc":
        return []
    claim = context.get("auth_oidc_claim") or ""
    fold_identities = claim in CASE_INSENSITIVE_CLAIMS
    findings: list[Finding] = []
    for user in users:
        listed = _access_members(user)
        if listed is None:
            continue
        name, members = listed
        by_principal: dict[str, list[str]] = {}
        for member in members:
            if not member.startswith(("user:", "domain:")):
                continue
            principal = _resolved_principal(name, member)
            if fold_identities and member.startswith("user:"):
                principal = principal.casefold()
            by_principal.setdefault(principal, []).append(member)
        for colliding in by_principal.values():
            if len(colliding) < 2:
                continue
            if len(set(colliding)) == 1:
                reason = "which list the same member more than once"
            elif colliding[0].startswith("domain:"):
                reason = "which name one domain, since domains compare without regard to case"
            else:
                reason = (
                    f"which the configured {claim!r} claim reads as one identity, since a "
                    "mailbox is the same in any case"
                )
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.duplicate_access_principal",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has the access members "
                        f"{colliding}, {reason}. The second grants nothing, and would keep "
                        "the card open after the first is removed. Write it once"
                    ),
                )
            )
    return findings


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


def _check_landing_group_types(web_terminals: dict[str, Any]) -> list[Finding]:
    """A ``type: apps`` landing section is the retired spelling of ``type: shared``.

    Read as ``type: shared`` for one release, with one warning naming the
    rewrite; ``osprey init --revalidate`` makes it.
    """
    from osprey.deployment.web_terminals.render import RETIRED_SHARED_GROUP_TYPE

    groups = as_dict(web_terminals.get("landing")).get("groups")
    if not isinstance(groups, list):
        return []
    return [
        Finding(
            severity="warn",
            code="web_terminals.retired_landing_group_type",
            message=(
                f"modules.web_terminals.landing.groups[{i}] spells `type: "
                f"{RETIRED_SHARED_GROUP_TYPE}`, which is read as `type: shared` (the shared "
                "logins section); `type: apps` is deprecated and will be refused in the next "
                "release. To rewrite it, run `osprey init --revalidate`"
            ),
        )
        for i, entry in enumerate(groups)
        if isinstance(entry, dict) and entry.get("type") == RETIRED_SHARED_GROUP_TYPE
    ]


def _check_landing_names(web_terminals: dict[str, Any]) -> list[Finding]:
    """A hidden roster needs a sign-in in front of it.

    ``names: hidden`` on a ``type: users`` section of ``landing.groups``
    replaces that section's name cards with one button that opens the card-less
    sign-in. That route exists only behind a login wall. Under ``token`` the
    name card is the way in, because each person enters through their own
    ``?token=`` URL and returns through the card; under ``none`` nobody signs
    in at all. Both are refused. The refusal is decided on the derived
    ``walled`` boolean, so any method that puts no login in front of the
    roster is refused; the message text is picked off ``auth_method``.

    The render reads the switch as a literal ``== "hidden"``, so any value other
    than ``shown`` or ``hidden`` (and ``names`` on a section that is not
    ``users``) is refused too: it would publish the names the operator meant to
    hide.
    """
    groups = as_dict(web_terminals.get("landing")).get("groups")
    if not isinstance(groups, list):
        return []

    findings: list[Finding] = []
    for i, entry in enumerate(groups):
        if not isinstance(entry, dict) or "names" not in entry:
            continue
        where = f"modules.web_terminals.landing.groups[{i}]"
        value = entry["names"]
        if entry.get("type") != "users":
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_landing_names",
                    message=(
                        f"{where} sets names, which only a `type: users` section reads; remove it."
                    ),
                )
            )
            continue
        if value not in ("shown", "hidden"):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_landing_names",
                    message=f"{where}.names is {value!r}; expected shown or hidden.",
                )
            )
            continue
        if value != "hidden":
            continue
        ctx = _auth_context(web_terminals)
        if ctx is None or ctx["walled"]:
            continue
        method = ctx["auth_method"]
        if method == "token":
            reason = (
                "under auth.method: token the name card is the way in: each person "
                "opens their terminal once from `osprey users login-url <name>` and "
                "comes back through their card."
            )
        else:
            reason = (
                f"under auth.method: {method} nobody signs in, so the button has "
                "nowhere to send them and the name card is the only way in."
            )
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.landing_names_hidden_without_sign_in",
                message=(
                    f"{where} sets names: hidden, but {reason} Set names: shown, or put "
                    "a login in front of the roster with auth.method: password or oidc."
                ),
            )
        )
    return findings


# --- authorization (role) checks ---------------------------------------------
#
# Like the auth seam checks above, these are scaffold-time feedback over a
# stanza whose authoritative gate lives on the render path: render.py's
# `_authorization_context` refuses an incoherent block outright. The two rules
# with no downstream mirror are the charset and the `$` scan — both describe
# strings that render perfectly well and only misbehave later, at the HTTP
# header or at compose interpolation.


def _authorization_stanza(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """The raw ``authorization`` stanza, read defensively as a dict."""
    return as_dict(web_terminals.get("authorization"))


def _authorization_ctx(web_terminals: dict[str, Any]) -> dict[str, Any] | None:
    """render.py's parsed view of the ``authorization`` stanza, or ``None``.

    The same wrapper as :func:`_auth_context`, for the same reason: every check
    keyed on the parsed tables is meaningless for a stanza that does not parse,
    so they skip themselves rather than reporting confused follow-on findings.
    :func:`_check_authorization` reports the parse failure itself.
    """
    try:
        return _authorization_context(web_terminals)
    except ValueError:
        return None


def _check_authorization(web_terminals: dict[str, Any]) -> list[Finding]:
    """The ``authorization`` stanza must be a mapping that parses.

    Two ERRORs:

    * A **wrong-typed** ``authorization``, ``roles`` or ``claims`` (a string, a
      list). The render reads a non-mapping as *no stanza at all*, so the
      deployment would come up with none of the privilege bindings the operator
      wrote — silently, since nothing downstream can tell a stanza that was
      never written from one that was written wrongly.
    * An **incoherent** stanza: a role naming no render, a claim map naming an
      undeclared role, half a ``claims`` block. The render raises on each of
      these; reporting it as a finding is what lets a scaffold command name
      every problem in one pass instead of dying on the first.
    """
    findings: list[Finding] = []
    raw = web_terminals.get("authorization")
    if raw is not None and not isinstance(raw, dict):
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_authorization_stanza",
                message=(
                    f"modules.web_terminals.authorization {raw!r} is not a mapping; it "
                    "must be a block with 'roles' and (optionally) 'claims' keys. A "
                    "non-mapping stanza is read as no authorization stanza at all, so "
                    "the deployment would render with none of these role bindings"
                ),
            )
        ]
    authorization = as_dict(raw)
    for key in ("roles", "claims"):
        value = authorization.get(key)
        if value is not None and not isinstance(value, dict):
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_authorization_stanza",
                    message=(
                        f"modules.web_terminals.authorization.{key} {value!r} is not a "
                        "mapping; it is read as absent, so the deployment would render "
                        "with none of the bindings written under it"
                    ),
                )
            )
    try:
        _authorization_context(web_terminals)
    except ValueError as exc:
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_authorization",
                message=str(exc),
            )
        )
    return findings


def _check_claims_without_oidc(web_terminals: dict[str, Any]) -> list[Finding]:
    """A ``claims`` block only does anything under single sign-on. WARN.

    The one inert source left in this stanza. Its sibling — a roster ``role:``
    under ``oidc`` — is *not* inert: the render resolves that entry's render
    from it, and the sidecar cross-checks the ID token's role against it, so it
    is emitted under both methods. ``claims`` has no such second job. It maps
    the values of an ID-token claim, and under ``password`` or ``none`` no ID
    token ever arrives, so nothing reads the map, the compose overlay renders
    none of it, and every role the operator wrote there is simply never granted.

    A WARN rather than an ERROR: the block is inert, not wrong, and it is what a
    facility staging a move to single sign-on writes before flipping the method.
    Saying so is the point — the alternative is a privilege table that looks
    live in the config file and is dark in the deployment.
    """
    claims = as_dict(_authorization_stanza(web_terminals).get("claims"))
    if not claims:
        return []
    context = _auth_context(web_terminals)
    if context is None or context["auth_method"] == "oidc":
        return []
    return [
        Finding(
            severity="warn",
            code="web_terminals.authorization_claims_without_oidc",
            message=(
                "modules.web_terminals.authorization.claims maps identity-provider "
                f"claim values onto roles, but auth.method is {context['auth_method']!r} "
                "so no ID token ever arrives to read them from; the block is not "
                "rendered and none of the roles it names is ever granted. Set "
                "auth.method: oidc, or bind these users with a roster 'role:' instead"
            ),
        )
    ]


def _check_role_charset(web_terminals: dict[str, Any]) -> list[Finding]:
    """A role name is held to the same charset as a username.

    It is carried on a roster entry's ``role:``, matched against an IdP claim's
    values, and forwarded to every terminal in the ``X-Osprey-Auth-Role``
    response header — which is latin-1 only, so a non-ASCII role name is not a
    cosmetic problem but a header nginx cannot emit. Reusing
    ``USERNAME_CHARSET_RE`` (see :func:`_check_username_charset`) keeps every
    identity-shaped name in this stanza held to one rule.

    A non-string key is reported by the same rule: the render drops it as
    unreferenceable, so this is the only place the operator is told why the role
    they wrote never applies.
    """
    findings: list[Finding] = []
    for role_name in as_dict(_authorization_stanza(web_terminals).get("roles")):
        if isinstance(role_name, str) and USERNAME_CHARSET_RE.fullmatch(role_name):
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.invalid_role_charset",
                message=(
                    f"modules.web_terminals.authorization.roles key {role_name!r} does "
                    f"not match {USERNAME_CHARSET_RE.pattern!r} (a role name is named by "
                    "a roster entry's 'role', matched against an IdP claim's values, and "
                    "forwarded in the latin-1-only X-Osprey-Auth-Role header)"
                ),
            )
        )
    return findings


def _authorization_strings(authorization: dict[str, Any]) -> list[tuple[str, str]]:
    """Every string the ``authorization`` stanza contributes, with its config path.

    One walk rather than five ad-hoc reads, so a value added to the stanza later
    is covered by the ``$`` scan the moment it is listed here.
    """
    strings: list[tuple[str, str]] = []
    for role_name, entry in as_dict(authorization.get("roles")).items():
        if not isinstance(role_name, str):
            continue
        strings.append(("authorization.roles", role_name))
        tier = as_dict(entry).get("tier")
        if isinstance(tier, str):
            strings.append((f"authorization.roles.{role_name}.tier", tier))
    claims = as_dict(authorization.get("claims"))
    claim = claims.get("claim")
    if isinstance(claim, str):
        strings.append(("authorization.claims.claim", claim))
    for claim_value, role_name in as_dict(claims.get("map")).items():
        if isinstance(claim_value, str):
            strings.append(("authorization.claims.map", claim_value))
        if isinstance(claim_value, str) and isinstance(role_name, str):
            strings.append((f"authorization.claims.map.{claim_value}", role_name))
    return strings


def _check_authorization_unsafe_values(web_terminals: dict[str, Any]) -> list[Finding]:
    """No string in the ``authorization`` stanza may contain ``$``.

    The same hazard as a ``$``-bearing ``oidc_subject`` (see
    :func:`_check_auth_oidc`), one layer up: these strings travel through the
    compose *document* rather than an env_file, so the deploy-time ``$`` scan
    over ``.env*`` never sees them and compose rewrites any ``$`` sequence on
    the way through. A rewritten role name or claim value is compared against
    the un-rewritten one the IdP and the roster carry, so the role is silently
    never granted — a privilege that quietly fails to apply, which is exactly
    the failure a static binding exists to prevent. Role names and claim values
    are near-universally plain identifiers, so a ``$`` is far more likely a typo
    than a real value; refusing it is the honest failure.
    """
    return [
        Finding(
            severity="error",
            code="web_terminals.authorization_unsafe_value",
            message=(
                f"modules.web_terminals.{path} {value!r} contains '$'. Roles and claim "
                "values are rendered into the compose document, where '$' sequences are "
                "interpolated — the sidecar would compare against a rewritten string, so "
                "this binding would silently never apply. Rename it, or map a claim "
                "value that carries no '$'"
            ),
        )
        for path, value in _authorization_strings(_authorization_stanza(web_terminals))
        if "$" in value
    ]


def _check_user_role(web_terminals: dict[str, Any], users: list[Any]) -> list[Finding]:
    """A roster entry's ``role`` must be a non-empty string naming a declared role.

    Three ERRORs, each of which otherwise degrades to a silent outcome — the
    entry renders with a privilege set nobody chose for it:

    * A **wrong-typed** ``role`` (an int, a YAML boolean, an empty string).
      :func:`~osprey.deployment.web_terminals.roster.normalize_users` drops it
      defensively, so nothing downstream can see that a role was meant at all.
    * A ``role`` **carried alongside a ``tier:`` pin**. Both bind the same
      slot — the role through
      ``modules.web_terminals.authorization.roles.<role>.render``, the pin
      directly — so which one governs is unwritten, and a later edit to the
      role's render would silently not reach this entry. Reported even when the
      two agree today, and reported *instead of* the checks below for that entry:
      the ambiguity is the finding.
    * A ``role`` **naming no declared role**. The render does not yet refuse
      this one: at the parse stage the roster's roles are unresolved, and
      nothing has looked one up. It is not undetectable, though —
      :func:`~osprey.deployment.web_terminals.roster.normalize_users` carries
      every non-empty string ``role`` through, so a resolver can tell "named a
      role that does not exist" apart from "named no role at all". The shared
      render helper (Task 4.2) is therefore expected to RAISE on a carried
      role absent from ``authorization_roles`` rather than fall back to the
      default tier, so that ``--no-lint`` cannot silently change a binding;
      this finding is the earlier, friendlier half of that pair.

    Skipped entirely when the stanza does not parse — :func:`_check_authorization`
    reports that, and every role would look undeclared against a table that was
    never built.
    """
    context = _authorization_ctx(web_terminals)
    if context is None:
        return []
    roles = context["authorization_roles"]
    findings: list[Finding] = []
    for user in users:
        if not isinstance(user, dict) or "role" not in user:
            continue
        role = user.get("role")
        name = _roster_entry_name(user)
        render = user.get("tier")
        if isinstance(role, str) and role and isinstance(render, str) and render:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.conflicting_user_tier_and_role",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has both tier "
                        f"{render!r} and role {role!r}; a role binds the tier "
                        f"(modules.web_terminals.authorization.roles.{role}.render) and a "
                        "direct tier pin binds the same slot, so which one governs is "
                        "unwritten and a later edit to the role's tier would silently "
                        "not reach this entry. Keep one: drop the tier to let the role "
                        "bind it, or drop 'role' to pin it directly"
                    ),
                )
            )
        elif not isinstance(role, str) or not role:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.invalid_user_role",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has role {role!r}, "
                        "which is not a non-empty string; it is dropped when the roster "
                        "is normalized, so the entry would silently render with the "
                        "deployment's default tier"
                    ),
                )
            )
        elif role not in roles:
            declared = ", ".join(repr(declared_role) for declared_role in roles) or "none"
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.unknown_role_reference",
                    message=(
                        f"modules.web_terminals.users entry {name!r} has role {role!r}, "
                        "which is not declared under "
                        f"modules.web_terminals.authorization.roles (declared: "
                        f"{declared}); the entry would silently render with the "
                        "deployment's default tier"
                    ),
                )
            )
    return findings


def _check_reserved_audit_identities(
    users: list[Any], app_identities: Mapping[str, str] | None = None
) -> list[Finding]:
    """A roster name may not be a service's own identity.

    Every container writes under ``var/spool/<identity>/`` and
    ``var/state/<identity>/`` and binds those subdirectories read-write, so a
    user named ``sidecar`` (the auth sidecar's fixed identity) or
    ``dispatch-worker-<n>`` (a worker's) would hold a read-write mount onto the
    spool and state of the component that records them. ``operator`` is refused
    for the same reason in the agent record's logins.

    render.py refuses this outright
    (:func:`~osprey.deployment.web_terminals.render._check_roster_audit_identities`,
    unconditional on ``auth.method`` because the name is a directory in every
    posture). This is the same rejection at scaffold time, where the render's
    raise is still a deploy attempt away and a rename is cheap — and the pattern
    is imported from there, so the two cannot drift.

    The neighbouring half of the same concern — a roster name that is not a
    usable path segment at all, which still names a spool bind source under
    ``auth.method: none`` — is already reported unconditionally by
    :func:`_check_username_charset`.

    A deployed application's audit identity (*app_identities*, identity to
    application) is refused the same way: its containers write facts as it.
    """
    app_identities = app_identities or {}
    findings: list[Finding] = []
    for user in users:
        name = _roster_entry_name(user)
        if name is not None and name in app_identities:
            findings.append(
                Finding(
                    severity="error",
                    code="web_terminals.reserved_audit_identity",
                    message=(
                        f"modules.web_terminals.users entry {name!r} takes the audit "
                        f"identity of application {app_identities[name]!r}. Each user's "
                        "container binds var/spool/<user>/ and var/state/<user>/ "
                        "read-write, so this user could write facts as that application "
                        "and read its state. Rename the roster entry"
                    ),
                )
            )
            continue
        if name is None or not _RESERVED_AUDIT_IDENTITY_RE.fullmatch(name):
            continue
        findings.append(
            Finding(
                severity="error",
                code="web_terminals.reserved_audit_identity",
                message=(
                    f"modules.web_terminals.users entry {name!r} collides with a "
                    "service's own identity (the auth sidecar writes "
                    f"var/spool/{AUTH_SIDECAR_AUDIT_IDENTITY}/, dispatch worker <n> "
                    f"writes var/spool/{DISPATCH_WORKER_SERVICE_PREFIX}-<n>/, a "
                    "queueserver writes var/spool/[<lane>-]queueserver/, a bridge "
                    f"writes var/spool/{{{','.join(sorted(BRIDGE_IDENTITIES.values()))}}}/), "
                    f"the history page's {HISTORY_SERVICE!r}, or the agent record's "
                    f"{OPERATOR_IDENTITY!r} credential. Each "
                    "user's container binds var/spool/<user>/ and var/state/<user>/ "
                    "read-write, so this user could read and write the spool and state "
                    "of the component that records them. Rename the roster entry"
                ),
            )
        )
    return findings


# --- the history page --------------------------------------------------------
#
# The across-users history page is served on its own origin, a second server of
# this nginx, and scoped by each person's ``history`` value. Every rule below is
# about that page, so each is asked only where it is deployed
# (:func:`~osprey.agent_record.config.history_page_deployed`), except the value
# check, which a typo deserves whatever the sign-in.

_WEB = "modules.web_terminals"

#: What a history scope may be: this person's sessions, or everyone's.
HISTORY_VALUES = ("own", "all")

#: The code of the start-blocking carrier-suffix collision.
HISTORY_CARRIER_COLLISION = "web_terminals.history_carrier_collision"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _roles(web_terminals: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(web_terminals.get("authorization")).get("roles"))


def _check_history_values(web_terminals: Mapping[str, Any], users: list[Any]) -> list[Finding]:
    """``history`` on a roster entry or a role is ``own`` or ``all``."""
    stated = [
        (f"{_WEB}.users entry {user.get('name')!r}", user["history"])
        for user in users
        if isinstance(user, Mapping) and "history" in user
    ]
    stated += [
        (f"{_WEB}.authorization.roles.{role}", entry["history"])
        for role, entry in _roles(web_terminals).items()
        if isinstance(entry, Mapping) and "history" in entry
    ]
    return [
        Finding(
            severity="error",
            code="web_terminals.invalid_history",
            message=(
                f"{where} has history {value!r}; it is 'own' (this person's "
                "sessions, the default) or 'all' (everyone's)"
            ),
        )
        for where, value in stated
        if value not in HISTORY_VALUES
    ]


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


def _check_history_port(root: Mapping[str, Any]) -> list[Finding]:
    """``history_port``, when set, is a port the history page's origin can listen on."""
    try:
        history_origin_port(root)
    except ValueError as exc:
        return [
            Finding(severity="error", code="web_terminals.invalid_history_port", message=str(exc))
        ]
    return []


def _check_history_origin(web_terminals: Mapping[str, Any]) -> list[Finding]:
    """``history_origin`` mirrors ``external_origin``, on the same host.

    When a load balancer is what browsers reach, nothing here can derive the
    history page's origin either, so it has to be stated beside
    ``external_origin``. And the sign-in cookie crosses ports on one host only:
    a history page's origin on another host would never see it.
    """
    external = web_terminals.get("external_origin")
    stated = web_terminals.get("history_origin")
    has_external = isinstance(external, str) and bool(external.strip())
    if stated is None:
        if not has_external:
            return []
        return [
            Finding(
                severity="error",
                code="web_terminals.history_origin_required",
                message=HISTORY_ORIGIN_REQUIRED,
            )
        ]
    stated_host = _origin_host(stated)
    if stated_host is None:
        return [
            Finding(
                severity="error",
                code="web_terminals.invalid_history_origin",
                message=invalid_history_origin(stated),
            )
        ]
    external_host = _origin_host(external) if has_external else None
    if external_host is not None and external_host != stated_host:
        return [
            Finding(
                severity="error",
                code="web_terminals.history_origin_host",
                message=(
                    f"{_WEB}.history_origin is on host {stated_host!r} and "
                    f"{_WEB}.external_origin on {external_host!r}. The sign-in cookie "
                    "crosses ports on one host only, so the history page on another "
                    "host would never see a signed-in person. Serve both from one "
                    "host, on two ports"
                ),
            )
        ]
    return []


def history_carrier_collision_problems(config: Any) -> list[tuple[str, str]]:
    """Named readers that would share one set of the sidecar's carriers.

    The sidecar's reader carriers are keyed by the ``agent_record.readers``
    names: ``jean-luc`` and ``jean_luc`` both key ``..._JEAN_LUC``, so one
    reader's subject would be read as the other's.

    Returns:
        One ``(problem, remedy)`` pair per shared suffix, naming both readers.
    """
    names = [entry["name"] for entry in reader_entries(as_dict(config))]
    return [
        (
            f"agent_record.readers {' and '.join(repr(name) for name in names)} share "
            f"the environment-variable suffix {suffix} (a name is upper-cased and '-' "
            "becomes '_'), so the sign-on settings of one would be read as the other's",
            "Rename one of them",
        )
        for suffix, names in env_var_suffix_collisions(names).items()
    ]


def history_preflight_problems(config: Any) -> list[tuple[str, str]]:
    """What ``osprey up`` refuses a start over for the history page.

    Readers sharing a carrier suffix, when the page that reads those carriers
    is deployed: the running page cannot tell them apart. Pure: reads the config
    and nothing else.
    """
    if not history_page_deployed(as_dict(config)):
        return []
    return history_carrier_collision_problems(config)


def _check_unbound_history_roles(
    web_terminals: Mapping[str, Any], users: list[Any]
) -> list[Finding]:
    """``history: all`` on a role nobody can hold widens nobody's view."""
    # Strings only: a wrong-typed role or claim target is another check's
    # finding, and binds nothing here.
    bound = {
        role
        for user in users
        if isinstance(user, Mapping) and isinstance(role := user.get("role"), str)
    }
    claims = _mapping(_mapping(web_terminals.get("authorization")).get("claims"))
    claimed = {role for role in _mapping(claims.get("map")).values() if isinstance(role, str)}
    return [
        Finding(
            severity="warn",
            code="web_terminals.unbound_history_role",
            message=(
                f"{_WEB}.authorization.roles.{role} has history 'all', but no roster "
                "entry carries that role and no authorization.claims.map entry grants "
                "it, so it widens nobody's view. Bind the role, or remove the key"
            ),
        )
        for role, entry in _roles(web_terminals).items()
        if isinstance(entry, Mapping)
        and entry.get("history") == "all"
        and role not in bound
        and role not in claimed
    ]


def host_identity_advisories(config: Any, runtime_user: str | None) -> list[str]:
    """What a command about to start the deployment says about who is running it.

    Host launches send to the agent record and are recorded under the account
    that BUILT the deployment (``deployment.host_identity``), so a start from
    another account is said once. About the invoking account rather than the
    config, so it is asked where something is launched and nowhere else. Empty
    when no account name is known, when the config records no identity, and
    when the account is the one the build recorded.
    """
    if runtime_user is None:
        return []
    identity = host_identity(as_dict(config))
    if identity is None or identity == runtime_user:
        return []
    return [
        f"deployment.host_identity is {identity!r} and this command runs as "
        f"{runtime_user!r}. Sessions started from this account are recorded as "
        f"{identity!r}, the account that built the deployment. Run `osprey build` as "
        "the account that will launch them, or set deployment.host_identity"
    ]
