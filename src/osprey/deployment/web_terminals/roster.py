"""Roster and render identity resolution for multi-user web terminal deployments.

Turns the raw ``modules.web_terminals`` roster into explicit per-user identity:
normalized ``{"name", "index"}`` entries (:func:`normalize_users`) and fully
resolved image/project/container-dir identity per user
(:func:`resolve_cards`). Who may *open* each of those entries is a second
roster vocabulary that lives here too: ``users[].access`` is parsed once, by
:func:`resolve_access_principals`, into the set of principals it admits, and
every consumer asks that function (or :func:`entry_is_shared`, the predicate
over it) rather than reading the raw key. The normalized entry carries the
authored value verbatim so both stay answerable from it.
Every reader of the roster goes through :func:`roster_entries` (each card the
deployment runs: the people, then one derived entry per shared app) or, when the
question is about a person, :func:`roster_people` / :func:`authored_people`.
Callers that write the roster *back* to
``config.yml`` rather than render from it use :func:`freeze_user_indices`, which
keeps the authored keys the normalizer projects away.
Port arithmetic lives in :mod:`osprey.deployment.web_terminals.ports`, and the
username→env-var-suffix mapping in :mod:`osprey.services.auth_sidecar.roster_env`.

One rule about the roster is *not* about identity at all and still lives here:
which renders can edit the deployment they run in (:func:`tier_privileges`),
and who may be handed one — the ``default_tier`` rule
(:func:`privileged_default_tier_problem`) and the shared-card rule
(:func:`shared_card_privileged_problems`). Two guards ask those questions on
three different surfaces —
:func:`osprey.cli.profile_cmd._privileged_tier_problems` at ``osprey init``,
and the lint belt in :mod:`osprey.deployment.web_terminals.lint` at profile and
rendered altitude (the second of which ``osprey up`` gates on) — and this is the
neutral module all of them already import; see the section comment above those
functions.
"""

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.agent_runner.tool_names import SHELL_TOOLS
from osprey.bluesky_bridge_connection import (
    LANE_KEYS,
    LANE_ONE,
)
from osprey.cli.profile_conventions import SETUP_PATCH_TOOL
from osprey.deployment.control_identity import validate_identity
from osprey.deployment.graphdb_service import resolve_graphdb_service_config
from osprey.deployment.web_terminals.render_naming import RENDER_KIND_APP, RENDER_KIND_TIER
from osprey.profiles.web_panels import panel_spec_enabled
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.services.auth_sidecar.identity_headers import CASE_INSENSITIVE_CLAIMS
from osprey_connectors import yaml_loader
from osprey_connectors.connection import ENV_NAME_RE, read_ca_bundle, read_credential_env_names
from osprey_connectors.types import (
    WRITES_KEY,
    archiver_settings_key,
    resolve_archiver_settings,
    tier_allows,
)
from osprey_connectors.workspace import BUILD_DIR_NAME

# Matches ${VAR} and $VAR env references inside modules.web_terminals.image_tag.
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

# The only wired value for `modules.web_terminals.mcp.topology`. Every
# other value (including the recognized-but-rejected `shared_http`) is
# fail-closed at render time — see render.py's `_check_mcp_topology()`. Lives
# in this neutral module so lint.py and render.py can both import it without
# one depending on the other. This key is scoped to the shared
# framework-MCP tier only; it has no bearing on a facility's own
# `claude_code.servers` custom entries (those render through the unrelated
# per-project `.mcp.json` pipeline, untouched by this module).
SUPPORTED_MCP_TOPOLOGY = "per_container_stdio"

# Usernames become nginx `location` keys and URL path segments (`/<user>/...`), so
# they're held to a stricter charset than a bare "no reserved collision" check.
# Public and defined here because this module owns what a roster username *is*:
# lint's scaffold-time rule, render's fail-closed gate and `auth_credentials`'
# deploy-time gate all import it from here, so the three cannot drift apart.
#
# Apply it with `.fullmatch()`, never `.match()`: Python's `$` also matches
# *before* a trailing newline, so `.match()` accepts "alice\n" — a name that
# goes on to render into an nginx location key mid-directive.
USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Roster entry -> render
#
# One answer to "which render does this roster entry run as", shared by every
# raw-roster read that needs a resolved tier: `resolve_cards` (and its
# `_render_ref_by_name` re-read), `served_render_names` below, lint's two
# roster walks, per-render env provisioning and the profile card. They used to
# each spell `entry.get("render") or default_tier` inline, which was fine
# while a direct pin was the only way to name a render; a `role:` that binds
# one through `modules.web_terminals.authorization` is a second way, and six
# independent readings of it is six chances for one surface to grant a
# privilege set another never saw.
# ---------------------------------------------------------------------------


class UnresolvedRoleError(ValueError):
    """A roster entry's render binding does not resolve to exactly one render.

    Raised by :func:`effective_tier` for the two roster shapes that would
    otherwise hand the entry a privilege set nobody chose: a ``role:`` naming no
    declared role, and an entry carrying both ``role:`` and ``tier:``.

    A ``ValueError`` subclass, so the lenient paths that already catch
    ``ValueError`` around render resolution keep behaving as they do; named, so
    a caller that wants to tell an unresolvable binding apart from every other
    config refusal can.
    """


def resolve_authorization_roles(web_terminals: Any) -> dict[str, str]:
    """The parsed ``{role: render}`` table for a ``modules.web_terminals`` block.

    A thin accessor over the single ``authorization`` parser
    (:func:`osprey.deployment.web_terminals.render._authorization_context`), so
    every :func:`effective_tier` caller reads the same table rather than
    re-deriving one. The parse is deliberately NOT repeated here: a second
    reading of the stanza is exactly the drift a single parser exists to
    prevent. ``render`` imports this module at module scope, so the import runs
    inside the function.

    Raises:
        ValueError: propagated from the parser for an incoherent stanza (see
            :func:`~osprey.deployment.web_terminals.render._authorization_context`).
            Reporting surfaces — lint, the profile card — catch it and carry on
            with an empty table and ``strict=False``, because a stanza that does
            not parse is already its own reported finding
            (``web_terminals.invalid_authorization``) and a report that raised
            would replace every other finding with a traceback.
    """
    from osprey.deployment.web_terminals.render import _authorization_context

    roles: dict[str, str] = _authorization_context(as_dict(web_terminals))["authorization_roles"]
    return roles


def effective_tier(
    entry: Any,
    authorization: Any,
    default_tier: Any,
    *,
    strict: bool = True,
) -> str | None:
    """Which render a raw roster entry runs as, or ``None`` for no render at all.

    Resolution, in order:

    * The entry's ``role:``, through *authorization* — a role names the catalog
      render every entry carrying it runs as.
    * Else the entry's own tier pin, ``tier:``.
    * Else *default_tier*.
    * Else ``None`` — no tier: only a roster with no ``default_tier``,
      which strict resolution refuses.

    An entry carrying no ``role:`` therefore resolves exactly as it did before
    roles existed, which is what keeps every pre-roles deployment rendering
    byte-identically. (:func:`normalize_users` drops an empty-string ``role``
    for the same reason it drops an empty ``oidc_subject``, so "carries a role"
    and "named a role" are the same question by the time anything reads one.)

    Two shapes raise :class:`UnresolvedRoleError` instead of resolving, because
    each would otherwise bind a privilege set nobody wrote:

    * **A ``role`` absent from** *authorization*. Falling back to
      *default_tier* here would let ``--no-lint`` silently swap a binding: the
      operator wrote a role, the deploy ran a different privilege set, and
      nothing said so. Lint reports the friendlier half of this pair first
      (``web_terminals.unknown_role_reference``).
    * **Both ``role:`` and a tier pin (``tier:``) on one
      entry.** A role binds the render
      and a direct ``tier:`` pin is the pre-authorization way to bind the same
      slot; an entry naming both leaves which mechanism governs unwritten. It is
      refused even when the two agree today, because the ambiguity is about
      which one governs tomorrow: a later edit to the role's render would
      silently not reach this entry. Lint reports it as
      ``web_terminals.conflicting_user_tier_and_role``.

    Args:
        entry: One RAW roster entry (``modules.web_terminals.users[i]``).
            Anything that is not a mapping resolves to *default_tier*, the
            same defensive read the rest of this module uses.
        authorization: The ``{role: render}`` table — the
            ``authorization_roles`` key of the parsed context, via
            :func:`resolve_authorization_roles`. Empty for a deployment that
            declares no roles.
        default_tier: ``modules.web_terminals.default_tier``. A
            non-string or empty value reads as absent. Callers that want only
            the entry's *own* reference (the default is added separately, or
            applied downstream) pass ``None``.
        strict: ``True`` — the binding surfaces — raises on the two shapes
            above. ``False`` — the reporting surfaces (lint's roster walks, the
            profile card), each of which pairs the degrade with an ERROR finding
            — resolves them the way the entry would have resolved before roles
            existed, so a report shows every finding rather than dying on the
            first bad entry. It mirrors :func:`resolve_cards`' own ``strict``
            split.

    Raises:
        UnresolvedRoleError: See above; only when *strict*.
    """
    source = entry if isinstance(entry, Mapping) else {}
    roles = authorization if isinstance(authorization, Mapping) else {}

    pinned = source.get("tier")
    tier = pinned if isinstance(pinned, str) and pinned else None
    role = source.get("role")
    role = role if isinstance(role, str) and role else None
    fallback = tier or (default_tier if isinstance(default_tier, str) and default_tier else None)

    if role is None:
        return fallback

    name = source.get("name")
    if tier is not None:
        if strict:
            raise UnresolvedRoleError(
                f"modules.web_terminals.users entry {name!r} carries both tier "
                f"{tier!r} and role {role!r}. A role binds the tier "
                f"(modules.web_terminals.authorization.roles.{role}) and a direct "
                "'tier' pin binds the same slot, so an entry naming both leaves which "
                "one governs unwritten — a later edit to the role's tier would silently "
                "not reach this entry. Keep one: drop 'tier' to let the role bind it, "
                "or drop 'role' to pin it directly"
            )
        return fallback

    bound = roles.get(role)
    if isinstance(bound, str) and bound:
        return bound
    if strict:
        declared = ", ".join(repr(declared_role) for declared_role in roles) or "none"
        raise UnresolvedRoleError(
            f"modules.web_terminals.users entry {name!r} has role {role!r}, which is not "
            "declared under modules.web_terminals.authorization.roles (declared: "
            f"{declared}). The entry is deliberately NOT resolved to the deployment's "
            "default tier: substituting a privilege set nobody chose is the failure a "
            "static role binding exists to prevent, and a scaffold run with --no-lint "
            "would be the only warning"
        )
    return fallback


def sign_in_decides_tier(entry: Any, web_terminals: Any) -> bool:
    """Whether single sign-on, not a file on this host, decides *entry*'s tier.

    True for a card naming no ``role:`` in a deployment whose ``auth.method``
    is ``oidc`` with an ``authorization`` claim: that person is granted the role
    the group claim maps to when they sign in, so no build can say which render
    they will run — whatever ``tier:`` pin or ``default_tier`` the roster
    carries. A card with an explicit role is bound statically and answers
    False, as does every card of a deployment without that sign-in.

    The one statement of the predicate: the login→tier map, status's tier
    column and every other reader call it rather than restate it.

    Args:
        entry: One RAW roster entry (``modules.web_terminals.users[i]``); a bare
            string entry carries no role.
        web_terminals: The ``modules.web_terminals`` subtree.

    Returns:
        Whether the card's tier is decided at sign-in. Never raises: an
        ``authorization`` stanza that does not parse names no claim.
    """
    from osprey.deployment.web_terminals.render import _authorization_context

    subtree = as_dict(web_terminals)
    auth = subtree.get("auth")
    if not (isinstance(auth, Mapping) and auth.get("method") == "oidc"):
        return False
    try:
        claim = _authorization_context(subtree)["authorization_claim"]
    except ValueError:
        return False
    if not claim:
        return False
    raw = entry if isinstance(entry, Mapping) else {}
    return not raw.get("role")


#: The in-terminal event-dispatcher dashboard's panel id. A render that declares
#: this panel is the one thing that entitles its containers to
#: ``EVENT_DISPATCHER_TOKEN`` — the panel declaration IS the intent, so there is
#: no separate config key to set (and to forget to set) alongside it.
EVENTS_PANEL_ID = "events"

#: The bluesky-web sidecar's tab. A render that declares it proxies into the
#: sidecar with its own operator secret, so the sidecar is handed that user's
#: secret variable (the web overlay's ``bluesky-web`` fragment) — the panel
#: declaration IS the entitlement, exactly as for :data:`EVENTS_PANEL_ID`.
BLUESKY_PANEL_ID = "bluesky"


def config_declares_panel(config: Any, panel_id: str) -> bool:
    """True if ``config`` declares ``web.panels.<panel_id>`` and hasn't disabled it.

    The one definition of "this project shows that panel", shared by the
    render-set predicates (:func:`renders_needing_dispatcher_token` and its
    siblings) and the reach contract, so they cannot answer differently for the
    same config — and read through the same predicate the web terminal and the
    health probe read a block with (:func:`panel_spec_enabled`). ``enabled:
    false`` counts as *not* declared: a panel switched off is one whose
    credential is not needed, and the build switches off every block a render
    carries for a tab its profile does not select.
    """
    panel = as_dict(as_dict(as_dict(config).get("web")).get("panels")).get(panel_id)
    return panel_spec_enabled(panel)


def config_needs_dispatcher_token(config: Any) -> bool:
    """True if ``config`` declares the EVENTS panel and so needs the dispatcher bearer.

    A thin, named wrapper around :func:`config_declares_panel` for
    :data:`EVENTS_PANEL_ID` specifically: the panel declaration IS the
    entitlement (see that constant's docstring — there is no separate config
    key to set, or to forget to set, alongside it), so this is the one
    predicate every dispatcher-token call site should read rather than each
    re-supplying the panel id itself.
    """
    return config_declares_panel(config, EVENTS_PANEL_ID)


def config_needs_ariel_password(config: Any) -> bool:
    """True if ``config`` carries a non-empty ``ariel:`` section.

    The ARIEL counterpart of :func:`config_declares_panel`, and deliberately not
    expressed as one: a web terminal has TWO ARIEL consumers that authenticate
    against the same Postgres — the panel's own server (``web.panels.ariel``)
    and the ``ariel`` MCP server the agent calls, which is selected during the
    build and appears nowhere in ``web.panels``. Gating the credential on the
    panel would leave the agent's logbook tools broken in exactly the projects
    that switched the tab off.

    What both consumers do read is the ``ariel:`` section — it is the input to
    :func:`osprey.services.ariel_search.config.resolve_ariel_dsn` — so its
    presence is the entitlement. An absent or empty section configures no
    logbook, and so entitles nothing.
    """
    return bool(as_dict(as_dict(config).get("ariel")))


def config_needs_facility_bundle(config: Any) -> bool:
    """True if ``config`` names a ``facility_knowledge.bundle_path``.

    The entitlement to have the deployment's knowledge bundle bound into this
    project's container. Read from the same key the bundle's own reader uses
    (:func:`osprey.deployment.compose_generator.resolve_facility_bundle_dir`),
    for the reason :func:`config_needs_ariel_password` gives about the ``ariel:``
    section: the key IS the entitlement, because it is what tells the project
    where its facility knowledge lives. A project that names no bundle path has
    no directory to be handed and nothing inside it that would read one.

    Deliberately not gated on a panel or an MCP server. The bundle has two
    unrelated consumers inside the container — the OKF panel and the
    ``facility_knowledge`` MCP server the agent calls — so gating on either
    would leave the other reading a directory that was never mounted.
    """
    bundle_path = as_dict(as_dict(config).get("facility_knowledge")).get("bundle_path")
    return isinstance(bundle_path, str) and bool(bundle_path.strip())


def config_needs_ariel_mirror(config: Any) -> bool:
    """True if ``config`` runs an ARIEL qmd export that writes a mirror.

    The entitlement to have the deployment's ARIEL markdown mirror bound into
    this project's container, and the mirror counterpart of
    :func:`config_needs_facility_bundle`. Read through the same reader the
    sidecar's corpus list and the deploy's provisioning use
    (:func:`osprey.deployment.compose_generator.configured_ariel_mirror_path`),
    because the key IS the entitlement: an enabled export with a path is a
    writer, and a writer inside a container that mounts nothing fills the
    container's writable layer — a tree the sidecar never indexes and the next
    recreate discards. A disabled export, or one naming no path, writes
    nothing and so entitles nothing.
    """
    from osprey.deployment.compose_generator import configured_ariel_mirror_path

    return configured_ariel_mirror_path(as_dict(config)) is not None


def bluesky_server_enabled(config: Any) -> bool:
    """True if ``config`` leaves the ``bluesky`` MCP server running.

    The one answer to "does this project run the bluesky server", shared by the
    launch-token entitlement here and by the reach contract's projections and
    consumers in :mod:`osprey.deployment.reach`, so the two halves of the
    bluesky contract cannot disagree about the same config.

    ``claude_code.servers.bluesky.enabled`` is an *override*, read exactly the
    way :func:`osprey.registry.mcp.resolve_servers` reads it: a literal ``False``
    switches the server off, a literal ``True`` switches it on, and absence
    leaves the registry's own default. That default is taken from the registry
    rather than restated here — the bluesky server is opt-in
    (``default_enabled=False``), and a hand-copied default is one that can drift
    away from the definition the render actually obeys.
    """
    servers = as_dict(as_dict(as_dict(config).get("claude_code")).get("servers"))
    value = as_dict(servers.get("bluesky")).get("enabled")
    if value is False:
        return False
    if value is True:
        return True
    return FRAMEWORK_SERVERS["bluesky"].default_enabled


def phoebus_server_runs(config: Any) -> bool:
    """True if ``config`` starts a Phoebus MCP server.

    The one answer to "does this project start a Phoebus MCP server", read
    exactly the way :func:`osprey.registry.mcp.resolve_servers` reads
    ``claude_code.servers``. The ``phoebus`` entry's ``enabled`` is an
    override: a literal ``False`` switches the framework server off, a literal
    ``True`` switches it on, and absence leaves the registry's own default,
    taken from the registry rather than restated here for the reason
    :func:`bluesky_server_enabled` gives.

    Any other entry with ``extends: phoebus`` counts too, unless it says
    ``enabled: false``: a declared clone is enabled, and every clone addresses a
    bridge that all terminals share, the same as the framework server does. A
    framework server name never counts as a clone, because the registry ignores
    ``extends`` on one. A malformed entry is not a Phoebus server.
    """
    servers = as_dict(as_dict(as_dict(config).get("claude_code")).get("servers"))
    for name, spec in servers.items():
        if name in FRAMEWORK_SERVERS:
            continue
        entry = as_dict(spec)
        if entry.get("extends") == "phoebus" and entry.get("enabled") is not False:
            return True
    value = as_dict(servers.get("phoebus")).get("enabled")
    if value is False:
        return False
    if value is True:
        return True
    return FRAMEWORK_SERVERS["phoebus"].default_enabled


def config_needs_phoebus_handles(config: Any) -> bool:
    """True if ``config`` starts a Phoebus server without ``phoebus.require_handle: false``.

    The entitlement for the ``PHOEBUS_REQUIRE_HANDLE`` stamp on a multi-user
    terminal. An explicit ``false`` is honoured by emitting nothing: the server
    reads the same ``false`` from its own config, and a stamped ``1`` would
    override it, because the environment variable wins in the Phoebus tools'
    resolution. An explicit ``true`` is stamped anyway, which is harmless and
    keeps the predicate to one rule.
    """
    return (
        phoebus_server_runs(config)
        and as_dict(as_dict(config).get("phoebus")).get("require_handle") is not False
    )


def _config_renders_lane(config: Any, lane: str) -> bool:
    """Whether this project's rendered config carries ``lane`` at all.

    A second lane exists as its own ``services.<lane>`` block, which is the
    deployment's own statement that it renders that lane; a render that carries
    no such block has no bridge there to arm, so the lane entitles nothing. Lane
    1 needs no block — every deployment has had it since the bridge shipped —
    and its presence is decided by :func:`bluesky_server_enabled` instead.
    """
    if lane == LANE_ONE:
        return True
    return isinstance(as_dict(as_dict(config).get("services")).get(lane), dict)


def config_needs_launch_token_for(config: Any, lane: str) -> bool:
    """True if ``config`` may arm a queue start on ``lane`` specifically.

    The per-lane form of :func:`config_needs_launch_token`, and the one every
    lane's own token is granted on. The lane matters only for whether this
    render carries it: whether writes may happen at all is the render's login
    tier, which is target-blind — a tier that allows writes allows them for
    every machine, and the person's switch picks which one at the moment of
    each write.

    Three conditions, each reading absence as "not granted":

    * **This render carries the lane** (:func:`_config_renders_lane`). An
      undeclared second lane has no bridge to arm, so it entitles nothing.
    * **The render's tier allows writes**
      (:func:`~osprey_connectors.types.tier_allows` of its ``control_system``
      section) — ``control_system.writes`` is ``asked`` or ``allowed``, the
      literal strings and nothing else. ``never``, silence and every other value
      arm nothing, which is the read-only tier's whole boundary.
    * **The bluesky server is left running** (:func:`bluesky_server_enabled`).
      A token for a server that never starts arms nothing while still handing
      the agent a live credential.

    Holding the token is not a write: each write a queued plan makes is still
    decided by the connector's verdict for the owner the plan names, with that
    owner's switch read at the moment of the write.

    :param config: One project's rendered config document.
    :param lane: A lane's ``services.<lane>`` key — see
        :data:`~osprey.bluesky_bridge_connection.LANE_KEYS`.
    """
    root = as_dict(config)
    if not bluesky_server_enabled(root):
        return False
    if not _config_renders_lane(root, lane):
        return False
    return tier_allows(root.get("control_system"))


def launch_token_writes_key(config: Any, lane: str) -> str:
    """The config key an operator sets to change ``lane``'s token entitlement.

    :data:`~osprey_connectors.types.WRITES_KEY` for every lane: the login tier
    is the one posture a render has, and it does not vary by the machine a lane
    drives. Kept per lane so a refusal can still say which lane's token it is
    about; ``config`` and ``lane`` do not change the answer.
    """
    del config, lane
    return WRITES_KEY


def config_needs_launch_token(config: Any) -> bool:
    """True if ``config`` both allows writes and runs the bluesky MCP server.

    ``BLUESKY_LAUNCH_TOKEN`` arms a queue start — it is what turns an agent's
    ``queue_start`` call into physical hardware motion — so it is granted only
    where both halves of that capability are actually configured. The consumer
    is the ``bluesky`` MCP server, not the BLUESKY panel, which is the same
    reasoning :func:`config_needs_ariel_password` applies to its own credential:
    a credential is gated on what its consumer reads, never on which panel tab
    happens to be visible. A project that switched the panel off still runs the
    agent's queue tools, and a project that shows the panel but runs no bluesky
    server has nothing to arm.

    This is :func:`config_needs_launch_token_for` on
    :data:`~osprey.bluesky_bridge_connection.LANE_ONE` — the lane every
    deployment has — so the rule lives in one place and a caller that knows
    nothing about lanes still asks about the lane it has always meant: lane 1 is
    carried by every render, so the answer is the render's tier and its bluesky
    server.
    """
    return config_needs_launch_token_for(config, LANE_ONE)


def config_needs_graphdb_password(config: Any) -> bool:
    """True if ``config`` configures a graph store and runs the ``graph`` MCP server.

    ``GRAPHDB_PASSWORD`` is the store's *single* credential: Neo4j has one
    account, ``neo4j``, and it is write-capable. There is no read-only account
    to hand out instead, so a render that may query the graph at all holds the
    password that could also rewrite it — and, because the credential arrives as
    an environment variable in the container the agent runs in, the agent can
    read it. Granting it is therefore a deliberate decision, not a default.

    The read-only tier receives it too, and that is intended rather than an
    oversight. Read-only-ness for the graph is enforced by the ``graph`` MCP
    server, which runs every query through ``Session.execute_read`` — the
    transaction never opens in write mode, whatever the credential would permit.
    That is the same posture ``ARIEL_DB_PASSWORD`` already has: one Postgres
    password shared by every ARIEL consumer, with the read-only guarantee living
    in what the consumer does rather than in which password it holds. The
    contrast is :func:`config_needs_launch_token`, which *is* gated on write
    posture — on the render's login tier — because there the
    credential itself is the capability: nothing downstream of holding it stops
    a queue start.

    The blast radius bounds the decision: the store holds a disposable mirror of
    a Turtle corpus, re-seeded from it by ``osprey build && osprey up``, so
    the worst case is corpus integrity rather than facility data or hardware.
    This predicate is consequently **never** gated on write posture — doing so
    would leave the read-only operator terminal, the very tier the graph search
    is meant to serve, unable to reach the store at all.

    Two conditions, read the way each key's own default reads:

    * ``services.graphdb`` must resolve to a block (see
      :func:`osprey.deployment.graphdb_service.resolve_graphdb_service_config`).
      That block IS the entitlement, for the reason
      :func:`config_needs_ariel_password` gives about the ``ariel:`` section: it
      is what
      :func:`osprey.deployment.graphdb_service.resolve_graphdb_connection` reads
      to decide what to dial, on the local-port path and the external-``uri:``
      path alike, and both then read this same variable for the password. A
      fully-defaulted ``graphdb: {}`` therefore entitles, while a bare
      ``graphdb:`` (which YAML parses as ``None``) configures no store and does
      not. A malformed block — a port that is not a port, a heap size the JVM
      would reject — entitles nothing either: it configures no store this
      deployment could reach, and refusing here would turn a typo in one
      render's ``config.yml`` into a failed render of the whole roster.
    * ``claude_code.servers.graph.enabled`` must merely be **not** ``False``,
      because that key is an *override* over the server's built-in default (see
      :func:`osprey.registry.mcp.resolve_servers`, which likewise disables only
      on a literal ``enabled: false``). An absent key leaves the server enabled,
      so reading absence as "disabled" would deny the password to every
      correctly-configured project that simply never wrote the override.
    """
    root = as_dict(config)
    try:
        if resolve_graphdb_service_config(root) is None:
            return False
    except ValueError:
        return False
    servers = as_dict(as_dict(root.get("claude_code")).get("servers"))
    return as_dict(servers.get("graph")).get("enabled", True) is not False


def config_archiver_credential_envs(config: Any) -> tuple[str, ...]:
    """The variables ``config``'s archiver connector authenticates with, in order.

    The archiver connector reads its secret from the environment variable its
    settings block names under ``auth:`` (``auth.token_env``, or
    ``auth.password_env`` beside ``auth.username``) and raises on every read
    when that variable is unset. For a store the project deploys itself,
    ``osprey up`` mints the value into the deploy ``.env`` under that name; for
    a facility-run store the operator puts it there. Either way the web
    terminal's agent can only authenticate if its container is handed *those*
    variables, which is why this returns the configured NAMES rather than a
    boolean. The names come from the connection-settings reader, so the grant
    matches what the connector will look up; an empty tuple is no grant.

    Only the SELECTED connector's block counts. The shipped ``config.yml``
    carries a filled-in ``mongodb_archiver:`` block under ``type:
    mock_archiver``; a block the selected type never reads entitles nothing,
    for the reason :func:`config_needs_ariel_password` gives — a credential is
    gated on what its consumer actually reads.

    Raises:
        ValueError: when a configured name is not a plain identifier. The
            name is emitted into a compose ``environment:`` line verbatim, so
            a value compose would mangle (a space, an ``=``, a ``${``, or
            surrounding whitespace) is refused at the deploy gate rather than
            rendered broken; the connector reads the name as written, so a
            padded name is refused here rather than granted under a spelling
            the connector never looks up. A refusal of the block itself by the
            connection-settings reader propagates.
    """
    archiver = as_dict(as_dict(config).get("archiver"))
    connector = archiver.get("type")
    if not isinstance(connector, str) or not connector:
        return ()
    where = archiver_settings_key(archiver)
    names: list[str] = []
    for key, raw in read_credential_env_names(resolve_archiver_settings(archiver), where=where):
        if not isinstance(raw, str) or not raw.strip():
            continue
        if not ENV_NAME_RE.match(raw):
            raise ValueError(
                f"{where}.{key} must name an environment variable (letters, digits and "
                f"underscores, not starting with a digit), got {raw!r}"
            )
        if raw not in names:
            names.append(raw)
    return tuple(names)


def ca_bundle_mounts(block: Any, *, where: str) -> tuple[str, ...]:
    """The host CA file ``block`` names under ``tls.ca_bundle``, when it is on this host.

    The file is mounted read-only at this same path into each container that
    reads the block, so the key names one file on the host and inside a
    container. Two cases mount nothing: a ``tls:`` the connection-settings
    reader refuses, and a named file that is not on this host. In both the
    connector inside the container refuses at connect exactly as it does on the
    host, naming the key, so the build adds no second report of the same fact.

    :param block: The connection block, as the consumer reads it.
    :param where: The block's dotted key.
    :return: ``(path,)`` for a named file on this host, else ``()``.
    """
    try:
        path = read_ca_bundle(block, where=where)
    except ValueError:
        return ()
    if path is None or not path.exists():
        return ()
    return (path.as_posix(),)


def config_archiver_ca_bundles(config: Any) -> tuple[str, ...]:
    """The CA file ``config``'s SELECTED archiver block names, when it is on this host.

    Only the selected connector's block counts, for the reason
    :func:`config_archiver_credential_envs` gives. See :func:`ca_bundle_mounts`
    for what is mounted and why nothing else is.
    """
    archiver = as_dict(as_dict(config).get("archiver"))
    connector = archiver.get("type")
    if not isinstance(connector, str) or not connector:
        return ()
    return ca_bundle_mounts(
        resolve_archiver_settings(archiver), where=archiver_settings_key(archiver)
    )


def dispatch_tier_name(config: Mapping[str, Any]) -> str | None:
    """The tier the dispatch workers run at, or ``None`` for the deployment's own posture.

    Read from ``services.dispatch_worker.tier``, which the build writes from
    the profile's ``dispatch.tier`` by name and never as an image string. The
    profile lint's view carries the same key
    (:func:`osprey.cli.build_profile_deploy.build_lint_view`), so one reader
    answers at both altitudes.
    """
    services = config.get("services") if isinstance(config, Mapping) else None
    worker = services.get("dispatch_worker") if isinstance(services, Mapping) else None
    tier = worker.get("tier") if isinstance(worker, Mapping) else None
    return tier if isinstance(tier, str) and tier else None


def served_render_names(
    web_terminals: Any, dispatch_tier: Any, *, strict: bool = False
) -> dict[str, str]:
    """Every render this deployment serves, by name, with its kind.

    The served set is every card's render, ``default_tier``, every role's
    tier and the dispatch tier. Each of the four is a way a container comes
    to run a render: a card is opened by its holder, ``default_tier`` is what
    the next person added inherits, a role may be handed to anyone single
    sign-on admits, and the dispatch workers run their tier headless. A render
    this set names is built, verified, granted and linted; one it does not
    name deploys nothing.

    Names come in that order — the cards in roster order (people, then apps),
    then ``default_tier``, then the roles, then the dispatch tier — each name
    once, at its first appearance. A reader that reports sorts.

    The kind is ``"app"`` for a name only ``shared:`` names and ``"tier"``
    for every other: a name that both a tier source and ``shared:`` name is a
    tier here, and the refusal that a card renders from one file reports the
    collision (:func:`resolve_cards`).

    Args:
        web_terminals: The ``modules.web_terminals`` subtree, as authored or
            as rendered; the renders projection is not read.
        dispatch_tier: The dispatch workers' tier name, or ``None``. Anything
            that is not a non-empty string names nothing.
        strict: ``True`` — the binding surfaces — raises on an
            ``authorization`` stanza that does not parse and on a roster entry
            whose binding does not resolve (:func:`effective_tier`). ``False``
            — the reporting surfaces — reads an unparseable stanza as no roles
            and an unresolvable entry the way :func:`effective_tier` degrades
            it, because each of those is already its own finding and a report
            that raised would hide every other one.

    Returns:
        ``{render name: RENDER_KIND_TIER | RENDER_KIND_APP}``.

    Raises:
        ValueError: Only when *strict*; see above.
    """
    subtree = as_dict(web_terminals)
    if strict:
        roles = resolve_authorization_roles(subtree)
    else:
        try:
            roles = resolve_authorization_roles(subtree)
        except ValueError:
            # `web_terminals.invalid_authorization` reports the stanza; the
            # query carries on against an empty table.
            roles = {}
    default_tier = subtree.get("default_tier")

    served: dict[str, str] = {}

    def serve(name: Any, kind: str) -> None:
        if not isinstance(name, str) or not name:
            return
        if served.get(name) != RENDER_KIND_TIER:
            served[name] = kind

    for user in authored_people(subtree):
        serve(effective_tier(user, roles, default_tier, strict=strict), RENDER_KIND_TIER)
    for entry in roster_entries(subtree, strict=False):
        if is_app_entry(entry):
            serve(entry.get("name"), RENDER_KIND_APP)
    serve(default_tier, RENDER_KIND_TIER)
    for tier in roles.values():
        serve(tier, RENDER_KIND_TIER)
    serve(dispatch_tier, RENDER_KIND_TIER)
    return served


@dataclass(frozen=True)
class ServedRender:
    """One served render, joined to its entry in the renders projection.

    The projection (``modules.web_terminals.renders``) is the build's account
    of where each render lives; this carries the fields every reader of a
    served render asks for, and the entry itself for the rest.
    """

    #: The render's name: a tier or app file's stem.
    name: str
    #: ``RENDER_KIND_TIER`` or ``RENDER_KIND_APP`` — the projection entry's
    #: own ``kind`` when it states one, else the kind the served set gave it.
    kind: str
    #: The image tag prefix the projection names, or ``None``.
    project: str | None
    #: The rendered project's directory, spelled against the repo root, or
    #: ``None`` when the entry names none.
    project_path: str | None
    #: The projection entry as written.
    entry: Mapping[str, Any]


def served_renders(config: Any, *, strict: bool = False) -> dict[str, ServedRender | None]:
    """The served set of a rendered config, each name joined to the renders projection.

    :func:`served_render_names` over ``modules.web_terminals`` and the
    dispatch tier (:func:`dispatch_tier_name`), each name looked up in
    ``modules.web_terminals.renders``. **A name the projection does not list
    maps to ``None``**: every reader skips it, because a credential is never
    granted on a guess and nothing can be built from no entry; lint reports it
    (``web_terminals.unknown_tier_reference``).

    Args:
        config: The parsed rendered (or lint-view) config.
        strict: Passed to :func:`served_render_names`.

    Returns:
        ``{render name: ServedRender | None}``, in :func:`served_render_names`'
        order.
    """
    root = as_dict(config)
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    projection = as_dict(web_terminals.get("renders"))
    served: dict[str, ServedRender | None] = {}
    for name, kind in served_render_names(
        web_terminals, dispatch_tier_name(root), strict=strict
    ).items():
        entry = projection.get(name)
        if not isinstance(entry, Mapping):
            served[name] = None
            continue
        entry_kind = entry.get("kind")
        project = entry.get("project")
        project_path = entry.get("project_path")
        served[name] = ServedRender(
            name=name,
            kind=entry_kind if entry_kind in (RENDER_KIND_TIER, RENDER_KIND_APP) else kind,
            project=project if isinstance(project, str) and project else None,
            project_path=project_path if isinstance(project_path, str) and project_path else None,
            entry=entry,
        )
    return served


def _renders_whose_config(
    config: Any, project_root: Any, predicate, layer_root: Any = None
) -> set[str]:
    """Referenced renders whose rendered ``config.yml`` satisfies ``predicate``.

    The one walk behind every per-render credential grant, so two credentials
    cannot disagree about which renders a roster deploys. Reads each render's
    ``config.yml`` off disk — which is why the render takes the result as a
    parameter rather than calling this itself (see
    :func:`osprey.deployment.web_terminals.render.render_web_terminals`'s
    determinism note). Resolution mirrors
    :func:`osprey.deployment.web_terminals.env_production._claude_code_auth_secret_vars`:
    a render whose ``project_path`` is unset, unrendered, or unreadable
    contributes nothing, because a credential is never granted on a guess.
    ``layer_root`` is :func:`_render_configs`'s.
    """
    return {
        render_name
        for render_name, render_config in _render_configs(config, project_root, layer_root)
        if predicate(render_config)
    }


def _render_dir(project_root: Any, project_path: str, layer_root: Any) -> Path:
    """Where a catalog entry's rendered project is read from.

    ``project_path`` is spelled against the repo root and, for every render
    ``osprey init`` catalogs, names the build zone (``build/<repo>-<render>``).
    A render in flight has not published that zone yet: ``osprey build`` writes
    every render into a staging tree that becomes ``build/`` only at the final
    swap, so a grant computed during the build must read the renders from
    THAT tree, or it reads the previous build's — or nothing. ``layer_root``
    is that tree; a ``project_path`` under the build zone is re-rooted beneath
    it, one outside the zone (an absolute pin, say) is read where it points.
    """
    rendered = Path(project_root, project_path)
    if layer_root is None:
        return rendered
    try:
        within_build_zone = rendered.relative_to(Path(project_root, BUILD_DIR_NAME))
    except ValueError:
        return rendered
    return Path(layer_root, within_build_zone)


def rendered_config_file(project_root: Any, project_path: str, layer_root: Any = None) -> Path:
    """The ``config.yml`` a catalog entry's rendered project is read from.

    The one join behind every reader of a render's render — the credential
    grants, the deploy-time lint belt, and the build-time belt for a render
    the build does not render itself (an entry with no ``build_profile``, whose
    ``project_path`` points at a project rendered somewhere else). Spelling the
    join once is what keeps the two altitudes reading the same file.

    :param project_root: The repo root ``project_path`` is spelled against.
    :param project_path: The catalog entry's ``project_path``, as written.
    :param layer_root: :func:`_render_dir`'s staging tree, or ``None``
        for the published build zone.
    :return: The path of that render's ``config.yml``; not checked to exist.
    """
    return _render_dir(project_root, project_path, layer_root) / "config.yml"


def _render_configs(
    config: Any, project_root: Any, layer_root: Any = None
) -> Iterable[tuple[str, Any]]:
    """Yield ``(render_name, parsed config.yml)`` for every readable served render.

    The renders walked are :func:`served_renders`' — every card's render,
    ``default_tier``, every role's tier and the dispatch tier — so a grant
    reaches every render a container can run, an app's included.

    The one disk walk behind :func:`_renders_whose_config` and
    :func:`renders_needing_archiver_credentials`; see the former for why a
    render that cannot be read is skipped rather than guessed at.

    :param layer_root: The directory standing in for ``<project_root>/build``
        — the staging tree a build in flight renders its renders into. ``None``
        reads the published build zone (see :func:`_render_dir`).
    """
    import yaml

    served = served_renders(config)

    for render_name in sorted(served):
        render = served[render_name]
        project_path = render.project_path if render is not None else None
        if project_path is None:
            continue
        config_yml = rendered_config_file(project_root, project_path, layer_root)
        if not config_yml.is_file():
            continue
        try:
            with config_yml.open("r", encoding="utf-8") as fh:
                render_config = yaml_loader.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        yield render_name, render_config


def rendered_configs(config: Any, project_root: Any) -> dict[str, Any]:
    """Referenced render name → its parsed rendered ``config.yml``.

    The public face of :func:`_render_configs` for callers that want the
    documents themselves rather than a predicate's verdict — today the lint
    belt's privilege check, which reads the same rendered files the credential
    grants read. Sharing the walk is the point: a second path join would be a
    second convention for where ``project_path`` resolves, and the two would
    drift the first time one of them learned about a deploy root and the other
    did not.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        One entry per referenced render whose rendered ``config.yml`` could be
        read. A render that is unset, unrendered or unreadable is absent —
        never present with a guessed document.
    """
    return dict(_render_configs(config, project_root))


def renders_needing_dispatcher_token(config: Any, project_root: Any) -> set[str]:
    """Names of catalog renders whose rendered project needs the dispatcher token.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names entitled to
        ``EVENT_DISPATCHER_TOKEN`` (see :func:`config_needs_dispatcher_token`).
    """
    return _renders_whose_config(config, project_root, config_needs_dispatcher_token)


def config_declares_bluesky_panel(config: Any) -> bool:
    """Whether this project shows the BLUESKY tab, and so proxies into the
    bluesky-web sidecar with its own operator secret."""
    return config_declares_panel(config, BLUESKY_PANEL_ID)


def renders_declaring_bluesky_panel(
    config: Any, project_root: Any, layer_root: Any = None
) -> set[str]:
    """Names of catalog renders whose rendered project declares the BLUESKY tab.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :param layer_root: Where a build in flight has rendered its renders;
        see :func:`_render_configs`.
    :return: The subset of referenced render names whose users' operator
        secrets the bluesky-web sidecar is handed (see
        :func:`config_declares_bluesky_panel`).
    """
    return _renders_whose_config(config, project_root, config_declares_bluesky_panel, layer_root)


def _bluesky_panel_roster_grants(
    config: Any, project_root: Any, layer_root: Any = None
) -> list[tuple[str, str]]:
    """The roster users the bluesky-web sidecar is handed a secret for.

    The one entitlement walk behind every rendered form of the grant: the
    variable list the sidecar's compose declares
    (:func:`bluesky_panel_secret_env_vars`) and the owner map its gate names
    people by (:func:`bluesky_panel_roster_owners`) are two views of this same
    answer, so a user can never be handed a key under one and be a stranger
    under the other.

    Entitlement is decided the way the web-terminal render decides every
    per-user grant: a user with a render (their own, else
    ``default_tier``) by that render's rendered ``config.yml``
    (:func:`renders_declaring_bluesky_panel`). Roster order, so the rendered compose is stable across runs.

    Read off the renders' rendered ``config.yml`` files, like every other
    per-render grant. The render that carries it is ``osprey build``'s — the
    start verbs are as-built and re-render nothing — so the build renders its
    renders before the services compose and passes ``layer_root``, the
    staging tree those renders sit in until the swap publishes it as
    ``build/``. Without it the walk reads the published zone: the previous
    build's renders, or none.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :param layer_root: Where a build in flight has rendered its renders;
        see :func:`_render_configs`.
    :return: ``(username, OSPREY_TERMINAL_SECRET_<SUFFIX>)`` pairs, one per
        entitled user, in roster order; empty when no user shows the tab.
    """
    from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var

    web_terminals = as_dict(as_dict(as_dict(config).get("modules")).get("web_terminals"))
    raw_users = authored_people(web_terminals)
    # A read, not a gate: an authorization block that does not parse binds no
    # role here and is lint's finding to raise, so the roster is resolved
    # non-strictly and a role-only entry is answered by the render its role
    # binds it to, exactly as the web-terminal render answers it.
    try:
        authorization_roles = resolve_authorization_roles(web_terminals)
    except ValueError:
        authorization_roles = {}
    refs = _render_ref_by_name(raw_users, authorization_roles, strict=False)
    default_tier = web_terminals.get("default_tier")
    if not isinstance(default_tier, str) or not default_tier:
        default_tier = None
    entitled_renders = renders_declaring_bluesky_panel(config, project_root, layer_root)

    grants: list[tuple[str, str]] = []
    for entry in roster_people(web_terminals):
        tier = refs.get(entry["name"]) or default_tier
        if tier in entitled_renders:
            grants.append((entry["name"], terminal_secret_var(entry["name"])))
    return grants


def bluesky_panel_secret_env_vars(
    config: Any, project_root: Any, layer_root: Any = None
) -> list[str]:
    """The per-user secret variables the bluesky-web sidecar is handed.

    One name per roster user whose terminal shows the BLUESKY tab — and so
    proxies into the sidecar with the operator secret ITS container holds,
    under the fixed ``OSPREY_TERMINAL_SECRET`` name. The sidecar's own compose
    file (``services/bluesky_web``) lists each of these variables so its web
    gate accepts every entitled user's secret beside the deployment-wide one,
    and no user's container is ever handed the deployment secret.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :param layer_root: Where a build in flight has rendered its renders;
        see :func:`_render_configs`.
    :return: Variable names (``OSPREY_TERMINAL_SECRET_<SUFFIX>``), one per
        entitled user; empty when no user shows the tab.
    """
    return [var for _, var in _bluesky_panel_roster_grants(config, project_root, layer_root)]


def bluesky_panel_roster_owners(config: Any, project_root: Any, layer_root: Any = None) -> str:
    """Whose secret each variable of the roster grant is.

    Rendered beside the grant as ``OSPREY_TERMINAL_ROSTER_OWNERS`` — comma
    separated ``SUFFIX=username`` pairs, in the grant's own order. The
    sidecar's web gate admits any of the granted secrets, and names the
    operator behind an admitted request by the one that matched; without this
    map a match says only which slot the credential filled, not whose account
    it is, and every panel action the deployment records would name nobody.

    Each key is sliced off the variable name the grant itself emits rather
    than derived from the username a second time, so a key here cannot name a
    variable the sidecar was not handed. The render refuses a roster name that
    is not a single ``[A-Za-z0-9][A-Za-z0-9._-]*`` component, which is what
    makes ``,`` and ``=`` safe as this map's punctuation.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :param layer_root: Where a build in flight has rendered its renders;
        see :func:`_render_configs`.
    :return: ``"<SUFFIX>=<username>,..."``; empty when no user shows the tab.
    """
    from osprey.deployment.web_terminals.auth_credentials import TERMINAL_SECRET_VAR_PREFIX

    return ",".join(
        f"{var.removeprefix(TERMINAL_SECRET_VAR_PREFIX)}={name}"
        for name, var in _bluesky_panel_roster_grants(config, project_root, layer_root)
    )


def renders_needing_ariel_password(config: Any, project_root: Any) -> set[str]:
    """Names of catalog renders whose rendered project configures ARIEL.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names entitled to
        ``ARIEL_DB_PASSWORD`` (see :func:`config_needs_ariel_password`).
    """
    return _renders_whose_config(config, project_root, config_needs_ariel_password)


def renders_needing_facility_bundle(
    config: Any, project_root: Any, layer_root: Any = None
) -> set[str]:
    """Names of catalog renders whose rendered project configures a knowledge bundle.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names whose container gets the
        deployment bundle bind-mounted (see :func:`config_needs_facility_bundle`).
    :param layer_root: :func:`_render_configs`'s staging tree, for a build in
        flight; ``None`` reads the published build zone.
    """
    return _renders_whose_config(config, project_root, config_needs_facility_bundle, layer_root)


def renders_needing_ariel_mirror(
    config: Any, project_root: Any, layer_root: Any = None
) -> set[str]:
    """Names of catalog renders whose rendered project writes the ARIEL mirror.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names whose container gets the
        deployment's mirror bind-mounted (see :func:`config_needs_ariel_mirror`).
    :param layer_root: :func:`_render_configs`'s staging tree, for a build in
        flight; ``None`` reads the published build zone.
    """
    return _renders_whose_config(config, project_root, config_needs_ariel_mirror, layer_root)


def renders_needing_phoebus_handles(config: Any, project_root: Any) -> set[str]:
    """Names of catalog renders whose rendered project must address Phoebus displays by handle.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names whose container gets
        ``PHOEBUS_REQUIRE_HANDLE=1`` (see :func:`config_needs_phoebus_handles`).
    """
    return _renders_whose_config(config, project_root, config_needs_phoebus_handles)


def renders_needing_launch_token_by_lane(config: Any, project_root: Any) -> dict[str, set[str]]:
    """Which renders may arm a queue start, per plan lane.

    The tier boundary in roster form, one set per lane, and the shape
    every lane's own token is granted from: each lane carries its own
    ``<PREFIX>_LAUNCH_TOKEN``, and a render holds a lane's token only where
    that lane's control target is armed in its rendered ``config.yml``. A
    deployment whose baseline is a live machine therefore hands out the VA
    lane's token while withholding lane 1's, which is the whole point of the
    posture being per target rather than per deployment.

    Asked once per lane through :func:`_renders_whose_config`, the same walk
    every other per-render grant uses, so no lane can disagree with the others
    about which renders a roster deploys.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: ``{lane: renders}`` for the lanes that entitle somebody. A lane no
        render may arm is ABSENT rather than present with an empty set — the
        deployment renders no such grant at all, and an empty entry would read
        like one that was withheld.
    """
    by_lane: dict[str, set[str]] = {}
    for lane in LANE_KEYS:
        entitled = _renders_whose_config(
            config,
            project_root,
            lambda render_config, lane=lane: config_needs_launch_token_for(render_config, lane),
        )
        if entitled:
            by_lane[lane] = entitled
    return by_lane


def renders_needing_graphdb_password(config: Any, project_root: Any) -> set[str]:
    """Names of catalog renders whose rendered project queries the graph store.

    Unlike :func:`renders_needing_launch_token_by_lane`, this set deliberately spans
    both tiers: a read-only render that configures a graph store belongs in it,
    because the store has exactly one account and the read-only guarantee lives
    in the ``graph`` MCP server's read transactions rather than in the
    credential. See :func:`config_needs_graphdb_password` for that reasoning and
    for the blast radius that bounds it.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced render names entitled to
        ``GRAPHDB_PASSWORD`` (see :func:`config_needs_graphdb_password`).
    """
    return _renders_whose_config(config, project_root, config_needs_graphdb_password)


def renders_needing_archiver_credentials(
    config: Any, project_root: Any
) -> dict[str, tuple[str, ...]]:
    """Map each catalog render whose archiver names credentials to the variables it reads.

    A map rather than a set because the grant carries the variable NAMES (see
    :func:`config_archiver_credential_envs`): two renders reading two stores each
    get their own lines, and the render emits exactly the names the connector
    will look up. Walks the same per-render ``config.yml`` files the other
    grants walk, so this cannot disagree with them about which renders a
    roster deploys.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: ``{render_name: (env_var_name, ...)}`` for the referenced renders
        whose selected archiver connector names a credential variable.
    :raises ValueError: when a render names a variable compose cannot carry
        (see :func:`config_archiver_credential_envs`).
    """
    grants: dict[str, tuple[str, ...]] = {}
    for render_name, render_config in _render_configs(config, project_root):
        names = config_archiver_credential_envs(render_config)
        if names:
            grants[render_name] = names
    return grants


def renders_needing_archiver_ca_bundles(
    config: Any, project_root: Any
) -> dict[str, tuple[str, ...]]:
    """Map each catalog render whose archiver names a CA file on this host to that file.

    Walks the same per-render ``config.yml`` files
    :func:`renders_needing_archiver_credentials` walks, so the two grants agree
    about which renders a roster deploys.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: ``{render_name: (ca_path,)}`` for the referenced renders whose
        selected archiver block names a CA file on this host.
    """
    grants: dict[str, tuple[str, ...]] = {}
    for render_name, render_config in _render_configs(config, project_root):
        paths = config_archiver_ca_bundles(render_config)
        if paths:
            grants[render_name] = paths
    return grants


def normalize_users(users_raw: Any, *, strict: bool = True) -> list[dict[str, Any]]:
    """Normalize ``modules.web_terminals.users`` into explicit ``{"name", "index"}`` dicts.

    This is the canonical users-normalizer for the module; render.py delegates to it
    directly rather than re-parsing the raw roster itself. Its purpose is to let a
    decommission migration *freeze* a bare roster's positional indices onto explicit
    ``index`` fields before removing a user — once frozen, deleting an earlier entry
    can no longer shift a later user's port allocation, because the survivor's index
    no longer depends on its position in the list.

    Legacy bare strings (``"alice"``) normalize to ``{"name": "alice", "index":
    <raw list position>}``, matching the fallback ``render.py`` already uses so
    ports stay identical to what an all-strings roster produces.
    Already-explicit object entries (``{"name": ..., "index": ...}``) pass through
    with their explicit index preserved, regardless of list position. This makes
    the function idempotent: normalizing an already-normalized list is a no-op,
    which is what lets the freeze step be applied without checking whether a
    roster has already been frozen.

    An object entry's optional ``display_name`` (a human-facing window/tab title,
    surfaced downstream as the per-user ``OSPREY_WEB_APP_NAME``) is carried through
    onto the normalized entry when it is a string, and dropped defensively
    otherwise (a non-string ``display_name`` is a config typo, reported separately
    by lint). Bare-string entries never carry one. Keeping this passthrough here
    means the normalized entry stays the single object :func:`resolve_cards`
    reads a user's identity off of, rather than re-deriving the field from the raw
    roster the way ``render`` is.

    An object entry's optional ``theme`` (that user's default web UI theme --
    a theme family like ``desy`` or a concrete id like ``desy-light``, surfaced
    downstream as the per-user ``OSPREY_WEB_THEME``) is carried through on
    exactly the same terms. Validity of the *value* is not checked here: the
    web terminal resolves it at startup and warns+falls back on an unknown one,
    and lint reports a non-string separately.

    An object entry's optional ``tour`` (that user's onboarding-tour invite
    policy -- ``once``, ``always`` or ``never``, surfaced downstream as the
    per-user ``OSPREY_WEB_TOUR``) is carried through on the same terms as
    ``theme``, with the same division of labour: the web terminal owns the
    vocabulary and warns+falls back on an unknown value.

    An object entry's optional ``oidc_subject`` (the value of the configured OIDC
    claim -- ``sub`` by default -- that identifies this roster user at the IdP) is
    carried through on the same terms, with one deliberate difference: an *empty*
    string is dropped along with every non-string, where ``display_name`` and
    ``theme`` would keep it. This field is an authorization mapping, not a
    cosmetic one -- carrying ``""`` through would let an identity whose claim is
    missing or empty match a roster user. Dropping it instead leaves that user
    with no mapping at all, which the callback answers with 403. Only the
    *non-secret* side of the mapping ever lives in config.yml; password hashes
    never do (they live in ``.env.auth``, keyed by
    :func:`~osprey.services.auth_sidecar.roster_env.env_var_suffix`).

    An object entry's optional ``role`` (the name of a
    ``modules.web_terminals.authorization.roles`` entry, which names the render
    this user runs as) is carried through on exactly the same terms as
    ``oidc_subject``, empty-string drop included: it is an authorization mapping
    too, and a carried ``""`` would leave the entry naming a role no table can
    answer. Dropping it leaves the entry with no binding at all, which the shared
    render helper reads as "the deployment's default tier". A ``role`` that
    names no declared role, and a non-string one, are reported by lint.

    An object entry's optional ``control_identity`` (the name the control
    system sees this card's writes arrive under, surfaced downstream as
    ``OSPREY_CONTROL_IDENTITY``) is carried on the same terms as
    ``oidc_subject``, empty-string drop included: a carried ``""`` would name
    nobody. Its charset and reserved names are judged by lint, not here.

    An object entry's optional ``access`` (which principals may open this
    entry's card, rather than only the one user it belongs to) is *validated*
    here and carried through **verbatim** — the authored form, never a resolved
    set. :func:`resolve_access_principals` is the single parser of that form, so
    the normalized entry holds exactly what that function reads: ``"own"`` and
    ``"any"`` stay those strings, and a principal list stays the list the
    operator wrote. Keeping the authored value is what leaves the entry
    JSON/YAML-serializable, lets every consumer that already reads
    ``entry["access"]`` keep reading it, keeps the two shorthands rendering
    byte-identically, and keeps one place that knows the vocabulary.

    The key rides through only when the resolved set is something other than
    "just this entry's own user" — the same present-only-when-it-changes-
    something rule the other optional fields follow. ``"own"``, an absent key,
    an ``access:`` with nothing after it, and ``[self]`` all resolve to
    owner-only, so a roster that shares nothing normalizes exactly as it did
    before the principal set existed.

    A value the vocabulary does not recognise is the one thing here that is not
    quietly dropped. Under the default ``strict=True`` the parser's
    ``ValueError`` propagates, naming the entry and the offending value; under
    ``strict=False`` the whole entry is dropped instead. Neither reading lets an
    unrecognised value become an owner-only card, and that is the point: the
    surfaces that admit a login and the guards that vet the deployment must
    resolve one authored value the same way, and a value silently rewritten to
    "owner only" here is one the guards then have nothing to say about. See
    :func:`entry_is_shared` for the single predicate consumers ask.

    Malformed entries — anything that isn't a string, and any dict missing a
    string ``name`` or an int ``index`` — are dropped rather than raising
    (well-formedness is lint.py's job; a malformed ``access`` is the one
    exception, above). ``bool`` is a subclass of ``int`` in
    Python, so a dict entry
    like ``{"name": "alice", "index": True}`` would technically satisfy
    ``isinstance(index, int)``; this function deliberately treats ``bool`` as an
    invalid index type and drops such entries, since ``index: true/false`` in a
    facility config is a config typo (e.g. a YAML boolean where an int was meant),
    not a meaningful index.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value. Anything other
            than a list (including ``None``) is treated as an empty roster.
        strict: What an unreadable ``access`` does. ``True`` (the default)
            propagates the parser's refusal, naming the entry and the value;
            ``False`` drops that entry instead. :func:`resolve_cards` threads
            its own ``strict`` through to here.

            The lenient path is deliberately NOT an escape hatch for the verbs
            that mutate a deployment. Every one of them reaches this function at
            the strict default first, before anything is written — the
            decommission migration through :func:`freeze_user_indices`, and
            ``prune``, ``nuke``, ``users passwd`` and the seeding walk through
            their own roster reads — so an unreadable ``access`` refuses the
            verb rather than quietly shortening the roster it is about to act
            on. Lenient serves the *readers* that resolve a roster without
            touching it, and that would otherwise be stopped by a value they are
            not there to judge: the ``osprey init`` privileged-render check,
            lint's rules, the feedback volume targets, and ``nuke``'s
            image-tag sweep.

    Returns:
        New ``{"name": str, "index": int}`` dicts (plus optional
        ``"display_name"``, ``"theme"``, ``"tour"``, ``"oidc_subject"``,
        ``"role"`` and ``"control_identity"`` string keys when the entry carried
        them, and the authored ``"access"`` value
        verbatim when it admits anyone beyond the entry's own user) in
        config-declaration order. Input dicts are never mutated or returned by
        reference.

    Raises:
        ValueError: Under ``strict``, for an ``access`` value
            :func:`resolve_access_principals` does not recognise. The message
            names the entry and the value.
    """
    if not isinstance(users_raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for position, entry in enumerate(users_raw):
        if isinstance(entry, str):
            normalized.append({"name": entry, "index": position})
        elif isinstance(entry, dict):
            name = entry.get("name")
            index = entry.get("index")
            if isinstance(name, str) and isinstance(index, int) and not isinstance(index, bool):
                normalized_entry: dict[str, Any] = {"name": name, "index": index}
                display_name = entry.get("display_name")
                if isinstance(display_name, str):
                    normalized_entry["display_name"] = display_name
                theme = entry.get("theme")
                if isinstance(theme, str):
                    normalized_entry["theme"] = theme
                # Per-user onboarding-tour invite policy (once/always/never),
                # carried on the same terms as `theme`: surfaced downstream as
                # OSPREY_WEB_TOUR, and the web terminal warns + falls back on
                # an unknown value, so only the type is checked here.
                tour = entry.get("tour")
                if isinstance(tour, str):
                    normalized_entry["tour"] = tour
                oidc_subject = entry.get("oidc_subject")
                if isinstance(oidc_subject, str) and oidc_subject:
                    normalized_entry["oidc_subject"] = oidc_subject
                # Carried on the same terms as `oidc_subject`, and for the same
                # reason: `role` is an authorization mapping, so an empty string is
                # dropped rather than kept. A carried `""` would leave the entry
                # claiming a role no table can answer; dropping it leaves no binding
                # at all, which every consumer reads as "the default tier".
                role = entry.get("role")
                if isinstance(role, str) and role:
                    normalized_entry["role"] = role
                # The name the control system sees this card write as. Carried on
                # the same terms as `oidc_subject`: a carried `""` would ask the
                # container to switch to an identity that names nobody, so an
                # empty or non-string value is dropped and lint reports it.
                control_identity = entry.get("control_identity")
                if isinstance(control_identity, str) and control_identity:
                    normalized_entry["control_identity"] = control_identity
                # `access` is validated through the single parser, then carried
                # verbatim: the authored form is what that parser reads, what
                # freeze writes back, and what keeps `own`/`any` rendering
                # byte-identically. An unreadable value is refused (or, for a
                # lenient caller, takes its whole entry with it) rather than
                # normalized into an owner-only card that no guard would then
                # have anything to say about.
                try:
                    principals = resolve_access_principals(entry)
                except ValueError:
                    if strict:
                        raise
                    continue
                if principals != _OWNER_ONLY:
                    normalized_entry["access"] = entry["access"]
                normalized.append(normalized_entry)
    return normalized


# The value a `domain:` principal carries is compared case-insensitively (a
# domain is), so it is folded once, here, instead of at every comparison. The
# fold is deliberately ASCII-only: str.lower() applies Unicode case mapping,
# which is locale-surprising, and the sidecar's comparator folds with this same
# table — the two have to agree byte for byte. A non-ASCII domain never reaches
# the fold at all; it is refused above.
_ACCESS_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

# Quoted in every refusal below, so an operator who trips one learns the whole
# vocabulary rather than only which member was wrong.
_ACCESS_MEMBERS = "'self', 'roster', 'user:<id>' or 'domain:<domain>'"

#: The resolved set every owner-only spelling comes to — `access: own`, an
#: unwritten key, an `access:` with nothing after it, and the explicit
#: `[self]`. Named because it is also the *dividing line*: an entry is a shared
#: card exactly when its resolved set is something other than this one (see
#: :func:`entry_is_shared`), and both the normalizer and the predicate must draw
#: that line in the same place.
_OWNER_ONLY = frozenset({"self"})

#: The resolved set ``access: any`` comes to. Named for the same reason
#: :data:`_OWNER_ONLY` is: it is the second set with a spelling of its own on
#: the wire (the literal ``any``), and the render and the sidecar must agree on
#: which set that token stands for.
_ROSTER_ONLY = frozenset({"roster"})


def _access_entry_label(entry: Mapping[str, Any]) -> str:
    """How an ``access`` refusal names the roster entry it is about.

    The name when the entry has one, since that is what an operator scans the
    file for. A nameless entry is not hypothetical — lint reads the authored
    roster, where a dropped or mistyped ``name`` key is exactly the kind of
    edit that also breaks ``access`` — and naming it ``None`` would send the
    reader looking for an entry called None. Its position is what identifies
    it instead, and ``index`` is the position the roster itself declares.

    Args:
        entry: A roster entry, authored rather than normalized: neither key is
            known to be present or of the right type.

    Returns:
        The subject phrase a refusal opens with, naming the config path either
        way so the message stands on its own.
    """
    name = entry.get("name")
    if isinstance(name, str) and name:
        return f"modules.web_terminals.users entry {name!r}"
    index = entry.get("index")
    if isinstance(index, int) and not isinstance(index, bool):
        return f"the unnamed modules.web_terminals.users entry at index {index}"
    return "an unnamed modules.web_terminals.users entry"


def _access_principal(member: Any, entry_label: str) -> str:
    """Resolve one member of an authored ``access`` list into a principal.

    Args:
        member: The authored member, of any type — a config value, not yet known
            to be a string.
        entry_label: :func:`_access_entry_label`'s naming of the entry, opening
            every refusal so the operator knows which entry to edit.

    Returns:
        The resolved principal: ``"self"``, ``"roster"``, the ``user:`` member
        byte-exact, or a ``domain:`` member with its domain ASCII-lower-cased.

    Raises:
        ValueError: For a non-string member, a bare word that is neither
            ``self`` nor ``roster``, the reserved ``group:`` prefix, an unknown
            prefix, a ``user:``/``domain:`` member carrying a ``$`` (which
            docker compose would interpolate away in transit), or a known
            prefix carrying a malformed value.
    """
    if not isinstance(member, str):
        raise ValueError(
            f"{entry_label} has an access member "
            f"{member!r}, which is not a string. Members are written as {_ACCESS_MEMBERS}"
        )
    if member in ("self", "roster"):
        return member
    prefix, separator, value = member.partition(":")
    if not separator:
        raise ValueError(
            f"{entry_label} has an access member "
            f"{member!r}, which names no principal kind. Only 'self' and 'roster' are "
            f"written bare; every other member carries a prefix ({_ACCESS_MEMBERS})"
        )
    if prefix == "group":
        raise ValueError(
            f"{entry_label} has an access member "
            f"{member!r}: the 'group:' prefix is reserved and not supported yet. Name the "
            "people individually with 'user:<id>', or admit their whole domain with "
            "'domain:<domain>'"
        )
    if prefix in ("user", "domain") and "$" in member:
        # Both prefixes at once, because the reason is the transport and not
        # the kind: the resolved member is written into the deployment's
        # compose file, and compose interpolates `$NAME`/`${NAME}` in a value
        # before the container ever sees it. Refused here rather than escaped,
        # so lint, `osprey init` and the render all say the same sentence about
        # it; no identity provider issues a `$` in a subject or a domain, so
        # nothing legitimate is being turned away.
        raise ValueError(
            f"{entry_label} has an access member "
            f"{member!r} containing '$'. The compose file substitutes '$' references before "
            "the sidecar compares the member. Name an identity or a domain without a '$'"
        )
    if prefix == "user":
        if not value or any(character.isspace() for character in value):
            raise ValueError(
                f"{entry_label} has an access member "
                f"{member!r} carrying no usable identity after 'user:'. Write the value the "
                "configured OIDC claim holds for one person, e.g. 'user:alice@example.org'"
            )
        return member
    if prefix == "domain":
        if (
            not value
            or any(character.isspace() or character in ":/" for character in value)
            or "@" in value
            or value.startswith(".")
            or value.endswith(".")
        ):
            raise ValueError(
                f"{entry_label} has an access member "
                f"{member!r} that is not a bare domain. Write the domain on its own, e.g. "
                "'domain:example.org' — an address ('domain:alice@example.org') admits one "
                "person, "
                "which is what 'user:' is for"
            )
        if not value.isascii():
            # An internationalised domain reaches a relying party as punycode:
            # that is the form the provider puts in the claim, and nothing here
            # translates between the two spellings. A Unicode-authored value
            # would therefore be a rule that admits nobody, ever — refused at
            # author time rather than deployed as a grant that silently never
            # fires.
            raise ValueError(
                f"{entry_label} has an access member "
                f"{member!r} carrying a non-ASCII domain. An identity provider asserts an "
                "internationalised domain in its punycode ('xn--') form, so this member "
                "would match nobody. Write the punycode spelling of the domain"
            )
        return f"domain:{value.translate(_ACCESS_ASCII_LOWER)}"
    unknown = f"{prefix}:"
    raise ValueError(
        f"{entry_label} has an access member {member!r} "
        f"with an unknown prefix {unknown!r}. Members are written as {_ACCESS_MEMBERS}"
    )


def resolve_access_principals(entry: Mapping[str, Any]) -> frozenset[str]:
    """Resolve a roster entry's authored ``access`` into its set of principals.

    ``users[].access`` says who may open an entry's card. It is authored as a
    principal set — allow-only, unioned, fail-closed — whose members are

    * ``self`` — the one user the entry belongs to,
    * ``roster`` — any authenticated user who has a roster entry,
    * ``user:<id>`` — one identity, named by the value of the configured OIDC
      claim,
    * ``domain:<domain>`` — every identity whose email is in that domain.

    Three authored forms are exact sugar for a one-member list, and are what
    every roster written before the set existed uses:

    * ``access: own`` ≡ ``[self]``,
    * ``access: any`` ≡ ``[roster]``,
    * an unwritten ``access`` ≡ ``[self]`` — owner-only is the default, so a
      roster that never mentions the key keeps meaning exactly what it did.

    This is the single parser of the authored form: consumers read the set it
    returns, never the raw value. That is the point of it. A value no consumer
    recognises must not reach an admission check as a shape that happens to
    look safe there, so an unrecognised one is refused here, by name, at the
    one place that looks at it. Unlike :func:`normalize_users`, which drops
    malformed roster entries rather than raising, this function raises: the
    caller decides whether to propagate the refusal (a render, a deploy
    preflight, or any verb about to mutate a deployment) or to degrade (a
    read-only summary, lint, or ``osprey init``'s privileged-render check),
    which it cannot do if the bad value has already been silently rewritten
    into a good one.

    ``group:`` is refused by name rather than as an unknown prefix, because it
    is the kind the vocabulary is expected to grow next: "not supported yet" is
    a different instruction to an operator than "unknown".

    A ``domain:`` value is ASCII-lower-cased, since domains compare
    case-insensitively and folding once here keeps every consumer comparing the
    same bytes. The comparator on the login side applies the same ASCII-only
    fold to the asserted email's domain. A domain carrying non-ASCII is refused
    rather than translated: an IdP asserts an internationalised domain in
    punycode, nothing here invents that translation, and a rule that could
    never match anybody is an error at author time and not a deployed grant. A
    ``user:`` value is left byte-exact: it is an identity
    string the IdP owns, and folding it would merge two identities an IdP may
    well keep distinct.

    Args:
        entry: A roster entry carrying the authored ``access`` value under that
            key, verbatim. Never mutated. Every refusal opens by naming the
            entry — by ``name``, or by ``index`` when it has none (see
            :func:`_access_entry_label`).

    Returns:
        The resolved principals, deduplicated. Never empty: every accepted form
        admits at least one principal.

    Raises:
        ValueError: When ``access`` is neither shorthand nor a list, when the
            list is empty, or for any malformed member (see
            :func:`_access_principal`). The message names the entry and the
            offending value.
    """
    authored = entry.get("access")
    entry_label = _access_entry_label(entry)
    if authored is None:
        # Both an absent key and an authored `access:` with nothing after it,
        # which YAML hands over as None. Owner-only is the fail-closed reading.
        return _OWNER_ONLY
    if authored == "own":
        return _OWNER_ONLY
    if authored == "any":
        return frozenset({"roster"})
    if not isinstance(authored, list):
        raise ValueError(
            f"{entry_label} has access {authored!r}, which "
            f"is neither of the two shorthands ('own', 'any') nor a list whose members are "
            f"{_ACCESS_MEMBERS}"
        )
    if not authored:
        raise ValueError(
            f"{entry_label} has an empty access list, "
            "which would admit nobody at all — not even its own user. Write 'access: own' "
            "(or drop the key) to keep the entry owner-only, or list the principals that "
            "may open it"
        )
    return frozenset(_access_principal(member, entry_label) for member in authored)


def access_wire_value(principals: frozenset[str]) -> str:
    """Spell a resolved principal set for the sidecar's per-entry env line.

    The inverse of the authored vocabulary: :func:`resolve_access_principals`
    reads ``users[].access`` from ``config.yml`` into a set, and this spells
    that set into the value ``OSPREY_AUTH_ROSTER_ACCESS_<SUFFIX>`` carries
    across the compose file to the auth sidecar, which parses it back. Three
    shapes, and the first two are the reason this is not simply ``json.dumps``:

    * owner-only (``{"self"}``) → ``""``, and the render emits no line at all.
      An absent variable is what the sidecar has always read as owner-only, so
      a roster that shares nothing renders exactly as it did before the
      principal set existed.
    * the whole roster (``{"roster"}``) → the literal ``"any"``, byte-identical
      to what ``access: any`` has always emitted. That shorthand is what nearly
      every existing roster carries, and upgrading into the principal set must
      not rewrite a deployment's compose file.
    * anything else → a compact JSON array of the members, sorted:
      ``["domain:example.org"]``, ``["self","user:carol@example.org"]``. Sorted because
      a set has no order of its own and an unstable render would churn the
      file; compact because the value shares a line with its key. ``self`` is
      kept when it appears beside others — there it is a member of the admitted
      set, not a marker. A ``roster`` member inside an array is NOT collapsed
      back to ``any`` either: the render never collapses a set it was given,
      and it is the sidecar that decides a roster member covers everyone.

    The members are already normalized by :func:`resolve_access_principals`
    (``self``, ``roster``, ``user:<id>`` byte-exact, ``domain:`` ASCII-lower-
    cased), so the sidecar compares the same bytes this side resolved and
    neither end folds anything a second time.

    Args:
        principals: A resolved set, as :func:`resolve_access_principals`
            returns it. Never empty — every accepted authored form admits at
            least one principal.

    Returns:
        The wire value, or ``""`` when the entry admits only its own user, for
        which the render emits nothing.
    """
    if principals == _OWNER_ONLY:
        return ""
    if principals == _ROSTER_ONLY:
        return "any"
    return json.dumps(sorted(principals), separators=(",", ":"))


def entry_is_shared(entry: dict[str, Any]) -> bool:
    """Whether this roster entry is a shared card — one somebody else may open.

    An entry is shared exactly when :func:`resolve_access_principals` admits
    anyone beyond the entry's own user: ``access: any`` (the whole roster),
    ``[domain:example.org]``, ``[user:alice@example.org]``, or any list that names more
    than ``self``. ``own``, an unwritten key and ``[self]`` are the owner-only
    set, and are the only forms that answer ``False``.

    This is the single *predicate* over the parsed set, and it is consumed at
    every call site that must not disagree about who may open an entry:
    credential provisioning (whose logins may mint a session for it), the deploy
    summary (which logins are seeded, and which cards are shared), the ``users
    passwd`` verb's refusal path, the render's per-entry service dict, and three
    of lint's rules. Two of those lint call sites pass an entry straight off the
    *raw* roster rather than a normalized one; both readings agree, because the
    normalized entry carries the authored value verbatim and this function
    parses that same value.

    Args:
        entry: A roster entry — authored or normalized — carrying the authored
            ``access`` value under that key. Never mutated.

    Returns:
        ``True`` when the entry admits any principal other than its own user.

    Raises:
        ValueError: For an ``access`` value the vocabulary does not recognise.
            The predicate refuses rather than answering: a value no consumer
            understands must not be read as an owner-only card here while
            another surface reads it as admitting somebody.
    """
    return resolve_access_principals(entry) != _OWNER_ONLY


def _is_shared_entry(user: Any) -> bool:
    """:func:`entry_is_shared` over a RAW roster entry, without its refusal.

    The predicate raises for an ``access`` value the vocabulary does not
    recognise, deliberately: a value no consumer understands must not be read
    as an owner-only card at one surface while another reads it as admitting
    somebody. Every caller that holds a *normalized* entry is already past that
    refusal — :func:`normalize_users` drops such an entry before it is
    resolved — but lint's shared-card rules and the control-identity builders
    ask the question of the raw roster, where the bad value is still there.

    So they absorb it, and read the entry as not shared. Linting is how the
    operator LEARNS the value is unreadable: lint's access rule reports it with
    the sentence that says what to write instead, and a rule that propagated
    the refusal instead would take that report — and every other finding in the
    run — down with it over the one config the operator most needs a report
    for. Treating the entry as not shared is also the fail-safe reading *for
    these callers specifically*: each only ever ADDS a finding about a shared
    card, so skipping the entry withholds a finding rather than blessing
    anything.

    Args:
        user: One entry straight off ``modules.web_terminals.users``, of either
            roster form.

    Returns:
        Whether the entry admits anyone beyond its own user; ``False`` for a
        bare-string entry and for an ``access`` value that cannot be read.
    """
    if not isinstance(user, dict):
        return False
    try:
        return entry_is_shared(user)
    except ValueError:
        return False


def freeze_user_indices(users_raw: Any) -> list[dict[str, Any]]:
    """The roster **as authored**, with :func:`normalize_users`' indices frozen onto it.

    :func:`normalize_users` answers "who is on the roster, at which index" and
    deliberately projects each entry down to the handful of fields the render
    reads. That projection is the right input for rendering and the WRONG thing
    to write back to ``config.yml``: every key it does not know about —
    ``render`` above all — would be dropped from the file, and a roster whose
    ``tier:`` keys are gone re-resolves every survivor onto
    ``default_tier`` (see :func:`resolve_cards`, which reads ``render``
    off the raw roster). For a user removal that is a silent privilege change,
    not a cosmetic one, whenever the default tier is the more privileged.

    ``role:`` is the OTHER key whose loss is that same silent privilege change,
    and it must reach the file for the same reason even though it reaches it by
    a different route: :func:`normalize_users` carries ``role`` through, so it
    survives the projection rather than being re-attached from the authored
    entry. Both keys are inputs to :func:`effective_tier`, so a survivor
    written back without either one drops onto ``default_tier`` on the next
    render. Named here because the two are one contract to a reader, whichever
    half of this function happens to deliver them.

    So this function starts from :func:`normalize_users`' output — survival and
    index-freezing stay entirely that function's contract, with no second copy
    of its rules here — and re-attaches the keys the author actually wrote.
    Entries are matched by ``name``, the same key
    :func:`_render_ref_by_name` and every other per-user artifact in this
    module (container names, volume names) is keyed by. A roster listing one
    name twice (a duplicate-name config error lint reports separately) therefore
    gives both entries the *last* authored entry's extra keys; each still keeps
    its own frozen index.

    Values are carried through verbatim, including ones
    :func:`normalize_users` would drop as malformed (a non-string
    ``display_name``, say). Removing one user must not quietly rewrite another
    user's config: what the author wrote about the survivors goes back to the
    file unchanged, and lint keeps reporting it.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value.

    Returns:
        One dict per surviving :func:`normalize_users` entry, in the same order,
        carrying every key its authored entry had plus an explicit ``index``. A
        bare-string entry becomes ``{"name", "index"}``, since a string carries
        nothing else — unless the same name is also spelled as an object
        somewhere in the list, which is the duplicate-name case above. Input
        dicts are never mutated.
    """
    authored_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(users_raw, list):
        for entry in users_raw:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                authored_by_name[entry["name"]] = entry

    frozen: list[dict[str, Any]] = []
    for normalized in normalize_users(users_raw):
        entry = dict(authored_by_name.get(normalized["name"], {}))
        # normalize_users wins on the keys it owns: `index` is the frozen one,
        # not whatever the file said, and `name` is already identical.
        entry.update(normalized)
        frozen.append(entry)
    return frozen


# ── The roster as its readers see it ─────────────────────────────────────────
#
# The roster is two lists in the profile: the people under `users:` and the
# shared app cards under `shared:`. A reader that asks "which cards does this
# deployment run" — containers, port slots, terminal secrets, liveness probes,
# orphan detection — takes `roster_entries`, so an app card is never missed. A
# reader whose question is about a *person* — a login, a tier or role binding,
# history scope, a control identity, a write-back of the roster — takes
# `roster_people` (normalized) or `authored_people` (as written). These three
# are the only readers of the raw keys; every other module goes through them.

#: The ``kind`` every derived app entry carries. People carry no ``kind`` at
#: all, so a roster without ``shared:`` normalizes exactly as it always has.
APP_ENTRY_KIND = "app"


def authored_people(web_terminals: Any) -> list[Any]:
    """The people of a ``modules.web_terminals`` subtree, exactly as authored.

    For the readers that need keys the normalizer projects away (``tier``,
    ``render``, ``role``, ``history``, ``oidc_subject`` on a malformed entry)
    or that report on entries the normalizer would drop. Never lists an app
    card. A copy: appending to it does not edit the profile.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree; anything that is
            not a mapping has no roster.

    Returns:
        The raw ``users:`` entries in roster order; empty when ``users`` is
        absent or not a list.
    """
    users = as_dict(web_terminals).get("users")
    return list(users) if isinstance(users, list) else []


def roster_people(web_terminals: Any, *, strict: bool = True) -> list[dict[str, Any]]:
    """The people of a ``modules.web_terminals`` subtree, normalized.

    :func:`normalize_users` over :func:`authored_people`, and the people half of
    :func:`roster_entries` exactly. For a reader whose question is about a
    person rather than a card.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree.
        strict: Passed to :func:`normalize_users`.

    Returns:
        Normalized people entries in roster order.
    """
    return normalize_users(authored_people(web_terminals), strict=strict)


def is_app_entry(entry: Any) -> bool:
    """Whether a roster entry is a derived shared app card rather than a person."""
    return isinstance(entry, dict) and entry.get("kind") == APP_ENTRY_KIND


def roster_entries(
    web_terminals: Any,
    *,
    strict: bool = True,
    app_titles: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Every card a ``modules.web_terminals`` subtree runs: people, then apps.

    The people are :func:`roster_people`, byte for byte. After them comes one
    derived entry per name in ``shared:``::

        {"kind": "app", "name": <app>, "app": <app>,
         "index": INDEX_MAX - <position in shared>,
         "display_name": <the app's title>, "access": "any"}

    The derived slots count down from the top of the family band, so adding
    people never moves an app and adding an app never moves a person.
    Reordering or removing an app may move the later apps' slots; nothing on a
    card is keyed by its port. A ``shared:`` item that is not a non-empty
    string derives nothing but keeps its position, so the apps after it keep
    their slots; lint reports the item. Duplicates and names that collide with
    a person are derived as written and are lint's to report, not this
    reader's to hide.

    Derived entries exist in memory only: nothing that writes the roster back
    to ``config.yml`` or ``profile.yml`` reads them (see
    :func:`freeze_user_indices`).

    Args:
        web_terminals: The ``modules.web_terminals`` subtree. Anything that is
            not a mapping has no roster.
        strict: Passed to :func:`normalize_users` for the people.
        app_titles: The app name -> title mapping (each app file's top-level
            ``name:``; see :func:`app_titles_in`). An app it does not title is
            shown under its name.

    Returns:
        People then app entries, each in authored order.
    """
    from osprey.port_layout import INDEX_MAX

    subtree = as_dict(web_terminals)
    entries = roster_people(subtree, strict=strict)
    shared = subtree.get("shared")
    titles = app_titles or {}
    for position, name in enumerate(shared if isinstance(shared, list) else []):
        if not isinstance(name, str) or not name:
            continue
        title = titles.get(name)
        entries.append(
            {
                "kind": APP_ENTRY_KIND,
                "name": name,
                "app": name,
                "index": INDEX_MAX - position,
                "display_name": title if isinstance(title, str) and title else name,
                "access": "any",
            }
        )
    return entries


def app_card_refusal(name: str) -> str:
    """The one message every per-person roster verb refuses a shared app card with.

    Args:
        name: The app name the operator gave.

    Returns:
        The refusal, naming where the app card is actually removed.
    """
    return (
        f"{name} is a shared app card; remove it from modules.web_terminals.shared "
        "in profile.yml and rebuild"
    )


def refuse_app_card(web_terminals: Any, name: str) -> None:
    """Refuse *name* when it is a shared app card rather than a person.

    A shared app card has no workspace of its own to decommission and no
    password of its own to rotate; it leaves the deployment by being dropped
    from ``shared:``. Resolved through
    :func:`roster_entries`, the one reader that knows both people and apps, so a person and an app are never
    confused. Called before any side effect, so a refusal touches nothing.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree.
        name: The name the operator gave.

    Raises:
        ValueError: If *name* is a shared app card.
    """
    entries = roster_entries(web_terminals, strict=False)
    if any(is_app_entry(entry) and entry["name"] == name for entry in entries):
        raise ValueError(app_card_refusal(name))


def app_titles_in(apps_dir: Any) -> dict[str, str]:
    """Each app file's title, read off the top-level ``name:`` of ``apps_dir/*.yml``.

    The mapping :func:`roster_entries` takes as ``app_titles``. A file that does
    not parse, or names no string title, is left out (its card falls back to
    the app name); a missing directory has no titles.

    Args:
        apps_dir: A repo's ``apps/`` directory.

    Returns:
        ``{app name: title}`` keyed by file stem.
    """
    import yaml

    directory = Path(apps_dir)
    titles: dict[str, str] = {}
    if not directory.is_dir():
        return titles
    for path in sorted(directory.glob("*.yml")):
        try:
            data = yaml_loader.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        title = data.get("name") if isinstance(data, dict) else None
        if isinstance(title, str) and title:
            titles[path.stem] = title
    return titles


# ── Deployment-editing privilege ─────────────────────────────────────────────
#
# Three config keys decide whether a render can edit the deployment it runs
# in, listed once in :data:`DEPLOYMENT_EDIT_DOORS`: the agent's own setup tool
# (``claude_code.permissions.deny``, read through
# :func:`~osprey.cli.profile_conventions.is_setup_patch_capable`), the
# browser's Config panel (``web.config_panel.enabled``) and the scaffold
# gallery's writes (``web.scaffold_gallery.write_enabled``). They are doors
# onto one capability — the agent edits ``config.yml`` through the tool, a
# person edits the same file through the panel, and the gallery authors the
# rules, skills and agents the deployment's agent reads — so everything below
# treats holding ANY of them as privileged.
#
# The judgement lives in this neutral module because three commands ask it at
# three different altitudes, from three shapes of input, and they must not
# answer it differently:
#
# * ``osprey init`` asks it of a resolved render PRESET
#   (:func:`osprey.cli.profile_cmd._layer_profile_texts`);
# * ``osprey profile validate`` / ``osprey build`` ask it of the host profile's
#   ``config:`` overrides with each ``tiers/<name>.yml`` delta layered on
#   top (:mod:`osprey.deployment.web_terminals.lint`);
# * ``osprey up`` asks it of each render's rendered ``config.yml`` (the same
#   lint module, one gate later).
#
# :func:`tier_privileges` takes all three by taking *layers*: any number of
# config documents, read newest-last, in whichever spelling each one uses.


#: How the setup-tool privilege is named inside a guard message. Prose, not a
#: key: the message names the key separately, and an operator reading "this
#: render can edit the deployment" acts on it faster than one reading a deny
#: list entry.
PRIVILEGE_SETUP_TOOL = "the agent's deployment-editing setup tool"

#: How the Config-panel privilege is named inside a guard message.
PRIVILEGE_CONFIG_PANEL = "the web Config panel"

#: How the scaffold-gallery privilege is named inside a guard message.
PRIVILEGE_SCAFFOLD_WRITES = "the scaffold gallery's writes"


#: The bundled tier every remedy below points at. Quoted rather than described
#: because a remedy an operator can paste is worth more than one they have to
#: translate; a deployment that renamed its tiers still reads it as "a render
#: without these privileges", which the same sentence says.
UNPRIVILEGED_TIER_EXAMPLE = "readonly"

#: The keys :func:`tier_privileges` reads, in their dotted spelling. Every
#: other spelling of the same path is reached by :func:`_declared_values`.
_DENY_KEY = "claude_code.permissions.deny"
_REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
_CONFIG_PANEL_KEY = "web.config_panel.enabled"
_SCAFFOLD_WRITES_KEY = "web.scaffold_gallery.write_enabled"


@dataclass(frozen=True)
class DeploymentEditDoor:
    """One config key through which a render can edit the deployment it runs in.

    Attributes:
        key: The door's dotted config key.
        closed: What closes it. ``False`` for a switch the server reads as a
            boolean (the door is open unless every spelling says ``False``);
            a tool name for a permissions list, which closes the door by
            containing it.
        privilege: How holding the door open is named in a guard message.
    """

    key: str
    closed: bool | str
    privilege: str

    @property
    def is_switch(self) -> bool:
        """Whether the door is a boolean switch rather than a list entry."""
        return isinstance(self.closed, bool)


#: Every door onto editing the deployment, in the order guard messages name
#: them. The one list read by the shared-card rule (through
#: :func:`tier_privileges`), the app-render composer
#: (:func:`osprey.cli.build_profile_merge.compose_app_render`), its check
#: (:func:`osprey.deployment.web_terminals.lint.lint_app_render`) and the
#: headless refusal of a dispatch tier
#: (:func:`osprey.cli.build_profile_schema.headless_refused_flags`).
DEPLOYMENT_EDIT_DOORS: tuple[DeploymentEditDoor, ...] = (
    DeploymentEditDoor(_DENY_KEY, SETUP_PATCH_TOOL, PRIVILEGE_SETUP_TOOL),
    DeploymentEditDoor(_CONFIG_PANEL_KEY, False, PRIVILEGE_CONFIG_PANEL),
    DeploymentEditDoor(_SCAFFOLD_WRITES_KEY, False, PRIVILEGE_SCAFFOLD_WRITES),
)

#: Every privilege :func:`tier_privileges` can report, in its order. Read by
#: guards that have to ask "does this deployment declare a privilege SPLIT at
#: all" without holding a tier: a baseline that already grants everything
#: leaves nothing for any render to rise above (see
#: :func:`privileges_beyond_baseline`), so a guard whose whole subject is the
#: split has nothing to say about that deployment — including when it cannot
#: read a render at all.
ALL_PRIVILEGES: tuple[str, ...] = tuple(door.privilege for door in DEPLOYMENT_EDIT_DOORS)

# The words the SERVER accepts for `web.config_panel.enabled`
# (`osprey.interfaces.web_terminal.app.coerce_config_flag`), restated rather
# than imported: this module is read by the deploy path and by lint, and
# importing the web application to borrow two frozensets would pull the whole
# interface package in behind them. Restated deliberately with the same
# contents — a guard that read `"false"` as truthy would report a panel ON that
# the deployment in fact serves OFF, and refuse a config that is already safe.
_PANEL_FALSE_WORDS = frozenset({"false", "no", "off", "0"})
_PANEL_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})

#: Distinguishes "this layer says nothing about the path" from "this layer sets
#: it to None", which are different answers for every key read here.
_UNSET = object()


def _declared_values(layer: Any, path: str) -> list[Any]:
    """Every value ``layer`` gives ``path``, in whichever spelling it uses.

    A profile's ``config:`` block is a flat bag of dotted keys, a rendered
    ``config.yml`` is fully nested, and both spellings reach the same key in
    the rendered file — ``config_update_fields`` takes either. A *privilege*
    check must not depend on which of two equivalent spellings its author
    chose, so every split point is tried: ``claude_code.permissions.deny`` as
    one top-level key, as ``claude_code.permissions:`` holding ``deny``, and as
    a nested ``claude_code:`` block, longest key first.

    Returns:
        One entry per spelling actually present, in longest-dotted-key-first
        order; empty when this layer says nothing about the path. A layer that
        spells the same path twice yields two entries, and each caller below
        says how it folds them.
    """
    if not isinstance(layer, Mapping):
        return []
    parts = path.split(".")
    found: list[Any] = []
    for cut in range(len(parts), 0, -1):
        key = ".".join(parts[:cut])
        if key not in layer:
            continue
        node: Any = layer[key]
        for part in parts[cut:]:
            if not isinstance(node, Mapping) or part not in node:
                node = _UNSET
                break
            node = node[part]
        if node is not _UNSET:
            found.append(node)
    return found


def _permission_entries(value: Any) -> list[str]:
    """The string entries of a permissions list — read only when it *is* a list.

    Delegates to :func:`osprey.cli.profile_conventions.permission_entries`
    rather than restating its rule, **including its refusal of a bare string**:
    the two read the same two lists one step apart, and a document must not be
    composed here on terms the predicate then rejects. See that function for
    why a loose spelling is read as nothing, and why guessing "one entry" would
    diverge from the render in both directions at once.

    For this side specifically, contributing no deny leans toward reporting
    MORE privilege — the safe direction for a guard.
    """
    from osprey.cli.profile_conventions import permission_entries

    return permission_entries(value)


def _switch_flag(value: Any) -> bool:
    """Read one declaration of a door switch the way the server does.

    A real boolean is honoured as written and a quoted boolean word with it
    (``"false"`` is a human writing a boolean). Anything else is a value the
    server itself discards in favour of the shipped default, which for both
    switch doors is ON — so it is read as ON here too, rather than as a quiet
    OFF that would let an unreadable value walk a privileged tier past the
    guard.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _PANEL_TRUE_WORDS:
            return True
        if token in _PANEL_FALSE_WORDS:
            return False
    return True


def render_capability_document(*layers: Any) -> dict[str, Any]:
    """The ``config.yml``-shaped document the setup-capability predicate reads.

    :func:`~osprey.cli.profile_conventions.is_setup_patch_capable` consumes one
    document with ``claude_code.permissions.deny`` and ``remove_deny`` beside
    each other, and owns the subtraction between them (exact membership,
    mirroring how ``settings.json`` renders). This assembles that document out
    of however many layers a caller holds, so the *composition* stays in one
    place and this owns only the reading:

    * **both spellings** of each key are read (see :func:`_declared_values`);
    * lists **union** across layers and across spellings, which is what profile
      resolution itself does with string lists across ``extends`` — a render
      delta cannot subtract from an inherited ``deny`` except through
      ``remove_deny``, which is exactly why that key exists.

    Args:
        *layers: Config documents, base first. A resolved profile's ``config:``
            bag, a layer's own ``config:`` bag, or a rendered
            ``config.yml`` — any mix. A rendered document carries no
            ``remove_deny`` (the render already applied it), which reduces the
            subtraction to a no-op rather than needing a separate path.

    :func:`osprey.cli.build_cmd._profile_setup_patch_capable` calls this with
    the one layer it holds, and answers a narrower question with the result
    (may the image chown ``build/config.yml`` to the agent's user). It used to
    assemble its own document from its own reading of the spellings, and the
    two readings drifted: it never learned the ``claude_code.permissions:``
    split point, so a profile that denied the setup tool that way was reported
    capable by the container and denied by the guard. One reader is the fix —
    the verdict already came from one predicate, and now the document does too.

    Returns:
        A document with both lists present, ready for
        :func:`~osprey.cli.profile_conventions.is_setup_patch_capable`.
    """
    deny: list[str] = []
    remove_deny: list[str] = []
    for layer in layers:
        for value in _declared_values(layer, _DENY_KEY):
            deny.extend(_permission_entries(value))
        for value in _declared_values(layer, _REMOVE_DENY_KEY):
            remove_deny.extend(_permission_entries(value))
    return {"claude_code": {"permissions": {"deny": deny, "remove_deny": remove_deny}}}


def tier_privileges(*layers: Any) -> tuple[str, ...]:
    """The deployment-editing privileges the render these layers describe holds.

    The one answer behind every guard in this module and behind lint's belt
    check: every door in :data:`DEPLOYMENT_EDIT_DOORS`, reported together so a
    message can name what an operator actually gets.

    The Config panel and the gallery's writes are ON unless a layer turns them
    off — that is the server's shipped default, so a render that never
    mentions a key has that door open, and reading an unmentioned key as "off"
    would walk exactly the tier this guards past it. Later layers win over
    earlier ones; a single layer that spells a key twice is read as ON unless
    *every* spelling turns it off, which is the same lean.

    Args:
        *layers: Config documents, base first — see
            :func:`render_capability_document`.

    Returns:
        The privilege of every open door, in :data:`ALL_PRIVILEGES` order;
        empty for a render holding none.
    """
    from osprey.cli.profile_conventions import is_setup_patch_capable

    privileges: list[str] = []
    for door in DEPLOYMENT_EDIT_DOORS:
        if door.is_switch:
            open_ = True
            for layer in layers:
                declarations = [_switch_flag(value) for value in _declared_values(layer, door.key)]
                if declarations:
                    open_ = any(declarations)
        else:
            # The one list door: the setup tool, denied through the deny list
            # less whatever remove_deny lifts.
            open_ = is_setup_patch_capable(render_capability_document(*layers))
        if open_:
            privileges.append(door.privilege)
    return tuple(privileges)


def privileges_beyond_baseline(held: Sequence[str], baseline: Sequence[str]) -> tuple[str, ...]:
    """What a render holds that the deployment it belongs to does not.

    The MIGRATION-SENSITIVE half of the guards' definition of privileged, and
    the reason that half is relative rather than absolute.

    **Which rule reads which.** Two rules are built on privilege, and they take
    their answer from different sides of this function on purpose:

    * :func:`privileged_default_tier_problem` reads this, the RELATIVE
      answer. A profile's ``default_tier`` is inherited from whatever the
      deployment was before tiers existed — nobody chose it as a privilege
      grant — so the finding is about an authoring mistake inside a declared
      split, and a legacy profile with no split has no better default to be
      pointed at.
    * :func:`shared_card_privileged_problems` reads :func:`tier_privileges`
      directly, the ABSOLUTE answer. ``access`` is not inherited from
      anything: it is a key somebody typed, on one entry, claiming that
      terminal may be opened by somebody other than the user it belongs to —
      every login the deployment has under ``any``, or the identities a
      ``user:``/``domain:`` list names. Whether the deployment declared a tier
      split has no bearing on what that terminal hands out — a floorless
      deployment sharing it hands out BOTH surfaces to everyone the entry
      admits — so exempting the floorless case is exactly the fail-open this
      belt exists to prevent.

    :func:`tier_privileges` answers an absolute question — can this document
    reach the setup tool, is the Config panel live — and every way of not having
    gated a surface reads as holding it, because that is the truth about the
    render. Refusing a build on that answer alone would refuse nearly every
    deployment in existence: the base tier's deny floor and
    ``web.config_panel.enabled: false`` are new, so a profile written before them
    opens every door for every render it has, and its ``default_tier``
    would fail a gate whose remedy ("point it at a tier that holds none of them") it
    has no such tier to satisfy.

    What these guards actually protect is the privilege SPLIT a deployment
    declared: a base that floors a capability and a tier that lifts it back for
    named people. A render is privileged when it rises above its own
    deployment's baseline, and a deployment whose baseline grants everything has
    no split to protect — every terminal is already equal, nothing is being
    handed to one entry that the others lack, and there is nothing here to
    report that would not simply be "you have not adopted tiers yet".

    The deliberate consequence, and the cost of being shippable: a deployment
    that floors nothing draws no ``default_tier`` finding even when every
    tier it has can edit it. That much is the *base tier's* to fix — it is what
    the shipped base's deny floor and panel default exist for. What it is NOT
    is a licence to serve such a render without a login; see the second bullet
    above.

    Args:
        held: :func:`tier_privileges` for the render.
        baseline: :func:`tier_privileges` for the deployment it belongs to —
            the host profile's own ``config:`` block, or the deploy
            ``config.yml`` that the renders beside.

    Returns:
        The subset of ``held`` the baseline does not also grant, in ``held``'s
        order; empty when the render is no more privileged than its base.
    """
    return tuple(privilege for privilege in held if privilege not in baseline)


def privilege_phrase(privileges: Sequence[str]) -> str:
    """Join names into the noun phrase a guard message reads with.

    The one joiner every guard message uses, here and in the belt in
    :mod:`~osprey.deployment.web_terminals.lint`, so a sentence built there
    reads like the ones built here.

    An empty sequence joins to the empty string rather than raising. No caller
    reaches this with nothing to name — every one of them is inside a branch
    that already found something — but a message helper that raises would turn
    a guard's own bug into an ``IndexError`` from the middle of a build, which
    is a worse way to learn about it than a sentence with a gap in it.
    """
    if not privileges:
        return ""
    if len(privileges) == 1:
        return privileges[0]
    return ", ".join(privileges[:-1]) + " and " + privileges[-1]


def privileged_default_tier_problem(default_tier: Any, privileges: Sequence[str]) -> str | None:
    """Why ``default_tier`` must not be this render, or ``None`` if it may.

    ``default_tier`` is what every roster entry without a ``tier:`` key of
    its own inherits — including entries added later, by someone who never read
    this key. So the default is the tier the deployment hands out by accident,
    and it has to be the one it is safe to hand out by accident.

    **Baseline-RELATIVE, unlike the shared-card rule beside it.** Callers pass
    :func:`privileges_beyond_baseline`' output here and
    :func:`tier_privileges`' raw output to
    :func:`shared_card_privileged_problems`, and the asymmetry is the point:
    this key is inherited rather than chosen. A profile written before the
    base tier's deny floor existed has a ``default_tier`` nobody picked as a
    privilege grant, and no unprivileged tier to be re-pointed at, so refusing
    it would refuse the migration itself. ``access`` is the opposite — typed
    deliberately, on one entry, about that entry — so it is judged on
    what the render actually holds. See :func:`privileges_beyond_baseline`
    for the full statement of the split.

    Args:
        default_tier: ``modules.web_terminals.default_tier``. A missing or
            non-string value is not this check's to report (lint's
            ``unknown_default_tier`` owns that), so it yields ``None``.
        privileges: What that render holds BEYOND its deployment's baseline —
            :func:`privileges_beyond_baseline`' output, not
            :func:`tier_privileges`'.

    Returns:
        A message naming the render, what it holds and the remedy, or ``None``.
    """
    if not isinstance(default_tier, str) or not default_tier or not privileges:
        return None
    return (
        f"modules.web_terminals.default_tier {default_tier!r} resolves to a render "
        f"that holds {privilege_phrase(privileges)}. Every roster entry with no tier: key of "
        f"its own inherits the default, so a privileged default is handed to every terminal "
        f"that forgets to name a tier — including ones added later. Point default_tier at "
        f"a render that holds none of them (the bundled stack's {UNPRIVILEGED_TIER_EXAMPLE!r}), "
        f"and name {default_tier!r} on the individual entries that need it"
    )


def auth_is_enforced(web_terminals: Any) -> bool:
    """Whether this deployment puts a login wall in front of its terminals at all.

    The render context's ``walled`` boolean: ``password``/``oidc`` stand a
    sidecar in front of the roster; ``token`` (the default, and what an absent
    ``auth`` stanza means) and ``none`` (open) do not, so under either no entry
    has a login — which is why the privileged-render guards report the
    deployment-wide exposure (:func:`deployment_wide_privileged_exposure_problems`)
    there instead of a per-entry one.

    Routed through render's parsed auth context rather than re-reading
    ``auth.method`` here, so the guards cannot disagree with the nginx seam
    about whether a stanza produces a wall. That function raises on exactly one
    input — a method name that does not exist — which lint reports on its own;
    a config already being rejected for that gets no second, confused finding
    from this guard, so it is read as walled.
    """
    from osprey.deployment.web_terminals.render import _auth_tls_context

    try:
        context = _auth_tls_context(as_dict(web_terminals))
    except ValueError:
        return True
    return bool(context["walled"])


def _privileged_entries(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_tier: Mapping[str, Sequence[str]],
) -> list[tuple[Mapping[str, Any], str, Sequence[str]]]:
    """``(entry, render, privileges)`` for every entry that can edit the deployment.

    A render missing from ``privileges_by_tier`` contributes nothing, on the
    same terms as :func:`_render_configs`: a config that could not be read is
    not evidence of a privilege, and every way of failing to read one is already
    reported by the check that owns it.
    """
    entries: list[tuple[Mapping[str, Any], str, Sequence[str]]] = []
    for entry in resolved_entries:
        tier = entry.get("render")
        if not isinstance(tier, str):
            continue
        privileges = privileges_by_tier.get(tier) or ()
        if privileges:
            entries.append((entry, tier, privileges))
    return entries


def deployment_wide_privileged_exposure_problems(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_tier: Mapping[str, Sequence[str]],
) -> list[str]:
    """Every privileged terminal of a deployment that stands no login wall.

    With no authentication configured every privileged terminal is served to
    whoever can reach the deployment — the real exposure, and worth naming,
    which is why this exists at all.

    It is reported ADVISORILY where the shared-card rule is an error, and the
    difference is what the config *claims*. An ``access`` set wider than the
    entry's own user singles that entry out as open to somebody else: the
    author asked for that and can act on the finding. ``auth.method: none`` is a whole-deployment posture — the
    shipped default, and a deliberate one for a stack bound to a loopback
    address — and failing a build over it would reject deployments that were
    never exposed to anyone, with no remedy but to configure authentication
    they do not need.

    Callers reach this only when :func:`auth_is_enforced` is ``False``, and when
    they do they use it INSTEAD of :func:`shared_card_privileged_problems`, not
    alongside it: with no wall standing there is no authenticated identity for
    sharing to widen, so reporting both would name one exposure twice and
    disagree with itself about severity.

    Args:
        resolved_entries: :func:`resolve_cards`' output (or anything with
            the same ``name``/``render`` keys).
        privileges_by_tier: Render name → :func:`tier_privileges` —
            the ABSOLUTE answer; see :func:`_privileged_entries` for what a
            missing render means.

    Returns:
        One message per privileged entry, in roster order.
    """
    problems: list[str] = []
    for entry, tier, privileges in _privileged_entries(resolved_entries, privileges_by_tier):
        name = entry.get("name")
        problems.append(
            f"modules.web_terminals user {name!r} resolves to render {tier!r}, which "
            f"holds {privilege_phrase(privileges)}, and modules.web_terminals.auth.method "
            f"puts no login wall in front of it — so anyone who can reach this deployment "
            f"edits it. Turn authentication on (auth.method: password or oidc), or point "
            f"{name!r} at a "
            f"render that holds none of them (the bundled stack's "
            f"{UNPRIVILEGED_TIER_EXAMPLE!r})"
        )
    return problems


def access_phrase(entry: Mapping[str, Any]) -> str:
    """The entry's authored ``access``, spelled for a message rich will render.

    ``osprey build`` renders these refusals through rich, which eats ``[...]``,
    so a principal list is spelled as its members rather than as its repr. That
    costs the one cue the brackets carried — whether the file holds a list or a
    scalar — so the phrase says which: ``access members 'user:a', 'domain:b'``
    for a list, ``access: 'any'`` for a shorthand. An operator reading either
    one back finds the same shape in their file.

    Args:
        entry: A roster entry carrying its authored ``access`` value.

    Returns:
        The whole phrase, including the ``access`` word, so every caller names
        the key the same way.
    """
    authored = entry.get("access")
    if isinstance(authored, list):
        return "access members " + ", ".join(repr(member) for member in authored)
    return f"access: {authored!r}"


def shared_card_privileged_problems(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_tier: Mapping[str, Sequence[str]],
) -> list[str]:
    """Every SHARED roster entry whose render can edit the deployment.

    An ``access`` set wider than the entry's own user opens that card to
    somebody else: every authenticated login under ``any``, or the identities a
    ``user:``/``domain:`` list names, may mint a session for it. The line is
    drawn once, by :func:`entry_is_shared` — shared is "the resolved set is
    anything other than the entry's own user" — so this rule generalizes with
    the vocabulary instead of naming one spelling of it. On an unprivileged
    render that is the feature — one control-room terminal everybody on shift
    opens. On a privileged one it undoes the tier split the roster declared:
    a deployment-editing door (:data:`DEPLOYMENT_EDIT_DOORS`) was opened for
    named people behind their own cards, and the one key hands it to somebody
    who was never given one.

    **Judged on what the render ABSOLUTELY holds**, not on what it holds
    beyond its deployment's baseline: sharing is an authored claim about one
    entry, made deliberately, and a deployment that floors no door hands every
    one of them through the shared card — the widest version of this
    exposure, not an exempt one. See :func:`privileges_beyond_baseline` for the
    rules that read the relative answer.

    Args:
        resolved_entries: :func:`resolve_cards`' output (or anything with
            the same ``name``/``render``/``access`` keys).
        privileges_by_tier: Render name → :func:`tier_privileges` — the
            ABSOLUTE answer; see :func:`_privileged_entries` for what a missing
            render means.

    Returns:
        One message per offending entry, in roster order, each naming the user,
        the render, what it holds and the remedy.
    """
    problems: list[str] = []
    for entry, tier, privileges in _privileged_entries(resolved_entries, privileges_by_tier):
        if not entry_is_shared(dict(entry)):
            continue
        name = entry.get("name")
        problems.append(
            f"modules.web_terminals user {name!r} is a shared card "
            f"({access_phrase(entry)}), but resolves to render {tier!r}, which holds "
            f"{privilege_phrase(privileges)}. Somebody other than that entry's own user "
            f"may open that terminal, so more people than its owner edit this deployment. "
            f"Set access: own for {name!r} — or drop the key, since owner-only is the "
            f"default — or point {name!r} at a render that holds none of them (the bundled "
            f"stack's {UNPRIVILEGED_TIER_EXAMPLE!r})"
        )
    return problems


CONTROL_IDENTITY_COLLISION_CODE = "web_terminals.control_identity_collision"
"""The code every :func:`control_identity_collision_warnings` WARN carries."""


def _roster_entry_name(user: Any) -> str | None:
    """A raw roster entry's name, uniformly for either roster form.

    A bare string is its own name. An object-form entry contributes its
    ``name`` field, but only when it is a string; a malformed dict without a
    str name yields ``None``, and the readers that use this skip it (name
    well-formedness is lint's own check, not theirs).
    """
    if isinstance(user, str):
        return user
    if isinstance(user, dict):
        name = user.get("name")
        if isinstance(name, str):
            return name
    return None


def _subject_key(user: Mapping[str, Any], fold: bool) -> str | None:
    """The entry's ``oidc_subject`` as the sidecar compares it, or ``None`` when unset.

    Stripped of surrounding whitespace, and case-folded when the configured
    claim is matched case-insensitively (:data:`CASE_INSENSITIVE_CLAIMS`).
    """
    subject = user.get("oidc_subject")
    if not isinstance(subject, str) or not subject.strip():
        return None
    key = subject.strip()
    return key.casefold() if fold else key


def _valid_control_identity_entries(raw_users: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """``(name, control_identity, entry)`` for every named entry carrying a valid value.

    The cross-entry rules compare only values that could reach ``/etc/passwd``;
    an invalid value is already an ERROR of its own, and a comparison over it
    would only pile a second finding onto the same typo.
    """
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for user in raw_users if isinstance(raw_users, list) else []:
        if not isinstance(user, dict) or "control_identity" not in user:
            continue
        name = _roster_entry_name(user)
        value = user.get("control_identity")
        if name is None or not isinstance(value, str):
            continue
        try:
            validate_identity(value, allow_service=False)
        except ValueError:
            continue
        entries.append((name, value, user))
    return entries


def _control_identity_groups(raw_users: Any, fold: bool) -> list[tuple[str, list[str], bool, bool]]:
    """Every control identity carried by more than one entry.

    Returns ``(value, entry names, two_people, all_subjects)`` in
    first-appearance order, where ``two_people`` is true when at least two of
    the entries carry ``oidc_subject`` values that still differ after the
    claim-dependent fold, and ``all_subjects`` is true when every entry carries
    one.
    """
    by_value: dict[str, list[tuple[str, str | None]]] = {}
    for name, value, user in _valid_control_identity_entries(raw_users):
        by_value.setdefault(value, []).append((name, _subject_key(user, fold)))
    groups: list[tuple[str, list[str], bool, bool]] = []
    for value, members in by_value.items():
        if len(members) < 2:
            continue
        subjects = [subject for _, subject in members if subject is not None]
        groups.append(
            (
                value,
                [name for name, _ in members],
                len(set(subjects)) > 1,
                len(subjects) == len(members),
            )
        )
    return groups


def control_identity_problems(raw_users: Any, *, claim: str = "") -> list[tuple[str, str]]:
    """Every roster ``control_identity`` that must be refused, as ``(code, message)``.

    ``control_identity`` is the account name the control system sees a card's
    writes arrive under, so three shapes are refused outright:

    * ``web_terminals.invalid_user_control_identity`` — a value
      :func:`~osprey.deployment.control_identity.validate_identity` refuses:
      not a string, outside the passwd-safe charset, reserved for OSPREY, or
      already an account in the base image.
    * ``web_terminals.shared_card_control_identity`` — any value on a shared
      card. Whoever opens a shared card writes through it, so one fixed name
      would attribute every opener's writes to one person.
    * ``web_terminals.duplicate_control_identity`` — one value carried by
      entries whose ``oidc_subject`` values name two different people after
      the claim-dependent case fold. The control system could no longer tell
      their writes apart. A value shared with a missing subject, or by one
      person holding two cards, is a WARN instead
      (:func:`control_identity_collision_warnings`).

    Read from the raw roster, so both roster forms are accepted and an entry
    whose ``access`` cannot be read is judged as not shared (see
    :func:`_is_shared_entry`). Messages name entries and the identity, never
    a subject value.

    Args:
        raw_users: ``modules.web_terminals.users`` as authored.
        claim: The configured OIDC claim; one in :data:`CASE_INSENSITIVE_CLAIMS`
            folds subject case before comparing. Empty means the sidecar
            default, which compares exactly.

    Returns:
        ``(code, message)`` per problem, in roster order: per-entry problems
        first, then one per duplicated value.
    """
    fold = claim in CASE_INSENSITIVE_CLAIMS
    problems: list[tuple[str, str]] = []
    for user in raw_users if isinstance(raw_users, list) else []:
        if not isinstance(user, dict) or "control_identity" not in user:
            continue
        name = user.get("name", user)
        value = user.get("control_identity")
        try:
            validate_identity(value, allow_service=False)
        except ValueError as exc:
            problems.append(
                (
                    "web_terminals.invalid_user_control_identity",
                    f"modules.web_terminals.users entry {name!r} sets an unusable "
                    f"control_identity: {exc}. Write the account name the control "
                    "system should see this card's writes arrive under, or drop the key",
                )
            )
        if _is_shared_entry(user):
            problems.append(
                (
                    "web_terminals.shared_card_control_identity",
                    f"modules.web_terminals.users entry {name!r} is a shared card "
                    f"({access_phrase(user)}) and sets control_identity. Everybody who "
                    "opens a shared card writes through it, so one fixed name would "
                    "attribute every opener's writes to one person. Drop control_identity "
                    f"from {name!r}, or make it owner-only",
                )
            )
    for value, names, two_people, _all_subjects in _control_identity_groups(raw_users, fold):
        if not two_people:
            continue
        problems.append(
            (
                "web_terminals.duplicate_control_identity",
                f"modules.web_terminals.users entries {names} all set control_identity "
                f"{value!r} but carry oidc_subject values naming different people, so "
                "the control system could not tell their writes apart. Give each "
                "person their own control_identity",
            )
        )
    return problems


def control_identity_collision_warnings(
    raw_users: Any, *, claim: str = ""
) -> list[tuple[str, str]]:
    """Every roster ``control_identity`` collision short of a refusal, as ``(code, message)``.

    Each is a value that may credit one person's writes to somebody else, but
    that the roster alone cannot prove wrong:

    * a value equal to ANOTHER entry's roster name;
    * a value equal to the email local part of ANOTHER entry's
      ``oidc_subject``;
    * a value carried by more than one entry that
      :func:`control_identity_problems` does not refuse — a subject is
      missing, so the roster cannot say whether one person holds the cards,
      or every subject agrees after the claim-dependent fold.

    The first two are skipped when both entries carry the same subject after
    the fold: that is one person holding two cards, and their own name is the
    right one. Invalid values are left to the ERROR and never compared.

    Args:
        raw_users: ``modules.web_terminals.users`` as authored.
        claim: The configured OIDC claim, as for
            :func:`control_identity_problems`.

    Returns:
        ``(CONTROL_IDENTITY_COLLISION_CODE, message)`` per collision, in roster
        order, then one per shared value.
    """
    fold = claim in CASE_INSENSITIVE_CLAIMS
    users = raw_users if isinstance(raw_users, list) else []
    others: list[tuple[str, str | None, str | None]] = []
    for user in users:
        name = _roster_entry_name(user)
        if name is None:
            continue
        subject = _subject_key(user, fold) if isinstance(user, dict) else None
        raw_subject = user.get("oidc_subject") if isinstance(user, dict) else None
        local = None
        if isinstance(raw_subject, str) and "@" in raw_subject:
            local = raw_subject.strip().rsplit("@", 1)[0].casefold() or None
        others.append((name, subject, local))

    warnings: list[tuple[str, str]] = []
    for name, value, user in _valid_control_identity_entries(users):
        own_subject = _subject_key(user, fold)
        for other_name, other_subject, other_local in others:
            if other_name == name:
                continue
            if own_subject is not None and own_subject == other_subject:
                continue
            if value == other_name:
                warnings.append(
                    (
                        CONTROL_IDENTITY_COLLISION_CODE,
                        f"modules.web_terminals.users entry {name!r} sets control_identity "
                        f"{value!r}, which is the roster name of entry {other_name!r}. The "
                        f"control system would credit {name!r}'s writes to that user. "
                        "Pick a name that belongs to this card's owner",
                    )
                )
            elif value == other_local:
                warnings.append(
                    (
                        CONTROL_IDENTITY_COLLISION_CODE,
                        f"modules.web_terminals.users entry {name!r} sets control_identity "
                        f"{value!r}, which matches the mailbox name in the oidc_subject of "
                        f"entry {other_name!r}. The control system would credit "
                        f"{name!r}'s writes to that person. Pick a name that belongs to "
                        "this card's owner",
                    )
                )
    for value, names, two_people, all_subjects in _control_identity_groups(users, fold):
        if two_people:
            continue
        if all_subjects:
            reason = "Every entry carries the same oidc_subject, so this reads as one person"
        else:
            reason = "Not every entry carries an oidc_subject, so the roster cannot say whether"
            reason += " one person holds them"
        warnings.append(
            (
                CONTROL_IDENTITY_COLLISION_CODE,
                f"modules.web_terminals.users entries {names} all set control_identity "
                f"{value!r}. {reason}. If more than one person opens these cards, give "
                "each person their own control_identity",
            )
        )
    return warnings


def roster_user_names(web_terminals: Any) -> list[str]:
    """The web-terminal user names a ``modules.web_terminals`` subtree declares.

    The whole rule in one place: a module that is not switched on has no roster
    at all, and the entries of one that is are read through
    :func:`normalize_users` rather than off the raw list. Both matter to callers
    who then create directories per user — ``osprey init`` seeding a
    per-user context slot and the build copying into it must agree on the roster
    down to the last name, or the build looks for a directory nobody made.

    Deliberately takes the *subtree*, not a whole config: its two callers reach
    it by different routes (a profile's ``config:`` block, which needs
    ``effective_web_terminals`` to fold the dotted keys; a built project's
    ``config.yml``, which is already nested), and only what they do with it
    afterwards is shared.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree. Anything that is
            not a mapping is an absent module, so the roster is empty.

    Returns:
        User names in roster order; empty for a disabled or absent module.
    """
    subtree = as_dict(web_terminals)
    if not subtree.get("enabled"):
        return []
    return [entry["name"] for entry in roster_people(subtree)]


def roster_role_by_name(web_terminals: Any) -> dict[str, str]:
    """The static role each roster entry declares, keyed by username.

    A roster ``role:`` reaches the deployment twice, and the two readings are
    deliberately kept apart. :func:`resolve_cards` *consumes* the role: it
    resolves through :func:`effective_tier` into the render a terminal is
    built from, and its output carries no ``role`` key at all — which is what
    makes a role-only roster resolve field for field like the ``tier:``-pinned
    roster it stands for. This accessor is the other reading: the role NAME
    itself, which the auth sidecar hands a password session as the privilege
    that login was granted. A ``tier:`` pin conveys none of it, so the two
    are separate facts and are read separately.

    Keyed by name rather than returned in roster order because that is how the
    render consumes it — one lookup per already-resolved entry — and because
    :func:`_render_ref_by_name` already keys the same roster the same way.

    Unlike :func:`roster_user_names` this does *not* consult ``enabled``: it is
    joined against :func:`resolve_cards`' output, which does not either, and
    a table that emptied itself on a key its partner ignores would answer "no
    role" for a roster that has one.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree. Anything that is
            not a mapping has no roster, so the table is empty.

    Returns:
        ``{username: role}`` for the entries declaring a non-empty ``role:``. An
        entry declaring none is *absent*, never empty-string keyed, on the same
        terms :func:`normalize_users` drops it: an empty role is "no privileges",
        and storing it as a value would invite a reader to treat it as a role
        named ``""``.
    """
    return {
        entry["name"]: entry["role"] for entry in roster_people(web_terminals) if entry.get("role")
    }


def effective_image_source(web_terminals: dict[str, Any]) -> str:
    """Coerce ``modules.web_terminals.image_source`` to one of the two real modes.

    ``"local"`` only for the exact literal ``"local"``; every other value
    (unset, ``"registry"``, or an invalid literal) resolves to ``"registry"``.
    This is the single source of that coercion — :func:`resolve_cards`,
    render, and lint all route through it so an invalid literal is treated
    identically everywhere (and reported once, separately, by lint's
    ``_check_unknown_image_source``).
    """
    return "local" if web_terminals.get("image_source") == "local" else "registry"


def resolve_image_tag(web_terminals: dict[str, Any]) -> str:
    """Resolve ``modules.web_terminals.image_tag`` to a literal registry tag.

    Defaults to ``"latest"`` (unset or non-string), then expands any ``${VAR}``
    / ``$VAR`` references against the **process environment at render time**, so
    the rendered compose artifact self-carries a fixed literal tag rather than a
    compose-side ``${...}`` the runtime would re-interpolate at ``up`` time. A
    referenced variable that is unset expands to the empty string (unlike
    :func:`os.path.expandvars`, which would leave the reference in place and leak
    a ``${...}`` into the output); an image_tag that resolves entirely empty is a
    lint warning (``_check_image_tag_empty``), never a silent bad tag.

    This is the single source of that resolution — :func:`resolve_cards` and
    lint both route through it so the tag is read and expanded identically.
    """
    raw = web_terminals.get("image_tag")
    if not isinstance(raw, str):
        raw = "latest"
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), raw)


def _render_ref_by_name(
    raw_users: Any, authorization: Mapping[str, str], *, strict: bool = True
) -> dict[str, str]:
    """Recover each roster entry's own render reference from the raw roster.

    normalize_users() drops both the `render` and the `role` field off each
    surviving entry; recover the render they resolve to here, keyed by name
    (the same key every other per-user artifact in this module — compose service
    names, volume names — is keyed by), since normalize_users()'s own
    index-freezing contract is orthogonal to render resolution.

    Routed through :func:`effective_tier`, so a `role:` binding and a
    `tier:` pin arrive at the same answer. ``default_tier`` is deliberately
    not applied here — :func:`resolve_cards` applies it to the entries this
    map has nothing for, exactly as it did when this returned only explicit
    pins.
    """
    refs: dict[str, str] = {}
    if isinstance(raw_users, list):
        for raw_entry in raw_users:
            if isinstance(raw_entry, dict):
                name = raw_entry.get("name")
                tier = effective_tier(raw_entry, authorization, None, strict=strict)
                if isinstance(name, str) and tier:
                    refs[name] = tier
    return refs


#: The refusal both the lint (``web_terminals.registry_mode_missing_url``) and
#: the render raise for registry mode with no ``registry.url``.
REGISTRY_MODE_MISSING_URL: str = (
    "modules.web_terminals.image_source is 'registry' (the default) but registry.url is not "
    "set, so no web-terminal image can be named; set the profile's deploy.registry.url, or "
    "registry.url in its config: block"
)

#: The refusal the render, the deploy preflight, the render image build and
#: the lint (``web_terminals.requires_catalog``) all raise for web terminals
#: with no render catalog or no ``default_tier``.
RENDER_CATALOG_REQUIRED: str = (
    "web terminals need a modules.web_terminals.renders catalog and a default_tier "
    "naming one of its entries; each terminal runs its render's project at /app/<project>"
)


def configured_registry_url(registry_cfg: Any) -> str:
    """Return the top-level ``registry.url`` every registry-mode image is named under.

    Args:
        registry_cfg: The top-level ``registry`` section, in any shape.

    Returns:
        ``registry_cfg["url"]`` when the section is a mapping and the value is a
        string; empty when the section, the key or a string value is missing.
    """
    if isinstance(registry_cfg, dict):
        url = registry_cfg.get("url")
        if isinstance(url, str):
            return url
    return ""


def resolve_cards(
    web_terminals: dict[str, Any],
    registry_cfg: dict[str, Any],
    *,
    strict: bool = True,
    project_name: str | None = None,
    app_titles: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve every card the roster runs into its image/project identity.

    Layered on :func:`roster_entries`: the people (:func:`normalize_users`'s
    output, whose signature and drop-don't-raise contract stay untouched), then
    one derived entry per shared app. For a person this function re-derives the
    tier reference (which :func:`normalize_users` discards) from the raw roster
    by matching on ``name``; the people resolve exactly as they did before
    shared apps existed.

    **A shared app** resolves to its own render, ``build/<project>-<app>``,
    named by :func:`~osprey.deployment.web_terminals.render_naming.render_project`
    from *project_name*. Without *project_name*, the render's ``project`` is
    read from the build's catalog entry for the app (``renders.<app>``).
    The app card keeps the derived entry's ``kind``,
    ``app``, ``display_name`` (the app's title, else its name) and
    ``access: any``; its ``render`` is the app's name, the render it runs. It
    declares no services of its own: the app render dials the host's, like
    every other card. A name that is both a tier and a shared app is refused
    (:func:`~osprey.deployment.web_terminals.render_naming.render_kind_conflict`),
    in either direction: a person, role or ``default_tier`` naming an app, or
    an app named like a tier the roster uses.

    A ``render`` reference resolves through :func:`effective_tier`: the
    entry's ``role:`` binding, else its own ``tier:`` key, else
    ``modules.web_terminals.default_tier``, else ``None``, which only a
    lenient caller sees: the strict path refuses a roster with no
    ``default_tier``. Under ``strict`` an unresolvable
    binding — a role naming no declared role, or an entry carrying both a role
    and a render — raises :class:`UnresolvedRoleError`; the lenient path
    degrades it to the pre-roles answer, which lint pairs with its own ERROR. Resolving against the catalog
    (``modules.web_terminals.renders.<name>: {project, project_path,
    build_profile}``) then follows these naming rules. In registry mode every
    image carries the tag :func:`resolve_image_tag` resolves from
    ``modules.web_terminals.image_tag`` (``<tag>`` below, default ``latest``);
    local ``:local`` images are unaffected by that field:

    * **Default tier** (resolved ``render`` equals ``default_tier``, and
      a catalog entry exists for it): registry mode uses the un-suffixed
      ``<registry_url>/web-terminal:<tag>`` image; local mode builds
      ``<render.project>:local`` like every other render.
      ``container_project_dir`` is ``/app/<project>`` from the catalog entry,
      exactly as for any other tier: the image is built FROM that project.
    * **Non-default tier**: registry mode uses
      ``<registry_url>/web-terminal-<render>:<tag>``; local mode uses
      ``<render.project>:local`` (same rule as the default tier) — the
      render image is tagged by its render alone, since the render name
      contributes nothing to the image beyond the tag;
      ``container_project_dir`` is derived from the render's own
      ``/app/<project>``.

    The resolution stays total: an empty ``registry_url`` still yields a
    leading-slash image name, and :func:`render_web_terminals` is where registry
    mode without one is refused.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section
            (``users``, ``shared``, ``renders``, ``default_tier``,
            ``image_source``).
        registry_cfg: The already-dict-coerced top-level ``registry`` section
            (only ``url`` is read).
        project_name: The deployment's ``project_name``, which names each
            shared app's render. ``None`` reads the render off the catalog.
        app_titles: The app name -> title mapping (see :func:`app_titles_in`),
            passed to :func:`roster_entries` for each app card's
            ``display_name``.
        strict: When ``True`` (render/build/seed callers), a config with no
            ``renders`` catalog or no default tier (in any image
            source), a render reference naming a catalog entry that doesn't
            exist, an in-effect catalog entry with no ``project``, a shared
            app whose render cannot be named, or a name that is both a tier
            and a shared app raises ``ValueError``. When ``False`` (lifecycle verbs, lint, reports), the
            same conditions yield the unresolved entry — ``image``,
            ``project`` and ``container_project_dir`` ``None`` — so a
            stale/bad render reference never blocks
            ``decommission``/``prune``/``nuke``.

    Returns:
        One ``{"name", "index", "render", "image", "project",
        "container_project_dir", "extra_mounts", "seed_base"}`` dict per
        :func:`roster_entries` entry, in the same order: each surviving
        :func:`normalize_users` entry, then each shared app, whose dict adds
        ``"kind": "app"``, ``"app"``, ``"display_name"`` and ``"access": "any"``. ``render``
        is the resolved catalog key; it is ``None`` only on the lenient path.
        ``extra_mounts`` is the render's ``extra_mounts`` list
        (compose volume strings applied to every user of that render),
        defaulting to ``[]``. ``seed_base`` is the catalog entry's
        ``seed_base`` (a bool; anything else is defensively coerced to
        ``True``), and ``True`` for an unresolved lenient entry — it controls whether the shared base context is prepended when
        seeding this entry's ``CLAUDE.md``. Optional ``"display_name"``,
        ``"theme"`` and ``"tour"`` keys are added — carried through from
        :func:`normalize_users` — only when the entry declared a non-empty string
        one (render emits them as ``OSPREY_WEB_APP_NAME``, ``OSPREY_WEB_THEME``
        and ``OSPREY_WEB_TOUR``); each is omitted entirely otherwise, so a roster
        declaring none resolves byte-identically to before these fields
        existed. An optional ``"oidc_subject"`` key rides through on the same
        terms, so the auth sidecar's roster→identity mapping is read off the same
        resolved entry as everything else rather than re-derived from the raw
        roster. An optional ``"control_identity"`` key rides through on the
        same terms, for the compose render's ``OSPREY_CONTROL_IDENTITY``.
        The authored ``"access"`` value rides through likewise, verbatim
        and whole — present only when the roster entry admits somebody beyond
        its own user (see :func:`entry_is_shared`), so both the render and the
        guards that read resolved entries resolve the same principals the
        authored roster does.

    Raises:
        ValueError: See ``strict`` above.
    """
    from osprey.deployment.web_terminals.render_naming import (
        RENDER_KIND_APP,
        RENDER_KIND_TIER,
        render_kind_conflict,
    )

    entries = roster_entries(web_terminals, strict=strict, app_titles=app_titles)
    normalized = [entry for entry in entries if not is_app_entry(entry)]
    apps = [entry for entry in entries if is_app_entry(entry)]
    app_names = frozenset(entry["name"] for entry in apps)

    renders_raw = web_terminals.get("renders")
    renders_catalog: dict[str, Any] = renders_raw if isinstance(renders_raw, dict) else {}
    # A person's access level is a tier: only tier renders are a people
    # catalog, and every app render joins the tier-or-app refusal below.
    tier_renders = {
        render_name: render
        for render_name, render in renders_catalog.items()
        if isinstance(render, dict) and render.get("kind") == RENDER_KIND_TIER
    }
    app_names = app_names | frozenset(
        render_name
        for render_name, render in renders_catalog.items()
        if isinstance(render, dict) and render.get("kind") == RENDER_KIND_APP
    )
    candidate = web_terminals.get("default_tier")
    default_tier_name = candidate if isinstance(candidate, str) and candidate else None

    image_source = effective_image_source(web_terminals)
    image_tag = resolve_image_tag(web_terminals)

    registry_url = configured_registry_url(registry_cfg)

    # The role table behind every entry's binding. Under `strict` an incoherent
    # `authorization` stanza stops the render here rather than resolving a
    # roster against a table that was never built; the lenient callers (lifecycle
    # verbs, lint) carry on with an empty one, and lint reports the stanza
    # itself as `web_terminals.invalid_authorization`.
    try:
        authorization_roles = resolve_authorization_roles(web_terminals)
    except ValueError:
        if strict:
            raise
        authorization_roles = {}

    render_ref_by_name = _render_ref_by_name(
        web_terminals.get("users"), authorization_roles, strict=strict
    )

    if strict and (not tier_renders or default_tier_name is None):
        raise ValueError(RENDER_CATALOG_REQUIRED)

    default_tier_image = f"{registry_url}/web-terminal:{image_tag}"

    def _with_optional_fields(entry: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        """Attach the optional per-user fields to a resolved entry, mirroring
        render.py's conditional-``sublabel`` convention: a key is present only
        for a non-empty string, so a roster declaring none leaves the entry
        byte-identical to a resolution from before these fields existed."""
        for field in ("display_name", "theme", "tour", "oidc_subject", "control_identity"):
            value = source.get(field)
            if isinstance(value, str) and value:
                entry[field] = value
        # The authored `access` value rides through on the same
        # present-only-when-set terms. It is carried whole, not reduced to a
        # marker: `normalize_users` has already validated it and dropped every
        # owner-only spelling, so anything still here admits somebody beyond
        # this entry's own user and must reach the render — and the guards that
        # read resolved entries — intact.
        access = source.get("access")
        if access is not None:
            entry["access"] = access
        return entry

    def _unresolved_entry(
        name: str, index: int, tier: str | None, source: dict[str, Any]
    ) -> dict[str, Any]:
        """The lenient answer for an entry that does not resolve.

        A lenient caller (lifecycle verbs, lint, reports) gets every roster
        name back, with ``render`` carried for logging. What the entry runs is
        unknown, so ``image``, ``project`` and ``container_project_dir`` are
        ``None`` instead of naming a directory no image has."""
        return _with_optional_fields(
            {
                "name": name,
                "index": index,
                "render": tier,
                "image": None,
                "project": None,
                "container_project_dir": None,
                "extra_mounts": [],
                "seed_base": True,
            },
            source,
        )

    # Every name the roster uses as a tier, for the app half of the
    # tier-or-app refusal: each person's binding, the default, every role's
    # target, and each tier the build's render projection lists.
    tier_names: set[str] = {
        render_ref_by_name.get(entry["name"]) or default_tier_name or "" for entry in normalized
    }
    if default_tier_name is not None:
        tier_names.add(default_tier_name)
    tier_names.update(target for target in authorization_roles.values() if target)
    tier_names.update(
        render_name
        for render_name, render in renders_catalog.items()
        if isinstance(render, dict) and render.get("kind") == RENDER_KIND_TIER
    )
    tier_names.discard("")

    resolved: list[dict[str, Any]] = []
    for entry in normalized:
        name = entry["name"]
        index = entry["index"]
        render_ref = render_ref_by_name.get(name) or default_tier_name

        if render_ref is not None:
            conflict = render_kind_conflict(render_ref, RENDER_KIND_TIER, apps=app_names)
            if conflict is not None:
                if strict:
                    raise ValueError(f"user {name!r}: {conflict}")
                resolved.append(_unresolved_entry(name, index, render_ref, entry))
                continue

        catalog_entry = tier_renders.get(render_ref) if render_ref is not None else None
        if not isinstance(catalog_entry, dict):
            if strict:
                raise ValueError(
                    f"user {name!r} references render {render_ref!r}, which has "
                    "no entry in modules.web_terminals.renders"
                )
            # Lenient: a stale or bad reference never blocks a lifecycle verb.
            resolved.append(_unresolved_entry(name, index, render_ref, entry))
            continue

        project = catalog_entry.get("project")
        if not isinstance(project, str) or not project:
            if strict:
                raise ValueError(
                    f"modules.web_terminals.renders[{render_ref!r}] has no project; "
                    "osprey build writes it for a render with a build_profile, and a "
                    "render whose image is built elsewhere states the project its "
                    "image was built from"
                )
            resolved.append(_unresolved_entry(name, index, render_ref, entry))
            continue

        # Render-level host mounts, applied to every user of this render. A
        # non-list drops to []; individual non-string/empty entries are dropped
        # (well-formedness — the colon-part syntax — is lint's job).
        extra_mounts_raw = catalog_entry.get("extra_mounts")
        extra_mounts = (
            [mount for mount in extra_mounts_raw if isinstance(mount, str) and mount]
            if isinstance(extra_mounts_raw, list)
            else []
        )

        # seed_base: whether this render's users get the shared base context
        # prepended ahead of their own extra context at seed time. Defaults to
        # True (always prepend); a non-bool value is a
        # config typo that lint reports separately, so coerce it back to the
        # safe default here rather than propagating garbage.
        seed_base = catalog_entry.get("seed_base")
        if not isinstance(seed_base, bool):
            seed_base = True

        is_default = render_ref == default_tier_name

        if image_source == "local":
            # A render image IS its render — the render name never reaches
            # the build beyond the tag — so the image is tagged by its project
            # alone. Lint enforces the distinctness this relies on
            # (`render_project_collision` / `render_project_shadows_worker_image`):
            # no two renders may share a `project` across different renders,
            # and none may take the deployment's own name, which the dispatch
            # worker's `<project>:local` tag occupies.
            image = f"{project}:local"
        elif is_default:
            image = default_tier_image
        else:
            image = f"{registry_url}/web-terminal-{render_ref}:{image_tag}"

        container_project_dir = f"/app/{project}"

        entry_resolved = _with_optional_fields(
            {
                "name": name,
                "index": index,
                "render": render_ref,
                "image": image,
                "project": project,
                "container_project_dir": container_project_dir,
                "extra_mounts": extra_mounts,
                "seed_base": seed_base,
            },
            entry,
        )
        resolved.append(entry_resolved)

    resolved.extend(
        _resolve_app_card(
            app_entry,
            project_name=project_name,
            catalog_entry=_app_catalog_entry(app_entry["name"], renders_catalog),
            tier_names=frozenset(tier_names),
            app_names=app_names,
            image_source=image_source,
            registry_url=registry_url,
            image_tag=image_tag,
            strict=strict,
        )
        for app_entry in apps
    )

    return resolved


def _app_catalog_entry(app: str, renders_catalog: Mapping[str, Any]) -> dict[str, Any]:
    """The build's catalog entry for one shared app's render, or ``{}``."""
    found = renders_catalog.get(app)
    return found if isinstance(found, dict) else {}


def _resolve_app_card(
    app_entry: dict[str, Any],
    *,
    project_name: str | None,
    catalog_entry: Mapping[str, Any],
    tier_names: frozenset[str],
    app_names: frozenset[str],
    image_source: str,
    registry_url: str,
    image_tag: str,
    strict: bool,
) -> dict[str, Any]:
    """One derived app entry resolved to the render it runs.

    See :func:`resolve_cards` for the rules. The derived entry's own keys
    (``kind``, ``app``, ``display_name``, ``access``) ride through unchanged;
    the render's identity is added beside them. A lenient caller gets the
    entry with ``image``, ``project`` and ``container_project_dir`` ``None``
    where the render cannot be named.

    Raises:
        ValueError: Under *strict*, when the app is also a tier or its render
            cannot be named.
    """
    from osprey.deployment.web_terminals.render_naming import (
        RENDER_KIND_APP,
        render_kind_conflict,
        render_project,
    )

    name = app_entry["name"]
    card: dict[str, Any] = {
        "kind": app_entry["kind"],
        "app": app_entry["app"],
        "name": name,
        "index": app_entry["index"],
        "render": name,
        "image": None,
        "project": None,
        "container_project_dir": None,
        "extra_mounts": [],
        "seed_base": True,
        "display_name": app_entry["display_name"],
        "access": app_entry["access"],
    }

    problem = render_kind_conflict(name, RENDER_KIND_APP, tiers=tier_names, apps=app_names)
    project: str | None = None
    if problem is None and isinstance(project_name, str) and project_name:
        project, _ = render_project(project_name, name, kind=RENDER_KIND_APP, tiers=tier_names)
    elif problem is None:
        cataloged = catalog_entry.get("project")
        if isinstance(cataloged, str) and cataloged:
            project = cataloged
        else:
            problem = (
                f"shared app {name!r} has no render to run: osprey build renders "
                f"build/<project>-{name} for it; pass the deployment's project_name, "
                "or run `osprey build`"
            )
    if problem is not None:
        if strict:
            raise ValueError(problem)
        return card

    extra_mounts_raw = catalog_entry.get("extra_mounts")
    seed_base = catalog_entry.get("seed_base")
    if image_source == "local":
        image = f"{project}:local"
    else:
        image = f"{registry_url}/web-terminal-{name}:{image_tag}"
    card.update(
        image=image,
        project=project,
        container_project_dir=f"/app/{project}",
        extra_mounts=(
            [mount for mount in extra_mounts_raw if isinstance(mount, str) and mount]
            if isinstance(extra_mounts_raw, list)
            else []
        ),
        seed_base=seed_base if isinstance(seed_base, bool) else True,
    )
    return card


# ---------------------------------------------------------------------------
# Shipped-artifact reads
#
# Everything above answers a question about a render's *config* — its authored
# intent. The two functions below deliberately do not: they read the built
# `.claude/settings.json` artifact instead, and so are kept out of the
# entitlement-predicate cluster above rather than joining it.
# ---------------------------------------------------------------------------


def settings_json_denies(project_dir: Any, tools: Iterable[str]) -> bool:
    """True if ``<project_dir>/.claude/settings.json`` denies every tool in *tools*.

    Reads the **shipped build artifact**, not the ``config.yml`` that produced
    it, and that distinction is the whole point of this function. ``osprey up``
    does not rebuild (rebuilding is a separate operator step), so a render's
    config edited after its last build does not change what its image actually
    ships. A caller that asked the config instead would see a permission set
    that has never been rendered: an operator who removed ``Bash`` from
    ``remove_deny`` would read as safe while the running image still permits the
    shell.

    The scope of that claim is the rendered project directory, which is what
    ``COPY . /app/<project>/`` bakes into the render image — so what this reads
    equals the image's own ``.claude/settings.json`` **as of that image's last
    build**, and the per-user compose services declare only ``image:``, with no
    ``build:`` stanza to rebuild one at start. A guard built on this is
    therefore sound *provided render images are rebuilt after a render*, and
    deliberately claims nothing beyond that: a render newer than its image can
    report a deny the running image does not yet carry. Reading ``config.yml``
    would not close that window either — it is one step further from the image,
    not one step closer.

    Matching is by **exact entry**, never by tool-name resolution: a scoped deny
    (``Bash(rm:*)``) constrains one command family and leaves the tool otherwise
    usable, and a wildcard entry such as ``mcp__plugin_*`` is compared as
    the literal string the artifact carries. Callers therefore spell each tool exactly as
    :data:`~osprey.agent_runner.tool_names.DENY_DEFAULTS` spells it, which is
    what the ``settings.json.j2`` template writes.

    Fails **closed**: an artifact that cannot be read and parsed into a
    ``permissions.deny`` list is not evidence that anything is denied, so every
    such case answers ``False`` rather than raising or defaulting to "safe".
    That covers an absent file, unreadable or invalid JSON, a top-level value
    that is not an object, and a missing or non-list ``permissions`` /
    ``permissions.deny``.

    Args:
        project_dir: The rendered render project directory (the one holding
            ``config.yml`` and ``.claude/``).
        tools: The ``permissions.deny`` entries that must **all** be present.
            Asked as one question rather than one call per tool so a caller
            wanting "denies the whole egress set" cannot accidentally accept a
            render that denies only part of it.

    Returns:
        ``True`` only when the artifact was read, parsed, and lists every entry
        in *tools*; ``False`` in every other case.
    """
    deny = settings_json_deny_entries(project_dir)
    return deny is not None and deny.issuperset(tools)


def settings_json_deny_entries(project_dir: Any) -> frozenset[str] | None:
    """The ``permissions.deny`` entries the shipped artifact lists, or ``None``.

    The read :func:`settings_json_denies` answers its yes/no question with, kept
    separately for the caller that needs the whole set rather than one verdict:
    a guard naming WHICH of several required entries a render is missing would
    otherwise re-read and re-parse the same small file once per entry.

    Every property of the read — the shipped artifact rather than the config,
    the BOM-tolerant decode, exact-entry matching, and failing closed on
    anything it cannot turn into a deny list — is described on
    :func:`settings_json_denies` and is unchanged here.

    Args:
        project_dir: The rendered render project directory (the one holding
            ``config.yml`` and ``.claude/``).

    Returns:
        The set of string ``permissions.deny`` entries, or ``None`` when the
        artifact is absent, unreadable, unparseable, or carries no
        ``permissions.deny`` list. ``None`` is deliberately distinct from an
        empty set: "nothing is denied here" and "there is nothing to read" are
        the same verdict for a guard but different sentences for an operator.
    """
    settings_json = settings_json_path(project_dir)
    try:
        # utf-8-sig, not utf-8: `json.load` does not strip a BOM, so a hand-edited
        # artifact saved with one would fail to parse and a genuinely Bash-denying
        # render would read as permissive — an opaque deploy refusal. The codec is
        # a strict superset for BOM-less files, which is everything OSPREY renders.
        with settings_json.open("r", encoding="utf-8-sig") as fh:
            settings = json.load(fh)
    except (OSError, ValueError):
        return None
    deny = as_dict(as_dict(settings).get("permissions")).get("deny")
    if not isinstance(deny, list):
        return None
    return frozenset(entry for entry in deny if isinstance(entry, str))


def settings_json_path(project_dir: Any) -> Path:
    """Where a rendered render project keeps the settings artifact this reads.

    One spelling of ``.claude/settings.json``, so a guard asking whether the
    artifact exists at all cannot look somewhere other than where the reads
    above look.
    """
    return Path(project_dir) / ".claude" / "settings.json"


def settings_json_is_rendered(project_dir: Any) -> bool:
    """Whether *project_dir* holds a settings artifact at all.

    Not a safety question — an artifact that exists proves nothing about what it
    denies — but a phrasing one. A render with no rendered project on this host
    fails every deny check for a reason no ``permissions.deny`` edit can fix,
    and telling that operator to add a deny entry sends them to a file that is
    not there. Callers that fail closed ask this only to choose the sentence.
    """
    return settings_json_path(project_dir).is_file()


def renders_not_denying(config: Any, project_root: Any, tools: Iterable[str]) -> set[str]:
    """Names of referenced renders whose shipped settings do **not** deny *tools*.

    The roster-shaped form of :func:`settings_json_denies`, phrased as the
    *unsafe* set so its caller can name every offending render in one error
    rather than re-deriving them.

    Walks the roster itself rather than routing through the shared
    ``config.yml`` engine the entitlement predicates use, because the question
    is about the built artifact and not about intent — see
    :func:`settings_json_denies`. Membership is inclusive for the same
    fail-closed reason: a render with no ``project_path``, or one whose project
    is not rendered, is reported here as not denying anything.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.
        tools: The ``permissions.deny`` entries a render must **all** ship to
            stay out of the returned set.

    Returns:
        The subset of referenced render names whose rendered
        ``.claude/settings.json`` does not deny every named tool.
    """
    return {
        render_name
        for render_name, project_dir in referenced_render_project_dirs(config, project_root).items()
        if project_dir is None or not settings_json_denies(project_dir, tools)
    }


def referenced_render_project_dirs(config: Any, project_root: Any) -> dict[str, Path | None]:
    """Every served render's rendered project directory, ``None`` when it has none.

    The roster walk behind :func:`renders_not_denying`, named on its own so a
    caller asking several questions of the same artifact walks the roster once
    and reads each render's ``settings.json`` once. Both readers stay bound to
    one definition of "referenced" and one resolution of ``project_path``, which
    is what keeps a raising guard and an ask-only reader from disagreeing about
    which renders a deployment even has.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        ``{render: directory}`` for every :func:`served_renders` name, with
        ``None`` for one the projection does not list or whose entry carries no
        usable ``project_path``. ``None`` is
        not "clean": such a render ships no artifact a guard can read, and
        every fail-closed caller must treat it as denying nothing.
    """
    served = served_renders(config)
    dirs: dict[str, Path | None] = {}
    for render_name in sorted(served):
        render = served[render_name]
        project_path = render.project_path if render is not None else None
        dirs[render_name] = Path(project_root, project_path) if project_path is not None else None
    return dirs


def renders_with_open_shell(config: Any, project_root: Any) -> set[str]:
    """Names of referenced renders whose shipped settings leave a shell open.

    A shell is open unless the render's ``.claude/settings.json`` denies
    every tool in :data:`~osprey.agent_runner.tool_names.SHELL_TOOLS`:
    ``Monitor`` runs shell commands in the background, so a render denying
    ``Bash`` alone still has one. Phrased as
    the *unsafe* set so its caller can name every offending render in one
    error rather than re-deriving them. A render granted
    ``BLUESKY_LAUNCH_TOKEN`` while its agent may also run a shell can read that
    token out of its own environment and arm a queue with a plain HTTP call,
    bypassing the chat approval entirely — the approval gates the
    ``queue_start`` tool, not a shell. Intersecting this set with the union of
    :func:`renders_needing_launch_token_by_lane` therefore names exactly the
    renders a deploy must refuse.

    The shell-tools case of :func:`renders_not_denying`, which describes the
    walk and the fail-closed membership rule. Being inclusive never over-blocks
    a deploy here, since a render whose project is missing cannot satisfy
    :func:`config_needs_launch_token` either and so drops out of the
    intersection anyway.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        The subset of referenced render names whose rendered
        ``.claude/settings.json`` does not deny every shell tool.
    """
    return renders_not_denying(config, project_root, SHELL_TOOLS)
