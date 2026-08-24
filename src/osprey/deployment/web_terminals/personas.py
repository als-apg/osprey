"""Roster and persona identity resolution for multi-user web terminal deployments.

Turns the raw ``modules.web_terminals`` roster into explicit per-user identity:
normalized ``{"name", "index"}`` entries (:func:`normalize_users`) and fully
resolved image/project/container-dir identity per user
(:func:`resolve_personas`). Callers that write the roster *back* to
``config.yml`` rather than render from it use :func:`freeze_user_indices`, which
keeps the authored keys the normalizer projects away. Also home to the
username→env-var-suffix mapping (:func:`env_var_suffix`) and its collision detector
(:func:`env_var_suffix_collisions`), which credential provisioning, the auth
sidecar and lint share so a user's credentials are keyed identically everywhere.
Port arithmetic lives separately in :mod:`osprey.deployment.web_terminals.ports`.

One rule about the roster is *not* about identity at all and still lives here:
which personas can edit the deployment they run in (:func:`persona_privileges`),
and who may be handed one — the ``default_persona`` rule
(:func:`privileged_default_persona_problem`) and the login rule
(:func:`unauthenticated_privileged_terminal_problems`). Two guards ask those
questions on three different surfaces —
:func:`osprey.cli.profile_cmd._privileged_persona_problems` at ``osprey init``,
and the lint belt in :mod:`osprey.deployment.web_terminals.lint` at profile and
rendered altitude (the second of which ``osprey up`` gates on) — and this is the
neutral module all of them already import; see the section comment above those
functions.
"""

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.deployment.graphdb_service import resolve_graphdb_service_config

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
# Public and defined here, alongside `env_var_suffix`, because this module owns
# what a roster username *is*: lint's scaffold-time rule, render's fail-closed
# gate and `auth_credentials`' deploy-time gate all import it from here, so the
# three cannot drift apart.
#
# Apply it with `.fullmatch()`, never `.match()`: Python's `$` also matches
# *before* a trailing newline, so `.match()` accepts "alice\n" — a name that
# goes on to render into an nginx location key mid-directive.
USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}


#: The in-terminal event-dispatcher dashboard's panel id. A persona that declares
#: this panel is the one thing that entitles its containers to
#: ``EVENT_DISPATCHER_TOKEN`` — the panel declaration IS the intent, so there is
#: no separate config key to set (and to forget to set) alongside it.
EVENTS_PANEL_ID = "events"

#: The bluesky-web sidecar's tab. A persona that declares it proxies into the
#: sidecar with its own operator secret, so the sidecar is handed that user's
#: secret variable (the web overlay's ``bluesky-web`` fragment) — the panel
#: declaration IS the entitlement, exactly as for :data:`EVENTS_PANEL_ID`.
BLUESKY_PANEL_ID = "bluesky"


def config_declares_panel(config: Any, panel_id: str) -> bool:
    """True if ``config`` declares ``web.panels.<panel_id>`` and hasn't disabled it.

    The one definition of "this project shows that panel", shared by the render
    (for persona-less roster entries) and by :func:`personas_needing_dispatcher_token`
    (for catalog personas), so the two cannot answer differently for the same
    config. ``enabled: false`` counts as *not* declared: a panel switched off is
    one whose credential is not needed.
    """
    panel = as_dict(as_dict(as_dict(config).get("web")).get("panels")).get(panel_id)
    if not isinstance(panel, dict):
        return False
    return panel.get("enabled", True) is not False


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

    The two conditions are deliberately asymmetric, because their defaults are:

    * ``control_system.writes_enabled`` must be **explicitly** ``True``. This is
      the read-only tier's whole boundary — a read-only persona expresses itself
      as ``writes_enabled: false``, and an absent or null key means writes were
      never granted. Anything other than a literal ``True`` therefore entitles
      nothing.
    * ``claude_code.servers.bluesky.enabled`` must merely be **not** ``False``.
      That key is an *override* over the server's built-in default (see
      :func:`osprey.registry.mcp.resolve_servers`, which likewise disables only
      on a literal ``enabled: false``), so an absent key leaves the server
      enabled. Reading absence as "disabled" here would deny the token to
      correctly-configured projects that simply never wrote the override.

    Neither condition entitles on its own: writes without the server have no
    caller, and the server without writes has nothing it may arm.
    """
    root = as_dict(config)
    if as_dict(root.get("control_system")).get("writes_enabled") is not True:
        return False
    servers = as_dict(as_dict(root.get("claude_code")).get("servers"))
    return as_dict(servers.get("bluesky")).get("enabled", True) is not False


def config_needs_graphdb_password(config: Any) -> bool:
    """True if ``config`` configures a graph store and runs the ``graph`` MCP server.

    ``GRAPHDB_PASSWORD`` is the store's *single* credential: Neo4j has one
    account, ``neo4j``, and it is write-capable. There is no read-only account
    to hand out instead, so a persona that may query the graph at all holds the
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
    contrast is :func:`config_needs_launch_token`, which *is* gated on
    ``control_system.writes_enabled``, because there the credential itself is
    the capability — nothing downstream of holding it stops a queue start.

    The blast radius bounds the decision: the store holds a disposable mirror of
    a Turtle corpus, re-seedable from it with ``osprey knowledge seed-graph``, so
    the worst case is corpus integrity rather than facility data or hardware.
    This predicate is consequently **never** gated on ``writes_enabled`` — doing
    so would leave the read-only operator terminal, the very tier the graph
    search is meant to serve, unable to reach the store at all.

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
      persona's ``config.yml`` into a failed render of the whole roster.
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


#: What a ``password_env`` name must look like to be emitted into a compose
#: ``environment:`` line verbatim. Anything else is refused rather than rendered.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def config_archiver_password_env(config: Any) -> str | None:
    """The variable ``config``'s archiver connector authenticates with, or ``None``.

    The archiver connector reads its password from the environment variable its
    own block names — ``archiver.<type>.password_env`` — and raises on every
    read when that variable is unset. For a store the project deploys itself,
    ``osprey up`` mints the value into the deploy ``.env`` under that name; for
    a facility-run store the operator puts it there. Either way the web
    terminal's agent can only authenticate if its container is handed *that*
    variable, which is why this returns the configured NAME rather than a
    boolean: the grant carries it, so a project reading a store under any
    spelling is served.

    Only the SELECTED connector's block counts. The shipped ``config.yml``
    carries a filled-in ``mongodb_archiver:`` block under ``type:
    mock_archiver``; a block the selected type never reads entitles nothing,
    for the reason :func:`config_needs_ariel_password` gives — a credential is
    gated on what its consumer actually reads.

    Raises:
        ValueError: when the configured name is not a plain identifier. The
            name is emitted into a compose ``environment:`` line verbatim, so
            a value compose would mangle (a space, an ``=``, a ``${``) is
            refused at the deploy gate rather than rendered broken.
    """
    archiver = as_dict(as_dict(config).get("archiver"))
    connector = archiver.get("type")
    if not isinstance(connector, str) or not connector:
        return None
    password_env = as_dict(archiver.get(connector)).get("password_env")
    if not isinstance(password_env, str) or not password_env.strip():
        return None
    password_env = password_env.strip()
    if not _ENV_VAR_NAME_RE.match(password_env):
        raise ValueError(
            f"archiver.{connector}.password_env must name an environment variable "
            f"(letters, digits and underscores, not starting with a digit), got "
            f"{password_env!r}"
        )
    return password_env


def _referenced_personas(config: Any) -> tuple[dict[str, Any], set[str]]:
    """The persona catalog and the names some roster entry actually resolves to.

    ``default_persona`` counts alongside every entry's explicit ``persona``: a
    roster entry that names none runs the default, so the default's project is
    as deployed as any other.
    """
    web_terminals = as_dict(as_dict(as_dict(config).get("modules")).get("web_terminals"))
    catalog = as_dict(web_terminals.get("personas"))

    referenced: set[str] = set()
    default_persona = web_terminals.get("default_persona")
    if isinstance(default_persona, str) and default_persona:
        referenced.add(default_persona)
    users = web_terminals.get("users")
    for user in users if isinstance(users, list) else []:
        if isinstance(user, dict) and isinstance(user.get("persona"), str) and user["persona"]:
            referenced.add(user["persona"])
    return catalog, referenced


def _personas_whose_config(config: Any, project_root: Any, predicate) -> set[str]:
    """Referenced personas whose rendered ``config.yml`` satisfies ``predicate``.

    The one walk behind every per-persona credential grant, so two credentials
    cannot disagree about which personas a roster deploys. Reads each persona's
    ``config.yml`` off disk — which is why the render takes the result as a
    parameter rather than calling this itself (see
    :func:`osprey.deployment.web_terminals.render.render_web_terminals`'s
    determinism note). Resolution mirrors
    :func:`osprey.deployment.web_terminals.env_production._claude_code_auth_secret_vars`:
    a persona whose ``project_path`` is unset, unrendered, or unreadable
    contributes nothing, because a credential is never granted on a guess.
    """
    return {
        persona_name
        for persona_name, persona_config in _persona_configs(config, project_root)
        if predicate(persona_config)
    }


def _persona_configs(config: Any, project_root: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(persona_name, parsed config.yml)`` for every readable referenced persona.

    The one disk walk behind :func:`_personas_whose_config` and
    :func:`personas_needing_archiver_password`; see the former for why a
    persona that cannot be read is skipped rather than guessed at.
    """
    import yaml

    catalog, referenced = _referenced_personas(config)

    for persona_name in sorted(referenced):
        entry = as_dict(catalog.get(persona_name))
        project_path = entry.get("project_path")
        if not isinstance(project_path, str) or not project_path:
            continue
        config_yml = Path(project_root, project_path) / "config.yml"
        if not config_yml.is_file():
            continue
        try:
            with config_yml.open("r", encoding="utf-8") as fh:
                persona_config = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        yield persona_name, persona_config


def rendered_persona_configs(config: Any, project_root: Any) -> dict[str, Any]:
    """Referenced persona name → its parsed rendered ``config.yml``.

    The public face of :func:`_persona_configs` for callers that want the
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
        One entry per referenced persona whose rendered ``config.yml`` could be
        read. A persona that is unset, unrendered or unreadable is absent —
        never present with a guessed document.
    """
    return dict(_persona_configs(config, project_root))


def personas_needing_dispatcher_token(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project needs the dispatcher token.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names entitled to
        ``EVENT_DISPATCHER_TOKEN`` (see :func:`config_needs_dispatcher_token`).
    """
    return _personas_whose_config(config, project_root, config_needs_dispatcher_token)


def config_declares_bluesky_panel(config: Any) -> bool:
    """Whether this project shows the BLUESKY tab, and so proxies into the
    bluesky-web sidecar with its own operator secret."""
    return config_declares_panel(config, BLUESKY_PANEL_ID)


def personas_declaring_bluesky_panel(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project declares the BLUESKY tab.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names whose users' operator
        secrets the bluesky-web sidecar is handed (see
        :func:`config_declares_bluesky_panel`).
    """
    return _personas_whose_config(config, project_root, config_declares_bluesky_panel)


def bluesky_panel_secret_env_vars(config: Any, project_root: Any) -> list[str]:
    """The per-user secret variables the bluesky-web sidecar is handed.

    One name per roster user whose terminal shows the BLUESKY tab — and so
    proxies into the sidecar with the operator secret ITS container holds,
    under the fixed ``OSPREY_TERMINAL_SECRET`` name. The sidecar's own compose
    file (``services/bluesky_web``) lists each of these variables so its web
    gate accepts every entitled user's secret beside the deployment-wide one,
    and no user's container is ever handed the deployment secret.

    Entitlement is decided the way the web-terminal render decides every
    per-user grant: a user with a persona (their own, else
    ``default_persona``) by that persona's rendered ``config.yml``
    (:func:`personas_declaring_bluesky_panel`); a persona-less user runs the
    deployment's own project, so the deploy config answers for them. Roster
    order, so the rendered compose is stable across runs.

    Read off the personas' rendered ``config.yml`` files, like every other
    per-persona grant — so the render that carries it is the deploy's:
    ``osprey up`` re-renders every services compose file from the repo root
    with the persona projects in place, while ``osprey build`` renders the
    services compose before its persona renders and so lists nothing (or what
    the previous build's personas declared).

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: Variable names (``OSPREY_TERMINAL_SECRET_<SUFFIX>``), one per
        entitled user; empty when no user shows the tab.
    """
    from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var

    web_terminals = as_dict(as_dict(as_dict(config).get("modules")).get("web_terminals"))
    raw_users = web_terminals.get("users")
    refs = _persona_ref_by_name(raw_users)
    default_persona = web_terminals.get("default_persona")
    if not isinstance(default_persona, str) or not default_persona:
        default_persona = None
    entitled_personas = personas_declaring_bluesky_panel(config, project_root)
    deploy_declares = config_declares_bluesky_panel(config)

    names: list[str] = []
    for entry in normalize_users(raw_users):
        persona = refs.get(entry["name"]) or default_persona
        entitled = persona in entitled_personas if persona else deploy_declares
        if entitled:
            names.append(terminal_secret_var(entry["name"]))
    return names


def personas_needing_ariel_password(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project configures ARIEL.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names entitled to
        ``ARIEL_DB_PASSWORD`` (see :func:`config_needs_ariel_password`).
    """
    return _personas_whose_config(config, project_root, config_needs_ariel_password)


def personas_needing_facility_bundle(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project configures a knowledge bundle.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names whose container gets the
        deployment bundle bind-mounted (see :func:`config_needs_facility_bundle`).
    """
    return _personas_whose_config(config, project_root, config_needs_facility_bundle)


def personas_needing_ariel_mirror(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project writes the ARIEL mirror.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names whose container gets the
        deployment's mirror bind-mounted (see :func:`config_needs_ariel_mirror`).
    """
    return _personas_whose_config(config, project_root, config_needs_ariel_mirror)


def personas_needing_launch_token(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project may arm a queue start.

    This is the tier boundary in roster form. :func:`config_needs_launch_token`
    decides the question per project; this wrapper is where that decision
    becomes a *set* a caller can hand a bearer token to, one persona at a
    time -- and it is therefore also where the security property the whole
    feature rests on actually holds: a read-only persona's rendered
    ``config.yml`` carries ``control_system.writes_enabled: false``, which can
    never satisfy the predicate, so a read-only persona can never appear in
    this set and can never be handed ``BLUESKY_LAUNCH_TOKEN``. That guarantee
    falls out of walking the same per-persona ``config.yml`` files
    :func:`personas_needing_dispatcher_token` and
    :func:`personas_needing_ariel_password` already walk -- there is no
    separate roster-level flag to keep in sync with the tier, and so nothing
    that a future persona addition could forget to set.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names entitled to
        ``BLUESKY_LAUNCH_TOKEN`` (see :func:`config_needs_launch_token`).
    """
    return _personas_whose_config(config, project_root, config_needs_launch_token)


def personas_needing_graphdb_password(config: Any, project_root: Any) -> set[str]:
    """Names of catalog personas whose rendered project queries the graph store.

    Unlike :func:`personas_needing_launch_token`, this set deliberately spans
    both tiers: a read-only persona that configures a graph store belongs in it,
    because the store has exactly one account and the read-only guarantee lives
    in the ``graph`` MCP server's read transactions rather than in the
    credential. See :func:`config_needs_graphdb_password` for that reasoning and
    for the blast radius that bounds it.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: The subset of referenced persona names entitled to
        ``GRAPHDB_PASSWORD`` (see :func:`config_needs_graphdb_password`).
    """
    return _personas_whose_config(config, project_root, config_needs_graphdb_password)


def personas_needing_archiver_password(config: Any, project_root: Any) -> dict[str, str]:
    """Map each catalog persona whose archiver reads a password to the variable it reads.

    A map rather than a set because the grant carries the variable NAME (see
    :func:`config_archiver_password_env`): two personas reading two stores each
    get their own line, and the render emits exactly the name the connector
    will look up. Walks the same per-persona ``config.yml`` files the other
    grants walk, so this cannot disagree with them about which personas a
    roster deploys.

    :param config: The parsed deploy config.
    :param project_root: Deploy project root; relative ``project_path`` values
        resolve against it.
    :return: ``{persona_name: env_var_name}`` for the referenced personas whose
        selected archiver connector names a ``password_env``.
    :raises ValueError: when a persona names a variable compose cannot carry
        (see :func:`config_archiver_password_env`).
    """
    grants: dict[str, str] = {}
    for persona_name, persona_config in _persona_configs(config, project_root):
        password_env = config_archiver_password_env(persona_config)
        if password_env is not None:
            grants[persona_name] = password_env
    return grants


def normalize_users(users_raw: Any) -> list[dict[str, Any]]:
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
    means the normalized entry stays the single object :func:`resolve_personas`
    reads a user's identity off of, rather than re-deriving the field from the raw
    roster the way ``persona`` is.

    An object entry's optional ``theme`` (that user's default web UI theme --
    a theme family like ``desy`` or a concrete id like ``desy-light``, surfaced
    downstream as the per-user ``OSPREY_WEB_THEME``) is carried through on
    exactly the same terms. Validity of the *value* is not checked here: the
    web terminal resolves it at startup and warns+falls back on an unknown one,
    and lint reports a non-string separately.

    An object entry's optional ``oidc_subject`` (the value of the configured OIDC
    claim -- ``sub`` by default -- that identifies this roster user at the IdP) is
    carried through on the same terms, with one deliberate difference: an *empty*
    string is dropped along with every non-string, where ``display_name`` and
    ``theme`` would keep it. This field is an authorization mapping, not a
    cosmetic one -- carrying ``""`` through would let an identity whose claim is
    missing or empty match a roster user. Dropping it instead leaves that user
    with no mapping at all, which the callback answers with 403. Only the
    *non-secret* side of the mapping ever lives in config.yml; password hashes
    never do (they live in ``.env.auth``, keyed by :func:`env_var_suffix`).

    An object entry's optional ``login`` (whether this entry sits behind the
    deployment's login wall when authentication is enabled) is carried through
    only when it is the literal boolean ``False`` — the one value that changes
    anything. ``true``, absence, and every malformed spelling all mean "login
    required", so dropping them is both the fail-closed reading and what keeps
    a config typo from silently opening an entry to the world (lint reports
    the typo separately). See :func:`entry_requires_login` for the single
    consumer-side reading of the carried key.

    Malformed entries — anything that isn't a string, and any dict missing a
    string ``name`` or an int ``index`` — are dropped rather than raising
    (well-formedness is lint.py's job). ``bool`` is a subclass of ``int`` in
    Python, so a dict entry
    like ``{"name": "alice", "index": True}`` would technically satisfy
    ``isinstance(index, int)``; this function deliberately treats ``bool`` as an
    invalid index type and drops such entries, since ``index: true/false`` in a
    facility config is a config typo (e.g. a YAML boolean where an int was meant),
    not a meaningful index.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value. Anything other
            than a list (including ``None``) is treated as an empty roster.

    Returns:
        New ``{"name": str, "index": int}`` dicts (plus optional
        ``"display_name"``, ``"theme"`` and ``"oidc_subject"`` string keys and
        a ``"login": False`` marker when the entry carried them) in
        config-declaration order. Input dicts are never mutated or returned by
        reference.
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
                oidc_subject = entry.get("oidc_subject")
                if isinstance(oidc_subject, str) and oidc_subject:
                    normalized_entry["oidc_subject"] = oidc_subject
                # Only the literal boolean False is carried: absent or True both
                # mean "login required", and any other value is a config typo
                # (reported by lint) whose safe reading is the same. Carrying
                # only the exempting value keeps the gate fail-closed — a typo
                # can never open an entry to the world.
                if entry.get("login") is False:
                    normalized_entry["login"] = False
                normalized.append(normalized_entry)
    return normalized


def entry_requires_login(entry: dict[str, Any]) -> bool:
    """Whether this normalized roster entry sits behind the login wall.

    ``login: false`` on a roster entry opts it out of authentication: with
    ``auth.method`` enabled, nginx still gates every other entry but proxies
    this one without an ``auth_request``, and no password is provisioned for
    it. The single reading of that key, shared by the render (which decides
    whether to emit the gate), credential provisioning (which decides whether a
    password exists) and the ``users passwd`` verb (which refuses to rotate a
    password that cannot exist) — three call sites that must never disagree
    about who has a login.

    Reads the *normalized* entry, so only the literal ``False``
    :func:`normalize_users` carries through exempts; every malformed spelling
    already normalized back to "login required".
    """
    return entry.get("login") is not False


def freeze_user_indices(users_raw: Any) -> list[dict[str, Any]]:
    """The roster **as authored**, with :func:`normalize_users`' indices frozen onto it.

    :func:`normalize_users` answers "who is on the roster, at which index" and
    deliberately projects each entry down to the handful of fields the render
    reads. That projection is the right input for rendering and the WRONG thing
    to write back to ``config.yml``: every key it does not know about —
    ``persona`` above all — would be dropped from the file, and a roster whose
    ``persona:`` keys are gone re-resolves every survivor onto
    ``default_persona`` (see :func:`resolve_personas`, which reads ``persona``
    off the raw roster). For a user removal that is a silent privilege change,
    not a cosmetic one, whenever the default persona is the more privileged.

    So this function starts from :func:`normalize_users`' output — survival and
    index-freezing stay entirely that function's contract, with no second copy
    of its rules here — and re-attaches the keys the author actually wrote.
    Entries are matched by ``name``, the same key
    :func:`_persona_ref_by_name` and every other per-user artifact in this
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


# ── Deployment-editing privilege ─────────────────────────────────────────────
#
# Two config keys decide whether a persona can edit the deployment it runs in:
# the agent's own setup tool (``claude_code.permissions.deny``, read through
# :func:`~osprey.cli.profile_conventions.is_setup_patch_capable`) and the
# browser's Config panel (``web.config_panel.enabled``). They are two doors
# onto one capability — the agent edits ``config.yml`` through the tool, a
# person edits the same file through the panel — so everything below treats
# holding EITHER as privileged.
#
# The judgement lives in this neutral module because three commands ask it at
# three different altitudes, from three shapes of input, and they must not
# answer it differently:
#
# * ``osprey init`` asks it of a resolved persona PRESET
#   (:func:`osprey.cli.profile_cmd._persona_profile_texts`);
# * ``osprey profile validate`` / ``osprey build`` ask it of the host profile's
#   ``config:`` overrides with each ``personas/<name>.yml`` delta layered on
#   top (:mod:`osprey.deployment.web_terminals.lint`);
# * ``osprey up`` asks it of each persona's rendered ``config.yml`` (the same
#   lint module, one gate later).
#
# :func:`persona_privileges` takes all three by taking *layers*: any number of
# config documents, read newest-last, in whichever spelling each one uses.


#: How the setup-tool privilege is named inside a guard message. Prose, not a
#: key: the message names the key separately, and an operator reading "this
#: persona can edit the deployment" acts on it faster than one reading a deny
#: list entry.
PRIVILEGE_SETUP_TOOL = "the agent's deployment-editing setup tool"

#: How the Config-panel privilege is named inside a guard message.
PRIVILEGE_CONFIG_PANEL = "the web Config panel"

#: Every privilege :func:`persona_privileges` can report, in its order. Read by
#: guards that have to ask "does this deployment declare a privilege SPLIT at
#: all" without holding a persona: a baseline that already grants everything
#: leaves nothing for any persona to rise above (see
#: :func:`privileges_beyond_baseline`), so a guard whose whole subject is the
#: split has nothing to say about that deployment — including when it cannot
#: read a persona at all.
ALL_PRIVILEGES = (PRIVILEGE_SETUP_TOOL, PRIVILEGE_CONFIG_PANEL)

#: The bundled tier every remedy below points at. Quoted rather than described
#: because a remedy an operator can paste is worth more than one they have to
#: translate; a deployment that renamed its tiers still reads it as "a persona
#: without these privileges", which the same sentence says.
UNPRIVILEGED_TIER_EXAMPLE = "readonly"

#: The three keys :func:`persona_privileges` reads, in their dotted spelling.
#: Every other spelling of the same path is reached by :func:`_declared_values`.
_DENY_KEY = "claude_code.permissions.deny"
_REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
_CONFIG_PANEL_KEY = "web.config_panel.enabled"

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


def _panel_flag(value: Any) -> bool:
    """Read one ``web.config_panel.enabled`` declaration the way the server does.

    A real boolean is honoured as written and a quoted boolean word with it
    (``"false"`` is a human writing a boolean). Anything else is a value the
    server itself discards in favour of the shipped default, which for this key
    is ON — so it is read as ON here too, rather than as a quiet OFF that would
    let an unreadable value walk a privileged persona past the guard.
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


def persona_capability_document(*layers: Any) -> dict[str, Any]:
    """The ``config.yml``-shaped document the setup-capability predicate reads.

    :func:`~osprey.cli.profile_conventions.is_setup_patch_capable` consumes one
    document with ``claude_code.permissions.deny`` and ``remove_deny`` beside
    each other, and owns the subtraction between them (exact membership,
    mirroring how ``settings.json`` renders). This assembles that document out
    of however many layers a caller holds, so the *composition* stays in one
    place and this owns only the reading:

    * **both spellings** of each key are read (see :func:`_declared_values`);
    * lists **union** across layers and across spellings, which is what profile
      resolution itself does with string lists across ``extends`` — a persona
      delta cannot subtract from an inherited ``deny`` except through
      ``remove_deny``, which is exactly why that key exists.

    Args:
        *layers: Config documents, base first. A resolved profile's ``config:``
            bag, a persona delta's own ``config:`` bag, or a rendered
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


def persona_privileges(*layers: Any) -> tuple[str, ...]:
    """The deployment-editing privileges the persona these layers describe holds.

    The one answer behind every guard in this module and behind lint's belt
    check: two surfaces onto the same capability, reported together so a
    message can name what an operator actually gets.

    The Config panel is ON unless a layer turns it off — that is the template's
    shipped default, so a persona that never mentions the key has the panel,
    and reading an unmentioned key as "off" would walk exactly the tier this
    guards past it. Later layers win over earlier ones; a single layer that
    spells the key twice is read as ON unless *every* spelling turns it off,
    which is the same lean.

    Args:
        *layers: Config documents, base first — see
            :func:`persona_capability_document`.

    Returns:
        :data:`PRIVILEGE_SETUP_TOOL` and/or :data:`PRIVILEGE_CONFIG_PANEL`, in
        that order; empty for a persona holding neither.
    """
    from osprey.cli.profile_conventions import is_setup_patch_capable

    panel_enabled = True
    for layer in layers:
        declarations = [_panel_flag(value) for value in _declared_values(layer, _CONFIG_PANEL_KEY)]
        if declarations:
            panel_enabled = any(declarations)

    privileges: list[str] = []
    if is_setup_patch_capable(persona_capability_document(*layers)):
        privileges.append(PRIVILEGE_SETUP_TOOL)
    if panel_enabled:
        privileges.append(PRIVILEGE_CONFIG_PANEL)
    return tuple(privileges)


def privileges_beyond_baseline(held: Sequence[str], baseline: Sequence[str]) -> tuple[str, ...]:
    """What a persona holds that the deployment it belongs to does not.

    The MIGRATION-SENSITIVE half of the guards' definition of privileged, and
    the reason that half is relative rather than absolute.

    **Which rule reads which.** Two rules are built on privilege, and they take
    their answer from different sides of this function on purpose:

    * :func:`privileged_default_persona_problem` reads this, the RELATIVE
      answer. A profile's ``default_persona`` is inherited from whatever the
      deployment was before tiers existed — nobody chose it as a privilege
      grant — so the finding is about an authoring mistake inside a declared
      split, and a legacy profile with no split has no better default to be
      pointed at.
    * :func:`unauthenticated_privileged_terminal_problems` reads
      :func:`persona_privileges` directly, the ABSOLUTE answer. ``login: false``
      is not inherited from anything: it is a key somebody typed, on one entry,
      claiming that terminal may be served to anyone. Whether the deployment
      declared a tier split has no bearing on what that terminal hands out —
      a floorless deployment serving it hands out BOTH surfaces to the whole
      internet — so exempting the floorless case is exactly the fail-open this
      belt exists to prevent. The remedy is the half that changes: with no
      split to point at, the message says how to floor the base tier rather
      than naming a tier that does not exist.

    :func:`persona_privileges` answers an absolute question — can this document
    reach the setup tool, is the Config panel live — and every way of not having
    gated a surface reads as holding it, because that is the truth about the
    render. Refusing a build on that answer alone would refuse nearly every
    deployment in existence: the base tier's deny floor and
    ``web.config_panel.enabled: false`` are new, so a profile written before them
    grants both surfaces to every persona it has, and its ``default_persona``
    would fail a gate whose remedy ("point it at a tier that holds neither") it
    has no such tier to satisfy.

    What these guards actually protect is the privilege SPLIT a deployment
    declared: a base that floors a capability and a tier that lifts it back for
    named people. A persona is privileged when it rises above its own
    deployment's baseline, and a deployment whose baseline grants everything has
    no split to protect — every terminal is already equal, nothing is being
    handed to one entry that the others lack, and there is nothing here to
    report that would not simply be "you have not adopted tiers yet".

    The deliberate consequence, and the cost of being shippable: a deployment
    that floors nothing draws no ``default_persona`` finding even when every
    tier it has can edit it. That much is the *base tier's* to fix — it is what
    the shipped base's deny floor and panel default exist for. What it is NOT
    is a licence to serve such a persona without a login; see the second bullet
    above.

    Args:
        held: :func:`persona_privileges` for the persona.
        baseline: :func:`persona_privileges` for the deployment it belongs to —
            the host profile's own ``config:`` block, or the deploy
            ``config.yml`` that the persona renders beside.

    Returns:
        The subset of ``held`` the baseline does not also grant, in ``held``'s
        order; empty when the persona is no more privileged than its base.
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


def privileged_default_persona_problem(
    default_persona: Any, privileges: Sequence[str]
) -> str | None:
    """Why ``default_persona`` must not be this persona, or ``None`` if it may.

    ``default_persona`` is what every roster entry without a ``persona:`` key of
    its own inherits — including entries added later, by someone who never read
    this key. So the default is the tier the deployment hands out by accident,
    and it has to be the one it is safe to hand out by accident.

    **Baseline-RELATIVE, unlike the ``login: false`` rule beside it.** Callers
    pass :func:`privileges_beyond_baseline`' output here and
    :func:`persona_privileges`' raw output to
    :func:`unauthenticated_privileged_terminal_problems`, and the asymmetry is
    the point: this key is inherited rather than chosen. A profile written
    before the base tier's deny floor existed has a ``default_persona`` nobody
    picked as a privilege grant, and no unprivileged tier to be re-pointed at,
    so refusing it would refuse the migration itself. ``login: false`` is the
    opposite — typed deliberately, on one entry, about that entry — so it is
    judged on what the persona actually holds. See
    :func:`privileges_beyond_baseline` for the full statement of the split.

    Args:
        default_persona: ``modules.web_terminals.default_persona``. A missing or
            non-string value is not this check's to report (lint's
            ``unknown_default_persona`` owns that), so it yields ``None``.
        privileges: What that persona holds BEYOND its deployment's baseline —
            :func:`privileges_beyond_baseline`' output, not
            :func:`persona_privileges`'.

    Returns:
        A message naming the persona, what it holds and the remedy, or ``None``.
    """
    if not isinstance(default_persona, str) or not default_persona or not privileges:
        return None
    return (
        f"modules.web_terminals.default_persona {default_persona!r} resolves to a persona "
        f"that holds {privilege_phrase(privileges)}. Every roster entry with no persona: key of "
        f"its own inherits the default, so a privileged default is handed to every terminal "
        f"that forgets to name a tier — including ones added later. Point default_persona at "
        f"a persona that holds neither (the bundled stack's {UNPRIVILEGED_TIER_EXAMPLE!r}), "
        f"and name {default_persona!r} on the individual entries that need it"
    )


def auth_is_enforced(web_terminals: Any) -> bool:
    """Whether this deployment puts a login wall in front of its terminals at all.

    ``auth.method: none`` (the default, and the whole ``auth`` stanza is
    optional) means no entry has a login — which makes every terminal what
    ``login: false`` makes one, and is why
    :func:`unauthenticated_privileged_terminal_problems` takes this as a
    parameter rather than reading ``login`` alone.

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
    return bool(context["auth_method"] != "none")


def _privileged_entries(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_persona: Mapping[str, Sequence[str]],
) -> list[tuple[Mapping[str, Any], str, Sequence[str]]]:
    """``(entry, persona, privileges)`` for every entry that can edit the deployment.

    A persona missing from ``privileges_by_persona`` contributes nothing, on the
    same terms as :func:`_persona_configs`: a config that could not be read is
    not evidence of a privilege, and every way of failing to read one is already
    reported by the check that owns it. Neither is an entry with no persona in
    effect — the zero-migration path has no tier to be privileged.
    """
    entries: list[tuple[Mapping[str, Any], str, Sequence[str]]] = []
    for entry in resolved_entries:
        persona = entry.get("persona")
        if not isinstance(persona, str):
            continue
        privileges = privileges_by_persona.get(persona) or ()
        if privileges:
            entries.append((entry, persona, privileges))
    return entries


def _floor_the_base_tier_remedy(name: Any, unfloored: Sequence[str]) -> str:
    """The remedy for an open privileged terminal where the base tier floors too little.

    The other remedy ("point it at a persona that holds neither") presupposes
    such a persona exists. Every persona holds whatever the base tier hands out,
    so on a deployment that floors nothing there is nothing to be re-pointed at
    — and on one that floors only ONE of the two surfaces there may equally be
    nothing, since a persona that holds neither has to disable the remaining
    surface itself. Either way the sentence would name a fix the operator cannot
    carry out. This one names the keys that create the split, in the spellings
    the profile's ``config:`` block takes them in, for exactly the surfaces this
    deployment leaves open, and keeps ``login: true`` as the one-key way out.

    No square brackets anywhere in the sentence, deliberately: ``osprey build``
    renders its refusals through rich, which reads ``[...]`` as a style tag and
    silently ate the tool name out of an earlier draft of this remedy. The
    entry is named in prose instead.

    Args:
        name: The roster user the open terminal belongs to.
        unfloored: The privileges the deployment's BASE tier still holds — a
            non-empty subset of :data:`ALL_PRIVILEGES`, in its order. A caller
            with nothing here has a fully floored base and wants the tier
            remedy instead.
    """
    from osprey.cli.profile_conventions import SETUP_PATCH_TOOL

    add: list[str] = []
    lift: list[str] = []
    if PRIVILEGE_SETUP_TOOL in unfloored:
        add.append(f"`{_DENY_KEY}` with the entry `{SETUP_PATCH_TOOL}`")
        lift.append(f"`{_REMOVE_DENY_KEY}`")
    if PRIVILEGE_CONFIG_PANEL in unfloored:
        add.append(f"`{_CONFIG_PANEL_KEY}: false`")
        lift.append(f"`{_CONFIG_PANEL_KEY}: true`")
    if len(add) > 1:
        opening = "This deployment floors neither surface, so every persona holds them."
        additions = f"{', and '.join(add)},"
        lifting = f"lift them only in the persona meant to hold them ({', '.join(lift)})"
    else:
        opening = (
            f"This deployment does not floor {privilege_phrase(unfloored)} for its base tier, so "
            f"every persona holds it."
        )
        additions = add[0]
        lifting = f"lift it only in the persona meant to hold it ({lift[0]})"
    return (
        f"{opening} Add {additions} to this profile's `config:` block, and {lifting} — or "
        f"set `login: true` for {name!r}"
    )


def unauthenticated_privileged_terminal_problems(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_persona: Mapping[str, Sequence[str]],
    *,
    baseline_privileges: Sequence[str] = (),
) -> list[str]:
    """Every roster entry that OPTED OUT of the login wall and can edit the deployment.

    The exposure is not subtle: a card on the landing page that opens straight
    into a terminal, with the Config panel live inside it or the setup tool in
    the agent's hands. Anyone who reaches the page edits the deployment.

    Scoped to ``login: false`` (see :func:`entry_requires_login`) — an authored
    claim that *this* terminal is public — because that is a claim about one
    entry, made deliberately, that the author can act on. A deployment with no
    authentication at all is the wider version of the same exposure and is
    reported separately and advisorily by
    :func:`deployment_wide_privileged_exposure_problems`; see that function for
    why the two are not one rule.

    **Judged on what the persona ABSOLUTELY holds**, not on what it holds beyond
    its deployment's baseline. A deployment that floors neither surface hands
    both of them to every persona it has, so a ``login: false`` entry there is
    the most exposed version of this, not an exempt one — and reading it
    relatively made it silent at every altitude. The baseline decides only which
    REMEDY is honest; see :func:`privileges_beyond_baseline` for why the
    ``default_persona`` rule beside this one goes the other way.

    Args:
        resolved_entries: :func:`resolve_personas`' output (or anything with the
            same ``name``/``persona``/``login`` keys).
        privileges_by_persona: Persona name → :func:`persona_privileges` —
            the ABSOLUTE answer; see :func:`_privileged_entries` for what a
            missing persona means.
        baseline_privileges: :func:`persona_privileges` for the deployment
            itself, read only to choose the remedy. The default ``()`` is a
            fully floored base, which is the deployment shape that has a tier to
            be pointed at, so a caller that does not pass it gets the tier
            remedy rather than an instruction to floor a base it may already
            have floored.

    Returns:
        One message per offending entry, in roster order, each naming the user,
        the persona, what it holds and the remedy.
    """
    # What the BASE tier still hands to every persona. Empty is the shape that
    # has an unprivileged tier to point at; anything else means a persona that
    # holds neither surface exists only if it disables what the base leaves
    # open, so the honest remedy names the surface instead of the tier.
    unfloored = tuple(privilege for privilege in ALL_PRIVILEGES if privilege in baseline_privileges)
    problems: list[str] = []
    for entry, persona, privileges in _privileged_entries(resolved_entries, privileges_by_persona):
        # The `login: false` filter lives HERE and not in `_privileged_entries`:
        # the deployment-wide rule below deliberately ignores the key, because
        # with no wall standing it means nothing.
        if entry_requires_login(dict(entry)):
            continue
        name = entry.get("name")
        remedy = (
            f"Set login: true for {name!r} — or drop the key, since a login is the "
            f"default — or point {name!r} at a persona that holds neither (the bundled "
            f"stack's {UNPRIVILEGED_TIER_EXAMPLE!r})"
            if not unfloored
            else _floor_the_base_tier_remedy(name, unfloored)
        )
        problems.append(
            f"modules.web_terminals user {name!r} is served without a login (login: false), "
            f"but resolves to persona {persona!r}, which holds {privilege_phrase(privileges)}. "
            f"Anyone who opens that terminal's page edits this deployment. {remedy}"
        )
    return problems


def deployment_wide_privileged_exposure_problems(
    resolved_entries: Iterable[Mapping[str, Any]],
    privileges_by_persona: Mapping[str, Sequence[str]],
) -> list[str]:
    """The same exposure, reached through ``auth.method: none`` instead of one key.

    With no authentication configured there is no login wall for any entry to be
    exempt from, so every privileged terminal is as open as a ``login: false``
    one — the real exposure, and worth naming, which is why this exists at all.

    It is reported ADVISORILY where the ``login: false`` rule is an error, and
    the difference is what the config *claims*. ``login: false`` singles one
    entry out as public while the rest of the roster sits behind a wall: the
    author asked for that entry to be reachable and can act on the finding.
    ``auth.method: none`` is a whole-deployment posture — the shipped default,
    and a deliberate one for a stack bound to a loopback address — and failing
    a build over it would reject deployments that were never exposed to anyone,
    with no remedy but to configure authentication they do not need. Lint
    already says the narrower version of this (``user_login_inert``: a
    ``login: false`` under ``auth.method: none`` changes nothing).

    Callers reach this only when :func:`auth_is_enforced` is ``False``, and when
    they do they use it INSTEAD of
    :func:`unauthenticated_privileged_terminal_problems`, not alongside it: with
    no wall standing, an entry's own ``login`` key is inert, so reporting both
    would name one exposure twice and disagree with itself about severity.

    Args:
        resolved_entries: See
            :func:`unauthenticated_privileged_terminal_problems`.
        privileges_by_persona: Likewise.

    Returns:
        One message per privileged entry, in roster order.
    """
    problems: list[str] = []
    for entry, persona, privileges in _privileged_entries(resolved_entries, privileges_by_persona):
        name = entry.get("name")
        problems.append(
            f"modules.web_terminals user {name!r} resolves to persona {persona!r}, which "
            f"holds {privilege_phrase(privileges)}, and modules.web_terminals.auth.method is "
            f"'none' — so anyone who can reach this deployment edits it. Turn "
            f"authentication on (auth.method: password or oidc), or point {name!r} at a "
            f"persona that holds neither (the bundled stack's "
            f"{UNPRIVILEGED_TIER_EXAMPLE!r})"
        )
    return problems


def env_var_suffix(username: str) -> str:
    """Map a roster username to the suffix its per-user env vars are keyed by.

    Uppercase, with ``-`` replaced by ``_`` — so ``alice-b`` keys
    ``OSPREY_AUTH_PW_HASH_ALICE_B``. This is the single definition of that
    mapping; credential provisioning, the sidecar's env lookup, and lint all
    route through it so a username can never be keyed one way at mint time and
    another at verify time.

    The mapping is intentionally total and lossy: it neither validates the
    username charset nor rejects anything. Two distinct usernames can therefore
    collide onto one suffix (``alice-b`` and ``alice_b``), which is exactly what
    :func:`env_var_suffix_collisions` exists to detect — enforcement is the
    caller's (a hard raise on the deploy preflight path, an ERROR in lint), not
    this function's.
    """
    return username.upper().replace("-", "_")


def env_var_suffix_collisions(usernames: Iterable[str]) -> dict[str, list[str]]:
    """Find roster usernames that :func:`env_var_suffix` maps onto one suffix.

    Without this check ``alice-b`` and ``alice_b`` would silently share a single
    ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry — one user's password would open the
    other's terminal, which is precisely the isolation the auth feature exists to
    establish.

    A username repeated verbatim in the roster is *not* a collision here: it is
    one user listed twice (a duplicate-name config error reported separately),
    not two users sharing a credential. Only distinct names count.

    Args:
        usernames: Roster usernames — typically ``entry["name"]`` for each
            :func:`normalize_users` entry. Non-string items are ignored, matching
            this module's drop-don't-raise convention.

    Returns:
        ``{suffix: [colliding usernames]}`` for suffixes claimed by two or more
        distinct usernames; empty when the roster is unambiguous. Suffix keys and
        the names under each are sorted, so a lint or preflight message built
        from this is byte-stable across runs.
    """
    by_suffix: dict[str, set[str]] = {}
    for username in usernames:
        if isinstance(username, str):
            by_suffix.setdefault(env_var_suffix(username), set()).add(username)
    return {suffix: sorted(names) for suffix, names in sorted(by_suffix.items()) if len(names) > 1}


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
    return [entry["name"] for entry in normalize_users(subtree.get("users"))]


def effective_image_source(web_terminals: dict[str, Any]) -> str:
    """Coerce ``modules.web_terminals.image_source`` to one of the two real modes.

    ``"local"`` only for the exact literal ``"local"``; every other value
    (unset, ``"registry"``, or an invalid literal) resolves to ``"registry"``.
    This is the single source of that coercion — :func:`resolve_personas`,
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

    This is the single source of that resolution — :func:`resolve_personas` and
    lint both route through it so the tag is read and expanded identically.
    """
    raw = web_terminals.get("image_tag")
    if not isinstance(raw, str):
        raw = "latest"
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), raw)


def _persona_ref_by_name(raw_users: Any) -> dict[str, str]:
    """Recover each roster entry's ``persona`` reference from the raw roster.

    normalize_users() drops any `persona` field off each surviving entry;
    recover it here, keyed by name (the same key every other per-user artifact
    in this module — compose service names, volume names — is keyed by), since
    normalize_users()'s own index-freezing contract is orthogonal to persona
    resolution.
    """
    refs: dict[str, str] = {}
    if isinstance(raw_users, list):
        for raw_entry in raw_users:
            if isinstance(raw_entry, dict):
                name = raw_entry.get("name")
                persona = raw_entry.get("persona")
                if isinstance(name, str) and isinstance(persona, str) and persona:
                    refs[name] = persona
    return refs


def resolve_personas(
    web_terminals: dict[str, Any],
    registry_cfg: dict[str, Any],
    facility_prefix: str,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Resolve each roster entry's persona reference into its image/project identity.

    Layered on :func:`normalize_users`'s output — that function's signature and
    drop-don't-raise contract stay untouched, and this function re-derives each
    surviving entry's ``persona`` field (which :func:`normalize_users` discards)
    from the raw roster by matching on ``name``.

    A ``persona`` reference resolves in this order: the entry's own ``persona:``
    key, else ``modules.web_terminals.default_persona``, else ``None`` (no persona
    system in effect for this entry at all — every config predating persona
    catalogs resolves this way for every entry). Resolving against the catalog
    (``modules.web_terminals.personas.<name>: {project, project_path,
    build_profile}``) then follows these naming rules. In registry mode every
    image carries the tag :func:`resolve_image_tag` resolves from
    ``modules.web_terminals.image_tag`` (``<tag>`` below, default ``latest``);
    local ``:local`` images are unaffected by that field:

    * **No persona in effect** (``persona`` is ``None``): the facility defaults —
      ``image`` is ``<registry_url>/web-terminal:<tag>`` (unsuffixed, the same
      string the compose template names directly whenever ``<tag>`` is its
      ``latest`` default),
      ``project`` and ``container_project_dir`` are ``<facility_prefix>-assistant``
      / ``/app/<facility_prefix>-assistant``. This is the zero-migration path: a
      config with no ``personas`` catalog at all resolves every entry here.
    * **Default persona** (resolved ``persona`` equals ``default_persona``, and
      a catalog entry exists for it): registry mode keeps the same un-suffixed
      ``<registry_url>/web-terminal:<tag>`` image (so the default persona's
      *image* never changes when a catalog is introduced); local mode still
      builds ``<persona.project>-<persona>:local`` like every other persona.
      ``container_project_dir`` is ``/app/<project>`` from the catalog entry,
      exactly as for any other persona: the image is built FROM that project, so
      pinning the directory to the facility default would name a path that
      image does not have. A catalog that gives the default persona a project
      other than ``<facility_prefix>-assistant`` therefore moves where its
      users' agent-data volume mounts — the volume itself is unchanged and
      keeps its contents, but they are no longer at the path the container
      reads.
    * **Non-default persona**: registry mode uses
      ``<registry_url>/web-terminal-<persona>:<tag>``; local mode uses
      ``<persona.project>-<persona>:local`` (same rule as the default persona);
      ``container_project_dir`` is derived from the persona's own
      ``/app/<project>``.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section
            (``users``, ``personas``, ``default_persona``, ``image_source``).
        registry_cfg: The already-dict-coerced top-level ``registry`` section
            (only ``url`` is read).
        facility_prefix: ``facility.prefix``, used for the zero-migration /
            default-persona project dir and image (``<prefix>-assistant``).
        strict: When ``True`` (render/build/seed callers), an unresolvable
            persona reference — an explicit or inherited ``persona:`` naming a
            catalog entry that doesn't exist, or ``image_source: local`` with no
            catalog/``default_persona`` configured at all — raises
            ``ValueError``. When ``False`` (lifecycle verbs), the same
            conditions degrade gracefully: an unresolved entry falls back to the
            zero-migration values (so a stale/bad persona reference never blocks
            ``decommission``/``prune``/``nuke``) instead of raising.

    Returns:
        One ``{"name", "index", "persona", "image", "project",
        "container_project_dir", "extra_mounts", "seed_base"}`` dict per
        surviving :func:`normalize_users` entry, in the same order. ``persona``
        is the resolved catalog key, or ``None`` when no persona is in effect for
        that entry. ``extra_mounts`` is the persona's ``extra_mounts`` list
        (compose volume strings applied to every user of that persona),
        defaulting to ``[]`` — both when no persona is in effect and when the
        catalog entry sets none. ``seed_base`` is the catalog entry's
        ``seed_base`` (a bool; anything else is defensively coerced to
        ``True``), and always ``True`` for the zero-migration / lenient-degrade
        paths — it controls whether the shared base context is prepended when
        seeding this entry's ``CLAUDE.md``. Optional ``"display_name"`` and
        ``"theme"`` keys are added — carried through from
        :func:`normalize_users` — only when the entry declared a non-empty string
        one (render emits them as ``OSPREY_WEB_APP_NAME`` and
        ``OSPREY_WEB_THEME``); each is omitted entirely otherwise, so a roster
        declaring neither resolves byte-identically to before these fields
        existed. An optional ``"oidc_subject"`` key rides through on the same
        terms, so the auth sidecar's roster→identity mapping is read off the same
        resolved entry as everything else rather than re-derived from the raw
        roster. An optional ``"login": False`` marker rides through likewise —
        present only when the roster entry opted out of the login wall (see
        :func:`entry_requires_login`). An optional ``"landing_group"`` key — the catalog entry's own
        ``landing_group``, present only for a non-empty string — names the
        landing-page section this entry's card belongs in; it is read by
        :func:`osprey.deployment.web_terminals.render._build_groups` and affects
        nothing else about the deployment.

    Raises:
        ValueError: See ``strict`` above.
    """
    normalized = normalize_users(web_terminals.get("users"))

    personas_raw = web_terminals.get("personas")
    personas_catalog: dict[str, Any] = personas_raw if isinstance(personas_raw, dict) else {}

    default_persona_name = web_terminals.get("default_persona")
    if not isinstance(default_persona_name, str) or not default_persona_name:
        default_persona_name = None

    image_source = effective_image_source(web_terminals)
    image_tag = resolve_image_tag(web_terminals)

    registry_url = ""
    if isinstance(registry_cfg, dict):
        url = registry_cfg.get("url")
        if isinstance(url, str):
            registry_url = url

    persona_ref_by_name = _persona_ref_by_name(web_terminals.get("users"))

    if (
        strict
        and image_source == "local"
        and (not personas_catalog or default_persona_name is None)
    ):
        raise ValueError(
            "modules.web_terminals.image_source: local requires both a "
            "modules.web_terminals.personas catalog and default_persona to be "
            "configured"
        )

    default_project = f"{facility_prefix}-assistant"
    default_container_dir = f"/app/{facility_prefix}-assistant"
    default_image = f"{registry_url}/web-terminal:{image_tag}"

    def _with_optional_fields(entry: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        """Attach the optional per-user fields to a resolved entry, mirroring
        render.py's conditional-``sublabel`` convention: a key is present only
        for a non-empty string, so a roster declaring none leaves the entry
        byte-identical to a resolution from before these fields existed."""
        for field in ("display_name", "theme", "oidc_subject"):
            value = source.get(field)
            if isinstance(value, str) and value:
                entry[field] = value
        # `login: False` rides through on the same present-only-when-set terms;
        # normalize_users already reduced every other spelling to absence.
        if source.get("login") is False:
            entry["login"] = False
        return entry

    def _zero_migration_entry(
        name: str, index: int, persona: str | None, source: dict[str, Any]
    ) -> dict[str, Any]:
        """The zero-migration resolution: the pre-persona values, with
        ``persona`` carried through for logging (``None`` when no persona is in
        effect, or the unresolvable reference on the lenient degrade path) and the
        optional per-user fields passed through unchanged. ``extra_mounts`` is
        empty here — the zero-migration path has no catalog entry to read
        persona-level host mounts from. ``seed_base`` is ``True`` — the shared
        base-context prepend is mandatory for a no-persona/zero-migration entry,
        and opting out is only expressible through a catalog entry."""
        return _with_optional_fields(
            {
                "name": name,
                "index": index,
                "persona": persona,
                "image": default_image,
                "project": default_project,
                "container_project_dir": default_container_dir,
                "extra_mounts": [],
                "seed_base": True,
            },
            source,
        )

    resolved: list[dict[str, Any]] = []
    for entry in normalized:
        name = entry["name"]
        index = entry["index"]
        persona_ref = persona_ref_by_name.get(name) or default_persona_name

        if persona_ref is None:
            # No persona system in effect for this entry — zero-migration path.
            resolved.append(_zero_migration_entry(name, index, None, entry))
            continue

        catalog_entry = personas_catalog.get(persona_ref)
        if not isinstance(catalog_entry, dict):
            if strict:
                raise ValueError(
                    f"user {name!r} references persona {persona_ref!r}, which has "
                    "no entry in modules.web_terminals.personas"
                )
            # Lenient degrade (lifecycle verbs): keep the requested persona name
            # visible for logging, but fall back to the zero-migration values so
            # a stale/bad reference never blocks a lifecycle verb.
            resolved.append(_zero_migration_entry(name, index, persona_ref, entry))
            continue

        project = catalog_entry.get("project")
        if not isinstance(project, str) or not project:
            project = default_project

        # Persona-level host mounts, applied to every user of this persona. A
        # non-list drops to []; individual non-string/empty entries are dropped
        # (well-formedness — the colon-part syntax — is lint's job).
        extra_mounts_raw = catalog_entry.get("extra_mounts")
        extra_mounts = (
            [mount for mount in extra_mounts_raw if isinstance(mount, str) and mount]
            if isinstance(extra_mounts_raw, list)
            else []
        )

        # seed_base: whether this persona's users get the shared base context
        # prepended ahead of their own extra context at seed time. Defaults to
        # True (always prepend); a non-bool value is a
        # config typo that lint reports separately, so coerce it back to the
        # safe default here rather than propagating garbage.
        seed_base = catalog_entry.get("seed_base")
        if not isinstance(seed_base, bool):
            seed_base = True

        # landing_group: which landing-page section this persona's users appear
        # under. A non-empty string lifts them out of the roster's default
        # terminals section into a section of that name — how a deployment says
        # "this login is not a person, it is a standalone service" (see
        # render._build_groups). Presentation only: it changes no image, port,
        # route, volume or entitlement. Absent or non-string — every catalog
        # predating this key — leaves the entry in the default section, so the
        # key is purely additive and a resolution without it is byte-identical
        # to what it was before (the `_with_optional_fields` convention).
        landing_group = catalog_entry.get("landing_group")

        is_default = persona_ref == default_persona_name

        if image_source == "local":
            image = f"{project}-{persona_ref}:local"
        elif is_default:
            image = default_image
        else:
            image = f"{registry_url}/web-terminal-{persona_ref}:{image_tag}"

        container_project_dir = f"/app/{project}"

        entry_resolved = _with_optional_fields(
            {
                "name": name,
                "index": index,
                "persona": persona_ref,
                "image": image,
                "project": project,
                "container_project_dir": container_project_dir,
                "extra_mounts": extra_mounts,
                "seed_base": seed_base,
            },
            entry,
        )
        if isinstance(landing_group, str) and landing_group:
            entry_resolved["landing_group"] = landing_group
        resolved.append(entry_resolved)

    return resolved


# ---------------------------------------------------------------------------
# Shipped-artifact reads
#
# Everything above answers a question about a persona's *config* — its authored
# intent. The two functions below deliberately do not: they read the built
# `.claude/settings.json` artifact instead, and so are kept out of the
# entitlement-predicate cluster above rather than joining it.
# ---------------------------------------------------------------------------


#: The exact ``permissions.deny`` entry that blocks the agent's shell wholesale.
#: A *scoped* deny (``Bash(rm:*)``) constrains one command family and leaves the
#: shell otherwise usable, so only this literal counts as "Bash is denied".
_BASH_DENY_ENTRY = "Bash"


def settings_json_denies_bash(project_dir: Any) -> bool:
    """True if ``<project_dir>/.claude/settings.json`` denies ``Bash`` outright.

    Reads the **shipped build artifact**, not the ``config.yml`` that produced
    it, and that distinction is the whole point of this function. ``osprey up``
    does not rebuild (rebuilding is a separate operator step), so a persona's
    config edited after its last build does not change what its image actually
    ships. A caller that asked the config instead would see a permission set
    that has never been rendered: an operator who removed ``Bash`` from
    ``remove_deny`` would read as safe while the running image still permits the
    shell.

    The scope of that claim is the rendered project directory, which is what
    ``COPY . /app/<project>/`` bakes into the persona image — so what this reads
    equals the image's own ``.claude/settings.json`` **as of that image's last
    build**, and the per-user compose services declare only ``image:``, with no
    ``build:`` stanza to rebuild one at start. A guard built on this is
    therefore sound *provided persona images are rebuilt after a render*, and
    deliberately claims nothing beyond that: a render newer than its image can
    report a deny the running image does not yet carry. Reading ``config.yml``
    would not close that window either — it is one step further from the image,
    not one step closer.

    Only the exact ``"Bash"`` entry counts (see :data:`_BASH_DENY_ENTRY`).

    Fails **closed**: an artifact that cannot be read and parsed into a
    ``permissions.deny`` list is not evidence that anything is denied, so every
    such case answers ``False`` rather than raising or defaulting to "safe".
    That covers an absent file, unreadable or invalid JSON, a top-level value
    that is not an object, and a missing or non-list ``permissions`` /
    ``permissions.deny``.

    Args:
        project_dir: The rendered persona project directory (the one holding
            ``config.yml`` and ``.claude/``).

    Returns:
        ``True`` only when the artifact was read, parsed, and lists ``"Bash"``
        in ``permissions.deny``; ``False`` in every other case.
    """
    settings_json = Path(project_dir) / ".claude" / "settings.json"
    try:
        # utf-8-sig, not utf-8: `json.load` does not strip a BOM, so a hand-edited
        # artifact saved with one would fail to parse and a genuinely Bash-denying
        # persona would read as permissive — an opaque deploy refusal. The codec is
        # a strict superset for BOM-less files, which is everything OSPREY renders.
        with settings_json.open("r", encoding="utf-8-sig") as fh:
            settings = json.load(fh)
    except (OSError, ValueError):
        return False
    deny = as_dict(as_dict(settings).get("permissions")).get("deny")
    if not isinstance(deny, list):
        return False
    return _BASH_DENY_ENTRY in [entry for entry in deny if isinstance(entry, str)]


def personas_not_denying_bash(config: Any, project_root: Any) -> set[str]:
    """Names of referenced personas whose shipped settings do **not** deny ``Bash``.

    The roster-shaped form of :func:`settings_json_denies_bash`, phrased as the
    *unsafe* set so its caller can name every offending persona in one error
    rather than re-deriving them. A persona granted ``BLUESKY_LAUNCH_TOKEN``
    while its agent may also run a shell can read that token out of its own
    environment and arm a queue with a plain HTTP call, bypassing the chat
    approval entirely — the approval gates the ``queue_start`` tool, not a
    shell. Intersecting this set with :func:`personas_needing_launch_token`
    therefore names exactly the personas a deploy must refuse.

    Walks the roster itself rather than routing through the shared
    ``config.yml`` engine the entitlement predicates use, because the question
    is about the built artifact and not about intent — see
    :func:`settings_json_denies_bash`. Membership is inclusive for the same
    fail-closed reason: a persona with no ``project_path``, or one whose project
    is not rendered, is reported here as not denying ``Bash``. That never
    over-blocks a deploy, since a persona whose project is missing cannot
    satisfy :func:`config_needs_launch_token` either and so drops out of the
    intersection anyway.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        The subset of referenced persona names whose rendered
        ``.claude/settings.json`` does not deny the shell.
    """
    catalog, referenced = _referenced_personas(config)

    permitting: set[str] = set()
    for persona_name in sorted(referenced):
        project_path = as_dict(catalog.get(persona_name)).get("project_path")
        if not isinstance(project_path, str) or not project_path:
            permitting.add(persona_name)
            continue
        if not settings_json_denies_bash(Path(project_root, project_path)):
            permitting.add(persona_name)
    return permitting
