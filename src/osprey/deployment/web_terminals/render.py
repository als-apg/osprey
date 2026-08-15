"""Renders the multi-user web-terminal deployment artifacts from a facility config's
``modules.web_terminals`` stanza.

Three artifacts come out of one facility config: the docker-compose overlay (one
service per user + nginx), the nginx routing fragment, and the static landing page.
All port arithmetic is delegated to :func:`osprey.deployment.web_terminals.ports.allocate_ports`
— this module only builds the per-service context list and hands it to Jinja2.
"""

from __future__ import annotations

import posixpath
import re
from importlib.resources import as_file, files
from pathlib import PurePosixPath
from typing import Any

from jinja2 import Environment, FileSystemLoader

from osprey.deployment.compose_generator import repo_identity, resolve_repo_root
from osprey.deployment.web_terminals.personas import (
    SUPPORTED_MCP_TOPOLOGY,
    USERNAME_CHARSET_RE,
    as_dict,
    config_needs_ariel_password,
    config_needs_dispatcher_token,
    config_needs_launch_token,
    effective_image_source,
    env_var_suffix,
    env_var_suffix_collisions,
    resolve_personas,
)
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.utils.facility import resolve_facility_name
from osprey.utils.workspace import agent_data_base_dir

# Package-relative location of the .j2 sources (Tasks 1.3/1.6). Resolved via
# importlib.resources, NOT Path(__file__).parent, so this works from an installed
# wheel too (hatchling ships all of src/osprey as package data).
_TEMPLATE_PACKAGE_PATH = "templates/modules/web_terminals"

_COMPOSE_TEMPLATE = "docker-compose.web.yml.j2"
_NGINX_TEMPLATE = "nginx.conf.j2"
_LANDING_TEMPLATE = "landing.html.j2"

# Output paths are relative to the rendered docker-compose.web.yml.j2 itself, per
# that template's own mount-path contract (nginx/nginx.conf, nginx/landing.html).
_COMPOSE_OUTPUT = "docker-compose.web.yml"
_NGINX_OUTPUT = "nginx/nginx.conf"
_LANDING_OUTPUT = "nginx/landing.html"

# Per-container constant: every per-user app's service families (web + every
# registry companion family) bind this host, never a routable interface —
# nginx's reverse proxy is the only off-host path. Not
# config-driven: unlike the per-family ports, there is no config knob
# for this, since a facility that wants a per-user port reachable directly
# off-host would defeat the single-origin chokepoint this module exists to
# provide.
_LOOPBACK_BIND_HOST = "127.0.0.1"

# Default nginx image when `modules.web_terminals.nginx_image` is unset. Kept
# byte-identical to docker-compose.web.yml.j2's own `| default(...)` fallback so
# an absent config value renders the same image from either side.
_DEFAULT_NGINX_IMAGE = "nginx:1.27-alpine"

#: The authentication methods this deployment can actually serve: ``none`` (no
#: auth at all), ``password`` (OSPREY-managed scrypt hashes) and ``oidc``
#: (Authlib client against a facility IdP). Anything else is a hard config
#: error rather than a forward-compatible passthrough — nginx would emit the
#: ``auth_request`` seam against a sidecar that cannot answer for that method,
#: which does fail closed but only as an unactionable 403 on every request.
SUPPORTED_AUTH_METHODS = ("none", "password", "oidc")

#: Port the auth sidecar listens on when ``auth.port`` is unset. Deliberately
#: outside the per-user port families (which start at ``*_base_port`` and grow
#: with the roster) so a default-port sidecar can't collide with user N's
#: terminal; lint joins the effective value into its port-collision check.
_DEFAULT_AUTH_PORT = 9070

#: How long an unlocked-user entry stays valid when ``auth.session_lifetime``
#: is unset, in seconds (12 hours — one operator shift plus slack).
_DEFAULT_SESSION_LIFETIME = 12 * 60 * 60

#: Env-var *names* (never values) the sidecar reads its OIDC client credentials
#: from when the config doesn't name its own. The credentials themselves live
#: in ``.env.auth`` and never enter config.yml.
_DEFAULT_OIDC_CLIENT_ID_ENV = "OSPREY_AUTH_OIDC_CLIENT_ID"
_DEFAULT_OIDC_CLIENT_SECRET_ENV = "OSPREY_AUTH_OIDC_CLIENT_SECRET"

#: Label key the auth sidecar's service carries the sha256 digest of
#: ``.env.auth``'s content under (repo label convention: dotted ``osprey.*``
#: keys, as in the service templates' ``osprey.project.name``). Compose bakes
#: ``env_file`` content into a container at CREATION time, and a
#: service-definition change is the only recreate trigger every compose
#: implementation honours — podman-compose in particular never recreates on a
#: content-only file change. Stamping the digest into the definition turns an
#: ``.env.auth`` edit into exactly that trigger. The digest is not a secret:
#: the file's values are high-entropy mints and IdP-issued credentials, and
#: only the hash of the whole file — never a variable's value — is emitted.
AUTH_ENV_DIGEST_LABEL = "osprey.auth.env.digest"


def _container_agent_data_dir(config: Any, container_project_dir: str) -> str:
    """Where a per-user container's agent-data volume mounts, as that container sees it.

    The mount target has to be the directory the process inside the container
    actually writes to, and that directory is decided by ``agent_data.base_dir``
    — the same key :func:`osprey.utils.workspace.resolve_agent_data_root`
    resolves at runtime, anchored on the config's ``project_root`` (in-container:
    *container_project_dir*). Derived through the shared
    :func:`~osprey.utils.workspace.agent_data_base_dir` reader for that reason:
    a literal here is a second, silent spelling of the same setting, and a
    project that relocates its agent-data root would mount the volume at the old
    path while the agent writes to the new one — an unbacked directory in the
    container's writable layer, discarded at the next recreate.

    The join happens here rather than in the template because an absolute
    ``base_dir`` names the same absolute path on both sides and must NOT be
    re-anchored under the project directory; a template-level
    ``{{ dir }}/{{ base }}`` concatenation cannot make that distinction.

    Known limit: a ``~``-relative ``base_dir`` is anchored here like any other
    relative path, while ``anchored_path`` expands it against the container's
    ``HOME`` — which the compose template sets to ``/data/claude-config``. Such
    a project's agent data therefore lands in the claude-config volume rather
    than the agent-data one: misfiled, but persisted by the claude-config volume,
    so this is not the unbacked-directory data loss above (the agent-data volume
    then mounts over an empty ``~`` directory and holds nothing). Left unhandled rather
    than expanded here because resolving it would mean spelling that ``HOME``
    path a second time in this module, free to drift from the template that
    actually sets it — the same duplication this function exists to remove.
    """
    base = PurePosixPath(agent_data_base_dir(config))
    if base.is_absolute():
        return base.as_posix()
    return (PurePosixPath(container_project_dir) / base).as_posix()


def render_web_terminals(
    config: Any,
    auth_env_digest: str | None = None,
    dispatcher_personas: set[str] | None = None,
    ariel_personas: set[str] | None = None,
    launch_token_personas: set[str] | None = None,
) -> dict[str, str]:
    """Render the compose overlay, nginx fragment, and landing page for one facility config.

    Args:
        config: The parsed facility config, read defensively as nested dicts (same
            convention as :func:`osprey.deployment.web_terminals.lint.lint_web_terminals`
            — no assumption that ``config`` is a particular schema/dataclass type).
            Deterministic: the same inputs always render the same three artifacts,
            with no clock/random/filesystem inputs — which is why the digest below
            arrives as a parameter instead of being read from disk here.
        auth_env_digest: sha256 hex digest of ``.env.auth``'s current content,
            emitted as the :data:`AUTH_ENV_DIGEST_LABEL` label on the auth
            sidecar's service so a content change becomes a service-DEFINITION
            change — the one recreate trigger every compose implementation
            honours. The deploy path computes it via
            :func:`osprey.deployment.web_terminals.artifacts.write_web_terminal_artifacts`;
            ``None`` (the default, and the ``osprey scaffold web-terminals
            render`` path, which has no project root to digest) emits no label —
            harmless, because every ``osprey up`` re-renders through the
            artifacts seam with a current digest before its ``up``, so a
            label-less render can never reach a running stack stale.
        dispatcher_personas: Persona names whose project declares the EVENTS
            panel, and whose users therefore need the event dispatcher's bearer
            token. Resolved from disk by
            :func:`osprey.deployment.web_terminals.personas.personas_needing_dispatcher_token`
            and passed in for the same reason ``auth_env_digest`` is: this
            function reads no filesystem. Each such user gets
            ``EVENT_DISPATCHER_TOKEN`` in its OWN ``environment:`` block,
            interpolated by compose from the deploy ``.env`` — never written
            into this rendered artifact, and never into the shared
            ``.env.users``, which every user's container reads (see
            :func:`osprey.deployment.web_terminals.env_production._build_env_production_subset`).
            That split IS the tier boundary: a read-only persona must not hold a
            credential that can fire triggers. ``None`` (the default, and the
            ``osprey scaffold web-terminals render`` path, which has no project
            root to resolve against) emits no token line at all.
        ariel_personas: Persona names whose project configures ARIEL, and whose
            users therefore need the Postgres password ``osprey up`` minted into
            the deploy ``.env`` (see
            :func:`osprey.deployment.web_terminals.personas.personas_needing_ariel_password`).
            Same placement and same reason as ``dispatcher_personas``: both ARIEL
            consumers inside the container — the panel's server and the ``ariel``
            MCP server — resolve their DSN through
            :func:`osprey.services.ariel_search.config.resolve_ariel_dsn`, which
            reads ``ARIEL_DB_PASSWORD`` from the environment and otherwise falls
            back to the shipped default, so a container that never receives it
            authenticates with the wrong password against a Postgres initialized
            with the minted one. ``None`` emits no line.
        launch_token_personas: Persona names entitled to arm a Bluesky queue
            start, resolved from disk by
            :func:`osprey.deployment.web_terminals.personas.personas_needing_launch_token`.
            Same placement and same reason as ``dispatcher_personas`` — see that
            argument for why a per-persona credential belongs in the per-user
            ``environment:`` block and never in the shared ``.env.users``.
            The tier boundary bites hardest here: ``BLUESKY_LAUNCH_TOKEN`` is
            what lets the ``bluesky`` MCP server arm a queue start without a
            second confirmation, so a read-only persona holding it would be able
            to move hardware. ``None`` emits no line.

    Returns:
        Mapping of output-relative-path to rendered content, for exactly three
        artifacts: ``docker-compose.web.yml``, ``nginx/nginx.conf``, and
        ``nginx/landing.html``.

    Raises:
        ValueError: If ``modules.web_terminals.nginx_port`` is missing/not an int,
            if a configured user can't resolve a full port-family set, if
            ``deploy.fqdn`` is missing while the deployment needs an external
            origin — at least one user configured (the host baked into
            ``OSPREY_TERMINAL_LANDING_URL``) or authentication enabled (see
            :func:`_external_origin`), if the ``auth``/``tls`` stanzas are
            incoherent (unknown ``auth.method``, ``tls.enabled`` without a
            cert/key pair, or authentication over cleartext HTTP without
            ``auth.allow_insecure_http``), if
            a roster entry's persona reference can't be resolved (see
            :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
            ``strict`` contract — render always resolves strictly), or if
            ``modules.web_terminals.mcp.topology`` is set to anything other than
            ``per_container_stdio`` (see :func:`_check_mcp_topology`), or if
            ``auth_env_digest`` is not a sha256 hex digest — the value is
            templated verbatim into compose YAML, so anything but lowercase hex
            is rejected here rather than trusted not to carry YAML structure.
    """
    if auth_env_digest and not re.fullmatch(r"[0-9a-f]{64}", auth_env_digest):
        raise ValueError("auth_env_digest must be a sha256 hex digest")
    root = as_dict(config)
    facility = as_dict(root.get("facility"))
    registry = as_dict(root.get("registry"))
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    facility_prefix = facility.get("prefix") or ""

    _check_mcp_topology(web_terminals)

    resolved_users = resolve_personas(web_terminals, registry, facility_prefix, strict=True)

    base_ports = base_ports_from_config(web_terminals)
    services = []
    for entry in resolved_users:
        user_ports = allocate_ports(base_ports, entry["index"])
        services.append(
            {
                "user": entry["name"],
                "image": entry["image"],
                "project": entry["project"],
                "container_project_dir": entry["container_project_dir"],
                # Where this user's agent-data volume mounts INSIDE the
                # container, resolved from the same key the process inside it
                # resolves its own agent-data root from.
                # Read from `agent_data.base_dir` through the shared helper
                # rather than spelled as a literal here, because the container
                # resolves its own agent-data root from that key
                # (workspace.resolve_agent_data_root, anchored on the config's
                # project_root, which in-container is container_project_dir).
                # A literal here silently decouples the two: the volume mounts
                # at one path while memory, sessions and artifacts are written
                # to another, which is a plain directory in the container's
                # writable layer — so every user's data is discarded the next
                # time the container is recreated, with nothing to see at
                # mount time or in any log.
                "container_agent_data_dir": _container_agent_data_dir(
                    root, entry["container_project_dir"]
                ),
                "extra_mounts": entry["extra_mounts"],
                # Optional per-user window/tab title -> OSPREY_WEB_APP_NAME. None
                # (the common case: resolve_personas omits the key unless a
                # non-empty string was set) leaves the template's `{% if
                # svc.display_name %}` guard false, so no env line is emitted and
                # the app falls back to config web.app_name.
                "display_name": entry.get("display_name"),
                # Optional per-user default theme -> OSPREY_WEB_THEME. Same
                # shape as display_name above: absent (the common case) leaves
                # the template's `{% if svc.theme %}` guard false, so no env
                # line is emitted and the app falls back to config web.theme.
                "theme": entry.get("theme"),
                # Optional per-user OIDC identity -> the auth sidecar's
                # OSPREY_AUTH_OIDC_SUBJECT_<SUFFIX>. Same shape again: absent
                # leaves the compose template's `{% if svc.oidc_subject %}`
                # guard false. Without this passthrough an `oidc` deployment
                # would render no subject mapping at all and every IdP identity
                # would be unmappable.
                "oidc_subject": entry.get("oidc_subject"),
                # The env-var name that subject is emitted under, resolved HERE
                # through env_var_suffix() — the single definition of the
                # username->env-var mapping, shared with credential
                # provisioning, the sidecar's own lookup and lint. The template
                # emits this verbatim rather than re-deriving `upper|replace`,
                # which would be a second copy of the mapping free to drift from
                # the one that keys the password hashes.
                "oidc_subject_env": f"OSPREY_AUTH_OIDC_SUBJECT_{env_var_suffix(entry['name'])}",
                **user_ports,
                # One env line per companion family, derived from the web-server
                # registry (PANEL_ENV_VARS) so a newly registered companion is
                # multi-user-wired without touching this module or the template.
                # The "web" family is not in this list — the template exports
                # its OSPREY_TERMINAL_WEB_PORT/OSPREY_WEB_PORT pair explicitly.
                "panel_env": [
                    {"name": env_var, "port": user_ports[family]}
                    for family, env_var in PANEL_ENV_VARS.items()
                ],
                # Whether this user's container gets the event dispatcher's
                # bearer (see the `dispatcher_personas` arg). A persona-less
                # roster entry — the zero-migration path, where the web image
                # IS the deploy project — is answered from this same config,
                # with no disk read, so the determinism contract holds either
                # way.
                "wants_dispatcher": (
                    entry["persona"] in (dispatcher_personas or set())
                    if entry.get("persona")
                    else config_needs_dispatcher_token(root)
                ),
                # Whether this user's container gets the ARIEL Postgres password
                # (see the `ariel_personas` arg). Persona-less entries are
                # answered from this same config, with no disk read, exactly as
                # above.
                "wants_ariel_db": (
                    entry["persona"] in (ariel_personas or set())
                    if entry.get("persona")
                    else config_needs_ariel_password(root)
                ),
                # Whether this user's container gets the Bluesky queue's launch
                # token (see the `launch_token_personas` arg). Persona-less
                # entries are answered from this same config, with no disk read,
                # exactly as above.
                "wants_launch_token": (
                    entry["persona"] in (launch_token_personas or set())
                    if entry.get("persona")
                    else config_needs_launch_token(root)
                ),
            }
        )

    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        raise ValueError("modules.web_terminals.nginx_port is required and must be an int")

    image_source = effective_image_source(web_terminals)

    auth_tls_ctx = _auth_tls_context(web_terminals)
    if auth_tls_ctx["tls_enabled"] and not (auth_tls_ctx["tls_cert"] and auth_tls_ctx["tls_key"]):
        raise ValueError(
            "modules.web_terminals.tls.enabled is true but tls.cert/tls.key are not both "
            "set — the gated `listen 443 ssl` seam needs both paths to emit a "
            "coherent ssl_certificate/ssl_certificate_key pair"
        )
    _check_tls_host_cert_dir(auth_tls_ctx)
    if (
        auth_tls_ctx["auth_method"] != "none"
        and not auth_tls_ctx["tls_enabled"]
        and not auth_tls_ctx["auth_allow_insecure_http"]
    ):
        raise ValueError(
            f"modules.web_terminals.auth.method is {auth_tls_ctx['auth_method']!r} but "
            "tls.enabled is false — session cookies would cross the network in "
            "cleartext, so login is refused rather than rendered insecurely. Enable "
            "tls, or set auth.allow_insecure_http: true to accept that risk (only "
            "sensible behind a TLS-terminating proxy or on an isolated network)"
        )
    _check_roster_charset(services, auth_tls_ctx["auth_method"])
    _check_roster_env_var_collisions(services, auth_tls_ctx["auth_method"])

    tls_enabled = auth_tls_ctx["tls_enabled"]
    # The one origin every absolute URL in this deployment is built from (see
    # _external_origin). Derived only when something actually needs it: a
    # roster-less config with no auth has no absolute URL to build, and must
    # keep rendering without a deploy.fqdn.
    external_origin = (
        _external_origin(root, nginx_port, tls_enabled=tls_enabled)
        if services or auth_tls_ctx["auth_method"] != "none"
        else ""
    )
    landing_url = _landing_url(root, nginx_port, tls_enabled=tls_enabled) if services else ""

    compose_ctx = {
        "facility_prefix": facility_prefix,
        "registry_url": registry.get("url") or "",
        "image_source": image_source,
        "services": services,
        "nginx_port": nginx_port,
        "landing_url": landing_url,
        "facility_timezone": facility.get("timezone") or "UTC",
        "bind_host": _LOOPBACK_BIND_HOST,
        # Read defensively (as everywhere else in this module); the template
        # carries the same default, so an absent/blank value still renders the
        # public tag. Facilities whose images all come from a private registry
        # override this so the nginx image is pullable too.
        "nginx_image": web_terminals.get("nginx_image") or _DEFAULT_NGINX_IMAGE,
        # The auth sidecar's service (image, listen port, session lifetime, the
        # env-var names its OIDC credentials arrive under) is rendered from the
        # same parsed stanza the nginx seam reads, so the two ends of the
        # auth_request contract can't drift apart.
        "external_origin": external_origin,
        # Empty string (not None) when no digest was supplied, so the template
        # can gate the label block on plain truthiness.
        "auth_env_digest": auth_env_digest or "",
        "auth_env_digest_label": AUTH_ENV_DIGEST_LABEL,
        # Which CHECKOUT this stack belongs to, baked into every container and
        # volume label the template emits. Derived through the same helper the
        # services stack renders from, so one deployment cannot end up with two
        # identities depending on which of its two compose files is read.
        "repo_id": repo_identity(resolve_repo_root(config)),
        **auth_tls_ctx,
    }

    nginx_ctx = {
        "nginx_port": nginx_port,
        "services": services,
        "bind_host": _LOOPBACK_BIND_HOST,
        "external_origin": external_origin,
        **auth_tls_ctx,
    }
    landing_ctx = {
        "facility_name": resolve_facility_name(root, ""),
        "groups": _build_groups(as_dict(web_terminals.get("landing")), resolved_users),
        "theme_blocks": _landing_theme_blocks(root),
    }

    template_dir = files("osprey").joinpath(_TEMPLATE_PACKAGE_PATH)
    with as_file(template_dir) as template_path:
        # Compose + nginx emit YAML/conf, not HTML — autoescape=False, matching
        # compose_generator.py's own Jinja2 convention. Landing emits HTML and is
        # rendered with autoescape=True; landing.html.j2 also `|e`-escapes every
        # interpolation itself as defense-in-depth.
        conf_env = Environment(loader=FileSystemLoader(str(template_path)), autoescape=False)
        html_env = Environment(loader=FileSystemLoader(str(template_path)), autoescape=True)

        rendered_compose = conf_env.get_template(_COMPOSE_TEMPLATE).render(**compose_ctx)
        rendered_nginx = conf_env.get_template(_NGINX_TEMPLATE).render(**nginx_ctx)
        rendered_landing = html_env.get_template(_LANDING_TEMPLATE).render(**landing_ctx)

    return {
        _COMPOSE_OUTPUT: rendered_compose,
        _NGINX_OUTPUT: rendered_nginx,
        _LANDING_OUTPUT: rendered_landing,
    }


#: The design-system custom properties the landing page's inline stylesheet
#: uses. Read by name out of the resolved theme rather than copied as a whole
#: emitted block: the landing page is a standalone file with no stylesheet to
#: fall back on, so it wants only what it uses — and a rename must fail the
#: render loudly (see ``theme_css_variables``) instead of leaving a deployed
#: page quietly off-palette.
_LANDING_THEME_VARIABLES = (
    "--bg-primary",
    "--bg-panel",
    "--bg-elevated",
    "--text-primary",
    "--text-secondary",
    "--text-muted",
    "--color-accent",
    "--border-default",
    "--border-accent",
    # The downward card shadow, not --shadow-panel: that one is a drawer edge
    # shadow (`-4px 0 ...`), which reads as a left-hand glow on a card.
    "--shadow-dropdown",
)

#: What a baked custom-property value is allowed to look like. Values come from
#: the framework's own token tree rather than from operator config, so this is a
#: belt-and-braces guard on a string that lands inside a ``<style>`` element,
#: where HTML escaping would not save us from a ``</style>`` sequence.
_SAFE_CSS_VALUE_RE = re.compile(r"^[#A-Za-z0-9 ,.()%/_-]+$")


def _landing_theme_blocks(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the landing page's inline theme blocks from the deployment's ``web.theme``.

    The landing page is served by nginx as a flat file with no application
    behind it — it cannot link ``tokens.css``, and it has no theme picker and no
    ``localStorage``. So its palette is resolved here, at render time, and baked
    into the page:

    - ``web.theme`` naming a **family** (``"desy"``) yields TWO blocks: that
      family's dark values under ``:root``, and its light values under a
      ``prefers-color-scheme: light`` media query. Light/dark then follows each
      viewer's OS, which is the same thing the terminals behind this page do
      when configured with a bare family.
    - ``web.theme`` naming a **concrete id** (``"desy-light"``) yields ONE
      block and no media query — the deployment pinned a mode, and the landing
      page honors that pin exactly as the terminals do.

    The landing page deliberately uses the deployment-level theme and never a
    per-user one: it is shown before anyone has identified themselves.

    Args:
        root: The parsed facility config.

    Returns:
        Block dicts for ``landing.html.j2``: ``{"media": str|None,
        "color_scheme": str, "variables": [(name, value), ...]}``.

    Raises:
        ValueError: If a resolved value fails :data:`_SAFE_CSS_VALUE_RE`.
        MissingThemeVariableError: If the resolved theme does not define one of
            :data:`_LANDING_THEME_VARIABLES`.
    """
    from osprey.interfaces.design_system.theme_config import (
        family_of,
        load_theme_registry,
        resolve_pinned_mode,
        resolve_theme_id,
        theme_css_variables,
    )

    configured = str(as_dict(root.get("web")).get("theme") or "main")
    entries, defaults = load_theme_registry()
    resolved_id = resolve_theme_id(configured, entries, defaults, config_key="web.theme")
    pinned_mode = resolve_pinned_mode(configured, entries)

    #: (media query or None, color-scheme, theme id) per block to emit.
    wanted: list[tuple[str | None, str, str]]
    if pinned_mode is not None:
        wanted = [(None, pinned_mode, resolved_id)]
    else:
        family = family_of(resolved_id, entries) or "main"
        family_modes = defaults[family]
        wanted = [
            (None, "dark", family_modes["dark"]),
            ("(prefers-color-scheme: light)", "light", family_modes["light"]),
        ]

    blocks: list[dict[str, Any]] = []
    for media, color_scheme, theme_id in wanted:
        variables = theme_css_variables(theme_id, _LANDING_THEME_VARIABLES)
        for name, value in variables.items():
            if not _SAFE_CSS_VALUE_RE.match(value):
                raise ValueError(
                    f"theme {theme_id!r} value for {name!r} is not safe to inline "
                    f"into the landing page's <style>: {value!r}"
                )
        blocks.append(
            {
                "media": media,
                "color_scheme": color_scheme,
                "variables": list(variables.items()),
            }
        )
    return blocks


def _external_origin(root: dict[str, Any], nginx_port: int, *, tls_enabled: bool) -> str:
    """Build the one origin every absolute URL this deployment emits is derived from.

    Two consumers need an absolute URL that a browser will actually resolve: the
    landing link baked into each container (:func:`_landing_url`) and the auth
    sidecar's OIDC ``redirect_uri``. Those two must agree exactly — an IdP
    rejects a ``redirect_uri`` that isn't character-for-character the registered
    one, and a landing link on a different origin would drop the session cookie
    — so both come from here rather than being assembled twice.

    Scheme and port follow ``tls.enabled``: with TLS on, the 443 server is the
    sole content server (the plain listener only redirects to it), and 443 is
    left implicit, which is both the canonical origin serialization and the form
    an IdP client registration is normally written in. With TLS off the origin
    is plain HTTP on the published ``nginx_port``.

    The host comes from ``deploy.fqdn``: the schema documents that field as
    reachable from developers' laptops (used in client-mode profiles), whereas
    ``deploy.host`` is only guaranteed SSH-resolvable (may be a bare
    `~/.ssh/config` alias, not a browser-reachable hostname).

    Args:
        root: The parsed facility config (only ``deploy.fqdn`` is read).
        nginx_port: The published plain-HTTP port, used only when TLS is off.
        tls_enabled: The parsed ``tls.enabled`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If ``deploy.fqdn`` is missing or blank.
    """
    deploy = as_dict(root.get("deploy"))
    host = str(deploy.get("fqdn") or "").strip()
    if not host:
        raise ValueError(
            "deploy.fqdn is required to render modules.web_terminals landing_url "
            "(OSPREY_TERMINAL_LANDING_URL) when at least one user is configured "
            "or authentication is enabled"
        )
    return f"https://{host}" if tls_enabled else f"http://{host}:{nginx_port}"


def _landing_url(root: dict[str, Any], nginx_port: int, *, tls_enabled: bool = False) -> str:
    """The absolute origin baked into every service's ``OSPREY_TERMINAL_LANDING_URL``.

    Per-user containers only get this value once, at container start (env vars, not
    request time) — unlike nginx.conf.j2's per-request ``$host`` redirect target,
    resolving it can't be deferred to the browser. It is the deployment's external
    origin verbatim (:func:`_external_origin`), which is what keeps a "back to
    landing" link and an OIDC ``redirect_uri`` on the same origin by construction.
    """
    return _external_origin(root, nginx_port, tls_enabled=tls_enabled)


def _user_card(resolved_user: dict[str, Any]) -> dict[str, Any]:
    """Build one auto-populated ``users``-group landing card from a resolved roster entry.

    The base shape is unchanged (``{label, url}``); a ``sublabel`` key is added
    only when the entry resolved to a persona, so a no-persona roster keeps
    producing exactly the same two-key items landing.html.j2 rendered before.

    Args:
        resolved_user: One :func:`osprey.deployment.web_terminals.personas.resolve_personas`
            entry (``name`` and ``persona`` are read).

    Returns:
        ``{"label", "url"}`` for a persona-less user, plus ``"sublabel"`` (the
        persona name) when ``persona`` is a non-empty string.
    """
    name = resolved_user["name"]
    card: dict[str, Any] = {"label": name, "url": f"/u/{name}/"}
    persona = resolved_user.get("persona")
    if isinstance(persona, str) and persona:
        card["sublabel"] = persona
    return card


def _build_groups(
    landing_cfg: dict[str, Any], resolved_users: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Transform config ``landing.groups`` into the template's ``groups`` shape:
    plain dicts with a ``label`` and an ``items`` key, since landing.html.j2
    uses bracket subscript (``group["items"]``) throughout.

    ``{type: "users"}`` auto-populates one card per configured user, using the
    relative ``/u/<user>/`` path that nginx.conf.j2 (bind-nginx-reverse-proxy)
    reverse-proxies to that user's loopback upstream — so, unlike ``landing_url``,
    no deploy-host needs baking into the landing cards themselves. When a user
    resolves to a persona (:func:`resolve_personas` returns a non-``None``
    ``persona``), that card also carries an optional ``sublabel`` holding the
    persona name, shown as a secondary badge on the card; users with no persona
    in effect (every bare-string roster) omit the key entirely, so
    landing.html.j2's ``{% if item["sublabel"] %}`` guard renders them without
    one. ``{type: "links", label, links}`` passes ``links`` straight through as
    ``items`` (link cards never carry a ``sublabel``). Unrecognized/malformed
    group entries are dropped rather than raising: lint is the authoritative
    gate on schema well-formedness, this is just the render-time adapter.

    Args:
        landing_cfg: The already-dict-coerced ``modules.web_terminals.landing``
            section (only ``groups`` is read).
        resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`
            output, in roster order — each entry's ``name`` becomes the card label
            and ``/u/<name>/`` url, and its ``persona`` (when not ``None``) the
            optional ``sublabel``.
    """
    groups_raw = landing_cfg.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        groups_raw = [{"type": "users"}]  # schema default when `landing.groups` is omitted

    groups: list[dict[str, Any]] = []
    for entry in groups_raw:
        entry = as_dict(entry)
        group_type = entry.get("type")
        if group_type == "users":
            items = [_user_card(user) for user in resolved_users]
            groups.append({"label": "Terminals", "items": items})
        elif group_type == "links":
            links = entry.get("links")
            items = [as_dict(link) for link in links] if isinstance(links, list) else []
            groups.append({"label": entry.get("label") or "", "items": items})
    return groups


def _auth_tls_context(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """Read the ``web_terminals.auth``/``web_terminals.tls`` stanzas into the context
    keys the nginx seam and the auth sidecar's compose service consume.

    This is the single place ``modules.web_terminals.auth`` is parsed: the nginx
    template, the compose overlay and the lint rules all read the keys returned
    here rather than the raw config, so there is one definition of what an
    ``auth`` stanza means. **This function renders nothing** — it derives values;
    the templates turn ``tls_enabled``/``auth_method`` into actual `listen 443
    ssl` / `auth_request` directives.

    Both stanzas remain entirely optional and default to the inert posture: no
    authentication, no TLS. Every value is read defensively (wrong-typed entries
    fall back to their default, as everywhere else in this module — lint is the
    authoritative gate on schema well-formedness) with one exception: an
    ``auth.method`` string naming a method that does not exist raises, because
    the resulting deployment would emit an auth seam nothing can answer.

    Args:
        web_terminals: The already-unwrapped ``modules.web_terminals`` dict (as
            passed to :func:`render_web_terminals`'s Jinja contexts).

    Returns:
        A dict with the auth keys ``auth_method`` (one of
        :data:`SUPPORTED_AUTH_METHODS`, defaults to ``"none"``), ``auth_port``
        (int, the sidecar's listen port), ``auth_session_lifetime`` (int
        seconds), ``auth_allow_insecure_http`` (bool), ``auth_image`` (str or
        ``None`` — required only in registry image-source mode),
        ``auth_oidc_issuer`` (str or ``None``),
        ``auth_oidc_client_id_env``/``auth_oidc_client_secret_env`` (the env-var
        *names* the sidecar reads its OIDC client credentials from — never the
        credentials themselves) and ``auth_oidc_claim`` (str or ``None``, the
        ID-token claim carrying the identity to map onto a roster user); plus
        the TLS keys ``tls_enabled`` (bool) and
        ``tls_cert``/``tls_key`` (str path or ``None``, read only when
        ``tls_enabled``).

    Raises:
        ValueError: If ``auth.method`` is a non-empty string outside
            :data:`SUPPORTED_AUTH_METHODS`.
    """
    auth = as_dict(web_terminals.get("auth"))
    tls = as_dict(web_terminals.get("tls"))
    oidc = as_dict(auth.get("oidc"))

    method = auth.get("method")
    auth_method = method if isinstance(method, str) and method else "none"
    if auth_method not in SUPPORTED_AUTH_METHODS:
        raise ValueError(
            f"modules.web_terminals.auth.method {auth_method!r} is not a supported "
            f"authentication method; expected one of {', '.join(SUPPORTED_AUTH_METHODS)}"
        )

    return {
        "auth_method": auth_method,
        "auth_port": _positive_int(auth.get("port"), _DEFAULT_AUTH_PORT),
        "auth_session_lifetime": _positive_int(
            auth.get("session_lifetime"), _DEFAULT_SESSION_LIFETIME
        ),
        "auth_allow_insecure_http": bool(auth.get("allow_insecure_http", False)),
        "auth_image": auth.get("image") or None,
        "auth_oidc_issuer": oidc.get("issuer") or None,
        "auth_oidc_client_id_env": _non_empty_str(
            oidc.get("client_id_env"), _DEFAULT_OIDC_CLIENT_ID_ENV
        ),
        "auth_oidc_client_secret_env": _non_empty_str(
            oidc.get("client_secret_env"), _DEFAULT_OIDC_CLIENT_SECRET_ENV
        ),
        # Which ID-token claim carries the identity that maps onto a roster
        # user. Left as None when unset (rather than defaulted here) so the
        # compose service emits no OSPREY_AUTH_OIDC_CLAIM at all and the
        # sidecar's own documented default applies — one default, in one place.
        "auth_oidc_claim": _non_empty_str(oidc.get("claim"), "") or None,
        "tls_enabled": bool(tls.get("enabled", False)),
        "tls_cert": tls.get("cert"),
        "tls_key": tls.get("key"),
        # The HOST directory holding the certificate and key. Optional, and the
        # only key in this stanza that names a path on the deploy host rather
        # than one inside the nginx container: setting it makes the overlay
        # bind-mount that directory read-only at the container-side directory
        # `tls.cert` sits in, which is the step that otherwise has to be done by
        # hand. Left unset, nothing is mounted and the operator supplies the
        # mount themselves — the escape hatch for a facility whose certificates
        # arrive by some route a directory bind cannot express.
        "tls_host_cert_dir": _non_empty_str(tls.get("host_cert_dir"), "") or None,
        # Where that directory lands inside the container: the parent of
        # `tls.cert`. Derived rather than configured so the mount and the
        # `ssl_certificate` directive cannot name different places.
        "tls_mount_target": _tls_mount_target(tls),
    }


def _check_tls_host_cert_dir(ctx: dict[str, Any]) -> None:
    """Refuse a ``tls.host_cert_dir`` that cannot deliver both files.

    One read-only bind mount covers the certificate *and* the key only when the
    two sit in the same directory inside the container, because the mount lands
    at the parent of ``tls.cert``. A key elsewhere would leave nginx reading a
    path nothing supplies — the exact silent failure this key exists to remove —
    so it is refused at render time rather than at nginx start.

    A relative ``host_cert_dir`` is refused for the same reason: compose
    resolves it against the project directory, so a deployment moved to another
    machine would mount a different (or missing) directory without saying so.

    Nothing is checked when the key is unset: that is the supported posture in
    which the operator supplies the mount themselves.
    """
    host_dir = ctx["tls_host_cert_dir"]
    if not host_dir:
        return
    if not ctx["tls_enabled"]:
        raise ValueError(
            "modules.web_terminals.tls.host_cert_dir is set but tls.enabled is false — "
            "the certificate directory would be mounted into an nginx that serves no "
            "HTTPS. Enable tls, or drop host_cert_dir"
        )
    if not posixpath.isabs(host_dir):
        raise ValueError(
            f"modules.web_terminals.tls.host_cert_dir {host_dir!r} must be an absolute "
            "path on the deploy host: compose resolves a relative bind source against "
            "the project directory, so the same config would mount a different "
            "directory elsewhere"
        )
    cert_dir = ctx["tls_mount_target"]
    key_dir = posixpath.dirname(_non_empty_str(ctx["tls_key"], "")) or None
    if cert_dir != key_dir:
        raise ValueError(
            "modules.web_terminals.tls.cert and tls.key must sit in the same directory "
            f"when host_cert_dir is set (cert is in {cert_dir!r}, key in {key_dir!r}): "
            "host_cert_dir is mounted at the certificate's directory, so a key outside "
            "it would not be present in the container"
        )


def _tls_mount_target(tls: dict[str, Any]) -> str | None:
    """The in-container directory ``tls.host_cert_dir`` is mounted at.

    The parent of ``tls.cert``, so a mounted directory always lands exactly
    where nginx's ``ssl_certificate`` looks. Returns ``None`` when there is
    nothing to mount or no certificate path to derive it from; the caller
    validates that ``tls.key`` shares the directory, which is what makes a
    single mount sufficient for both files.
    """
    cert = _non_empty_str(tls.get("cert"), "")
    if not cert:
        return None
    return posixpath.dirname(cert) or None


def _positive_int(value: Any, default: int) -> int:
    """A config value read as a positive int, falling back to ``default``.

    ``bool`` is excluded explicitly: it passes ``isinstance(..., int)``, and
    ``auth.port: true`` becoming port 1 would be a baffling deployment.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _non_empty_str(value: Any, default: str) -> str:
    """A config value read as a non-empty string, falling back to ``default``."""
    return value if isinstance(value, str) and value.strip() else default


def _check_roster_charset(services: list[dict[str, Any]], auth_method: str) -> None:
    """Fail-closed render gate: with authentication on, every roster name must
    match the username charset.

    The charset is enforced in two other places, and a deployment can miss both:
    lint is skippable (``--no-lint``), and ``auth_credentials`` only runs on the
    password-mode deploy path. Nothing on the render path itself checked, so a
    name carrying ``&`` or ``?`` rendered an nginx auth location whose verify
    query held *two* user parameters
    (``/_osprey_auth/bob&user=alice`` -> ``...verify?user=bob&user=alice``) —
    the exact ambiguity per-user internal locations exist to prevent. Since the
    username is the authorization identity, an unusable one must stop the render
    rather than produce a seam whose meaning depends on how the sidecar's query
    parser breaks a tie.

    Scoped to authentication being on, deliberately: with ``auth.method: none``
    the username is a routing label, not an identity, and raising here would
    break an otherwise valid render. Lint still reports it in that case.

    Args:
        services: The resolved per-user service entries (each with a ``user``).
        auth_method: The parsed ``auth.method`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If authentication is on and any roster name is outside the
            charset. The message names every offender.
    """
    if auth_method == "none":
        return
    # fullmatch, not match: `$` also matches *before* a trailing newline, so
    # `match` would accept "alice\n" — a name that then renders into an nginx
    # location key mid-directive.
    offenders = [
        service["user"]
        for service in services
        if not USERNAME_CHARSET_RE.fullmatch(service["user"])
    ]
    if offenders:
        raise ValueError(
            f"modules.web_terminals.users {', '.join(repr(name) for name in offenders)} "
            f"must match {USERNAME_CHARSET_RE.pattern!r} because "
            f"modules.web_terminals.auth.method is {auth_method!r}: an authenticated "
            "deployment makes the username the authorization identity, and it is "
            "rendered literally into an nginx location key and into the auth "
            "subrequest's ?user= value, where a name containing '&' or '?' would "
            "silently authorize a different user"
        )


def _check_roster_env_var_collisions(services: list[dict[str, Any]], auth_method: str) -> None:
    """Fail-closed render gate: with authentication on, no two roster names may
    share one per-user env-var suffix.

    :func:`~osprey.deployment.web_terminals.personas.env_var_suffix` is total and
    lossy — ``alice-b`` and ``alice_b`` both key ``..._ALICE_B`` — so a colliding
    pair would share a single ``OSPREY_AUTH_PW_HASH_ALICE_B``: one user's
    password would open the other's terminal, which is the precise isolation
    this feature exists to establish.

    Caught here rather than only downstream because the downstream signals are
    both poor. Credential provisioning raises on the password-mode deploy path
    only, and in ``oidc`` mode the collision surfaces (if at all) as a
    whole-deployment 503 from the sidecar — an outage whose message says nothing
    about which two roster names caused it. Render time knows both names.

    Args:
        services: The resolved per-user service entries (each with a ``user``).
        auth_method: The parsed ``auth.method`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If authentication is on and two or more names collide. The
            message names each colliding group and the suffix they share.
    """
    if auth_method == "none":
        return
    collisions = env_var_suffix_collisions([service["user"] for service in services])
    if collisions:
        detail = "; ".join(
            f"{', '.join(repr(name) for name in names)} all key {suffix!r}"
            for suffix, names in collisions.items()
        )
        raise ValueError(
            f"modules.web_terminals.users collide on their per-user env-var suffix "
            f"({detail}) and modules.web_terminals.auth.method is {auth_method!r}: "
            "colliding names would share one OSPREY_AUTH_PW_HASH_* entry, so one "
            "user's password would open the other's terminal. Rename one of them"
        )


def _check_mcp_topology(web_terminals: dict[str, Any]) -> None:
    """Fail closed on any ``modules.web_terminals.mcp.topology`` value other than
    the one wired topology, ``per_container_stdio``.

    Only two of the framework's eight MCP servers (``channel-finder`` and
    ``facility-knowledge``) are safely shareable across a shared HTTP tier
    without per-user-state corruption — not enough to justify building and
    securing a whole shared tier. ``shared_http`` is therefore a *recognized
    but rejected* schema value: it lints as an ERROR and raises here at render
    time. See ``references/modules/web-terminals.md`` for the rationale.

    This check is scoped to the shared **framework**-MCP tier only. It has
    nothing to do with, and never rejects, a facility's own
    ``claude_code.servers`` custom ``url``/HTTP entries — those are a
    separate, already-supported path (resolved by
    :func:`osprey.registry.mcp.resolve_servers` into each project's own
    ``.mcp.json``) that this module never reads or touches.

    Args:
        web_terminals: The already-unwrapped ``modules.web_terminals`` dict.

    Raises:
        ValueError: If ``mcp.topology`` is set to anything other than
            ``per_container_stdio`` (including ``shared_http`` and any other
            unrecognized value).
    """
    mcp_cfg = as_dict(web_terminals.get("mcp"))
    topology = mcp_cfg.get("topology") or SUPPORTED_MCP_TOPOLOGY
    if topology != SUPPORTED_MCP_TOPOLOGY:
        raise ValueError(
            f"modules.web_terminals.mcp.topology {topology!r} is not wired yet for "
            "the shared framework-MCP tier; per_container_stdio is the only "
            "supported topology (a facility's own claude_code.servers custom "
            "`url` entries are a separate, already-supported path and are "
            "unaffected by this restriction)."
        )
