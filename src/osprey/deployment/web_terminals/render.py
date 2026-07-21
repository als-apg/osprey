"""Renders the multi-user web-terminal deployment artifacts from a facility config's
``modules.web_terminals`` stanza.

Three artifacts come out of one facility config: the docker-compose overlay (one
service per user + nginx), the nginx routing fragment, and the static landing page.
All port arithmetic is delegated to :func:`osprey.deployment.web_terminals.ports.allocate_ports`
— this module only builds the per-service context list and hands it to Jinja2.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Any

from jinja2 import Environment, FileSystemLoader

from osprey.deployment.web_terminals.personas import (
    SUPPORTED_MCP_TOPOLOGY,
    as_dict,
    effective_image_source,
    resolve_personas,
)
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)

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

# Per-container constant (Task 1.1): every per-user app's service families
# (web + every registry companion family) bind this host, never a routable
# interface —
# nginx's reverse proxy (Task 1.2) becomes the only off-host path. Not
# config-driven: unlike the per-family ports, there is no facility-config knob
# for this, since a facility that wants a per-user port reachable directly
# off-host would defeat the single-origin chokepoint this module exists to
# provide.
_LOOPBACK_BIND_HOST = "127.0.0.1"


def render_web_terminals(config: Any) -> dict[str, str]:
    """Render the compose overlay, nginx fragment, and landing page for one facility config.

    Args:
        config: The parsed facility config, read defensively as nested dicts (same
            convention as :func:`osprey.deployment.web_terminals.lint.lint_web_terminals`
            — no assumption that ``config`` is a particular schema/dataclass type).
            Deterministic: the same config always renders the same three artifacts,
            with no clock/random inputs.

    Returns:
        Mapping of output-relative-path to rendered content, for exactly three
        artifacts: ``docker-compose.web.yml``, ``nginx/nginx.conf``, and
        ``nginx/landing.html``.

    Raises:
        ValueError: If ``modules.web_terminals.nginx_port`` is missing/not an int,
            if a configured user can't resolve a full port-family set, if
            ``deploy.fqdn`` is missing while at least one user is configured (the
            landing-origin host baked into ``OSPREY_TERMINAL_LANDING_URL``), if
            a roster entry's persona reference can't be resolved (see
            :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
            ``strict`` contract — render always resolves strictly), or if
            ``modules.web_terminals.mcp.topology`` is set to anything other than
            ``per_container_stdio`` (see :func:`_check_mcp_topology`).
    """
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
                "extra_mounts": entry["extra_mounts"],
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
            }
        )

    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        raise ValueError("modules.web_terminals.nginx_port is required and must be an int")

    landing_url = _landing_url(root, nginx_port) if services else ""

    image_source = effective_image_source(web_terminals)

    compose_ctx = {
        "facility_prefix": facility_prefix,
        "registry_url": registry.get("url") or "",
        "image_source": image_source,
        "services": services,
        "nginx_port": nginx_port,
        "landing_url": landing_url,
        "facility_timezone": facility.get("timezone") or "UTC",
        "bind_host": _LOOPBACK_BIND_HOST,
    }
    auth_tls_ctx = _auth_tls_context(web_terminals)
    if auth_tls_ctx["tls_enabled"] and not (auth_tls_ctx["tls_cert"] and auth_tls_ctx["tls_key"]):
        raise ValueError(
            "modules.web_terminals.tls.enabled is true but tls.cert/tls.key are not both "
            "set — the gated `listen 443 ssl` seam (Task 1.3) needs both paths to emit a "
            "coherent ssl_certificate/ssl_certificate_key pair"
        )

    nginx_ctx = {
        "nginx_port": nginx_port,
        "services": services,
        "bind_host": _LOOPBACK_BIND_HOST,
        **auth_tls_ctx,
    }
    landing_ctx = {
        "facility_name": facility.get("name") or "",
        "groups": _build_groups(as_dict(web_terminals.get("landing")), resolved_users),
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


def _landing_url(root: dict[str, Any], nginx_port: int) -> str:
    """Build the absolute origin baked into every service's ``OSPREY_TERMINAL_LANDING_URL``.

    Per-user containers only get this value once, at container start (env vars, not
    request time) — unlike nginx.conf.j2's per-request ``$host`` redirect target,
    resolving it can't be deferred to the browser. It comes from ``deploy.fqdn``:
    the schema documents that field as reachable from developers' laptops (used in
    client-mode profiles), whereas ``deploy.host`` is only guaranteed
    SSH-resolvable (may be a bare `~/.ssh/config` alias, not a browser-reachable
    hostname).
    """
    deploy = as_dict(root.get("deploy"))
    host = str(deploy.get("fqdn") or "").strip()
    if not host:
        raise ValueError(
            "deploy.fqdn is required to render modules.web_terminals landing_url "
            "(OSPREY_TERMINAL_LANDING_URL) when at least one user is configured"
        )
    return f"http://{host}:{nginx_port}"


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
    """Transform config ``landing.groups`` (Task 1.2 shape) into template ``groups``
    (Task 1.6 shape): plain dicts with a ``label`` and an ``items`` key, since
    landing.html.j2 uses bracket subscript (``group["items"]``) throughout.

    ``{type: "users"}`` auto-populates one card per configured user, using the
    relative ``/u/<user>/`` path that nginx.conf.j2 (bind-nginx-reverse-proxy)
    reverse-proxies to that user's loopback upstream — so, unlike ``landing_url``,
    no deploy-host needs baking into the landing cards themselves. When a user
    resolves to a persona (:func:`resolve_personas` returns a non-``None``
    ``persona``), that card also carries an optional ``sublabel`` holding the
    persona name, shown as a secondary badge on the card; users with no persona
    in effect (every pre-persona bare-string roster) omit the key entirely, so
    landing.html.j2's ``{% if item["sublabel"] %}`` guard renders them exactly as
    before. ``{type: "links", label, links}`` passes ``links`` straight through as
    ``items`` (link cards never carry a ``sublabel``). Unrecognized/malformed
    group entries are dropped rather than raising: the lint (Task 1.5) is the
    authoritative gate on schema well-formedness, this is just the render-time
    adapter.

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
    """Read the optional, forward-looking ``web_terminals.auth``/``web_terminals.tls``
    stanzas (Task 1.4's config contract) into the context keys the gated nginx seam
    render (Task 1.3) consumes.

    Both stanzas are entirely optional and default to the inert v1 posture: no
    authentication, no TLS — identical trust model to Phase 1, only the access
    boundary moves from host-port to URL-path. **This function does not render any
    nginx seam block itself** — it only derives the defensively-read values; Task
    1.3 is responsible for turning ``tls_enabled``/``auth_method`` into actual
    `listen 443 ssl` / `auth_request` directives.

    Args:
        web_terminals: The already-unwrapped ``modules.web_terminals`` dict (as
            passed to :func:`render_web_terminals`'s Jinja contexts).

    Returns:
        A dict with exactly four keys: ``auth_method`` (str, defaults to
        ``"none"``), ``tls_enabled`` (bool, defaults to ``False``), and
        ``tls_cert``/``tls_key`` (str path or ``None``, present only to be read
        when ``tls_enabled`` is true).
    """
    auth = as_dict(web_terminals.get("auth"))
    tls = as_dict(web_terminals.get("tls"))
    auth_method = auth.get("method")
    return {
        "auth_method": auth_method if isinstance(auth_method, str) and auth_method else "none",
        "tls_enabled": bool(tls.get("enabled", False)),
        "tls_cert": tls.get("cert"),
        "tls_key": tls.get("key"),
    }


def _check_mcp_topology(web_terminals: dict[str, Any]) -> None:
    """Fail closed on any ``modules.web_terminals.mcp.topology`` value other than
    the one wired topology, ``per_container_stdio`` (Task 2.5).

    Only two of the framework's eight MCP servers (``channel-finder`` and
    ``facility-knowledge``) were found to be safely shareable across a shared
    HTTP tier without per-user-state corruption — not enough to justify
    building and securing a whole shared tier this phase. ``shared_http`` is
    therefore a *recognized but rejected* schema value: it lints as an ERROR
    (Task 2.4) and raises here at render time. See
    ``references/modules/web-terminals.md`` for the full deferral rationale.

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
