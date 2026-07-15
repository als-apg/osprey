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

from osprey.deployment.web_terminals.ports import allocate_ports, base_ports_from_config

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

# Per-container constant (Task 1.1): every per-user app's four service families
# (web/artifact/ariel/lattice) bind this host, never a routable interface —
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
            if a configured user can't resolve a full four-family port set, or if
            ``deploy.fqdn`` is missing while at least one user is configured (the
            landing-origin host baked into ``OSPREY_TERMINAL_LANDING_URL``).
    """
    root = _as_dict(config)
    facility = _as_dict(root.get("facility"))
    registry = _as_dict(root.get("registry"))
    web_terminals = _as_dict(_as_dict(root.get("modules")).get("web_terminals"))

    users_raw = web_terminals.get("users")
    users = (
        [user for user in users_raw if isinstance(user, str)] if isinstance(users_raw, list) else []
    )

    base_ports = base_ports_from_config(web_terminals)
    services = [
        {"user": user, **allocate_ports(base_ports, index)} for index, user in enumerate(users)
    ]

    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        raise ValueError("modules.web_terminals.nginx_port is required and must be an int")

    landing_url = _landing_url(root, nginx_port) if services else ""

    compose_ctx = {
        "facility_prefix": facility.get("prefix") or "",
        "registry_url": registry.get("url") or "",
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
        "groups": _build_groups(_as_dict(web_terminals.get("landing")), users),
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
    deploy = _as_dict(root.get("deploy"))
    host = str(deploy.get("fqdn") or "").strip()
    if not host:
        raise ValueError(
            "deploy.fqdn is required to render modules.web_terminals landing_url "
            "(OSPREY_TERMINAL_LANDING_URL) when at least one user is configured"
        )
    return f"http://{host}:{nginx_port}"


def _build_groups(landing_cfg: dict[str, Any], users: list[str]) -> list[dict[str, Any]]:
    """Transform config ``landing.groups`` (Task 1.2 shape) into template ``groups``
    (Task 1.6 shape): plain dicts with a ``label`` and an ``items`` key, since
    landing.html.j2 uses bracket subscript (``group["items"]``) throughout.

    ``{type: "users"}`` auto-populates one card per configured user, using the
    relative ``/u/<user>/`` path that nginx.conf.j2 (bind-nginx-reverse-proxy)
    reverse-proxies to that user's loopback upstream — so, unlike ``landing_url``,
    no deploy-host needs baking into the landing cards themselves. ``{type:
    "links", label, links}`` passes ``links`` straight through as ``items``.
    Unrecognized/malformed group entries are dropped rather than raising: the
    lint (Task 1.5) is the authoritative gate on schema well-formedness, this is
    just the render-time adapter.
    """
    groups_raw = landing_cfg.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        groups_raw = [{"type": "users"}]  # schema default when `landing.groups` is omitted

    groups: list[dict[str, Any]] = []
    for entry in groups_raw:
        entry = _as_dict(entry)
        group_type = entry.get("type")
        if group_type == "users":
            items = [{"label": user, "url": f"/u/{user}/"} for user in users]
            groups.append({"label": "Terminals", "items": items})
        elif group_type == "links":
            links = entry.get("links")
            items = [_as_dict(link) for link in links] if isinstance(links, list) else []
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
    auth = _as_dict(web_terminals.get("auth"))
    tls = _as_dict(web_terminals.get("tls"))
    auth_method = auth.get("method")
    return {
        "auth_method": auth_method if isinstance(auth_method, str) and auth_method else "none",
        "tls_enabled": bool(tls.get("enabled", False)),
        "tls_cert": tls.get("cert"),
        "tls_key": tls.get("key"),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}
