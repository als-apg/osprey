"""Data-driven catalog of OSPREY companion web servers.

Defines metadata for each web server that ``ServerLauncher`` can start.
The infrastructure layer uses ``importlib`` to resolve factory paths at
call time — no direct imports from interfaces/ or services/.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebServerDefinition:
    """Metadata for one companion web server.

    Attributes:
        name: Human-readable name for logging.
        factory_path: Dotted import path with colon separator, e.g.
            ``"osprey.interfaces.artifacts.app:create_app"``.
        config_key: Top-level ``config.yml`` key, e.g. ``"artifact_server"``.
        panel_id: The id this server's panel is keyed on everywhere OUTSIDE the
            registry — ``web.panels.<panel_id>`` in config, the frontend tab,
            the ``/panel/<panel_id>/`` proxy mount, and
            ``profiles.web_panels.BUILTIN_PANELS``. It is a second namespace,
            not a spelling variant of the registry key: the gallery is registry
            key ``artifact`` and panel ``artifacts``, and the hyphenated ids
            (``channel-finder``, ``system-health``) are not valid registry keys.
            Declared here so the two namespaces are related in exactly one
            place — every consumer that needs to cross between them reads
            :data:`PANEL_ID_TO_REGISTRY_KEY` or this field instead of carrying
            its own table.
        config_web_subkey: Optional nested subkey for host/port/auto_launch,
            e.g. ``"web"`` when config is ``ariel.web.host``. Navigating it is
            :func:`web_server_config_section`'s job, never a caller's — see that
            function for why writing the key at the other depth is an error.
        host_default: Fallback host when not in config.
        port_default: Fallback port when not in config.
        pass_workspace: If True, ``workspace_root`` is passed to the factory.
        auto_launch_default: Default ``auto_launch`` value when key is absent.
        require_section: If True, missing/empty top-level section → auto_launch=False.
        factory_config_kwargs: Maps factory kwarg names to dotted config paths.
            E.g. ``{"bundle_path": "facility_knowledge.bundle_path"}`` reads
            ``config["facility_knowledge"]["bundle_path"]`` and passes it as
            ``bundle_path=``.
        import_error_message: Custom message when the factory import fails.
            If None, ImportError propagates normally.
        port_family: Multi-user port-family name for this server (drives the
            ``modules.web_terminals.<family>_base_port`` config field). ``None``
            means the family name is the server's registry key. Only set when a
            server's family is conventionally named something else
            (``lattice_dashboard`` → ``lattice``).
        multi_user_base_port: Default first per-user port for this server's
            family in multi-user deployments (user *i* gets ``base + i``; see
            ``deployment/web_terminals/ports.py``). Every entry MUST set it —
            per-user containers share the host network namespace, so a server
            without its own family collides with itself across users. Config
            overrides it via ``<family>_base_port``. Convention: ×100 spacing
            in the 9091+ range.
    """

    name: str
    factory_path: str
    config_key: str
    panel_id: str
    config_web_subkey: str | None = None
    host_default: str = "127.0.0.1"
    port_default: int = 8080
    pass_workspace: bool = False
    auto_launch_default: bool = True
    require_section: bool = False
    factory_config_kwargs: dict[str, str] = field(default_factory=dict)
    import_error_message: str | None = None
    port_family: str | None = None
    multi_user_base_port: int | None = None

    @property
    def port_env_var(self) -> str:
        """The env var that overrides this server's listen port.

        Single derivation shared by :func:`resolve_web_server_address` — which
        every launcher entry is partial-bound to, and which every caller that
        builds a panel URL goes through — and the multi-user compose render
        (``deployment/web_terminals``), the two ends of the same contract, so
        they can never drift.
        """
        return f"OSPREY_{self.config_key.upper()}_PORT"


FRAMEWORK_WEB_SERVERS: dict[str, WebServerDefinition] = {
    "artifact": WebServerDefinition(
        name="Artifact gallery",
        factory_path="osprey.interfaces.artifacts.app:create_app",
        config_key="artifact_server",
        panel_id="artifacts",
        port_default=8086,
        pass_workspace=True,
        multi_user_base_port=9291,
    ),
    "ariel": WebServerDefinition(
        name="ARIEL server",
        factory_path="osprey.interfaces.ariel.app:create_app",
        config_key="ariel",
        panel_id="ariel",
        config_web_subkey="web",
        port_default=8085,
        multi_user_base_port=9391,
    ),
    "channel_finder": WebServerDefinition(
        name="Channel Finder",
        factory_path="osprey.interfaces.channel_finder.app:create_app",
        config_key="channel_finder",
        panel_id="channel-finder",
        config_web_subkey="web",
        port_default=8092,
        require_section=True,
        multi_user_base_port=9591,
    ),
    "lattice_dashboard": WebServerDefinition(
        name="Lattice dashboard",
        factory_path="osprey.interfaces.lattice_dashboard.app:create_app",
        config_key="lattice_dashboard",
        panel_id="lattice",
        port_default=8097,
        pass_workspace=True,
        require_section=True,
        port_family="lattice",
        multi_user_base_port=9491,
    ),
    # OKF "KNOWLEDGE" panel. config_key is the shared facility_knowledge section
    # (also read by the MCP server + CLI); require_section gates auto-launch on
    # that section existing, and factory_config_kwargs feeds the resolved
    # bundle_path into create_app (None → the panel's guarded mode). Port lives
    # directly under the section (no config_web_subkey); env override is
    # OSPREY_FACILITY_KNOWLEDGE_PORT.
    "okf": WebServerDefinition(
        name="OKF Knowledge Panel",
        factory_path="osprey.interfaces.okf_panel.app:create_app",
        config_key="facility_knowledge",
        panel_id="okf",
        port_default=8093,
        require_section=True,
        factory_config_kwargs={"bundle_path": "facility_knowledge.bundle_path"},
        multi_user_base_port=9691,
    ),
    # System-health dashboard panel. Port/host live under the nested `web`
    # subkey (config is `health.web.host`/`health.web.port`); env override is
    # OSPREY_HEALTH_PORT. require_section is False — the panel is always
    # launchable (auto_launch defaults on) since the health framework ships a
    # usable default even when config.yml has no `health` section.
    "system_health": WebServerDefinition(
        name="System Health Dashboard",
        factory_path="osprey.interfaces.health.app:create_app",
        config_key="health",
        panel_id="system-health",
        config_web_subkey="web",
        port_default=8094,
        require_section=False,
        multi_user_base_port=9791,
    ),
}


#: Panel id (as it appears under ``web.panels``, in the frontend tab set and in
#: the ``/panel/<id>/`` proxy mount) → key into :data:`FRAMEWORK_WEB_SERVERS`.
#:
#: Derived from :attr:`WebServerDefinition.panel_id`, never hand-listed. The two
#: namespaces genuinely differ (``artifacts``/``artifact``,
#: ``channel-finder``/``channel_finder``, ``lattice``/``lattice_dashboard``,
#: ``system-health``/``system_health``), and every consumer that used to keep its
#: own copy of that table — the health panel probe, the ``osprey web``
#: port pre-flight, the web terminal's panel proxy — drifted from the others.
#: Read this instead of writing a fifth one.
PANEL_ID_TO_REGISTRY_KEY: dict[str, str] = {
    definition.panel_id: key for key, definition in FRAMEWORK_WEB_SERVERS.items()
}


def panel_url_state_attr(key: str) -> str:
    """The ``app.state`` attribute a companion server's panel URL is published under.

    The convention is the registry key plus ``_server_url``. Two modules depend
    on it: the web terminal's lifespan writes it (``web_terminal/app.py``) and
    the panel reverse proxy reads it (``web_terminal/routes/proxy.py``). Nothing
    else would notice if one side changed its spelling — the panel would simply
    report unavailable, which is indistinguishable from a server that is legitimately
    switched off. It lives here, beside the other two cross-namespace facts
    (:attr:`WebServerDefinition.panel_id` and
    :attr:`WebServerDefinition.port_env_var`), because the two modules that share
    it cannot import each other: ``app`` already imports the route package.

    Args:
        key: Key into :data:`FRAMEWORK_WEB_SERVERS`, e.g. ``"artifact"``.

    Returns:
        The ``app.state`` attribute name, e.g. ``"artifact_server_url"``.
    """
    return f"{key}_server_url"


#: The keys whose meaning depends entirely on their nesting depth.
_ADDRESS_KEYS: tuple[str, ...] = ("host", "port", "auto_launch")


class WebServerConfigDepthError(ValueError):
    """A companion server's address key is written at the wrong nesting depth.

    Raised instead of silently ignoring the key. ``ariel.auto_launch: false``
    (correct depth: ``ariel.web.auto_launch``) used to read back as the default
    ``True`` and launch the panel the operator had just switched off; the
    mirror-image mistake, ``artifact_server.web.port``, left the gallery on
    8086 while the operator believed they had moved it.

    Subclasses ``ValueError`` so the launch surfaces that already fail open on a
    bad config section (``osprey chat``'s per-server guard, the web terminal's
    per-panel guard) keep failing open — but loudly, and with the panel *not*
    started, which is the outcome ``auto_launch: false`` asked for anyway.
    """


def web_server_config_section(
    definition: WebServerDefinition,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the mapping holding *definition*'s host/port/auto_launch.

    The single navigation of :attr:`WebServerDefinition.config_web_subkey`. The
    six companion servers disagree about depth — three take ``<section>.web.*``
    (ariel, channel_finder, system_health), three take ``<section>.*`` (artifact,
    lattice_dashboard, okf) — because three of the sections are shared with
    non-web config and three are the panel's own. That disagreement is a shipped,
    documented config format, so this function reconciles it in one place and
    rejects the wrong depth rather than letting each caller re-derive it.

    Args:
        definition: The server whose config section to locate.
        config: Already-loaded ``config.yml`` mapping.

    Returns:
        The mapping to read ``host``/``port``/``auto_launch`` from; empty when
        the server has no config section at all.

    Raises:
        WebServerConfigDepthError: An address key sits at the other depth, where
            nothing reads it.
    """
    top = config.get(definition.config_key)
    top = top if isinstance(top, Mapping) else {}

    if definition.config_web_subkey:
        _reject_stray_keys(definition, top, wrong_depth="")
        nested = top.get(definition.config_web_subkey)
        return nested if isinstance(nested, Mapping) else {}

    nested = top.get("web")
    if isinstance(nested, Mapping):
        _reject_stray_keys(definition, nested, wrong_depth="web")
    return top


def _reject_stray_keys(
    definition: WebServerDefinition,
    section: Mapping[str, Any],
    *,
    wrong_depth: str,
) -> None:
    """Raise if *section* — a level nothing reads — holds any address key.

    Args:
        definition: The server being configured.
        section: The mapping at the depth that is NOT read for this server.
        wrong_depth: The subkey *section* sits under, ``""`` for the top level of
            the server's config section. Used to spell both paths in the message.

    Raises:
        WebServerConfigDepthError: *section* holds at least one address key.
    """
    stray = [key for key in _ADDRESS_KEYS if key in section]
    if not stray:
        return

    def _path(depth: str, key: str) -> str:
        return ".".join(part for part in (definition.config_key, depth, key) if part)

    right_depth = definition.config_web_subkey or ""
    listed = ", ".join(_path(wrong_depth, key) for key in stray)
    raise WebServerConfigDepthError(
        f"{definition.name}: {listed} "
        f"{'is' if len(stray) == 1 else 'are'} at the wrong nesting depth and "
        f"nothing reads {'it' if len(stray) == 1 else 'them'}. "
        f"Write {' and '.join(_path(right_depth, key) for key in stray)} instead. "
        "OSPREY refuses the config rather than starting a panel you switched off "
        "or binding a port you did not ask for."
    )


def resolve_web_server_address(
    key: str,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, int]:
    """Resolve the ``(host, port)`` a companion web server listens on.

    The single derivation every producer and consumer of a companion server's
    address shares: the server's config section (honouring
    ``config_web_subkey``), falling back to the definition's defaults, with
    ``WebServerDefinition.port_env_var`` overriding the port.

    The env override is not optional detail. Multi-user compose renders export
    ``OSPREY_<CONFIG_KEY>_PORT`` per user because the per-user containers share
    the host network namespace, so a consumer that reads the config port alone
    builds a URL pointing at a port nothing listens on.

    A ``port_env_var`` that is set but not an integer is logged and ignored
    rather than raised: the callers are URL builders on hot paths (MCP tool
    responses, the approval hook), and failing one deployment's typo closed at
    the config port beats failing the tool call.

    Args:
        key: Key into :data:`FRAMEWORK_WEB_SERVERS`, e.g. ``"artifact"``.
        config: Already-loaded ``config.yml`` mapping. Loaded on demand when
            omitted — pass it when the caller has one in hand.

    Returns:
        The resolved ``(host, port)`` pair.

    Raises:
        KeyError: *key* is not a known companion web server.
        WebServerConfigDepthError: The config writes ``host``/``port``/
            ``auto_launch`` at the depth this server does not read.
    """
    import os

    definition = FRAMEWORK_WEB_SERVERS[key]

    if config is None:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()

    section = web_server_config_section(definition, config)

    host = section.get("host") or definition.host_default
    port = int(section.get("port") or definition.port_default)

    env_value = os.environ.get(definition.port_env_var)
    if env_value:
        try:
            port = int(env_value)
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: not an integer port; using %d",
                definition.port_env_var,
                env_value,
                port,
            )

    return host, port


def resolve_web_server_base_url(
    key: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Return the ``http://host:port`` base URL for a companion web server.

    Thin wrapper over :func:`resolve_web_server_address` for the callers that
    only ever build a URL. See that function for the resolution order.
    """
    host, port = resolve_web_server_address(key, config)
    return f"http://{host}:{port}"
