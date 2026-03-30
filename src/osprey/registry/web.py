"""Data-driven catalog of OSPREY companion web servers.

Defines metadata for each web server that ``ServerLauncher`` can start.
The infrastructure layer uses ``importlib`` to resolve factory paths at
call time — no direct imports from interfaces/ or services/.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WebServerDefinition:
    """Metadata for one companion web server.

    Attributes:
        name: Human-readable name for logging.
        factory_path: Dotted import path with colon separator, e.g.
            ``"osprey.interfaces.artifacts.app:create_app"``.
        config_key: Top-level ``config.yml`` key, e.g. ``"artifact_server"``.
        config_web_subkey: Optional nested subkey for host/port/auto_launch,
            e.g. ``"web"`` when config is ``ariel.web.host``.
        host_default: Fallback host when not in config.
        port_default: Fallback port when not in config.
        pass_workspace: If True, ``workspace_root`` is passed to the factory.
        auto_launch_default: Default ``auto_launch`` value when key is absent.
        require_section: If True, missing/empty top-level section → auto_launch=False.
        factory_config_kwargs: Maps factory kwarg names to dotted config paths.
            E.g. ``{"tuning_api_url": "tuning.api_url"}`` reads
            ``config["tuning"]["api_url"]`` and passes it as ``tuning_api_url=``.
        import_error_message: Custom message when the factory import fails.
            If None, ImportError propagates normally.
    """

    name: str
    factory_path: str
    config_key: str
    config_web_subkey: str | None = None
    host_default: str = "127.0.0.1"
    port_default: int = 8080
    pass_workspace: bool = False
    auto_launch_default: bool = True
    require_section: bool = False
    factory_config_kwargs: dict[str, str] = field(default_factory=dict)
    import_error_message: str | None = None


FRAMEWORK_WEB_SERVERS: dict[str, WebServerDefinition] = {
    "artifact": WebServerDefinition(
        name="Artifact gallery",
        factory_path="osprey.interfaces.artifacts.app:create_app",
        config_key="artifact_server",
        port_default=8086,
        pass_workspace=True,
    ),
    "ariel": WebServerDefinition(
        name="ARIEL server",
        factory_path="osprey.interfaces.ariel.app:create_app",
        config_key="ariel",
        config_web_subkey="web",
        port_default=8085,
    ),
    "tuning": WebServerDefinition(
        name="Tuning panel",
        factory_path="osprey.interfaces.tuning.app:create_app",
        config_key="tuning",
        config_web_subkey="web",
        port_default=8090,
        factory_config_kwargs={"tuning_api_url": "tuning.api_url"},
    ),
    "channel_finder": WebServerDefinition(
        name="Channel Finder",
        factory_path="osprey.interfaces.channel_finder.app:create_app",
        config_key="channel_finder",
        config_web_subkey="web",
        port_default=8092,
        require_section=True,
    ),
    "lattice_dashboard": WebServerDefinition(
        name="Lattice dashboard",
        factory_path="osprey.interfaces.lattice_dashboard.app:create_app",
        config_key="lattice_dashboard",
        port_default=8097,
        pass_workspace=True,
        require_section=True,
    ),
}
