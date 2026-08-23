"""ARIEL Web Interface - FastAPI Application.

A production-grade web interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), providing search, browsing, and entry creation
for scientific logbook data.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse

from osprey.interfaces._app_setup import configure_interface_app
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger("ariel")

STATIC_DIR = Path(__file__).parent / "static"

#: The panel starts on every one of these; they only describe how much of it
#: works. ``ok`` — the config parsed clean; ``configuration_warning`` — the
#: config parsed and the service can run, but a key is wrong;
#: ``configuration_invalid`` — the config did not parse, or the facility
#: vocabulary it asked for could not be loaded, so search is dead.
CONFIG_STATUS_OK = "ok"
CONFIG_STATUS_WARNING = "configuration_warning"
CONFIG_STATUS_INVALID = "configuration_invalid"

#: Remedies are derived from the error class, never written per call site, so
#: the banner, ``/api/capabilities`` and the UI name the same operator action.
REMEDY_NO_CONFIG_FILE = "mount config.yml at /app/config.yml or set CONFIG_FILE, then restart"
REMEDY_NAMED_KEY = "fix the named key in config.yml and restart"


def load_ariel_config_with_path(
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load ARIEL configuration from config.yml, reporting the file it came from.

    Looks for config in:
    1. Provided config_path argument
    2. /app/config.yml (Docker mount)
    3. CONFIG_FILE environment variable
    4. Current directory config.yml

    Environment variable overrides:
    - ARIEL_DATABASE_HOST: Override database host (for Docker networking)

    The path matters as much as the dictionary: a relative
    ``ariel.vocabulary.path`` resolves against the directory of the file that
    was actually read, and the settings editor writes back to it.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Tuple of the ARIEL configuration dictionary and the resolved path of
        the config.yml it was read from.

    Raises:
        RuntimeError: If no config file is found.
    """
    config_paths = [
        Path(config_path) if config_path else None,
        Path("/app/config.yml"),
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            logger.info(f"Loading config from {path}")
            with open(path) as f:
                # resolve_env_vars matches the framework's ConfigBuilder
                # behavior (load_osprey_config): a DSN written out in config.yml
                # may carry a ${ARIEL_DB_PASSWORD:-ariel} placeholder that must
                # expand here too, or the web interface would hand psycopg a
                # literal `${…}` password.
                from osprey.utils.config import resolve_env_vars

                config = resolve_env_vars(yaml.safe_load(f))
                ariel_config = config.get("ariel", {})
                services = config.get("services") or {}

            if ariel_config:
                # Resolve the DSN before the host override below rewrites it:
                # with `ariel.database.uri` unset the DSN is derived from
                # `services.postgresql`, and the derived host is `localhost` —
                # exactly what the container override has to replace.
                from osprey.services.ariel_search.config import resolve_ariel_dsn

                database = ariel_config.get("database") or {}
                database["uri"] = resolve_ariel_dsn(ariel_config, services.get("postgresql") or {})
                ariel_config["database"] = database

            # Apply environment variable overrides for Docker networking
            db_host_override = os.environ.get("ARIEL_DATABASE_HOST")
            if db_host_override and "database" in ariel_config:
                uri = ariel_config["database"].get("uri", "")
                if uri:
                    # Replace localhost or 127.0.0.1 with the override host
                    new_uri = re.sub(
                        r"@(localhost|127\.0\.0\.1):",
                        f"@{db_host_override}:",
                        uri,
                    )
                    if new_uri != uri:
                        logger.info(f"Overriding database host with {db_host_override}")
                        ariel_config["database"]["uri"] = new_uri

            return ariel_config, path

    raise RuntimeError(
        "No config.yml found. Set CONFIG_FILE environment variable "
        "or mount config.yml at /app/config.yml"
    )


def load_ariel_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load ARIEL configuration from config.yml.

    Thin delegation to :func:`load_ariel_config_with_path` for callers that do
    not need the file the configuration came from.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        ARIEL configuration dictionary.

    Raises:
        RuntimeError: If no config file is found.
    """
    return load_ariel_config_with_path(config_path)[0]


def _derive_config_status(
    config_errors: list[str],
    *,
    vocabulary_errors: list[str],
    parse_failed: bool,
    config_found: bool,
) -> tuple[str, str | None]:
    """Classify configuration errors and derive the matching operator remedy.

    Two classes, because they degrade different things: a config that did not
    parse at all, or a facility vocabulary that could not be loaded, kills the
    search path (``configuration_invalid``); a pre-existing ``validate()``
    error leaves a working service and only warns (``configuration_warning``).
    The remedy follows from the class, so no surface writes its own.

    Args:
        config_errors: Every error to report, in the order they were collected.
        vocabulary_errors: The vocabulary subset of ``config_errors``.
        parse_failed: Whether loading or parsing raised.
        config_found: Whether a config.yml was discovered at all.

    Returns:
        Tuple of the status string and the derived remedy (None when ok).
    """
    if not config_errors:
        return CONFIG_STATUS_OK, None
    if vocabulary_errors:
        from osprey.services.ariel_search.exceptions import VocabularyError

        return CONFIG_STATUS_INVALID, VocabularyError.remedy
    if not config_found:
        return CONFIG_STATUS_INVALID, REMEDY_NO_CONFIG_FILE
    if parse_failed:
        return CONFIG_STATUS_INVALID, REMEDY_NAMED_KEY
    return CONFIG_STATUS_WARNING, REMEDY_NAMED_KEY


@dataclass(frozen=True)
class _ConfigState:
    """Everything the panel knows about its configuration before it opens a socket.

    Attributes:
        config: The parsed ``ARIELConfig``, or None when loading or parsing
            raised. A None here is what stops the database attempt.
        path: The config.yml actually read, or None when none was found. Routes
            resolve config-relative paths against its directory and the settings
            editor writes back to it.
        errors: Every error to report, in the order they were collected.
        status: One of the ``CONFIG_STATUS_*`` values.
        remedy: The class-derived operator action, None when ok.
    """

    config: Any
    path: Path | None
    errors: list[str]
    status: str
    remedy: str | None


def _resolve_config_state(config_path: str | Path | None) -> _ConfigState:
    """Load, parse, validate and classify the configuration.

    Kept out of the lifespan and out of the database attempt on purpose: a
    broken config.yml must not be reported as an unreachable database, and it
    must not stop the panel from starting — the settings editor is how the
    operator repairs it. Nothing here raises; every failure becomes an entry in
    ``errors`` and a status a surface can render.

    Args:
        config_path: Explicit config file to read, or None for the search order
            :func:`load_ariel_config_with_path` implements.

    Returns:
        The classified state, ready to publish on ``app.state``.
    """
    ariel_config: Any = None
    resolved_path: Path | None = None
    parse_failed = False
    try:
        from osprey.services.ariel_search import ARIELConfig

        # load_ariel_config_with_path has already resolved the DSN (and applied
        # the container host override to it), so the section carries an explicit
        # `database.uri` by the time it is parsed here.
        config_dict, resolved_path = load_ariel_config_with_path(config_path)
        ariel_config = ARIELConfig.from_dict(config_dict, config_dir=resolved_path.parent)
        config_errors = list(ariel_config.validate())
    except Exception as e:
        parse_failed = True
        ariel_config = None
        config_errors = [str(e)]

    # A test-double config is not a list — only a real list counts as
    # vocabulary errors, so the classification cannot misfire on one.
    raw_vocabulary_errors = getattr(ariel_config, "vocabulary_errors", None)
    vocabulary_errors = raw_vocabulary_errors if isinstance(raw_vocabulary_errors, list) else []

    status, remedy = _derive_config_status(
        config_errors,
        vocabulary_errors=vocabulary_errors,
        parse_failed=parse_failed,
        config_found=resolved_path is not None,
    )
    return _ConfigState(
        config=ariel_config,
        path=resolved_path,
        errors=config_errors,
        status=status,
        remedy=remedy,
    )


def _config_banner(status: str, config_errors: list[str], remedy: str | None) -> str:
    """Render the configuration banner logged at startup.

    Deliberately distinct from the database banner: an operator reading the log
    must be able to tell a broken config.yml from an unreachable database.

    Args:
        status: One of the ``CONFIG_STATUS_*`` values.
        config_errors: The errors to list.
        remedy: The derived operator action.

    Returns:
        The multi-line banner text.
    """
    if status == CONFIG_STATUS_INVALID:
        headline = "CONFIGURATION INVALID"
        impact = (
            "  STILL WORKING: Web UI, drafts, config/settings endpoints\n  NOT WORKING:   Search\n"
        )
    else:
        headline = "CONFIGURATION WARNING"
        impact = "  The panel is running; the named key is not doing what it says.\n"
    listed = "\n".join(f"    {error}" for error in config_errors)
    return (
        "\n"
        "============================================================\n"
        f"  ARIEL: {headline} — {len(config_errors)} errors\n"
        "============================================================\n"
        f"{listed}\n"
        "\n"
        f"{impact}"
        f"\n  Remedy: {remedy}\n"
        "============================================================"
    )


def _create_lifespan(config_path: str | Path | None = None):
    """Create a lifespan context manager with the given config path.

    Args:
        config_path: Optional path to config file.

    Returns:
        Async context manager for FastAPI lifespan.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle.

        Initialize ARIEL service on startup, cleanup on shutdown.
        Gracefully degrades when the database is unavailable — the web UI,
        draft routes, config/settings endpoints still work.
        """
        logger.info("Starting ARIEL Web Interface...")

        # Initialize framework-only registry so ARIEL search modules can
        # resolve connectors and services via get_service() / get_connector().
        try:
            import osprey.registry.manager as _reg_mod

            _reg_mod._registry = _reg_mod.RegistryManager(registry_path=None)
            _reg_mod._registry.initialize(silent=True)
            logger.info("Framework registry initialized for ARIEL")
        except Exception as e:
            logger.warning(f"Registry initialization failed (non-fatal): {e}")

        # Resolved before any database work, and never raising, so a broken
        # config.yml cannot be reported as an unreachable database.
        state = _resolve_config_state(config_path)
        ariel_config = state.config

        # Set on BOTH paths, so every route can read them with getattr.
        app.state.config_path = state.path
        app.state.ariel_config = state.config
        app.state.config_errors = state.errors
        app.state.config_status = state.status
        app.state.config_remedy = state.remedy

        if state.errors:
            logger.warning(_config_banner(state.status, state.errors, state.remedy))

        # Try to connect to the database; degrade gracefully if unavailable.
        # This runs whenever a config object exists, INDEPENDENT of
        # config_errors: a vocabulary error degrades only the search path,
        # while browse, entries, status and publishing keep working.
        service = None
        if ariel_config is None:
            logger.info("Skipping database connection: no usable configuration")
        else:
            try:
                from osprey.services.ariel_search import create_ariel_service

                service = await create_ariel_service(ariel_config)

                healthy, message = await service.health_check()
                if healthy:
                    logger.info(f"ARIEL service ready: {message}")
                else:
                    logger.warning(f"ARIEL service degraded: {message}")
            except Exception as e:
                logger.warning(
                    "\n"
                    "============================================================\n"
                    "  ARIEL: DATABASE NOT AVAILABLE — DEGRADED MODE\n"
                    "============================================================\n"
                    "  The database connection failed. ARIEL is running without\n"
                    "  search, browse, or entry persistence.\n"
                    "\n"
                    "  STILL WORKING: Web UI, drafts, config/settings endpoints\n"
                    "  NOT WORKING:   Search, browse, entry creation, status\n"
                    "\n"
                    "  To restore full functionality, start PostgreSQL and\n"
                    "  restart the server.\n"
                    f"\n  Error: {e}\n"
                    "============================================================"
                )

        app.state.ariel_service = service

        yield

        # Cleanup
        logger.info("Shutting down ARIEL Web Interface...")
        if service is not None:
            await service.__aexit__(None, None, None)

    return lifespan


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the ARIEL Web Interface FastAPI application.

    App factory for ASGI servers and testing.

    Args:
        config_path: Optional path to config.yml file. If not provided,
            will search standard locations.

    Returns:
        Configured FastAPI application instance.
    """
    from osprey.interfaces.ariel.api.drafts import draft_router
    from osprey.interfaces.ariel.api.routes import router as api_router

    app = FastAPI(
        title="ARIEL Search Interface",
        description="Agentic Retrieval Interface for Electronic Logbooks",
        version="1.0.0",
        lifespan=_create_lifespan(config_path),
    )

    app.include_router(api_router)
    app.include_router(draft_router)

    @app.get("/")
    async def root():
        """Serve main index.html."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        """Simple health check endpoint."""
        config_status = getattr(app.state, "config_status", None)
        service = getattr(app.state, "ariel_service", None)
        if service is not None:
            healthy, message = await service.health_check()
            return {
                "status": "healthy" if healthy else "degraded",
                "message": message,
                "config_status": config_status,
            }
        return {
            "status": "degraded",
            "message": "Database unavailable — drafts, UI, and settings work",
            "config_status": config_status,
        }

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app


def run_web(
    host: str = "127.0.0.1",
    port: int = 8085,
    reload: bool = False,
    config_path: str | None = None,
) -> None:
    """Run the ARIEL web interface.

    CLI entry point for launching the web server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        reload: Enable auto-reload for development.
        config_path: Optional path to config file.
    """
    import uvicorn

    if reload:
        # Reload mode requires a string import path (uvicorn re-imports on change)
        uvicorn.run(
            "osprey.interfaces.ariel.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    else:
        app = create_app(config_path)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
