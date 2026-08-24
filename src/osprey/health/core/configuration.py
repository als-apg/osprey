"""Core ``configuration`` health category.

Reports on the project's ``config.yml``: that it exists, parses as a YAML
mapping, has a valid ``health:`` section, and that its structure, referenced
environment variables and timezone are sound.

Unlike the other core categories, this one *reports on config loading itself*,
so it consumes a :class:`ConfigState` describing the outcome of the CLI's
single ``config.yml`` load rather than a bare config mapping. The category
never touches disk — the CLI performs the one load and constructs the state.

Row names, statuses and messages are pinned by the CLI contract tests. The
``health_config`` error row is emitted when the ``health:`` section is invalid.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from osprey.health.models import CheckResult, Status

_CATEGORY = "configuration"

# In the MCP architecture, only python_code_generator is actively consumed by
# live code (basic_generator.py). Other model roles are optional and
# application-specific.
_RECOMMENDED_MODELS = ["python_code_generator"]

# services.postgresql.port_host is the host-side published port of *this*
# project's Postgres, so it only bears on a DSN aimed back at the host.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass
class ConfigState:
    """Outcome of the CLI's single ``config.yml`` load, consumed by this category.

    The CLI performs the one load and constructs this; the ``configuration``
    category translates it into result rows without touching disk.

    Attributes:
        config_path: Path the loader looked at (``<project>/config.yml``).
        exists: Whether ``config_path`` exists.
        cwd: Directory searched, used in the not-found diagnostic.
        config: Parsed config mapping, or ``None`` when it could not be parsed
            into a mapping (missing file, empty file, non-mapping, parse error).
        yaml_error: Human-readable reason the YAML did not yield a mapping;
            ``None`` when parsing succeeded. Drives the ``yaml_valid`` row.
        health_error: Human-readable reason the ``health:`` section is invalid;
            ``None`` when it is valid or absent. Drives the ``health_config`` row.
    """

    config_path: Path
    exists: bool
    cwd: Path
    config: dict[str, Any] | None = None
    yaml_error: str | None = None
    health_error: str | None = None


def configuration(config: ConfigState, context: Any = None) -> Any:
    """Build the ``configuration`` category callable.

    Args:
        config: Pre-loaded :class:`ConfigState` from the CLI's ``config.yml`` load.
        context: Unused; configuration reports on config and needs no connector.

    Returns:
        A no-argument callable producing the category's :class:`CheckResult` rows.
    """
    state = config

    def _run() -> list[CheckResult]:
        return _check_configuration(state)

    return _run


def _check_configuration(state: ConfigState) -> list[CheckResult]:
    """Produce all ``configuration`` category rows from a loaded config state."""
    results: list[CheckResult] = []

    # config.yml existence.
    if not state.exists:
        results.append(
            CheckResult(
                name="config_file_exists",
                category=_CATEGORY,
                status=Status.ERROR,
                # The path goes in the message, not the details: details are only
                # printed under --verbose, and "not found" without saying where
                # is unactionable on its own. The command resolves the repository
                # enclosing the working directory, so naming the cwd would point
                # at somewhere it never looked.
                message=f"config.yml not found at {state.config_path}",
                details=(
                    f"Searched from: {state.cwd}\n"
                    "Run this from inside a deployment repository, or pass "
                    "--project with the directory holding config.yml."
                ),
            )
        )
        return results

    results.append(
        CheckResult(
            name="config_file_exists",
            category=_CATEGORY,
            status=Status.OK,
            message=f"Found at {state.config_path}",
        )
    )

    # YAML validity: any failure to yield a mapping is reported here, then stop.
    if state.yaml_error is not None or state.config is None:
        message = state.yaml_error or "Config file is empty"
        results.append(
            CheckResult(
                name="yaml_valid",
                category=_CATEGORY,
                status=Status.ERROR,
                message=message,
            )
        )
        return results

    results.append(
        CheckResult(
            name="yaml_valid",
            category=_CATEGORY,
            status=Status.OK,
            message="Valid YAML syntax",
        )
    )

    # health: section validity (new row; only emitted when invalid). Does not
    # short-circuit — the rest of the report still renders.
    if state.health_error is not None:
        results.append(
            CheckResult(
                name="health_config",
                category=_CATEGORY,
                status=Status.ERROR,
                message=f"Invalid health: section: {state.health_error}",
            )
        )

    config = state.config
    results.extend(_check_config_structure(config))
    results.extend(_check_environment_variables(config))
    results.append(_check_timezone(config))
    results.extend(_check_ariel_dsn_port(config))
    return results


def _check_config_structure(config: dict[str, Any]) -> list[CheckResult]:
    """Check configuration structure and required sections."""
    results: list[CheckResult] = []

    models = config.get("models", {})

    if not models:
        results.append(
            CheckResult(
                name="model_configs",
                category=_CATEGORY,
                status=Status.WARNING,
                message="No models section in config",
                details=(
                    "The models section is optional but python_code_generator is needed "
                    "for Python code execution."
                ),
            )
        )
    else:
        missing_recommended = [m for m in _RECOMMENDED_MODELS if m not in models]
        if missing_recommended:
            missing_str = ", ".join(missing_recommended)
            results.append(
                CheckResult(
                    name="recommended_models",
                    category=_CATEGORY,
                    status=Status.WARNING,
                    message=f"Missing recommended models: {missing_str}",
                    details="python_code_generator is used by the Python execution service.",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="recommended_models",
                    category=_CATEGORY,
                    status=Status.OK,
                    message=f"{len(models)} model(s) defined (including python_code_generator)",
                )
            )

    # Check model configurations.
    invalid_models = []
    for model_name, model_config in models.items():
        if not isinstance(model_config, dict):
            invalid_models.append(model_name)
            continue
        if "provider" not in model_config:
            invalid_models.append(f"{model_name} (missing provider)")
        if "model_id" not in model_config:
            invalid_models.append(f"{model_name} (missing model_id)")

    if invalid_models:
        results.append(
            CheckResult(
                name="model_configs_valid",
                category=_CATEGORY,
                status=Status.WARNING,
                message=f"Invalid model configurations: {', '.join(invalid_models)}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="model_configs_valid",
                category=_CATEGORY,
                status=Status.OK,
                message="All model configurations valid",
            )
        )

    # Check deployed_services. An empty list is a supported shape, not a fault:
    # presets that attach to an existing stack and the standalone templates ship
    # it empty on purpose. The sibling ``containers`` category already treats the
    # same empty list as normal, so this one must not scold either.
    deployed_services = config.get("deployed_services", [])
    if not deployed_services:
        results.append(
            CheckResult(
                name="deployed_services",
                category=_CATEGORY,
                status=Status.SKIP,
                message="No services deployed — attached or service-free project",
            )
        )
    else:
        results.append(
            CheckResult(
                name="deployed_services",
                category=_CATEGORY,
                status=Status.OK,
                message=(
                    f"{len(deployed_services)} services configured: {', '.join(deployed_services)}"
                ),
            )
        )

    # Check if services in deployed_services exist in the services section.
    services = config.get("services", {})
    undefined_services = [s for s in deployed_services if s not in services]
    if undefined_services:
        results.append(
            CheckResult(
                name="service_definitions",
                category=_CATEGORY,
                status=Status.ERROR,
                message=f"Services not defined: {', '.join(undefined_services)}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="service_definitions",
                category=_CATEGORY,
                status=Status.OK,
                message="All deployed services defined",
            )
        )

    # Check API providers.
    api_providers = config.get("api", {}).get("providers", {})
    if not api_providers:
        results.append(
            CheckResult(
                name="api_providers",
                category=_CATEGORY,
                status=Status.WARNING,
                message="No API providers configured",
            )
        )
    else:
        results.append(
            CheckResult(
                name="api_providers",
                category=_CATEGORY,
                status=Status.OK,
                message=(
                    f"{len(api_providers)} providers configured: {', '.join(api_providers.keys())}"
                ),
            )
        )

    return results


def _check_environment_variables(config: dict[str, Any]) -> list[CheckResult]:
    """Check if environment variables referenced in config are set.

    Emits no row when the config references no ``${VAR}`` placeholders.
    """
    config_str = str(config)
    env_vars = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", config_str)
    env_vars = list(set(env_vars))  # Remove duplicates.

    missing_vars = [var for var in env_vars if var not in os.environ]

    if missing_vars:
        return [
            CheckResult(
                name="environment_variables",
                category=_CATEGORY,
                status=Status.WARNING,
                message=f"Missing environment variables: {', '.join(missing_vars)}",
                details=("These variables are referenced in config.yml but not set in environment"),
            )
        ]
    if env_vars:
        return [
            CheckResult(
                name="environment_variables",
                category=_CATEGORY,
                status=Status.OK,
                message=f"All {len(env_vars)} environment variables set",
            )
        ]
    return []


def _check_timezone(config: dict[str, Any]) -> CheckResult:
    """Check if timezone is configured (not left as the UTC default)."""
    from osprey.utils.config import resolve_env_vars

    tz_raw = config.get("system", {}).get("timezone", "UTC")
    tz = resolve_env_vars(tz_raw) if isinstance(tz_raw, str) else tz_raw
    if tz == "UTC":
        return CheckResult(
            name="timezone",
            category=_CATEGORY,
            status=Status.WARNING,
            message="Timezone is UTC (default)",
            details=(
                "Set system.timezone under `config:` in profile.yml to your "
                "facility timezone (e.g., America/New_York, Europe/Berlin) "
                "and run `osprey build`."
            ),
        )
    return CheckResult(
        name="timezone",
        category=_CATEGORY,
        status=Status.OK,
        message=f"Timezone: {tz}",
    )


def _check_ariel_dsn_port(config: dict[str, Any]) -> list[CheckResult]:
    """Cross-check an explicit ARIEL DSN against the Postgres port it should use.

    With ``ariel.database.uri`` unset the DSN is derived from
    ``services.postgresql`` (see
    :func:`osprey.services.ariel_search.config.resolve_ariel_dsn`), so it follows
    a port move by construction and there is nothing to cross-check. An explicit
    uri — including one still spelled with the retired ``connection_string``
    alias, which is honored as the effective uri — is a second, independent copy
    of the same facts: a config rendered before the project moved ``port_host``
    keeps its literal ``:5432`` and points at a database that is no longer there.

    Only a loopback DSN is checked. A uri naming an external database host is a
    legitimate override that ``port_host`` says nothing about, as is any config
    with no ``services.postgresql`` block at all.

    Returns:
        A single row when both ports are known and comparable; no rows at all
        when there is nothing to compare (derived DSN, external database, no
        Postgres service, or a uri whose port cannot be read).
    """
    from osprey.utils.config import resolve_env_vars

    database = (config.get("ariel") or {}).get("database") or {}
    if "uri" in database:
        key = "ariel.database.uri"
        uri = database["uri"]
    elif "connection_string" in database:
        key = "ariel.database.connection_string"
        uri = database["connection_string"]
    else:
        return []

    if not isinstance(uri, str):
        return []

    postgresql = (config.get("services") or {}).get("postgresql") or {}
    if "port_host" not in postgresql:
        return []

    try:
        service_port = int(postgresql["port_host"])
    except (TypeError, ValueError):
        return []

    uri_port = _loopback_dsn_port(resolve_env_vars(uri))
    if uri_port is None:
        return []

    if uri_port == service_port:
        return [
            CheckResult(
                name="ariel_dsn_port",
                category=_CATEGORY,
                status=Status.OK,
                message=f"ARIEL DSN port {uri_port} matches services.postgresql.port_host",
            )
        ]

    return [
        CheckResult(
            name="ariel_dsn_port",
            category=_CATEGORY,
            status=Status.WARNING,
            message=(
                f"{key} points at port {uri_port}, but "
                f"services.postgresql.port_host is {service_port}"
            ),
            details=(
                f"The explicit {key} is a second copy of the Postgres port and no longer "
                f"matches services.postgresql.port_host ({service_port}), the port this "
                f"project actually publishes — so ARIEL connects to nothing. Either delete "
                f"{key} from config.yml, which derives the DSN from services.postgresql "
                f"(username, database_name, port_host) so it follows any future port move, "
                f"or set ariel.database.uri to port {service_port}."
            ),
        )
    ]


def _loopback_dsn_port(uri: Any) -> int | None:
    """Read the port from a DSN aimed at the local host.

    Returns:
        The explicit port, or ``None`` when the uri is not a parseable loopback
        DSN carrying one — an external host, a portless DSN, or a value with an
        unresolved ``${VAR}`` placeholder where the port belongs.
    """
    if not isinstance(uri, str):
        return None
    try:
        parts = urlsplit(uri)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if host is None or host.lower() not in _LOOPBACK_HOSTS:
        return None
    return port


# Re-exported so the CLI can build states without importing private helpers.
__all__ = ["ConfigState", "configuration"]
