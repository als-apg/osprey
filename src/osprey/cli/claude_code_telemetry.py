"""OTEL / Claude Code telemetry env-block construction.

Builds the ``OTEL_*`` / ``CLAUDE_CODE_ENABLE_TELEMETRY`` environment block that
:class:`osprey.cli.claude_code_resolver.ClaudeCodeModelResolver` injects into a
launch when ``claude_code.telemetry`` is enabled. Kept separate from the
resolver's provider/model logic because telemetry is a distinct observability
concern: :func:`_build_telemetry_env` is pure and side-effect-free (all
``${VAR}``-expansion and container detection happen upstream), which makes it
directly unit-testable — see ``tests/cli/test_telemetry_env.py``.
"""

from __future__ import annotations

import base64
import os

# OTEL / Claude Code telemetry vars the resolver may inject into the env block.
#
# These are deliberately NOT in MANAGED_ENV_VARS: they configure observability,
# not the provider backend or model, so a stale shell export of one of them
# cannot silently reroute the agent. They are enumerated here so
# ``detect_env_conflicts`` can exempt them — a pre-existing operator OTEL export
# is a legitimate override, not a launch-blocking conflict.
TELEMETRY_ENV_VARS: frozenset[str] = frozenset(
    {
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "OTEL_METRICS_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_LOG_USER_PROMPTS",
        "OTEL_LOG_ASSISTANT_RESPONSES",
        "OTEL_LOG_TOOL_DETAILS",
        "OTEL_LOG_RAW_API_BODIES",
    }
)

# Content-capture gates: OTEL env var → config key that suppresses it.
# Each defaults ON (emitted as "1") and is dropped only when its config key is
# explicitly ``false``. ``OTEL_LOG_TOOL_CONTENT`` is intentionally absent — it
# requires tracing, which is out of scope for this metrics/logs pipeline.
_TELEMETRY_CONTENT_GATES: dict[str, str] = {
    "OTEL_LOG_USER_PROMPTS": "log_user_prompts",
    "OTEL_LOG_ASSISTANT_RESPONSES": "log_assistant_responses",
    "OTEL_LOG_TOOL_DETAILS": "log_tool_details",
    "OTEL_LOG_RAW_API_BODIES": "log_raw_api_bodies",
}


def _running_in_container() -> bool:
    """Best-effort detection of whether this process runs inside a container.

    Consulted once per :meth:`ClaudeCodeModelResolver.resolve` call to pick the
    OpenObserve default host (``openobserve`` service name vs ``localhost``).
    This is the single place container/filesystem/env state is read; the pure
    :func:`_build_telemetry_env` helper never touches ``os``.
    """
    return os.path.exists("/.dockerenv") or bool(os.environ.get("OSPREY_IN_CONTAINER"))


def _parse_header_map(value: dict | str) -> dict[str, str]:
    """Parse OTLP headers config into a ``{key: value}`` dict.

    Accepts either a mapping (used directly) or a comma-separated ``k=v`` string
    (as OTLP expects on the wire). Only the first ``=`` in each pair is treated
    as the separator, so header values may themselves contain ``=``.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    parsed: dict[str, str] = {}
    for pair in str(value).split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _sep, val = pair.partition("=")
        parsed[key.strip()] = val.strip()
    return parsed


def _render_kv_map(value: dict | str) -> str:
    """Render a mapping as a comma-joined ``k=v`` string (OTLP wire format).

    A pre-formatted string is returned unchanged; a mapping is rendered as
    comma-joined ``key=value`` pairs. Used for both resource attributes and the
    OTLP headers block.
    """
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _resolve_telemetry_endpoint(telemetry_cfg: dict, *, in_container: bool) -> str:
    """Resolve the OTLP exporter endpoint, failing loud on misconfiguration.

    Priority: an explicit ``endpoint`` wins verbatim; otherwise an
    ``openobserve`` backend derives the org endpoint from the container context.
    Any other enabled-but-unaddressed config is a misconfiguration and raises.

    Raises:
        ValueError: If no endpoint can be resolved, or if the resolved endpoint
            still carries a literal ``${VAR}`` — an unresolved placeholder whose
            referenced env var was unset. An unresolved var keeps the literal
            ``${VAR}`` (it does not become empty), so shipping it would point the
            exporter at a nonsense URL; refuse instead.
    """
    endpoint = telemetry_cfg.get("endpoint")
    if not endpoint:
        if telemetry_cfg.get("backend") == "openobserve":
            host = "openobserve" if in_container else "localhost"
            org = telemetry_cfg.get("openobserve", {}).get("org", "default")
            endpoint = f"http://{host}:5080/api/{org}"
        else:
            raise ValueError(
                "claude_code.telemetry is enabled but no 'endpoint' is set and "
                "'backend' is not 'openobserve'; cannot resolve an OTLP endpoint. "
                "Set claude_code.telemetry.endpoint or backend: openobserve."
            )
    endpoint = str(endpoint)
    if "${" in endpoint:
        raise ValueError(
            f"OTLP endpoint still contains an unresolved '${{VAR}}': {endpoint!r}. "
            "The referenced environment variable is unset — refusing to ship a "
            "literal placeholder in the exporter URL."
        )
    return endpoint


def _openobserve_auth_header(telemetry_cfg: dict) -> tuple[str, str]:
    """Compute the HTTP Basic auth header for an OpenObserve backend.

    Both credentials arrive already ``${VAR}``-expanded in the config (the
    config loader expands ``claude_code.telemetry`` before the resolver runs),
    so this reads them straight from the config — never ``os.environ``. The
    base64 header value cannot be expressed as ``${VAR}``; it is computed here.

    Returns:
        ``("Authorization", "Basic <base64(user:password)>")``.

    Raises:
        ValueError: If either credential is missing or blank — an
            unauthenticated exporter to an auth-gated store would silently drop
            every span, so refuse to emit one — or if either credential still
            carries a literal ``${VAR}`` (an unresolved placeholder whose env
            var is unset).
    """
    oo = telemetry_cfg.get("openobserve") or {}
    user = oo.get("user")
    password = oo.get("password")
    if not user or not password:
        raise ValueError(
            "claude_code.telemetry.backend is 'openobserve' but "
            "openobserve.user / openobserve.password are missing or blank; "
            "refusing to emit an unauthenticated OTLP exporter to an "
            "auth-gated store."
        )
    # Fail loud on an unresolved ${VAR} — same contract as the endpoint check.
    # The config loader leaves the literal ``${VAR}`` when the referenced env
    # var is unset; base64-encoding that literal would silently 401 at runtime
    # instead of failing clearly here at resolve() time.
    for field_name, cred in (("openobserve.user", user), ("openobserve.password", password)):
        if "${" in str(cred):
            raise ValueError(
                f"{field_name} still contains an unresolved '${{VAR}}': {cred!r}. "
                "The referenced environment variable is unset — refusing to "
                "encode a literal placeholder into the OpenObserve auth header."
            )
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return "Authorization", f"Basic {token}"


def _build_telemetry_env(
    telemetry_cfg: dict | None, *, in_container: bool = False
) -> dict[str, str]:
    """Build the OTEL/telemetry env block from the ``claude_code.telemetry`` config.

    Pure and side-effect-free: it reads no environment and does no I/O — the
    ``${VAR}``-expansion and container detection happen upstream. Returns an
    empty dict when telemetry is absent or disabled.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section (already
            ``${VAR}``-expanded), or ``None``.
        in_container: Whether the agent runs in a container — selects the
            OpenObserve default host (``openobserve`` vs ``localhost``).

    Returns:
        A ``{VAR: "value"}`` dict; every value is a string (never bool). Keys
        are a subset of :data:`TELEMETRY_ENV_VARS`.

    Raises:
        ValueError: On an unresolvable endpoint, a leaked ``${VAR}`` in the
            endpoint, or an ``openobserve`` backend with missing/blank creds.
    """
    if not telemetry_cfg or not telemetry_cfg.get("enabled"):
        return {}

    env: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": str(telemetry_cfg.get("protocol", "http/protobuf")),
        "OTEL_EXPORTER_OTLP_ENDPOINT": _resolve_telemetry_endpoint(
            telemetry_cfg, in_container=in_container
        ),
    }

    # Headers: config headers first, then the computed OpenObserve Basic auth,
    # which wins on key collision. OTLP wire format is comma-separated k=v.
    headers: dict[str, str] = {}
    configured = telemetry_cfg.get("headers")
    if configured:
        headers.update(_parse_header_map(configured))
    if telemetry_cfg.get("backend") == "openobserve":
        key, value = _openobserve_auth_header(telemetry_cfg)
        headers[key] = value
    if headers:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = _render_kv_map(headers)

    resource_attrs = telemetry_cfg.get("resource_attributes")
    if resource_attrs:
        env["OTEL_RESOURCE_ATTRIBUTES"] = _render_kv_map(resource_attrs)

    # Content-capture gates default ON; each is dropped only on an explicit false.
    for env_var, cfg_key in _TELEMETRY_CONTENT_GATES.items():
        if telemetry_cfg.get(cfg_key) is not False:
            env[env_var] = "1"

    return env
