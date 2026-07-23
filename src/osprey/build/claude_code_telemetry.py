"""OTEL / Claude Code telemetry env-block construction.

Builds the ``OTEL_*`` / ``CLAUDE_CODE_ENABLE_TELEMETRY`` environment block that
:class:`osprey.build.claude_code_resolver.ClaudeCodeModelResolver` injects into a
launch when ``claude_code.telemetry`` is enabled. Kept separate from the
resolver's provider/model logic because telemetry is a distinct observability
concern: :func:`_build_telemetry_env` does no I/O and reads no env (all
``${VAR}``-expansion and container detection happen upstream), which makes it
directly unit-testable — see ``tests/cli/test_telemetry_env.py``.
"""

from __future__ import annotations

import base64
import os
import warnings

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


# Config values that suppress a content gate. ``resolve_env_vars`` turns
# ``${VAR:-false}`` into the *string* "false" (YAML never re-coerces it to a
# bool), so a bare ``is not False`` check would silently keep the gate ON. Treat
# the common false-y spellings as OFF, matching bash/YAML intuition.
_GATE_FALSEY: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


def _gate_is_on(value: object) -> bool:
    """Whether a content-capture gate is enabled — default ON.

    A gate is suppressed only by an explicit false-y value: the bool ``False``
    (bare YAML ``false``) or a case-insensitive false-y string
    (``"false"``/``"0"``/``"no"``/``"off"``/``""``) — the latter is what
    ``resolve_env_vars`` yields for ``${VAR:-false}``. A missing key (``None``)
    or any other value leaves the gate ON.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in _GATE_FALSEY
    return bool(value)


class TelemetryConfigError(ValueError):
    """A ``claude_code.telemetry`` block is enabled but misconfigured.

    A subclass of :class:`ValueError` so every existing caller that catches
    ``ValueError`` (the template-render paths) keeps working unchanged, while a
    caller that must not let an observability misconfiguration take down an
    orthogonal concern — the dispatch worker's provider-auth injection — can
    catch this type specifically and degrade telemetry without dropping auth.
    """


def _running_in_container() -> bool:
    """Best-effort detection of whether this process runs inside a container.

    Consulted once per :meth:`ClaudeCodeModelResolver.resolve` call to pick the
    OpenObserve default host (``openobserve`` service name vs ``localhost``).
    This is the single place container/filesystem/env state is read; the pure
    :func:`_build_telemetry_env` helper never touches ``os``.
    """
    return os.path.exists("/.dockerenv") or bool(os.environ.get("OSPREY_IN_CONTAINER"))


def _openobserve_host_override() -> str | None:
    """Explicit OpenObserve host from the deploy environment, or ``None``.

    ``_running_in_container`` cannot know the network topology: whether the
    ``openobserve`` compose service is reachable by DNS name depends on the
    network mode, which only the compose author knows. A bridge-networked
    service sets ``OSPREY_OTEL_OPENOBSERVE_HOST=openobserve`` so the emitter
    targets the service DNS name regardless of runtime (docker *or* podman); a
    host-networked deployment leaves it unset so the derived ``localhost`` wins.
    Kept deliberately docker-agnostic — this env var, not filesystem sniffing,
    is the reliable in-container signal.
    """
    return os.environ.get("OSPREY_OTEL_OPENOBSERVE_HOST") or None


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


def _resolve_telemetry_endpoint(
    telemetry_cfg: dict, *, in_container: bool, openobserve_host: str | None = None
) -> str:
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
            # An explicit host from the deploy env wins (the compose author knows
            # the network topology); otherwise derive from container context.
            host = openobserve_host or ("openobserve" if in_container else "localhost")
            org = telemetry_cfg.get("openobserve", {}).get("org", "default")
            endpoint = f"http://{host}:5080/api/{org}"
        else:
            raise TelemetryConfigError(
                "claude_code.telemetry is enabled but no 'endpoint' is set and "
                "'backend' is not 'openobserve'; cannot resolve an OTLP endpoint. "
                "Set claude_code.telemetry.endpoint or backend: openobserve."
            )
    endpoint = str(endpoint)
    if "${" in endpoint:
        raise TelemetryConfigError(
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
        raise TelemetryConfigError(
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
            raise TelemetryConfigError(
                f"{field_name} still contains an unresolved '${{VAR}}': {cred!r}. "
                "The referenced environment variable is unset — refusing to "
                "encode a literal placeholder into the OpenObserve auth header."
            )
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return "Authorization", f"Basic {token}"


def _build_telemetry_env(
    telemetry_cfg: dict | None,
    *,
    in_container: bool = False,
    openobserve_host: str | None = None,
) -> dict[str, str]:
    """Build the OTEL/telemetry env block from the ``claude_code.telemetry`` config.

    Reads no environment and does no filesystem/network I/O — the
    ``${VAR}``-expansion, container detection, and host override happen
    upstream. Its only side effect is a single advisory ``warnings.warn`` when
    full content capture is active on a non-openobserve backend. Returns an
    empty dict when telemetry is absent or disabled.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section (already
            ``${VAR}``-expanded), or ``None``.
        in_container: Whether the agent runs in a container — selects the
            OpenObserve default host (``openobserve`` vs ``localhost``).
        openobserve_host: Explicit OpenObserve host from the deploy env; when
            set it wins over the ``in_container`` derivation (the compose author
            knows the network topology).

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
            telemetry_cfg, in_container=in_container, openobserve_host=openobserve_host
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

    # Content-capture gates default ON; each is dropped only on an explicit
    # false-y value (bool False, or a false-y string from ${VAR:-false}).
    on_gates = [
        env_var
        for env_var, cfg_key in _TELEMETRY_CONTENT_GATES.items()
        if _gate_is_on(telemetry_cfg.get(cfg_key))
    ]
    for env_var in on_gates:
        env[env_var] = "1"

    # Safe-state advisory: full-fidelity content capture is the default only
    # because the openobserve backend is local and air-gapped. On any other
    # backend the captured transcripts leave the host, so warn once (the
    # warnings registry deduplicates identical messages) when content still
    # ships off-host.
    if on_gates and telemetry_cfg.get("backend") != "openobserve":
        warnings.warn(
            "claude_code.telemetry ships full content capture "
            f"({', '.join(sorted(on_gates))}) to a non-openobserve backend; "
            "these transcripts leave the host. Set log_user_prompts / "
            "log_assistant_responses / log_tool_details / log_raw_api_bodies "
            "to false to suppress categories you do not want to emit.",
            UserWarning,
            stacklevel=2,
        )

    return env
