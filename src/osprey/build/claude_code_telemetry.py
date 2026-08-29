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
import re
import warnings
from collections.abc import Mapping
from typing import Any

from osprey.port_layout import default_port, resolve_port_base

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
# Each defaults ON (emitted as "1") and is emitted as an explicit "0" when its
# config key is ``false``. ``OTEL_LOG_TOOL_CONTENT`` is intentionally absent —
# it requires tracing, which is out of scope for this metrics/logs pipeline.
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


class ObservabilityCredentialError(TelemetryConfigError):
    """The observability backend's credentials are missing or unresolved.

    The narrow case of a telemetry misconfiguration whose remedy is a
    credential in the deployment's own secret store — never a model, a
    provider, or an endpoint. Callers that resolve a provider spec catch
    broad error types and phrase their own remedy from what they caught, so
    without this distinction a missing observability password is reported as
    a bad model or an unreadable provider, sending the operator to fix
    something that was never wrong.

    Subclasses :class:`TelemetryConfigError` (and through it ``ValueError``),
    so every existing handler keeps catching it exactly as before.

    Attributes:
        unresolved_vars: The environment variables the refused credentials
            still name as literal ``${VAR}`` placeholders — empty when the
            credentials were simply missing or blank, which no environment can
            fill in. Carried on the exception rather than left to be recovered
            from the message text: a caller that has to decide whether the
            missing value is one its own deploy issues later (see
            ``container_lifecycle._STORE_ISSUED_VARS``) is deciding on a
            variable name, and re-extracting names from prose makes that
            decision hostage to the wording of a sentence written for a human.
    """

    def __init__(self, message: str, unresolved_vars: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.unresolved_vars = unresolved_vars


def telemetry_creds_are_store_issued(exc: ObservabilityCredentialError) -> bool:
    """Whether every credential the telemetry check refused is one a deploy issues.

    A store-issued credential (``container_lifecycle._STORE_ISSUED_VARS``) is
    minted by the service it authenticates against and written to the
    deployment's ``.env`` by ``osprey up``. Before the first start it is
    genuinely absent, and no operator can supply it — so on a freshly
    scaffolded deployment the rendered telemetry block names a variable that
    cannot exist yet. Every launch path that resolves a provider spec resolves
    telemetry with it, so refusing over that would make the deployment
    unreachable until an unrelated verb has run.

    Read off the registry rather than restated as a name list, so a credential
    added there is covered here on the same commit. All-or-nothing on purpose:
    one refused name an operator does have to set — an ordinary missing secret
    — and the refusal stands for the whole set, because a launch told "nothing
    to do here" about half the story is worse than the error. The missing/blank
    case names no variable at all and is likewise a refusal.

    Lives beside the exception it inspects rather than in any one caller: the
    launch paths that have to make this decision (``osprey chat``, the headless
    agent-run primitives, the Web Terminal lifespan) share no other module, and
    a second copy of the rule is a second thing to keep in step with the
    registry.

    Imported inside the function: ``container_lifecycle`` pulls in a large
    deployment surface that a launch has no other reason to load.

    Args:
        exc: The caught :class:`ObservabilityCredentialError`.

    Returns:
        True when the refusal is only about credentials this deployment issues.
    """
    from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS

    return bool(exc.unresolved_vars) and all(
        name in _STORE_ISSUED_VARS for name in exc.unresolved_vars
    )


#: Env-var names inside a ``${VAR}`` / ``${VAR:-default}`` reference. Applied to
#: the credential *value* the check refused, never to a rendered message, so the
#: names it yields are exactly the ones the config asked for.
_CREDENTIAL_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")


def _running_in_container() -> bool:
    """Best-effort detection of whether this process runs inside a container.

    Consulted once per :meth:`ClaudeCodeModelResolver.resolve` call to pick the
    OpenObserve default host (``openobserve`` service name vs ``localhost``),
    and only when :func:`_openobserve_host_override` named none. This is the
    single place container/filesystem/env state is read; the pure
    :func:`_build_telemetry_env` helper never touches ``os``.

    The signal is the runtime's own marker file — ``/.dockerenv`` under
    Docker, ``/run/.containerenv`` under Podman; probing only the Docker one
    answered ``False`` inside every Podman container. ``OSPREY_IN_CONTAINER``
    is an operator override on top of that, which the shipped compose
    deliberately does not set.

    None of it decides network topology, which is why
    :func:`_openobserve_host_override` is consulted first and wins: whether the
    ``openobserve`` service name resolves depends on the network mode, and only
    the compose author knows it. Kept identical to
    :func:`osprey.health.derive._in_container`, which is the same question asked
    by a package that must not import this one.
    """
    return (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or bool(os.environ.get("OSPREY_IN_CONTAINER"))
    )


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


#: The port the OpenObserve container LISTENS on. Fixed by the image, and the
#: container half of the publish line in
#: ``templates/services/openobserve/docker-compose.yml.j2``
#: (``<bind>:<published>:5080/tcp``) — which is why this literal survives the
#: port layout: it is a property of the image, not a host port ``port_base``
#: can move. It is the right port only from inside the store's own compose
#: network, where the store is reached by DNS name; from anywhere else — the
#: host, a host-networked container — the store is reached at the port it
#: PUBLISHES, which the layout places and a project moves freely.
OPENOBSERVE_LISTEN_PORT = 5080

#: The variable a compose author sets beside ``OSPREY_OTEL_OPENOBSERVE_HOST``
#: when the host it names is reached on a port the config cannot know — the
#: bridge case, where the compose DNS name answers on the listen port. Same
#: convention as the host variable: topology is declared, never sniffed.
OPENOBSERVE_PORT_ENV_VAR = "OSPREY_OTEL_OPENOBSERVE_PORT"


def openobserve_published_port(config: Mapping[str, Any] | None) -> int:
    """The port this deployment publishes the telemetry store on.

    Read from ``services.openobserve.port`` — the one key that moves it, and the
    same key the service template, the deploy preflight and ``osprey health``
    read. With no such key the port is the layout's ``openobserve`` slot at the
    base *this config* resolved, which is the number the service template
    publishes when nothing overrides it. A persona render carries both: profile
    inheritance copies the hosting profile's ``config:`` overlay into every
    attached render, so a moved store — or a moved base — moves every client.

    The base is always the one the config in hand names, never
    :data:`~osprey.port_layout.DEFAULT_PORT_BASE`. Two deployments coexist on a
    host by taking different bases, and a client that fell back to the default
    base would dial its neighbour's store rather than its own.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The published port, as an ``int``.

    Raises:
        TelemetryConfigError: The key is present but not an integer port.
        ValueError: ``deployment.port_base`` is set to a base no block can
            start at (:func:`~osprey.port_layout.resolve_port_base`).
    """
    services = (config or {}).get("services") or {}
    block = services.get("openobserve") if isinstance(services, Mapping) else None
    raw = block.get("port") if isinstance(block, Mapping) else None
    if raw is None:
        return default_port("openobserve", base=resolve_port_base(config))
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise TelemetryConfigError(
            f"services.openobserve.port must be an integer port, got {raw!r}"
        ) from None


def _openobserve_port_override() -> int | None:
    """Explicit OpenObserve port from the deploy environment, or ``None``.

    The port half of :func:`_openobserve_host_override`, read the same way and
    for the same reason. An unset or empty variable is no override; a set one
    that is not an integer is a compose-authoring error and is refused rather
    than silently ignored, because the value was written deliberately.

    Raises:
        TelemetryConfigError: The variable is set but not an integer port.
    """
    raw = os.environ.get(OPENOBSERVE_PORT_ENV_VAR)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise TelemetryConfigError(
            f"{OPENOBSERVE_PORT_ENV_VAR} must be an integer port, got {raw!r}"
        ) from None


def resolve_openobserve_port(config: Mapping[str, Any] | None) -> int:
    """The port a runtime launch reaches the telemetry store on.

    The deploy environment's declaration wins (:func:`_openobserve_port_override`);
    otherwise the port this deployment publishes
    (:func:`openobserve_published_port`), which with no
    ``services.openobserve.port`` key is the layout's ``openobserve`` slot at
    the base this config resolved. Runtime launch paths call this; build-time
    renders, which must not read the builder's environment, call
    :func:`openobserve_published_port` directly.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The port to dial the store on, as an ``int``.

    Raises:
        TelemetryConfigError: The environment variable or the config key is set
            but is not an integer port.
        ValueError: ``deployment.port_base`` is set to a base no block can
            start at.
    """
    return _openobserve_port_override() or openobserve_published_port(config)


def resolve_runtime_telemetry_endpoint(config: Mapping[str, Any] | None) -> str | None:
    """The OTLP endpoint an agent launched from this process would post to.

    The same inputs the launch path hands :func:`_build_telemetry_env` —
    container context, the deploy environment's host and port overrides, the
    port the deployment publishes — so what the ``reach`` health category
    knocks on is what the exporter dials from here, not a re-derivation.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The endpoint URL, or ``None`` when telemetry is off, not the
        OpenObserve backend, or misconfigured (the launch path refuses that
        loudly; the probe has nothing to knock on).
    """
    telemetry_cfg = ((config or {}).get("claude_code") or {}).get("telemetry")
    if not isinstance(telemetry_cfg, dict) or not telemetry_cfg.get("enabled"):
        return None
    try:
        return _resolve_telemetry_endpoint(
            telemetry_cfg,
            in_container=_running_in_container(),
            openobserve_host=_openobserve_host_override(),
            openobserve_port=resolve_openobserve_port(config),
        )
    except (TelemetryConfigError, ValueError):
        return None


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
    telemetry_cfg: dict,
    *,
    in_container: bool,
    openobserve_host: str | None = None,
    openobserve_port: int | None = None,
) -> str:
    """Resolve the OTLP exporter endpoint, failing loud on misconfiguration.

    Priority: an explicit ``endpoint`` wins verbatim; otherwise an
    ``openobserve`` backend derives the org endpoint from the container context.
    Any other enabled-but-unaddressed config is a misconfiguration and raises.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` section.
        in_container: Whether the agent runs in a container — selects the
            OpenObserve default host (``openobserve`` vs ``localhost``).
        openobserve_host: Explicit store host from the deploy env; wins over
            the ``in_container`` derivation.
        openobserve_port: The port the store is reached on from where this
            launch runs — the deploy env's declaration or the port the
            deployment publishes (:func:`resolve_openobserve_port`), which the
            layout places at ``port_base + 50``. ``None`` falls back to
            :data:`OPENOBSERVE_LISTEN_PORT`, the port the image listens on
            inside the store's own compose network, and the only caller that
            may take it is one on that network; every launch path passes the
            resolved port.

    Raises:
        ValueError: If no endpoint can be resolved; if ``protocol`` is ``grpc``
            while the endpoint would be derived from the OpenObserve backend
            (which serves HTTP only); or if the resolved endpoint still carries a
            literal ``${VAR}`` — an unresolved placeholder whose referenced env
            var was unset. An unresolved var keeps the literal ``${VAR}`` (it does
            not become empty), so shipping it would point the exporter at a
            nonsense URL; refuse instead.
    """
    endpoint = telemetry_cfg.get("endpoint")
    if not endpoint:
        if telemetry_cfg.get("backend") == "openobserve":
            # The derived URL below is the OpenObserve HTTP ingest API — the only
            # port that service publishes. A gRPC exporter aimed there connects to
            # nothing and drops every metric and log without an error, so refuse
            # the combination rather than emit a silently-dead exporter.
            if str(telemetry_cfg.get("protocol", "http/protobuf")).strip().lower() == "grpc":
                raise TelemetryConfigError(
                    "claude_code.telemetry.protocol is 'grpc' but the OTLP endpoint "
                    "is auto-derived from backend: openobserve, which serves HTTP "
                    "only — a gRPC exporter aimed there would drop every "
                    "metric and log silently. Use protocol: http/protobuf, or set an "
                    "explicit claude_code.telemetry.endpoint pointing at a collector "
                    "that speaks gRPC."
                )
            # An explicit host from the deploy env wins (the compose author knows
            # the network topology); otherwise derive from container context.
            host = openobserve_host or ("openobserve" if in_container else "localhost")
            # No resolved port means the caller is on the store's own compose
            # network, where the container answers on the port the image
            # listens on — the `:5080/tcp` container half of the publish line
            # in `templates/services/openobserve/docker-compose.yml.j2`. That
            # half is not a host port, so the layout does not move it; only the
            # published half in front of it moves with `port_base`.
            port = openobserve_port or OPENOBSERVE_LISTEN_PORT
            org = telemetry_cfg.get("openobserve", {}).get("org", "default")
            endpoint = f"http://{host}:{port}/api/{org}"
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


def _openobserve_auth_header(
    telemetry_cfg: dict, *, defer_unresolved: bool = False
) -> tuple[str, str] | None:
    """Compute the HTTP Basic auth header for an OpenObserve backend.

    Both credentials arrive already ``${VAR}``-expanded in the config (the
    config loader expands ``claude_code.telemetry`` before the resolver runs),
    so this reads them straight from the config — never ``os.environ``. The
    base64 header value cannot be expressed as ``${VAR}``; it is computed here.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section.
        defer_unresolved: When True, an unresolved ``${VAR}`` credential returns
            ``None`` (with a warning) instead of raising. Build-time callers set
            this: they render a scaffold whose credentials belong to the
            *deployment*, and the runtime re-resolves them at agent-spawn. A
            missing or blank credential still raises — no later step can fill
            that in.

    Returns:
        ``("Authorization", "Basic <base64(user:password)>")``, or ``None`` when
        a credential is deferred.

    Raises:
        ObservabilityCredentialError: If either credential is missing or blank
            — an unauthenticated exporter to an auth-gated store would silently
            drop every span, so refuse to emit one — or if either credential
            still carries a literal ``${VAR}`` (an unresolved placeholder whose
            env var is unset) and ``defer_unresolved`` is False. Both cases are
            fixed in the deployment's secret store, which is why they carry
            their own type rather than the general telemetry one. The
            unresolved case also carries the placeholder's variable names on
            ``unresolved_vars``; the missing/blank case carries none, since
            there is no variable to name.
    """
    oo = telemetry_cfg.get("openobserve") or {}
    user = oo.get("user")
    password = oo.get("password")
    if not user or not password:
        raise ObservabilityCredentialError(
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
            if defer_unresolved:
                warnings.warn(
                    f"{field_name} still contains an unresolved '${{VAR}}': {cred!r}. "
                    "Omitting the OpenObserve auth header from the rendered "
                    "scaffold; the runtime must resolve it at agent-spawn.",
                    stacklevel=2,
                )
                return None
            raise ObservabilityCredentialError(
                f"{field_name} still contains an unresolved '${{VAR}}': {cred!r}. "
                "The referenced environment variable is unset — refusing to "
                "encode a literal placeholder into the OpenObserve auth header.",
                tuple(sorted(set(_CREDENTIAL_PLACEHOLDER_RE.findall(str(cred))))),
            )
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return "Authorization", f"Basic {token}"


def _build_telemetry_env(
    telemetry_cfg: dict | None,
    *,
    in_container: bool = False,
    openobserve_host: str | None = None,
    openobserve_port: int | None = None,
    defer_unresolved_creds: bool = False,
) -> dict[str, str]:
    """Build the OTEL/telemetry env block from the ``claude_code.telemetry`` config.

    Reads no environment and does no filesystem/network I/O — the
    ``${VAR}``-expansion, container detection, and host/port resolution happen
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
        openobserve_port: The port the store is reached on from where this
            launch runs (see :func:`_resolve_telemetry_endpoint`).

    Returns:
        A ``{VAR: "value"}`` dict; every value is a string (never bool). Keys
        are a subset of :data:`TELEMETRY_ENV_VARS`.

    Raises:
        TelemetryConfigError: On an unresolvable endpoint, ``protocol: grpc``
            against an auto-derived (HTTP-only) OpenObserve endpoint, or a
            leaked ``${VAR}`` in the endpoint.
        ObservabilityCredentialError: On an ``openobserve`` backend whose
            credentials are missing, blank, or an unresolved ``${VAR}``.
    """
    if not telemetry_cfg or not telemetry_cfg.get("enabled"):
        return {}

    env: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": str(telemetry_cfg.get("protocol", "http/protobuf")),
        "OTEL_EXPORTER_OTLP_ENDPOINT": _resolve_telemetry_endpoint(
            telemetry_cfg,
            in_container=in_container,
            openobserve_host=openobserve_host,
            openobserve_port=openobserve_port,
        ),
    }

    # Headers: config headers first, then the computed OpenObserve Basic auth,
    # which wins on key collision. OTLP wire format is comma-separated k=v.
    headers: dict[str, str] = {}
    configured = telemetry_cfg.get("headers")
    if configured:
        headers.update(_parse_header_map(configured))
    if telemetry_cfg.get("backend") == "openobserve":
        auth = _openobserve_auth_header(telemetry_cfg, defer_unresolved=defer_unresolved_creds)
        if auth is not None:
            headers[auth[0]] = auth[1]
    if headers:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = _render_kv_map(headers)

    resource_attrs = telemetry_cfg.get("resource_attributes")
    if resource_attrs:
        env["OTEL_RESOURCE_ATTRIBUTES"] = _render_kv_map(resource_attrs)

    # Content-capture gates default ON; each turns off only on an explicit
    # false-y value (bool False, or a false-y string from ${VAR:-false}).
    # A disabled gate must be emitted as an explicit "0", never omitted: the
    # CLI resolves an absent var through its own fallback chain (e.g.
    # OTEL_LOG_ASSISTANT_RESPONSES ?? OTEL_LOG_USER_PROMPTS), which would
    # silently re-enable capture the config turned off.
    on_gates = [
        env_var
        for env_var, cfg_key in _TELEMETRY_CONTENT_GATES.items()
        if _gate_is_on(telemetry_cfg.get(cfg_key))
    ]
    for env_var in _TELEMETRY_CONTENT_GATES:
        env[env_var] = "1" if env_var in on_gates else "0"

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
