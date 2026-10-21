"""OTEL / Claude Code telemetry env-block construction.

Builds the ``OTEL_*`` / ``CLAUDE_CODE_ENABLE_TELEMETRY`` environment block that
:class:`osprey.agent_runner.provider_env.ClaudeCodeModelResolver` injects into a
launch when ``claude_code.telemetry`` is enabled. Kept separate from the
resolver's provider/model logic because telemetry is a distinct observability
concern: :func:`_build_telemetry_env` does no I/O and reads no env (all
``${VAR}``-expansion and gateway addressing happen upstream), which makes it
directly unit-testable — see ``tests/cli/test_telemetry_env.py``.
"""

from __future__ import annotations

import base64
import os
import uuid
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any

from osprey_connectors.connection import ENV_NAME_RE

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
        "OTEL_TRACES_EXPORTER",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA",
        "OTEL_LOG_TOOL_CONTENT",
        "CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH",
    }
)

# Content-capture gates: each content-capture env var → the config key that
# suppresses it. Each defaults ON (emitted as "1") and is emitted as an
# explicit "0" when its config key is false-y.
_TELEMETRY_CONTENT_GATES: dict[str, str] = {
    "OTEL_LOG_USER_PROMPTS": "log_user_prompts",
    "OTEL_LOG_ASSISTANT_RESPONSES": "log_assistant_responses",
    "OTEL_LOG_TOOL_DETAILS": "log_tool_details",
    "OTEL_LOG_RAW_API_BODIES": "log_raw_api_bodies",
    "OTEL_LOG_TOOL_CONTENT": "log_tool_content",
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


def _content_max_length(telemetry_cfg: dict) -> str | None:
    """The ``CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH`` value, or ``None`` to emit nothing.

    The longest content value Claude Code puts in one telemetry attribute
    before it cuts it and marks it truncated, in UTF-16 code units. Unset (or
    ``""``, what ``${VAR:-}`` yields) leaves Claude Code's own default (61440).
    Read from config only, never from ``os.environ``.

    Raises:
        TelemetryConfigError: When the value is not a positive integer.
    """
    raw = telemetry_cfg.get("content_max_length")
    if raw is None or raw == "":
        return None
    try:
        # YAML ``true`` is an int subclass; refuse it rather than read it as 1.
        if isinstance(raw, bool):
            raise ValueError(raw)
        value = int(str(raw).strip())
        if value < 1:
            raise ValueError(raw)
    except ValueError:
        raise TelemetryConfigError(
            "claude_code.telemetry.content_max_length must be a positive integer "
            f"(UTF-16 code units), got {raw!r}"
        ) from None
    return str(value)


#: Each signal ``claude_code.telemetry.signals`` may name -> the variable that
#: selects its exporter. Ordered as the refusal message lists them.
_SIGNAL_EXPORTERS: dict[str, str] = {
    "metrics": "OTEL_METRICS_EXPORTER",
    "logs": "OTEL_LOGS_EXPORTER",
    "traces": "OTEL_TRACES_EXPORTER",
}


def _exported_signals(telemetry_cfg: dict) -> frozenset[str]:
    """The signals the agent exports, from ``claude_code.telemetry.signals``.

    Absent (or ``None``) means every signal. Otherwise the value is a non-empty
    list drawn from ``metrics``, ``logs`` and ``traces``; a repeated member
    counts once. An empty list is refused rather than read as "export nothing",
    because ``enabled: false`` already says that and says it for the whole
    block.

    Raises:
        TelemetryConfigError: When the value is not a list, is empty, or names
            anything other than the three signals.
    """
    raw = telemetry_cfg.get("signals")
    if raw is None:
        return frozenset(_SIGNAL_EXPORTERS)
    allowed = ", ".join(_SIGNAL_EXPORTERS)
    if not isinstance(raw, list):
        raise TelemetryConfigError(
            f"claude_code.telemetry.signals must be a list drawn from {allowed}, got {raw!r}"
        )
    if not raw:
        raise TelemetryConfigError(
            "claude_code.telemetry.signals is empty; set claude_code.telemetry.enabled: "
            "false to export nothing"
        )
    unknown = sorted({str(member) for member in raw} - set(_SIGNAL_EXPORTERS))
    if unknown:
        raise TelemetryConfigError(
            f"claude_code.telemetry.signals names {', '.join(unknown)}; the signals are {allowed}"
        )
    return frozenset(raw)


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
        unresolved_vars: The environment variables that must be set for the
            launch to authenticate — empty when a credential is set but
            malformed. Carried on the exception rather
            than left to be recovered from the message text, so a caller that
            names the variable to set never re-extracts it from prose.
    """

    def __init__(self, message: str, unresolved_vars: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.unresolved_vars = unresolved_vars


#: Characters a bearer token may not carry: each would split or corrupt the
#: comma-joined ``OTEL_EXPORTER_OTLP_HEADERS`` list the exporter parses.
_HEADER_LIST_BREAKERS = frozenset(",\r\n")


#: The ``claude_code.telemetry.backend`` value for the deployment's own
#: telemetry gateway.
GATEWAY_BACKEND = "gateway"

#: The ``claude_code.telemetry.backend`` value for a collector the deployment
#: does not run, named by ``claude_code.telemetry.endpoint``. An unset backend
#: means the same.
GENERIC_BACKEND = "generic"

_BACKENDS: tuple[str, ...] = (GATEWAY_BACKEND, GENERIC_BACKEND)

#: The OTLP protocols the gateway's receiver answers. It listens for OTLP over
#: HTTP only, so a gRPC exporter aimed there would drop everything silently.
GATEWAY_PROTOCOLS: tuple[str, ...] = ("http/protobuf", "http/json")

#: The variable carrying a sender's ``<identity>:<password>`` login to the
#: gateway (``osprey.deployment.telemetry_credentials.LOGIN_ENV``). A literal
#: so this module stays importable without the deployment package; a test pins
#: the two together.
TELEMETRY_LOGIN_ENV = "OSPREY_TELEMETRY_LOGIN"

#: The launch-scoped terminal session (``osprey.audit.envelope.TERMINAL_SESSION_ENV``).
#: Read by exact name, never by prefix. A literal for the same reason as
#: :data:`TELEMETRY_LOGIN_ENV`.
TERMINAL_SESSION_ENV = "OSPREY_TERMINAL_SESSION"

#: The OTLP resource attribute that carries :data:`TERMINAL_SESSION_ENV`, so a
#: record in the store joins the ledger line its launch wrote.
TERMINAL_SESSION_ATTRIBUTE = "terminal.session"

_RESOURCE_ATTRIBUTES_ENV = "OTEL_RESOURCE_ATTRIBUTES"

#: Every key the ``claude_code.telemetry`` block may carry: exactly the keys the
#: readers in this module and :mod:`osprey.deployment.telemetry_config` take
#: from it. The profile parser closes the block against this set, so a key no
#: reader takes is refused at build instead of being carried silently.
#: ``resource_attributes`` is an open mapping below it; ``auth`` is closed to
#: :data:`TELEMETRY_AUTH_KEYS`.
TELEMETRY_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "backend",
        "protocol",
        "endpoint",
        "auth",
        "signals",
        "resource_attributes",
        *_TELEMETRY_CONTENT_GATES.values(),
        "content_max_length",
    }
)

#: Every key ``claude_code.telemetry.auth`` may carry.
TELEMETRY_AUTH_KEYS: frozenset[str] = frozenset({"token_env"})


def telemetry_backend(telemetry_cfg: Mapping[str, Any]) -> str:
    """The block's backend: :data:`GATEWAY_BACKEND` or :data:`GENERIC_BACKEND`.

    An unset backend is :data:`GENERIC_BACKEND`: the block names a collector
    of its own through ``claude_code.telemetry.endpoint``.

    Raises:
        TelemetryConfigError: When ``backend`` names anything else.
    """
    backend = telemetry_cfg.get("backend")
    if backend is None or backend == "":
        return GENERIC_BACKEND
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise TelemetryConfigError(
            f"claude_code.telemetry.backend is {backend!r}; it is '{GATEWAY_BACKEND}' "
            f"(the deployment's telemetry gateway) or '{GENERIC_BACKEND}' (a collector "
            "named by claude_code.telemetry.endpoint)."
        )
    return backend


def _enabled_block(config: Mapping[str, Any] | None) -> dict | None:
    """``claude_code.telemetry`` when it is a mapping with ``enabled`` set, else ``None``."""
    telemetry_cfg = ((config or {}).get("claude_code") or {}).get("telemetry")
    if not isinstance(telemetry_cfg, dict) or not telemetry_cfg.get("enabled"):
        return None
    return telemetry_cfg


def build_gateway_url(config: Mapping[str, Any] | None) -> str | None:
    """The gateway address a build-time render bakes in, or ``None`` for no gateway.

    The host-side address this config publishes (``telemetry_config.gateway_url``
    with ``bridge=False``) — never the builder's environment, which a rendered
    artifact must not carry. ``None`` when telemetry is off or the block names
    no gateway backend.

    Raises:
        TelemetryConfigError: When the backend is unknown.
    """
    telemetry_cfg = _enabled_block(config)
    if telemetry_cfg is None:
        return None
    if telemetry_backend(telemetry_cfg) != GATEWAY_BACKEND:
        return None
    from osprey.deployment.telemetry_config import gateway_url

    return gateway_url(config, bridge=False)


def resolve_runtime_telemetry_endpoint(
    config: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None
) -> str | None:
    """The OTLP endpoint an agent launched with *environ* would post to.

    The same inputs the launch path hands :func:`_build_telemetry_env`, so
    what the ``reach`` health category knocks on is what the exporter dials.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        environ: The launch environment; ``os.environ`` when omitted.

    Returns:
        The endpoint URL, or ``None`` when telemetry is off or misconfigured
        (the launch path refuses that loudly; the probe has nothing to knock on).
    """
    telemetry_cfg = _enabled_block(config)
    if telemetry_cfg is None:
        return None
    try:
        gateway_url = None
        if telemetry_backend(telemetry_cfg) == GATEWAY_BACKEND:
            from osprey.deployment.telemetry_config import runtime_gateway_url

            gateway_url = runtime_gateway_url(config, os.environ if environ is None else environ)
        return _resolve_telemetry_endpoint(telemetry_cfg, gateway_url=gateway_url)
    except ValueError:
        return None


def telemetry_auth_token_env(telemetry_cfg: Mapping[str, Any]) -> str | None:
    """The variable ``claude_code.telemetry.auth.token_env`` names, or ``None``.

    The one reader of the key: the launch builder, the web-terminal render and
    the deploy gate all ask it, so they agree on which variable carries the
    collector's bearer token. The value is a variable NAME, like
    ``archiver.settings.password_env``; a refusal never echoes what was written
    there, because a ``${VAR}`` in this key has already been expanded by the
    config loader into the secret itself.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section.

    Returns:
        The variable name, or ``None`` when the block has no ``auth``.

    Raises:
        TelemetryConfigError: When ``auth`` is set on the ``gateway``
            backend (each sender authenticates with its own login), is not a mapping
            whose only member is ``token_env``, or ``token_env`` is not a
            variable name.
    """
    auth = telemetry_cfg.get("auth")
    if auth is None:
        return None
    if telemetry_backend(telemetry_cfg) == GATEWAY_BACKEND:
        raise TelemetryConfigError(
            "claude_code.telemetry.auth is set on the gateway backend, where each "
            f"sender authenticates with its own login ({TELEMETRY_LOGIN_ENV}). "
            "Remove claude_code.telemetry.auth."
        )
    if not isinstance(auth, Mapping) or set(auth) != TELEMETRY_AUTH_KEYS:
        raise TelemetryConfigError(
            "claude_code.telemetry.auth takes one member: the collector takes a "
            "bearer token named by claude_code.telemetry.auth.token_env."
        )
    token_env = auth["token_env"]
    if not isinstance(token_env, str) or not ENV_NAME_RE.match(token_env):
        raise TelemetryConfigError(
            "claude_code.telemetry.auth.token_env names the environment variable "
            "that holds the collector's bearer token; it holds a name, not a "
            "value or a ${VAR} reference."
        )
    return token_env


def _bearer_header(
    token_env: str, auth_token: str | None, *, defer_unresolved: bool
) -> tuple[str, str] | None:
    """``("Authorization", "Bearer <token>")`` for a collector's token.

    Args:
        token_env: The variable :func:`telemetry_auth_token_env` read.
        auth_token: That variable's value from the launch environment, handed
            in by the caller.
        defer_unresolved: When True, an absent token returns ``None`` without a
            warning: a build never carries the secret, so its absence there is
            the correct state, not a fault.

    Raises:
        ObservabilityCredentialError: When the token is unset or blank and
            ``defer_unresolved`` is False (compose's ``${VAR:-}`` makes an
            unset host variable a blank one, so the two are one fault), or when
            it carries a comma or a line break.
    """
    if auth_token is None or not auth_token.strip():
        if defer_unresolved:
            return None
        raise ObservabilityCredentialError(
            f"claude_code.telemetry.auth.token_env names {token_env}, which is "
            "unset or blank; set it in the deployment's .env.",
            (token_env,),
        )
    if _HEADER_LIST_BREAKERS & set(auth_token):
        raise ObservabilityCredentialError(
            f"{token_env} (claude_code.telemetry.auth.token_env) contains a comma "
            "or a line break, which a bearer token in the OTLP header list "
            "cannot carry."
        )
    return "Authorization", f"Bearer {auth_token}"


def _render_kv_map(value: dict | str) -> str:
    """Render a mapping as a comma-joined ``k=v`` string (OTLP wire format).

    A pre-formatted string is returned unchanged; a mapping is rendered as
    comma-joined ``key=value`` pairs. Used for both resource attributes and the
    OTLP headers block.
    """
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _resolve_telemetry_endpoint(telemetry_cfg: dict, *, gateway_url: str | None = None) -> str:
    """Resolve the OTLP exporter endpoint, failing loud on misconfiguration.

    Under ``backend: gateway`` the endpoint is the gateway address the caller
    resolved for where the launch runs (``telemetry_config.runtime_gateway_url``
    at runtime, ``gateway_url`` at build time); the block names no endpoint of
    its own and speaks one of :data:`GATEWAY_PROTOCOLS`. Without a backend the
    block's ``endpoint`` is used verbatim.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` section.
        gateway_url: The gateway's OTLP/HTTP base URL from where this launch
            runs. Required under ``backend: gateway``.

    Raises:
        TelemetryConfigError: On an unknown backend; under ``backend: gateway``
            on an ``endpoint`` key, a protocol the gateway does not answer
            (``grpc`` among them) or no gateway address; without a backend on
            a missing endpoint; and on an endpoint that still carries a literal
            ``${VAR}`` (an unresolved placeholder whose variable is unset).
    """
    if telemetry_backend(telemetry_cfg) == GATEWAY_BACKEND:
        if telemetry_cfg.get("endpoint"):
            raise TelemetryConfigError(
                "claude_code.telemetry.endpoint is set with backend: gateway, which "
                "sends to the deployment's own gateway. Remove the endpoint, or remove "
                "the backend to export to that collector instead."
            )
        protocol = str(telemetry_cfg.get("protocol", GATEWAY_PROTOCOLS[0])).strip().lower()
        if protocol not in GATEWAY_PROTOCOLS:
            raise TelemetryConfigError(
                f"claude_code.telemetry.protocol is {protocol!r}, but the telemetry gateway "
                f"receives OTLP over HTTP only ({', '.join(GATEWAY_PROTOCOLS)}); an exporter "
                "speaking anything else would drop every metric, log and span silently. "
                f"Use {GATEWAY_PROTOCOLS[0]}, or set backend: {GENERIC_BACKEND} and name a "
                "gRPC collector in claude_code.telemetry.endpoint."
            )
        if not gateway_url:
            raise TelemetryConfigError(
                "claude_code.telemetry.backend is 'gateway' but no gateway address was "
                "resolved for this launch."
            )
        endpoint = str(gateway_url)
    else:
        endpoint = str(telemetry_cfg.get("endpoint") or "")
        if not endpoint:
            raise TelemetryConfigError(
                "claude_code.telemetry is enabled but names neither backend: gateway nor "
                "an 'endpoint'; cannot resolve an OTLP endpoint."
            )
    if "${" in endpoint:
        raise TelemetryConfigError(
            f"OTLP endpoint still contains an unresolved '${{VAR}}': {endpoint!r}. "
            "The referenced environment variable is unset — refusing to ship a "
            "literal placeholder in the exporter URL."
        )
    return endpoint


def _gateway_auth_header(
    telemetry_login: str | None, *, defer_unresolved: bool
) -> tuple[str, str] | None:
    """``("Authorization", "Basic <base64 login>")`` for a sender's gateway login.

    Args:
        telemetry_login: The launch's :data:`TELEMETRY_LOGIN_ENV` value,
            ``<identity>:<password>``, handed in by the caller.
        defer_unresolved: When True, an absent login returns ``None``: a build
            never carries the login, so its absence there is the correct state.

    Raises:
        ObservabilityCredentialError: When the login is unset or blank and
            ``defer_unresolved`` is False — the gateway refuses an
            unauthenticated sender, so every record would be dropped — or when
            it is not ``<identity>:<password>`` or carries a comma or a line
            break.
    """
    login = (telemetry_login or "").strip()
    if not login:
        if defer_unresolved:
            return None
        raise ObservabilityCredentialError(
            f"{TELEMETRY_LOGIN_ENV} is unset or blank, so this launch has no login for "
            "the telemetry gateway and every record it sent would be refused. "
            "`osprey build` writes each sender's login to .env.telemetry-sender.<identity>; "
            "on the host, run it and check that this launch's identity is a configured "
            "sender. A container is handed its sender's .env.telemetry-sender.<identity> "
            "as an env file: compose does this, and `docker run` takes it with --env-file.",
            (TELEMETRY_LOGIN_ENV,),
        )
    identity, sep, password = login.partition(":")
    if not sep or not identity or not password or _HEADER_LIST_BREAKERS & set(login):
        raise ObservabilityCredentialError(
            f"{TELEMETRY_LOGIN_ENV} is not an <identity>:<password> login the OTLP header "
            "list can carry; delete the sender's .env.telemetry-sender.<identity> file and "
            "run `osprey build` to mint a new one."
        )
    token = base64.b64encode(login.encode()).decode()
    return "Authorization", f"Basic {token}"


def _attribute_pairs(value: dict | str) -> list[tuple[str, str]]:
    """``OTEL_RESOURCE_ATTRIBUTES`` as ordered ``(key, value)`` pairs."""
    if isinstance(value, dict):
        return [(str(k), str(v)) for k, v in value.items()]
    pairs: list[tuple[str, str]] = []
    for pair in str(value).split(","):
        if not pair.strip():
            continue
        key, _sep, val = pair.partition("=")
        pairs.append((key.strip(), val.strip()))
    return pairs


def _gateway_resource_attributes(telemetry_cfg: dict, terminal_session: str | None) -> str:
    """The resource attributes a gateway sender carries: the block's own plus the session.

    :data:`TERMINAL_SESSION_ATTRIBUTE` is always present, empty when the
    session is not known yet: a launch site that spawns a session later fills
    it (:func:`terminal_session_env`), and the gateway drops an empty one.

    Raises:
        TelemetryConfigError: When ``resource_attributes`` sets the session
            attribute itself; it is the launch's, never the config's.
    """
    pairs = _attribute_pairs(telemetry_cfg.get("resource_attributes") or {})
    if any(key == TERMINAL_SESSION_ATTRIBUTE for key, _ in pairs):
        raise TelemetryConfigError(
            f"claude_code.telemetry.resource_attributes sets {TERMINAL_SESSION_ATTRIBUTE}, "
            "which each launch sets to its own terminal session. Remove it."
        )
    pairs.append((TERMINAL_SESSION_ATTRIBUTE, terminal_session or ""))
    return ",".join(f"{key}={value}" for key, value in pairs)


def terminal_session_env(session: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """What a launch site adds to a child's environment to give it terminal session *session*.

    :data:`TERMINAL_SESSION_ENV` always — the ledger writers fall back to it
    when the child has no posture session. When the child inherits a gateway
    sender's resource attributes (they name :data:`TERMINAL_SESSION_ATTRIBUTE`),
    those attributes too, with the session filled in, so the child's OTLP
    resource and its ledger lines carry the same value. Any other backend's
    attributes are left as they are.

    Args:
        session: The launch-scoped id.
        inherited: The environment the child is otherwise spawned with.

    Returns:
        The variables to set on the child.
    """
    env = {TERMINAL_SESSION_ENV: session}
    attributes = inherited.get(_RESOURCE_ATTRIBUTES_ENV)
    if attributes is None:
        return env
    pairs = _attribute_pairs(attributes)
    if not any(key == TERMINAL_SESSION_ATTRIBUTE for key, _ in pairs):
        return env
    env[_RESOURCE_ATTRIBUTES_ENV] = ",".join(
        f"{key}={session if key == TERMINAL_SESSION_ATTRIBUTE else value}" for key, value in pairs
    )
    return env


#: The posture session a terminal's child carries (``osprey.audit.posture``).
#: Read only, never set here: it is a write-safety input.
_POSTURE_SESSION_ENV = "OSPREY_POSTURE_SESSION"

#: The prefix of the terminal session minted for a launch on the host.
HOST_SESSION_PREFIX = "chat-"

#: The prefix of a dispatched job's terminal session, ``dispatch-<run id>``.
DISPATCH_SESSION_PREFIX = "dispatch-"


def export_host_terminal_session(environ: MutableMapping[str, str]) -> str:
    """Give a session launched on the host (``osprey chat``/``query``) its terminal session.

    A launch already inside a terminal session keeps it — the one it was
    handed (:data:`TERMINAL_SESSION_ENV`), else the posture session its
    ledger lines are filed under — so the two never part. Otherwise a fresh
    ``chat-<uuid4>``. Only :data:`TERMINAL_SESSION_ENV` is set: the posture
    session decides approval stamps and write blocking, and a host launch has
    none.

    Returns:
        The session the launch carries.
    """
    for name in (TERMINAL_SESSION_ENV, _POSTURE_SESSION_ENV):
        existing = (environ.get(name) or "").strip()
        if existing:
            environ[TERMINAL_SESSION_ENV] = existing
            return existing
    session = f"{HOST_SESSION_PREFIX}{uuid.uuid4()}"
    environ[TERMINAL_SESSION_ENV] = session
    return session


def _build_telemetry_env(
    telemetry_cfg: dict | None,
    *,
    gateway_url: str | None = None,
    telemetry_login: str | None = None,
    terminal_session: str | None = None,
    defer_unresolved_creds: bool = False,
    auth_token: str | None = None,
) -> dict[str, str]:
    """Build the OTEL/telemetry env block from the ``claude_code.telemetry`` config.

    Reads no environment and does no filesystem/network I/O — the
    ``${VAR}``-expansion, the gateway address and the login are resolved
    upstream. Its only side effect is a single advisory ``warnings.warn`` when
    full content capture ships to a collector outside the deployment. Returns
    an empty dict when telemetry is absent or disabled.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section (already
            ``${VAR}``-expanded), or ``None``.
        gateway_url: The gateway's OTLP/HTTP base URL from where this launch
            runs (see :func:`_resolve_telemetry_endpoint`).
        telemetry_login: The launch's :data:`TELEMETRY_LOGIN_ENV` value.
        terminal_session: The launch's :data:`TERMINAL_SESSION_ENV` value,
            carried as :data:`TERMINAL_SESSION_ATTRIBUTE` under the gateway.
        defer_unresolved_creds: When True, a credential that is not resolvable
            yet is omitted instead of refused (build-time renders).
        auth_token: The value of the variable
            ``claude_code.telemetry.auth.token_env`` names, read by the caller
            from the launch environment it was handed.

    Returns:
        A ``{VAR: "value"}`` dict; every value is a string (never bool). Keys
        are a subset of :data:`TELEMETRY_ENV_VARS`.

    Raises:
        TelemetryConfigError: On an unknown backend, an unresolvable endpoint, a
            protocol the gateway does not answer, a leaked ``${VAR}`` in the
            endpoint, a ``content_max_length`` that is not a positive integer,
            a ``signals`` value that is empty, not a list, or names anything
            but ``metrics``, ``logs`` and ``traces``, an ``auth`` block the
            collector cannot use (see :func:`telemetry_auth_token_env`), a
            ``headers`` key, or ``resource_attributes`` naming the terminal
            session under the gateway.
        ObservabilityCredentialError: On a gateway login that is unset, blank
            or malformed, or on a collector token that is unset, blank, or
            carries a comma or a line break.
    """
    if not telemetry_cfg or not telemetry_cfg.get("enabled"):
        return {}

    gateway = telemetry_backend(telemetry_cfg) == GATEWAY_BACKEND
    # The exporter appends ``/v1/<signal>`` to the base endpoint, so one
    # endpoint serves logs, metrics and traces.
    signals = _exported_signals(telemetry_cfg)
    env: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1" if "traces" in signals else "0",
        # A signal left out is written as ``none``, never omitted: an omitted
        # exporter keeps whatever the launching shell or the project's ``.env``
        # holds, because the block is overlaid key by key. The tracing switch is
        # written either way for the same reason.
        **{
            exporter: "otlp" if signal in signals else "none"
            for signal, exporter in _SIGNAL_EXPORTERS.items()
        },
        "OTEL_EXPORTER_OTLP_PROTOCOL": str(telemetry_cfg.get("protocol", "http/protobuf")),
        "OTEL_EXPORTER_OTLP_ENDPOINT": _resolve_telemetry_endpoint(
            telemetry_cfg, gateway_url=gateway_url
        ),
    }

    # The header map holds the one Authorization value: the sender's gateway
    # login or the collector's bearer token, never both. OTLP wire format is
    # comma-separated k=v.
    if "headers" in telemetry_cfg:
        raise TelemetryConfigError(
            "claude_code.telemetry.headers is not a setting; name the variable that "
            "holds the collector's bearer token in claude_code.telemetry.auth.token_env"
        )
    headers: dict[str, str] = {}
    token_env = telemetry_auth_token_env(telemetry_cfg)
    if token_env is not None:
        bearer = _bearer_header(token_env, auth_token, defer_unresolved=defer_unresolved_creds)
        if bearer is not None:
            headers[bearer[0]] = bearer[1]
    if gateway:
        basic = _gateway_auth_header(telemetry_login, defer_unresolved=defer_unresolved_creds)
        if basic is not None:
            headers[basic[0]] = basic[1]
    if headers:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = _render_kv_map(headers)

    if gateway:
        env[_RESOURCE_ATTRIBUTES_ENV] = _gateway_resource_attributes(
            telemetry_cfg, terminal_session
        )
    else:
        resource_attrs = telemetry_cfg.get("resource_attributes")
        if resource_attrs:
            env[_RESOURCE_ATTRIBUTES_ENV] = _render_kv_map(resource_attrs)

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

    limit = _content_max_length(telemetry_cfg)
    if limit is not None:
        env["CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH"] = limit

    # Safe-state advisory: full-fidelity content capture is the default only
    # because the gateway feeds the deployment's own store. A collector named
    # by ``endpoint`` sits outside it, so the captured transcripts leave the
    # host; warn once (the warnings registry deduplicates identical messages).
    if on_gates and not gateway:
        warnings.warn(
            "claude_code.telemetry ships full content capture "
            f"({', '.join(sorted(on_gates))}) to a collector outside this deployment; "
            "these transcripts leave the host. Set log_user_prompts / "
            "log_assistant_responses / log_tool_details / log_tool_content / "
            "log_raw_api_bodies "
            "to false to suppress categories you do not want to emit.",
            UserWarning,
            stacklevel=2,
        )

    return env
