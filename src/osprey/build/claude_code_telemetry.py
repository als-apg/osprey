"""Claude Code's OTLP export: every launch sends its log events to the agent record.

:func:`record_telemetry_env` builds the ``OTEL_*`` / ``CLAUDE_CODE_*``
environment block that :class:`osprey.agent_runner.provider_env.ClaudeCodeModelResolver`
puts into a launch and :func:`osprey.agent_runner.provider_env.host_record_launch`
publishes to a host session's children. There is one destination and nothing in
a profile selects another: agent-record's ``/v1/logs``, at the address the
launch was handed (``OSPREY_AGENT_RECORD_URL``), authenticated with the
launch's own record login (``OSPREY_AGENT_RECORD_LOGIN``), so the record files
each event under the identity that sent it.

Only logs are exported. agent-record answers 404 on traces and metrics, because
their facts are derived from entries. The content gates are off, because the
session transcript carries the content. The export interval is one second, so a
read that drains the record sees a batch the agent sent just before it.

The block reads only the launch environment it is handed and does no I/O,
which makes it directly unit-testable (``tests/cli/test_telemetry_env.py``).
"""

from __future__ import annotations

import base64
import logging
import uuid
from collections.abc import Mapping, MutableMapping

logger = logging.getLogger(__name__)

#: The record's address (``osprey.agent_record.client.URL_ENV``). A literal so
#: this module stays importable without the client's HTTP stack; a test pins the
#: two together.
RECORD_URL_ENV = "OSPREY_AGENT_RECORD_URL"

#: The launch's record login, ``<login>:<token>``
#: (``osprey.agent_record.credentials.LOGIN_ENV``). Pinned by a test.
RECORD_LOGIN_ENV = "OSPREY_AGENT_RECORD_LOGIN"

#: The record's off switch and its one "off" value
#: (``osprey.agent_record.contract.RECORD_OFF_ENV`` / ``RECORD_OFF_VALUE``).
#: Pinned by a test.
RECORD_OFF_ENV = "OSPREY_AGENT_RECORD_OFF"
RECORD_OFF_VALUE = "1"

#: The path agent-record receives OTLP logs on
#: (``osprey.agent_record.ingest.otlp.LOGS_PATH``). Pinned by a test.
LOGS_PATH = "/v1/logs"

#: The protocol the exporter speaks; agent-record decodes protobuf and JSON.
OTLP_PROTOCOL = "http/protobuf"

#: How often, in milliseconds, the exporter sends a batch.
LOGS_EXPORT_INTERVAL_MS = "1000"

#: Claude Code's content-capture switches, every one written ``"0"``. Written,
#: never omitted: the CLI resolves an absent one through its own fallback chain
#: (``OTEL_LOG_ASSISTANT_RESPONSES`` falls back to ``OTEL_LOG_USER_PROMPTS``),
#: and an inherited shell export would otherwise turn capture back on.
CONTENT_GATES: tuple[str, ...] = (
    "OTEL_LOG_USER_PROMPTS",
    "OTEL_LOG_ASSISTANT_RESPONSES",
    "OTEL_LOG_TOOL_DETAILS",
    "OTEL_LOG_RAW_API_BODIES",
    "OTEL_LOG_TOOL_CONTENT",
)

#: The launch-scoped terminal session (``osprey.audit.envelope.TERMINAL_SESSION_ENV``).
#: Read by exact name, never by prefix.
TERMINAL_SESSION_ENV = "OSPREY_TERMINAL_SESSION"

#: The OTLP resource attribute that carries :data:`TERMINAL_SESSION_ENV`, so an
#: exported event joins the entries its launch wrote.
TERMINAL_SESSION_ATTRIBUTE = "terminal.session"

_RESOURCE_ATTRIBUTES_ENV = "OTEL_RESOURCE_ATTRIBUTES"

#: Every variable :func:`record_telemetry_env` may write. Deliberately not in
#: ``MANAGED_ENV_VARS``: they configure observability, not the provider backend
#: or model, so ``detect_env_conflicts`` exempts them and ``strip_claude_code_env``
#: keeps the ``CLAUDE_CODE_*`` ones.
TELEMETRY_ENV_VARS: frozenset[str] = frozenset(
    {
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_TRACES_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_LOGS_EXPORT_INTERVAL",
        _RESOURCE_ATTRIBUTES_ENV,
        *CONTENT_GATES,
    }
)

#: Characters a login may not carry: each would split or corrupt the
#: comma-joined ``OTEL_EXPORTER_OTLP_HEADERS`` list the exporter parses.
_HEADER_LIST_BREAKERS = frozenset(",\r\n")

#: What a launch with no record to send to exports: nothing. Every exporter is
#: written ``none`` rather than omitted, so a shell export of an exporter or an
#: endpoint cannot send the agent's events anywhere else.
_EXPORT_NOTHING: Mapping[str, str] = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_TRACES_EXPORTER": "none",
}


def _record_login(environ: Mapping[str, str]) -> str | None:
    """The launch's ``<login>:<token>``, or ``None`` when it has no usable one."""
    login = (environ.get(RECORD_LOGIN_ENV) or "").strip()
    name, sep, token = login.partition(":")
    if not sep or not name or not token or _HEADER_LIST_BREAKERS & set(login):
        return None
    return login


def record_telemetry_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The OTLP block for a Claude Code launch whose environment is *environ*.

    * With the record's off switch set, nothing: the run records nothing, so
      it carries no OTLP setting at all.
    * With the record's address and a ``<login>:<token>`` record login, logs
      to ``<address>/v1/logs`` under that login, traces and metrics off, every
      content gate off, a one-second export interval, and the launch's
      terminal session as the :data:`TERMINAL_SESSION_ATTRIBUTE` resource
      attribute (empty when the session is not known yet: a launch site that
      spawns one later fills it, see :func:`terminal_session_env`).
    * Otherwise, export nothing: there is no record this launch may send to.
      A login that is set but unusable is logged, never echoed.

    Args:
        environ: The launch environment.

    Returns:
        A ``{VAR: value}`` dict whose keys are a subset of
        :data:`TELEMETRY_ENV_VARS`.
    """
    if environ.get(RECORD_OFF_ENV) == RECORD_OFF_VALUE:
        return {}
    url = (environ.get(RECORD_URL_ENV) or "").strip().rstrip("/")
    login = _record_login(environ)
    if not url or login is None:
        if url and (environ.get(RECORD_LOGIN_ENV) or "").strip():
            logger.warning(
                "%s is not a <login>:<token> pair; this launch exports no OTLP events",
                RECORD_LOGIN_ENV,
            )
        return dict(_EXPORT_NOTHING)
    token = base64.b64encode(login.encode()).decode()
    session = (environ.get(TERMINAL_SESSION_ENV) or "").strip()
    return {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "0",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": OTLP_PROTOCOL,
        "OTEL_EXPORTER_OTLP_ENDPOINT": url,
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": f"{url}{LOGS_PATH}",
        "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Basic {token}",
        "OTEL_LOGS_EXPORT_INTERVAL": LOGS_EXPORT_INTERVAL_MS,
        _RESOURCE_ATTRIBUTES_ENV: f"{TERMINAL_SESSION_ATTRIBUTE}={session}",
        **dict.fromkeys(CONTENT_GATES, "0"),
    }


def _attribute_pairs(value: str) -> list[tuple[str, str]]:
    """``OTEL_RESOURCE_ATTRIBUTES`` as ordered ``(key, value)`` pairs."""
    pairs: list[tuple[str, str]] = []
    for pair in value.split(","):
        if not pair.strip():
            continue
        key, _sep, val = pair.partition("=")
        pairs.append((key.strip(), val.strip()))
    return pairs


def terminal_session_env(session: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """What a launch site adds to a child's environment to give it terminal session *session*.

    :data:`TERMINAL_SESSION_ENV` always — the ledger writers fall back to it
    when the child has no posture session. When the child inherits resource
    attributes that name :data:`TERMINAL_SESSION_ATTRIBUTE`, those attributes
    too, with the session filled in, so the child's OTLP events and its
    entries carry the same value.

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
