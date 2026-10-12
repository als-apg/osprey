"""The telemetry gateway's Collector configuration, rendered from the config.

The gateway is an OpenTelemetry Collector (contrib distribution, pinned by
:data:`~osprey.deployment.telemetry_config.DEFAULT_GATEWAY_IMAGE`). Every
sender posts OTLP/HTTP to it with its own send-only login; the gateway checks
the login against an htpasswd file, stamps the login's name on what it
received, and writes it to the store with the insert-only ``ingest`` account.
Nothing a sender claims about who it is survives the gateway.

This module builds that configuration as a plain mapping
(:func:`collector_config`) and serialises it deterministically
(:func:`render_collector_config`), so two renders of one config are byte-equal
and a golden can pin the result. :func:`gateway_render_context` hands the
compose template everything it cannot compute itself — the memory limit, which
depends on how many ledger receivers the gateway has, and the rendered file.

The pipelines
-------------
``logs/otlp``, ``traces`` and ``metrics`` carry what senders post:
``memory_limiter`` → ``transform/identity`` → ``filter/clamp`` → the
``clickhouse/otlp`` exporter. ``memory_limiter`` comes first so a sender is
refused (and retries) before the gateway runs out of memory.

``logs/ledger`` carries the audit ledger the gateway reads from the host. It has
**no** ``memory_limiter``: a ledger batch the limiter refused stays held by its
file reader and is retried without end, so memory never drops and the refusal
never clears. The reader's fixed batch and the disk-backed export queue bound
that pipeline instead, and the container's memory limit
(:func:`~osprey.deployment.telemetry_config.gateway_memory_floor`) is sized for
it. The pipeline exists only when the gateway is given ledger receivers.

Identity
--------
``transform/identity`` first deletes every ``osprey.*`` attribute a sender set,
at every level it can reach (resource, scope, log, span, span event,
datapoint), drops span links and exemplars (their attributes are sender claims
no statement can reach), and then sets ``osprey.user`` from the authenticated
login — on the resource and on every log, span and datapoint, because Claude
Code copies resource attributes onto each record and a reader may look at
either. A ``session.id`` or ``terminal.session`` that is not a plain token is
deleted, so a stored session column is empty rather than garbage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment.telemetry_credentials import (
    HTPASSWD_FILE,
    INGEST_PASSWORD_ENV,
    terminal_identities,
)

__all__ = [
    "ANCHOR_REGEX",
    "AUDIT_CONTAINER_ROOT",
    "ENVELOPE_REGEX",
    "TOOL_CALL_REGEX",
    "LEDGER_PATH_FILTER",
    "LEDGER_TRANSFORM",
    "QuarantinedLedger",
    "configured_ledger_identities",
    "ledger_identities",
    "ledger_include_identity",
    "ledger_receiver",
    "ledger_receiver_id",
    "ledger_receivers",
    "ledger_transform_keys",
    "non_regular_ledgers",
    "quarantine_hostile_ledgers",
    "trusted_on_behalf_of",
    "BASICAUTH",
    "COLLECTOR_CONFIG_FILE",
    "CONFIG_CONTAINER_PATH",
    "FEATURE_GATES",
    "HTPASSWD_CONTAINER_PATH",
    "LEDGER_EXPORTER",
    "LEDGER_PIPELINE",
    "OTLP_EXPORTER",
    "OTLP_PIPELINES",
    "QUEUE_BATCH_BYTES",
    "QUEUE_SIZE_BYTES",
    "STATE_CONTAINER_DIR",
    "STATE_VOLUME",
    "STORAGE",
    "collector_config",
    "gateway_render_context",
    "render_collector_config",
]

#: The rendered Collector configuration's file name in the gateway's build
#: directory, and where the container reads it.
COLLECTOR_CONFIG_FILE = "config.yaml"
CONFIG_CONTAINER_PATH = "/etc/otelcol/config.yaml"
#: Where the senders' htpasswd file is mounted, read-only.
HTPASSWD_CONTAINER_PATH = "/etc/otelcol/senders.htpasswd"
#: Where the audit ledger root is mounted, read-only.
AUDIT_CONTAINER_ROOT = "/audit"
#: The named volume holding the export queues and the file checkpoints, and
#: where it is mounted.
STATE_VOLUME = "gateway-state"
STATE_CONTAINER_DIR = "/var/lib/otelcol"

#: The Collector feature gates the gateway runs with. The synchronous emitter
#: makes a file reader wait until its batch is accepted downstream before it
#: checkpoints past it, so a stop never skips ledger lines.
FEATURE_GATES = ("stanza.synchronousLogEmitter",)

BASICAUTH = "basicauth/senders"
STORAGE = "file_storage/state"
OTLP_EXPORTER = "clickhouse/otlp"
LEDGER_EXPORTER = "clickhouse/ledger"
MEMORY_LIMITER = "memory_limiter"
IDENTITY = "transform/identity"
CLAMP = "filter/clamp"
LEDGER_TRANSFORM = "transform/ledger"
LEDGER_PATH_FILTER = "filter/ledger-path"

#: Each export queue's ceiling on disk, and the largest insert it sends.
QUEUE_SIZE_BYTES = 1024 * 1024 * 1024
QUEUE_BATCH_BYTES = 8 * 1024 * 1024

#: The pipelines senders post into, with the processors each runs in order.
OTLP_PIPELINES: Mapping[str, tuple[str, ...]] = {
    "logs/otlp": (MEMORY_LIMITER, IDENTITY, CLAMP),
    "traces": (MEMORY_LIMITER, IDENTITY, CLAMP),
    "metrics": (MEMORY_LIMITER, IDENTITY, CLAMP),
}
#: The ledger pipeline's name and its processors: never a memory limiter.
LEDGER_PIPELINE = "logs/ledger"
LEDGER_PROCESSORS: tuple[str, ...] = (LEDGER_TRANSFORM, LEDGER_PATH_FILTER, CLAMP)

#: What a stored session token may look like; anything else is deleted.
_SESSION_TOKEN_RE = "^[A-Za-z0-9._-]{1,128}$"
_SESSION_KEYS = ("session.id", "terminal.session")
_USER = 'otelcol.client.auth.attributes["username"]'
_RESERVED = '"^osprey\\\\."'
_STAMP = '"osprey.user"'

#: How far ahead of the gateway's clock a record may claim to be.
_FUTURE_SKEW = 'Duration("1h")'


def _session_guards(context: str) -> list[str]:
    """Delete a session attribute in *context* that is not a plain token."""
    return [
        f'delete_key({context}.attributes, "{key}") where {context}.attributes["{key}"] != nil '
        f'and (not IsString({context}.attributes["{key}"]) or '
        f'not IsMatch({context}.attributes["{key}"], "{_SESSION_TOKEN_RE}"))'
        for key in _SESSION_KEYS
    ]


def _scrub(context: str) -> str:
    return f"delete_matching_keys({context}.attributes, {_RESERVED})"


def _stamp(context: str) -> str:
    return f"set({context}.attributes[{_STAMP}], {_USER})"


def _resource_and_scope() -> list[dict[str, Any]]:
    return [
        {
            "context": "resource",
            "statements": [_scrub("resource"), _stamp("resource"), *_session_guards("resource")],
        },
        {"context": "scope", "statements": [_scrub("scope")]},
    ]


def _identity_processor() -> dict[str, Any]:
    """``transform/identity``: delete sender claims, stamp the login, guard sessions."""
    return {
        "error_mode": "ignore",
        "log_statements": [
            *_resource_and_scope(),
            {
                "context": "log",
                "statements": [
                    _scrub("log"),
                    _stamp("log"),
                    *_session_guards("log"),
                    'set(log.attributes["event.name"], log.event_name) '
                    'where log.event_name != "" and log.attributes["event.name"] == nil',
                    "set(log.time, log.observed_time) where log.time_unix_nano == 0",
                ],
            },
        ],
        "trace_statements": [
            *_resource_and_scope(),
            {
                "context": "span",
                "statements": [
                    _scrub("span"),
                    "set(span.links, nil)",
                    _stamp("span"),
                    *_session_guards("span"),
                ],
            },
            {"context": "spanevent", "statements": [_scrub("spanevent")]},
        ],
        "metric_statements": [
            *_resource_and_scope(),
            {
                "context": "datapoint",
                "statements": [
                    _scrub("datapoint"),
                    "set(datapoint.exemplars, nil)",
                    _stamp("datapoint"),
                    *_session_guards("datapoint"),
                ],
            },
        ],
    }


def _clamp_processor(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """``filter/clamp``: drop what claims a time outside the store's retention."""
    past = f'Duration("{tc.retention_days(config) * 24}h")'

    def outside(field: str) -> str:
        return f"{field} < Now() - {past} or {field} > Now() + {_FUTURE_SKEW}"

    return {
        "error_mode": "ignore",
        "logs": {"log_record": [outside("log.time")]},
        "traces": {"span": [outside("span.start_time")]},
        "metrics": {"datapoint": [outside("datapoint.time")]},
    }


def _exporter(config: Mapping[str, Any] | None, logs_table: str, *, block: bool) -> dict[str, Any]:
    """A ClickHouse exporter writing as ``ingest`` through a disk-backed queue."""
    queue: dict[str, Any] = {
        "enabled": True,
        "storage": STORAGE,
        "sizer": "bytes",
        "queue_size": QUEUE_SIZE_BYTES,
    }
    if block:
        queue["block_on_overflow"] = True
    queue["batch"] = {"sizer": "bytes", "max_size": QUEUE_BATCH_BYTES}
    return {
        "endpoint": tc.store_url(config, bridge=True),
        "username": "ingest",
        "password": f"${{env:{INGEST_PASSWORD_ENV}}}",
        "database": "telemetry",
        "async_insert": False,
        "create_schema": False,
        "ttl": 0,
        "logs_table_name": logs_table,
        "timeout": "30s",
        "sending_queue": queue,
        "retry_on_failure": {"max_elapsed_time": 0},
    }


def _storage_extension() -> dict[str, Any]:
    return {
        "directory": f"{STATE_CONTAINER_DIR}/state",
        "create_directory": True,
        "fsync": True,
        "recreate": True,
        "compaction": {
            "on_start": True,
            "on_rebound": True,
            "directory": f"{STATE_CONTAINER_DIR}/compact",
            "cleanup_on_start": True,
            "rebound_needed_threshold_mib": 100,
            "rebound_trigger_threshold_mib": 10,
        },
    }


# --- the ledger feed -----------------------------------------------------------

logger = logging.getLogger(__name__)

#: The names an identity directory found on disk must have to get a receiver:
#: the roster's charset. A directory outside it is skipped with a warning; one
#: named by the config always gets its receiver, whatever its characters.
LEDGER_DIRECTORY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
#: The ledger files a receiver reads in its identity's directory.
LEDGER_GLOB = "*.jsonl"
#: How long a reader holds a partial line before emitting it as it stands.
LEDGER_FORCE_FLUSH = "60s"
#: The suffix a non-regular ledger path is renamed with, before its UTC stamp.
REFUSED_SUFFIX = ".refused-"
#: The identity whose ``on_behalf_of`` is trusted on every surface.
_SIDECAR = "sidecar"
#: The one surface a queueserver's ``on_behalf_of`` is trusted on.
_PREFLIGHT = "preflight"
_TOOL_CALL = "tool_call"
_UNPARSED = "unparsed"

#: One JSON string's contents, escapes included; never crosses its closing quote.
_STR = r'(?:[^"\\]|\\.)*'
#: The two fields every ledger line opens with, in this order, both writers.
ANCHOR_REGEX = r'^\{"ts":"(?P<ledger_ts>[^"]+)","surface":"(?P<ledger_surface>[^"\\]{1,256})"'
_HEAD = r'^\{"ts":"[^"]+","surface":"[^"\\]{1,256}"'
_POSTURE = f',"posture":"{_STR}","posture_source":"{_STR}"'

#: A ``tool_call`` record (:func:`osprey.audit.tool_call.build_record`): every
#: field the store keeps precedes the payloads, so a truncated line still has
#: them. Unanchored at the end.
TOOL_CALL_REGEX = (
    _HEAD
    + f',"actor":"(?P<ledger_actor>{_STR})"'
    + _POSTURE
    + f',"session":(?:null|"(?P<ledger_session>{_STR})")'
    + f',"session_id":(?:null|"(?P<ledger_session_id>{_STR})")'
    + f',"tool_use_id":(?:null|"(?P<ledger_tool_use_id>{_STR})")'
    + r',"duration_ms":(?:null|"?(?P<ledger_duration_ms>-?Infinity|NaN|-?[0-9][0-9.eE+-]*)"?)'
    + f',"server":"{_STR}","subject":"{_STR}"'
    + f',"decision":"(?P<ledger_decision>{_STR})"'
)

#: Every other surface (:meth:`osprey.audit.envelope.AuditEnvelope.to_dict` and
#: the hook twin): positional, and unanchored at the end, because an executor
#: record carries its ``source`` after ``tool_use_id`` and must still match.
ENVELOPE_REGEX = (
    _HEAD
    + f',"actor":"(?P<ledger_actor>{_STR})"'
    + _POSTURE
    + f',"session":(?:null|"(?P<ledger_session>{_STR})")'
    + f'(?:,"on_behalf_of":"(?P<ledger_on_behalf_of>{_STR})")?'
    + f',"subject":"{_STR}"'
    + f',"decision":"(?P<ledger_decision>{_STR})"'
    + f',"reason":"{_STR}"'
    + f'(?:,"detail":"{_STR}")?'
    + f'(?:,"role":"{_STR}")?'
    + r'(?:,"tool_use_id":"(?P<ledger_tool_use_id>[A-Za-z0-9_-]{1,128})")?'
)

#: Both ``ts`` layouts the writers have used (``…Z`` and ``…+00:00``).
LEDGER_TIME_LAYOUT = "2006-01-02T15:04:05.999999999Z07:00"

#: What ``transform/ledger`` copies from the parsed line, by the ``ledger_*``
#: attribute the operators write it to: the ``osprey.audit.*`` key it lands in.
_CAPTURED = (
    "session_id",
    "session",
    "tool_use_id",
    "duration_ms",
    "actor",
    "decision",
    "surface",
    "on_behalf_of",
)
#: The markers ``transform/ledger`` sets on a line it could not read whole.
_MARKERS = ("incomplete", "truncated", "unparsed", "time_unparsed")


def ledger_transform_keys() -> tuple[str, ...]:
    """Every ``osprey.audit.*`` key ``transform/ledger`` can set on a row."""
    return tuple(f"osprey.audit.{name}" for name in (*_CAPTURED, *_MARKERS))


def ledger_receiver_id(identity: str) -> str:
    """The receiver's component id: a hash, since a name may hold any character.

    Depends on the name alone, so a directory keeps its id — and the checkpoint
    stored under it — for its whole life, whatever else is configured.
    """
    return f"file_log/d{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


_GLOB_SPECIAL = re.compile(r"([\\*?\[\]{}!])")
_GLOB_UNESCAPE = re.compile(r"\[(.)\]|\\(.)")


def _glob_escape(name: str) -> str:
    return _GLOB_SPECIAL.sub(r"\\\1", name)


def ledger_include_identity(pattern: str) -> str | None:
    """The identity directory a ledger receiver's include *pattern* reads.

    The inverse of the pattern :func:`ledger_receiver` writes; ``None`` for a
    pattern of any other shape.
    """
    prefix = f"{AUDIT_CONTAINER_ROOT}/"
    suffix = f"/{LEDGER_GLOB}"
    if not (pattern.startswith(prefix) and pattern.endswith(suffix)):
        return None
    middle = pattern[len(prefix) : -len(suffix)]
    return _GLOB_UNESCAPE.sub(lambda m: m.group(1) or m.group(2), middle)


def _ledger_operators(identity: str) -> list[dict[str, Any]]:
    """Parse the fields the store keeps, by position; never the whole line."""
    return [
        {
            "id": "ledger_anchor",
            "type": "regex_parser",
            "regex": ANCHOR_REGEX,
            "on_error": "send_quiet",
        },
        {
            "id": "ledger_route",
            "type": "router",
            "routes": [
                {
                    "expr": f'attributes.ledger_surface == "{_TOOL_CALL}"',
                    "output": "ledger_tool_call",
                }
            ],
            "default": "ledger_envelope",
        },
        {
            "id": "ledger_tool_call",
            "type": "regex_parser",
            "regex": TOOL_CALL_REGEX,
            "on_error": "send_quiet",
            "output": "ledger_time",
        },
        {
            "id": "ledger_envelope",
            "type": "regex_parser",
            "regex": ENVELOPE_REGEX,
            "on_error": "send_quiet",
            "output": "ledger_time",
        },
        {
            "id": "ledger_time",
            "type": "time_parser",
            "parse_from": "attributes.ledger_ts",
            "layout_type": "gotime",
            "layout": LEDGER_TIME_LAYOUT,
            "on_error": "send_quiet",
        },
        {
            "id": "ledger_identity",
            "type": "add",
            "field": 'resource["osprey.user"]',
            # An expression holding one string literal, so a name that itself
            # looks like an expression is stored as written.
            "value": f"EXPR({json.dumps(identity)})",
        },
    ]


def ledger_receiver(config: Mapping[str, Any] | None, identity: str) -> dict[str, Any]:
    """The ``file_log`` receiver reading one identity directory's ledgers."""
    return {
        "include": [f"{AUDIT_CONTAINER_ROOT}/{_glob_escape(identity)}/{LEDGER_GLOB}"],
        "include_file_path": True,
        "include_file_path_resolved": True,
        "start_at": "beginning",
        "storage": STORAGE,
        "max_concurrent_files": 2,
        "max_log_size": tc.gateway_max_log_size(config),
        "max_log_size_behavior": "truncate",
        "on_truncate": "read_whole_file",
        "force_flush_period": LEDGER_FORCE_FLUSH,
        "retry_on_failure": {
            "enabled": True,
            "initial_interval": "1s",
            "max_interval": "30s",
            "max_elapsed_time": 0,
        },
        "operators": _ledger_operators(identity),
    }


def configured_ledger_identities(config: Mapping[str, Any] | None) -> list[str]:
    """Every identity directory this deployment's config names a writer for.

    Roster terminals, the auth sidecar, dispatch workers, the fixed recording
    services, the lane queueservers, the history page and
    ``deployment.host_identity``. Empty when the gateway is not deployed.
    """
    if not tc.gateway_deployed(config):
        return []
    from osprey.deployment.compose_generator import (
        dispatch_worker_audit_identities,
        lane_queueserver_audit_identities,
        service_audit_identities,
    )

    conf = dict(config or {})
    identities: list[str] = []
    web_terminals = (conf.get("modules") or {}).get("web_terminals") or {}
    if isinstance(web_terminals, Mapping) and web_terminals.get("enabled"):
        from osprey.deployment.web_terminals.render import (
            AUTH_SIDECAR_AUDIT_IDENTITY,
            _auth_tls_context,
        )

        identities.extend(str(name) for name in terminal_identities(conf))
        if _auth_tls_context(dict(web_terminals))["sidecar_active"]:
            identities.append(AUTH_SIDECAR_AUDIT_IDENTITY)
    identities.extend(dispatch_worker_audit_identities(conf))
    identities.extend(service_audit_identities(conf))
    identities.extend(lane_queueserver_audit_identities(conf))
    if tc.history_page_deployed(conf):
        identities.append(tc.HISTORY_SERVICE)
    host = tc.host_identity(conf)
    if host is not None:
        identities.append(host)
    return list(dict.fromkeys(identities))


def _existing_directories(audit_root: Path | None) -> list[str]:
    if audit_root is None:
        return []
    names: list[str] = []
    try:
        with os.scandir(audit_root) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    names.append(entry.name)
    except OSError:
        return []
    return names


def ledger_identities(config: Mapping[str, Any] | None, audit_root: Path | None) -> list[str]:
    """The identities the gateway reads: configured ∪ already on disk, sorted.

    A directory that exists keeps its receiver after its user leaves the
    roster, so its checkpoint survives and nothing is read twice. A directory
    on disk outside :data:`LEDGER_DIRECTORY_RE` is skipped with a warning.
    Empty when the gateway is not deployed.
    """
    if not tc.gateway_deployed(config):
        return []
    configured = configured_ledger_identities(config)
    found: list[str] = []
    for name in _existing_directories(audit_root):
        if name in configured:
            continue
        if not LEDGER_DIRECTORY_RE.fullmatch(name):
            logger.warning(
                "audit directory %r is not fed: its name is outside %s",
                name,
                LEDGER_DIRECTORY_RE.pattern,
            )
            continue
        found.append(name)
    return sorted({*configured, *found})


def ledger_receivers(
    config: Mapping[str, Any] | None,
    audit_root: Path | None,
    identities: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """The ledger's ``file_log`` receivers by component id, one per identity.

    Args:
        config: The rendered deployment config.
        audit_root: The host's ``var/audit``; its existing directories join the set.
        identities: The identity set to render, when the caller already has it.
    """
    names = ledger_identities(config, audit_root) if identities is None else identities
    return {ledger_receiver_id(name): ledger_receiver(config, name) for name in sorted(names)}


def trusted_on_behalf_of(config: Mapping[str, Any] | None) -> tuple[str, tuple[str, ...]]:
    """Whose ``on_behalf_of`` the store keeps: the sidecar, and the queueservers.

    Returns:
        The sidecar's identity (trusted on every surface) and the rendered
        queueserver identities (trusted on ``preflight`` rows only), exact names.
    """
    from osprey.deployment.compose_generator import lane_queueserver_audit_identities

    return _SIDECAR, tuple(lane_queueserver_audit_identities(dict(config or {})))


def _ledger_transform(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """``transform/ledger``: the parsed fields become ``osprey.audit.*``, nothing else."""
    max_log_size = tc.gateway_max_log_size(config)
    parsed = 'log.attributes["ledger_surface"] != nil'
    user = 'resource.attributes["osprey.user"]'
    sidecar, queueservers = trusted_on_behalf_of(config)
    trusted = f'{user} == "{sidecar}"'
    if queueservers:
        names = " or ".join(f'{user} == "{name}"' for name in queueservers)
        trusted = f'{trusted} or (({names}) and log.attributes["ledger_surface"] == "{_PREFLIGHT}")'
    statements = [
        'delete_matching_keys(log.attributes, "^osprey\\\\.")',
        *(
            f'set(log.attributes["osprey.audit.{name}"], log.attributes["ledger_{name}"]) '
            f'where log.attributes["ledger_{name}"] != nil and log.attributes["ledger_{name}"] != ""'
            for name in _CAPTURED
        ),
        f'set(log.attributes["osprey.audit.surface"], "{_UNPARSED}") where not ({parsed})',
        f'set(log.attributes["osprey.audit.unparsed"], true) where not ({parsed})',
        f'set(log.attributes["osprey.audit.truncated"], true) where Len(log.body) >= {max_log_size}',
        f'set(log.attributes["osprey.audit.incomplete"], true) where {parsed} '
        f'and not IsMatch(log.body, "\\\\}}$") and Len(log.body) < {max_log_size}',
        f'set(log.attributes["osprey.audit.time_unparsed"], true) '
        f"where {parsed} and log.time_unix_nano == 0",
        "set(log.time, log.observed_time) where log.time_unix_nano == 0",
        f'set(log.attributes["osprey.user"], {user})',
        f'delete_key(log.attributes, "osprey.audit.on_behalf_of") where not ({trusted})',
        'delete_matching_keys(log.attributes, "^ledger_")',
    ]
    return {
        "error_mode": "ignore",
        "log_statements": [{"context": "log", "statements": statements}],
    }


#: ``filter/ledger-path``: a line read through a symlink anywhere on its path.
LEDGER_PATH_CONDITION = (
    'log.attributes["log.file.path_resolved"] != log.attributes["log.file.path"] '
    'or log.attributes["log.file.path_resolved"] == nil'
)


def _ledger_path_filter() -> dict[str, Any]:
    return {"error_mode": "ignore", "logs": {"log_record": [LEDGER_PATH_CONDITION]}}


class QuarantinedLedger:
    """One non-regular ledger path ``osprey up`` moved out of the readers' glob."""

    __slots__ = ("path", "moved_to")

    def __init__(self, path: Path, moved_to: Path) -> None:
        self.path = path
        self.moved_to = moved_to

    def __repr__(self) -> str:
        return f"QuarantinedLedger({str(self.path)!r} -> {str(self.moved_to)!r})"


def _release_fifo(path: str) -> None:
    """Open a FIFO for writing and close it, so a reader blocked in ``open()`` sees EOF."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


def non_regular_ledgers(audit_root: Path) -> list[Path]:
    """Every ``*.jsonl`` one level under an identity directory that is not a regular file.

    Walked with :func:`os.scandir`, so dotfiles are seen and no symlink is followed.
    """
    found: list[Path] = []
    for identity in _existing_directories(audit_root):
        with contextlib.suppress(OSError), os.scandir(Path(audit_root) / identity) as entries:
            for entry in entries:
                if entry.name.endswith(".jsonl") and not entry.is_file(follow_symlinks=False):
                    found.append(Path(entry.path))
    return sorted(found)


def quarantine_hostile_ledgers(
    audit_root: Path, *, now: datetime | None = None
) -> list[QuarantinedLedger]:
    """Move every ``*.jsonl`` that is not a regular file out of the readers' glob.

    Finds them with :func:`non_regular_ledgers` (dotfiles included,
    no symlink followed). A FIFO is first opened for writing and closed, which
    releases a gateway blocked opening it; then the entry is renamed
    ``<name>.refused-<UTC ts>`` and kept as evidence.

    Raises:
        RuntimeError: When an entry cannot be renamed, naming its path.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    moved: list[QuarantinedLedger] = []
    for path in non_regular_ledgers(audit_root):
        if not path.is_symlink() and not path.is_dir():
            _release_fifo(str(path))
        target = path.with_name(f"{path.name}{REFUSED_SUFFIX}{stamp}")
        try:
            os.rename(path, target)
        except OSError as exc:
            raise RuntimeError(
                f"{path} is not a regular file and could not be moved aside "
                f"({exc.strerror or exc}); remove it, then run `osprey up` again"
            ) from exc
        moved.append(QuarantinedLedger(path, target))
    return moved


def collector_config(
    config: Mapping[str, Any] | None,
    *,
    ledger_receivers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The gateway's Collector configuration as a mapping.

    Args:
        config: The rendered deployment config.
        ledger_receivers: The ledger's file receivers, by component id. When
            there are any, the ``clickhouse/ledger`` exporter and the
            ``logs/ledger`` pipeline are rendered to carry them.
    """
    ledger = dict(ledger_receivers or {})
    receivers: dict[str, Any] = {
        "otlp": {
            "protocols": {
                "http": {
                    "endpoint": f"0.0.0.0:{tc.GATEWAY_OTLP_CONTAINER_PORT}",
                    "max_request_body_size": tc.gateway_max_request_body_size(config),
                    "compression_algorithms": ["", "gzip"],
                    "include_metadata": False,
                    "auth": {"authenticator": BASICAUTH},
                }
            }
        },
        **ledger,
    }
    exporters: dict[str, Any] = {OTLP_EXPORTER: _exporter(config, "otel_logs", block=False)}
    pipelines: dict[str, Any] = {
        name: {"receivers": ["otlp"], "processors": list(processors), "exporters": [OTLP_EXPORTER]}
        for name, processors in OTLP_PIPELINES.items()
    }
    processors: dict[str, Any] = {
        MEMORY_LIMITER: {
            "check_interval": "1s",
            "limit_percentage": 80,
            "spike_limit_percentage": 20,
        },
        IDENTITY: _identity_processor(),
        CLAMP: _clamp_processor(config),
    }
    if ledger:
        processors[LEDGER_TRANSFORM] = _ledger_transform(config)
        processors[LEDGER_PATH_FILTER] = _ledger_path_filter()
        exporters[LEDGER_EXPORTER] = _exporter(config, "audit_ledger", block=True)
        pipelines[LEDGER_PIPELINE] = {
            "receivers": list(ledger),
            "processors": list(LEDGER_PROCESSORS),
            "exporters": [LEDGER_EXPORTER],
        }
    return {
        "extensions": {
            BASICAUTH: {"htpasswd": {"file": HTPASSWD_CONTAINER_PATH}},
            STORAGE: _storage_extension(),
        },
        "receivers": receivers,
        "processors": processors,
        "exporters": exporters,
        "service": {
            "telemetry": {
                "metrics": {
                    "readers": [
                        {
                            "pull": {
                                "exporter": {
                                    "prometheus": {
                                        "host": "0.0.0.0",
                                        "port": tc.GATEWAY_METRICS_CONTAINER_PORT,
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "extensions": [BASICAUTH, STORAGE],
            "pipelines": pipelines,
        },
    }


def render_collector_config(
    config: Mapping[str, Any] | None,
    *,
    ledger_receivers: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """:func:`collector_config` as the YAML file the gateway reads.

    Deterministic: keys keep their construction order and no line is folded,
    so equal inputs give byte-equal files.
    """
    return yaml.safe_dump(
        collector_config(config, ledger_receivers=ledger_receivers),
        sort_keys=False,
        default_flow_style=False,
        width=1 << 16,
    )


def gateway_render_context(
    config: Mapping[str, Any] | None,
    *,
    ledger_receivers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """What the gateway's compose template and config file render from.

    Returns:
        ``receivers`` (R, the number of ledger receivers), ``mem_limit`` (bytes,
        :func:`~osprey.deployment.telemetry_config.gateway_memory_bytes` for R),
        ``image``, ``command`` (the Collector's arguments), the published
        ``port`` and ``metrics_port`` with the container ports they map to,
        the source and target of each mount, and ``collector_config`` (the
        rendered file's text).
    """
    receivers = len(ledger_receivers or {})
    service = tc.GATEWAY_SERVICE
    return {
        "receivers": receivers,
        "mem_limit": tc.gateway_memory_bytes(config, receivers),
        "image": _gateway_image(config),
        "port": tc.gateway_port(config),
        "metrics_port": tc.metrics_port(config),
        "command": [
            f"--config={CONFIG_CONTAINER_PATH}",
            f"--feature-gates={','.join(FEATURE_GATES)}",
        ],
        "otlp_container_port": tc.GATEWAY_OTLP_CONTAINER_PORT,
        "metrics_container_port": tc.GATEWAY_METRICS_CONTAINER_PORT,
        "config_source": f"./build/services/{service}/{COLLECTOR_CONFIG_FILE}",
        "config_target": CONFIG_CONTAINER_PATH,
        "htpasswd_source": f"./{HTPASSWD_FILE}",
        "htpasswd_target": HTPASSWD_CONTAINER_PATH,
        "audit_target": AUDIT_CONTAINER_ROOT,
        "state_volume": STATE_VOLUME,
        "state_target": STATE_CONTAINER_DIR,
        "collector_config": render_collector_config(config, ledger_receivers=ledger_receivers),
    }


def _gateway_image(config: Mapping[str, Any] | None) -> str:
    """``services.telemetry_gateway.image``, or the pinned default."""
    value = tc.service_block(config, tc.GATEWAY_SERVICE).get("image")
    return value.strip() if isinstance(value, str) and value.strip() else tc.DEFAULT_GATEWAY_IMAGE
