"""The ``agent_record:`` block — where the deployment's one agent record lives.

Every deployment keeps an agent record, so the block is never required: a
profile that does not spell it gets :class:`AgentRecordConfig`'s defaults,
which are the embedded backend (the record kept by the agent-record service
itself, with no store container), two weeks of retention and no named
readers. Spelling the block changes those facts for the deployment as a
whole, which is why it is a root-profile key that a tier may not spell.

Shapes, parsing, rules and the config the block contributes are all here, on
the pattern of the other per-block modules:

* :func:`parse_agent_record_block` turns the raw block into its typed shape
  and refuses structural problems (an unknown key, a wrong type) all at once;
* :func:`agent_record_errors` holds the range and value rules, reported from
  ``BuildProfile.validate()`` beside the rest of the profile's problems;
* :func:`reader_errors` and :func:`reader_warnings` judge the named readers
  against the roster and the sign-in method, which only the merged lint view
  knows — ``deploy_aware_config_errors`` asks them;
* :func:`agent_record_config_overrides` is the resolved block as dotted keys,
  written into the rendered ``config.yml`` under the top-level key
  ``agent_record`` so the service and the build read one config source.
* :func:`refuse_retired_config_keys` refuses a ``config:`` block that still
  spells a key the record replaced (:data:`RETIRED_CONFIG_KEYS`), naming what
  to write instead.

Beside the block live the names the record's deployment shares with the
history page and the compose render: the history service's name, the two
history ports, the ClickHouse container's port and memory ratio, and
:func:`parse_memory`, which reads every memory size the block states.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from osprey.agent_record.contract import KINDS
from osprey.port_layout import default_port, resolve_port_base
from osprey_connectors.errors import BuildProfileError

#: The profile key, and the top-level key of the rendered ``config.yml``.
AGENT_RECORD_BLOCK = "agent_record"

#: The record kept by the agent-record service in its own file: no store
#: container, which is what a hello-world deployment runs.
BACKEND_EMBEDDED = "embedded"

#: The record kept in a ClickHouse store the deployment runs beside it.
BACKEND_CLICKHOUSE = "clickhouse"

#: Every backend the record can be kept in, the default first.
BACKENDS: tuple[str, ...] = (BACKEND_EMBEDDED, BACKEND_CLICKHOUSE)

#: Days the record keeps an entry, counted from the day it arrived.
DEFAULT_RETENTION_DAYS = 14

#: The agent-record container's memory limit.
DEFAULT_MEMORY = "1g"

#: The ClickHouse server image a ``backend: clickhouse`` deployment runs.
DEFAULT_CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server:26.8.18.2"

#: The ClickHouse container's memory limit.
DEFAULT_CLICKHOUSE_MEMORY = "2g"

#: The history page's compose service, its audit identity and its template
#: directory. Derived (see :func:`history_page_deployed`), never a
#: ``deployed_services`` entry.
HISTORY_SERVICE = "history"

#: The port the ClickHouse image listens on for HTTP. A property of the image,
#: not a host port, so ``deployment.port_base`` does not move it.
STORE_HTTP_CONTAINER_PORT = 8123

#: The share of its memory limit the ClickHouse server lets itself use
#: (``max_server_memory_usage_to_ram_ratio`` in its server config).
STORE_MEMORY_RATIO = 0.9

_KIB = 1024
_MIB = 1024 * _KIB
_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([bkmg]?)(?:i?b)?\s*$", re.IGNORECASE)
_MEMORY_UNITS = {"": 1, "b": 1, "k": _KIB, "m": _MIB, "g": 1024 * _MIB}

#: The keys one ``readers`` entry may spell.
READER_KEYS: tuple[str, ...] = ("name", "kinds", "days_back", "oidc_subject")

#: What a reader may be called: the roster's charset, because a reader's name
#: is part of its login and of the credential file minted for it.
READER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_READERS = f"{AGENT_RECORD_BLOCK}.readers"
_WEB = "modules.web_terminals"


@dataclass(frozen=True)
class ClickHouseConfig:
    """The ``agent_record.clickhouse:`` sub-block — read only with ``backend: clickhouse``."""

    image: str = DEFAULT_CLICKHOUSE_IMAGE
    """The ClickHouse server image the store container runs."""

    memory: str | int = DEFAULT_CLICKHOUSE_MEMORY
    """The store container's memory limit, as compose's ``mem_limit`` takes it."""


@dataclass(frozen=True)
class AgentRecordConfig:
    """The parsed ``agent_record:`` block, defaults filled in."""

    backend: str = BACKEND_EMBEDDED
    """Where the record is kept (see :data:`BACKENDS`)."""

    retention_days: int = DEFAULT_RETENTION_DAYS
    """Days an entry is kept after it arrived. A reader's ``days_back`` is at
    most this, because the record holds nothing older."""

    readers: tuple[Any, ...] = ()
    """The named readers, as the profile states them: mappings of
    :data:`READER_KEYS`. Kept as stated so :func:`reader_errors` can name the
    entry it refuses; a ``kinds`` left out means every kind, a ``days_back``
    left out means the whole retention window."""

    port: int = default_port("agent_record")
    """The service's port, at the layout's ``agent_record`` slot — that slot at
    the layout's own base here, and at the profile's resolved base once
    :func:`parse_agent_record_block` fills an unspelled key."""

    memory: str | int = DEFAULT_MEMORY
    """The agent-record container's memory limit."""

    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    """The ClickHouse store's knobs (``agent_record.clickhouse``)."""


#: Keys the block recognizes, derived from the dataclasses rather than listed
#: by hand so the check cannot drift from what the parser honors.
_KNOWN_AGENT_RECORD_KEYS: frozenset[str] = frozenset(f.name for f in fields(AgentRecordConfig))
_KNOWN_CLICKHOUSE_KEYS: frozenset[str] = frozenset(f.name for f in fields(ClickHouseConfig))


def uses_clickhouse(raw: Mapping[str, Any] | None) -> bool:
    """Whether *raw*'s ``agent_record:`` block names the ClickHouse backend.

    Read from the raw block, so a block the build refuses never raises here.
    """
    block = (raw or {}).get(AGENT_RECORD_BLOCK)
    return isinstance(block, Mapping) and block.get("backend") == BACKEND_CLICKHOUSE


#: The web-terminal sign-in methods that scope a history page by person.
HISTORY_AUTH_METHODS: tuple[str, ...] = ("password", "oidc")


def history_page_deployed(config: Mapping[str, Any] | None) -> bool:
    """Whether this deployment serves the across-users history page.

    The page reads the agent record, which every deployment keeps, and needs a
    sign-in to scope by: it exists exactly when the web terminals are enabled
    and authenticate people (``auth.method`` of ``password`` or ``oidc``). A
    deployment without that sign-in, hello-world among them, has no page.
    """
    modules = _mapping(_mapping(config).get("modules"))
    web_terminals = _mapping(modules.get("web_terminals"))
    if not web_terminals.get("enabled"):
        return False
    return _mapping(web_terminals.get("auth")).get("method") in HISTORY_AUTH_METHODS


def history_origin_port(config: Mapping[str, Any] | None) -> int:
    """The port of the history page's origin (``modules.web_terminals.history_port``).

    Raises:
        ValueError: If the key is set to something that is not a port.
    """
    value = _mapping(_mapping(_mapping(config).get("modules")).get("web_terminals")).get(
        "history_port"
    )
    if value is None:
        return default_port("history_origin", base=resolve_port_base(config))
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < 65536:
        raise ValueError(f"{_WEB}.history_port must be an integer port, got {value!r}")
    return value


def history_app_port(config: Mapping[str, Any] | None) -> int:
    """The loopback port the history app is published on."""
    return default_port(HISTORY_SERVICE, base=resolve_port_base(config))


def parse_memory(value: Any, key: str) -> int:
    """Return a compose memory value in bytes.

    Args:
        value: An integer byte count, or a string as compose's ``mem_limit``
            takes it — ``2g``, ``512m``, ``1.5gb``, ``2GiB``. Units are binary.
        key: The dotted config key the value came from, for the refusal.

    Raises:
        ValueError: If the value is not a positive memory size.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = _MEMORY_RE.match(value) if isinstance(value, str) else None
    size = float(match.group(1)) * _MEMORY_UNITS[match.group(2).lower()] if match else 0
    if size < 1:
        raise ValueError(
            f"{key} must be a memory size such as 2g or 512m (or a byte count), got {value!r}"
        )
    return int(size)


def parse_agent_record_block(
    raw: Mapping[str, Any], *, base: int | None = None
) -> AgentRecordConfig:
    """Parse a profile's ``agent_record:`` block into its typed shape.

    Args:
        raw: The resolved raw profile dict.
        base: The base the deployment resolved from ``deployment.port_base``.
            An unspelled ``port`` is filled from it, so the service lands in
            this deployment's own port block. ``None`` keeps the dataclass
            default, the slot at the layout's default base.

    Returns:
        The parsed config. A profile with no block gets the defaults: every
        deployment keeps a record, and the embedded backend needs nothing
        stated.

    Raises:
        BuildProfileError: If the block is not a mapping, names a key it does
            not define, or gives one the wrong type — every such problem in one
            message. Range and value rules are :func:`agent_record_errors`'.
    """
    block = raw.get(AGENT_RECORD_BLOCK)
    port_default: dict[str, Any] = (
        {} if base is None else {"port": default_port("agent_record", base=base)}
    )
    if block is None:
        return AgentRecordConfig(**port_default)
    if not isinstance(block, Mapping):
        raise BuildProfileError(
            f"Profile 'agent_record' must be a mapping (got {type(block).__name__})"
        )

    problems: list[str] = []
    _reject_unknown_keys(block, _KNOWN_AGENT_RECORD_KEYS, AGENT_RECORD_BLOCK, problems)

    values: dict[str, Any] = dict(port_default)
    if "backend" in block:
        values["backend"] = _parse_str(block["backend"], "backend", problems)
    for key in ("retention_days", "port"):
        if key in block:
            values[key] = _parse_int(block[key], key, problems)
    if "memory" in block:
        values["memory"] = _parse_memory_value(block["memory"], "memory", problems)
    if block.get("readers") is not None:
        stated = block["readers"]
        if isinstance(stated, list):
            values["readers"] = tuple(
                dict(entry) if isinstance(entry, Mapping) else entry for entry in stated
            )
        else:
            problems.append(
                f"'readers' must be a list of reader entries (`- name: <who>`), got {stated!r}."
            )
    if block.get("clickhouse") is not None:
        values["clickhouse"] = _parse_clickhouse(block["clickhouse"], problems)

    if problems:
        raise BuildProfileError(
            "Profile 'agent_record' block is invalid:\n  - " + "\n  - ".join(problems)
        )
    return AgentRecordConfig(**values)


def agent_record_errors(agent_record: AgentRecordConfig, config: Any = None) -> list[str]:
    """Check the block's values, and that ``config:`` does not state them again.

    Args:
        agent_record: The parsed block.
        config: The profile's ``config:`` block, or ``None``.

    Returns:
        The problems found, each naming its fix; empty when the block is clean.
    """
    cfg = agent_record
    errors: list[str] = []
    if cfg.backend not in BACKENDS:
        errors.append(
            f"agent_record.backend must be one of "
            f"{', '.join(repr(name) for name in BACKENDS)} (got {cfg.backend!r}). Remove "
            f"the key to keep the record in the agent-record service itself, with no "
            f"store container, or set `backend: {BACKEND_CLICKHOUSE}` to keep it in a "
            f"ClickHouse store the deployment runs"
        )
    if cfg.retention_days < 1:
        errors.append(
            f"agent_record.retention_days must be a whole number of days, 1 or more "
            f"(got {cfg.retention_days}). Remove the key to keep "
            f"{DEFAULT_RETENTION_DAYS} days"
        )
    if not (1 <= cfg.port <= 65535):
        errors.append(
            f"agent_record.port must be in 1..65535 (got {cfg.port}). Remove the key to "
            f"take the agent_record slot of this deployment's port block"
        )
    for key, value in (
        ("agent_record.memory", cfg.memory),
        ("agent_record.clickhouse.memory", cfg.clickhouse.memory),
    ):
        try:
            parse_memory(value, key)
        except ValueError as exc:
            errors.append(str(exc))
    if not cfg.clickhouse.image.strip():
        errors.append(
            f"agent_record.clickhouse.image must be a non-empty image reference. Remove "
            f"the key to run {DEFAULT_CLICKHOUSE_IMAGE}"
        )
    errors.extend(_duplicate_config_key_errors(config))
    return errors


def reader_errors(agent_record: AgentRecordConfig, config: Mapping[str, Any] | None) -> list[str]:
    """The named-reader problems that must fail a command.

    Args:
        agent_record: The parsed block.
        config: The merged lint view as nested mappings — the roster
            (``modules.web_terminals.users``) and the sign-in method are read
            from it.
    """
    return [message for severity, message in _reader_findings(agent_record, config) if severity]


def reader_warnings(agent_record: AgentRecordConfig, config: Mapping[str, Any] | None) -> list[str]:
    """The advisory named-reader findings."""
    return [message for severity, message in _reader_findings(agent_record, config) if not severity]


def agent_record_config_overrides(agent_record: AgentRecordConfig) -> dict[str, Any]:
    """The resolved block as the dotted keys the rendered ``config.yml`` carries.

    Every key is written, defaults included, so the service reads the facts the
    build resolved rather than defaulting them a second time on its own.
    """
    cfg = agent_record
    return {
        "agent_record.backend": cfg.backend,
        "agent_record.retention_days": cfg.retention_days,
        "agent_record.readers": [
            dict(entry) if isinstance(entry, Mapping) else entry for entry in cfg.readers
        ],
        "agent_record.port": cfg.port,
        "agent_record.memory": cfg.memory,
        "agent_record.clickhouse.image": cfg.clickhouse.image,
        "agent_record.clickhouse.memory": cfg.clickhouse.memory,
    }


# --- readers -----------------------------------------------------------------


def _reader_findings(
    agent_record: AgentRecordConfig, config: Mapping[str, Any] | None
) -> list[tuple[bool, str]]:
    """Every reader finding as ``(is_error, message)``."""
    from osprey.deployment.web_terminals.roster import authored_people

    web_terminals = _mapping(_mapping(_mapping(config).get("modules")).get("web_terminals"))
    roster = {
        name
        for user in authored_people(web_terminals)
        if isinstance(name := user.get("name") if isinstance(user, Mapping) else user, str)
    }
    auth_method = _mapping(web_terminals.get("auth")).get("method")

    findings: list[tuple[bool, str]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for position, entry in enumerate(agent_record.readers):
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not READER_NAME_RE.fullmatch(name):
            findings.append(
                (
                    True,
                    f"{_READERS}[{position}] must be a mapping whose 'name' is lowercase "
                    "letters, digits, '_' and '-', starting with a letter or digit; "
                    f"got {entry!r}",
                )
            )
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
        unknown = sorted(str(key) for key in entry if key not in READER_KEYS)
        if unknown:
            findings.append(
                (
                    True,
                    f"{_READERS} entry {name!r} names {', '.join(unknown)}, which a reader "
                    f"entry does not have; the keys are {', '.join(READER_KEYS)}",
                )
            )
        findings.extend(_check_kinds(name, entry))
        findings.extend(_check_days_back(name, entry, agent_record.retention_days))
        findings.extend(_check_reader_subject(name, entry, name in roster, auth_method))
    if duplicates:
        findings.append(
            (
                True,
                f"{_READERS} names {', '.join(repr(name) for name in duplicates)} more "
                "than once. A list in a profile is appended to the list it extends, so an "
                "entry repeated to change it becomes a second entry; state each reader once",
            )
        )
    return findings


def _check_kinds(name: str, entry: Mapping[str, Any]) -> list[tuple[bool, str]]:
    """``kinds`` is a list of the registry's kinds; left out, every kind."""
    if "kinds" not in entry:
        return []
    stated = entry.get("kinds")
    if not isinstance(stated, list | tuple) or not all(isinstance(k, str) for k in stated):
        return [
            (
                True,
                f"{_READERS} entry {name!r} kinds must be a list of kind names, got "
                f"{stated!r}. Remove the key to let this reader see every kind",
            )
        ]
    unknown = sorted(set(stated) - set(KINDS))
    if unknown:
        return [
            (
                True,
                f"{_READERS} entry {name!r} names {', '.join(unknown)}, which the record "
                f"does not have; the kinds are {', '.join(KINDS)}",
            )
        ]
    return []


def _check_days_back(name: str, entry: Mapping[str, Any], retention: int) -> list[tuple[bool, str]]:
    """``days_back`` is a whole number of days, at most what the record keeps."""
    if "days_back" not in entry:
        return []
    days = entry.get("days_back")
    key = f"{_READERS} entry {name!r} days_back"
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        return [
            (
                True,
                f"{key} must be a whole number of days, 1 or more, got {days!r}. Remove "
                "the key to let this reader see everything the record keeps",
            )
        ]
    if retention >= 1 and days > retention:
        return [
            (
                True,
                f"{key} is {days}, more than agent_record.retention_days ({retention}): "
                f"the record holds nothing older than {retention} days, so this window "
                "promises entries that do not exist. Lower it or raise retention_days",
            )
        ]
    return []


def _check_reader_subject(
    name: str, entry: Mapping[str, Any], on_roster: bool, auth_method: Any
) -> list[tuple[bool, str]]:
    """``oidc_subject`` recognises a reader who is NOT on the roster, under OIDC."""
    if "oidc_subject" not in entry:
        return []
    subject = entry.get("oidc_subject")
    key = f"{_READERS} entry {name!r} oidc_subject"
    if not isinstance(subject, str) or not subject.strip():
        return [(True, f"{key} must be a non-empty string")]
    if on_roster:
        return [
            (
                True,
                f"{key} is set, and {name!r} is also a {_WEB}.users entry. A roster user "
                "is recognised in the browser by their own sign-in; a second subject on "
                "the reader entry would let a different sign-in read as them. Remove "
                "oidc_subject from the reader entry",
            )
        ]
    if auth_method != "oidc":
        return [
            (
                False,
                f"{key} is set but {_WEB}.auth.method is {auth_method!r}: nothing asserts "
                "a subject without OIDC sign-on, so this reader has a key for scripts and "
                "no way into the history page",
            )
        ]
    return []


# --- retired keys ------------------------------------------------------------

#: Every config key the agent record replaced, with what a profile writes
#: instead. A key here is refused wherever a profile spells it — the key
#: itself or anything below it, in a nested, dotted or mixed spelling — and
#: the refusal names the replacement, so the operator edits the profile once
#: and builds.
RETIRED_CONFIG_KEYS: Mapping[str, str] = {
    "claude_code.telemetry": (
        "remove it: Claude Code's OTLP logs always go to the agent record, and no "
        "key selects another destination, signal or capture level"
    ),
    "claude_code.transcripts": (
        "set `agent_record.retention_days` instead: Claude Code keeps its "
        "transcripts for the longer of that and 30 days"
    ),
    "audit.tool_call": (
        "remove it: every tool call is an agent-record entry, kept for "
        "`agent_record.retention_days`"
    ),
    "services.telemetry_store": (
        "state the store in the top-level `agent_record:` block instead — "
        "`backend: clickhouse`, with `retention_days`, `readers`, `memory`, "
        "`clickhouse.image` and `clickhouse.memory`"
    ),
    "services.telemetry_gateway": (
        "remove it: every sender logs in to the agent record with a credential the build mints"
    ),
    "web.feedback.max_store_bytes": (
        "remove it: feedback is an agent-record entry, kept for `agent_record.retention_days`"
    ),
    f"{_WEB}.reader_port": f"rename it to `{_WEB}.history_port`, the history page's port",
    f"{_WEB}.reader_origin": f"rename it to `{_WEB}.history_origin`, the history page's origin",
}

#: The ``deployed_services`` entries the agent record replaced.
RETIRED_DEPLOYED_SERVICES: frozenset[str] = frozenset({"telemetry_store", "telemetry_gateway"})

_RETIRED_SERVICE_ENTRY = (
    "remove it: the agent record is always deployed and is never a "
    "`deployed_services` entry; state its settings in the top-level `agent_record:` block"
)


def refuse_retired_config_keys(config: Any) -> None:
    """Refuse a ``config:`` block that spells a key the agent record replaced.

    Every key of :data:`RETIRED_CONFIG_KEYS` and every key below one is
    refused, and so is a :data:`RETIRED_DEPLOYED_SERVICES` entry in
    ``deployed_services``. Nested, dotted and mixed spellings address the same
    keys, so the refusal cannot be stepped around by spelling one differently.
    Every offender is reported at once, each with what to write instead.

    Args:
        config: A profile's ``config:`` block, or a rendered config mapping.

    Raises:
        BuildProfileError: Naming every retired key the block spells and its
            replacement.
    """
    if not isinstance(config, Mapping):
        return
    spelled: dict[str, set[str]] = {}
    services: set[str] = set()
    for path, value in _leaf_paths(config):
        retired = next(
            (key for key in RETIRED_CONFIG_KEYS if path == key or path.startswith(f"{key}.")),
            None,
        )
        if retired is not None:
            spelled.setdefault(retired, set()).add(path)
        elif path == "deployed_services" and isinstance(value, list):
            services.update(
                entry
                for entry in value
                if isinstance(entry, str) and entry in RETIRED_DEPLOYED_SERVICES
            )
    problems = [
        f"{', '.join(sorted(paths))} — {RETIRED_CONFIG_KEYS[key]}"
        for key, paths in sorted(spelled.items())
    ]
    if services:
        problems.append(
            f"deployed_services: {', '.join(sorted(services))} — {_RETIRED_SERVICE_ENTRY}"
        )
    if problems:
        raise BuildProfileError(
            "The profile's config: block states key(s) the agent record replaced. "
            "Edit the profile as directed and build again:\n  " + "\n  ".join(problems)
        )


# --- helpers -----------------------------------------------------------------


def _leaf_paths(node: Mapping[Any, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf of *node* as a dotted path and its value, whatever the spelling.

    A key may itself carry dots, so ``{"a.b": {"c": 1}}`` and
    ``{"a": {"b.c": 1}}`` both give ``a.b.c``. An empty mapping is a leaf.
    """
    leaves: list[tuple[str, Any]] = []
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            leaves.extend(_leaf_paths(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _duplicate_config_key_errors(config: Any) -> list[str]:
    """Refuse ``config:`` spelling a key this block owns, in either spelling."""
    if not isinstance(config, Mapping):
        return []
    stated = sorted(
        str(key)
        for key in config
        if key == AGENT_RECORD_BLOCK or str(key).startswith(f"{AGENT_RECORD_BLOCK}.")
    )
    if not stated:
        return []
    return [
        f"the profile's `config:` block states {', '.join(stated)}. The record's settings "
        f"live in the top-level `agent_record:` block, and the build writes them into "
        f"config.yml from there; move them into that block"
    ]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_clickhouse(value: Any, problems: list[str]) -> ClickHouseConfig:
    if not isinstance(value, Mapping):
        problems.append(f"'clickhouse' must be a mapping (got {type(value).__name__}).")
        return ClickHouseConfig()
    _reject_unknown_keys(value, _KNOWN_CLICKHOUSE_KEYS, "agent_record.clickhouse", problems)
    values: dict[str, Any] = {}
    if "image" in value:
        values["image"] = _parse_str(value["image"], "clickhouse.image", problems)
    if "memory" in value:
        values["memory"] = _parse_memory_value(value["memory"], "clickhouse.memory", problems)
    return ClickHouseConfig(**values)


def _reject_unknown_keys(
    block: Mapping[str, Any], known: frozenset[str], where: str, problems: list[str]
) -> None:
    """Reject anything the block does not define, naming the closest spelling."""
    names = sorted(known)
    for key in sorted(str(k) for k in block if k not in known):
        close = difflib.get_close_matches(key, names, n=1)
        suggestion = f" (did you mean '{close[0]}'?)" if close else ""
        problems.append(
            f"unknown key '{key}'{suggestion} — valid {where} keys are: {', '.join(names)}."
        )


def _parse_int(value: Any, key: str, problems: list[str]) -> int:
    """Read one integer knob, refusing the bool YAML would otherwise let pass."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    problems.append(f"'{key}' must be an integer (got {value!r}).")
    return 1


def _parse_str(value: Any, key: str, problems: list[str]) -> str:
    """Read one string knob."""
    if not isinstance(value, str):
        problems.append(f"'{key}' must be a string (got {type(value).__name__}).")
        return ""
    return value


def _parse_memory_value(value: Any, key: str, problems: list[str]) -> str | int:
    """Read a memory size — a string such as ``2g`` or a byte count."""
    if isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    problems.append(f"'{key}' must be a memory size such as 2g or 512m (got {value!r}).")
    return DEFAULT_MEMORY
