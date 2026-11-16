"""Where each health row is shown: layers, components, row labels, probe methods.

The health suite produces rows per *category* — the unit a check runs in. A
reader of the System Health panel asks a different question: which part of
the running deployment is this about? This module is the one place that
answers it. A single table (:data:`RULES`) maps every core row, by category
and name pattern, to

* a **component** — the tile the row belongs to (``graph store``, ``hooks``);
  each component belongs to one **layer** (:data:`LAYERS`), the band it is
  drawn in, operator-facing at the top, the host at the bottom;
* a **label** — what was checked, in a few words (``bolt connection``);
* a **method** — how it was measured (``bolt connect + RETURN 1``);
* optionally a **group** inside the component (one per panel, one per
  agent-record part).

The checks never name their tile, so a check can move between categories and
a tile can be re-cut without touching a probe. A row no rule matches — a
facility's YAML or plugin category — gets a component of its own in the
trailing ``facility`` layer, so nothing a deployment declares is hidden.
Rows about machine physics rather than OSPREY itself (the simulator's status
channels) map to no layer: the panel reports OSPREY's own health, and they stay
in the CLI report.

:func:`build_layers` turns a serialized report into the panel's ``layers``
structure (see the panel's ``/checks`` envelope): rows annotated in place
with ``layer``, ``component``, ``label``, ``method``, ``group`` and
``display``, and per component its status, one-fact summary, issue text and
row indices. Pure: no I/O, no clock.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from osprey.health.models import NOT_USED
from osprey.health.runner import ON_DEMAND_PREFIX

Row = dict[str, Any]

#: Severity order of the wire statuses, best to worst.
_RANK = {"skip": 0, "ok": 1, "warning": 2, "error": 3}


@dataclass(frozen=True)
class Layer:
    """One band of the panel."""

    id: str
    label: str


#: The bands, top to bottom. ``facility`` is drawn only when a row lands in it.
LAYERS: tuple[Layer, ...] = (
    Layer("op", "Operator interface"),
    Layer("agent", "Agent"),
    Layer("chain", "Safety chain"),
    Layer("mcp", "MCP servers"),
    Layer("data", "Data services"),
    Layer("out", "Connections out"),
    Layer("cfg", "Configuration & build"),
    Layer("host", "Host"),
    Layer("facility", "Facility checks"),
)

#: Layer id of rows the panel does not draw (machine physics, not OSPREY).
MACHINE = "machine"

Fact = Callable[[list[Row]], str]


@dataclass(frozen=True)
class Component:
    """One tile: its layer, its name, and how its one fact is read off its rows."""

    layer: str
    name: str
    fact: Fact | None = None
    header: Fact | None = None


# -- fact helpers ---------------------------------------------------------------


def _shown(rows: list[Row], name: str) -> str:
    """The value (else display text) of the row called *name*, or ``""``."""
    return str(next((r.get("value") or r["display"] for r in rows if r["name"] == name), ""))


def _ok_of(rows: list[Row]) -> str:
    counted = [r for r in rows if r["status"] != "skip"]
    ok = sum(1 for r in counted if r["status"] == "ok")
    return f"{ok} / {len(counted)}"


def _first_number(text: str) -> str:
    match = re.search(r"\d[\d,.]*", text)
    return match.group(0) if match else ""


def _logins(rows: list[Row]) -> str:
    n = sum(1 for r in rows if r["name"].startswith("web_terminals."))
    return f"{n} login{'s' if n != 1 else ''}"


def _panels(rows: list[Row]) -> str:
    groups: dict[str, str] = {}
    for r in rows:
        g = r.get("group") or r["name"]
        if _RANK[r["status"]] > _RANK.get(groups.get(g, "skip"), 0):
            groups[g] = r["status"]
        groups.setdefault(g, r["status"])
    up = sum(1 for s in groups.values() if s in ("ok", "skip"))
    return f"{up} / {len(groups)}"


def _version(rows: list[Row]) -> str:
    for r in rows:
        match = re.search(r"\d+\.\d+\.\d+", f"{r.get('value', '')} {r['message']}")
        if match:
            return match.group(0)
    return ""


def _hooks(rows: list[Row]) -> str:
    hooks = [r for r in rows if r["name"].startswith("hooks.osprey_")]
    ok = sum(1 for r in hooks if r["status"] == "ok")
    return f"{ok} / {len(hooks)}" + (" = render" if hooks and ok == len(hooks) else "")


def _switch(rows: list[Row]) -> str:
    positions = {
        r["display"]
        for r in rows
        if r["name"].startswith("write_switch.") and not r["display"].startswith("never")
    }
    if len(positions) == 1:
        return str(positions.pop())
    return f"{len(positions)} positions" if positions else "never"


def _watchdogs(rows: list[Row]) -> str:
    dogs = [r for r in rows if r["name"].startswith("watchdog.") and r["status"] != "skip"]
    dogs = [r for r in dogs if r["name"] != "watchdog.flags_open"]
    flags = _first_number(_shown(rows, "watchdog.flags_open")) or "0"
    if not dogs:
        return "none configured"
    return f"{len(dogs)} · {flags} flags"


def _graph_store(rows: list[Row]) -> str:
    for r in rows:
        match = re.search(r"bolt://[^:/\s]+:(\d+)", r["message"])
        if match:
            return f"bolt :{match.group(1)}"
    return _shown(rows, "graphdb_resources")


def _telemetry_backend(rows: list[Row]) -> str:
    return _shown(rows, "agent_record_backend").split(" · ")[0]


def _telemetry_header(rows: list[Row]) -> str:
    backend = _shown(rows, "agent_record_backend")
    return f"agent record · {backend}" if backend else "agent record"


def _subagents(rows: list[Row]) -> str:
    agents = _first_number(_shown(rows, "rendered.agents"))
    skills = _first_number(_shown(rows, "rendered.skills"))
    return " · ".join(x for x in (agents, skills) if x)


def _render_drift(rows: list[Row]) -> str:
    return "no drift" if all(r["status"] in ("ok", "skip") for r in rows) else "drift"


def _reachable(_rows: list[Row]) -> str:
    return "reachable"


def _valid(_rows: list[Row]) -> str:
    return "valid"


def _disk(rows: list[Row]) -> str:
    match = re.search(r"(\d+)%", _shown(rows, "disk_space"))
    return f"{match.group(1)} % used" if match else ""


def _runtime(rows: list[Row]) -> str:
    running = [r for r in rows if r["name"].startswith("container_") and r.get("value")]
    if running:
        return f"{sum(1 for r in running if r['status'] == 'ok')} / {len(running)} running"
    return rows[0]["display"] if rows else ""


def _keys(rows: list[Row]) -> str:
    return _shown(rows, "environment_variables") or _ok_of(rows)


def _knowledge(rows: list[Row]) -> str:
    n = _first_number(_shown(rows, "qmd_okf"))
    return f"{n} docs" if n else _shown(rows, "okf.bundle")


def _providers_used(rows: list[Row]) -> str:
    names = [r["name"] for r in rows if r["category"] == "providers"]
    return " · ".join(dict.fromkeys(names))


def _value_of(name: str) -> Fact:
    return lambda rows: _shown(rows, name)


def _prefixed(prefix: str, name: str) -> Fact:
    return lambda rows: f"{prefix} {_shown(rows, name)}".strip()


#: Every tile, in drawing order within its layer.
COMPONENTS: dict[str, Component] = {
    "web-terminal": Component("op", "web terminal", _logins),
    "panels": Component("op", "panels", _panels),
    "claude-code": Component("agent", "Claude Code", _version),
    "model-provider": Component("agent", "model provider", None),
    "embed-vision": Component("agent", "embedding · vision", _providers_used),
    "subagents-skills": Component("agent", "sub-agents · skills", _subagents),
    "dispatch": Component("agent", "dispatch workers", _value_of("dispatch.pool")),
    "hooks": Component("chain", "hooks", _hooks),
    "write-switch": Component("chain", "write switch", _switch),
    "limits": Component(
        "chain", "limits file", lambda r: _shown(r, "limits.parse").split(" · ")[0]
    ),
    "watchdogs": Component("chain", "watchdogs", _watchdogs),
    "sandbox": Component("chain", "exec sandbox", _value_of("exec_sandbox.guards")),
    "mcp-servers": Component("mcp", "MCP servers", None),
    "graph-store": Component("data", "graph store", _graph_store),
    "channel-db": Component("data", "channel database", _value_of("channel_finder_pipeline")),
    "pv-directory": Component("data", "PV directory", _ok_of),
    "ariel-db": Component("data", "ARIEL database", _value_of("ariel_entries")),
    "knowledge-index": Component("data", "knowledge index", _knowledge),
    "agent-record": Component(
        "data", "telemetry store", _telemetry_backend, header=_telemetry_header
    ),
    "shared-services": Component("data", "shared services", _reachable),
    "cs-connector": Component("out", "control-system connector", _reachable),
    "archiver-connector": Component("out", "archiver connector", _reachable),
    "config-yml": Component("cfg", "config.yml", _valid),
    "rendered": Component("cfg", ".claude/ · .mcp.json", _render_drift),
    "versions": Component("cfg", "versions", _prefixed("osprey", "versions.osprey")),
    "provider-keys": Component("cfg", "provider keys", _keys),
    "disk": Component("host", "disk", _disk),
    "python": Component("host", "python", _version),
    "container-runtime": Component("host", "container runtime", _runtime),
}


# -- row rules ------------------------------------------------------------------

Text = str | Callable[[Row], str]


@dataclass(frozen=True)
class Rule:
    """Map rows of *category* whose name matches *pattern* (one ``*`` at most).

    ``label`` and ``group`` may use ``{tail}`` (what ``*`` matched) or be
    callables of the row; ``display`` replaces the row's shown value; ``when``
    restricts the rule to rows it accepts. ``component=None`` means machine
    scope: annotated, never drawn.
    """

    category: str
    pattern: str
    component: str | None
    label: Text
    method: str
    group: Text | None = None
    display: Callable[[Row], str] | None = None
    when: Callable[[Row], bool] | None = None
    tags: tuple[str, ...] = field(default=())


def _msg_prefix(row: Row) -> str:
    return str(row["message"].split(":", 1)[0].strip())


def _panel_group(row: Row) -> str:
    return _msg_prefix(row).capitalize()


def _after(prefix: str) -> Callable[[Row], str]:
    """Display: the message after *prefix*, when the row has no value."""

    def show(row: Row) -> str:
        message = row["message"]
        return str(message[len(prefix) :].strip() if message.startswith(prefix) else message)

    return show


def _after_colon(row: Row) -> str:
    """Display: the row's value, else its message after the ``"<consumer>: "`` prefix."""
    if row.get("value") and row["status"] == "ok":
        return str(row["value"])
    return str(row["message"].split(": ", 1)[-1])


def _ok_word(word: str) -> Callable[[Row], str]:
    """Display: *word* when the row is ok, else its own text."""
    return lambda row: word if row["status"] == "ok" else row.get("value") or row["message"]


def _is_connector(row: Row) -> bool:
    return "connector" in row["message"]


def _is_graph(row: Row) -> bool:
    return "graph" in str(row.get("value", ""))


def _freshness_category() -> str:
    """The category the build writes the archiver freshness check into."""
    from osprey.cli.build_profile_archiver import FRESHNESS_CATEGORY

    return FRESHNESS_CATEGORY


RULES: tuple[Rule, ...] = (
    # configuration
    Rule(
        "configuration",
        "config_file_exists",
        "config-yml",
        "file",
        "file stat",
        display=_after("Found at"),
    ),
    Rule(
        "configuration",
        "yaml_valid",
        "config-yml",
        "syntax",
        "YAML parse",
        display=_ok_word("0 errors"),
    ),
    Rule("configuration", "health_config", "config-yml", "health section", "config parse health:"),
    Rule("configuration", "model_configs", "config-yml", "models section", "config parse models:"),
    Rule(
        "configuration",
        "recommended_models",
        "config-yml",
        "recommended models",
        "config parse models:",
    ),
    Rule(
        "configuration",
        "model_configs_valid",
        "config-yml",
        "model configs",
        "schema check",
        display=_ok_word("0 errors"),
    ),
    Rule(
        "configuration",
        "deployed_services",
        "config-yml",
        "deployed services",
        "config parse deployed_services",
    ),
    Rule(
        "configuration",
        "service_definitions",
        "config-yml",
        "service definitions",
        "config parse services:",
        display=_ok_word("all defined"),
    ),
    Rule(
        "configuration",
        "api_providers",
        "model-provider",
        "catalogue",
        "config parse api.providers",
        display=lambda r: (
            f"{_first_number(r['message'])} providers" if r["status"] == "ok" else r["message"]
        ),
    ),
    Rule(
        "configuration",
        "environment_variables",
        "provider-keys",
        "environment variables",
        "env lookup",
    ),
    Rule(
        "configuration",
        "timezone",
        "config-yml",
        "timezone",
        "config parse system.timezone",
        display=_after("Timezone:"),
    ),
    Rule("configuration", "ariel_dsn_port", "ariel-db", "DSN port", "config parse ariel.database"),
    Rule("configuration", "control_system", "cs-connector", "restart notice", "config change"),
    Rule("configuration", "*", "config-yml", "{tail}", "config parse"),
    Rule("control_system", "*", "cs-connector", "control target", "config parse control_system"),
    # file system / python / containers
    Rule("file_system", "project_root_path", "disk", "project root", "dir stat"),
    Rule(
        "file_system",
        "agent_data_dir",
        "disk",
        "agent data dir",
        "write probe",
        display=_ok_word("writable"),
    ),
    Rule("file_system", "env_file", "provider-keys", ".env file", "file stat"),
    Rule("file_system", "users_env", "provider-keys", ".env.users", "compare with env chain"),
    Rule("file_system", "disk_space", "disk", "project volume", "statvfs"),
    Rule("file_system", "*", "disk", "{tail}", "file stat"),
    Rule(
        "python_environment",
        "python_version",
        "python",
        "interpreter",
        "sys.version",
        display=_after("Python"),
    ),
    Rule("python_environment", "virtual_environment", "python", "environment", "sys.prefix"),
    Rule(
        "python_environment",
        "core_dependencies",
        "python",
        "core dependencies",
        "importlib.metadata",
    ),
    Rule("python_environment", "*", "python", "{tail}", "importlib.metadata"),
    Rule(
        "containers",
        "container_runtime",
        "container-runtime",
        "deployment record",
        "deployment record read",
    ),
    Rule("containers", "*_available", "container-runtime", "runtime", "exec --version"),
    Rule("containers", "container_*", "container-runtime", "{tail}", "container inspect"),
    Rule("containers", "*", "container-runtime", "{tail}", "container inspect"),
    Rule("systemd_unit", "*", "container-runtime", "systemd unit", "file stat osprey.service"),
    # agent
    Rule(
        "claude_cli",
        "*",
        "claude-code",
        "CLI version",
        "exec claude --version",
        display=lambda r: _version([r]) or r["message"],
    ),
    Rule("claude_cli_pinned", "*", "claude-code", "version pin", "compare with build pin"),
    Rule("model_chat", "*", "model-provider", "chat completion", "chat completion"),
    Rule(
        "rendered",
        "rendered.agents",
        "subagents-skills",
        "sub-agents rendered",
        "sha256 vs build manifest",
    ),
    Rule(
        "rendered",
        "rendered.skills",
        "subagents-skills",
        "skills rendered",
        "dir walk + YAML parse .claude/skills",
    ),
    Rule("rendered", "rendered.claude_dir", "rendered", ".claude/", "sha256 vs build manifest"),
    Rule(
        "rendered",
        "rendered.settings_json",
        "rendered",
        ".claude/settings.json",
        "sha256 vs build manifest",
    ),
    Rule("rendered", "rendered.mcp_json", "rendered", ".mcp.json", "sha256 vs build manifest"),
    Rule("rendered", "*", "rendered", "{tail}", "sha256 vs build manifest"),
    Rule("dispatch", "dispatch.pool", "dispatch", "worker pool", "http GET dispatcher /health"),
    Rule(
        "dispatch",
        "dispatch.failed_24h",
        "dispatch",
        "failed runs, 24 h",
        "http GET dispatcher /dashboard/state",
    ),
    Rule("dispatch", "*", "dispatch", "{tail}", "http GET dispatcher"),
    Rule(
        "web_terminals",
        "web_terminals.dispatch_worker*",
        "dispatch",
        "worker{tail} identity",
        "config parse web_terminals",
    ),
    Rule(
        "web_terminals", "web_terminals.*", "web-terminal", "{tail}", "config parse web_terminals"
    ),
    # safety chain
    Rule("safety_chain", "hooks.order", "hooks", "enforcement order", "PreToolUse array order"),
    Rule("safety_chain", "hooks.*", "hooks", "{tail}", "JSON parse settings.json + sha256"),
    Rule("safety_chain", "hooks", "hooks", "hooks", "JSON parse settings.json"),
    Rule("safety_chain", "write_switch.*", "write-switch", "{tail}", "state record read"),
    Rule("safety_chain", "write_switch", "write-switch", "write switch", "state record read"),
    Rule(
        "safety_chain",
        "write_authorizations",
        "write-switch",
        "active write authorizations",
        "state dir listing",
    ),
    Rule("safety_chain", "limits.parse", "limits", "parse", "YAML parse limits file"),
    Rule("safety_chain", "limits.age", "limits", "file age", "file stat"),
    Rule("safety_chain", "limits*", "limits", "limits", "YAML parse limits file"),
    Rule("safety_chain", "watchdog.flags_open", "watchdogs", "open flags", "hold record stat"),
    Rule("safety_chain", "watchdog.*", "watchdogs", "{tail}", "config parse watchdogs"),
    Rule("safety_chain", "watchdog*", "watchdogs", "watchdogs", "config parse watchdogs"),
    Rule("safety_chain", "exec_sandbox.guards", "sandbox", "guards", "compile wrapper per mode"),
    Rule("safety_chain", "exec_sandbox.timeout", "sandbox", "execution timeout", "config parse"),
    Rule("safety_chain", "*", "sandbox", "{tail}", "compile wrapper per mode"),
    # MCP
    Rule("mcp_servers", "*", "mcp-servers", "{tail}", "http MCP initialize + tools/list"),
    # panels
    Rule(
        "web_panels",
        "web_panels.*",
        "panels",
        "HTTP /health",
        "http GET /health",
        group=_panel_group,
        display=_after_colon,
    ),
    Rule(
        "service_capabilities",
        "bluesky.capability",
        "panels",
        "bridge capability",
        "http GET bridge /health",
        group="Bluesky",
    ),
    Rule(
        "reach",
        "reach.bluesky",
        "panels",
        "backend: bluesky bridge",
        "tcp connect",
        group="Bluesky",
    ),
    Rule("reach", "reach.event_dispatcher", "dispatch", "event dispatcher", "tcp connect"),
    # data services
    Rule(
        "graphdb",
        "graphdb_connection",
        "graph-store",
        "bolt connection",
        "bolt connect + RETURN 1",
        display=_ok_word("open"),
    ),
    Rule("graphdb", "graphdb_resources", "graph-store", "Resource nodes", "bolt count(Resource)"),
    Rule("graphdb", "graphdb_seed", "graph-store", "seed marker", "bolt query vs corpus sha256"),
    Rule("graphdb", "*", "graph-store", "{tail}", "bolt query"),
    Rule("reach", "reach.graphdb", "graph-store", "bolt port", "tcp connect"),
    Rule(
        "channel_finder",
        "channel_finder_pipeline",
        "graph-store",
        "channel finder pipeline",
        "config parse channel_finder.pipeline_mode",
        when=_is_graph,
    ),
    Rule(
        "channel_finder",
        "channel_finder_store",
        "graph-store",
        "bolt connection",
        "bolt connect + RETURN 1",
        display=_ok_word("open"),
    ),
    Rule(
        "channel_finder",
        "channel_finder_resources",
        "graph-store",
        "Resource nodes",
        "bolt count(Resource)",
    ),
    Rule(
        "channel_finder",
        "channel_finder_seed",
        "graph-store",
        "seed marker",
        "bolt query vs corpus sha256",
    ),
    Rule(
        "channel_finder",
        "channel_finder_search_index",
        "graph-store",
        "search index",
        "file stat + manifest",
    ),
    Rule(
        "channel_finder",
        "channel_finder_pipeline",
        "channel-db",
        "pipeline",
        "config parse channel_finder.pipeline_mode",
    ),
    Rule("channel_finder", "*", "channel-db", "{tail}", "file stat"),
    Rule(
        "pv_directory",
        "*",
        "pv-directory",
        _msg_prefix,
        "service probe",
        display=_after_colon,
    ),
    Rule(
        "pv_directory_alarm_feed",
        "*",
        "pv-directory",
        "alarm feed",
        "alarm snapshot, read after catch-up",
        display=_after_colon,
    ),
    Rule(
        "ariel", "ariel_status", "ariel-db", "service", "http GET /health", display=_ok_word("200")
    ),
    Rule("ariel", "ariel_entries", "ariel-db", "entries", "sql count"),
    Rule("ariel", "ariel_last_ingestion", "ariel-db", "last ingestion", "sql max(ingested_at)"),
    Rule("ariel", "ariel_search_modules", "ariel-db", "search modules", "config parse"),
    Rule("ariel", "ariel_enhancement_modules", "ariel-db", "enhancement modules", "config parse"),
    Rule("ariel", "*", "ariel-db", "{tail}", "http GET"),
    Rule("reach", "reach.postgresql", "ariel-db", "database port", "tcp connect"),
    Rule("qmd", "qmd_ariel", "ariel-db", "hybrid search (qmd)", "qmd index status"),
    Rule("reach", "reach.qmd-ariel", "ariel-db", "hybrid search port", "tcp connect"),
    Rule("qmd", "qmd_okf", "knowledge-index", "OKF index (qmd)", "qmd index status"),
    Rule("qmd", "*", "knowledge-index", "qmd index {tail}", "qmd index status"),
    Rule("reach", "reach.qmd-okf", "knowledge-index", "OKF index port", "tcp connect"),
    Rule(
        "service_capabilities",
        "okf.bundle",
        "knowledge-index",
        "OKF bundle",
        "sha256 of bundle vs manifest",
    ),
    Rule("service_capabilities", "okf.bundle_age", "knowledge-index", "bundle age", "file stat"),
    Rule("service_capabilities", "okf*", "knowledge-index", "OKF bundle", "file stat"),
    Rule(
        "ariel_modules",
        "ariel.enhancement.*",
        "embed-vision",
        "ARIEL {tail}",
        "module health_check()",
    ),
    Rule(
        "ariel_modules",
        "ariel.enhancement",
        "embed-vision",
        "ARIEL modules",
        "module health_check()",
    ),
    Rule(
        "service_capabilities",
        "bluesky*",
        "panels",
        "bridge capability",
        "http GET bridge /health",
        group="Bluesky",
    ),
    Rule(
        "agent_record",
        "agent_record_backend",
        "agent-record",
        "backend",
        "http GET /health/sources",
        group="store",
    ),
    Rule(
        "agent_record",
        "agent_record.retention",
        "agent-record",
        "retention",
        "config parse agent_record.retention_days",
        group="store",
    ),
    Rule(
        "agent_record",
        "agent_record.disk",
        "agent-record",
        "disk used",
        "http GET /health/sources",
        group="store",
    ),
    Rule(
        "agent_record",
        "agent_record.otlp*",
        "agent-record",
        "OTLP{tail}",
        "http GET /health/sources",
        group="OTLP receiver",
    ),
    Rule(
        "agent_record",
        "agent_record.spool.backlog",
        "agent-record",
        "backlog",
        "http GET /health/sources",
        group="spool",
    ),
    Rule(
        "agent_record",
        "agent_record.spool.oldest",
        "agent-record",
        "oldest uncommitted",
        "http GET /health/sources",
        group="spool",
    ),
    Rule(
        "agent_record",
        "agent_record_source.*",
        "agent-record",
        "{tail}",
        "http GET /health/sources",
        group="spool",
    ),
    Rule(
        "agent_record",
        "agent_record_transcripts.*",
        "agent-record",
        "{tail}",
        "dir stat transcript root",
        group="transcripts",
    ),
    Rule(
        "agent_record",
        "agent_record_export",
        "agent-record",
        "daily export",
        "file stat export dir",
        group="export",
    ),
    Rule(
        "agent_record",
        "agent_record.readers.*",
        "agent-record",
        "{tail}",
        "config parse agent_record.readers",
        group="readers",
    ),
    Rule("agent_record", "*", "agent-record", "agent record", "http GET /health/sources"),
    Rule(
        "archive",
        "*",
        "agent-record",
        "last archive pass",
        "file stat var/archive",
        group="archive",
    ),
    # connections out
    Rule(
        "reach",
        "reach.*",
        "cs-connector",
        _msg_prefix,
        "tcp connect",
        when=_is_connector,
        display=_after_colon,
    ),
    Rule("reach", "reach.*", "shared-services", _msg_prefix, "tcp connect", display=_after_colon),
    # versions
    Rule("versions", "versions.osprey", "versions", "osprey", "package metadata"),
    Rule("versions", "versions.canopy", "versions", "canopy", "bundle manifest vs pin"),
    Rule("versions", "*", "versions", "{tail}", "package metadata"),
    # machine physics: reported by the CLI, not drawn
    Rule("simulator", "*", None, "{tail}", "channel read"),
)


def _tail(pattern: str, name: str) -> str:
    if "*" not in pattern:
        return name
    head, _, rest = pattern.partition("*")
    return name[len(head) : len(name) - len(rest) if rest else None]


def _text(value: Text, row: Row, tail: str) -> str:
    return value(row) if callable(value) else value.replace("{tail}", tail)


def _rule_for(row: Row) -> Rule | None:
    category = row["category"]
    if category == _freshness_category():
        return Rule(category, "*", "archiver-connector", "archive freshness", "archiver read")
    for rule in RULES:
        if rule.category != category or not fnmatchcase(row["name"], rule.pattern):
            continue
        if rule.when is not None and not rule.when(row):
            continue
        return rule
    return None


def _display(row: Row, text: str | None = None) -> str:
    """The row's shown value: *text* (else value, else message) with its latency once."""
    shown = str(text or row.get("value") or row["message"])
    ms = row.get("latency_ms")
    if ms and row["status"] != "skip" and not re.search(r"\d ms\b", shown):
        return f"{shown} · {round(ms)} ms"
    return shown


def annotate(row: Row, provider_roles: Mapping[str, str]) -> None:
    """Add ``layer``, ``component``, ``label``, ``method``, ``group``, ``display`` to *row*.

    Args:
        row: One serialized result (``CheckResult.to_dict()``), updated in place.
        provider_roles: Provider name → ``"agent"`` for the deployment's own
            agent provider, ``"other"`` for one some other role names; a
            provider absent from it is not used.
    """
    rule = _rule_for(row)
    component: str | None
    if row["category"] == "providers":
        role = provider_roles.get(row["name"])
        component = "model-provider" if role == "agent" or role is None else "embed-vision"
        rule = Rule("providers", "*", component, row["name"], _provider_method(row))
    if rule is None:
        component = f"facility.{row['category']}"
        layer = "facility"
        row.update(
            layer=layer,
            component=component,
            label=row["name"],
            method="declared check",
            group=None,
        )
    else:
        tail = _tail(rule.pattern, row["name"])
        if rule.component is None:
            layer, component = MACHINE, None
        else:
            layer, component = COMPONENTS[rule.component].layer, rule.component
        row.update(
            layer=layer,
            component=component,
            label=_text(rule.label, row, tail).strip() or row["name"],
            method=rule.method,
            group=_text(rule.group, row, tail) if rule.group is not None else None,
        )
    display = rule.display(row) if rule is not None and rule.display is not None else None
    if row["status"] == "skip" and row["message"].startswith(ON_DEMAND_PREFIX):
        # The runner's placeholder for a costly category the poll never runs.
        display = "on demand · osprey health --full"
    elif row.get("value") == NOT_USED:
        # The group these rows are drawn under already says why.
        display = "not probed"
    row["display"] = _display(row, display)


def _provider_method(row: Row) -> str:
    if row.get("value") == NOT_USED:
        return "config parse provider roles"
    if "key not set" in row["message"].lower():
        return "env lookup"
    return "provider health check"


def provider_roles(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Provider name → ``"agent"`` / ``"other"`` for every provider a role names."""
    from osprey.health.core.providers import used_providers

    cfg = config or {}
    claude_code = cfg.get("claude_code") if isinstance(cfg.get("claude_code"), Mapping) else {}
    own = claude_code.get("provider") if isinstance(claude_code, Mapping) else None
    roles = dict.fromkeys(used_providers(cfg), "other")
    if isinstance(own, str) and own:
        roles[own] = "agent"
    return roles


# -- assembly -------------------------------------------------------------------


def _worst(rows: Sequence[Row]) -> str:
    status = "skip"
    for r in rows:
        if _RANK[r["status"]] > _RANK[status]:
            status = r["status"]
    return status


#: Longest issue text a tile carries before it is cut; the drawer has the rest.
_ISSUE_CHARS = 64


def _clip(text: str) -> str:
    return text if len(text) <= _ISSUE_CHARS else text[: _ISSUE_CHARS - 1].rstrip() + "…"


def _said(row: Row) -> str:
    """What a faulty row says, without its latency: its value, else its display."""
    return str(row.get("value") or row["display"])


def _issue(rows: list[Row]) -> str | None:
    """The tile's issue text: the worst rows, said once."""
    bad = [r for r in rows if r["status"] in ("warning", "error")]
    if not bad:
        return None
    worst = _worst(bad)
    bad = [r for r in bad if r["status"] == worst]
    first = bad[0]
    if len(bad) == 1:
        where = " ".join(x for x in (first.get("group"), first["label"]) if x)
        return _clip(f"{where}: {_said(first)}")
    groups = {r.get("group") for r in bad}
    shown = {r.get("value") or r["display"] for r in bad}
    group = first.get("group")
    if len(groups) == 1 and group:
        total = sum(1 for r in rows if r.get("group") == group)
        what = shown.pop() if len(shown) == 1 else "faults"
        return _clip(f"{group}: {len(bad)} / {total} {what}")
    return _clip(f"{len(bad)} faults · {first['label']}: {_said(first)}")


def _group_summary(rows: list[Row]) -> str:
    if len(rows) == 1:
        return str(rows[0]["display"])
    counted = [r for r in rows if r["status"] != "skip"]
    bad = [r for r in counted if r["status"] in ("warning", "error")]
    if bad:
        return f"{len(counted) - len(bad)} / {len(counted)} OK"
    return f"{len(rows)} OK" if counted else rows[0]["display"]


def build_layers(
    results: list[Row],
    config: Mapping[str, Any] | None,
    *,
    mcp_configured: Sequence[str] = (),
    mcp_answered: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Annotate *results* in place and return the panel's ``layers`` structure.

    Args:
        results: The serialized report rows (``CheckResult.to_dict()`` each).
        config: The expanded config the suite ran against, for provider roles.
        mcp_configured: Server names from the rendered ``.mcp.json``; when any,
            the MCP tile is drawn even if no row reports on a server.
        mcp_answered: ``(answered, stdio)`` from the introspection cache, or
            ``None`` when nothing has been introspected: the tile is then not
            probed (``skip``), never passing on configuration alone.

    Returns:
        One dict per drawn layer, in :data:`LAYERS` order, each with its
        components in :data:`COMPONENTS` order (facility components in
        first-seen order).
    """
    roles = provider_roles(config)
    by_component: dict[str, list[int]] = {}
    for index, row in enumerate(results):
        annotate(row, roles)
        if row["component"] is not None:
            by_component.setdefault(row["component"], []).append(index)
    if mcp_configured:
        by_component.setdefault("mcp-servers", [])

    layers: list[dict[str, Any]] = []
    for layer in LAYERS:
        ids = [cid for cid, c in COMPONENTS.items() if c.layer == layer.id]
        if layer.id == "facility":
            ids = [cid for cid in by_component if cid.startswith("facility.")]
        components = [
            _component(cid, by_component[cid], results, mcp_configured, mcp_answered)
            for cid in ids
            if cid in by_component
        ]
        if not components:
            continue
        layers.append(
            {
                "id": layer.id,
                "label": layer.label,
                "status": _worst([{"status": c["status"]} for c in components]),
                "components": components,
            }
        )
    return layers


def _component(
    cid: str,
    indices: list[int],
    results: list[Row],
    mcp_configured: Sequence[str],
    mcp_answered: tuple[int, int] | None = None,
) -> dict[str, Any]:
    spec = COMPONENTS.get(cid) or Component("facility", cid.split(".", 1)[1].replace("_", " "))
    unused = [i for i in indices if results[i].get("value") == NOT_USED]
    shown = [i for i in indices if i not in unused]
    shown_rows = [results[i] for i in shown]

    groups: dict[str, list[int]] = {}
    direct: list[int] = []
    for i in shown:
        group = results[i].get("group")
        if group:
            groups.setdefault(group, []).append(i)
        else:
            direct.append(i)

    status = _worst(shown_rows) if shown_rows else "skip"
    issue = _issue(shown_rows)
    if cid == "mcp-servers":
        fact = f"{len(mcp_configured)} servers"
        if mcp_answered is None:
            fact += " · not probed"
        else:
            answered, stdio = mcp_answered
            probed = "ok" if answered == stdio else "warning" if answered else "error"
            if stdio:
                status = _worst([{"status": status}, {"status": probed}])
    elif issue:
        fact = issue
    elif spec.fact is not None:
        fact = spec.fact(shown_rows) or _ok_of(shown_rows)
    elif cid == "model-provider":
        fact = next((r["name"] for r in shown_rows if r["category"] == "providers"), "")
    else:
        fact = _ok_of(shown_rows)
    if status == "skip" and shown_rows and not issue:
        fact = shown_rows[0]["display"]

    used_by: list[str] = []
    for r in shown_rows:
        for consumer in r.get("used_by", []):
            if consumer not in used_by:
                used_by.append(consumer)

    return {
        "id": cid,
        "name": spec.name,
        "status": status,
        "fact": fact,
        "issue": issue,
        "header": spec.header(shown_rows) if spec.header is not None else None,
        "used_by": used_by,
        "rows": direct,
        "groups": [
            {
                "id": name,
                "name": name,
                "status": _worst([results[i] for i in idx]),
                "summary": _group_summary([results[i] for i in idx]),
                "rows": idx,
                "fix": next((results[i]["fix"] for i in idx if results[i].get("fix")), None),
            }
            for name, idx in groups.items()
        ],
        "unused": unused,
    }
