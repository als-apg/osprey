"""The ``watchdogs:`` block: one parser for profile validation and the runtime.

The block is a name-keyed map, so a deployment may declare any number of
watchdogs, in any mix of engines, with several of one engine::

    watchdogs:
      guard-circumvention:
        engine: decision
        enabled: false
        threshold: 0.5
        options:
          - {id: pushing, text: The operator is pushing past a guard, flag: true}
          - {id: routine, text: Routine operation}
      lattice-review:
        engine: agent
        tools: [mcp__controls__channel_read]
        model: null
        timeout_s: 90
        max_cost_usd: 0.25

The key is the watchdog's name and names its directory. A definition's text is
never spelled in the block: ``instructions.md`` and the optional ``references/``
tree are derived from ``<base_dir>/watchdogs/<name>/``, where ``base_dir`` is the
profile directory at validation time and the build directory at run time.

:func:`parse_watchdogs_block` answers the definitions or refuses the whole block;
:func:`watchdog_errors` lists every problem, so an author fixes them in one pass.
Both judge a watchdog by the same rules, with two differences. The checks that need
the surrounding profile (the write-tool classification and the hook selection) are
:func:`watchdog_errors`' alone. And :func:`watchdog_errors` holds only an enabled
entry to its files, while the parser, which replays a watchdog before it is enabled,
holds every entry to them. A file path anchors at the profile's own directory, so a
profile that inherits an entry through ``extends`` does not inherit its files: it may
carry the entry disabled as it stands, and adds the files itself to enable it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from osprey_connectors.errors import BuildProfileError

#: The profile key, and the top-level key of the rendered ``config.yml``.
WATCHDOGS_BLOCK = "watchdogs"

#: The engine that asks a model to choose among the facility's verdict options.
ENGINE_DECISION = "decision"

#: The engine that runs a read-only agent over the session.
ENGINE_AGENT = "agent"

#: Every engine a watchdog may name.
ENGINES: tuple[str, ...] = (ENGINE_DECISION, ENGINE_AGENT)

#: The summed flag-option probability at which a decision watchdog flags.
DEFAULT_THRESHOLD = 0.5

#: Seconds one agent watchdog run may take.
DEFAULT_TIMEOUT_S = 90.0

#: Dollars one agent watchdog run may spend.
DEFAULT_MAX_COST_USD = 0.25

#: What a watchdog may be called: its name is a directory under ``watchdogs/``.
WATCHDOG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: The file a watchdog's instructions live in, inside its directory.
INSTRUCTIONS_FILENAME = "instructions.md"

#: The directory a watchdog's reference material lives in, inside its directory.
REFERENCES_DIRNAME = "references"

#: The hook that tells the web terminal when a turn starts and ends. Without it
#: a Stop could never confirm the session is idle.
TURN_STATE_HOOK = "turn-state"

_COMMON_KEYS = frozenset({"engine", "enabled"})
_DECISION_KEYS = frozenset({"threshold", "options"})
_AGENT_KEYS = frozenset({"tools", "model", "timeout_s", "max_cost_usd"})
_OPTION_KEYS = frozenset({"id", "text", "flag"})

#: What may not appear in a ``tools`` entry once ``.*`` reads as ``*``: a pattern
#: spelled with these could reach a write tool the check has no way to see.
_UNREADABLE_PATTERN_CHARS = frozenset(".|()[]{}+?^$\\")
_CLASSIFICATION_PATTERN_START = re.compile(r"[*.|()\[\]{}+?^$\\]")


@dataclass(frozen=True)
class VerdictOption:
    """One answer a decision watchdog may give."""

    id: str
    text: str
    flag: bool = False


@dataclass(frozen=True)
class WatchdogDefinition:
    """One watchdog, with its files already read.

    ``threshold`` is ``None`` on an agent watchdog; ``options`` is empty there,
    and ``tools`` and ``model`` are unset on a decision watchdog.
    """

    name: str
    engine: str
    enabled: bool = False
    instructions: str = ""
    options: tuple[VerdictOption, ...] = ()
    threshold: float | None = None
    references: tuple[Path, ...] = ()
    tools: tuple[str, ...] = ()
    model: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_cost_usd: float = DEFAULT_MAX_COST_USD

    @property
    def flag_threshold(self) -> float:
        """The score at or above which this watchdog flags; the default when it names none."""
        return self.threshold if self.threshold is not None else DEFAULT_THRESHOLD


def _is_number(value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _option_errors(label: str, options: Any) -> list[str]:
    """Check a decision watchdog's options, which must be able to say both answers."""
    if not isinstance(options, list):
        return [
            f"{label}.options must be a list of `{{id, text, flag}}` entries, with at least "
            f"one flag option and one non-flag option"
        ]

    errors: list[str] = []
    seen: set[str] = set()
    flags: list[bool] = []
    for index, option in enumerate(options):
        where = f"{label}.options[{index}]"
        if not isinstance(option, Mapping):
            errors.append(f"{where} must be a mapping with `id`, `text` and `flag`")
            continue
        unknown = sorted(str(key) for key in set(option) - _OPTION_KEYS)
        if unknown:
            errors.append(f"{where} has unknown keys: {', '.join(unknown)}")
        for key in ("id", "text"):
            value = option.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}.{key} must be a non-empty string")
        option_id = option.get("id")
        if isinstance(option_id, str):
            if option_id in seen:
                errors.append(f"{where}.id {option_id!r} is used twice")
            seen.add(option_id)
        flag = option.get("flag", False)
        if isinstance(flag, bool):
            flags.append(flag)
        else:
            errors.append(f"{where}.flag must be true or false (got {flag!r})")

    if True not in flags:
        errors.append(f"{label}.options needs a flag option (`flag: true`): it could never flag")
    if False not in flags:
        errors.append(
            f"{label}.options needs a non-flag option (`flag: false`): it could only flag"
        )
    return errors


def _is_unreadable_pattern(tool: str) -> bool:
    """Whether a ``tools`` entry is more than a name with one trailing wildcard."""
    wildcard = tool.replace(".*", "*")
    return bool(_UNREADABLE_PATTERN_CHARS & set(wildcard)) or "*" in wildcard[:-1]


def _engine_key_errors(label: str, entry: Mapping[Any, Any], engine: str) -> list[str]:
    """Check the keys and values only one engine reads."""
    errors: list[str] = []
    if engine == ENGINE_DECISION:
        foreign = _AGENT_KEYS & set(entry)
        errors.extend(_option_errors(label, entry.get("options")))
        threshold = entry.get("threshold", DEFAULT_THRESHOLD)
        if not _is_number(threshold) or not 0 < threshold < 1:
            errors.append(
                f"{label}.threshold must be between 0 and 1, exclusive (got {threshold!r})"
            )
    else:
        foreign = _DECISION_KEYS & set(entry)
        tools = entry.get("tools", [])
        if not isinstance(tools, list) or not all(isinstance(tool, str) and tool for tool in tools):
            errors.append(f"{label}.tools must be a list of tool names (got {tools!r})")
        else:
            errors.extend(
                f"{label}.tools entry {tool!r} is a pattern the write-tool check cannot read: "
                f"name the tool, or end it with `*` or `.*` as the wildcard"
                for tool in tools
                if _is_unreadable_pattern(tool)
            )
        model = entry.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            errors.append(f"{label}.model must be a model name or null (got {model!r})")
        for key, default in (
            ("timeout_s", DEFAULT_TIMEOUT_S),
            ("max_cost_usd", DEFAULT_MAX_COST_USD),
        ):
            value = entry.get(key, default)
            if not _is_number(value) or value <= 0:
                errors.append(f"{label}.{key} must be a positive number (got {value!r})")
    if foreign:
        errors.append(
            f"{label} is a {engine} watchdog and does not read: {', '.join(sorted(foreign))}"
        )
    return errors


def instructions_error(name: str, base_dir: Path) -> str | None:
    """Say what is wrong with a watchdog's instructions file, or ``None``."""
    label = f"{WATCHDOGS_BLOCK}.{name}"
    path = base_dir / WATCHDOGS_BLOCK / name / INSTRUCTIONS_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return (
            f"{label} has no readable {path}: add the file. A profile that inherits this "
            f"watchdog through `extends` does not inherit its files, so it adds its own copy"
        )
    if not text.strip():
        return f"{label} has an empty {path}: a watchdog with no instructions judges nothing"
    return None


def _definition_errors(
    name: Any, entry: Any, base_dir: Path, *, files_when_disabled: bool
) -> list[str]:
    """Every problem one watchdog has on its own, whatever profile surrounds it.

    ``files_when_disabled`` holds an entry that says ``enabled: false``, or nothing,
    to its files as well; any other ``enabled`` is held to them whatever its spelling.
    """
    if not isinstance(name, str) or not WATCHDOG_NAME_RE.fullmatch(name):
        return [
            f"{WATCHDOGS_BLOCK} name {name!r} is unsafe: a watchdog's name is a directory, so "
            f"use lowercase letters, digits, '-' and '_', starting with a letter or digit"
        ]
    label = f"{WATCHDOGS_BLOCK}.{name}"
    if not isinstance(entry, Mapping):
        return [f"{label} must be a mapping (got {type(entry).__name__})"]

    errors: list[str] = []
    unknown = set(entry) - _COMMON_KEYS - _DECISION_KEYS - _AGENT_KEYS
    if unknown:
        errors.append(f"{label} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    if not isinstance(entry.get("enabled", False), bool):
        errors.append(f"{label}.enabled must be true or false (got {entry['enabled']!r})")

    engine = entry.get("engine")
    if engine in ENGINES:
        errors.extend(_engine_key_errors(label, entry, engine))
    else:
        errors.append(f"{label}.engine must be one of {', '.join(ENGINES)} (got {engine!r})")

    if files_when_disabled or entry.get("enabled", False) is not False:
        instructions = instructions_error(name, base_dir)
        if instructions is not None:
            errors.append(instructions)
    return errors


def _block_items(raw: Any) -> list[tuple[Any, Any]]:
    """The block's entries; refuses anything but a mapping, reads nothing as empty."""
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise BuildProfileError(
            f"Profile '{WATCHDOGS_BLOCK}' must be a mapping of watchdog names "
            f"(got {type(raw).__name__})"
        )
    return list(raw.items())


def _matches_as_regex(pattern: str, tool: str) -> bool:
    try:
        return re.fullmatch(pattern, tool) is not None
    except re.error:
        return False


def _patterns_overlap(wildcard: str, write_tool: str) -> bool:
    """Whether a trailing-wildcard entry can match a tool a classification pattern matches.

    Each side accepts every name extending its literal prefix, so they can only
    meet when one prefix extends the other.
    """
    start = _CLASSIFICATION_PATTERN_START.search(write_tool)
    if start is None or not wildcard.endswith("*"):
        return False
    entry_prefix, classified_prefix = wildcard[:-1], write_tool[: start.start()]
    return entry_prefix.startswith(classified_prefix) or classified_prefix.startswith(entry_prefix)


def _grants_write(tool: str, write_tools: Iterable[str]) -> bool:
    """Whether a ``tools`` entry reaches a write tool, under any spelling of it.

    An entry reaches a write tool by naming it, by naming its whole server
    (``mcp__<server>``), or by a wildcard that matches it. A classification entry
    may itself name a whole server, as ``mcp__<server>``, or be a pattern, read
    both as the glob a permission rule is and as the regex a hook matcher is.
    :func:`_engine_key_errors` has already refused every other pattern spelling.
    """
    wildcard = tool.replace(".*", "*")
    for write_tool in write_tools:
        if (
            tool == write_tool
            or write_tool.startswith(f"{tool}__")
            or fnmatchcase(write_tool, wildcard)
            or (write_tool.count("__") == 1 and tool.startswith(f"{write_tool}__"))
            or fnmatchcase(tool, write_tool.replace(".*", "*"))
            or _matches_as_regex(write_tool, tool)
            or _patterns_overlap(wildcard, write_tool)
        ):
            return True
    return False


def watchdog_errors(
    raw: Any,
    *,
    base_dir: Path,
    write_tools: Iterable[str],
    hook_selection: Iterable[Any],
) -> list[str]:
    """List every problem in a ``watchdogs:`` block.

    Args:
        raw: The block itself, the name-keyed map; ``None`` reads as empty.
        base_dir: The directory holding ``watchdogs/<name>/``, read for the
            enabled entries only.
        write_tools: The write-tool classification an agent watchdog's ``tools``
            must stay clear of.
        hook_selection: The library hooks the profile selects, as
            :func:`~osprey.cli.profile_conventions.library_selection` answers it.

    Returns:
        The problems found, each naming its fix; empty when the block is clean.
    """
    try:
        items = _block_items(raw)
    except BuildProfileError as exc:
        return [str(exc)]

    write_tools = tuple(write_tools)
    turn_state = TURN_STATE_HOOK in tuple(hook_selection)
    errors: list[str] = []
    for name, entry in items:
        errors.extend(_definition_errors(name, entry, base_dir, files_when_disabled=False))
        if not isinstance(entry, Mapping):
            continue
        tools = entry.get("tools")
        if entry.get("engine") == ENGINE_AGENT and isinstance(tools, list):
            writes = [t for t in tools if isinstance(t, str) and _grants_write(t, write_tools)]
            if writes:
                errors.append(
                    f"{WATCHDOGS_BLOCK}.{name}.tools reaches write tools: "
                    f"{', '.join(writes)}. A watchdog reads and never acts; remove them"
                )
        if entry.get("enabled") is True and not turn_state:
            errors.append(
                f"{WATCHDOGS_BLOCK}.{name} is enabled but `hooks:` does not select "
                f"`{TURN_STATE_HOOK}`, so a Stop could never confirm the session is idle. "
                f"Add `{TURN_STATE_HOOK}` to `hooks:`, or set `enabled: false`"
            )
    return errors


def _references(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.rglob("*") if path.is_file()))


def _definition(name: str, entry: Mapping[str, Any], base_dir: Path) -> WatchdogDefinition:
    directory = base_dir / WATCHDOGS_BLOCK / name
    engine = entry["engine"]
    decision = engine == ENGINE_DECISION
    return WatchdogDefinition(
        name=name,
        engine=engine,
        enabled=entry.get("enabled", False),
        instructions=(directory / INSTRUCTIONS_FILENAME).read_text(encoding="utf-8"),
        options=tuple(
            VerdictOption(id=option["id"], text=option["text"], flag=option.get("flag", False))
            for option in entry.get("options", ())
        ),
        threshold=float(entry.get("threshold", DEFAULT_THRESHOLD)) if decision else None,
        references=_references(directory / REFERENCES_DIRNAME),
        tools=tuple(entry.get("tools", ())),
        model=entry.get("model"),
        timeout_s=float(entry.get("timeout_s", DEFAULT_TIMEOUT_S)),
        max_cost_usd=float(entry.get("max_cost_usd", DEFAULT_MAX_COST_USD)),
    )


def parse_watchdogs_block(raw: Any, *, base_dir: Path) -> tuple[WatchdogDefinition, ...]:
    """Parse a ``watchdogs:`` block into its definitions, in block order.

    Args:
        raw: The block itself, the name-keyed map; ``None`` reads as empty.
        base_dir: The directory holding ``watchdogs/<name>/``.

    Returns:
        One definition per entry, enabled or not.

    Raises:
        BuildProfileError: If the block is not a mapping, or any entry breaks a
            rule of its own (:func:`watchdog_errors` without the profile checks, and
            with a disabled entry held to its files too), every such problem in one
            message.
    """
    items = _block_items(raw)
    problems = [
        problem
        for name, entry in items
        for problem in _definition_errors(name, entry, base_dir, files_when_disabled=True)
    ]
    if problems:
        raise BuildProfileError(
            f"Profile '{WATCHDOGS_BLOCK}' block is invalid:\n  - " + "\n  - ".join(problems)
        )
    return tuple(_definition(name, entry, base_dir) for name, entry in items)
