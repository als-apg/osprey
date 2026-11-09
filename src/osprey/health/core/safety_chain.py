"""Core ``safety_chain`` health category.

Reports on the machinery that stands between the agent and a hardware write —
the hooks, the write switch, the limits database, the watchdogs and the
execution sandbox — read from the render and the deployment's state files, the
same files the web terminal's routes read. Nothing here writes, and nothing is
asked over HTTP. A row about a channel *value* does not belong here: this is
the chain's own state, not the machine's.

The factory takes ``cwd``, the render directory (the one holding the rendered
``config.yml``, ``.claude/`` and the limits file). The deployment's repo root is
the config's ``project_root`` when it names a directory here, else the render's
parent (:func:`~osprey_connectors.workspace.repo_root_for_config`).

Rows, all in category ``safety_chain``:

* ``hooks.osprey_writes_check``, ``hooks.osprey_limits``,
  ``hooks.osprey_approval`` — the hook is registered under ``PreToolUse`` in
  ``.claude/settings.json`` and its file in ``.claude/hooks/`` equals the
  framework template the build copies (sha256). The render's write tools are
  ``write_tools`` in ``.claude/hooks/hook_config.json``, the build's own list:
  ``writes_check`` and ``approval`` must cover every one of them, ``limits``
  at least one. Not registered (or missing on a write tool) and a missing hook
  file are ``error``; a file whose sha differs from the template is
  ``warning`` with fix ``osprey build``. A render with no write tools skips the
  three rows ("no write tools in this render"); one with no ``.claude/``
  skips them ("not rendered"); an unreadable ``settings.json`` is ``error``,
  and so is a missing or unreadable ``hook_config.json`` (fail closed).
* ``hooks.order`` — in every ``PreToolUse`` entry the three run in the order
  ``writes_check → limits → approval``; any other order is ``error``.
* ``write_switch.<login>`` — per login (``modules.web_terminals.users`` when
  the multi-user stack is enabled, else this process's acting identity): the
  positions on the login's control-context record
  (:func:`~osprey_connectors.control_context.switch_position`), capped by the
  login's tier ceiling (the tier render's ``control_system.writes``, or this
  config's on a single-user deployment). Informational (``ok``); an unreadable
  record is ``warning``.
* ``write_authorizations`` — the number of write authorizations in force
  across those logins' ``authorizations`` directories (``ok``).
* ``limits.parse`` — the limits database at
  ``control_system.limits_checking.database_path``, loaded by the connector's
  own loader: ``ok`` with the channel count; a file that does not parse is
  ``error``; fail closed — a missing or unconfigured file while writes are on
  is ``error``, and ``skip`` ("not configured") only while writes are
  ``never``. Limits checking switched off while writes are on is ``warning``.
* ``limits.age`` — the file's age and mtime, when the file exists (``ok``).
* ``watchdog.<name>`` — one ``ok`` row per enabled watchdog the build runs.
* ``watchdog.flags_open`` — open hold flags across the session-hold store:
  ``ok`` "0 · no hold", ``warning`` when any is open, ``skip`` when no watchdog
  is configured, ``error`` when the rendered block is invalid.
* ``exec_sandbox.guards`` — the execution wrapper, guards included, is built
  for both modes and compiles (``error`` otherwise).
* ``exec_sandbox.timeout`` — ``python_executor.execution_timeout_seconds``
  through the executor's own validator; an invalid value is ``error``.

A part that raises is isolated to one ``error`` row named after it
(``hooks``, ``write_switch``, ``limits``, ``watchdog``, ``exec_sandbox``); the
callable never raises. With no config at all the category contributes no rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from osprey.deployment.web_terminals.roster import authored_people
from osprey.health.models import CheckResult, Status
from osprey.utils.workspace import session_hold_dir
from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import acting_identity
from osprey_connectors.types import tier_writes
from osprey_connectors.workspace import (
    agent_data_base_dir,
    anchored_path,
    repo_root_for_config,
)

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime
    from osprey.services.watchdog.config import WatchdogDefinition

CATEGORY = "safety_chain"

#: The write chain, in the order it must run.
CHAIN = ("osprey_writes_check", "osprey_limits", "osprey_approval")

#: Hooks that must cover every write tool; ``osprey_limits`` guards the direct
#: channel write only, so it is required on at least one.
_EVERY_WRITE_TOOL = ("osprey_writes_check", "osprey_approval")

_ORDER_TEXT = "writes_check → limits → approval"
_FIX_BUILD = "osprey build"
_MODES = ("readonly", "readwrite")


def template_hooks_dir() -> Path:
    """The framework's hook templates, which the build copies into ``.claude/hooks/``."""
    import osprey.templates

    return Path(osprey.templates.__file__).parent / "claude_code" / "claude" / "hooks"


def safety_chain(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature
    *,
    cwd: Path | str | None = None,
) -> CategoryCallable:
    """Build the ``safety_chain`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Health runtime. Unused — every row reads files.
        cwd: The render directory; ``None`` uses the process working directory.

    Returns:
        A no-argument sync callable returning the category's rows.
    """
    cfg = config
    render = Path(cwd) if cwd is not None else Path.cwd()

    def _run() -> list[CheckResult]:
        if not cfg:
            return []
        repo = _repo_root(cfg, render)
        parts: tuple[tuple[str, Callable[[], list[CheckResult]]], ...] = (
            ("hooks", lambda: _hook_rows(render)),
            ("write_switch", lambda: _switch_rows(cfg, repo)),
            ("limits", lambda: _limits_rows(cfg, render)),
            ("watchdog", lambda: _watchdog_rows(cfg, render, repo)),
            ("exec_sandbox", lambda: _sandbox_rows(cfg)),
        )
        rows: list[CheckResult] = []
        for name, part in parts:
            try:
                rows.extend(part())
            except Exception as exc:  # isolation: one part never sinks the rest
                rows.append(
                    _row(
                        name,
                        Status.ERROR,
                        f"{name} check raised {type(exc).__name__}",
                        f"check raised {type(exc).__name__}",
                        details=str(exc),
                    )
                )
        return rows

    return _run


def _row(name: str, status: Status, message: str, value: str, **kw: Any) -> CheckResult:
    return CheckResult(name, CATEGORY, status, message, value=value, **kw)


def _repo_root(cfg: Mapping[str, Any], render: Path) -> Path:
    declared = cfg.get("project_root")
    if isinstance(declared, str) and declared and Path(declared).is_dir():
        return Path(declared)
    return repo_root_for_config(render / "config.yml")


# -- hooks ---------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches(matcher: Any, tool: str) -> bool:
    if not isinstance(matcher, str):
        return False
    try:
        return re.fullmatch(matcher, tool) is not None
    except re.error:
        return matcher == tool


def _chain_hooks(entry: Mapping[str, Any]) -> list[str]:
    """The chain hooks one ``PreToolUse`` entry runs, in its order."""
    found: list[str] = []
    for hook in entry.get("hooks") or []:
        command = hook.get("command") if isinstance(hook, Mapping) else None
        if not isinstance(command, str):
            continue
        found.extend(name for name in CHAIN if re.search(rf"\b{name}\.py\b", command))
    return found


def _hook_rows(render: Path) -> list[CheckResult]:
    claude = render / ".claude"
    if not claude.is_dir():
        return [
            _row(f"hooks.{h}", Status.SKIP, f"{h}: no rendered .claude/ here", "not rendered")
            for h in CHAIN
        ]
    try:
        settings = json.loads((claude / "settings.json").read_text(encoding="utf-8"))
        pre_tool_use = settings.get("hooks", {}).get("PreToolUse", []) or []
        entries = [e for e in pre_tool_use if isinstance(e, Mapping)]
    except (OSError, ValueError, AttributeError) as exc:
        return [
            _row(
                f"hooks.{h}",
                Status.ERROR,
                f"{h}: .claude/settings.json cannot be read",
                "settings.json unreadable",
                details=str(exc),
                fix=_FIX_BUILD,
            )
            for h in CHAIN
        ]
    try:
        write_tools = _write_tools(claude / "hooks" / "hook_config.json")
    except (OSError, ValueError, AttributeError) as exc:
        # The hooks' own contract: without it nothing says which tools write,
        # so the chain cannot be shown to cover them. Fail closed.
        return [
            _row(
                f"hooks.{h}",
                Status.ERROR,
                f"{h}: .claude/hooks/hook_config.json cannot be read",
                "hook_config.json unreadable",
                details=str(exc),
                fix=_FIX_BUILD,
            )
            for h in CHAIN
        ]
    if not write_tools:
        return [
            _row(
                f"hooks.{h}",
                Status.SKIP,
                f"{h}: the render exposes no write tool",
                "no write tools in this render",
            )
            for h in CHAIN
        ]

    covered: dict[str, set[str]] = {h: set() for h in CHAIN}
    for tool in write_tools:
        for entry in entries:
            if _matches(entry.get("matcher"), tool):
                for hook in _chain_hooks(entry):
                    covered[hook].add(tool)

    rows = [_hook_row(claude, hook, covered[hook], write_tools) for hook in CHAIN]

    misordered = [
        str(entry.get("matcher"))
        for entry in entries
        if (found := _chain_hooks(entry)) != sorted(found, key=CHAIN.index)
    ]
    if misordered:
        rows.append(
            _row(
                "hooks.order",
                Status.ERROR,
                f"write chain out of order on {', '.join(misordered)}; it must run {_ORDER_TEXT}",
                f"out of order on {', '.join(misordered)}",
                fix=_FIX_BUILD,
            )
        )
    else:
        rows.append(_row("hooks.order", Status.OK, f"write chain runs {_ORDER_TEXT}", _ORDER_TEXT))
    return rows


def _write_tools(hook_config: Path) -> list[str]:
    """The render's write tools; raises when ``hook_config.json`` is missing or malformed."""
    tools = json.loads(hook_config.read_text(encoding="utf-8")).get("write_tools")
    return [t for t in tools or [] if isinstance(t, str) and t]


def _hook_row(claude: Path, hook: str, covered: set[str], write_tools: list[str]) -> CheckResult:
    name = f"hooks.{hook}"
    missing = [t for t in write_tools if t not in covered] if hook in _EVERY_WRITE_TOOL else []
    if not covered:
        return _row(
            name,
            Status.ERROR,
            f"{hook} is not registered under PreToolUse for any write tool",
            "not registered",
            fix=_FIX_BUILD,
        )
    if missing:
        return _row(
            name,
            Status.ERROR,
            f"{hook} is not registered for {', '.join(missing)}",
            f"not registered for {', '.join(missing)}",
            fix=_FIX_BUILD,
        )
    rendered = claude / "hooks" / f"{hook}.py"
    if not rendered.is_file():
        return _row(
            name,
            Status.ERROR,
            f"{hook} is registered but .claude/hooks/{hook}.py is missing",
            "registered · file missing",
            fix=_FIX_BUILD,
        )
    template = template_hooks_dir() / f"{hook}.py"
    if template.is_file() and _sha256(template) != _sha256(rendered):
        return _row(
            name,
            Status.WARNING,
            f"{hook} differs from the framework template the build writes",
            "registered · sha differs from render",
            fix=_FIX_BUILD,
        )
    return _row(
        name,
        Status.OK,
        f"{hook} registered for {len(covered)} write tool(s); file matches the template",
        "registered · sha = render",
    )


# -- write switch + authorizations ---------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _logins(cfg: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """``(login, tier)`` per login; the acting identity alone without the stack."""
    wt = _as_dict(_as_dict(cfg.get("modules")).get("web_terminals"))
    if wt.get("enabled"):
        default = wt.get("default_tier")
        out: list[tuple[str, str | None]] = []
        # As authored: the normalizer projects the per-person tier away.
        for user in authored_people(wt):
            if isinstance(user, Mapping) and isinstance(user.get("name"), str) and user["name"]:
                tier = user.get("tier") or default
                out.append((user["name"], tier if isinstance(tier, str) else None))
        return out
    return [(acting_identity(), None)]


def _agent_data_root(cfg: Mapping[str, Any], repo: Path) -> Path:
    stamped = posture_store.stamped_agent_data_root()
    if stamped is not None:
        return stamped
    return anchored_path(agent_data_base_dir(cfg), repo)


def _login_dir(cfg: Mapping[str, Any], repo: Path, login: str) -> Path | None:
    tree = os.environ.get(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "").strip()
    if tree and Path(tree).is_absolute():
        return Path(tree) / login
    bound = posture_store.bound_state_dir()
    if bound is not None:
        return bound if login == acting_identity() else None
    return _agent_data_root(cfg, repo) / posture_store.STATE_DIR_NAME / login


def _ceiling(cfg: Mapping[str, Any], repo: Path, tier: str | None) -> str | None:
    """The login's write ceiling: its tier render's ``writes``, else this config's."""
    if tier is None:
        return tier_writes(cfg.get("control_system"))
    wt = _as_dict(_as_dict(cfg.get("modules")).get("web_terminals"))
    render = _as_dict(_as_dict(wt.get("renders")).get(tier))
    rel = render.get("project_path")
    if not isinstance(rel, str) or not rel:
        return None
    try:
        tier_cfg = yaml.safe_load((anchored_path(rel, repo) / "config.yml").read_text())
    except (OSError, yaml.YAMLError):
        return None
    return tier_writes(_as_dict(tier_cfg).get("control_system"))


_RANK = {"off": 0, "never": 0, "asked": 1, "allowed": 2}


def _switch_rows(cfg: Mapping[str, Any], repo: Path) -> list[CheckResult]:
    rows: list[CheckResult] = []
    in_force = 0
    now_ns = time.time_ns()
    for login, tier in _logins(cfg):
        directory = _login_dir(cfg, repo, login)
        rows.append(_switch_row(cfg, repo, login, tier, directory))
        if directory is not None:
            from osprey_connectors.write_authorization import list_grants

            try:
                grants, _ = list_grants(directory / "authorizations")
            except OSError:
                continue
            in_force += sum(1 for g in grants if g.live_at(now_ns))
    rows.append(
        _row(
            "write_authorizations",
            Status.OK,
            f"{in_force} write authorization(s) in force",
            str(in_force),
        )
    )
    return rows


def _switch_row(
    cfg: Mapping[str, Any], repo: Path, login: str, tier: str | None, directory: Path | None
) -> CheckResult:
    name = f"write_switch.{login}"
    ceiling = _ceiling(cfg, repo, tier)
    if ceiling == "never":
        why = f"tier {tier}" if tier else "writes: never"
        return _row(name, Status.OK, f"{login}: writes never ({why})", f"never ({why})")
    if directory is None:
        return _row(name, Status.SKIP, f"{login}: control record not visible here", "not visible")
    record, status = control_context.read_record_status(path=directory / "control_context.json")
    if status == "unreadable":
        return _row(
            name,
            Status.WARNING,
            f"{login}: control-context record cannot be read; writes refuse",
            "record unreadable",
        )
    armed = (
        {}
        if record is None
        else {t: control_context.switch_position(record, t) for t in record.switch}
    )
    armed = {t: p for t, p in armed.items() if p != "off"}
    if not armed:
        return _row(name, Status.OK, f"{login}: write switch off on every target", "off")
    parts = []
    for target, position in sorted(armed.items()):
        text = f"{position} on {target}"
        if ceiling in _RANK and _RANK[position] > _RANK[ceiling]:
            text += f" · capped at {ceiling}"
        parts.append(text)
    value = ", ".join(parts)
    return _row(name, Status.OK, f"{login}: {value}", value)


# -- limits --------------------------------------------------------------------


def _fmt_age(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(0, int(seconds // 60))} m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{int(seconds // 86400)} d"


def _limits_rows(cfg: Mapping[str, Any], render: Path) -> list[CheckResult]:
    from osprey_connectors.control_system import (
        LIMITS_DATABASE_CONFIG_KEY,
        LimitsValidator,
        mapping_config_lookup,
    )

    section = _as_dict(cfg.get("control_system"))
    writes_on = tier_writes(section) != "never"
    checking = _as_dict(section.get("limits_checking"))
    if checking.get("enabled") is False:
        if writes_on:
            return [
                _row(
                    "limits.parse",
                    Status.WARNING,
                    "limits checking is off while writes are on",
                    "limits checking off",
                )
            ]
        return [_row("limits.parse", Status.SKIP, "limits checking is off", "not configured")]

    lookup = mapping_config_lookup(dict(cfg))
    config_path = str(render / "config.yml")
    db_path = LimitsValidator.configured_database_path(
        config_lookup=lookup, config_path=config_path
    )
    if db_path is None or not Path(db_path).exists():
        if not writes_on:
            return [
                _row(
                    "limits.parse",
                    Status.SKIP,
                    "no limits database; writes never",
                    "not configured",
                )
            ]
        where = db_path or f"{LIMITS_DATABASE_CONFIG_KEY} (unset)"
        return [
            _row(
                "limits.parse",
                Status.ERROR,
                f"no limits database at {where}; every write is refused",
                "file missing" if db_path else "not configured",
            )
        ]

    loaded, reason = LimitsValidator.load_configured_database(
        config_lookup=lookup, config_path=config_path
    )
    if loaded is None:
        parse = _row(
            "limits.parse",
            Status.ERROR,
            f"limits database does not load; every write is refused: {reason}",
            "parse error",
            details=reason or "",
        )
    else:
        count = len(loaded[0])
        parse = _row(
            "limits.parse",
            Status.OK,
            f"{count:,} channels loaded from {db_path}",
            f"{count:,} channels · 0 errors",
        )
    mtime = Path(db_path).stat().st_mtime
    stamp = datetime.fromtimestamp(mtime, UTC).strftime("%H:%MZ")
    age = _row(
        "limits.age",
        Status.OK,
        f"limits database last written {_fmt_age(time.time() - mtime)} ago",
        f"{_fmt_age(time.time() - mtime)} · mtime {stamp}",
    )
    return [parse, age]


# -- watchdogs -------------------------------------------------------------------


def _load_watchdogs(cfg: Mapping[str, Any], render: Path) -> tuple[WatchdogDefinition, ...]:
    from osprey.services.watchdog.definitions import load_watchdogs

    return load_watchdogs(cfg, render)


def _watchdog_rows(cfg: Mapping[str, Any], render: Path, repo: Path) -> list[CheckResult]:
    try:
        definitions = _load_watchdogs(cfg, render)
    except Exception as exc:  # the rendered block breaks a rule
        return [
            _row(
                "watchdog.flags_open",
                Status.ERROR,
                "the rendered watchdogs block is invalid; no watchdog runs",
                "watchdogs block invalid",
                details=str(exc),
                fix=_FIX_BUILD,
            )
        ]
    if not definitions:
        return [
            _row(
                "watchdog.flags_open",
                Status.SKIP,
                "no watchdog is configured",
                "no watchdogs configured",
            )
        ]
    rows = [
        _row(
            f"watchdog.{d.name}",
            Status.OK,
            f"watchdog {d.name} enabled ({d.engine} engine)",
            f"enabled · {d.engine}",
        )
        for d in definitions
    ]
    rows.append(_flags_row(session_hold_dir(repo)))
    return rows


def _flags_row(root: Path) -> CheckResult:
    from osprey_connectors.session_hold import (
        UNREADABLE_FLAG_ID,
        is_safe_session_key,
        open_flags,
    )

    flags = 0
    held = 0
    unreadable: list[str] = []
    try:
        keys = sorted(p.name for p in root.iterdir() if p.is_dir())
    except FileNotFoundError:
        keys = []
    for key in keys:
        if not is_safe_session_key(key):
            continue
        found = open_flags(key, root=root)
        if any(f.flag_id == UNREADABLE_FLAG_ID for f in found):
            # Still held (fail closed), but there is no operator question to
            # answer: the record itself is the fault.
            unreadable.append(key)
            continue
        flags += len(found)
        held += 1 if found else 0
    if not flags and not unreadable:
        return _row("watchdog.flags_open", Status.OK, "no open hold flag", "0 · no hold")
    parts: list[str] = []
    noun = "session" if held == 1 else "sessions"
    if flags:
        parts.append(f"{flags} open · {held} {noun} held")
    if unreadable:
        parts.append(
            f"{len(unreadable)} hold record{'s' if len(unreadable) != 1 else ''} unreadable"
        )
    message = (
        f"{flags} open hold flag(s); {held} {noun} held until an operator answers"
        if flags
        else "a session's hold record is missing or unreadable, so the session is held"
    )
    if flags and unreadable:
        message += f"; {len(unreadable)} more held on an unreadable record"
    return _row(
        "watchdog.flags_open",
        Status.WARNING,
        message,
        " · ".join(parts),
        details="\n".join(str(root / key / "record.json") for key in unreadable),
    )


# -- exec sandbox -----------------------------------------------------------------


def _wrapper_source(mode: str) -> str:
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    return ExecutionWrapper(execution_mode=mode).create_wrapper("pass")


def _sandbox_rows(cfg: Mapping[str, Any]) -> list[CheckResult]:
    built = 0
    errors: list[str] = []
    for mode in _MODES:
        try:
            compile(_wrapper_source(mode), f"<wrapper:{mode}>", "exec")
            built += 1
        except Exception as exc:  # a guard that fails to render or compile
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    guards = _row(
        "exec_sandbox.guards",
        Status.OK if built == len(_MODES) else Status.ERROR,
        f"execution wrapper with its guards builds and compiles in {built} of {len(_MODES)} modes",
        f"{built} / {len(_MODES)} modes compile",
        details="\n".join(errors),
    )

    from osprey.mcp_server.python_executor.executor import execution_timeout_seconds
    from osprey.utils.execution_method import DEFAULT_EXECUTION_TIMEOUT_SECONDS

    raw = _as_dict(cfg.get("python_executor")).get(
        "execution_timeout_seconds", DEFAULT_EXECUTION_TIMEOUT_SECONDS
    )
    try:
        seconds = execution_timeout_seconds(raw)
    except ValueError as exc:
        timeout = _row(
            "exec_sandbox.timeout",
            Status.ERROR,
            str(exc),
            f"invalid: {raw!r}",
        )
    else:
        shown = f"{seconds:g} s"
        timeout = _row("exec_sandbox.timeout", Status.OK, f"execution timeout {shown}", shown)
    return [guards, timeout]
