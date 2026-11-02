#!/usr/bin/env python3
"""
---
name: Config Drift Guard
description: Warns at session start when config.yml has drifted from the rendered Claude Code artifacts
summary: Catch-all so a hand-edited config.yml (or raw `claude` launch) never silently runs stale safety settings
event: SessionStart
---

## Flow

```
SessionStart ──► stat config.yml & .claude/settings.json
                     │
                     ▼
   (a) config.yml mtime > settings.json mtime? ──► general drift
                     │
                     ▼
   (b) control_system.writes readable?  ── no ─► claim nothing about writes
                     │ yes
                     ▼
    allowed/never vs this render's kill-switch tools in deny? ──► precise drift
                     │
                     ▼
   drift?  ── yes ─► warn (stderr + additionalContext) ─► exit 0
           ── no  ─► silent ─────────────────────────────► exit 0
```

## Details

`config.yml` is a build-time input: safety-critical fields (notably the write
posture, which bakes this render's write tools into ``settings.json``'s
``permissions.deny``) only take effect once the artifacts are re-rendered by
``osprey build``. The web and CLI entry points auto-regenerate, but a
hand-edited config.yml launched via raw ``claude`` would otherwise run stale
settings with no signal. This hook is that signal.

The tools check (b) looks for come from this render's own
``.claude/hooks/hook_config.json`` — the ``write_tools`` list the writes-check
hook and the MCP audit middleware read, plus the ``facility_write_tools`` set
the ``never`` kill switch denies — not from a literal spelled here. A deployment
whose controls server is an ``extends`` clone denies
``mcp__<clone>__channel_write`` and nothing named ``controls``, so a hook probing
the framework literal claimed nothing while looking like it had checked. Tools
whose reads survive the kill switch (``mixed_read_write_tools``) and the
project's ``control_system.write_tools`` are dropped: neither is denied
outright, so neither says anything about the posture. An empty set abstains.

The posture check (b) reads is the tier, ``control_system.writes``, exactly as
the kill switch does (``tier_writes``): only the literals ``asked`` and
``allowed`` arm. The key is looked up as a direct child of the top-level
``control_system`` section, never section-blind, because a facility server's
``writes:`` list lives under ``claude_code.servers`` and says nothing about the
tier. A retired
``writes_enabled`` leaf, flat or in a connector block, is never read: the kill
switch does not read it either. Anything the regex cannot read for certain — no
tier key at all, a duplicated key, a value that is not a plain scalar —
abstains, and check (a) still covers such a config.

Warn-only by design — it never mutates artifacts and never blocks the session
(writes stay blocked until a proper regen, which is the safe direction). It is
stdlib-only and must NOT import osprey or PyYAML: under a raw ``claude`` launch
the project venv's interpreter is not guaranteed to be the one running this hook,
so YAML is matched with a regex and JSON is read with the stdlib ``json`` module.

Fails open on any IO/parse error — stays silent and exits 0.
"""

# A hook can be executed by whatever bare ``python3`` is on PATH — on macOS that
# is still 3.9, which parses no PEP 604 union in an evaluated annotation. Making
# annotations lazy keeps this module importable there; without it the hook dies
# at import and its whole check goes dark.
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

#: The section the tier lives in, and its keys (the kill switch's `tier_writes`).
_SECTION = "control_system"
_WRITES_LEAF = "writes"
_TIER_ARMING = ("asked", "allowed")

#: One block-mapping line, `<indent><key>: <rest>`. A key is a plain YAML word;
#: quoted or complex keys are not posture keys and are passed over.
_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_.-]+):(?:\s+(?P<rest>.*))?$")
#: A block-sequence entry, `<indent>- ...`. Keys inside one are never posture keys.
_LIST_LINE = re.compile(r"^(?P<indent> *)-(?:\s|$)")
#: A trailing YAML comment: `#` after whitespace.
_COMMENT = re.compile(r"\s+#.*$")
#: Whole-server hook matcher `mcp__<srv>__.*`: as permission syntax it is `mcp__<srv>`.
_WHOLE_SERVER = re.compile(r"^(?P<server>mcp__[A-Za-z0-9_-]+?)__\.\*$")
#: Suffix marking a `mcp__<server>__.*` entry in `write_tools` — a whole server
#: gated by the writes-check hook, whose tool names are unknown at render time.
#: Such an entry is not itself in `permissions.deny`; the kill switch denies the
#: server as `mcp__<server>` instead, which `facility_write_tools` supplies.
_MATCHER_WILDCARD = ".*"

#: Read when `hook_config.json` cannot be, so this check keeps working on a
#: render that predates the file. It is the framework's own controls tool: the
#: literal this hook used to be written against.
_FALLBACK_WRITE_TOOLS = ("mcp__controls__channel_write",)

_REGEN_HINT = "run 'osprey build' and restart for changes to take effect"


def _control_system_keys(text: str) -> dict[tuple[str, ...], list[str]]:
    """Every block-mapping key under the top-level ``control_system`` section.

    Maps each key path (``("control_system", "connector", "epics",
    "prefix")``) to the raw value text of every line that spells it, in
    file order, so a duplicated key shows up as two values. A line-level
    indentation walk rather than a YAML parse (this hook may run without
    PyYAML): keys inside a block sequence or a block scalar are never recorded,
    a tab-indented or otherwise unreadable line hides what nests under it, and
    a key outside ``control_system`` — a facility server's ``writes:`` list
    under ``claude_code.servers`` included — is never a posture key.
    """
    found: dict[tuple[str, ...], list[str]] = {}
    stack: list[tuple[int, str | None]] = []
    block_scalar_indent: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block_scalar_indent is not None:
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        match = None if line[indent : indent + 1] == "\t" else _KEY_LINE.match(line.rstrip())
        if match is None or _LIST_LINE.match(line):
            # A sequence entry, a tab, a quoted key, a continuation: whatever
            # nests under it is not a mapping path this walk can name.
            stack.append((indent, None))
            continue
        key = match.group("key")
        rest = _COMMENT.sub("", match.group("rest") or "").strip()
        path = tuple(k for _, k in stack) + (key,)
        stack.append((indent, key))
        if rest[:1] in ("|", ">"):
            block_scalar_indent = indent
        if path[0] != _SECTION or any(k is None for k in path):
            continue
        found.setdefault(path, []).append(rest)  # type: ignore[arg-type]
    return found


def _plain_scalar(raw: str) -> str | None:
    """A value's string, or None when a regex cannot know what YAML makes of it.

    An empty value (a nested block), a flow collection, an alias, anchor or tag,
    a block scalar and an unexpanded ``${VAR}`` all abstain. Matching quotes
    are stripped: ``"asked"`` is the same string as ``asked``.
    """
    if not raw or raw[0] in "[{*&!|>" or "${" in raw:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    if raw[0] in ("'", '"'):
        return None
    return raw


def _tier_in_config(config_path: Path) -> tuple[bool, str] | None:
    """Whether config.yml's tier allows writes, and the key that says so.

    Returns ``(allows_writes, "<dotted key>: <value>")``, or None when the
    answer is undeterminable. Mirrors the kill switch's ``tier_writes``:
    ``control_system.writes`` decides when present, and only the literals
    ``asked`` and ``allowed`` arm: this is the raw scalar, so ``on`` and ``true``
    stay on the never side. A retired ``writes_enabled`` leaf is never read. No
    tier key, a duplicated key, or a value the regex cannot read abstains: for a
    warn-only signal a wrong warning is worse than none.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    keys = _control_system_keys(text)

    tier = keys.get((_SECTION, _WRITES_LEAF))
    if tier is None or len(tier) != 1:
        return None
    value = _plain_scalar(tier[0])
    if value is None:
        return None
    return value in _TIER_ARMING, f"{_SECTION}.{_WRITES_LEAF}: {value}"


def _kill_switch_tools(hook_config_path: Path) -> tuple[str, ...]:
    """The tools the ``never`` kill switch renders into ``permissions.deny``.

    Read from this render's own ``hook_config.json`` rather than spelled here,
    because the answer is per deployment: an ``extends`` clone of the controls
    server contributes ``mcp__<clone>__channel_write``, and a facility server
    contributes what ``facility_write_tools`` names. A hook that probes one
    framework literal claims nothing about either — and on a deployment whose
    only write tool is a clone it claimed nothing at all while looking like it
    had checked.

    Two sources, the same two the kill switch denies from:

    * ``write_tools`` — every writes-check-gated matcher, minus three kinds of
      entry the kill switch never denies as spelled:

      * wildcard matchers — they name no tool that could be in ``permissions.deny``;
      * ``mixed_read_write_tools`` — tools whose reads stay available with writes
        off, so they are not denied outright and their absence from ``deny`` says
        nothing about the posture;
      * ``control_system_write_tools`` — tools the project appended via
        ``control_system.write_tools``, gated per call rather than by the kill
        switch, so they never enter ``deny`` either. Read with ``.get``, so a
        render from before the key existed behaves as it did then.

    * ``facility_write_tools`` — the facility write set, every entry denied,
      a whole-server ``mcp__<srv>__.*`` as the permission rule ``mcp__<srv>``.
      Read with ``.get`` too: a render from before the key existed carries its
      facility exact names in ``write_tools`` already.

    An unreadable or malformed file falls back to the framework's own controls
    tool, which is what this check was written against before there was a file
    to read. An empty result means "claim nothing", which the caller handles.
    """
    try:
        data = json.loads(hook_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _FALLBACK_WRITE_TOOLS
    if not isinstance(data, dict):
        return _FALLBACK_WRITE_TOOLS
    tools = data.get("write_tools")
    if not isinstance(tools, list):
        return _FALLBACK_WRITE_TOOLS
    excluded: set[str] = set()
    for key in ("mixed_read_write_tools", "control_system_write_tools"):
        value = data.get(key)
        if isinstance(value, list):
            excluded.update(t for t in value if isinstance(t, str))
    result: list[str] = []
    for tool in tools:
        if (
            isinstance(tool, str)
            and not tool.endswith(_MATCHER_WILDCARD)
            and tool not in excluded
            and tool not in result
        ):
            result.append(tool)
    facility = data.get("facility_write_tools")
    for matcher in facility if isinstance(facility, list) else ():
        if not isinstance(matcher, str):
            continue
        whole = _WHOLE_SERVER.match(matcher)
        rule = whole.group("server") if whole else matcher
        if rule and not rule.endswith(_MATCHER_WILDCARD) and rule not in result:
            result.append(rule)
    return tuple(result)


def _write_tools_denied(settings_path: Path, tools: tuple[str, ...]) -> bool | None:
    """Whether *tools* are in permissions.deny (None if undeterminable).

    All-or-nothing on purpose: the kill switch renders the whole set together,
    so a partial overlap is not a posture — it is a hand-edited settings.json,
    and check (a) is the signal for that.
    """
    if not tools:
        return None
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    perms = data.get("permissions")
    if not isinstance(perms, dict):  # e.g. hand-edited {"permissions": null}
        return None
    deny = perms.get("deny", [])
    if not isinstance(deny, list):
        return None
    denied = {tool for tool in tools if tool in deny}
    if not denied:
        return False
    if len(denied) == len(tools):
        return True
    return None


def _detect_drift(config_path: Path, settings_path: Path, hook_config_path: Path) -> str | None:
    """Return a drift warning message, or None when artifacts look in sync."""
    if not config_path.exists() or not settings_path.exists():
        return None

    # (b) Targeted, high-signal check: does the kill-switch match the config?
    tier = _tier_in_config(config_path)
    denied = _write_tools_denied(settings_path, _kill_switch_tools(hook_config_path))
    if tier is not None and denied is not None:
        allows_writes, stated = tier
        if allows_writes and denied:
            return (
                f"config.yml sets {stated}, a tier that allows writes, but the agent "
                f"config still blocks writes — {_REGEN_HINT}."
            )
        if not allows_writes and not denied:
            return (
                f"config.yml sets {stated}, a tier that never writes, but the agent "
                f"config still permits writes — {_REGEN_HINT}."
            )

    # (a) General signal: config edited more recently than the rendered artifacts.
    try:
        if config_path.stat().st_mtime > settings_path.stat().st_mtime:
            return f"config.yml changed since the agent config was last generated — {_REGEN_HINT}."
    except OSError:
        return None

    return None


def main() -> int:
    try:
        project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
        config_path = project_dir / "config.yml"
        settings_path = project_dir / ".claude" / "settings.json"
        hook_config_path = project_dir / ".claude" / "hooks" / "hook_config.json"

        message = _detect_drift(config_path, settings_path, hook_config_path)
        if not message:
            return 0

        warning = f"⚠ {message}"
        # Visible in the session transcript ...
        sys.stderr.write(warning + "\n")
        # ... and injected as context so the agent can relay it to the operator.
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": warning,
                    }
                }
            )
        )
        return 0
    except Exception:
        # Last-resort fail-open: never block a session start.
        return 0


if __name__ == "__main__":
    sys.exit(main())
