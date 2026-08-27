#!/usr/bin/env python3
"""
---
name: Memory Write Guard
description: Gates every file-writing tool — Write/MultiEdit to Claude memory files, NotebookEdit to the agent-data artifacts tree
summary: Restricts Write/MultiEdit to the Claude memory directory and NotebookEdit to agent-data artifacts
event: PreToolUse
tools: Write|MultiEdit|NotebookEdit
safety_layer: 0
wiring: standalone
timeout: 5
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         tool_name is one of      ──NO──► EXIT (no opinion)
         Write/MultiEdit/NotebookEdit
              │
             YES
              │
              ├──── Write / MultiEdit ────┐
              │                           ▼
              │                  Resolve file_path
              │                           │
              │                           ▼
              │                  Inside Claude memory  ──YES──► ALLOW
              │                  directory? (.md only)
              │                           │
              │                          NO
              │                           ▼
              │                  DENY: not a memory file
              │
              └──── NotebookEdit ─────────┐
                                          ▼
                                 Resolve notebook_path
                                          │
                                          ▼
                                 Under <agent_data_root>/  ──YES──► ALLOW
                                 artifacts/ ?
                                          │
                                         NO
                                          ▼
                                 DENY: not an artifact notebook
```

## Details

Gate for every tool that can put bytes on disk. Claude Code ships three:
``Write``, ``MultiEdit`` and ``NotebookEdit``. Leaving any of them ungated is
the whole failure this hook exists to prevent — a build that gated only
``Write`` still let the agent create arbitrary files through the other two.

``Write`` and ``MultiEdit`` are held to the Claude Code memory directory for
the current project (``$CLAUDE_CONFIG_DIR/projects/<encoded>/memory/``, with
``~/.claude`` as the root when the variable is unset), and only for
``.md`` files directly inside it. This lets the agent use Claude Code's native
memory system while preventing arbitrary file creation. The memory gallery
frontend reads from the same directory, so saved memories appear in the UI
automatically.

``NotebookEdit`` is held to ``<agent_data_root>/artifacts/`` instead, mirroring
the scoped ``NotebookEdit(<agent_data_root>/artifacts/**)`` allow rule that
``settings.json.j2`` renders. Notebooks are the agent's own output and land in
the artifact tree the gallery serves; holding them to the memory directory
would deny the agent its own artifacts, and holding them to nothing at all
would reopen arbitrary writes under a different tool name.

All other targets are denied — the agent never sees a user prompt for a write
this guard does not recognise.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_REFUSED,
    emit_audit,
    get_hook_input,
    get_project_dir,
    get_repo_root,
    load_osprey_config,
    log_hook,
)

# Keep this regex in sync with
# ``osprey.agent_runner.project_paths.encode_claude_project_path``. Duplicated inline
# because this hook runs in user projects where the ``osprey`` package may
# not be importable at hook-execution time.
_CLAUDE_PROJECT_DIR_NORMALIZE = re.compile(r"[^A-Za-z0-9-]")

#: Tools held to the Claude memory directory. ``MultiEdit`` writes the same
#: bytes ``Write`` does, so the two share one rule; splitting them is how the
#: guard came to cover only half the surface it names.
_MEMORY_TOOLS = frozenset({"Write", "MultiEdit"})

#: The one tool held to the artifact tree instead.
_NOTEBOOK_TOOL = "NotebookEdit"

#: Every tool this hook has an opinion about. Anything else exits silently.
#: Must stay in step with the ``tools:`` matcher in the frontmatter above —
#: that string becomes the ``PreToolUse`` matcher verbatim, so a tool listed
#: there and missing here would reach the hook and be waved through.
_GUARDED_TOOLS = frozenset(_MEMORY_TOOLS | {_NOTEBOOK_TOOL})

#: Where each tool names its target. ``NotebookEdit`` uses ``notebook_path``;
#: the others use ``file_path``.
_PATH_KEYS = {
    "Write": ("file_path",),
    "MultiEdit": ("file_path",),
    _NOTEBOOK_TOOL: ("notebook_path", "file_path"),
}

#: Subdirectory of the agent-data root that holds the agent's own outputs.
#: This is the ``artifacts`` in ``NotebookEdit(<agent_data_root>/artifacts/**)``
#: as rendered by ``settings.json.j2``.
_ARTIFACTS_SUBDIR = "artifacts"

# The framework DEFAULT agent-data root, imported rather than spelled out here
# so the two cannot drift apart. Only the default is imported: a project that
# overrides ``agent_data.base_dir`` is honoured through the config read in
# :func:`agent_data_base_dir` below, the same key ``settings.json.j2`` renders
# its allow rule from. The literal fallback covers a hook running with osprey
# off the path, the one case where guessing beats crashing.
try:
    from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR as _DEFAULT_AGENT_DATA_ROOT
except Exception:  # pragma: no cover - hooks must never crash the agent
    _DEFAULT_AGENT_DATA_ROOT = "var/agent_data"


def resolve_memory_dir(project_dir: str) -> Path:
    """Compute the Claude Code memory directory for a project.

    Claude Code stores memory files in
    ``<config-dir>/projects/<encoded>/memory/`` where ``<encoded>`` is the
    absolute project path with every non-alphanumeric character normalized
    to ``-``, and ``<config-dir>`` is ``$CLAUDE_CONFIG_DIR`` when set and
    ``~/.claude`` otherwise. The web-terminal container sets the variable
    (and ``HOME``) to one mounted volume, so the ``~/.claude`` spelling there
    would name a directory nothing writes to — and deny every memory write.
    Mirrors ``osprey.agent_runner.project_paths.claude_config_dir``, inlined
    for the same reason as the regex above.
    """
    abs_project = str(Path(project_dir).resolve())
    encoded = _CLAUDE_PROJECT_DIR_NORMALIZE.sub("-", abs_project)
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    config_dir = Path(configured).expanduser() if configured else Path.home() / ".claude"
    return config_dir / "projects" / encoded / "memory"


def agent_data_base_dir(config: dict | None) -> str:
    """Read ``agent_data.base_dir`` out of an already-loaded config mapping.

    A restatement of ``osprey_connectors.workspace.agent_data_base_dir`` using
    the standard library only, for the same reason the project-path regex above
    is duplicated: the hook runs in user projects where ``osprey`` may not be
    importable. The key is the single spelling of the agent-data root, so a
    project that relocates it keeps this guard and the rendered
    ``NotebookEdit`` allow rule pointing at the same tree.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to a project anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, dict):
        return _DEFAULT_AGENT_DATA_ROOT
    return str(section.get("base_dir") or _DEFAULT_AGENT_DATA_ROOT)


def resolve_artifacts_dirs(hook_input=None) -> list[Path]:
    """Resolve the artifact directories ``NotebookEdit`` may write into.

    A relative ``agent_data.base_dir`` — the normal case — needs an anchor, and
    under the four-zone layout there are two plausible ones: the repo root that
    owns durable agent state (:func:`get_repo_root`) and the render Claude Code
    actually runs in (:func:`get_project_dir`). They coincide in a flat layout
    and diverge in a zoned one, and the gallery has been observed writing under
    each, so both are accepted rather than picking one and denying the agent its
    own notebooks under the other. An absolute ``base_dir`` needs no anchor and
    yields exactly one directory.

    Args:
        hook_input: The parsed hook payload, used to locate the project.

    Returns:
        Resolved ``.../artifacts`` directories, deduplicated, possibly empty.
    """
    base = Path(agent_data_base_dir(load_osprey_config(hook_input))).expanduser()

    if base.is_absolute():
        candidates = [base]
    else:
        candidates = [
            Path(anchor).expanduser() / base
            for anchor in (get_repo_root(hook_input), get_project_dir(hook_input))
            if anchor
        ]

    artifacts_dirs = []
    for candidate in candidates:
        try:
            resolved = (candidate / _ARTIFACTS_SUBDIR).resolve()
        except (OSError, ValueError):
            continue
        if resolved not in artifacts_dirs:
            artifacts_dirs.append(resolved)
    return artifacts_dirs


def is_allowed_memory_write(file_path: str, memory_dir: Path) -> bool:
    """Check whether a file path targets a .md file inside the memory directory."""
    try:
        target = Path(file_path).expanduser().resolve()
    except (OSError, ValueError):
        return False

    if target.suffix.lower() != ".md":
        return False

    # Must be a direct child of the memory dir (no subdirectories / traversal)
    try:
        target.relative_to(memory_dir)
    except ValueError:
        return False

    if target.parent != memory_dir:
        return False

    # Reject path traversal components in the original input
    if ".." in file_path:
        return False

    return True


def is_allowed_notebook_edit(notebook_path: str, artifacts_dirs: list[Path]) -> bool:
    """Check whether a notebook path sits inside one of the artifact directories.

    Recursive, unlike the memory rule: the rendered allow is
    ``<agent_data_root>/artifacts/**``, and the artifact tree is nested by
    design. The directory itself is not a writable target.

    Args:
        notebook_path: The path the tool call named, unmodified.
        artifacts_dirs: Resolved directories from :func:`resolve_artifacts_dirs`.

    Returns:
        ``True`` when the path resolves to something strictly inside one of them.
    """
    try:
        target = Path(notebook_path).expanduser().resolve()
    except (OSError, ValueError):
        return False

    # Reject path traversal components in the original input
    if ".." in notebook_path:
        return False

    for artifacts_dir in artifacts_dirs:
        try:
            target.relative_to(artifacts_dir)
        except ValueError:
            continue
        if target != artifacts_dir:
            return True

    return False


def _target_path(tool_name: str, tool_input: dict | None) -> str:
    """Pull the path a tool call names, by the key that tool uses.

    ``NotebookEdit`` carries ``notebook_path``; ``file_path`` is read only as a
    fallback, so a rename of the key upstream degrades to checking the path
    that *is* present rather than denying every notebook edit outright. The
    order matters: whichever key is read is the one checked against the
    allowlist, so the tool's own key must win.
    """
    if not isinstance(tool_input, dict):
        return ""
    for key in _PATH_KEYS.get(tool_name, ("file_path",)):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _deny(reason: str) -> dict:
    """Build the PreToolUse deny payload carrying *reason*."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _record_deny(hook_input, subject, reason, detail=None):
    """Put one refusal in the deployment's audit ledger, and never cost the deny.

    Both refusal branches route through here so the two records carry one
    vocabulary. The wrap is the hook's fail-open rule: an unwritable audit zone
    downgrades the trail, it does not let the write through.
    """
    try:
        emit_audit(
            "memory-guard",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=subject,
            reason=reason,
            detail=detail,
        )
    except Exception:
        pass  # the audit trail must never cost the deny


def _denied_header(tool_name: str) -> str:
    """Name the refusal after the tool the agent actually reached for."""
    if tool_name == _NOTEBOOK_TOOL:
        return "\U0001f512 NOTEBOOK EDIT DENIED"
    return "\U0001f512 WRITE DENIED"


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    if tool_name not in _GUARDED_TOOLS:
        sys.exit(0)

    target_path = _target_path(tool_name, hook_input.get("tool_input", {}))

    if not target_path:
        log_hook("memory-guard", hook_input, status="deny", detail=f"no-path tool={tool_name}")
        _record_deny(hook_input, subject=tool_name, reason="no_path")
        json.dump(_deny(f"{_denied_header(tool_name)}\n\nNo file path provided."), sys.stdout)
        sys.exit(0)

    if tool_name == _NOTEBOOK_TOOL:
        artifacts_dirs = resolve_artifacts_dirs(hook_input)
        allowed = is_allowed_notebook_edit(target_path, artifacts_dirs)
        listed = "\n".join(f"  {d}/" for d in artifacts_dirs) or "  (none resolved)"
        reason = (
            f"{_denied_header(tool_name)}\n\n"
            "Notebook edits are restricted to the agent-data artifacts tree.\n"
            f"Allowed directories:\n{listed}\n\n"
            "Write notebooks through the workspace tools so they land in the "
            "artifact gallery."
        )
    else:
        project_dir = get_project_dir(hook_input) or os.getcwd()
        memory_dir = resolve_memory_dir(project_dir)
        allowed = is_allowed_memory_write(target_path, memory_dir)
        reason = (
            f"{_denied_header(tool_name)}\n\n"
            "File writes are restricted to Claude Code memory files only.\n"
            f"Allowed directory: {memory_dir}/\n"
            "Allowed extension: .md\n\n"
            "To save memories, write .md files to your Claude memory directory.\n"
            "Use Claude Code's native /memory command to view saved memories."
        )

    if allowed:
        log_hook("memory-guard", hook_input, status="allow", detail=f"tool={tool_name}")
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }
    else:
        log_hook(
            "memory-guard", hook_input, status="deny", detail=f"tool={tool_name} path={target_path}"
        )
        _record_deny(
            hook_input,
            subject=target_path,
            reason="path_not_allowed",
            detail=f"tool={tool_name}",
        )
        output = _deny(reason)

    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
