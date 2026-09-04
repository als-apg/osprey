#!/usr/bin/env python3
"""
---
name: Notebook Update
description: Invalidates the cached notebook HTML after NotebookEdit and badges the NOTEBOOKS panel when the edit lands in the notebooks tree
summary: Tracks notebook edits as workspace artifacts and shows the operator which notebook the agent edited
event: PostToolUse
tools: NotebookEdit
safety_layer: 99
wiring: standalone
timeout: 5
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Has notebook_path? ──NO──► EXIT
              │
             YES
              │
              ▼
         Locate _notebook_cache/
         {stem}_rendered.html
              │
              ├──► exists? ──YES──► Delete cached HTML
              │
              ▼
         Under <agent-data root>/notebooks/? ──NO──► EXIT
              │
             YES
              │
              ▼
         POST /api/agent-activity
         (1 s, fail-open)
              │
              ▼
         EXIT
```

## Details

Lightweight utility hook with no safety implications, doing two things after
the agent edits a notebook.

**Cache invalidation.** The gallery's cached HTML rendering goes stale the
moment the notebook changes, so this hook deletes it and the next gallery view
triggers a fresh render.

**Panel badge.** An edit inside the agent's own notebooks tree is something the
operator should see: the hook reports it to the web terminal as
``POST /api/agent-activity`` with
``{"tool": "NotebookEdit", "target": {"kind": "panel", "panel": "jupyter",
"detail": "<path relative to notebooks/>"}}``, which the frontend turns into a
glow and a badge on the NOTEBOOKS rail entry. Only that tree emits — an edit
under ``artifacts/`` belongs to the gallery, and badging NOTEBOOKS for it would
point the operator at the wrong panel.

## Notebooks tree

The tree is ``<agent-data root>/notebooks/``, resolved exactly the way
``osprey_memory_guard.py`` resolves the sibling trees it gates: the
``agent_data.base_dir`` config key, anchored — when it is relative, the normal
case — on both the repo root that owns durable agent state and the render
Claude Code runs in, which coincide in a flat layout and diverge in a zoned
one. The two hooks must agree on the answer, or an edit the guard allowed would
badge nothing. The resolution is restated here rather than imported: these are
standalone scripts copied into a deployment, and a badge has no business
importing a write gate.

## Terminal-API authorization

The POST carries ``Authorization: Bearer <OSPREY_PANEL_TOKEN>`` whenever that
variable holds a non-blank value in the environment the hook inherits — the web
terminal exports it into the agent it launches. When it is unset, empty or
whitespace-only the header is omitted entirely rather than sent blank, matching
how the server-side ``mcp_server.http._panel_auth_headers`` reads the same
carrier.

Everything about the emit is fail-open and bounded at a second: a terminal that
is down, refuses the call, or rejects the payload costs the operator a badge,
never the tool call.

stdlib-only — the sibling ``osprey_hook_log`` aside, this file must NOT import
osprey, yaml, requests, or any third-party lib. That contract is why the web
terminal's port is written out here rather than looked up: ``osprey.port_layout``
is where every framework port lives, and this file may not import it. The
literal is the ``web`` slot at the layout's default base — what
``default_port('web', 0)`` returns — and it is a *fallback*, reached only when
``OSPREY_WEB_PORT`` is unset. A deployment that moved its block exports the
variable, so the literal is never what such a deployment dials. It carries the
``osprey:not-a-port`` marker so the retired-number lint reads it as the stated
exception it is; move it if the layout's ``web`` offset or default base ever
moves.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    get_hook_input,
    get_project_dir,
    get_repo_root,
    load_osprey_config,
    log_hook,
)

#: Subdirectory of the agent-data root that holds the agent's notebooks. This
#: is the ``notebooks`` in ``NotebookEdit(<agent_data_root>/notebooks/**)`` as
#: rendered by ``settings.json.j2`` and gated by ``osprey_memory_guard.py``.
_NOTEBOOKS_SUBDIR = "notebooks"

# The framework DEFAULT agent-data root, imported rather than spelled out here
# so the two cannot drift apart. Only the default is imported: a project that
# overrides ``agent_data.base_dir`` is honoured through the config read in
# :func:`_agent_data_base_dir` below. The literal fallback covers a hook running
# with osprey off the path, the one case where guessing beats crashing.
try:
    from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR as _DEFAULT_AGENT_DATA_ROOT
except Exception:  # pragma: no cover - hooks must never crash the agent
    _DEFAULT_AGENT_DATA_ROOT = "var/agent_data"

#: The panel the badge lands on, and the tool the frame reports. The panel id
#: is the one ``jupyter`` panel the web terminal registers.
_ACTIVITY_PANEL = "jupyter"
_ACTIVITY_TOOL = "NotebookEdit"

#: Seconds the emit may spend. The hook's own budget is five, and the operator
#: is waiting on the tool call behind it.
_ACTIVITY_TIMEOUT = 1

#: The web terminal's port when ``OSPREY_WEB_PORT`` names none. Written out
#: rather than looked up because this file may not import osprey; the module
#: docstring states the contract and why the literal is the honest fallback.
_DEFAULT_WEB_PORT = "10100"  # osprey:not-a-port — stdlib-only hook contract (see module docstring); equals default_port('web', 0)


def _agent_data_base_dir(config):
    """Read ``agent_data.base_dir`` out of an already-loaded config mapping.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to a project anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, dict):
        return _DEFAULT_AGENT_DATA_ROOT
    return str(section.get("base_dir") or _DEFAULT_AGENT_DATA_ROOT)


def resolve_notebooks_dirs(hook_input=None):
    """Resolve the notebook directories whose edits badge the NOTEBOOKS panel.

    A relative ``agent_data.base_dir`` — the normal case — needs an anchor, and
    under the four-zone layout there are two plausible ones: the repo root that
    owns durable agent state and the render Claude Code actually runs in. They
    coincide in a flat layout and diverge in a zoned one, so both are accepted.
    An absolute ``base_dir`` needs no anchor and yields exactly one directory.

    Args:
        hook_input: The parsed hook payload, used to locate the project.

    Returns:
        Resolved ``.../notebooks`` directories, deduplicated, possibly empty.
    """
    base = Path(_agent_data_base_dir(load_osprey_config(hook_input))).expanduser()

    if base.is_absolute():
        candidates = [base]
    else:
        candidates = [
            Path(anchor).expanduser() / base
            for anchor in (get_repo_root(hook_input), get_project_dir(hook_input))
            if anchor
        ]

    notebooks_dirs = []
    for candidate in candidates:
        try:
            resolved = (candidate / _NOTEBOOKS_SUBDIR).resolve()
        except (OSError, ValueError):
            continue
        if resolved not in notebooks_dirs:
            notebooks_dirs.append(resolved)
    return notebooks_dirs


def notebook_relpath(notebook_path, notebooks_dirs):
    """Return the edited notebook's path relative to its notebooks root.

    Args:
        notebook_path: The path the tool call named, unmodified.
        notebooks_dirs: Resolved roots from :func:`resolve_notebooks_dirs`.

    Returns:
        A POSIX-style relative path when the edit landed strictly inside one of
        the roots, else None — which is the whole test for whether this edit
        badges anything. Traversal components are rejected on the raw input for
        the same reason ``osprey_memory_guard.py`` rejects them: a path the
        guard would have refused must not be reported as one the agent edited.
    """
    if ".." in notebook_path:
        return None

    try:
        target = Path(notebook_path).expanduser().resolve()
    except (OSError, ValueError):
        return None

    for notebooks_dir in notebooks_dirs:
        try:
            relative = target.relative_to(notebooks_dir)
        except ValueError:
            continue
        if target != notebooks_dir:
            return relative.as_posix()

    return None


def _activity_request(detail):
    """Build the ``POST /api/agent-activity`` request carrying *detail*.

    The body is the route's fixed contract; the bearer is sent only when
    ``OSPREY_PANEL_TOKEN`` holds a non-blank value, because a blank one would
    be a credential claim this hook cannot back.
    """
    payload = json.dumps(
        {
            "tool": _ACTIVITY_TOOL,
            "target": {"kind": "panel", "panel": _ACTIVITY_PANEL, "detail": detail},
        }
    ).encode()

    headers = {"Content-Type": "application/json"}
    token = (os.environ.get("OSPREY_PANEL_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    port = os.environ.get("OSPREY_WEB_PORT", _DEFAULT_WEB_PORT)
    return urllib.request.Request(
        f"http://127.0.0.1:{port}/api/agent-activity",
        data=payload,
        headers=headers,
        method="POST",
    )


def notify_notebook_edit(detail):
    """Badge the NOTEBOOKS panel with the notebook the agent just edited.

    Fire-and-forget: an unreachable, refusing or slow web terminal costs the
    operator a badge and nothing else.
    """
    try:
        urllib.request.urlopen(_activity_request(detail), timeout=_ACTIVITY_TIMEOUT).close()
    except Exception:
        pass


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    notebook_path = tool_input.get("notebook_path", "")

    if not notebook_path:
        sys.exit(0)

    # Invalidate cached HTML for this notebook
    try:
        nb_path = Path(notebook_path)
        # Check in the artifact cache directory
        cache_dir = nb_path.parent / "_notebook_cache"
        cached_html = cache_dir / f"{nb_path.stem}_rendered.html"
        if cached_html.exists():
            cached_html.unlink()
            log_hook(
                "notebook-update", hook_input, status="invalidated", detail=f"path={notebook_path}"
            )
        else:
            log_hook(
                "notebook-update", hook_input, status="no-cache", detail=f"path={notebook_path}"
            )
    except Exception:
        pass  # Never block on cache invalidation failure

    # Badge the NOTEBOOKS panel, but only for the agent's own notebooks tree.
    try:
        detail = notebook_relpath(notebook_path, resolve_notebooks_dirs(hook_input))
        if detail:
            notify_notebook_edit(detail)
    except Exception:
        pass  # Never block on a badge

    sys.exit(0)


if __name__ == "__main__":
    main()
