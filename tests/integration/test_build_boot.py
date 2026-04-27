"""Tier 1 build + MCP boot smoke tests.

Runs ``osprey build`` as a real subprocess (so the project venv is created and
dependencies are installed), then for every entry in the generated
``.mcp.json`` it:

  1. Asserts the configured ``command`` actually exists on disk and lives
     inside the build output (catches the stale-path bug from worktree
     renames).
  2. Spawns the server and performs a JSON-RPC ``initialize`` + ``tools/list``
     handshake to verify the documented tools are advertised.

Each preset becomes a parametrized test ID. CI matrices invoke the right
preset via ``pytest -k <preset>``.

Wall-clock budget: ~2 min/preset with a warm uv cache; cold cache may take
5-6 min on the first run after ``uv.lock`` changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration._mcp_handshake import (
    MCPHandshakeError,
    assert_tools_superset,
    list_mcp_tools,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Presets shipped in src/osprey/profiles/presets that ship with this repo.
# osprey-edu adds ``education`` via an additional parametrize entry on top of
# this list; education-specific assertions live in osprey-edu's own test file.
PRESETS = ["hello-world", "control-assistant"]


# Hard-coded expected tools per logical .mcp.json server name. Subset
# assertions (``actual ⊇ expected``) so adding new tools doesn't break the
# build smoke. Verified against ``@mcp.tool()`` decorators in
# src/osprey/mcp_server/<server>/tools/.
EXPECTED_TOOLS: dict[str, set[str]] = {
    "controls": {"channel_read", "channel_write", "channel_limits", "archiver_read"},
    "python": {"execute", "execute_file"},
    "workspace": {
        "archiver_downsample",
        "create_static_plot",
        "screenshot_capture",
        "session_log",
    },
    "ariel": {"keyword_search", "entry_get", "sql_query", "semantic_search"},
}
# channel-finder: enumerated dynamically since which backend is enabled
# (hierarchical | in_context | middle_layer) depends on preset config.

# Forbidden path fragments in any rendered command/args/env value. Catches
# the worktree-rename bug class: any reference to a host source-tree path
# (where the repo lives) means the build wired itself up to outside-the-
# project files and will break the moment the user moves the project.
def _forbidden_path_fragments() -> list[str]:
    """Build the forbidden-path list dynamically from REPO_ROOT.

    Includes the repo path itself plus common siblings/legacy names that
    show up in this developer's tree. CI runs in a clean checkout where
    only ``REPO_ROOT`` is in scope, so this is purely a local-dev safety
    net; the in-build-dir check below is the real contract.
    """
    fragments = [str(REPO_ROOT)]
    parent = REPO_ROOT.parent
    for sibling in ("osprey-edu", "osprey-3d-lab", "osprey-main"):
        candidate = parent / sibling
        if candidate.exists() and candidate != REPO_ROOT:
            fragments.append(str(candidate))
    return fragments


@pytest.fixture(scope="module")
def build_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build every preset once per module run; reuse across tests.

    Module-scoped because ``osprey build`` is the slow step (uv sync inside
    the project venv). Each test asserts a different invariant against the
    same artefact.
    """
    outputs: dict[str, Path] = {}
    base = tmp_path_factory.mktemp("preset_builds")
    osprey_bin = _find_osprey_console_script()

    for preset in PRESETS:
        project_dir = base / preset
        cmd = [
            str(osprey_bin),
            "build",
            preset,  # project name
            "--preset",
            preset,
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if proc.returncode != 0:
            pytest.fail(
                f"osprey build {preset} failed (rc={proc.returncode}):\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        if not project_dir.exists():
            pytest.fail(f"build succeeded but project dir missing: {project_dir}")
        outputs[preset] = project_dir
    return outputs


def _find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for the active interpreter.

    Prefers a sibling of ``sys.executable`` (works inside both venvs and
    CI's ``setup-uv``). Falls back to ``shutil.which("osprey")``.
    """
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError(
        "Could not locate the 'osprey' console script. "
        f"Tried {Path(sys.executable).parent / 'osprey'} and PATH."
    )


def _load_mcp_json(project_dir: Path) -> dict:
    mcp_path = project_dir / ".mcp.json"
    assert mcp_path.is_file(), f"missing .mcp.json in {project_dir}"
    return json.loads(mcp_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_build_paths_are_self_consistent(build_outputs: dict[str, Path], preset: str) -> None:
    """Every command + path string in .mcp.json must live inside the build dir.

    This is the test that would catch the stale-path-after-worktree-rename
    bug. If ``mv`` is run on a built project, the venv-internal paths break;
    here we assert they were valid the moment the build finished.
    """
    project_dir = build_outputs[preset]
    project_str = str(project_dir)
    forbidden = _forbidden_path_fragments()

    cfg = _load_mcp_json(project_dir)
    servers = cfg.get("mcpServers", {})
    assert servers, f".mcp.json has no servers for preset {preset}"

    failures: list[str] = []
    for name, entry in servers.items():
        if "url" in entry:
            continue  # HTTP/SSE — no command to verify
        cmd = entry.get("command")
        if not cmd:
            failures.append(f"{name}: missing 'command' (and not http)")
            continue
        cmd_path = Path(cmd)
        # Compare literal paths (not Path.resolve()): a venv's bin/python is
        # a symlink to the host interpreter, so resolving would always escape
        # the project dir. The contract we care about is "the command string
        # written into .mcp.json points inside the build output".
        if not cmd_path.is_file():
            failures.append(f"{name}: command path does not exist: {cmd}")
        elif not cmd.startswith(project_str + "/"):
            failures.append(f"{name}: command escapes project dir: {cmd}")

        # Walk args + env for any host-tree leakage.
        for arg in entry.get("args") or []:
            for frag in forbidden:
                if isinstance(arg, str) and frag in arg and not arg.startswith(project_str):
                    failures.append(f"{name}: arg references host source path: {arg}")
        for k, v in (entry.get("env") or {}).items():
            if not isinstance(v, str):
                continue
            for frag in forbidden:
                if frag in v and not v.startswith(project_str) and "${" not in v:
                    failures.append(f"{name}: env {k}={v} references host source path")

    assert not failures, (
        f".mcp.json for preset {preset!r} has stale or escaping paths:\n  "
        + "\n  ".join(failures)
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_stale_command_path_fails_fast(
    build_outputs: dict[str, Path], preset: str, tmp_path: Path
) -> None:
    """Per-preset: a stale command path in .mcp.json must fail fast, not hang.

    Takes each stdio server's real .mcp.json entry and rewrites only its
    ``command`` to a non-existent location. Every other arg/env is preserved,
    so this exercises the spawn path with the exact framing a moved-build
    failure would hit. Locks the contract that future refactors of
    ``_mcp_handshake.list_mcp_tools`` must not swallow ``FileNotFoundError``
    and replace fast failure with hangs.
    """
    cfg = _load_mcp_json(build_outputs[preset])
    servers = cfg.get("mcpServers", {})
    assert servers, f".mcp.json has no servers for preset {preset}"

    saw_stdio = False
    for name, entry in servers.items():
        if "url" in entry or "command" not in entry:
            continue
        saw_stdio = True
        bogus_cmd = str(tmp_path / "definitely-not-here" / Path(entry["command"]).name)
        with pytest.raises(MCPHandshakeError, match="command not found:"):
            list_mcp_tools(
                command=bogus_cmd,
                args=list(entry.get("args") or []),
                env=entry.get("env"),
                timeout=5.0,
            )
    assert saw_stdio, f"preset {preset!r} has no stdio MCP server to probe"


@pytest.mark.parametrize("preset", PRESETS)
def test_mcp_servers_register_expected_tools(
    build_outputs: dict[str, Path], preset: str
) -> None:
    """Spawn every stdio MCP server and verify it advertises the documented tools."""
    project_dir = build_outputs[preset]
    cfg = _load_mcp_json(project_dir)
    servers = cfg.get("mcpServers", {})

    if not servers:
        pytest.skip(f"no MCP servers configured for preset {preset}")

    failures: list[str] = []
    for name, entry in servers.items():
        if "url" in entry:
            continue  # http/sse — out of scope for stdio handshake
        try:
            tool_names = list_mcp_tools(
                command=entry["command"],
                args=list(entry.get("args") or []),
                env=entry.get("env"),
                timeout=30.0,
            )
        except MCPHandshakeError as exc:
            failures.append(f"{name}: handshake failed: {exc}")
            continue

        expected = _expected_tools_for(name)
        if expected is None:
            # Unknown server name (custom/facility-specific). Soft check:
            # at least *some* tools must register.
            if not tool_names:
                failures.append(f"{name}: server returned zero tools")
            continue
        try:
            assert_tools_superset(name, tool_names, expected)
        except AssertionError as exc:
            failures.append(str(exc))

    assert not failures, f"MCP boot smoke failed for preset {preset!r}:\n" + "\n".join(failures)


def _expected_tools_for(server_name: str) -> set[str] | None:
    """Return the hard-coded expected-tools subset for a known server name.

    Returns ``None`` for unknown servers — callers fall back to a
    "at least one tool registered" check.
    """
    if server_name in EXPECTED_TOOLS:
        return EXPECTED_TOOLS[server_name]
    # channel-finder: tool names diverge across hierarchical / in_context /
    # middle_layer. Skip the subset check; the "≥1 tool" floor in the caller
    # is sufficient to catch a server that fails to register anything (the
    # original bug class).
    return None
