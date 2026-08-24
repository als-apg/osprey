"""Coverage for the shipped ``osprey_memory_guard`` hook across every write tool.

Claude Code can put bytes on disk through three tools — ``Write``, ``MultiEdit``
and ``NotebookEdit`` — and the guard used to declare only ``Write``. The other
two reached no hook at all, so a build that looked fully gated let the agent
create arbitrary files under a different tool name. Nothing about that gap was
visible in a rendered project: ``settings.json`` carried a ``PreToolUse`` rule,
it just carried a matcher one tool wide.

These tests pin the widened contract at the template level, where the gap was
introduced:

* the ``tools:`` frontmatter names all three tools, because that string is
  copied verbatim into the ``PreToolUse`` matcher
  (``osprey.cli.templates.claude_code`` builds the rule from it), so the
  frontmatter *is* the matcher;
* the ``NotebookEdit`` scope agrees with the ``NotebookEdit(<agent_data_root>/
  artifacts/**)`` allow rule ``settings.json.j2`` renders — two independent
  spellings of one policy, which is exactly the pair that silently forks;
* each tool is actually gated end to end, by running the hook the way Claude
  Code runs it.

The behavioural cases go through a subprocess rather than an in-process import:
the hook's stdin/stdout JSON contract is the thing being checked, and a
subprocess is the only way to exercise it as shipped. ``$HOME`` is redirected at
a temp directory for every run, because the memory branch resolves
``~/.claude/projects/...`` and must never be pointed at the developer's real
home.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
MEMORY_GUARD = HOOKS_DIR / "osprey_memory_guard.py"
SETTINGS_TEMPLATE = (
    REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "claude" / "settings.json.j2"
)

#: Every tool Claude Code offers that writes a file. The guard must have an
#: opinion about each one; a tool missing here is a tool the agent can write
#: through unchallenged.
WRITE_TOOLS = frozenset({"Write", "MultiEdit", "NotebookEdit"})

#: Subdirectory of the agent-data root that ``NotebookEdit`` is scoped to.
ARTIFACTS_SUBDIR = "artifacts"


def _frontmatter() -> dict:
    """Parse the YAML frontmatter out of the hook's module docstring."""
    parts = MEMORY_GUARD.read_text(encoding="utf-8").split("---")
    return yaml.safe_load(parts[1])


@pytest.fixture
def hook_home(tmp_path):
    """A throwaway ``$HOME`` the hook subprocess resolves ``~`` against."""
    home = tmp_path / "home"
    home.mkdir()
    return home


@pytest.fixture
def project(tmp_path):
    """A project directory the hook can treat as both render and repo root."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def write_config(project):
    """Write a ``config.yml`` naming the project root and agent-data base dir.

    ``project_root`` is set explicitly so the hook's repo-root resolution stops
    at the config instead of walking out of ``tmp_path`` into whatever
    ``profile.yml`` happens to sit above it.
    """

    def _write(base_dir=None):
        config = {"project_root": str(project)}
        if base_dir is not None:
            config["agent_data"] = {"base_dir": base_dir}
        (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    return _write


@pytest.fixture
def run_guard(project, hook_home):
    """Run the guard as Claude Code does: JSON on stdin, JSON on stdout.

    The environment is built from scratch rather than inherited so a stray
    ``OSPREY_CONFIG`` in the developer's shell cannot redirect the config the
    hook reads.
    """

    def _run(tool_name, tool_input):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(hook_home),
            "USERPROFILE": str(hook_home),
            "CLAUDE_PROJECT_DIR": str(project),
        }
        payload = json.dumps(
            {"tool_name": tool_name, "tool_input": tool_input, "cwd": str(project)}
        )
        completed = subprocess.run(
            [sys.executable, str(MEMORY_GUARD)],
            input=payload,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Traceback" not in completed.stderr, completed.stderr
        if not completed.stdout.strip():
            return None
        return json.loads(completed.stdout)

    return _run


def _decision(result):
    """Pull the permission decision out of a hook result, or ``None``."""
    if result is None:
        return None
    return result["hookSpecificOutput"]["permissionDecision"]


def memory_dir_for(hook_home, project):
    """The memory directory the hook computes, rebuilt from its documented rule."""
    import re

    encoded = re.sub(r"[^A-Za-z0-9-]", "-", str(project.resolve()))
    return hook_home / ".claude" / "projects" / encoded / "memory"


# -- Frontmatter is the matcher --


@pytest.mark.unit
def test_frontmatter_declares_every_write_tool():
    """``tools:`` names all three write tools.

    The build copies this string into the ``PreToolUse`` matcher verbatim, so a
    tool absent here never reaches the hook — the gap is invisible in the
    rendered settings, which still show a rule.
    """
    declared = {tool.strip() for tool in _frontmatter()["tools"].split("|")}

    assert declared == WRITE_TOOLS


@pytest.mark.unit
def test_frontmatter_uses_alternation_not_commas():
    """The matcher is a regex, so the tools must be pipe-separated.

    A comma-separated list renders as a literal matcher that matches no tool
    name at all, which would take the guard dark while leaving the rule in
    place.
    """
    assert _frontmatter()["tools"] == "Write|MultiEdit|NotebookEdit"


@pytest.mark.unit
def test_guard_stays_the_outermost_pretooluse_gate():
    """Widening must not disturb the hook's ordering or wiring."""
    meta = _frontmatter()

    assert meta["event"] == "PreToolUse"
    assert meta["wiring"] == "standalone"
    assert meta["safety_layer"] == 0


# -- Agreement with the rendered allow rule --


@pytest.mark.unit
def test_settings_template_still_scopes_notebookedit_to_artifacts():
    """``settings.json.j2`` grants ``NotebookEdit`` only under the artifacts tree.

    The hook denies outside that same directory. If this allow rule is ever
    respelled, the hook's scope has to move with it or the two halves of one
    policy disagree.
    """
    template = SETTINGS_TEMPLATE.read_text(encoding="utf-8")

    assert f"NotebookEdit(' ~ agent_data_root ~ '/{ARTIFACTS_SUBDIR}/**)" in template


@pytest.mark.unit
def test_hook_scopes_notebookedit_to_the_same_subdirectory():
    """The hook names the same ``artifacts`` subdirectory the allow rule does."""
    source = MEMORY_GUARD.read_text(encoding="utf-8")

    assert f'_ARTIFACTS_SUBDIR = "{ARTIFACTS_SUBDIR}"' in source


# -- MultiEdit is gated like Write --


@pytest.mark.unit
def test_multiedit_outside_memory_dir_is_denied(run_guard, project):
    """``MultiEdit`` to an arbitrary file is refused, as ``Write`` always was."""
    target = str(project / "evil.py")

    result = run_guard("MultiEdit", {"file_path": target, "edits": []})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_multiedit_to_memory_file_is_allowed(run_guard, project, hook_home):
    """``MultiEdit`` on a memory ``.md`` file is allowed, like ``Write``."""
    target = str(memory_dir_for(hook_home, project) / "channels.md")

    result = run_guard("MultiEdit", {"file_path": target, "edits": []})

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_multiedit_deny_message_names_the_memory_directory(run_guard, project):
    """The refusal tells the agent where it may write instead."""
    result = run_guard("MultiEdit", {"file_path": str(project / "evil.py"), "edits": []})

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "WRITE DENIED" in reason
    assert "memory" in reason.lower()


# -- NotebookEdit is scoped to the artifacts tree --


@pytest.mark.unit
def test_notebookedit_inside_artifacts_is_allowed(run_guard, project, write_config):
    """A notebook under ``<agent_data_root>/artifacts`` is the agent's own output."""
    write_config()
    target = project / "var" / "agent_data" / ARTIFACTS_SUBDIR / "run.ipynb"
    target.parent.mkdir(parents=True)

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_notebookedit_nested_under_artifacts_is_allowed(run_guard, project, write_config):
    """The allow is recursive — ``artifacts/**``, not just direct children."""
    write_config()
    target = project / "var" / "agent_data" / ARTIFACTS_SUBDIR / "2026" / "run.ipynb"
    target.parent.mkdir(parents=True)

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_notebookedit_outside_artifacts_is_denied(run_guard, project, write_config):
    """A notebook anywhere else is refused — this is the tool's whole gap."""
    write_config()
    target = project / "escape.ipynb"

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_notebookedit_into_memory_dir_is_denied(run_guard, project, hook_home, write_config):
    """The two scopes are separate: a notebook may not target the memory dir."""
    write_config()
    target = memory_dir_for(hook_home, project) / "notes.ipynb"

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_notebookedit_honours_relocated_agent_data_root(run_guard, project, write_config):
    """An overridden ``agent_data.base_dir`` moves the allowed directory with it.

    ``settings.json.j2`` interpolates the same key into its allow rule, so a
    guard hard-coding the default would refuse the agent its own artifacts on
    any project that relocates the root.
    """
    write_config(base_dir="custom_state/agent_data")
    target = project / "custom_state" / "agent_data" / ARTIFACTS_SUBDIR / "run.ipynb"
    target.parent.mkdir(parents=True)

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_notebookedit_at_default_root_denied_when_root_relocated(run_guard, project, write_config):
    """Relocating the root also *narrows* the allow — the default no longer counts."""
    write_config(base_dir="custom_state/agent_data")
    target = project / "var" / "agent_data" / ARTIFACTS_SUBDIR / "run.ipynb"
    target.parent.mkdir(parents=True)

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_notebookedit_traversal_is_denied(run_guard, project, write_config):
    """``..`` in the raw path is refused even when it resolves back inside."""
    write_config()
    artifacts = project / "var" / "agent_data" / ARTIFACTS_SUBDIR
    artifacts.mkdir(parents=True)
    target = artifacts / "sub" / ".." / "run.ipynb"

    result = run_guard("NotebookEdit", {"notebook_path": str(target)})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_notebookedit_without_a_path_is_denied(run_guard, write_config):
    """A notebook edit naming no target is refused, not waved through."""
    write_config()

    result = run_guard("NotebookEdit", {"new_source": "print('hi')"})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_notebookedit_deny_message_names_the_artifacts_directory(run_guard, project, write_config):
    """The refusal points at the artifact tree, not at the memory directory."""
    write_config()

    result = run_guard("NotebookEdit", {"notebook_path": str(project / "escape.ipynb")})

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "NOTEBOOK EDIT DENIED" in reason
    assert ARTIFACTS_SUBDIR in reason


# -- The original Write contract is unchanged --


@pytest.mark.unit
def test_write_to_memory_file_still_allowed(run_guard, project, hook_home):
    """Widening the guard did not disturb the branch it already had."""
    target = str(memory_dir_for(hook_home, project) / "MEMORY.md")

    result = run_guard("Write", {"file_path": target, "content": "# Memories\n"})

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_write_outside_memory_dir_still_denied(run_guard, project):
    """Arbitrary ``Write`` remains refused."""
    result = run_guard("Write", {"file_path": str(project / "CLAUDE.md"), "content": "x"})

    assert _decision(result) == "deny"


@pytest.mark.unit
def test_read_tool_passes_through(run_guard):
    """A tool that writes nothing gets no opinion, so other gates still decide."""
    result = run_guard("Read", {"file_path": "/etc/passwd"})

    assert result is None
