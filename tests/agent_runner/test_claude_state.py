"""Seeding Claude Code's first-run state for containerized deployments.

A fresh container volume is, from Claude Code's point of view, a brand-new
machine: the interactive CLI walks the operator through onboarding (theme
picker, security notes, terminal setup), asks whether to trust the project
folder, and — under a raw-key provider — whether to use the ambient API key.
None of that is an operator's decision in a deployment OSPREY rendered: the
render already states the theme, the permissions, and the provider. Worse, the
trust dialog is not cosmetic: the rendered ``permissions.allow`` list does not
apply until the folder is trusted, so an unseeded first session runs with a
degraded permission surface.

:func:`osprey.agent_runner.claude_state.seed_claude_state` writes the state
Claude Code would have recorded had the operator answered, into the
``.claude.json`` the container's ``CLAUDE_CONFIG_DIR``/``HOME`` names. The
properties asserted here:

- **Merge-only.** A key that exists is never rewritten — a returning
  operator's live volume keeps every choice they made, including an explicit
  ``hasCompletedOnboarding: false``.
- **Never clobber.** A file that does not parse is left byte-for-byte alone;
  losing an operator's OAuth state to a seed step would be strictly worse
  than showing the prompts.
- **Deterministic trust key.** Trust is recorded against the resolved render
  directory — the cwd every launcher starts Claude Code in — exactly as
  Claude Code keys it for a non-git directory.
- **Provider-conditional key approval.** Only a provider whose auth reaches
  Claude Code as ``ANTHROPIC_API_KEY`` triggers the CLI's key-approval
  prompt, so only that shape is seeded; token-auth proxies get nothing.
- **Idempotent.** A second run against its own output writes nothing.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from osprey.agent_runner.claude_state import (
    CLAUDE_CONFIG_VOLUME_SUFFIX,
    CLAUDE_STATE_FILENAME,
    is_untrusted_allow_rules_notice,
    seed_claude_state,
)

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def render_dir(tmp_path: Path) -> Path:
    """A minimal render: a directory holding a config.yml with a pinned CLI."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(
        "claude_code:\n  provider: anthropic\n  cli_version: '2.1.239'\n"
    )
    return render


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """The volume-backed directory ``CLAUDE_CONFIG_DIR`` points at."""
    d = tmp_path / "claude-config"
    d.mkdir()
    return d


def _seed(render: Path, config_dir: Path, **env: str) -> list[str]:
    return seed_claude_state(render, env={"CLAUDE_CONFIG_DIR": str(config_dir), **env})


def _state(config_dir: Path) -> dict:
    return json.loads((config_dir / CLAUDE_STATE_FILENAME).read_text())


# ── fresh volume ─────────────────────────────────────────────────────────────


def test_fresh_volume_gets_onboarding_trust_and_version(render_dir, config_dir):
    seeded = _seed(render_dir, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is True
    assert state["lastOnboardingVersion"] == "2.1.239"
    assert state["projects"][str(render_dir)]["hasTrustDialogAccepted"] is True
    assert seeded  # every action is reported for the entrypoint log


def test_config_dir_is_created_when_missing(render_dir, tmp_path):
    """A named volume mounts as an existing dir, but a bare HOME may not hold one."""
    config_dir = tmp_path / "not-yet"
    seed_claude_state(render_dir, env={"CLAUDE_CONFIG_DIR": str(config_dir)})
    assert (config_dir / CLAUDE_STATE_FILENAME).is_file()


def test_home_fallback_when_config_dir_unset(render_dir, tmp_path):
    """The single-user image sets HOME only; state lands at ~/.claude.json."""
    home = tmp_path / "home"
    home.mkdir()
    seed_claude_state(render_dir, env={"HOME": str(home)})
    assert (home / CLAUDE_STATE_FILENAME).is_file()


def test_no_target_dir_is_a_reported_noop(render_dir):
    """Neither CLAUDE_CONFIG_DIR nor HOME: nowhere to write, and no crash."""
    assert seed_claude_state(render_dir, env={}) == []


# ── merge-only semantics ─────────────────────────────────────────────────────


def test_existing_keys_and_unrelated_state_survive(render_dir, config_dir):
    """The seed adds what is missing and rewrites nothing that exists."""
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": False,  # an explicit choice, kept
                "oauthAccount": {"email": "op@example.org"},  # untouched
                "projects": {
                    "/somewhere/else": {"hasTrustDialogAccepted": False},
                },
            }
        )
    )

    _seed(render_dir, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is False
    assert state["oauthAccount"] == {"email": "op@example.org"}
    assert state["projects"]["/somewhere/else"] == {"hasTrustDialogAccepted": False}
    # ...while the render's own trust entry is still added beside it.
    assert state["projects"][str(render_dir)]["hasTrustDialogAccepted"] is True


def test_existing_project_entry_gains_only_the_missing_key(render_dir, config_dir):
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps({"projects": {str(render_dir): {"exampleFiles": ["a.py"]}}})
    )

    _seed(render_dir, config_dir)

    entry = _state(config_dir)["projects"][str(render_dir)]
    assert entry["exampleFiles"] == ["a.py"]
    assert entry["hasTrustDialogAccepted"] is True


def test_corrupt_state_file_is_left_untouched(render_dir, config_dir):
    """A parse failure must not cost the operator their file."""
    corrupt = '{"oauthAccount": '  # truncated write
    (config_dir / CLAUDE_STATE_FILENAME).write_text(corrupt)

    assert _seed(render_dir, config_dir) == []
    assert (config_dir / CLAUDE_STATE_FILENAME).read_text() == corrupt


def test_second_run_is_a_silent_noop(render_dir, config_dir):
    _seed(render_dir, config_dir)
    before = (config_dir / CLAUDE_STATE_FILENAME).read_text()

    assert _seed(render_dir, config_dir) == []
    assert (config_dir / CLAUDE_STATE_FILENAME).read_text() == before


# ── provider-conditional API-key approval ────────────────────────────────────


def test_anthropic_key_is_pre_approved_by_its_last_20_chars(render_dir, config_dir):
    key = "sk-ant-api03-" + "x" * 40
    _seed(render_dir, config_dir, ANTHROPIC_API_KEY=key)

    approved = _state(config_dir)["customApiKeyResponses"]["approved"]
    assert approved == [key[-20:]]


def test_token_auth_provider_seeds_no_key_approval(config_dir, tmp_path):
    """A proxy provider authenticates with ANTHROPIC_AUTH_TOKEN — no prompt exists."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(
        "claude_code:\n"
        "  provider: cborg\n"
        "api:\n"
        "  providers:\n"
        "    cborg:\n"
        "      default_model: claude-haiku-4-5\n"
        "      models: [claude-haiku-4-5]\n"
    )

    _seed(render, config_dir, CBORG_API_KEY="cborg-secret", ANTHROPIC_AUTH_TOKEN="cborg-secret")

    assert "customApiKeyResponses" not in _state(config_dir)


def test_missing_key_value_seeds_no_approval(render_dir, config_dir):
    """Provider says ANTHROPIC_API_KEY but the container got no key: nothing to approve."""
    _seed(render_dir, config_dir)
    assert "customApiKeyResponses" not in _state(config_dir)


def test_already_rejected_digest_is_respected(render_dir, config_dir):
    key = "sk-ant-api03-" + "y" * 40
    (config_dir / CLAUDE_STATE_FILENAME).write_text(
        json.dumps({"customApiKeyResponses": {"approved": [], "rejected": [key[-20:]]}})
    )

    _seed(render_dir, config_dir, ANTHROPIC_API_KEY=key)

    responses = _state(config_dir)["customApiKeyResponses"]
    assert responses["approved"] == []
    assert responses["rejected"] == [key[-20:]]


# ── degraded configs ─────────────────────────────────────────────────────────


def test_render_without_config_still_seeds_onboarding_and_trust(config_dir, tmp_path):
    """No config.yml: no version, no provider — the universal keys still land."""
    render = tmp_path / "bare"
    render.mkdir()

    _seed(render, config_dir)

    state = _state(config_dir)
    assert state["hasCompletedOnboarding"] is True
    assert "lastOnboardingVersion" not in state
    assert state["projects"][str(render)]["hasTrustDialogAccepted"] is True


# ── the per-user volume that holds this state ────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUFFIX_OWNER = _REPO_ROOT / "src" / "osprey" / "agent_runner" / "claude_state.py"


def test_claude_config_volume_suffix_is_the_deployed_spelling():
    """Deployed hosts hold each user's state volume under exactly this name, so
    this literal changes only alongside a migration of those volumes, never on
    its own."""
    assert CLAUDE_CONFIG_VOLUME_SUFFIX == "-claude-config"


def _docstring_constants(tree: ast.AST) -> set[int]:
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }


def test_no_other_module_spells_the_volume_suffix():
    """Every producer of a per-user state-volume name reads the one constant.

    Python sources are scanned for string constants (f-string parts included),
    docstrings excepted; templates are scanned line by line with Jinja comments
    and ``#`` comment lines removed, since prose may name the volume.
    """
    offenders: list[str] = []
    for root in (_REPO_ROOT / "src", _REPO_ROOT / "packages"):
        for path in sorted(root.rglob("*.py")):
            if path == _SUFFIX_OWNER:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = _docstring_constants(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and "-claude-config" in node.value
                ):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    for path in sorted((_REPO_ROOT / "src" / "osprey" / "templates").rglob("*.j2")):
        text = re.sub(
            r"\{#.*?#\}",
            lambda m: "\n" * m.group(0).count("\n"),
            path.read_text(encoding="utf-8"),
            flags=re.S,
        )
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip().startswith("#") and "-claude-config" in line:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    assert offenders == []


# ── the untrusted allow-rules notice ─────────────────────────────────────────

_SAMPLE_RENDER = "/var/osprey/build"
_SAMPLE_CONFIG_DIR = "/var/osprey/agent_data/claude-config"


def _notice(count: str) -> str:
    """The CLI's untrusted-workspace notice, verbatim from the pinned CLI."""
    return (
        f"Ignoring {count} from .claude/settings.json: this workspace has not been "
        "trusted. Run Claude Code interactively here once and accept the trust dialog, or set "
        f'projects["{_SAMPLE_RENDER}"].hasTrustDialogAccepted: true in '
        f"{_SAMPLE_CONFIG_DIR}/.claude.json."
    )


@pytest.mark.parametrize("count", ["1 permissions.allow entry", "4 permissions.allow entries"])
def test_the_notice_is_recognised_in_both_numbers(count: str) -> None:
    assert is_untrusted_allow_rules_notice(_notice(count))


@pytest.mark.parametrize(
    "line",
    [
        _notice("1 permissions.additionalDirectories entry"),
        "error: " + _notice("1 permissions.allow entry"),
        "Warning: no stdin data received in 3s, proceeding without it. If piping from a slow "
        "command, redirect stdin explicitly: < /dev/null to skip, or wait longer.",
        "",
    ],
)
def test_other_stderr_lines_are_not_the_notice(line: str) -> None:
    assert not is_untrusted_allow_rules_notice(line)


def test_the_pinned_cli_prints_the_notice_in_the_recognised_form(tmp_path: Path) -> None:
    """The bundled CLI's notice still matches the recognised form.

    The notice is observed in the pinned CLI, not documented, so this runs the
    real binary: a CLI bump that rewords it must fail here instead of letting
    the notice back into dispatch run records. Never skips — the binary ships
    with the SDK this project depends on.
    """
    from osprey.agent_runner.launcher import bundled_cli_path

    binary = bundled_cli_path()
    if binary is None:
        pytest.fail("the Agent SDK ships no bundled Claude CLI")

    render = tmp_path / "render"
    (render / ".claude").mkdir(parents=True)
    (render / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Read(/data/**)"], "deny": ["WebFetch"]}})
    )
    config = tmp_path / "config"
    config.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(config),
        "ANTHROPIC_API_KEY": "sk-test",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }

    proc = subprocess.Popen(
        [str(binary), "-p", "hi", "--max-turns", "1", "--setting-sources=project"],
        cwd=render,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    timer = threading.Timer(30, proc.kill)
    timer.start()
    seen: list[str] = []
    notice: str | None = None
    try:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip()
            seen.append(line)
            if line.startswith("Ignoring "):
                notice = line
                break
    finally:
        timer.cancel()
        proc.kill()
        proc.wait()

    assert notice is not None, "the CLI printed no notice:\n" + "\n".join(seen)
    assert is_untrusted_allow_rules_notice(notice), notice
    assert "1 permissions.allow entry" in notice
    assert not any("deny" in line for line in seen)

    state_file = config / CLAUDE_STATE_FILENAME
    if state_file.exists():
        projects = json.loads(state_file.read_text()).get("projects", {})
        assert not any(entry.get("hasTrustDialogAccepted") for entry in projects.values())
