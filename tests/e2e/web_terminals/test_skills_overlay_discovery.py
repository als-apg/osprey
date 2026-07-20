"""Smoke check: is a seeded skill discovered by ``claude`` under the flags the
web terminal launches with?

The multi-user web terminal launches Claude Code with ``--setting-sources
project`` unconditionally appended (``osprey.utils.claude_launcher``). This test
answers the gating question for the per-user *skills overlay*: when a skill is
dropped into a filesystem location and ``claude`` starts with
``--setting-sources project``, does the harness DISCOVER that skill?

Discovery signal
----------------
Skills surface as slash commands. The Claude Code CLI emits a
``{"type":"system","subtype":"init", ...}`` event as the FIRST stream-json
record, and that record carries a ``skills`` array (and a matching
``slash_commands`` array). This init event is emitted at session start, *before*
the model is contacted — so the signal is deterministic and needs no working
provider credentials: a bogus ``ANTHROPIC_API_KEY`` still yields a full init
record before the run ends with a 401. That keeps this test runnable in CI
without spending tokens or requiring VPN/provider access.

VERDICT (recorded empirically against claude 2.1.210)
-----------------------------------------------------
Skill *scope* is bound to ``--setting-sources``:

    | skill on disk at                | scope   | seen with --setting-sources project |
    | ------------------------------- | ------- | ----------------------------------- |
    | ``$CLAUDE_CONFIG_DIR/skills/``  | user    | NO  — suppressed                    |
    | ``<cwd>/.claude/skills/``       | project | YES — discovered                    |

So:

* Seeding a skill into the per-user ``claude-config`` volume (which becomes
  ``$CLAUDE_CONFIG_DIR`` = the *user* scope) is **INERT** under the web
  terminal's ``--setting-sources project`` launch — the skill is silently
  invisible.
* Seeding the same skill into the launched project's ``.claude/skills/``
  directory (``app.state.project_cwd``, the *project* scope) is **LIVE**.

The user-scope skill is not malformed — a control run with
``--setting-sources user,project`` discovers it — it is purely the scope flag
that hides it. The repo's prior assumption ("``--setting-sources`` governs
settings.json only; skills are a plain filesystem convention") is therefore
FALSE for the user scope.

Actionable consequence for Task 3.1 / docs (5.1): the ``skills/`` seed must land
in the web terminal's project working directory ``<project_cwd>/.claude/skills/``
(project scope), NOT only in ``$CLAUDE_CONFIG_DIR/skills/``. The CLAUDE.md
concatenation seeded into ``$CLAUDE_CONFIG_DIR`` is a separate mechanism and is
out of scope for this discovery check.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.sdk_helpers import is_claude_code_available

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

# Mirror the flag the web terminal appends unconditionally
# (osprey.utils.claude_launcher._SETTING_SOURCES_ARGS).
_SETTING_SOURCES_PROJECT = ["--setting-sources", "project"]
_SETTING_SOURCES_USER_PROJECT = ["--setting-sources", "user,project"]

_SKILLIFY = """---
name: {name}
description: {desc}
---
# {name}

Marker skill used by the OSPREY web-terminal skills-discovery smoke check.
"""


def _write_skill(skills_root: Path, name: str, desc: str) -> None:
    """Create ``<skills_root>/<name>/SKILL.md`` with valid frontmatter."""
    d = skills_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(_SKILLIFY.format(name=name, desc=desc), encoding="utf-8")


def _clean_env(config_dir: Path) -> dict[str, str]:
    """A subprocess env that reaches the init event without real credentials.

    ``CLAUDECODE``/``CLAUDE_CODE_ENTRYPOINT`` are stripped so the CLI does not
    treat this as a nested session. Any inherited Anthropic auth (including a
    proxy base URL) is removed and replaced with a deliberately-bogus key: the
    init record is emitted before the API is contacted, so the run reaches a fast
    401 without us needing — or spending — a valid provider credential.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY_o",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
        }
    }
    env["ANTHROPIC_API_KEY"] = "sk-ant-osprey-smoke-bogus"
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    return env


def _discovered_skills(*, config_dir: Path, cwd: Path, setting_sources: list[str]) -> list[str]:
    """Return the ``skills`` list from the CLI's stream-json init record.

    Invokes ``claude --print --output-format stream-json --verbose`` with the
    given ``--setting-sources`` and parses the first ``subtype == "init"``
    record. Raises ``AssertionError`` if no init record is produced (a CLI/format
    change we want to surface loudly rather than mistake for "nothing
    discovered").
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        *setting_sources,
        "--model",
        "haiku",
        "probe",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_clean_env(config_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "system" and record.get("subtype") == "init":
            skills = record.get("skills")
            assert isinstance(skills, list), (
                "init record present but has no 'skills' list — CLI schema changed?\n"
                f"init keys: {sorted(record)}"
            )
            return skills
    raise AssertionError(
        "No stream-json init record produced by the claude CLI. This test "
        "depends on the init event being emitted before the (bogus-auth) run "
        "ends.\n--- stdout ---\n"
        f"{proc.stdout[:2000]}\n--- stderr ---\n{proc.stderr[:2000]}"
    )


@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
def test_project_scope_skill_is_live_under_setting_sources_project(tmp_path: Path) -> None:
    """A skill in ``<cwd>/.claude/skills/`` IS discovered with the web-terminal flag.

    This is the LIVE path and the recommended seed target: the web terminal
    launches ``claude`` with ``cwd = app.state.project_cwd``, so seeding into
    ``<project_cwd>/.claude/skills/`` places the skill in the *project* scope that
    ``--setting-sources project`` loads.
    """
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    (config_dir / "skills").mkdir(parents=True)
    _write_skill(
        project_dir / ".claude" / "skills",
        "osprey-proj-probe",
        "Project-scope marker skill; token OSPREY_PROJ_PROBE_XYZ.",
    )

    skills = _discovered_skills(
        config_dir=config_dir,
        cwd=project_dir,
        setting_sources=_SETTING_SOURCES_PROJECT,
    )
    assert "osprey-proj-probe" in skills, (
        "A project-scope skill (<cwd>/.claude/skills/) must be discovered under "
        "--setting-sources project — this is the LIVE seed target for the web "
        f"terminal. Discovered skills: {skills}"
    )


@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
def test_user_scope_skill_is_inert_under_setting_sources_project(tmp_path: Path) -> None:
    """A skill in ``$CLAUDE_CONFIG_DIR/skills/`` is INERT with the web-terminal flag.

    This is the location the deploy currently seeds (the per-user
    ``claude-config`` volume becomes ``$CLAUDE_CONFIG_DIR``). Under the web
    terminal's ``--setting-sources project``, the *user*-scope skill is NOT
    discovered. A control run with ``--setting-sources user,project`` proves the
    skill file itself is valid — only the scope flag hides it — so this is a
    genuine scoping suppression, not a malformed-skill artifact.
    """
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_skill(
        config_dir / "skills",
        "osprey-user-probe",
        "User-scope marker skill; token OSPREY_USER_PROBE_XYZ.",
    )

    # Under the web terminal's flag: user-scope skill suppressed (INERT).
    project_only = _discovered_skills(
        config_dir=config_dir,
        cwd=project_dir,
        setting_sources=_SETTING_SOURCES_PROJECT,
    )
    assert "osprey-user-probe" not in project_only, (
        "A $CLAUDE_CONFIG_DIR/skills/ (user-scope) skill was discovered under "
        "--setting-sources project — the web terminal's launch flag was expected "
        "to SUPPRESS it. If this assertion starts failing, the seed location in "
        f"Task 3.1 may now be viable. Discovered skills: {project_only}"
    )

    # Control: with the user scope included, the very same skill IS discovered,
    # proving the file is well-formed and only the scope flag hid it above.
    with_user = _discovered_skills(
        config_dir=config_dir,
        cwd=project_dir,
        setting_sources=_SETTING_SOURCES_USER_PROJECT,
    )
    assert "osprey-user-probe" in with_user, (
        "Control run failed: the user-scope skill was not discovered even with "
        "--setting-sources user,project, so the suppression above cannot be "
        f"attributed to scoping. Discovered skills: {with_user}"
    )
