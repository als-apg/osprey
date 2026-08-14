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
record. That keeps this test runnable in CI without spending tokens or requiring
VPN/provider access.

We read that init record off the stream and then KILL the CLI, rather than
waiting for it to exit. The exit path is not part of the contract under test and
is not ours to depend on: with an unusable credential the CLI retries the API
call with backoff (``{"type":"system","subtype":"api_retry"}``) rather than
terminating promptly, so waiting for exit wedges the test for as long as the
retry schedule runs while the answer has already been sitting in stdout since
the first second.

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
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from tests.e2e.sdk_helpers import is_claude_code_available

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

# Mirror the flag the web terminal appends unconditionally
# (osprey.utils.claude_launcher._SETTING_SOURCES_ARGS).
_SETTING_SOURCES_PROJECT = ["--setting-sources", "project"]
_SETTING_SOURCES_USER_PROJECT = ["--setting-sources", "user,project"]

# The init record is emitted at session start, before the model is contacted, so
# it lands in about a second. This bound only has to cover CLI startup on a cold,
# loaded runner — it is a wedged-process backstop, not an expected wait.
_INIT_RECORD_TIMEOUT_S = 60.0

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
    init record is emitted before the API is contacted, so we get the signal
    without needing — or spending — a valid provider credential. What the CLI
    does with the doomed request afterwards is immaterial; the caller stops
    reading and kills it once the init record is in hand.
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


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the CLI's whole process group, tolerating an already-dead child."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _discovered_skills(*, config_dir: Path, cwd: Path, setting_sources: list[str]) -> list[str]:
    """Return the ``skills`` list from the CLI's stream-json init record.

    Invokes ``claude --print --output-format stream-json --verbose`` with the
    given ``--setting-sources`` and returns the first ``subtype == "init"``
    record's skills, killing the CLI the moment that record is in hand — see the
    module docstring on why the exit path is deliberately not awaited.

    Raises ``AssertionError`` if the stream ends without an init record (a
    CLI/format change we want to surface loudly rather than mistake for "nothing
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
    # stderr goes to a file, not a pipe: we stop reading stdout as soon as the
    # init record lands, and an unread stderr pipe would let the CLI block
    # forever on a full buffer instead of dying to the kill below.
    stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        cmd,
        cwd=str(cwd),
        env=_clean_env(config_dir),
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        bufsize=1,
        # Own process group, so the teardown below reaps the CLI's children too.
        # We kill it mid-run rather than letting it exit, and a bare proc.kill()
        # would signal only the node parent and orphan anything it spawned.
        start_new_session=True,
    )
    # readline() blocks, so a deadline between lines cannot bound a CLI that has
    # gone quiet. Killing the child is what makes stdout reach EOF and the loop
    # below terminate.
    watchdog = threading.Timer(_INIT_RECORD_TIMEOUT_S, lambda: _kill_group(proc))
    watchdog.start()
    stdout_seen: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_seen.append(line)
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "system" and record.get("subtype") == "init":
                skills = record.get("skills")
                assert isinstance(skills, list), (
                    "init record present but has no 'skills' list — CLI schema changed?\n"
                    f"init keys: {sorted(record)}"
                )
                return skills
    finally:
        watchdog.cancel()
        _kill_group(proc)
        proc.wait(timeout=30)
        if proc.stdout is not None:
            proc.stdout.close()
        stderr_file.seek(0)
        stderr_text = stderr_file.read()
        stderr_file.close()
    raise AssertionError(
        "No stream-json init record produced by the claude CLI before the stream "
        f"ended or {_INIT_RECORD_TIMEOUT_S:.0f}s elapsed. This test depends on the "
        "init event being emitted at session start.\n--- stdout ---\n"
        f"{''.join(stdout_seen)[:2000]}\n--- stderr ---\n{stderr_text[:2000]}"
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
