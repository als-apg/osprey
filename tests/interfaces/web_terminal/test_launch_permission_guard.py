"""Guard: the web-terminal launch path never enables a permission-bypassing mode.

OSPREY's human-in-the-loop guarantee for the web terminal rests on the
``permissions.deny`` array in the rendered ``.claude/settings.json``: ``Bash``
and ``Edit`` are denied there, so the agent cannot shell out or patch files
without going through a tool that carries its own approval.

That array is only authoritative while the agent runs under normal permission
arbitration. A Claude Code process started with ``--dangerously-skip-permissions``,
or with a permission mode of ``bypassPermissions``/``acceptEdits``, ignores the
deny list entirely — the settings file would still render exactly as before, and
every other test in the suite would still pass, while the safety property was
silently gone. This module is the test that would fail.

**Scope is the web-terminal launch path only, deliberately — not a repo-wide grep.**
``tests/e2e/sdk_helpers.py`` legitimately drives the SDK with
``permission_mode="bypassPermissions"``, and the headless runner in
``osprey.agent_runner.primitives`` defaults to it, because those paths are
read-only by construction (they pass an architectural ``disallowed_tools`` guard
instead). Those are different launch paths with different guarantees; widening
this guard to the whole repo would fail on them and teach the next person to
delete it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.utils.claude_launcher import build_claude_launch_argv

REPO_ROOT = Path(__file__).resolve().parents[3]

# Every source surface that participates in starting the web terminal's agent:
# the argv builder itself, the deployment renderers that materialise a project,
# the templates they render from, and the two callers that hand the argv to a PTY.
LAUNCH_PATH = (
    "src/osprey/utils/claude_launcher.py",
    "src/osprey/deployment/web_terminals",
    "src/osprey/templates/modules/web_terminals",
    "src/osprey/templates/claude_code",
    "src/osprey/interfaces/web_terminal/app.py",
    "src/osprey/cli/web_cmd.py",
)

# Restricted to surfaces that can actually launch or configure a process.
# Markdown is excluded on purpose: a rule or skill doc may legitimately *name*
# a bypass mode in prose ("never run in bypassPermissions mode") without
# enabling one, and a guard that fires on documentation is a guard people delete.
SCANNED_SUFFIXES = frozenset({".py", ".j2", ".json", ".yml", ".yaml", ".sh", ".toml"})

# Tokens with no legitimate use on this path: the two permissive SDK/CLI modes
# plus both spellings of the skip-permissions flag.
BYPASS_TOKENS = (
    "--dangerously-skip-permissions",
    "dangerouslySkipPermissions",
    "bypassPermissions",
    "acceptEdits",
)


def _launch_path_files() -> list[Path]:
    """Every scannable file on the web-terminal launch path."""
    found: list[Path] = []
    for entry in LAUNCH_PATH:
        target = REPO_ROOT / entry
        assert target.exists(), (
            f"Launch-path entry {entry!r} no longer exists. This guard is only "
            f"meaningful while it covers the real launch path — update LAUNCH_PATH "
            f"to follow the code, do not just drop the entry."
        )
        if target.is_file():
            found.append(target)
            continue
        found.extend(
            p
            for p in target.rglob("*")
            if p.is_file() and p.suffix in SCANNED_SUFFIXES and "__pycache__" not in p.parts
        )
    return found


class TestLaunchArgvCarriesNoPermissionBypass:
    """``build_claude_launch_argv()`` is the single choke point for the argv."""

    @pytest.mark.parametrize(
        "cc_config,no_pin",
        [
            ({}, False),
            ({"cli_version": "2.1.146"}, False),
            ({"cli_version": "2.1.146"}, True),
            ({"provider": "anthropic", "default_model": "haiku"}, False),
        ],
    )
    def test_argv_contains_no_bypass_token(self, cc_config, no_pin):
        argv = build_claude_launch_argv(cc_config, no_pin=no_pin)
        joined = " ".join(argv)
        for token in BYPASS_TOKENS:
            assert token not in joined, (
                f"build_claude_launch_argv({cc_config!r}, no_pin={no_pin}) emitted "
                f"{token!r}. The web terminal's permissions.deny array stops being "
                f"authoritative the moment the agent launches in a bypassing mode."
            )

    @pytest.mark.parametrize("cc_config", [{}, {"cli_version": "2.1.146"}])
    def test_argv_does_not_choose_a_permission_mode_at_all(self, cc_config):
        """The launch argv leaves permission arbitration to the settings file.

        Passing any ``--permission-mode`` on the command line would override the
        project's settings; the safe default is to pass none and let
        ``.claude/settings.json`` decide.
        """
        argv = build_claude_launch_argv(cc_config)
        assert "--permission-mode" not in argv
        assert "--permissionMode" not in argv


class TestLaunchPathSourcesCarryNoPermissionBypass:
    """No file on the launch path may introduce a bypass by another route.

    The argv test above only covers the one function. A future change could just
    as easily set a bypassing ``defaultMode`` in the settings template, or spawn
    the CLI from a different call site with the flag attached — neither would
    touch ``build_claude_launch_argv()``.
    """

    def test_scan_covers_a_plausible_file_set(self):
        """Fail loudly if the scan silently matches nothing (a vacuous pass)."""
        files = _launch_path_files()
        assert len(files) > 10, (
            f"Only {len(files)} files scanned on the launch path — the guard has "
            f"probably lost its target and would now pass vacuously."
        )

    def test_no_bypass_token_on_the_launch_path(self):
        offenders: list[str] = []
        for path in _launch_path_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for token in BYPASS_TOKENS:
                    if token in line:
                        rel = path.relative_to(REPO_ROOT)
                        offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "A permission-bypassing mode reached the web-terminal launch path:\n  "
            + "\n  ".join(offenders)
            + "\n\nThe web terminal's safety model assumes the agent runs under "
            "normal permission arbitration, so that the permissions.deny array in "
            "the rendered settings.json actually binds. If this is intentional, the "
            "deny-array argument has to be replaced with a different one — not just "
            "this assertion relaxed."
        )


class TestDenyArrayIsTheThingBeingProtected:
    """Pin the other half of the invariant: the deny defaults still exist.

    The bypass guard above protects the deny array's authority. This protects the
    array itself — together they state the whole property. Without this, deleting
    ``Bash`` from the template would leave the bypass guard passing and guarding
    nothing.
    """

    def test_settings_template_denies_bash_and_edit_by_default(self):
        """Read the constant the template renders, not the template's source.

        settings.json.j2 no longer declares the list inline — it takes
        ``deny_defaults`` from the render context, sourced from DENY_DEFAULTS —
        so scraping the Jinja file would now guard a copy nothing ships.
        """
        from osprey.cli.templates.claude_code import DENY_DEFAULTS

        assert DENY_DEFAULTS, (
            "DENY_DEFAULTS is empty — the web terminal's tool restrictions are "
            "rendered from it into settings.json's permissions.deny."
        )
        for tool in ("Bash", "Edit"):
            assert tool in DENY_DEFAULTS, (
                f"{tool!r} is no longer in DENY_DEFAULTS ({list(DENY_DEFAULTS)}), "
                f"which settings.json.j2 renders into permissions.deny. The web "
                f"terminal relies on it to keep the agent from shelling out or "
                f"patching files unmediated."
            )
