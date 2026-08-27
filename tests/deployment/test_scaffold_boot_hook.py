"""The boot-hook template, pinned to the bytes it emits.

The hook runs from cron at boot, on a host nobody is watching, in a shell with
no session and no bus — the one context where a mistake is both silent and
permanent until the next reboot. It is also a shell script, where a lost quote
or an unbalanced ``if`` is not a style problem but a job that does nothing and
mails a parse error to an account nobody reads. So this template is held to its
output byte for byte, against ``goldens/osprey-boot-hook.sh``, and the render
is additionally parsed by a real shell.

The byte specification lives here for the same reason the unit's does (see
``test_scaffold_systemd_unit``): the hook is not part of what a deployment repo
carries — it is emitted for a host, from absolute paths the repo does not know,
and wired into that account's crontab outside the repo entirely.

**Update discipline.** The golden is renderer output, never hand-edited. When a
template change is deliberate, regenerate it in the same reviewed change::

    PYTHONPATH=src .venv/bin/python -c "
    from pathlib import Path
    from tests.deployment.test_scaffold_boot_hook import GOLDEN_PATH, render_hook
    GOLDEN_PATH.write_text(render_hook(), encoding='utf-8')"

so the diff a reviewer reads is the template edit and its consequence, side by
side.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey
from osprey.cli.deploy_scaffold_templates import (
    BOOT_HOOK_LOG,
    BOOT_HOOK_LOG_DIR,
    BOOT_HOOK_MARKER,
    BOOT_HOOK_OUTPUT_NAME,
    BOOT_HOOK_PATH,
    BOOT_HOOK_POLL_SEC,
    BOOT_HOOK_TEMPLATE,
    BOOT_HOOK_TOTAL_WAIT_SEC,
    SYSTEMD_UNIT_NAME,
    boot_hook_crontab_lines,
    build_boot_hook_context,
    render,
)

GOLDENS = Path(__file__).parent / "goldens"
EXEMPLAR_DIR = GOLDENS / "exemplar-profile"
GOLDEN_PATH = GOLDENS / BOOT_HOOK_OUTPUT_NAME
DEPLOY_HOWTO = Path(__file__).parents[2] / "docs" / "source" / "how-to" / "deploy-a-facility.rst"

#: Passed in place of the installed version so the golden does not change with
#: the release the test happens to run under.
FROZEN_VERSION = "OSPREY_VERSION"

#: The deploy host's coordinates, frozen. All three are properties of a machine
#: rather than of the profile, so a render that read them off the host running
#: the tests would produce a different hook on every checkout.
FROZEN_REPO_ROOT = Path("/srv/osprey/demo-facility")
FROZEN_OSPREY_BIN = "/usr/local/bin/osprey"
FROZEN_HOME = Path("/home/osprey")

#: Where the frozen host's hook lands, as the crontab has to name it.
FROZEN_HOOK = f"{FROZEN_REPO_ROOT}/{BOOT_HOOK_PATH}"


def render_hook(profile: dict[str, Any] | None = None) -> str:
    """Render the hook for a profile, with the host's half held fixed."""
    if profile is None:
        profile = yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))
    context = build_boot_hook_context(
        profile, FROZEN_REPO_ROOT, FROZEN_OSPREY_BIN, FROZEN_VERSION, home=FROZEN_HOME
    )
    return render(BOOT_HOOK_TEMPLATE, context)


@pytest.fixture(scope="module")
def rendered() -> str:
    """The hook the exemplar renders."""
    return render_hook()


@pytest.fixture(scope="module")
def code(rendered: str) -> str:
    """The hook with its comments stripped — the lines a shell acts on.

    Asserting against the whole file would let a promise made only in a comment
    pass for the behaviour itself, which is exactly the failure mode this hook
    exists to fix.
    """
    return "\n".join(line for line in rendered.splitlines() if not line.lstrip().startswith("#"))


def test_render_matches_the_golden_byte_for_byte(rendered: str) -> None:
    """The template emits exactly what is committed under ``goldens/``."""
    assert GOLDEN_PATH.is_file(), f"missing golden fixture: {GOLDEN_PATH}"
    assert rendered == GOLDEN_PATH.read_text(encoding="utf-8")


def test_installed_version_is_the_only_value_that_moves_with_a_release(
    rendered: str,
) -> None:
    """Two renders of one profile differ in exactly the ``osprey-version:`` line.

    The scaffolder normalizes that line away before deciding whether a re-emit
    would change anything, so an upgrade rewrites no hook. A second
    release-dependent value would be masked by the same normalization and would
    silently stop re-emission from noticing a real change.
    """
    profile = yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))
    context = build_boot_hook_context(
        profile, FROZEN_REPO_ROOT, FROZEN_OSPREY_BIN, osprey.__version__, home=FROZEN_HOME
    )
    current = render(BOOT_HOOK_TEMPLATE, context)

    differing = [
        (left, right)
        for left, right in zip(rendered.splitlines(), current.splitlines(), strict=True)
        if left != right
    ]
    assert differing == [
        (f"# osprey-version: {FROZEN_VERSION}", f"# osprey-version: {osprey.__version__}")
    ]


def test_the_hook_carries_the_provenance_marker_in_its_header(rendered: str) -> None:
    """Re-emission is only safe for a file the scaffolder can recognize.

    The marker is looked for in the first twenty lines alone, so it has to be
    there — a hook that lost it would be treated as hand-written and never
    updated again.
    """
    header = rendered.splitlines()[:20]
    assert f"# osprey-scaffold: {BOOT_HOOK_MARKER}" in header


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_the_rendered_script_parses(rendered: str, tmp_path: Path, shell: str) -> None:
    """A rendered-string assertion cannot catch a shell syntax error.

    The hook's only reader is cron, which will not tell anybody the job failed
    to parse, so the emitted text is handed to a real shell here. Both are
    checked: ``sh`` because that is the interpreter the shebang names, and
    ``bash`` because on many hosts ``/bin/sh`` *is* bash and would quietly
    accept a bashism that a dash-based host rejects at boot.
    """
    binary = shutil.which(shell)
    if binary is None:
        pytest.skip(f"{shell} is not installed on this host")

    script = tmp_path / BOOT_HOOK_OUTPUT_NAME
    script.write_text(rendered, encoding="utf-8")
    result = subprocess.run(
        [binary, "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_hook_exports_the_user_managers_runtime_directory(code: str) -> None:
    """The one line without which the whole script is a no-op.

    A ``cron @reboot`` job inherits no ``XDG_RUNTIME_DIR`` and no session bus,
    and ``systemctl --user`` without one does not fail loudly — it just fails.
    This is the classic reason a hand-written boot hook appears to run and
    changes nothing, so it gets its own pin rather than riding on the golden.
    """
    assert 'XDG_RUNTIME_DIR="/run/user/$(id -u)"' in code
    assert "export XDG_RUNTIME_DIR" in code


def test_the_hook_waits_for_every_late_path_before_touching_systemd(code: str) -> None:
    """The mount, the deployment, the executable, and the manager itself.

    All four can be absent when cron fires: the first three sit under the home
    that has not been mounted yet, and the runtime directory is created by the
    user manager, so its appearance is the only signal that there is a manager
    to talk to.
    """
    waits = [line.strip() for line in code.splitlines() if line.strip().startswith("wait_for ")]
    assert waits == [
        'wait_for "$HOME" "the home directory"',
        f'wait_for "{FROZEN_REPO_ROOT}" "the deployment repo"',
        f'wait_for "{FROZEN_OSPREY_BIN}" "the osprey executable"',
        'wait_for "$XDG_RUNTIME_DIR" "the user manager\'s runtime directory"',
    ]
    assert code.index('wait_for "$XDG_RUNTIME_DIR"') < code.index("systemctl --user daemon-reload")


def test_the_wait_is_bounded_and_says_what_never_arrived(code: str) -> None:
    """An unbounded wait holds a cron slot open on a host nobody is watching.

    Giving up has to be loud: cron mails what the job prints, and that mail is
    the only account of a boot that did not come back.
    """
    assert f"POLL_SECONDS={BOOT_HOOK_POLL_SEC}" in code
    assert f"TOTAL_WAIT_SECONDS={BOOT_HOOK_TOTAL_WAIT_SEC}" in code
    assert 'sleep "$POLL_SECONDS"' in code
    assert "die " in code and "gave up after" in code
    assert BOOT_HOOK_POLL_SEC < BOOT_HOOK_TOTAL_WAIT_SEC


def test_the_hook_reloads_before_it_starts(code: str) -> None:
    """Reloading is the entire point: the manager read its unit search path
    once, before the home was there, so without this the start command finds no
    such unit."""
    reload_at = code.index("systemctl --user daemon-reload")
    start_at = code.index(f"systemctl --user start {SYSTEMD_UNIT_NAME}")
    assert reload_at < start_at


def test_running_it_twice_is_harmless(code: str) -> None:
    """An operator debugging a failed boot runs the hook by hand, on a host
    where the unit may already be up. It has to leave an active deployment
    alone rather than restart it underneath its users."""
    assert f"systemctl --user is-active --quiet {SYSTEMD_UNIT_NAME}" in code
    guard_at = code.index("is-active")
    start_at = code.index(f"systemctl --user start {SYSTEMD_UNIT_NAME}")
    assert guard_at < start_at


def test_the_header_shows_the_crontab_lines_that_install_it(rendered: str) -> None:
    """The hook is wired up by hand, so the file has to say how — every line.

    The lines come from the same function the console prints them from, and
    the job names the hook by the same constant the emitter writes to, so what
    an operator pastes cannot drift from where the file lands or from what the
    verb told them.
    """
    header = rendered[: rendered.index("set -u")]
    lines = boot_hook_crontab_lines(FROZEN_HOOK)
    assert "crontab -e" in header
    assert "".join(f"#   {line}\n" for line in lines) in header


def test_the_crontab_job_survives_a_home_that_is_not_there_yet() -> None:
    """Two things kill a cron job before its command runs on this host.

    cron changes into the crontab's ``HOME`` first and dies silently when it is
    missing, so the preamble has to hand it a directory that exists at boot.
    Then ``sh`` has to read the script, which sits on the same late mount, so
    the job — on the local disk, in the crontab — must wait for the file
    itself before running it, and must say it fired somewhere that is not the
    home before it waits. Both were learned from real reboots, where a bare
    ``@reboot <hook>`` never launched at all.
    """
    shell_line, home_line, job = boot_hook_crontab_lines(FROZEN_HOOK)
    # The job is POSIX sh; an existing crontab may have set SHELL to a csh.
    assert shell_line == "SHELL=/bin/sh"
    assert home_line == "HOME=/"
    assert job.startswith("@reboot ")
    # cron reads a `%` as a newline, and the job would be cut there.
    assert "%" not in job
    fired_at = job.index('cron fired" >> "$log"')
    wait_at = job.index(f"until [ -x {FROZEN_HOOK} ]")
    run_at = job.index(f"exec {FROZEN_HOOK}")
    assert fired_at < wait_at < run_at
    # Bounded by the same budget the script uses, and loud when it runs out.
    assert f"[ $n -ge {BOOT_HOOK_TOTAL_WAIT_SEC // BOOT_HOOK_POLL_SEC} ]" in job
    assert f"sleep {BOOT_HOOK_POLL_SEC}" in job
    # On stdout as well as in the log: this is the branch cron's mail is for.
    assert 'never appeared" | tee -a "$log"' in job


def test_both_writers_refuse_a_log_directory_they_do_not_own() -> None:
    """``/tmp`` is shared and the name is predictable.

    Appending to a bare file there follows a symlink any local user could
    have planted, turning the log into an append to a file of their choosing
    as the deploying account. So the log lives in a directory each writer
    creates with mode 700 and uses only if it is a real directory it owns —
    ``/tmp``'s sticky bit keeps anyone else from swapping it out afterwards.
    """
    *_, job = boot_hook_crontab_lines(FROZEN_HOOK)
    assert BOOT_HOOK_LOG == f"{BOOT_HOOK_LOG_DIR}/boot.log"
    assert f'd={BOOT_HOOK_LOG_DIR}; mkdir -m 700 "$d"' in job
    assert 'if [ -d "$d" ] && [ ! -L "$d" ] && [ -O "$d" ]; then log=' in job
    assert "else log=/dev/null; fi" in job
    assert job.index("mkdir -m 700") < job.index("cron fired")


def test_the_script_guards_its_log_directory_the_same_way(code: str) -> None:
    """Same directory, same ownership check, before the first write."""
    assert f'LOG_DIR="{BOOT_HOOK_LOG_DIR}"' in code
    assert 'mkdir -m 700 "$LOG_DIR"' in code
    guard = 'if [ -d "$LOG_DIR" ] && [ ! -L "$LOG_DIR" ] && [ -O "$LOG_DIR" ]; then'
    assert guard in code
    assert code.index(guard) < code.index(f'LOG="{BOOT_HOOK_LOG}"') < code.index("launched")


def test_the_script_restores_the_real_home_before_anything_else(code: str) -> None:
    """The crontab sets ``HOME=/`` so cron can start the job at all.

    Everything the script waits for sits under the real home, so it has to
    put that back first — from a literal the scaffolder knew, not from
    ``$HOME`` (which is ``/``) or from ``getent`` (which asks an identity
    service that may not be up yet).
    """
    restore_at = code.index(f'HOME="{FROZEN_HOME}"')
    assert "export HOME" in code
    assert restore_at < code.index("export HOME") < code.index("wait_for ")


def test_the_launch_marker_is_written_before_the_first_wait(code: str) -> None:
    """A boot on which the home never came leaves no other trace.

    The marker goes to the local disk, appended after the line the crontab job
    already wrote, and before the script waits for anything — so the log
    splits "cron never fired" from "still waiting" from "ran and said why".
    """
    assert f'LOG="{BOOT_HOOK_LOG}"' in code
    marker_at = code.index("launched")
    assert '>> "$LOG"' in code[marker_at : marker_at + 120]
    assert marker_at < code.index('wait_for "$HOME"')
    # Every later line lands in the same file.
    assert 'tee -a "$LOG"' in code


def test_the_how_to_shows_the_same_crontab_job() -> None:
    """The docs cannot render the line, so they carry a copy — pinned here.

    The how-to spells the hook as ``/path/to/repo/...``; the job around it has
    to be the one the verb prints, or an operator following the page gets a
    job the field evidence says does not launch.
    """
    docs = DEPLOY_HOWTO.read_text(encoding="utf-8")
    lines = boot_hook_crontab_lines(f"/path/to/repo/{BOOT_HOOK_PATH}")
    assert "".join(f"   {line}\n" for line in lines) in docs


def test_the_header_names_the_facility_and_the_unit(rendered: str) -> None:
    """One host can carry a hook per deployment, and cron mails all of their
    output to the same account."""
    header = rendered[: rendered.index("set -u")]
    assert "Demo Facility" in header
    assert SYSTEMD_UNIT_NAME in header


def test_the_script_stays_posix(code: str) -> None:
    """``sh -n`` under a shell that happens to be bash would accept several of
    these, so they are named explicitly. The shebang promises ``/bin/sh``, which
    on a Debian-family deploy host is dash."""
    for bashism in ("[[", "declare ", "local ", "function ", "+=("):
        assert bashism not in code, f"bashism in a POSIX sh script: {bashism!r}"
