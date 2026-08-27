"""``osprey scaffold systemd`` — the verb, not the unit it renders.

The rendered bytes are pinned elsewhere: ``tests/deployment/
test_scaffold_systemd_unit.py`` holds the template to a golden, directive by
directive. What is tested here is everything around the render — where the file
lands, which repo the verb decides it is acting on, what happens when something
is already at that path, and what an operator is told to do with the file once
it is written.

Three of those are worth naming.

*The host, not the profile.* The unit is rendered from a repo with its
``deploy:`` block commented out, which is what ``osprey init`` emits. A verb
that reached for deployment coordinates it does not need would fail here, and
would fail on every repo that is deployed by hand.

*The two absolute paths.* A unit starts with no working directory and a short
PATH, so both the repo and the ``osprey`` program have to be written into it in
full. The executable is resolved through a helper that reads the machine, so
every test here freezes it: a unit whose contents depend on the developer's
PATH is a unit nobody can pin.

*The instruction.* The unit is written to the repo root and takes effect from
``~/.config/systemd/user/``, so the verb's output is the only thing standing
between "emitted" and "installed".

*The pair.* The verb emits two files with independent histories — the unit, and
the boot hook under ``scripts/`` — and the printing has to keep them apart. A
hand-written hook is a refusal, but it is not a reason to stop telling an
operator how to install the unit they did just get.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_scaffold import (
    BOOT_HOOK_OUTPUT_PATH,
    SYSTEMD_OUTPUT_NAME,
    detect_network_home,
    scaffold_systemd_unit,
)
from osprey.cli.deploy_scaffold_templates import (
    BOOT_HOOK_MARKER,
    SYSTEMD_MARKER,
    boot_hook_crontab_lines,
)
from osprey.cli.main import cli
from osprey.cli.scaffold_cmd import scaffold
from osprey.errors import ConfigurationError
from tests.fixtures.lifecycle_repo import build_exemplar_repo

#: The ``osprey`` the emitted unit names. Frozen, because the real resolver
#: reads PATH and the user-local bin directories of whoever runs the tests.
FROZEN_BIN = "/opt/osprey/bin/osprey"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def frozen_osprey_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the executable the unit is rendered against.

    Patched where it is used rather than where it is defined: the templates
    module imported the name, so patching the helper's own module would leave
    this binding pointing at the real one.
    """
    monkeypatch.setattr(
        "osprey.cli.deploy_scaffold_templates.resolve_shell_command",
        lambda command: FROZEN_BIN,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The exemplar repo as ``osprey init`` leaves it: no deploy coordinates."""
    return build_exemplar_repo(tmp_path / "als-exemplar")


def emit(runner: CliRunner, repo: Path, *args: str):
    """Run the verb against *repo* from outside it."""
    return runner.invoke(scaffold, ["systemd", "--repo", str(repo), *args])


def unit_of(repo: Path) -> Path:
    return repo / SYSTEMD_OUTPUT_NAME


def hook_of(repo: Path) -> Path:
    return repo.joinpath(*BOOT_HOOK_OUTPUT_PATH)


# ── What it writes, and where ────────────────────────────────────────────────


def test_the_unit_lands_at_the_repo_root(runner: CliRunner, repo: Path) -> None:
    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    unit = unit_of(repo)
    assert unit.is_file()
    assert unit.name == "osprey.service"
    assert f"osprey-scaffold: {SYSTEMD_MARKER}" in unit.read_text(encoding="utf-8")


def test_a_repo_with_no_deploy_block_still_gets_a_unit(runner: CliRunner, repo: Path) -> None:
    """The unit runs the deployment in place, so it reads no coordinates."""
    profile = (repo / "profile.yml").read_text(encoding="utf-8")
    assert "\ndeploy:" not in profile, "the fixture is supposed to have no deploy: block"

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit_of(repo).is_file()


def test_both_host_paths_are_written_in_full(runner: CliRunner, repo: Path) -> None:
    """A unit inherits no working directory and a short PATH."""
    emit(runner, repo)

    text = unit_of(repo).read_text(encoding="utf-8")
    assert f"WorkingDirectory={repo}" in text
    assert f"ExecStart={FROZEN_BIN} up -d" in text
    assert f"ExecStop={FROZEN_BIN} down" in text


def test_the_description_comes_from_the_profile(runner: CliRunner, repo: Path) -> None:
    emit(runner, repo)

    assert "Description=Als Exemplar OSPREY deployment" in unit_of(repo).read_text(encoding="utf-8")


def test_no_temporary_file_is_left_behind(runner: CliRunner, repo: Path) -> None:
    """The write goes through a temp file in the destination directory."""
    emit(runner, repo)

    assert not list(repo.glob("*.tmp"))
    assert not list(repo.glob(".osprey.service.*"))


def test_the_hook_lands_under_scripts(runner: CliRunner, repo: Path) -> None:
    """One run, two files: the unit at the root and the hook beside it."""
    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    hook = hook_of(repo)
    assert hook.is_file()
    assert hook.parent == repo / "scripts"
    assert hook.name == "osprey-boot-hook.sh"
    assert f"osprey-scaffold: {BOOT_HOOK_MARKER}" in hook.read_text(encoding="utf-8")


def test_both_files_are_reported(runner: CliRunner, repo: Path) -> None:
    """Each emitted file gets its own status line, named repo-relative."""
    result = emit(runner, repo)

    text = flat(result)
    assert f"{SYSTEMD_OUTPUT_NAME} (created)" in text
    assert "scripts/osprey-boot-hook.sh (created)" in text


def test_the_hook_is_executable_and_the_unit_is_not(runner: CliRunner, repo: Path) -> None:
    """cron runs the hook directly; systemd only reads the unit."""
    emit(runner, repo)

    assert hook_of(repo).stat().st_mode & 0o777 == 0o755
    assert unit_of(repo).stat().st_mode & 0o777 == 0o644


# ── Which repo it acts on ────────────────────────────────────────────────────


def test_it_finds_the_repo_from_a_subdirectory(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --repo means the repo the operator is standing in, at any depth."""
    inside = repo / "data" / "raw"
    inside.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(inside)

    result = runner.invoke(scaffold, ["systemd"])

    assert result.exit_code == 0, result.output
    assert unit_of(repo).is_file()


def test_outside_a_repo_it_reports_the_missing_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(scaffold, ["systemd"])

    assert result.exit_code == 1
    assert "profile.yml" in result.output


def test_a_profileless_directory_is_told_to_re_run_this_verb(tmp_path: Path) -> None:
    """The engine names its caller, since two verbs render from one profile.

    Reached directly rather than through the command: repo discovery is what
    finds the profile, so a CLI run that got past it has one by construction.
    The engine is called by ``osprey init`` too, and the message it raises is
    the one an operator sees.
    """
    with pytest.raises(ConfigurationError) as raised:
        scaffold_systemd_unit(tmp_path)

    assert "osprey scaffold systemd" in str(raised.value)


# ── Re-running ───────────────────────────────────────────────────────────────


def test_a_second_run_writes_nothing(runner: CliRunner, repo: Path) -> None:
    emit(runner, repo)
    unit = unit_of(repo)
    first = unit.read_bytes()

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit.read_bytes() == first
    assert "unchanged" in result.output


def test_a_second_run_reports_both_files_as_unchanged(runner: CliRunner, repo: Path) -> None:
    """Neither file may drift on a re-run, and both have to say so."""
    emit(runner, repo)
    before = (unit_of(repo).read_bytes(), hook_of(repo).read_bytes())

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert (unit_of(repo).read_bytes(), hook_of(repo).read_bytes()) == before
    text = flat(result)
    assert f"{SYSTEMD_OUTPUT_NAME} (unchanged)" in text
    assert "scripts/osprey-boot-hook.sh (unchanged)" in text


def test_a_stale_version_stamp_alone_is_not_a_change(runner: CliRunner, repo: Path) -> None:
    """An OSPREY upgrade on its own must not produce a diff."""
    emit(runner, repo)
    unit = unit_of(repo)
    aged = unit.read_text(encoding="utf-8").replace("osprey-version:", "osprey-version: 0.0.1 #")
    unit.write_text(aged, encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit.read_text(encoding="utf-8") == aged
    assert "unchanged" in result.output


def test_a_moved_repo_rewrites_the_unit(runner: CliRunner, repo: Path, tmp_path: Path) -> None:
    """The paths in the unit are the point, so a real change is written."""
    emit(runner, repo)
    unit = unit_of(repo)
    moved = unit.read_text(encoding="utf-8").replace(
        f"WorkingDirectory={repo}", "WorkingDirectory=/old"
    )
    unit.write_text(moved, encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert f"WorkingDirectory={repo}" in unit.read_text(encoding="utf-8")
    assert "updated" in result.output


# ── A file we did not write ──────────────────────────────────────────────────


def test_a_hand_written_unit_is_left_alone(runner: CliRunner, repo: Path) -> None:
    unit = unit_of(repo)
    unit.write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert unit.read_text(encoding="utf-8") == "[Unit]\nDescription=mine\n"
    assert "osprey scaffold systemd --force" in result.output


def test_force_replaces_it(runner: CliRunner, repo: Path) -> None:
    unit = unit_of(repo)
    unit.write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo, "--force")

    assert result.exit_code == 0, result.output
    assert f"osprey-scaffold: {SYSTEMD_MARKER}" in unit.read_text(encoding="utf-8")


# ── No osprey to name ────────────────────────────────────────────────────────


def test_an_unresolvable_osprey_writes_nothing(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit naming a program that is not there fails at boot, unwatched."""

    def missing(command: str) -> str:
        raise FileNotFoundError(f"{command!r} was not found on PATH.")

    monkeypatch.setattr("osprey.cli.deploy_scaffold_templates.resolve_shell_command", missing)

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert not unit_of(repo).exists()
    assert "command -v osprey" in result.output


# ── What the operator is told ────────────────────────────────────────────────


def test_the_output_says_how_to_install_it(runner: CliRunner, repo: Path) -> None:
    result = emit(runner, repo)

    # The copy names the file in full: an operator who passed --repo, or who is
    # standing in a subdirectory, cannot copy a repo-relative path.
    assert f"cp {unit_of(repo)} ~/.config/systemd/user/" in " ".join(result.output.split())
    assert "systemctl --user daemon-reload" in result.output
    assert "systemctl --user enable --now osprey.service" in result.output


def test_the_output_says_what_boot_also_needs(runner: CliRunner, repo: Path) -> None:
    """A user unit that nobody is logged in for does not start on its own."""
    result = emit(runner, repo)

    assert "loginctl enable-linger" in result.output


def test_a_refusal_prints_no_install_instruction(runner: CliRunner, repo: Path) -> None:
    """Nothing was written, so there is nothing to install."""
    unit_of(repo).write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert "systemctl --user enable" not in result.output


# ── Reachable as an osprey verb ──────────────────────────────────────────────


def test_the_verb_is_reachable_through_the_top_level_group(runner: CliRunner) -> None:
    """The group is lazily loaded, so the help path is worth exercising."""
    result = runner.invoke(cli, ["scaffold", "systemd", "--help"])

    assert result.exit_code == 0, result.output
    assert "--repo" in result.output
    assert "--force" in result.output


def test_the_group_help_lists_the_verb(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["scaffold", "--help"])

    assert result.exit_code == 0, result.output
    assert "systemd" in result.output


# ── A home the user manager cannot see at boot ───────────────────────────────


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``$HOME`` the test owns, so the warning has a path to name."""
    account = tmp_path / "home" / "operator"
    account.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(account))
    return account


@pytest.fixture(autouse=True)
def findmnt(monkeypatch: pytest.MonkeyPatch):
    """Own the detection seam: is ``findmnt`` there, and what does it say.

    Both halves are patched in the module that reaches for them — the binary
    lookup as well as the call — because the tests run on macOS too, where
    ``findmnt`` does not exist and an unpatched lookup would answer for every
    case at once.

    Autouse, so the seam is closed from setup even for the tests that never
    mention it: ``shutil.which`` answers ``None``, which is the quiet default
    (no binary, no warning, and the answer those tests already expect), and
    ``subprocess.run`` is replaced by a callable that raises rather than
    letting an unpatched path shell out. Without that, every test that calls
    ``emit`` would run the real detection against the machine's own ``$HOME``
    — a live ``findmnt`` per test on Linux, with a two-second timeout hanging
    off it. No assertion here depends on the outcome, so nothing was failing;
    what the default removes is the subprocess and the machine-dependence.

    The returned ``install`` is how a test opts into a specific answer, exactly
    as before; it patches over the default and returns a fresh list of the
    argv the seam is called with. A test that patches the same two names by
    hand also wins, because its patches land after this setup.
    """

    def unexpected_run(argv, **kwargs):
        raise AssertionError(
            f"the findmnt seam was called without an answer installed: argv={list(argv)}"
        )

    monkeypatch.setattr("osprey.cli.deploy_scaffold.shutil.which", lambda name: None)
    monkeypatch.setattr("osprey.cli.deploy_scaffold.subprocess.run", unexpected_run)

    def install(fstype: str = "", *, present: bool = True, returncode: int = 0) -> list[list[str]]:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "osprey.cli.deploy_scaffold.shutil.which",
            lambda name: "/usr/bin/findmnt" if present and name == "findmnt" else None,
        )

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(list(argv), returncode, f"{fstype}\n", "")

        monkeypatch.setattr("osprey.cli.deploy_scaffold.subprocess.run", fake_run)
        return calls

    return install


def flat(result) -> str:
    """The output with its wrapping collapsed, for asserting on prose."""
    return " ".join(result.output.split())


def test_an_nfs_home_gets_the_root_only_drop_in(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """Linger alone loses the unit here, and the fix is not ours to apply."""
    findmnt("nfs")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    text = flat(result)
    assert "nfs" in text
    assert str(home) in text
    assert "linger alone" in text
    assert f"/etc/systemd/system/user@{os.getuid()}.service.d/network-home.conf" in text
    assert "[Unit]" in text
    assert f"RequiresMountsFor={home}" in text
    assert "After=remote-fs.target autofs.service" in text
    assert "sudo systemctl daemon-reload" in text


def test_an_autofs_home_gets_it_too(runner: CliRunner, repo: Path, home: Path, findmnt) -> None:
    findmnt("autofs")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert "autofs" in flat(result)
    assert f"RequiresMountsFor={home}" in flat(result)


def test_the_warning_does_not_replace_the_install_instructions(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """It is a warning beside them, not a different set of instructions."""
    findmnt("nfs4")

    result = emit(runner, repo)

    assert "systemctl --user daemon-reload" in result.output
    assert "systemctl --user enable --now osprey.service" in result.output
    assert "loginctl enable-linger" in result.output
    assert "RequiresMountsFor" in flat(result)


def test_the_home_is_the_path_findmnt_is_asked_about(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    calls = findmnt("nfs")

    emit(runner, repo)

    assert calls == [["/usr/bin/findmnt", "-T", str(home), "-no", "FSTYPE"]]


def test_a_local_home_says_nothing_extra(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    findmnt("ext4")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert "RequiresMountsFor" not in flat(result)
    assert "loginctl enable-linger" in result.output


def test_no_findmnt_says_nothing_extra(runner: CliRunner, repo: Path, home: Path, findmnt) -> None:
    """macOS and minimal containers have no findmnt, and get no guesswork."""
    findmnt(present=False)

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert "RequiresMountsFor" not in flat(result)


def test_a_findmnt_that_fails_says_nothing_extra(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    findmnt("nfs", returncode=1)

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert "RequiresMountsFor" not in flat(result)


def test_a_refusal_prints_no_drop_in_either(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """Nothing was written, so there is nothing for a mount to hide."""
    findmnt("nfs")
    unit_of(repo).write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert "RequiresMountsFor" not in flat(result)


def test_a_hand_written_hook_still_leaves_the_unit_fully_explained(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """A refused hook must not swallow the unit's instructions.

    The two files have independent histories. A hook somebody wrote by hand is
    refused and makes the run non-zero, but the unit beside it was written
    perfectly — and on a network home the drop-in is the whole reason this
    warning exists. Gating any of that on "did anything refuse" would hide the
    drop-in from exactly the operator who needs it.
    """
    findmnt("nfs")
    hook = hook_of(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\necho mine\n"
    text = flat(result)
    assert "osprey scaffold systemd --force" in text
    assert f"cp {unit_of(repo)} ~/.config/systemd/user/" in text
    assert "systemctl --user enable --now osprey.service" in text
    assert "loginctl enable-linger" in text
    assert f"RequiresMountsFor={home}" in text


def test_a_network_home_says_how_to_wire_the_hook_up(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """The no-root route: the hook's full path, and both crontab lines.

    The job is the one :func:`boot_hook_crontab_lines` builds — the same
    spelling the hook's own header carries — printed without Rich reading its
    ``[ -x ... ]`` tests as style tags.
    """
    findmnt("nfs")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    text = flat(result)
    home_line, job = boot_hook_crontab_lines(str(hook_of(repo)))
    assert str(hook_of(repo)) in text
    assert "crontab -e" in text
    assert home_line in text
    assert job in text
    assert text.index("HOME=/") < text.index("@reboot")


def test_the_drop_in_is_scoped_to_mounts_systemd_manages(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """A home served by the autofs daemon has no mount unit to order against.

    ``findmnt`` says ``autofs`` for a systemd automount too, so the verb cannot
    tell the two apart and has to say where the drop-in applies rather than
    pretend it always does. Where it does apply, cron only repairs each boot
    after it — and that is still said, in the order an operator reads it: root
    first, cron after.
    """
    findmnt("autofs")

    result = emit(runner, repo)

    text = flat(result)
    assert "When systemd manages the mount" in text
    assert "autofs daemon has no mount unit" in text
    assert "fallback there, not a replacement" in text
    assert text.index("RequiresMountsFor") < text.index("@reboot")


def test_a_local_home_never_mentions_the_hook(
    runner: CliRunner, repo: Path, home: Path, findmnt
) -> None:
    """The hook is written either way, but wiring it up here is noise."""
    findmnt("ext4")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    text = flat(result)
    assert "@reboot" not in text
    assert "crontab" not in text
    assert str(hook_of(repo)) not in text


# ── The detection itself ─────────────────────────────────────────────────────


@pytest.mark.parametrize("fstype", ["nfs", "nfs4", "autofs"])
def test_detection_names_the_network_filesystem(fstype: str, home: Path, findmnt) -> None:
    findmnt(fstype)

    assert detect_network_home(home) == fstype


@pytest.mark.parametrize("fstype", ["ext4", "xfs", "btrfs", "overlay", ""])
def test_detection_passes_over_local_storage(fstype: str, home: Path, findmnt) -> None:
    findmnt(fstype)

    assert detect_network_home(home) is None


def test_a_findmnt_that_hangs_is_not_waited_on(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck mount is exactly where findmnt blocks, so the call is bounded."""
    monkeypatch.setattr(
        "osprey.cli.deploy_scaffold.shutil.which",
        lambda name: "/usr/bin/findmnt" if name == "findmnt" else None,
    )

    def timing_out(argv, **kwargs):
        assert kwargs.get("timeout"), "the call has to carry a timeout"
        raise subprocess.TimeoutExpired(list(argv), kwargs["timeout"])

    monkeypatch.setattr("osprey.cli.deploy_scaffold.subprocess.run", timing_out)

    assert detect_network_home(home) is None


def test_an_unrunnable_findmnt_is_not_an_error(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osprey.cli.deploy_scaffold.shutil.which",
        lambda name: "/usr/bin/findmnt" if name == "findmnt" else None,
    )

    def unrunnable(argv, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("osprey.cli.deploy_scaffold.subprocess.run", unrunnable)

    assert detect_network_home(home) is None
