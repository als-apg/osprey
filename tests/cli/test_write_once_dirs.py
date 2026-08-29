"""Contract tests for the WRITE-ONCE directories a materialization seeds.

A seed directory is copied out of the app bundle on a repo's first init and
never again: it becomes the operator's, so a re-materialization must find it
already there and leave it alone. That is why seeds are deliberately absent
from ``MATERIALIZED_SOURCE_ENTRIES`` — the entries a later ``--force`` is
allowed to replace — and why the one case that DOES remove a seed is a run that
created it and then failed, which owns the tree it just wrote.

The bundle these tests seed from is synthetic: a template root assembled in
``tmp_path`` that symlinks the real one everywhere except the bundle under
test, whose ``mcp_servers/`` is planted here. That keeps the assertions about
copy SEMANTICS (byte-code dropped, existing tree untouched, failure cleaned up)
independent of whatever the packaged bundle happens to ship on any given day.
The verb-level section at the end runs ``osprey init`` against the PACKAGED
bundle instead, because what a first init seeds into an operator's repo is the
shipped package and nothing else; those tests skip when the bundle ships no
such tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import osprey.cli.build_profile as build_profile_mod
from osprey.cli.build_profile_schema import McpServerDef
from osprey.cli.init_cmd import _missing_server_packages, init
from osprey.cli.profile_cmd import _materialize_profile_directory
from osprey.cli.profile_conventions import BUILD_OUTPUT_DIR
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError

#: The preset materialized by most tests here, and the bundle it names.
PRESET = "hello-world"
BUNDLE = "hello_world"

#: What a caller passes: profile-root directory name → bundle-relative source.
SEED_DIRS = {"mcp_servers": "mcp_servers"}


def _packaged_template_root() -> Path:
    """The installed template root, read before any monkeypatching."""
    return Path(TemplateManager().template_root)


#: The seed the shipped bundle carries. The tests below read the real tree
#: rather than a planted one, because what a first ``osprey init`` puts in an
#: operator's repo IS this package — a synthetic stand-in could not catch a
#: bundle that stopped shipping it.
_PACKAGED_SEED = _packaged_template_root() / "apps" / BUNDLE / "mcp_servers"

needs_packaged_seed = pytest.mark.skipif(
    not _PACKAGED_SEED.is_dir(),
    reason=f"the packaged {BUNDLE} bundle ships no mcp_servers/ tree",
)


@pytest.fixture
def synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every ``TemplateManager`` at a root whose bundle ships a known seed.

    Everything but the bundle under test is symlinked, so the root costs
    nothing to build and the rest of the materialization (the ``claude_code``
    context baseline, the deploy templates) behaves exactly as it does against
    the real installation.

    Returns:
        The bundle's planted ``mcp_servers/`` source tree.
    """
    real = _packaged_template_root()
    root = tmp_path / "template-root"
    (root / "apps").mkdir(parents=True)
    for entry in real.iterdir():
        if entry.name != "apps":
            (root / entry.name).symlink_to(entry)
    for entry in (real / "apps").iterdir():
        if entry.name != BUNDLE:
            (root / "apps" / entry.name).symlink_to(entry)

    # Copied without the packaged seed: what this bundle ships as `mcp_servers/`
    # is planted below, so these tests read the same either side of the day the
    # real seed lands.
    shutil.copytree(
        real / "apps" / BUNDLE,
        root / "apps" / BUNDLE,
        ignore=shutil.ignore_patterns("mcp_servers", "__pycache__"),
    )
    seed = root / "apps" / BUNDLE / "mcp_servers"
    (seed / "fake_server").mkdir(parents=True)
    (seed / "fake_server" / "__init__.py").write_text('"""Planted."""\n', encoding="utf-8")
    # Byte-code a source checkout accumulates and a wheel never ships. It must
    # not reach the operator's repo, or the same seed would differ by where
    # osprey was installed from.
    (seed / "fake_server" / "__pycache__").mkdir()
    (seed / "fake_server" / "__pycache__" / "x.pyc").write_bytes(b"\x00stale byte-code")

    monkeypatch.setattr(TemplateManager, "_get_template_root", lambda self: root)
    return seed


def _fail_on_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make only the CLOSING resolve — the read-back of the written file — fail.

    The opening resolve takes ``None`` and a preset name; the round-trip takes
    the path of the ``profile.yml`` just written. Failing on the second is what
    puts the failure AFTER the seed, which is the case this file is about.
    """
    real = build_profile_mod.resolve_build_profile

    def resolve(profile_path, preset_name, *args, **kwargs):
        if profile_path is not None:
            raise BuildProfileError("planted round-trip failure")
        return real(profile_path, preset_name, *args, **kwargs)

    monkeypatch.setattr(build_profile_mod, "resolve_build_profile", resolve)


def test_failed_first_run_leaves_no_seed(
    synthetic_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that seeds and then fails removes the tree it just created.

    The seed is not in ``MATERIALIZED_SOURCE_ENTRIES``, so the only thing that
    can remove it is the run's own record of having made it — which is exactly
    the guarantee a first init needs: it either produces a repo or nothing.
    """
    target = tmp_path / "repo"
    real_resolve = build_profile_mod.resolve_build_profile
    _fail_on_round_trip(monkeypatch)

    with pytest.raises(click.UsageError) as refusal:
        _materialize_profile_directory(target, PRESET, seed_dirs=SEED_DIRS)

    assert not (target / "mcp_servers").exists()
    assert not (target / "profile.yml").exists()
    assert not (target / "data").exists()
    # `_cleanup`'s own verdict: nothing it owns survived, seed included.
    assert "Nothing was materialized." in str(refusal.value)

    # And the retry — the operator fixing whatever broke and running again —
    # seeds, because the failed run left the name free.
    monkeypatch.setattr(build_profile_mod, "resolve_build_profile", real_resolve)
    materialized = _materialize_profile_directory(target, PRESET, seed_dirs=SEED_DIRS)

    assert materialized.seeded == ("mcp_servers",)
    assert (target / "mcp_servers" / "fake_server" / "__init__.py").is_file()


def test_seed_carries_no_byte_code(synthetic_bundle: Path, tmp_path: Path) -> None:
    """A seed from a source checkout matches a seed from a wheel."""
    target = tmp_path / "repo"

    materialized = _materialize_profile_directory(target, PRESET, seed_dirs=SEED_DIRS)

    seeded = target / "mcp_servers"
    assert materialized.seeded == ("mcp_servers",)
    assert (seeded / "fake_server" / "__init__.py").read_text(
        encoding="utf-8"
    ) == '"""Planted."""\n'
    assert list(seeded.rglob("__pycache__")) == []
    assert list(seeded.rglob("*.pyc")) == []


def test_bundle_without_the_tree_seeds_nothing(tmp_path: Path) -> None:
    """A bundle that ships no such directory is not an error, and writes nothing.

    The control assistant ships no ``mcp_servers/``, so asking for one is a
    no-op rather than a refusal: the same caller passes the same seed table for
    every preset, and only the bundles that ship the tree get it.
    """
    target = tmp_path / "repo"

    materialized = _materialize_profile_directory(target, "control-assistant", seed_dirs=SEED_DIRS)

    assert materialized.seeded == ()
    assert not (target / "mcp_servers").exists()


def test_existing_directory_is_left_alone(synthetic_bundle: Path, tmp_path: Path) -> None:
    """Write-once: a name already in the target is the operator's, not the bundle's.

    It is also not reported as seeded — the caller uses that list to decide what
    a failed run may delete, and deleting a directory this run did not create is
    the one thing the write-once rule exists to prevent.
    """
    target = tmp_path / "repo"
    mine = target / "mcp_servers" / "my_server"
    mine.mkdir(parents=True)
    (mine / "server.py").write_text("# the operator's own server\n", encoding="utf-8")

    materialized = _materialize_profile_directory(target, PRESET, seed_dirs=SEED_DIRS)

    assert materialized.seeded == ()
    assert (mine / "server.py").read_text(encoding="utf-8") == "# the operator's own server\n"
    assert not (target / "mcp_servers" / "fake_server").exists()


def test_no_seed_dirs_asked_for_seeds_nothing(synthetic_bundle: Path, tmp_path: Path) -> None:
    """Today's callers pass nothing, and see no behavior change from the seam."""
    target = tmp_path / "repo"

    materialized = _materialize_profile_directory(target, PRESET)

    assert materialized.seeded == ()
    assert not (target / "mcp_servers").exists()


@needs_packaged_seed
def test_packaged_bundle_seeds_its_servers(tmp_path: Path) -> None:
    """The real bundle, seeded through the real template root."""
    target = tmp_path / "repo"

    materialized = _materialize_profile_directory(target, PRESET, seed_dirs=SEED_DIRS)

    assert materialized.seeded == ("mcp_servers",)
    assert (target / "mcp_servers").is_dir()
    assert list((target / "mcp_servers").rglob("__pycache__")) == []
    assert list((target / "mcp_servers").rglob("*.pyc")) == []


# ---------------------------------------------------------------------------
# Through the verb, on the packaged bundle
# ---------------------------------------------------------------------------

#: Content an operator adds to the seeded tree, looked for after the verb runs
#: again. Distinctive, so a directory re-seeded with the bundle's own files
#: reads as destroyed rather than as preserved.
SENTINEL = "written-by-the-operator\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _init(runner: CliRunner, target: Path, preset: str = PRESET, *extra: str):
    """Run the verb the way an operator does, minus the git repository.

    ``--no-git`` throughout: every assertion here is about which files are on
    disk afterwards, and a repository around them would only slow the run down.
    """
    return runner.invoke(init, [str(target), "--preset", preset, "--no-git", *extra])


def _zone_row(repo: Path) -> str:
    """The README sentence that spells the source zone out in full.

    Read as a line rather than searched for in the whole file, so that prose
    elsewhere in the README mentioning a directory by name cannot stand in for
    the zone table having claimed it.
    """
    readme = (repo / "README.md").read_text(encoding="utf-8")
    return next(line for line in readme.splitlines() if line.startswith("In full, the first row"))


def _reset_runs_without_a_container_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``--reset`` past the machinery that wants a live container runtime.

    ``init`` refuses ``--reset`` up front unless a runtime answers, then hands
    the previous deployment's containers and volumes to ``reset_for_reinit``.
    Both are stubbed exactly as ``test_init_verb.py`` stubs them: what is under
    test here is which files the re-materialization leaves standing, and that
    must not depend on whether the developer has Docker Desktop open.
    """
    from osprey.deployment.reset import ResetOutcome

    monkeypatch.setattr(
        "osprey.deployment.runtime_helper.verify_runtime_is_running",
        lambda config=None: (True, ""),
    )
    monkeypatch.setattr(
        "osprey.deployment.reset.reset_for_reinit",
        lambda repo_root, **kw: ResetOutcome.COMPLETED,
    )
    monkeypatch.setattr("osprey.cli.init_cmd._surviving_project_resources", lambda target: [])


@needs_packaged_seed
def test_first_init_seeds_the_shipped_server_package(runner: CliRunner, tmp_path: Path) -> None:
    """A brand-new repo comes with the bundle's server package already in it.

    Byte-compared against what is packaged rather than checked for existence:
    the point of shipping a worked example is that the operator reads the code
    the framework's own tests exercise, not an approximation of it.
    """
    target = tmp_path / "demo"

    result = _init(runner, target)

    assert result.exit_code == 0, result.output
    seeded = target / "mcp_servers" / "example_server"
    for name in ("__init__.py", "server.py", "__main__.py"):
        assert (
            seeded.joinpath(name).read_bytes()
            == (_PACKAGED_SEED / "example_server" / name).read_bytes()
        ), name
    # A source checkout accumulates byte-code beside the package; a wheel never
    # ships it. Neither may reach the operator's repo.
    assert list((target / "mcp_servers").rglob("__pycache__")) == []
    assert list((target / "mcp_servers").rglob("*.py[co]")) == []


@needs_packaged_seed
def test_the_zone_row_names_the_directory_only_where_it_was_seeded(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The README describes the repo the operator got, not the seeding table.

    A preset whose bundle ships no server package must not hand its operator a
    README naming a directory they will not find.
    """
    seeding = tmp_path / "seeded"
    bare = tmp_path / "bare"
    assert _init(runner, seeding).exit_code == 0
    assert _init(runner, bare, "control-assistant").exit_code == 0

    assert "`mcp_servers/`" in _zone_row(seeding)
    assert "mcp_servers" not in _zone_row(bare)


@needs_packaged_seed
def test_a_file_added_to_the_seed_survives_force_and_reset(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write-once through the verb: re-materializing never costs the operator a server.

    The sentinel goes one level below ``mcp_servers/`` because that directory is
    a profile convention directory — one directory per server — so a bare file
    directly inside it would make the repo fail validation rather than model
    anything an operator would have written.
    """
    target = tmp_path / "demo"
    assert _init(runner, target).exit_code == 0
    mine = target / "mcp_servers" / "example_server" / "sentinel.txt"
    mine.write_text(SENTINEL, encoding="utf-8")

    forced = _init(runner, target, PRESET, "--force")

    assert forced.exit_code == 0, forced.output
    assert mine.read_text(encoding="utf-8") == SENTINEL

    _reset_runs_without_a_container_runtime(monkeypatch)
    reset = _init(runner, target, PRESET, "--reset")

    assert reset.exit_code == 0, reset.output
    assert mine.read_text(encoding="utf-8") == SENTINEL


@needs_packaged_seed
def test_a_deleted_seed_is_never_put_back(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the directory is a decision, and re-running the verb respects it.

    This is why the "has this name ever been initialized?" question is asked in
    ``init`` and not in the materializer: inside it, the hold-aside has already
    moved ``profile.yml`` away, so every ``--force`` run looks like a first init
    and would put the deleted tree straight back.
    """
    target = tmp_path / "demo"
    assert _init(runner, target).exit_code == 0
    shutil.rmtree(target / "mcp_servers")

    forced = _init(runner, target, PRESET, "--force")

    assert forced.exit_code == 0, forced.output
    assert not (target / "mcp_servers").exists()

    _reset_runs_without_a_container_runtime(monkeypatch)
    reset = _init(runner, target, PRESET, "--reset")

    assert reset.exit_code == 0, reset.output
    assert not (target / "mcp_servers").exists()


# ── The pairing warning ──────────────────────────────────────────────────────
# A server entry and the package it starts are two halves of one thing, and
# only one half is write-once. `--force` on a repo whose `mcp_servers/` was
# deleted writes profile.yml again, live block and all, and never puts the
# package back: the pair is broken, and nothing said so until this warning.


#: What the warning says once the entry name and module are filled in. Spelled
#: out here rather than imported from the code under test, so a reworded
#: warning has to be read by someone before it can pass.
WARNING_TAIL = "is missing; delete the block too, or copy the package back."
WARNING_SUMMARY = (
    f"profile.yml declares example_server but mcp_servers/example_server/ {WARNING_TAIL}"
)
WARNING_DETAIL = "Every session otherwise waits 20 s for a server that cannot start."


@needs_packaged_seed
def test_a_paired_first_init_prints_no_warning(runner: CliRunner, tmp_path: Path) -> None:
    """The repo the verb just created is paired, so it has nothing to say.

    This is the common case by a wide margin, and a warning here would train
    everyone to ignore the one case that matters.
    """
    result = _init(runner, tmp_path / "demo")

    assert result.exit_code == 0, result.output
    assert WARNING_TAIL not in result.output


def test_a_preset_declaring_no_server_prints_no_warning(runner: CliRunner, tmp_path: Path) -> None:
    """The control assistant declares no Python server, so there is no pair to break.

    The check reads the profile rather than the directory: a preset with no
    ``mcp_servers`` block must not be told that a package is missing just
    because it has no ``mcp_servers/`` either.
    """
    result = _init(runner, tmp_path / "bare", "control-assistant")

    assert result.exit_code == 0, result.output
    assert WARNING_TAIL not in result.output


@needs_packaged_seed
def test_a_deleted_package_draws_a_warning_on_the_next_forced_run(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Deleting the directory alone leaves a declaration that cannot start.

    Removing the package is a decision the verb respects (see
    ``test_a_deleted_seed_is_never_put_back``), and the profile written beside
    it still names the server. Saying so is the whole point: the operator meant
    to drop the example and has done half of it.
    """
    target = tmp_path / "demo"
    assert _init(runner, target).exit_code == 0
    shutil.rmtree(target / "mcp_servers")

    forced = _init(runner, target, PRESET, "--force")

    assert forced.exit_code == 0, forced.output
    assert WARNING_SUMMARY in forced.output
    assert WARNING_DETAIL in forced.output


def test_the_warning_reads_only_servers_this_repo_could_pair(tmp_path: Path) -> None:
    """Only a package the build copies out of this repo can be missing from it.

    A remote server has no package here at all, and one started from a binary
    on the host has one this command cannot see. Both are fine as they stand,
    and a warning about either would be wrong rather than merely noisy.
    """
    target = tmp_path / "repo"
    (target / "mcp_servers" / "present_server").mkdir(parents=True)
    build_zone = f"{{project_root}}/{BUILD_OUTPUT_DIR}/_mcp_servers"
    servers = {
        "gone": McpServerDef(
            command="{current_python_env}",
            args=["-m", "gone_server"],
            env={"PYTHONPATH": build_zone},
        ),
        "present": McpServerDef(
            command="{current_python_env}",
            args=["-m", "present_server"],
            env={"PYTHONPATH": build_zone},
        ),
        "remote": McpServerDef(url="http://lattice.example.org:8400/mcp", port=8400),
        "binary": McpServerDef(
            command="/usr/local/bin/facility-mcp",
            args=["--stdio"],
            env={"PYTHONPATH": build_zone},
        ),
        "elsewhere": McpServerDef(
            command="{current_python_env}",
            args=["-m", "vendored_server"],
            env={"PYTHONPATH": "/opt/vendor/lib"},
        ),
    }

    assert _missing_server_packages(servers, target) == [("gone", "gone_server")]
