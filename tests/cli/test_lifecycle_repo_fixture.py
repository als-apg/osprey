"""The exemplar deployment repo fixture holds its shape.

The fixture in :mod:`tests.fixtures.lifecycle_repo` is the gold-standard
three-zone repo every lifecycle verb is developed against, and the target
``osprey init`` must reproduce. These tests pin the invariants that make it
usable in both roles: the zone layout, an anchored ``.gitignore``, a profile
the real loader accepts, and byte-stable content.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from osprey import __version__
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_deploy import deploy_aware_config_errors
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.repo_resolver import find_repo_root
from tests.fixtures.lifecycle_repo import (
    EXEMPLAR_DIRNAME,
    EXEMPLAR_PRESET,
    IGNORED_ZONE_PATTERNS,
    PERSONA_PRESETS,
    STATE_DIRS,
    build_exemplar_repo,
    exemplar_source_files,
)

pytestmark = pytest.mark.unit

#: Command strings the redesign retires. None may survive in an artifact the
#: exemplar ships, because ``osprey init`` emits these same files (SC-8).
#:
#: These are DATA — the retired spellings themselves — so they are the one place
#: in this file that must keep naming a dead verb. A sweep that rewrites verb
#: names across the tree has to skip this tuple, or it silently converts the
#: criterion into "the exemplar may not mention the LIVE verbs", which the
#: exemplar does on purpose.
RETIRED_STRINGS = (
    "osprey deploy",
    "osprey claude",
    "deploy scaffold",
    "osprey profile new",
    "profile try",
    "claude regen",
    "set-control-system",
    "set-epics-gateway",
    "OSPREY_PROJECT",
    "--project",
    "_agent_data",
)


# ── Zone layout ──────────────────────────────────────────────────────────────


def test_repo_root_holds_the_profile(lifecycle_repo: Path) -> None:
    """profile.yml sits at the repo root — there is no profile/ nesting."""
    assert lifecycle_repo.name == EXEMPLAR_DIRNAME
    assert (lifecycle_repo / "profile.yml").is_file()
    assert not (lifecycle_repo / "profile").exists()


@pytest.mark.parametrize(
    "rel",
    [
        "profile.yml",
        "triggers.yml",
        "README.md",
        ".gitignore",
        ".env.example",
        "ci-extra.yml",
        "personas/readonly.yml",
        "personas/readwrite.yml",
        "data/README.md",
        "data/channel_limits.json",
        "data/machine_state_channels.json",
        "data/channel_databases/tiers/tier3/hierarchical.json",
        "data/facility_knowledge/index.md",
        "data/simulation/scenarios/nominal/scenario.json",
        "web-terminal-context/alice/.gitkeep",
        "web-terminal-context/bob/.gitkeep",
    ],
)
def test_source_zone_file_present(lifecycle_repo: Path, rel: str) -> None:
    assert (lifecycle_repo / rel).is_file(), f"missing source-zone file: {rel}"


def test_state_zone_is_present_and_empty(lifecycle_repo: Path) -> None:
    """var/ exists so a build finds it, and holds nothing — not even a marker.

    A ``.gitkeep`` here would be git-ignored anyway and would be the one thing
    ``osprey reset``'s wipe of ``var/agent_data`` has to work around.
    """
    for rel in STATE_DIRS:
        state_dir = lifecycle_repo / rel
        assert state_dir.is_dir(), f"missing state dir: {rel}"
        assert list(state_dir.iterdir()) == []


def test_output_zone_is_absent(lifecycle_repo: Path) -> None:
    """No build/ — the exemplar is the repo before any build, and after a clone."""
    assert not (lifecycle_repo / "build").exists()


def test_secrets_zone_is_opt_in(lifecycle_repo_factory) -> None:
    """A freshly emitted repo has no .env; seeding one is a separate act."""
    bare = lifecycle_repo_factory()
    assert not (bare / ".env").exists()
    assert (bare / ".env.example").is_file()

    seeded = lifecycle_repo_factory(bare.parent / "seeded", seed_env=True)
    assert (seeded / ".env").is_file()
    assert "ANTHROPIC_API_KEY=" in (seeded / ".env").read_text()
    assert (seeded / ".env").stat().st_mode & 0o777 == 0o600


def test_verify_script_is_executable(lifecycle_repo_factory, tmp_path: Path) -> None:
    repo = lifecycle_repo_factory(tmp_path / "with-ci", with_ci=True)

    verify = repo / "scripts" / "verify.sh"
    assert verify.stat().st_mode & 0o111, "scripts/verify.sh must be executable"
    assert verify.read_text().startswith("#!/usr/bin/env bash")


def test_deploy_job_never_copies_over_the_durable_zones(
    lifecycle_repo_factory, tmp_path: Path
) -> None:
    """The host re-renders from the commit; it is not rsynced over.

    ``git reset --hard`` moves only tracked files, so the deploy host's ``.env``
    and ``var/`` survive by construction. A working-tree copy would have to be
    trusted not to delete them, and ``rsync --delete`` would not.
    """
    repo = lifecycle_repo_factory(tmp_path / "with-ci", with_ci=True)
    pipeline = (repo / ".gitlab-ci.yml").read_text()
    # Comments explain why rsync is absent, so judge the pipeline's own lines.
    directives = [ln for ln in pipeline.splitlines() if not ln.lstrip().startswith("#")]

    assert not [ln for ln in directives if "rsync" in ln]
    assert "git reset --hard $CI_COMMIT_SHA" in pipeline
    # A control-room service is not deployed by a push landing on main.
    assert "when: manual" in pipeline
    assert "resource_group: production" in pipeline
    assert 'ssh-keyscan -H "$DEPLOY_HOST" >> ~/.ssh/known_hosts' in pipeline


def test_env_production_never_renders_to_the_job_log(
    lifecycle_repo_factory, tmp_path: Path
) -> None:
    """``users env-production`` must carry ``--output``, or secrets hit stdout.

    ``--output`` is what selects file mode: without it the command echoes the
    fully assembled secrets file, and in a CI job stdout is the retained job
    log. The flag also creates the file at 0600 from its first byte.
    """
    repo = lifecycle_repo_factory(tmp_path / "with-ci", with_ci=True)
    directives = [
        ln
        for ln in (repo / ".gitlab-ci.yml").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    ]

    renders = [ln for ln in directives if "users env-production" in ln]
    assert renders, "the deploy job must render .env.production before starting containers"
    for line in renders:
        assert "--output" in line, (
            f"env-production without --output writes secrets to stdout: {line}"
        )


# ── .gitignore ───────────────────────────────────────────────────────────────


def test_gitignore_anchors_every_zone_entry(lifecycle_repo: Path) -> None:
    """Zone entries carry a leading slash; an unanchored spelling is the bug.

    An unanchored ``build/`` or ``.env*`` also matches a same-named path
    anywhere deeper in the tree — including a file moved there later — and does
    it silently.
    """
    lines = [
        line.strip()
        for line in (lifecycle_repo / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    for pattern in IGNORED_ZONE_PATTERNS:
        assert pattern in lines, f".gitignore is missing {pattern}"
    assert "!/.env.example" in lines

    unanchored = {"build/", "build", "var/", "var", ".env", ".env*"}
    assert not (unanchored & set(lines)), f"unanchored zone entry in .gitignore: {lines}"


def test_git_status_is_clean_from_birth(lifecycle_repo_factory, tmp_path: Path) -> None:
    """SC-1: an emitted repo's committed source zone leaves nothing untracked."""
    if shutil.which("git") is None:
        pytest.fail("git is required to check the shipped .gitignore against real git")

    repo = lifecycle_repo_factory(tmp_path / "committed", seed_env=True, git=True)
    (repo / "var" / "agent_data" / "sessions.json").write_text("{}")
    (repo / "build").mkdir()
    (repo / "build" / "config.yml").write_text("rendered: true\n")

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == "", f"unexpected git status:\n{status.stdout}"


# ── The profile the repo carries ─────────────────────────────────────────────


def test_profile_resolves_and_validates(lifecycle_repo: Path) -> None:
    """The real loader accepts the profile, data tree and web-stack lint included."""
    profile, profile_dir = resolve_build_profile(lifecycle_repo / "profile.yml", None)

    profile.validate(profile_dir)
    # The lint `osprey validate` and `osprey build` both gate on — judged on the
    # merged view the build renders, not on the raw config block.
    assert deploy_aware_config_errors(profile.deploy, profile.config) == []

    assert profile.name == "Als Exemplar"
    assert profile_dir == lifecycle_repo
    assert profile.config["control_system.type"] == "virtual_accelerator"


def test_ci_variant_resolves_and_validates(lifecycle_repo_factory, tmp_path: Path) -> None:
    """Filling in the deploy coordinates keeps the profile valid and lint-clean."""
    repo = lifecycle_repo_factory(tmp_path / "with-ci", with_ci=True)

    profile, profile_dir = resolve_build_profile(repo / "profile.yml", None)

    profile.validate(profile_dir)
    assert deploy_aware_config_errors(profile.deploy, profile.config) == []

    assert profile.deploy is not None
    assert profile.deploy.host.fqdn == "appsdev2.example.org"
    # image_source has one home, and with a deploy block that home is the block.
    assert profile.deploy.image_source == "local"
    assert "image_source" not in profile.config["modules.web_terminals"]


@pytest.mark.parametrize("persona", sorted(PERSONA_PRESETS))
def test_persona_delta_resolves_over_the_repo_profile(lifecycle_repo: Path, persona: str) -> None:
    """A persona is a delta whose root is the repo, not personas/."""
    profile, profile_dir = resolve_build_profile(
        lifecycle_repo / "personas" / f"{persona}.yml", None
    )

    profile.validate(profile_dir)
    assert deploy_aware_config_errors(profile.deploy, profile.config) == []

    assert profile_dir == lifecycle_repo
    assert profile.name == f"Als Exemplar ({persona})"
    assert profile.deploy_services is False
    assert profile.config["control_system.writes_enabled"] is (persona == "readwrite")


def test_persona_renders_land_under_the_output_zone(lifecycle_repo: Path) -> None:
    """Persona renders are build output; their paths must say so."""
    profile, _ = resolve_build_profile(lifecycle_repo / "profile.yml", None)
    personas = profile.config["modules.web_terminals"]["personas"]

    for name, entry in personas.items():
        assert entry["project_path"].startswith("build/"), name
        # Both sides derive the name from the repo, which is why a render lands
        # exactly where the roster mounts it.
        assert entry["project"] == Path(entry["project_path"]).name
        assert entry["build_profile"] == f"personas/{name}.yml"


def test_provenance_records_the_live_preset(lifecycle_repo: Path) -> None:
    """The two stamped values are resolved at materialization, never frozen."""
    text = (lifecycle_repo / "profile.yml").read_text()

    assert f"preset: {EXEMPLAR_PRESET}" in text
    assert f"preset_hash: {compute_preset_hash(EXEMPLAR_PRESET)}" in text
    assert f"emitted by OSPREY {__version__}" in text


def test_repo_is_discoverable_from_any_subdirectory(lifecycle_repo: Path) -> None:
    """The single discovery rule finds this repo from inside it (FR-9)."""
    nested = lifecycle_repo / "data" / "facility_knowledge" / "subsystems"

    assert find_repo_root(nested) == lifecycle_repo.resolve()
    assert find_repo_root(lifecycle_repo) == lifecycle_repo.resolve()


# ── Emitted content ──────────────────────────────────────────────────────────


def _all_text_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git/" not in str(path.relative_to(root))
    }


def test_no_retired_command_strings(lifecycle_repo: Path) -> None:
    """SC-8: nothing the exemplar ships names a verb the redesign deletes."""
    offenders = [
        f"{rel}: {retired}"
        for rel, text in _all_text_files(lifecycle_repo).items()
        for retired in RETIRED_STRINGS
        if retired in text
    ]
    assert offenders == [], "retired command strings in emitted artifacts:\n" + "\n".join(offenders)


def test_no_unexpanded_sentinels(lifecycle_repo: Path) -> None:
    """Every ``@…@`` placeholder is resolved before it reaches disk."""
    offenders = [
        rel
        for rel, text in _all_text_files(lifecycle_repo).items()
        if "@OSPREY_VERSION@" in text or "@PRESET_HASH:" in text
    ]
    assert offenders == []


@pytest.mark.parametrize("with_ci", [False, True])
def test_disk_matches_the_declared_manifest(
    lifecycle_repo_factory, tmp_path: Path, with_ci: bool
) -> None:
    """``exemplar_source_files()`` IS what lands — the byte-comparison contract.

    Which variant a downstream comparison wants is fixed, not a preference:
    Task 2.1 compares ``osprey init``'s emission against ``with_ci=False`` (a
    bare init has no deploy coordinates, so it renders no pipeline), and Tasks
    2.5/2.8 compare ``osprey scaffold ci`` against ``with_ci=True``. Either
    way the mapping has to be the same bytes the factory writes, not a
    paraphrase of them.
    """
    repo = lifecycle_repo_factory(tmp_path / f"ci-{with_ci}", with_ci=with_ci)
    declared = exemplar_source_files(with_ci=with_ci)
    on_disk = _all_text_files(repo)

    assert set(declared) == set(on_disk)
    for rel, text in declared.items():
        assert on_disk[rel] == text, f"{rel} differs from the declared manifest"


def test_materialization_is_deterministic(lifecycle_repo_factory, tmp_path: Path) -> None:
    first = lifecycle_repo_factory(tmp_path / "one")
    second = lifecycle_repo_factory(tmp_path / "two")

    assert _all_text_files(first) == _all_text_files(second)


def test_ci_pipeline_follows_the_deploy_block(lifecycle_repo: Path) -> None:
    """No deploy coordinates, no pipeline — there is nothing to render one from.

    This is the default shape precisely because it is what a bare
    ``osprey init --preset control-assistant`` can reproduce.
    """
    assert not (lifecycle_repo / ".gitlab-ci.yml").exists()
    assert not (lifecycle_repo / "scripts").exists()
    # ci-extra.yml is the facility's own file: created once, never regenerated,
    # and never conditional on the pipeline existing.
    assert (lifecycle_repo / "ci-extra.yml").is_file()

    profile, _ = resolve_build_profile(lifecycle_repo / "profile.yml", None)
    assert profile.deploy is None


def test_materializing_into_an_existing_directory_is_allowed(tmp_path: Path) -> None:
    """`init` runs inside a cloned empty repo, so the factory must too."""
    dest = tmp_path / "cloned"
    dest.mkdir()
    (dest / ".git").mkdir()

    repo = build_exemplar_repo(dest)

    assert repo == dest.resolve()
    assert (repo / "profile.yml").is_file()
