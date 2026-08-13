"""Tests for ``osprey scaffold`` CLI subcommands.

Claiming moves an artifact into the profile the project was built from; the
build's convention scan is what registers ownership afterwards. The tests below
therefore assert two things above all: that the artifact ends up in the right
profile slot, and that a claim writes no ``config.yml`` and no manifest — the
second source of truth this design removes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.scaffold_cmd import (
    ScaffoldClaimError,
    _owned_row,
    claim_into_profile,
    profile_root_or_none,
    profile_slot_for,
    resolve_claim_target,
    scaffold,
)
from osprey.cli.templates.manifest import MANIFEST_FILENAME
from osprey.services.build_artifacts.ownership import update_config_add_user_owned


@pytest.fixture()
def repo_dir(tmp_path):
    """A deployment repo whose build zone holds a real render.

    Materialized and built through the real verbs rather than assembled: what
    these tests need from the render is GENUINE artifacts — the claim/diff/
    unclaim surface reads a manifest, a config and the files they name, and a
    hand-written stand-in would prove only that the fixture agrees with itself.
    """
    runner = CliRunner()
    repo = tmp_path / "demo-repo"

    result = runner.invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture()
def project_dir(repo_dir):
    """The build zone of :func:`repo_dir`, where the render landed."""
    return repo_dir / "build"


class Pair:
    """A deployment repo: its source zone, and the build zone beside it.

    Stands in for a real ``osprey build`` wherever a test is about the claim
    logic rather than the build: the only thing claiming reads from a build
    zone is its manifest, and the only thing it reads from the source zone is
    the convention slot it is about to write.

    ``profile`` is the repo root — in a three-zone repo the profile lives at
    the root, so the repo the verbs resolve and the profile a claim writes into
    are the same directory.
    """

    def __init__(self, project: Path, profile: Path):
        self.project = project
        self.profile = profile

    @property
    def repo(self) -> Path:
        """What ``--repo`` names: the repo root, which is the profile root."""
        return self.profile

    def artifact(self, rel: str, content: str = "# artifact\n") -> Path:
        """Write a project file at ``rel``, creating its parents."""
        path = self.project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def config(self) -> dict:
        return yaml.safe_load((self.project / "config.yml").read_text(encoding="utf-8")) or {}


def _write_manifest(project: Path, build_args: dict) -> None:
    (project / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "build_args": build_args,
                "reproducible_command": "osprey build demo --preset hello-world",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def pair(tmp_path) -> Pair:
    """A deployment repo whose build zone names the profile at its root."""
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")

    project = repo / "build"
    project.mkdir()
    (project / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    _write_manifest(project, {"profile_path_abs": str(repo / "profile.yml")})
    return Pair(project, repo)


# ── Name resolution ──────────────────────────────────────────────────


class TestResolveClaimTarget:
    """The name → (project path, profile slot) mapping."""

    def test_markdown_artifact(self):
        target = resolve_claim_target("rules/safety")
        assert target.project_rel == ".claude/rules/safety.md"
        assert target.profile_rel == "rules/safety.md"
        assert not target.is_directory

    def test_nested_markdown_keeps_full_path(self):
        """A nested command is `commands/osprey/scan`, never the basename."""
        target = resolve_claim_target("commands/osprey/scan")
        assert target.project_rel == ".claude/commands/osprey/scan.md"
        assert target.profile_rel == "commands/osprey/scan.md"

    def test_skill_is_a_whole_directory(self):
        target = resolve_claim_target("skills/orbit-check")
        assert target.project_rel == ".claude/skills/orbit-check"
        assert target.profile_rel == "skills/orbit-check"
        assert target.is_directory

    def test_service_is_a_whole_directory(self):
        target = resolve_claim_target("services/postgresql")
        assert target.project_rel == "services/postgresql"
        assert target.profile_rel == "services/postgresql"
        assert target.is_directory

    def test_trailing_md_is_accepted(self):
        """Canonical names carry no suffix, but the spelling with one resolves."""
        assert resolve_claim_target("rules/safety.md") == resolve_claim_target("rules/safety")

    def test_nested_name_under_a_directory_shape_errors(self):
        with pytest.raises(ScaffoldClaimError) as exc:
            resolve_claim_target("skills/orbit-check/SKILL")
        assert "skills/orbit-check" in str(exc.value)

    def test_bare_convention_name_errors(self):
        with pytest.raises(ScaffoldClaimError) as exc:
            resolve_claim_target("rules")
        assert "one rule at a time" in str(exc.value)

    @pytest.mark.parametrize(
        "name", ["", "   ", "..", "rules/../../etc/passwd", "rules/../safety", "skills/.."]
    )
    def test_malformed_names_error(self, name):
        with pytest.raises(ScaffoldClaimError):
            resolve_claim_target(name)

    def test_surrounding_slashes_and_space_are_tolerated(self):
        """Leniency, not an escape: the paths are rebuilt from the convention table."""
        assert resolve_claim_target(" /rules/safety/ ") == resolve_claim_target("rules/safety")

    def test_hook_resolves_by_filename_not_by_stem(self):
        """A hook is known by its filename: ``settings.json`` wires it by name."""
        target = resolve_claim_target("hooks/osprey_cf_feedback_capture.py")
        assert target.project_rel == ".claude/hooks/osprey_cf_feedback_capture.py"
        assert target.profile_rel == "hooks/osprey_cf_feedback_capture.py"
        assert not target.is_directory

    def test_a_hooks_catalog_canonical_resolves_to_the_same_artifact(self):
        """`scaffold list` prints the dashed canonical; the file has another name.

        Both spellings must reach the same artifact, and the name that comes
        back is the one ownership uses — otherwise a claim would rename the
        file out from under the ``settings.json`` entry that invokes it.
        """
        by_canonical = resolve_claim_target("hooks/cf-feedback-capture")
        by_filename = resolve_claim_target("hooks/osprey_cf_feedback_capture.py")
        assert by_canonical == by_filename
        assert by_canonical.name == "hooks/osprey_cf_feedback_capture.py"

    @pytest.mark.parametrize("name", ["hooks/hook-config", "hooks/hook_config.json"])
    def test_a_generated_file_under_a_claimable_channel_is_refused(self, name):
        """``hook_config.json`` sits in a claimable directory but is generated.

        It is the write-safety layer's runtime configuration, rendered from the
        resolved config. Claiming it would move a generated file into the
        profile and freeze it against the build that regenerates it — a stale
        ``write_tools`` list would then silently outlive every config change
        meant to update it. Being inside ``hooks/`` must not exempt it from the
        rule that already refuses ``settings.json`` and ``.mcp.json``.

        Both spellings are refused: the reserved path is a property of the
        file, not of how the operator happened to name it.
        """
        with pytest.raises(ScaffoldClaimError) as exc:
            resolve_claim_target(name)
        message = str(exc.value)
        assert ".claude/hooks/hook_config.json" in message
        assert "write" in message  # the channel explains what it configures

    def test_refusing_a_generated_hook_does_not_close_the_channel(self):
        """Only the generated file is reserved — real hooks stay claimable."""
        assert resolve_claim_target("hooks/osprey_limits.py").project_rel == (
            ".claude/hooks/osprey_limits.py"
        )

    def test_a_catalog_name_pointing_inside_a_directory_claims_the_directory(self):
        """The catalog lists a skill by its SKILL.md; the artifact is the directory."""
        target = resolve_claim_target("skills/session-report")
        assert target.project_rel == ".claude/skills/session-report"
        assert target.is_directory

    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("claude-md", "claude_md_template"),
            ("settings-json", "claude_code.permissions"),
            ("mcp-json", "mcp_servers:"),
        ],
    )
    def test_a_config_generated_artifact_names_the_key_that_configures_it(self, name, key):
        """The clean break: these are generated FROM config, so config is where they change.

        Refusing is only half an answer — an operator who cannot claim
        ``settings.json`` needs to be told that ``claude_code.permissions`` is
        what they were reaching for.
        """
        with pytest.raises(ScaffoldClaimError) as exc:
            resolve_claim_target(name)
        assert key in str(exc.value)

    def test_unknown_name_lists_the_claimable_prefixes(self):
        with pytest.raises(ScaffoldClaimError) as exc:
            resolve_claim_target("nonexistent/thing")
        message = str(exc.value)
        assert "Unknown artifact" in message
        assert "rules/" in message and "skills/" in message

    # Names in the ownership vocabulary — what the build registers, what a
    # persona delta excludes by. Resolving one must return it unchanged.
    OWNERSHIP_NAMES = (
        "rules/safety",
        "agents/channel-finder",
        "commands/osprey/scan",
        "output-styles/terse",
        "skills/orbit-check",
        "skills/session-report",
        "services/postgresql",
        "hooks/osprey_limits.py",
    )

    # Catalog canonicals whose spelling differs from the artifact's own name.
    ALIAS_NAMES = ("hooks/cf-feedback-capture",)

    @pytest.mark.parametrize("name", OWNERSHIP_NAMES + ALIAS_NAMES)
    def test_names_round_trip_through_the_builds_ownership_derivation(self, name):
        """The name a claim resolves to is the name the build registers.

        The two mappings are the same contract read in opposite directions: if
        they ever disagree, a claimed artifact would come back from the profile
        registered under a name nothing recognizes — and for a hook, under a
        filename ``settings.json`` does not invoke.
        """
        from osprey.cli.build_persistence import ownership_canonical
        from osprey.cli.profile_conventions import ConventionCopy

        target = resolve_claim_target(name)
        copy = ConventionCopy(
            category=target.profile_rel.split("/", 1)[0],
            source=Path("/unused"),
            destination=target.project_rel,
            is_directory=target.is_directory,
        )
        assert ownership_canonical(copy) == target.name

    @pytest.mark.parametrize("name", OWNERSHIP_NAMES)
    def test_an_ownership_name_resolves_to_itself(self, name):
        """Only the catalog aliases are renamed; the vocabulary is a fixed point."""
        assert resolve_claim_target(name).name == name


class TestPublishedResolution:
    """``profile_slot_for`` / ``profile_root_or_none`` — resolution without a move."""

    def test_slot_is_pure_resolution(self, pair):
        """Answers for an artifact that exists in neither place, and moves nothing."""
        slot = profile_slot_for(pair.project, "rules/safety")

        assert slot == pair.profile / "rules" / "safety.md"
        assert not slot.exists()
        assert not (pair.profile / "rules").exists()

    def test_slot_follows_a_claimed_artifact(self, pair):
        """The gallery's case: after a claim, the profile copy is the live one."""
        pair.artifact(".claude/rules/safety.md", "# custom\n")
        outcome = claim_into_profile(pair.project, "rules/safety")

        assert profile_slot_for(pair.project, "rules/safety") == outcome.profile_path
        assert outcome.profile_path.read_text(encoding="utf-8") == "# custom\n"

    def test_slot_accepts_either_hook_spelling(self, pair):
        assert profile_slot_for(pair.project, "hooks/cf-feedback-capture") == (
            pair.profile / "hooks" / "osprey_cf_feedback_capture.py"
        )

    def test_slot_is_none_without_a_profile(self, pair):
        """A container-mode project resolves no profile; that is a branch, not an error."""
        _write_manifest(pair.project, {"source": "preset"})

        assert profile_slot_for(pair.project, "rules/safety") is None

    def test_slot_still_raises_on_a_bad_name(self, pair):
        """A caller error stays an error — same contract as claim_into_profile."""
        with pytest.raises(ScaffoldClaimError):
            profile_slot_for(pair.project, "nonexistent/thing")

    def test_profile_root_resolves(self, pair):
        assert profile_root_or_none(pair.project) == pair.profile

    def test_profile_root_is_none_when_unresolvable(self, pair):
        _write_manifest(pair.project, {"source": "preset"})
        assert profile_root_or_none(pair.project) is None

    def test_profile_root_is_none_when_the_profile_is_gone(self, pair):
        """The deployed-container case: the recorded path is a build-machine path."""
        (pair.profile / "profile.yml").unlink()
        assert profile_root_or_none(pair.project) is None


# ── Claiming into the profile ────────────────────────────────────────


class TestClaimIntoProfile:
    """``claim_into_profile`` moves the artifact and touches nothing else."""

    def test_file_moves_into_the_profile(self, pair):
        pair.artifact(".claude/rules/safety.md", "# custom safety\n")

        outcome = claim_into_profile(pair.project, "rules/safety")

        assert outcome.profile_path == pair.profile / "rules" / "safety.md"
        assert outcome.profile_path.read_text(encoding="utf-8") == "# custom safety\n"
        assert not (pair.project / ".claude" / "rules" / "safety.md").exists()

    def test_nested_command_creates_its_profile_parents(self, pair):
        pair.artifact(".claude/commands/osprey/scan.md", "# scan\n")

        outcome = claim_into_profile(pair.project, "commands/osprey/scan")

        assert outcome.profile_path == pair.profile / "commands" / "osprey" / "scan.md"
        assert outcome.profile_path.is_file()

    def test_skill_moves_as_a_whole_directory(self, pair):
        pair.artifact(".claude/skills/orbit-check/SKILL.md", "# orbit check\n")
        pair.artifact(".claude/skills/orbit-check/reference.md", "# reference\n")
        pair.artifact(".claude/skills/orbit-check/scripts/run.py", "print('hi')\n")

        outcome = claim_into_profile(pair.project, "skills/orbit-check")

        moved = pair.profile / "skills" / "orbit-check"
        assert outcome.profile_path == moved
        assert outcome.target.is_directory
        assert (moved / "SKILL.md").read_text(encoding="utf-8") == "# orbit check\n"
        assert (moved / "reference.md").is_file()
        assert (moved / "scripts" / "run.py").is_file()
        assert not (pair.project / ".claude" / "skills" / "orbit-check").exists()

    def test_service_moves_as_a_whole_directory(self, pair):
        pair.artifact("services/postgresql/docker-compose.yml.j2", "services: {}\n")

        outcome = claim_into_profile(pair.project, "services/postgresql")

        assert (outcome.profile_path / "docker-compose.yml.j2").is_file()
        assert not (pair.project / "services" / "postgresql").exists()

    def test_claim_writes_no_yaml_and_no_manifest(self, pair):
        """The build's convention scan owns registration — a claim must not."""
        pair.artifact(".claude/rules/safety.md")
        config_before = (pair.project / "config.yml").read_text(encoding="utf-8")
        manifest_before = (pair.project / MANIFEST_FILENAME).read_text(encoding="utf-8")

        claim_into_profile(pair.project, "rules/safety")

        assert (pair.project / "config.yml").read_text(encoding="utf-8") == config_before
        assert (pair.project / MANIFEST_FILENAME).read_text(encoding="utf-8") == manifest_before
        assert "scaffold" not in pair.config()

    def test_persona_delta_claims_into_its_root_profile(self, pair, tmp_path):
        """Convention directories live at the root, so a delta claims there."""
        personas = pair.profile / "personas"
        personas.mkdir()
        (personas / "readonly.yml").write_text("name: readonly\n", encoding="utf-8")
        _write_manifest(pair.project, {"profile_path_abs": str(personas / "readonly.yml")})
        pair.artifact(".claude/rules/safety.md")

        outcome = claim_into_profile(pair.project, "rules/safety")

        assert outcome.is_persona_delta
        assert outcome.profile_root == pair.profile
        assert outcome.profile_path == pair.profile / "rules" / "safety.md"


class TestClaimRefusals:
    """Every refusal happens before anything moves."""

    def test_nonexistent_artifact(self, pair):
        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")
        assert "Nothing to claim" in str(exc.value)
        assert not (pair.profile / "rules").exists()

    def test_nonexistent_but_catalog_known_artifact_says_so(self, pair):
        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "agents/channel-finder")
        assert "did not render it here" in str(exc.value)

    def test_already_in_the_profile(self, pair):
        pair.artifact(".claude/rules/safety.md", "# same\n")
        (pair.profile / "rules").mkdir()
        (pair.profile / "rules" / "safety.md").write_text("# same\n", encoding="utf-8")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "already in the profile" in str(exc.value)
        # Neither copy was disturbed.
        assert (pair.project / ".claude" / "rules" / "safety.md").is_file()
        assert (pair.profile / "rules" / "safety.md").read_text(encoding="utf-8") == "# same\n"

    def test_destination_collision_reports_the_difference(self, pair):
        pair.artifact(".claude/rules/safety.md", "# project version\n")
        (pair.profile / "rules").mkdir()
        (pair.profile / "rules" / "safety.md").write_text("# profile version\n", encoding="utf-8")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "differs" in str(exc.value)
        assert (pair.profile / "rules" / "safety.md").read_text(
            encoding="utf-8"
        ) == "# profile version\n"

    def test_directory_collision_is_refused(self, pair):
        pair.artifact(".claude/skills/orbit-check/SKILL.md", "# project\n")
        existing = pair.profile / "skills" / "orbit-check"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text("# profile\n", encoding="utf-8")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "skills/orbit-check")

        assert "already in the profile" in str(exc.value)
        assert (existing / "SKILL.md").read_text(encoding="utf-8") == "# profile\n"

    def test_shape_mismatch_is_refused(self, pair):
        """A skill is a directory; a file sitting at its path is not one."""
        pair.artifact(".claude/skills/orbit-check", "# not a directory\n")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "skills/orbit-check")

        assert "whole directory" in str(exc.value)
        assert (pair.project / ".claude" / "skills" / "orbit-check").is_file()

    def test_no_profile_recorded(self, pair):
        _write_manifest(pair.project, {"source": "preset"})
        pair.artifact(".claude/rules/safety.md")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        message = str(exc.value)
        assert "names no profile" in message
        assert "osprey build" in message
        assert (pair.project / ".claude" / "rules" / "safety.md").is_file()

    def test_no_manifest_at_all(self, pair):
        (pair.project / MANIFEST_FILENAME).unlink()
        pair.artifact(".claude/rules/safety.md")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "names no profile" in str(exc.value)

    def test_profile_recorded_but_missing_on_disk(self, pair):
        (pair.profile / "profile.yml").unlink()
        pair.artifact(".claude/rules/safety.md")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        message = str(exc.value)
        assert "missing" in message
        assert str(pair.profile / "profile.yml") in message
        assert (pair.project / ".claude" / "rules" / "safety.md").is_file()

    def test_profile_inside_the_installed_package_is_refused(self, pair):
        """A claim must never write into the wheel — the build never reads there."""
        import osprey

        package_profile = (
            Path(osprey.__file__).resolve().parent
            / "profiles"
            / "presets"
            / "control-assistant.yml"
        )
        assert package_profile.is_file(), "expected a bundled preset to point the manifest at"
        _write_manifest(pair.project, {"profile_path_abs": str(package_profile)})
        pair.artifact(".claude/rules/safety.md")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "installed osprey package" in str(exc.value)
        assert (pair.project / ".claude" / "rules" / "safety.md").is_file()

    def test_unwritable_destination_leaves_the_artifact_in_the_project(self, pair):
        """A failed move must lose nothing, and must still explain itself.

        Asserts ``ScaffoldClaimError`` specifically, not ``(ScaffoldClaimError,
        OSError)``: a raw filesystem error reaching the caller is a traceback
        where an operator-facing message belongs, and a tuple here would let
        that regress silently.
        """
        pair.artifact(".claude/rules/safety.md", "# irreplaceable\n")
        # `rules` is a FILE in the profile, so the slot's parent cannot be made.
        (pair.profile / "rules").write_text("not a directory\n", encoding="utf-8")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        message = str(exc.value)
        assert str(pair.profile / "rules") in message
        assert "Nothing was moved" in message
        assert (pair.project / ".claude" / "rules" / "safety.md").read_text(
            encoding="utf-8"
        ) == "# irreplaceable\n"

    def test_unwritable_destination_reports_through_the_cli(self, pair):
        """The CLI must print the reason, not exit silently on a traceback."""
        pair.artifact(".claude/rules/safety.md", "# irreplaceable\n")
        (pair.profile / "rules").write_text("not a directory\n", encoding="utf-8")

        result = CliRunner().invoke(scaffold, ["claim", "rules/safety", "--repo", str(pair.repo)])

        assert result.exit_code != 0
        assert result.output.strip(), "the command exited without saying why"
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_symlinked_artifact_is_refused(self, pair, tmp_path):
        """Claiming the link would move a link, not the content.

        A profile must be self-contained, so the build refuses a convention
        directory symlinking out of it. Moving the link would report success
        and then break every later build of that profile — with the project
        copy already gone.
        """
        outside = tmp_path / "shared-safety.md"
        outside.write_text("# shared\n", encoding="utf-8")
        rules = pair.project / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "safety.md").symlink_to(outside)

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "symlink" in str(exc.value)
        assert (rules / "safety.md").is_symlink()
        assert not (pair.profile / "rules" / "safety.md").exists()

    def test_a_skill_containing_an_escaping_symlink_is_refused(self, pair, tmp_path):
        outside = tmp_path / "shared-reference.md"
        outside.write_text("# shared\n", encoding="utf-8")
        pair.artifact(".claude/skills/orbit-check/SKILL.md", "# orbit\n")
        (pair.project / ".claude/skills/orbit-check/reference.md").symlink_to(outside)

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "skills/orbit-check")

        assert "reference.md" in str(exc.value)
        assert (pair.project / ".claude" / "skills" / "orbit-check").is_dir()
        assert not (pair.profile / "skills" / "orbit-check").exists()

    def test_a_skills_internal_symlink_is_left_alone(self, pair):
        """The build permits a link resolving inside the profile, so a claim must too."""
        pair.artifact(".claude/skills/orbit-check/SKILL.md", "# orbit\n")
        (pair.project / ".claude/skills/orbit-check/alias.md").symlink_to("SKILL.md")

        outcome = claim_into_profile(pair.project, "skills/orbit-check")

        assert (outcome.profile_path / "alias.md").read_text(encoding="utf-8") == "# orbit\n"

    def test_a_broken_symlink_in_the_slot_counts_as_occupied(self, pair, tmp_path):
        """`exists()` follows links, so a broken one must not read as a free slot."""
        pair.artifact(".claude/rules/safety.md", "# project\n")
        (pair.profile / "rules").mkdir()
        (pair.profile / "rules" / "safety.md").symlink_to(tmp_path / "gone")

        with pytest.raises(ScaffoldClaimError) as exc:
            claim_into_profile(pair.project, "rules/safety")

        assert "already in the profile" in str(exc.value)
        assert (pair.project / ".claude" / "rules" / "safety.md").is_file()


# ── CLI surface ──────────────────────────────────────────────────────


class TestClaimCommand:
    """``osprey scaffold claim`` — exit codes and operator-facing reporting."""

    def test_successful_claim_reports_the_destination(self, pair):
        pair.artifact(".claude/rules/safety.md")

        result = CliRunner().invoke(scaffold, ["claim", "rules/safety", "--repo", str(pair.repo)])

        assert result.exit_code == 0, result.output
        assert "Moved" in result.output
        assert (pair.profile / "rules" / "safety.md").is_file()

    def test_no_profile_exits_non_zero(self, pair):
        _write_manifest(pair.project, {"source": "preset"})
        pair.artifact(".claude/rules/safety.md")

        result = CliRunner().invoke(scaffold, ["claim", "rules/safety", "--repo", str(pair.repo)])

        assert result.exit_code != 0
        assert "no profile" in result.output

    def test_unknown_name_exits_non_zero(self, pair):
        result = CliRunner().invoke(
            scaffold, ["claim", "nonexistent/thing", "--repo", str(pair.repo)]
        )

        assert result.exit_code != 0
        assert "Unknown artifact" in result.output

    def test_claim_on_a_real_build_moves_the_rule_out_of_the_project(self, repo_dir, project_dir):
        profile_root = repo_dir
        assert (project_dir / ".claude" / "rules" / "safety.md").is_file()

        result = CliRunner().invoke(scaffold, ["claim", "rules/safety", "--repo", str(repo_dir)])

        assert result.exit_code == 0, result.output
        assert (profile_root / "rules" / "safety.md").is_file()
        assert not (project_dir / ".claude" / "rules" / "safety.md").exists()
        # No YAML edit: ownership comes back from the build's convention scan.
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        assert "rules/safety" not in config.get("scaffold", {}).get("user_owned", [])

    def test_claim_twice_errors_on_the_second(self, repo_dir, project_dir):
        runner = CliRunner()
        assert (
            runner.invoke(scaffold, ["claim", "rules/safety", "--repo", str(repo_dir)]).exit_code
            == 0
        )
        result = runner.invoke(scaffold, ["claim", "rules/safety", "--repo", str(repo_dir)])

        assert result.exit_code != 0
        assert "already in the profile" in result.output


class TestPromptsList:
    """Tests for ``osprey scaffold list``."""

    def test_list_shows_all_artifacts(self, repo_dir, project_dir):
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--repo", str(repo_dir)])
        assert result.exit_code == 0
        assert "Build Artifacts" in result.output
        assert "claude-md" in result.output
        assert "agents/channel-finder" in result.output
        assert "hooks/error-guidance" in result.output

    def test_list_shows_framework_managed(self, repo_dir, project_dir):
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--repo", str(repo_dir)])
        assert "Framework-managed" in result.output

    def test_list_shows_facility(self, repo_dir, project_dir):
        """rules/facility appears either as framework-managed or user-owned."""
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--repo", str(repo_dir)])
        assert "rules/facility" in result.output

    def test_list_shows_ownership_the_catalog_does_not_know(self, pair):
        """A profile's own skill is user-owned though no framework template renders it.

        Asserts on names only: the listing is rendered through rich, which wraps
        rows at the terminal width, so any assertion on a longer phrase would be
        a coin toss. The row's content is pinned on ``_owned_row`` below.
        """
        skill = pair.profile / "skills" / "orbit-check"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# orbit\n", encoding="utf-8")
        update_config_add_user_owned(pair.project, "skills/orbit-check")

        result = CliRunner().invoke(scaffold, ["list", "--repo", str(pair.repo)])

        assert result.exit_code == 0, result.output
        assert "skills/orbit-check" in result.output
        assert "User-owned" in result.output


class TestOwnedRow:
    """What ``list`` says about one user-owned name."""

    def test_profile_supplied_ownership(self, pair):
        (pair.profile / "rules").mkdir()
        (pair.profile / "rules" / "facility-ops.md").write_text("# ops\n", encoding="utf-8")

        name, project_rel, origin = _owned_row(pair.profile, "rules/facility-ops")

        assert (name, project_rel, origin) == (
            "rules/facility-ops",
            ".claude/rules/facility-ops.md",
            "from profile",
        )

    def test_ownership_with_no_profile_slot_behind_it(self, pair):
        assert _owned_row(pair.profile, "rules/facility")[2] == "in project"

    def test_no_profile_at_all(self):
        assert _owned_row(None, "rules/facility")[2] == "in project"

    def test_a_skill_row_shows_the_directory_it_owns(self, pair):
        """The catalog lists a skill by its SKILL.md; ownership is the directory."""
        assert _owned_row(pair.profile, "skills/diagnose")[1] == ".claude/skills/diagnose"

    def test_a_name_with_no_convention_falls_back_to_the_catalog_path(self, pair):
        assert _owned_row(pair.profile, "claude-md")[1] == "CLAUDE.md"


class TestPromptsDiff:
    """Tests for ``osprey scaffold diff``."""

    def test_diff_not_owned_errors(self, repo_dir, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["diff", "rules/safety", "--repo", str(repo_dir)],
        )
        assert result.exit_code != 0
        assert "not user-owned" in result.output

    def test_diff_no_changes(self, repo_dir, project_dir):
        # Ownership as a build derives it: registered in config.yml, file in place.
        update_config_add_user_owned(project_dir, "rules/safety")
        result = CliRunner().invoke(
            scaffold,
            ["diff", "rules/safety", "--repo", str(repo_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "no differences" in result.output

    def test_diff_shows_changes(self, repo_dir, project_dir):
        update_config_add_user_owned(project_dir, "rules/safety")

        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.write_text("# Modified safety rules\nCustom content.\n")

        result = CliRunner().invoke(
            scaffold,
            ["diff", "rules/safety", "--repo", str(repo_dir)],
        )
        assert result.exit_code == 0
        assert "---" in result.output  # unified diff header
        assert "+++" in result.output


class TestPromptsUnclaim:
    """Tests for ``osprey scaffold unclaim``."""

    def test_unclaim_removes_config_entry(self, repo_dir, project_dir):
        update_config_add_user_owned(project_dir, "rules/safety")
        result = CliRunner().invoke(
            scaffold,
            ["unclaim", "rules/safety", "--repo", str(repo_dir)],
        )
        assert result.exit_code == 0, result.output

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        user_owned = config.get("scaffold", {}).get("user_owned", [])
        assert "rules/safety" not in user_owned

    def test_unclaim_removes_manifest_entry(self, repo_dir, project_dir):
        from osprey.cli.templates.manager import TemplateManager
        from osprey.services.build_artifacts.ownership import update_manifest_add_user_owned

        update_config_add_user_owned(project_dir, "rules/safety")
        update_manifest_add_user_owned(project_dir, TemplateManager(), {}, "rules/safety")

        CliRunner().invoke(
            scaffold,
            ["unclaim", "rules/safety", "--repo", str(repo_dir)],
        )

        manifest = json.loads((project_dir / MANIFEST_FILENAME).read_text())
        assert "rules/safety" not in manifest.get("user_owned", {})

    def test_unclaim_keeps_file(self, repo_dir, project_dir):
        """Unclaim keeps the file in place (user decides what to do)."""
        update_config_add_user_owned(project_dir, "rules/safety")
        CliRunner().invoke(
            scaffold,
            ["unclaim", "rules/safety", "--repo", str(repo_dir)],
        )

        assert (project_dir / ".claude" / "rules" / "safety.md").exists()

    def test_unclaim_not_owned_errors(self, repo_dir, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["unclaim", "rules/safety", "--repo", str(repo_dir)],
        )
        assert result.exit_code != 0
        assert "not user-owned" in result.output

    def test_unclaim_says_when_the_profile_still_supplies_it(self, pair):
        """Releasing config ownership does not stop the profile re-registering it."""
        (pair.profile / "rules").mkdir()
        (pair.profile / "rules" / "safety.md").write_text("# from profile\n", encoding="utf-8")
        update_config_add_user_owned(pair.project, "rules/safety")

        result = CliRunner().invoke(scaffold, ["unclaim", "rules/safety", "--repo", str(pair.repo)])

        assert result.exit_code == 0, result.output
        assert "still supplies it" in result.output


def _web_terminals_repo(root: Path, *, users=("alice", "bob")) -> Path:
    """Write a deployment repo whose built config.yml carries a clean stanza.

    Only the sections the generator reads, in the shape a build emits them from
    the profile's ``config:`` and ``deploy:`` blocks. Deliberately hand-written
    rather than built: these tests are about where the verbs read the stanza
    from, not about what a build puts in it — which is why the profile at the
    repo root is a stub. It is there because it is what makes this directory a
    repo, and the stanza the verbs read comes from the build zone beside it.
    """
    repo = root / "wt-repo"
    project = repo / "build"
    project.mkdir(parents=True)
    (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    config = {
        "facility": {"name": "Demo Light Source", "prefix": "dls"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 9080,
                "web_base_port": 9091,
                "artifact_base_port": 9291,
                "ariel_base_port": 9391,
                "lattice_base_port": 9491,
                "users": [{"name": name, "index": i} for i, name in enumerate(users)],
                "landing": {"groups": [{"type": "users"}]},
            }
        },
    }
    (project / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return repo


class TestWebTerminalsRetiredConfigOption:
    """`--config` is a tombstone: it names where the stanza went, always."""

    def test_lint_config_is_refused_and_names_the_replacements(self, tmp_path, caplog):
        """The criterion: a `--config` path the CLI never even opens is refused.

        `any.yml` does not exist, which is the point — the refusal has to reach
        the operator instead of a Click existence check that would say nothing
        about the move. Read the message off the record: the Rich handler wraps
        the rendered line to the console width, so a phrase asserted on
        `result.output` can break mid-match.
        """
        import logging

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(
                scaffold, ["web-terminals", "lint", "--config", str(tmp_path / "any.yml")]
            )

        assert result.exit_code != 0, result.output
        reported = caplog.text + (str(result.exception) if result.exception else "")
        assert "--config is no longer supported" in reported
        assert "osprey scaffold ci" in reported
        assert "`deploy:` block" in reported
        assert "/osprey-build-interview" in reported

    def test_refusal_is_unconditional_for_a_readable_config(self, tmp_path, caplog):
        """A file that parses cleanly is refused just the same — no fallback read."""
        import logging

        config = tmp_path / "facility-config.yml"
        config.write_text(yaml.safe_dump({"modules": {"web_terminals": {}}}), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(
                scaffold, ["web-terminals", "lint", "--config", str(config)]
            )

        assert result.exit_code != 0, result.output
        assert "--config is no longer supported" in caplog.text

    def test_render_config_is_refused_before_anything_is_written(self, tmp_path, caplog):
        """Render refuses on the same terms, and leaves --output untouched."""
        import logging

        out = tmp_path / "deploy"

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(
                scaffold,
                ["web-terminals", "render", "--config", "any.yml", "-o", str(out)],
            )

        assert result.exit_code != 0, result.output
        assert "--config is no longer supported" in caplog.text
        assert not out.exists()


class TestWebTerminalsReadsTheBuiltConfig:
    """Both verbs take their stanza from the repo's built config.yml."""

    def test_lint_reads_the_named_repo(self, tmp_path):
        repo = _web_terminals_repo(tmp_path)

        result = CliRunner().invoke(scaffold, ["web-terminals", "lint", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "no issues found" in result.output

    def test_lint_defaults_to_the_current_directory(self, tmp_path, monkeypatch):
        repo = _web_terminals_repo(tmp_path)
        monkeypatch.chdir(repo)

        result = CliRunner().invoke(scaffold, ["web-terminals", "lint"])

        assert result.exit_code == 0, result.output

    def test_lint_reports_the_repos_own_findings(self, tmp_path):
        """A duplicate in this repo's roster is what lint fails on."""
        repo = _web_terminals_repo(tmp_path, users=("alice", "alice"))

        result = CliRunner().invoke(scaffold, ["web-terminals", "lint", "--repo", str(repo)])

        assert result.exit_code != 0, result.output
        # One word: the finding line is rendered through rich, which wraps at
        # the console width, and a longer phrase could break mid-match.
        assert "duplicate" in result.output

    def test_lint_without_a_build_says_so(self, tmp_path, monkeypatch):
        """A repo that has not been built yet gets the remedy, not a puzzle."""
        repo = tmp_path / "unbuilt"
        repo.mkdir()
        (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")
        monkeypatch.chdir(repo)

        result = CliRunner().invoke(scaffold, ["web-terminals", "lint"])

        assert result.exit_code != 0
        assert "No build at" in result.output
        assert "osprey build" in result.output

    def test_render_writes_the_repos_artifacts(self, tmp_path):
        from osprey.deployment.web_terminals.render import render_web_terminals

        repo = _web_terminals_repo(tmp_path)
        project = repo / "build"
        out = tmp_path / "deploy"

        result = CliRunner().invoke(
            scaffold, ["web-terminals", "render", "--repo", str(repo), "-o", str(out)]
        )

        assert result.exit_code == 0, result.output
        expected = render_web_terminals(yaml.safe_load((project / "config.yml").read_text()))
        assert expected, "fixture produced no artifacts; the assertion below would be vacuous"
        for relative_path, content in expected.items():
            assert (out / relative_path).read_text(encoding="utf-8") == content

    def test_render_expands_vars_the_way_the_deploy_path_does(self, tmp_path, monkeypatch):
        """One loader: the verb resolves `${VAR}` through the `.env` passthrough.

        The discriminating case is a variable set in BOTH the shell environment
        and the project's `.env`. `ConfigBuilder` — the loader the deploy path
        uses — loads `.env` over the environment, so the `.env` value is the one
        a deploy would bake in. A bare `yaml.safe_load` + `resolve_env_vars`
        here would never read `.env` at all and would render the shell value,
        producing an artifact the deploy would not have produced.

        `monkeypatch.setenv` first is deliberate: the product code mutates
        `os.environ` for real, and pre-registering the name is what gets it
        restored at teardown.
        """
        project = _web_terminals_repo(tmp_path) / "build"
        monkeypatch.setenv("OSPREY_WT_TEST_IMAGE", "shell-value:latest")
        (project / ".env").write_text(
            "OSPREY_WT_TEST_IMAGE=dotenv-value:latest\n", encoding="utf-8"
        )

        config = yaml.safe_load((project / "config.yml").read_text())
        config["modules"]["web_terminals"]["nginx_image"] = "${OSPREY_WT_TEST_IMAGE}"
        (project / "config.yml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        # `.env` is read from the working directory, which is how a deploy run
        # from inside the project sees it.
        monkeypatch.chdir(project)
        out = tmp_path / "deploy"

        result = CliRunner().invoke(scaffold, ["web-terminals", "render", "-o", str(out)])

        assert result.exit_code == 0, result.output
        compose = (out / "docker-compose.web.yml").read_text(encoding="utf-8")
        assert "dotenv-value:latest" in compose
        assert "shell-value:latest" not in compose
        assert "${OSPREY_WT_TEST_IMAGE}" not in compose


def test_main_cli_lists_scaffold_not_prompts():
    """`osprey --help` exposes the renamed `scaffold` group, not the legacy `prompts`."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scaffold" in result.stdout
    # The old name must not appear as a command — match only at line start with
    # surrounding whitespace so we don't false-positive on documentation prose.
    for line in result.stdout.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("prompts "), f"legacy 'prompts' command leaked: {line!r}"
