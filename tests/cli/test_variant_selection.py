"""Tests for the host-local overlay-variant setting.

Covers the four resolution outcomes (`.env.variant` absent, named-and-resolved,
named-with-no-variants-in-repo, named-unknown), the non-fatal git-tracked probe,
and the two containment properties the design rests on: the setting file is not
a member of the env chain, and the `.gitignore` `osprey init` authors already
covers it.

`TestBuildIntegration` is the other half: the same repo, built twice with two
different settings, must produce the two renders the overlays describe — which
is the whole point of the feature and the only test that proves the setting
reaches `osprey build` at all.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import compute_profile_hash
from osprey.cli.init_cmd import _repo_gitignore
from osprey.cli.profile_conventions import unknown_root_entries
from osprey.cli.repo_resolver import PROFILE_FILENAME
from osprey.cli.validate_cmd import validate
from osprey.cli.variant_selection import (
    VARIANT_DIRNAME,
    VARIANT_SETTING_FILENAME,
    VARIANT_SETTING_KEY,
    VariantSelection,
    active_variant_overlay,
    known_variants,
    read_variant_setting,
    resolve_variant_selection,
    variant_setting_path,
)
from osprey.deployment import staleness
from osprey.deployment.staleness import DriftState, check_drift, profile_fingerprint
from osprey.errors import BuildProfileError
from osprey.utils.dotenv import ENV_CHAIN_FILENAMES
from tests.cli._lifecycle_build import stub_build


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo root with a tracked ``profile.yml`` and nothing else."""
    (tmp_path / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    return tmp_path


def write_setting(repo: Path, text: str) -> Path:
    path = repo / VARIANT_SETTING_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def define_variants(repo: Path, *names: str) -> None:
    root = repo / VARIANT_DIRNAME
    root.mkdir(exist_ok=True)
    for name in names:
        (root / f"{name}.yml").write_text(f"# {name}\n", encoding="utf-8")


class TestReadSetting:
    """``read_variant_setting`` — what the file says, in env-file syntax."""

    def test_absent_file_is_no_variant(self, repo: Path) -> None:
        assert read_variant_setting(repo) is None

    def test_reads_the_key(self, repo: Path) -> None:
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert read_variant_setting(repo) == "teststand"

    def test_env_file_syntax_is_honored(self, repo: Path) -> None:
        """Comments, an ``export`` prefix, and quoted values, like any ``.env``."""
        write_setting(
            repo,
            f'# which overlay this host builds\nexport {VARIANT_SETTING_KEY}="control-room"\n',
        )
        assert read_variant_setting(repo) == "control-room"

    def test_other_keys_are_ignored(self, repo: Path) -> None:
        write_setting(repo, f"SOMETHING_ELSE=1\n{VARIANT_SETTING_KEY}=teststand\n")
        assert read_variant_setting(repo) == "teststand"

    def test_file_without_the_key_is_no_variant(self, repo: Path) -> None:
        write_setting(repo, "# nothing selected yet\n")
        assert read_variant_setting(repo) is None

    def test_empty_value_is_no_variant(self, repo: Path) -> None:
        """An emptied setting means "build the tracked profile", not a variant named ''."""
        write_setting(repo, f"{VARIANT_SETTING_KEY}=\n")
        assert read_variant_setting(repo) is None

    def test_setting_path_is_at_the_repo_root(self, repo: Path) -> None:
        assert variant_setting_path(repo) == repo / VARIANT_SETTING_FILENAME


class TestKnownVariants:
    """``known_variants`` — the ``profiles/*.yml`` set, and only that."""

    def test_no_directory_is_no_variants(self, repo: Path) -> None:
        assert known_variants(repo) == []

    def test_sorted_stems(self, repo: Path) -> None:
        define_variants(repo, "teststand", "control-room")
        assert known_variants(repo) == ["control-room", "teststand"]

    def test_non_yml_entries_are_not_variants(self, repo: Path) -> None:
        define_variants(repo, "teststand")
        (repo / VARIANT_DIRNAME / "README.md").write_text("notes\n", encoding="utf-8")
        (repo / VARIANT_DIRNAME / "nested").mkdir()
        assert known_variants(repo) == ["teststand"]


class TestResolution:
    """``resolve_variant_selection`` — the four outcomes."""

    def test_no_setting_selects_nothing(self, repo: Path) -> None:
        define_variants(repo, "teststand")
        selection = resolve_variant_selection(repo)
        assert selection == VariantSelection(name=None, path=None)
        assert not selection.selected

    def test_named_variant_resolves_to_its_overlay(self, repo: Path) -> None:
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        selection = resolve_variant_selection(repo)

        assert selection.name == "teststand"
        assert selection.path == repo / VARIANT_DIRNAME / "teststand.yml"
        assert selection.path is not None and selection.path.is_file()
        assert selection.selected
        assert selection.notices == ()

    def test_unknown_variant_raises_listing_the_known_ones(self, repo: Path) -> None:
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=staging\n")

        with pytest.raises(BuildProfileError) as excinfo:
            resolve_variant_selection(repo)

        message = str(excinfo.value)
        assert "staging" in message
        assert "teststand" in message
        assert "control-room" in message
        assert VARIANT_SETTING_KEY in message

    def test_setting_without_variants_in_repo_is_a_notice_not_an_error(
        self, repo: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The repo dropped its variants; the tracked profile still builds."""
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        with caplog.at_level(logging.WARNING):
            selection = resolve_variant_selection(repo)

        assert selection.name == "teststand"
        assert selection.path is None
        assert not selection.selected
        assert len(selection.notices) == 1
        assert "defines no variants" in selection.notices[0]
        assert "defines no variants" in caplog.text

    def test_empty_variants_directory_is_the_same_notice(self, repo: Path) -> None:
        (repo / VARIANT_DIRNAME).mkdir()
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        selection = resolve_variant_selection(repo)

        assert selection.path is None
        assert selection.notices and "defines no variants" in selection.notices[0]

    def test_a_traversing_name_is_refused_as_unknown(self, repo: Path) -> None:
        """Membership is checked against the known set, so a path cannot escape."""
        define_variants(repo, "teststand")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=../../etc/passwd\n")

        with pytest.raises(BuildProfileError):
            resolve_variant_selection(repo)


class TestGitTrackedProbe:
    """The committed-by-accident warning — loud, and never fatal."""

    def test_untracked_setting_is_silent(self, repo: Path) -> None:
        define_variants(repo, "teststand")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert resolve_variant_selection(repo).notices == ()

    def test_tracked_setting_warns_and_still_resolves(
        self, repo: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        define_variants(repo, "teststand")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        _git_init_and_track(repo, VARIANT_SETTING_FILENAME)

        with caplog.at_level(logging.WARNING):
            selection = resolve_variant_selection(repo)

        assert selection.selected
        assert len(selection.notices) == 1
        assert "tracked by git" in selection.notices[0]
        assert "git rm --cached" in selection.notices[0]
        assert "tracked by git" in caplog.text

    def test_probe_failure_is_not_fatal(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No git on the host, or a git that hangs, must not fail a build."""
        define_variants(repo, "teststand")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        def explode(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr("osprey.cli.variant_selection.subprocess.run", explode)

        selection = resolve_variant_selection(repo)

        assert selection.selected
        assert selection.notices == ()


class TestContainment:
    """The two properties that keep the setting out of everything else."""

    def test_setting_file_is_not_in_the_env_chain(self) -> None:
        """It picks a profile layer; it is not a variable a deployment reads."""
        assert VARIANT_SETTING_FILENAME not in ENV_CHAIN_FILENAMES

    def test_init_gitignore_covers_and_documents_the_setting(self) -> None:
        gitignore = _repo_gitignore()
        assert "/.env*" in gitignore
        assert VARIANT_SETTING_FILENAME in gitignore
        assert VARIANT_SETTING_KEY in gitignore
        assert f"!/{VARIANT_SETTING_FILENAME}" not in gitignore


#: Both builds in this module render without a venv and without lifecycle
#: hooks: neither is what the overlay decides, and a real dependency install per
#: build would dominate the module's runtime.
BUILD_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: Two overlays that differ in one visible key each, so a render can be told
#: apart from the other by reading `build/config.yml` alone.
TESTSTAND_OVERLAY = """\
config:
  facility.name: Test Stand
  web.theme: dark
"""

CONTROL_ROOM_OVERLAY = """\
config:
  facility.name: Control Room
  web.theme: light
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _build(runner: CliRunner, repo: Path):
    """Run ``osprey build`` from *repo*, as an operator standing there would."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return runner.invoke(build, BUILD_FLAGS)
    finally:
        os.chdir(previous)


def _rendered_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


def _write_variant(repo: Path, name: str, body: str) -> None:
    root = repo / VARIANT_DIRNAME
    root.mkdir(exist_ok=True)
    (root / f"{name}.yml").write_text(body, encoding="utf-8")


class TestBuildIntegration:
    """``osprey build`` renders what THIS host's setting selects."""

    @pytest.fixture
    def variant_repo(self, lifecycle_repo: Path) -> Path:
        """The exemplar repo, given two host variants to choose between."""
        _write_variant(lifecycle_repo, "teststand", TESTSTAND_OVERLAY)
        _write_variant(lifecycle_repo, "control-room", CONTROL_ROOM_OVERLAY)
        return lifecycle_repo

    def test_two_settings_from_one_repo_produce_the_two_renders(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """The feature, end to end: same source, two hosts, two builds."""
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        first = _build(runner, variant_repo)
        assert first.exit_code == 0, first.output
        teststand = _rendered_config(variant_repo)

        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=control-room\n")
        second = _build(runner, variant_repo)
        assert second.exit_code == 0, second.output
        control_room = _rendered_config(variant_repo)

        assert teststand["facility"]["name"] == "Test Stand"
        assert teststand["web"]["theme"] == "dark"
        assert control_room["facility"]["name"] == "Control Room"
        assert control_room["web"]["theme"] == "light"

    def test_no_setting_builds_the_tracked_profile(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """Variants in the repo change nothing until a host selects one."""
        result = _build(runner, variant_repo)

        assert result.exit_code == 0, result.output
        config = _rendered_config(variant_repo)
        assert config["web"]["theme"] == "light"
        assert config.get("facility", {}).get("name") != "Test Stand"

    def test_the_overlay_only_overrides_what_it_names(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """A layer, not a replacement: the tracked profile still supplies the rest.

        ``control_system.type`` is the un-named key here — the overlay says
        nothing about it, so it stays whatever the tracked profile baselined,
        which for the exemplar is the live stand-in."""
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        assert _build(runner, variant_repo).exit_code == 0

        config = _rendered_config(variant_repo)
        assert config["control_system"]["type"] == "live_standin"
        assert config["project_name"] == variant_repo.name

    def test_the_selected_variant_is_named_in_the_build_output(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """An operator reading a render must be told which host shape it is."""
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        result = _build(runner, variant_repo)

        assert result.exit_code == 0, result.output
        assert "teststand" in result.output

    def test_an_unknown_variant_stops_the_build(
        self, runner: CliRunner, variant_repo: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """And leaves the previous ``build/`` alone, like every other refusal."""
        assert _build(runner, variant_repo).exit_code == 0
        before = _rendered_config(variant_repo)

        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=staging\n")
        with caplog.at_level(logging.ERROR):
            result = _build(runner, variant_repo)

        assert result.exit_code != 0
        # The refusal reaches the operator through the build's error channel,
        # naming both the variant asked for and the ones that would work.
        assert "staging" in caplog.text
        assert "teststand" in caplog.text
        assert _rendered_config(variant_repo) == before

    def test_a_real_build_stamps_a_fingerprint_that_covers_its_variant(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """Stamp side and check side agree, and both see the overlay.

        The end of the finding this test exists for: the build's own manifest,
        not a stubbed one, must read CLEAN right after it is written and stale
        as soon as the overlay it rendered from moves.
        """
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert _build(runner, variant_repo).exit_code == 0

        assert check_drift(variant_repo).state is DriftState.CLEAN

        _write_variant(
            variant_repo, "teststand", TESTSTAND_OVERLAY.replace("Test Stand", "Test Stand 2")
        )

        assert check_drift(variant_repo).state is DriftState.DRIFT

    def test_switching_hosts_makes_the_existing_build_stale(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """Otherwise ``osprey up`` starts the other host's render without a word."""
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert _build(runner, variant_repo).exit_code == 0

        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=control-room\n")

        assert check_drift(variant_repo).state is DriftState.DRIFT

    def test_the_variants_directory_is_not_an_unrecognized_entry(self, variant_repo: Path) -> None:
        """``profiles/`` is SOURCE material, so no build warns about it."""
        assert unknown_root_entries(variant_repo) == []

    def test_a_variant_reaches_the_persona_renders_too(
        self, runner: CliRunner, variant_repo: Path
    ) -> None:
        """A persona ships in the same stack, so it is built for the same host."""
        personas = sorted((variant_repo / "personas").glob("*.yml"))
        if not personas:
            pytest.skip("the exemplar repo carries no persona deltas")
        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        assert _build(runner, variant_repo).exit_code == 0

        rendered = variant_repo / "build" / f"{variant_repo.name}-{personas[0].stem}"
        config = yaml.safe_load((rendered / "config.yml").read_text(encoding="utf-8"))
        assert config["facility"]["name"] == "Test Stand"


def _overlay(repo: Path, name: str) -> Path:
    return repo / VARIANT_DIRNAME / f"{name}.yml"


class TestActiveOverlay:
    """``active_variant_overlay`` — the quiet reading the fingerprint uses."""

    def test_no_setting_selects_nothing(self, repo: Path) -> None:
        define_variants(repo, "teststand")
        assert active_variant_overlay(repo) is None

    def test_the_selected_overlay(self, repo: Path) -> None:
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert active_variant_overlay(repo) == _overlay(repo, "teststand")

    def test_an_unhonorable_name_is_quiet_rather_than_fatal(self, repo: Path) -> None:
        """The build refuses on it; the fingerprint must still be computable."""
        define_variants(repo, "teststand")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=staging\n")
        assert active_variant_overlay(repo) is None

    def test_a_directory_that_is_not_a_repo_selects_nothing(self, tmp_path: Path) -> None:
        assert active_variant_overlay(tmp_path / "personas") is None


class TestFingerprintFollowsTheVariant:
    """The drift fingerprint covers the overlay THIS host builds.

    Without this, the overlay is the one build input nothing hashes: an operator
    edits ``profiles/teststand.yml``, or points ``.env.variant`` at another
    host's overlay, and every staleness check still says the render is current.
    """

    def _hash(self, repo: Path) -> str | None:
        return compute_profile_hash(repo / PROFILE_FILENAME)

    def test_variants_alone_do_not_move_the_hash(self, repo: Path) -> None:
        """Nothing selected, nothing folded — no repo is invalidated by the feature."""
        before = self._hash(repo)
        define_variants(repo, "teststand", "control-room")
        assert self._hash(repo) == before

    def test_selecting_a_variant_moves_the_hash(self, repo: Path) -> None:
        define_variants(repo, "teststand")
        before = self._hash(repo)
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        assert self._hash(repo) != before

    def test_editing_the_selected_overlay_moves_the_hash(self, repo: Path) -> None:
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        before = self._hash(repo)

        _overlay(repo, "teststand").write_text("config:\n  web.theme: dark\n", encoding="utf-8")

        assert self._hash(repo) != before

    def test_editing_an_unselected_overlay_does_not(self, repo: Path) -> None:
        """Another host's overlay is not an input to this host's render."""
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        before = self._hash(repo)

        _overlay(repo, "control-room").write_text("config:\n  web.theme: dark\n", encoding="utf-8")

        assert self._hash(repo) == before

    def test_switching_between_identical_overlays_moves_the_hash(self, repo: Path) -> None:
        """The SELECTION is part of the fingerprint, not only the bytes it names."""
        define_variants(repo, "teststand")
        (repo / VARIANT_DIRNAME / "control-room.yml").write_text("# teststand\n", encoding="utf-8")
        assert (
            _overlay(repo, "teststand").read_bytes() == _overlay(repo, "control-room").read_bytes()
        )
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        before = self._hash(repo)

        write_setting(repo, f"{VARIANT_SETTING_KEY}=control-room\n")

        assert self._hash(repo) != before

    def test_an_unhonorable_name_hashes_as_the_tracked_profile(self, repo: Path) -> None:
        """A name the repo cannot honor is a refusal at build time, not a hash."""
        define_variants(repo, "teststand")
        before = self._hash(repo)
        write_setting(repo, f"{VARIANT_SETTING_KEY}=staging\n")
        assert self._hash(repo) == before

    def test_the_selected_overlay_is_named_in_the_key_digests(self, repo: Path) -> None:
        """So a refusal can point at the file the operator edited."""
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        fingerprint = profile_fingerprint(repo / PROFILE_FILENAME)

        assert fingerprint is not None
        keys = set(fingerprint.key_digests)
        assert f"{staleness.MATERIAL_PREFIX}{VARIANT_DIRNAME}/teststand.yml" in keys
        assert f"{staleness.MATERIAL_PREFIX}{VARIANT_DIRNAME}/control-room.yml" not in keys


def _stamp_build(repo: Path) -> Path:
    """A ``build/`` stamped exactly as a build of *repo* would stamp it now.

    Both halves of the provenance the drift check reads: the fingerprint itself
    and the per-key digests that name what moved. Written from the same
    functions the build calls, which is the property under test — a stamp side
    that computed the fingerprint differently from the check side would report
    every build as drifted, or none.
    """
    build_dir = stub_build(repo)
    fingerprint = profile_fingerprint(repo / PROFILE_FILENAME)
    assert fingerprint is not None
    manifest_path = build_dir / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["creation"][staleness.KEY_DIGESTS_MANIFEST_KEY] = fingerprint.key_digests
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return build_dir


class TestVariantDrift:
    """``osprey up``'s gate, over a repo that builds a host overlay."""

    @pytest.fixture
    def variant_repo(self, repo: Path) -> Path:
        define_variants(repo, "teststand", "control-room")
        write_setting(repo, f"{VARIANT_SETTING_KEY}=teststand\n")
        return repo

    def test_a_stamped_variant_build_is_clean(self, variant_repo: Path) -> None:
        _stamp_build(variant_repo)
        assert check_drift(variant_repo).state is DriftState.CLEAN

    def test_editing_the_selected_overlay_is_drift(self, variant_repo: Path) -> None:
        _stamp_build(variant_repo)

        _overlay(variant_repo, "teststand").write_text(
            "config:\n  web.theme: dark\n", encoding="utf-8"
        )

        report = check_drift(variant_repo)
        assert report.state is DriftState.DRIFT
        # And the refusal names the file that moved, not "files the profile
        # points at" — the operator is one directory away from the fix.
        assert f"{VARIANT_DIRNAME}/teststand.yml" in report.changed_keys

    def test_switching_the_setting_is_drift(self, variant_repo: Path) -> None:
        """``build/`` holds the other host's render until it is rebuilt."""
        _stamp_build(variant_repo)

        write_setting(variant_repo, f"{VARIANT_SETTING_KEY}=control-room\n")

        assert check_drift(variant_repo).state is DriftState.DRIFT

    def test_dropping_the_setting_is_drift(self, variant_repo: Path) -> None:
        _stamp_build(variant_repo)

        variant_setting_path(variant_repo).unlink()

        assert check_drift(variant_repo).state is DriftState.DRIFT

    def test_editing_an_unselected_overlay_is_not_drift(self, variant_repo: Path) -> None:
        """No host builds it here, so it cannot make this build stale."""
        _stamp_build(variant_repo)

        _overlay(variant_repo, "control-room").write_text(
            "config:\n  web.theme: dark\n", encoding="utf-8"
        )

        assert check_drift(variant_repo).state is DriftState.CLEAN

    def test_a_repo_without_variants_is_unaffected(self, repo: Path) -> None:
        """The no-variant path is byte-for-byte the behaviour it always had."""
        _stamp_build(repo)
        assert check_drift(repo).state is DriftState.CLEAN

        (repo / PROFILE_FILENAME).write_text("name: renamed\n", encoding="utf-8")

        assert check_drift(repo).state is DriftState.DRIFT


def _run_validate(runner: CliRunner, repo: Path):
    """Run ``osprey validate`` from *repo*, zero-argument, as an operator would."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return runner.invoke(validate, [])
    finally:
        os.chdir(previous)


class TestValidateHonorsTheVariant:
    """``osprey validate`` judges the document the build renders, not another.

    Validation that resolved the tracked profile while the build merged an
    overlay would pass on a repo whose real render is broken — and the operator
    would learn it from ``osprey build``, the verb they ran validate to stay
    ahead of.
    """

    def test_the_merged_document_is_what_is_judged(
        self, runner: CliRunner, lifecycle_repo: Path
    ) -> None:
        _write_variant(lifecycle_repo, "teststand", "name: demo-teststand\n")
        write_setting(lifecycle_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        result = _run_validate(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        # The reported name comes from the overlay, so the overlay was merged.
        assert "demo-teststand" in result.output

    def test_the_variant_is_named_the_way_the_build_names_it(
        self, runner: CliRunner, lifecycle_repo: Path
    ) -> None:
        _write_variant(lifecycle_repo, "teststand", TESTSTAND_OVERLAY)
        write_setting(lifecycle_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        result = _run_validate(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert f"host variant teststand ({VARIANT_DIRNAME}/teststand.yml" in result.output

    def test_no_setting_judges_the_tracked_profile_silently(
        self, runner: CliRunner, lifecycle_repo: Path
    ) -> None:
        _write_variant(lifecycle_repo, "teststand", "name: demo-teststand\n")

        result = _run_validate(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert "demo-teststand" not in result.output
        assert "host variant" not in result.output

    def test_an_unknown_variant_is_refused(self, runner: CliRunner, lifecycle_repo: Path) -> None:
        """The same refusal the build gives, at the verb that exists to find it first."""
        _write_variant(lifecycle_repo, "teststand", TESTSTAND_OVERLAY)
        write_setting(lifecycle_repo, f"{VARIANT_SETTING_KEY}=staging\n")

        result = _run_validate(runner, lifecycle_repo)

        assert result.exit_code != 0

    def test_a_persona_delta_has_no_variant(self, runner: CliRunner, lifecycle_repo: Path) -> None:
        """A delta named directly is a file, not a deployment — nothing overlays it."""
        personas = sorted((lifecycle_repo / "personas").glob("*.yml"))
        if not personas:
            pytest.skip("the exemplar repo carries no persona deltas")
        _write_variant(lifecycle_repo, "teststand", "name: demo-teststand\n")
        write_setting(lifecycle_repo, f"{VARIANT_SETTING_KEY}=teststand\n")

        previous = Path.cwd()
        os.chdir(lifecycle_repo)
        try:
            result = runner.invoke(validate, [str(personas[0])])
        finally:
            os.chdir(previous)

        assert result.exit_code == 0, result.output
        assert "host variant" not in result.output


def _git_init_and_track(repo: Path, relative: str) -> None:
    """Make ``relative`` a tracked path in a throwaway repo, or skip."""
    env = {
        "GIT_CONFIG_GLOBAL": str(repo / ".gitconfig-none"),
        "GIT_CONFIG_SYSTEM": str(repo / ".gitconfig-none"),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(repo),
    }
    try:
        for args in (
            ("init", "--quiet"),
            ("add", "--force", "--", relative),
        ):
            subprocess.run(
                ["git", *args], cwd=repo, env=env, capture_output=True, check=True, timeout=30
            )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
