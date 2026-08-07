"""Tests for ScaffoldGalleryService — bridges BuildArtifactCatalog + TemplateManager for web UI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.interfaces.web_terminal.scaffold_gallery_service import (
    ScaffoldGalleryService,
    restore_scaffold_bodies,
)
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_ARTIFACT = "rules/safety"  # Always available, real content


@pytest.fixture()
def project_dir(tmp_path):
    """Create a real OSPREY project via ``osprey build --preset control-assistant``.

    The gallery tests assert presence of channel-finder/data-visualizer agents
    and the diagnose skill, which only ship with the full control-assistant
    preset (not hello-world).
    """
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "gallery-test",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "gallery-test"


@pytest.fixture()
def service(project_dir):
    return ScaffoldGalleryService(project_dir)


def _unreachable_profile(project_dir: Path) -> Path:
    """Delete the profile a project names, leaving the manifest pointing at it.

    What a deployed container looks like: the manifest records the path the
    build machine resolved, and that path does not exist here.
    """
    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
    profile_path = manifest.get("build_args", {}).get("profile_path_abs")
    assert profile_path, "preset builds are expected to materialize a profile"
    shutil.rmtree(Path(profile_path).parent)
    return project_dir


@pytest.fixture()
def detached_project_dir(project_dir, monkeypatch):
    """A project that names no profile at all — a pre-profile manifest.

    The only topology where ownership still belongs in the project's own
    config.yml. Note that this drops ``build_args.profile_path_abs`` as well as
    the profile directory: a project that *names* a profile it cannot reach is
    a different case, and must refuse rather than fall back here (see
    :class:`TestDegradedTopology`).
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    _unreachable_profile(project_dir)

    manifest_path = project_dir / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.get("build_args", {}).pop("profile_path_abs", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return project_dir


@pytest.fixture()
def detached_service(detached_project_dir):
    return ScaffoldGalleryService(detached_project_dir)


@pytest.fixture()
def degraded_project_dir(project_dir, monkeypatch):
    """A project naming a profile that cannot be reached, with no volume."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return _unreachable_profile(project_dir)


@pytest.fixture()
def volume_dir(tmp_path):
    """Stand-in for the per-user claude-config volume mount."""
    return tmp_path / "claude-config"


@pytest.fixture()
def container_project(project_dir, volume_dir, monkeypatch):
    """A deployed web terminal: image-baked project tree plus a durable volume.

    The manifest still names its profile — that is exactly what an image
    carries — and the path is unreachable from inside the container. What
    survives a recreation is the volume, and nothing else.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(volume_dir))
    return _unreachable_profile(project_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_index(volume_dir: Path) -> dict:
    """The durable ownership index as written on the volume."""
    path = volume_dir / "osprey" / "scaffold" / "user_owned.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recreate_container(project_dir: Path, destination: Path) -> Path:
    """Copy a pristine image-baked tree, as a container recreation would.

    The point of the copy is that it is taken *before* anything is claimed: it
    is the image, unchanged. Pairing it with the same volume is the whole test
    — the operator's work has to come back from somewhere, and the tree is not
    that somewhere.
    """
    shutil.copytree(project_dir, destination)
    return destination


def _read_config(project_dir: Path) -> dict:
    with open(project_dir / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_user_owned(project_dir: Path) -> list[str]:
    cfg = _read_config(project_dir)
    return cfg.get("scaffold", {}).get("user_owned", [])


# ===========================================================================
# List artifacts
# ===========================================================================


class TestListArtifacts:
    """Tests for ScaffoldGalleryService.list_artifacts()."""

    def test_list_artifacts_only_existing_files(self, service, project_dir):
        """Every returned artifact corresponds to a file that exists on disk."""
        result = service.list_artifacts()
        assert len(result) > 0, "Expected at least some artifacts from osprey build"
        for art in result:
            fpath = project_dir / art["output_path"]
            assert fpath.exists(), f"{art['output_path']} listed but not on disk"

    def test_list_artifacts_bounded_by_registry(self, service):
        """Returned count is at most the registry size (no phantom artifacts)."""
        registry = BuildArtifactCatalog.default()
        result = service.list_artifacts()
        assert len(result) <= len(registry.all_artifacts())

    def test_list_artifacts_status_framework(self, service):
        """An artifact without user-ownership has status 'framework'."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert SAFE_ARTIFACT in by_name
        assert by_name[SAFE_ARTIFACT]["status"] == "framework"

    def test_list_artifacts_status_user_owned(self, service, project_dir):
        """After claiming, the artifact shows status 'user-owned'."""
        service.scaffold_override(SAFE_ARTIFACT)
        # Re-create service to pick up refreshed config
        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[SAFE_ARTIFACT]["status"] == "user-owned"

    def test_list_artifacts_categories(self, service):
        """Category is derived from the canonical name prefix."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}

        # "agents/channel-finder" -> category "agents"
        assert by_name["agents/channel-finder"]["category"] == "agents"
        # "rules/safety" -> category "rules"
        assert by_name["rules/safety"]["category"] == "rules"
        # "claude-md" (no slash) -> category "config"
        assert by_name["claude-md"]["category"] == "config"

    def test_list_artifacts_summary_counts(self, service):
        """Sum of framework + overridden equals total."""
        result = service.list_artifacts()
        framework = sum(1 for a in result if a["status"] == "framework")
        owned = sum(1 for a in result if a["status"] == "user-owned")
        assert framework + owned == len(result)

    def test_list_artifacts_reads_disk_metadata(self, service, project_dir):
        """Modifying front-matter on disk is reflected in list_artifacts()."""
        # Claim and overwrite with custom front-matter
        service.scaffold_override(SAFE_ARTIFACT)
        disk_path = project_dir / ".claude" / "rules" / "safety.md"
        disk_path.write_text(
            "---\nsummary: Custom safety summary\n"
            "description: Custom safety description\n---\n# Custom\n",
            encoding="utf-8",
        )

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[SAFE_ARTIFACT]["summary"] == "Custom safety summary"
        assert by_name[SAFE_ARTIFACT]["description"] == "Custom safety description"

    def test_list_artifacts_excludes_missing_files(self, project_dir):
        """Artifacts not on disk are not returned (filesystem-first)."""
        # Delete a known artifact file from the initialized project
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        if safety_file.exists():
            safety_file.unlink()

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        names = {a["name"] for a in result}
        assert SAFE_ARTIFACT not in names


# ===========================================================================
# Content retrieval
# ===========================================================================


class TestGetContent:
    """Tests for get_content, get_framework_content, get_override_content."""

    def test_get_content_framework(self, service):
        """Framework artifact returns non-empty content with source='framework'."""
        result = service.get_content(SAFE_ARTIFACT)
        assert result["source"] == "framework"
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_get_content_user_owned(self, service, project_dir):
        """After claim + modify, get_content returns the user's version."""
        service.scaffold_override(SAFE_ARTIFACT)
        custom = "# Custom safety rules\nDo not touch anything.\n"
        service.save_override(SAFE_ARTIFACT, custom)

        svc = ScaffoldGalleryService(project_dir)
        result = svc.get_content(SAFE_ARTIFACT)
        assert result["source"] == "user-owned"
        assert result["content"] == custom

    def test_get_framework_content_renders(self, service):
        """get_framework_content returns non-empty rendered content."""
        content = service.get_framework_content(SAFE_ARTIFACT)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_get_framework_content_unknown(self, service):
        """Unknown artifact name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_framework_content("nonexistent/artifact")

    def test_get_override_content_not_owned(self, service):
        """Non-owned artifact returns None."""
        result = service.get_override_content(SAFE_ARTIFACT)
        assert result is None

    def test_get_override_content_exists(self, service, project_dir):
        """After claiming, get_override_content returns file content."""
        scaffold_result = service.scaffold_override(SAFE_ARTIFACT)
        expected_content = scaffold_result["content"]

        svc = ScaffoldGalleryService(project_dir)
        content = svc.get_override_content(SAFE_ARTIFACT)
        assert content is not None
        assert content == expected_content


# ===========================================================================
# Diff
# ===========================================================================


class TestComputeDiff:
    """Tests for compute_diff."""

    def test_compute_diff_identical(self, service):
        """Claim without modification yields has_diff=False."""
        service.scaffold_override(SAFE_ARTIFACT)
        result = service.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is False
        assert result["additions"] == 0
        assert result["deletions"] == 0

    def test_compute_diff_with_changes(self, service, project_dir):
        """Claim, modify, diff shows additions and deletions."""
        service.scaffold_override(SAFE_ARTIFACT)
        service.save_override(SAFE_ARTIFACT, "# Completely replaced content\n")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is True
        assert result["additions"] > 0
        assert result["deletions"] > 0
        assert isinstance(result["unified_diff"], str)
        assert len(result["unified_diff"]) > 0

    def test_compute_diff_not_owned(self, service):
        """Diff on a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.compute_diff(SAFE_ARTIFACT)


# ===========================================================================
# Scaffold (Claim)
# ===========================================================================


class TestScaffold:
    """Tests for scaffold_override (claim)."""

    def test_scaffold_claims_and_updates_config(self, detached_service, detached_project_dir):
        """With no profile to claim into, ownership is recorded in config.yml."""
        result = detached_service.scaffold_override(SAFE_ARTIFACT)

        assert result["status"] == "claimed"
        assert "output_path" in result
        assert len(result["content"]) > 0

        # config.yml has the user_owned entry
        user_owned = _get_user_owned(detached_project_dir)
        assert SAFE_ARTIFACT in user_owned

    def test_scaffold_already_owned(self, service):
        """Claiming the same artifact twice raises FileExistsError."""
        service.scaffold_override(SAFE_ARTIFACT)
        with pytest.raises(FileExistsError, match="already user-owned"):
            service.scaffold_override(SAFE_ARTIFACT)


# ===========================================================================
# Save
# ===========================================================================


class TestSaveOverride:
    """Tests for save_override."""

    def test_save_override_writes_file(self, detached_service, detached_project_dir):
        """Save writes new content to the user-owned file."""
        detached_service.scaffold_override(SAFE_ARTIFACT)
        new_content = "# Updated safety rules\nAll writes require approval.\n"
        result = detached_service.save_override(SAFE_ARTIFACT, new_content)

        assert result["status"] == "saved"

        # Verify file content on disk
        output_file = detached_project_dir / result["path"]
        assert output_file.read_text(encoding="utf-8") == new_content

    def test_save_override_not_owned(self, service):
        """Saving to a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.save_override(SAFE_ARTIFACT, "some content")


# ===========================================================================
# Unclaim
# ===========================================================================


class TestUnoverride:
    """Tests for unoverride (unclaim)."""

    def test_unclaim_removes_config(self, detached_service, detached_project_dir):
        """Unclaim removes the config entry.

        Config mode, because that is the only topology where config.yml holds
        ownership and removing the entry is what releasing means. Asserting
        this against a profile-built project used to pass for the wrong reason:
        the claim there never wrote config.yml in the first place, so "the
        entry is gone" was true before the unclaim ran. What profile mode does
        instead is pinned in :class:`TestUnclaimIsHonestAboutTheProfile`.
        """
        detached_service.scaffold_override(SAFE_ARTIFACT)
        assert SAFE_ARTIFACT in _get_user_owned(detached_project_dir)

        result = ScaffoldGalleryService(detached_project_dir).unoverride(SAFE_ARTIFACT)

        assert result["status"] == "removed"
        assert SAFE_ARTIFACT not in _get_user_owned(detached_project_dir)

    def test_unclaim_not_owned(self, service):
        """Unclaiming a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.unoverride(SAFE_ARTIFACT)


# ===========================================================================
# Description extraction
# ===========================================================================


class TestDescriptionExtraction:
    """Tests for two-tier summary/description extraction from front matter."""

    def test_agent_has_summary_from_front_matter(self, service):
        """Agent artifact summary comes from template front matter, not registry."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert art["summary"] == ("Creates plots, charts, dashboards, and compiles LaTeX reports")
        # Summary should differ from the full description
        assert art["summary"] != art["description"]

    def test_hook_has_summary_from_front_matter(self, service):
        """Hook artifact summary comes from docstring YAML front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["hooks/limits"]
        assert art["summary"] == ("Validates channel write values against the limits database")

    def test_config_falls_back_to_registry(self, service):
        """Config artifacts (JSON templates) fall back to registry description."""
        registry = BuildArtifactCatalog.default()
        reg_art = registry.get("claude-md")
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["claude-md"]
        # No front matter in JSON templates -> falls back to registry
        assert art["summary"] == reg_art.description

    def test_description_is_full_from_front_matter(self, service):
        """Agent description field comes from full front matter description."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert "Creates data visualizations" in art["description"]

    def test_rule_has_summary_and_description(self, service):
        """Rule artifacts get both summary and description from front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["rules/safety"]
        assert art["summary"] == "Safety boundaries, channel write safety, and data integrity"
        assert "tool confinement" in art["description"]

    def test_skill_diagnose_has_summary_from_front_matter(self, service):
        """Diagnose skill gets summary from skill front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["skills/diagnose"]
        assert art["summary"] == "Investigate OSPREY infrastructure and agent failures"


# ===========================================================================
# Untracked file detection
# ===========================================================================


class TestScanUntracked:
    """Tests for scan_untracked — detecting files active in Claude Code but not managed."""

    def test_scan_untracked_finds_orphaned_files(self, service, project_dir):
        """A .md file in .claude/rules/ not in the registry is reported as untracked."""
        orphan = project_dir / ".claude" / "rules" / "my-custom-rule.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# My Custom Rule\nDo something special.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/my-custom-rule" in names

    def test_scan_untracked_excludes_registered(self, service, project_dir):
        """Registry artifacts that exist on disk are NOT reported as untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\nExisting framework content.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/safety" not in names

    def test_scan_untracked_excludes_user_owned(self, service, project_dir):
        """Custom files already in user_owned are NOT reported as untracked."""
        orphan = project_dir / ".claude" / "rules" / "already-claimed.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Already Claimed\n", encoding="utf-8")

        service.register_untracked("rules/already-claimed")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/already-claimed" not in names

    def test_scan_untracked_returns_correct_category(self, service, project_dir):
        """Category is derived from the first path component."""
        orphan = project_dir / ".claude" / "agents" / "rogue-agent.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Rogue Agent\n", encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["agents/rogue-agent"]["category"] == "agents"

    def test_scan_untracked_returns_preview(self, service, project_dir):
        """Untracked files include a text preview."""
        orphan = project_dir / ".claude" / "rules" / "preview-test.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Preview Test\nSome content here.\n"
        orphan.write_text(content, encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["rules/preview-test"]["preview"] == content

    def test_scan_untracked_empty_when_no_orphans(self, service):
        """Returns empty list when all files are tracked."""
        result = service.scan_untracked()
        assert result == []


class TestRegisterUntracked:
    """Tests for register_untracked — adding custom files to config."""

    def test_register_untracked_adds_to_config(self, detached_service, detached_project_dir):
        """Registering adds the canonical name to scaffold.user_owned in config."""
        orphan = detached_project_dir / ".claude" / "rules" / "new-rule.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# New Rule\n", encoding="utf-8")

        result = detached_service.register_untracked("rules/new-rule")
        assert result["status"] == "registered"

        user_owned = _get_user_owned(detached_project_dir)
        assert "rules/new-rule" in user_owned

    def test_register_untracked_file_must_exist(self, service):
        """Registering a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.register_untracked("rules/nonexistent")

    def test_register_untracked_already_registered(self, service, project_dir):
        """Registering an already-registered file raises FileExistsError."""
        orphan = project_dir / ".claude" / "rules" / "dupe.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Dupe\n", encoding="utf-8")

        service.register_untracked("rules/dupe")
        svc = ScaffoldGalleryService(project_dir)
        with pytest.raises(FileExistsError, match="already registered"):
            svc.register_untracked("rules/dupe")


class TestDeleteUntracked:
    """Tests for delete_untracked — removing orphaned files from disk."""

    def test_delete_untracked_removes_file(self, service, project_dir):
        """Deleting removes the file from disk."""
        orphan = project_dir / ".claude" / "rules" / "to-delete.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Delete Me\n", encoding="utf-8")

        result = service.delete_untracked("rules/to-delete")
        assert result["status"] == "deleted"
        assert not orphan.exists()

    def test_delete_untracked_file_must_exist(self, service):
        """Deleting a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.delete_untracked("rules/ghost")

    def test_delete_untracked_rejects_framework_artifact(self, service, project_dir):
        """Cannot delete a framework-registered artifact via delete_untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\n", encoding="utf-8")

        with pytest.raises(ValueError, match="framework artifact"):
            service.delete_untracked("rules/safety")


class TestCustomArtifacts:
    """Tests for custom user artifacts appearing in list_artifacts and get_content."""

    def test_list_artifacts_includes_custom(self, service, project_dir):
        """After registering a custom file, it appears in list_artifacts."""
        orphan = project_dir / ".claude" / "rules" / "custom.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("---\nsummary: My custom rule\n---\n# Custom\n", encoding="utf-8")

        service.register_untracked("rules/custom")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert "rules/custom" in by_name
        art = by_name["rules/custom"]
        assert art["status"] == "user-owned"
        assert art["custom"] is True
        assert art["category"] == "rules"
        assert art["summary"] == "My custom rule"

    def test_get_content_custom_artifact(self, service, project_dir):
        """get_content works for registered custom files."""
        orphan = project_dir / ".claude" / "rules" / "readable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Readable Custom Rule\nContent here.\n"
        orphan.write_text(content, encoding="utf-8")

        service.register_untracked("rules/readable")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.get_content("rules/readable")
        assert result["source"] == "user-owned"
        assert result["content"] == content

    def test_get_content_unknown_artifact_raises(self, service):
        """get_content for a completely unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_content("rules/totally-unknown")

    def test_save_override_custom_artifact(self, detached_service, detached_project_dir):
        """save_override works for registered custom files."""
        orphan = detached_project_dir / ".claude" / "rules" / "editable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Original\n", encoding="utf-8")

        detached_service.register_untracked("rules/editable")

        svc = ScaffoldGalleryService(detached_project_dir)
        new_content = "# Edited Custom Rule\nUpdated content.\n"
        result = svc.save_override("rules/editable", new_content)
        assert result["status"] == "saved"
        assert orphan.read_text(encoding="utf-8") == new_content

    def test_unoverride_custom_with_delete(self, detached_service, detached_project_dir):
        """Unclaiming a custom artifact with delete_file=True removes the file."""
        orphan = detached_project_dir / ".claude" / "rules" / "removable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Removable\n", encoding="utf-8")

        detached_service.register_untracked("rules/removable")

        svc = ScaffoldGalleryService(detached_project_dir)
        result = svc.unoverride("rules/removable", delete_file=True)
        assert result["status"] == "removed"
        assert result["deleted_file"] is True
        assert not orphan.exists()

        user_owned = _get_user_owned(detached_project_dir)
        assert "rules/removable" not in user_owned


# ===========================================================================
# Unoverride framework restore
# ===========================================================================


# ===========================================================================
# Create artifact
# ===========================================================================


class TestCreateArtifact:
    """Creating a custom artifact, on whichever surface will keep it.

    In profile mode that surface is the profile, not the project: the project's
    ``config.yml`` is derived output the next build regenerates from the
    profile, so a registration written there would be dropped and the file
    beside it pruned as unowned. The operator would watch the artifact appear
    and then quietly vanish at the next build.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def test_create_artifact_writes_the_profile_copy(self, service, project_dir):
        """The new artifact lands in the profile's convention directory."""
        result = service.create_artifact("rules", "my-rule")
        assert result["status"] == "created"
        assert result["created_in_profile"] is True
        assert result["output_path"] == ".claude/rules/my-rule.md"

        slot = self._profile_root(project_dir) / "rules" / "my-rule.md"
        assert slot.is_file()
        assert len(slot.read_text(encoding="utf-8")) > 0
        assert not (project_dir / ".claude" / "rules" / "my-rule.md").exists(), (
            "the project copy is the build's to make, from the profile"
        )

    def test_create_artifact_writes_no_project_ownership(self, service, project_dir):
        """Registration is the next build's convention scan, not the gallery's."""
        service.create_artifact("agents", "my-agent")

        assert "agents/my-agent" not in _get_user_owned(project_dir)
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        assert "agents/my-agent" not in manifest.get("user_owned", {})

    def test_a_created_artifact_is_owned_and_visible_immediately(self, service, project_dir):
        """It must not disappear from the gallery until someone rebuilds."""
        service.create_artifact("rules", "shift-handover", "# Shift handover\nCheck the orbit.\n")

        svc = ScaffoldGalleryService(project_dir)
        assert "rules/shift-handover" in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed["rules/shift-handover"]["status"] == "user-owned"
        assert svc.get_content("rules/shift-handover")["content"].startswith("# Shift handover")

    def test_a_created_artifact_can_be_edited_where_it_now_lives(self, service, project_dir):
        """Create → edit → save has to land on the profile copy, like a claim."""
        service.create_artifact("rules", "editable", "# First\n")

        svc = ScaffoldGalleryService(project_dir)
        svc.save_override("rules/editable", "# Second\n")

        slot = self._profile_root(project_dir) / "rules" / "editable.md"
        assert slot.read_text(encoding="utf-8") == "# Second\n"

    def test_create_refuses_when_the_project_already_carries_the_file(self, service, project_dir):
        """That is a claim — which moves the body — not a creation."""
        existing = project_dir / ".claude" / "rules" / "already-here.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# Written by hand\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="claim it instead"):
            service.create_artifact("rules", "already-here")
        assert existing.read_text(encoding="utf-8") == "# Written by hand\n"

    def test_create_refuses_in_a_degraded_project(self, degraded_project_dir):
        """No reachable profile means nowhere durable to author into."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.create_artifact("rules", "nowhere-to-put-this")
        assert not (degraded_project_dir / ".claude" / "rules" / "nowhere-to-put-this.md").exists()

    def test_create_still_registers_in_config_mode(self, detached_service, detached_project_dir):
        """A pre-profile project has no profile, so config.yml is still the place."""
        detached_service.create_artifact("agents", "legacy-agent")

        assert "agents/legacy-agent" in _get_user_owned(detached_project_dir)
        assert (detached_project_dir / ".claude" / "agents" / "legacy-agent.md").is_file()

    def test_create_artifact_invalid_category_raises(self, service):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            service.create_artifact("widgets", "bad-widget")

    def test_create_artifact_conflict_with_framework_raises(self, service):
        """Creating an artifact with a framework name raises ValueError."""
        with pytest.raises(ValueError, match="framework artifact"):
            service.create_artifact("rules", "safety")

    def test_create_artifact_duplicate_raises(self, service, project_dir):
        """Creating the same artifact twice raises FileExistsError."""
        service.create_artifact("rules", "dupe-test")
        svc = ScaffoldGalleryService(project_dir)
        with pytest.raises(FileExistsError, match="already exists"):
            svc.create_artifact("rules", "dupe-test")

    def test_create_artifact_hook_gets_py_extension(self, service, project_dir):
        """Hook artifacts are created with .py extension, not .md."""
        result = service.create_artifact("hooks", "my-hook")
        assert result["output_path"] == ".claude/hooks/my-hook.py"

        slot = self._profile_root(project_dir) / "hooks" / "my-hook.py"
        assert slot.is_file()
        assert not slot.with_suffix(".md").exists()

    def test_a_created_hook_is_owned_under_its_filename(self, service, project_dir):
        """A hook is known by its filename — that is what settings.json wires.

        Assembling the name as ``<category>/<name>`` instead would drop the
        ``.py``, and every later lookup would resolve it back to a ``.md`` path
        that was never written: the body would be stored against a file nothing
        reads, and releasing it would delete nothing.
        """
        result = service.create_artifact("hooks", "shift-check")

        assert result["canonical_name"] == "hooks/shift-check.py"
        svc = ScaffoldGalleryService(project_dir)
        assert svc._canonical_to_path("hooks/shift-check.py") == ".claude/hooks/shift-check.py"
        assert svc.get_content("hooks/shift-check.py")["content"]

    def test_create_artifact_hook_executable(self, service, project_dir):
        """Hook .py files get the execute permission bit."""
        import stat

        service.create_artifact("hooks", "exec-hook")
        slot = self._profile_root(project_dir) / "hooks" / "exec-hook.py"
        assert slot.stat().st_mode & stat.S_IXUSR

    def test_create_artifact_starter_content(self, service, project_dir):
        """Default content is category-appropriate starter content."""
        profile_root = self._profile_root(project_dir)

        service.create_artifact("rules", "starter-test")
        content = (profile_root / "rules" / "starter-test.md").read_text(encoding="utf-8")
        assert content.startswith("# Starter Test")

        svc = ScaffoldGalleryService(project_dir)
        svc.create_artifact("skills", "starter-skill")
        skill_content = (profile_root / "skills" / "starter-skill" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "name: starter-skill" in skill_content

    def test_a_created_skill_is_a_directory_with_a_skill_md(self, service, project_dir):
        """A skill is a directory; a flat ``skills/<name>.md`` is not loaded.

        The convention table says so — ``skills/`` is the one directory-shaped
        category the gallery can create — and Claude Code agrees: without a
        ``SKILL.md`` inside its own directory there is nothing for it to pick
        up. The old flat path produced a file that looked created and was
        never a skill.
        """
        result = service.create_artifact("skills", "orbit-check")

        assert result["canonical_name"] == "skills/orbit-check"
        assert result["output_path"] == ".claude/skills/orbit-check/SKILL.md"
        slot = self._profile_root(project_dir) / "skills" / "orbit-check"
        assert slot.is_dir()
        assert (slot / "SKILL.md").is_file()
        assert not slot.with_suffix(".md").exists()

    @pytest.mark.parametrize("bad", ["../escape", "../../etc/passwd"])
    def test_create_refuses_a_name_that_climbs_out(self, service, project_dir, bad):
        """The name is resolved, not concatenated, so it cannot escape.

        Assembling ``.claude/rules/<name>.md`` by hand let a name containing
        ``..`` address a file outside the project entirely.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        with pytest.raises((ScaffoldClaimError, ValueError)):
            service.create_artifact("rules", bad)
        assert not (project_dir.parent / "escape.md").exists()

    def test_a_nested_name_is_legitimate(self, service, project_dir):
        """Markdown categories nest — ``commands/osprey/scan`` is a real name.

        Refusing every name with a slash would have been the easy way to stop
        the traversal above, and it would have broken nesting the convention
        table explicitly allows.
        """
        result = service.create_artifact("commands", "osprey/handover")

        assert result["canonical_name"] == "commands/osprey/handover"
        slot = self._profile_root(project_dir) / "commands" / "osprey" / "handover.md"
        assert slot.is_file()

    def test_register_untracked_writes_manifest(self, detached_service, detached_project_dir):
        """Regression: config-mode register_untracked also writes a manifest entry."""
        project_dir = detached_project_dir
        orphan = project_dir / ".claude" / "rules" / "manifest-reg.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Manifest Reg\n", encoding="utf-8")

        detached_service.register_untracked("rules/manifest-reg")

        manifest_path = project_dir / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("user_owned", {}).get("rules/manifest-reg")
        assert entry is not None
        assert "claimed_at" in entry


class TestUnoverrideFrameworkRestore:
    """Tests for unoverride restoring framework file content on disk.

    Config mode throughout: releasing is only the last word on ownership where
    nothing else supplies the artifact. Where a profile does, the next build
    registers it again — see :class:`TestProfileMode`.
    """

    def test_unoverride_restores_framework_file(self, detached_service, detached_project_dir):
        """Claim, customize, release → get_content returns framework content."""
        # Claim the artifact
        detached_service.scaffold_override(SAFE_ARTIFACT)

        # Customize it with user content
        custom_text = "# My custom safety rules\nUser wrote this.\n"
        detached_service.save_override(SAFE_ARTIFACT, custom_text)

        # Release to framework with delete_file=True (what the web UI sends)
        svc = ScaffoldGalleryService(detached_project_dir)
        svc.unoverride(SAFE_ARTIFACT, delete_file=True)

        # After release, get_content should return framework content, not custom
        svc2 = ScaffoldGalleryService(detached_project_dir)
        result = svc2.get_content(SAFE_ARTIFACT)
        assert result["source"] == "framework"
        assert result["content"] != custom_text

    def test_unoverride_restored_content_matches_render(
        self, detached_service, detached_project_dir
    ):
        """After release, disk file matches _render_framework() output exactly."""
        # Get expected framework content before any changes
        expected = detached_service.get_framework_content(SAFE_ARTIFACT)

        # Claim and customize
        detached_service.scaffold_override(SAFE_ARTIFACT)
        detached_service.save_override(SAFE_ARTIFACT, "# Totally different content\n")

        # Release to framework
        svc = ScaffoldGalleryService(detached_project_dir)
        result = svc.unoverride(SAFE_ARTIFACT, delete_file=True)
        assert result["restored_file"] is True

        # Verify disk matches rendered template exactly
        art = svc._registry.get(SAFE_ARTIFACT)
        disk_path = detached_project_dir / art.output_path
        assert disk_path.read_text(encoding="utf-8") == expected

    def test_unoverride_without_delete_file_preserves_disk(
        self, detached_service, detached_project_dir
    ):
        """delete_file=False does NOT touch the file (CLI deferred-regen contract)."""
        # Claim and customize
        detached_service.scaffold_override(SAFE_ARTIFACT)
        custom_text = "# My custom rules — should persist\n"
        detached_service.save_override(SAFE_ARTIFACT, custom_text)

        # Release WITHOUT delete_file (CLI semantics)
        svc = ScaffoldGalleryService(detached_project_dir)
        svc.unoverride(SAFE_ARTIFACT, delete_file=False)

        # File on disk should still be the user's customized version
        art = svc._registry.get(SAFE_ARTIFACT)
        disk_path = detached_project_dir / art.output_path
        assert disk_path.read_text(encoding="utf-8") == custom_text


# ===========================================================================
# Profile mode — the project names a profile this process can reach
# ===========================================================================


class TestProfileMode:
    """The gallery delegates to the CLI claim path and follows the artifact.

    A claim MOVES the artifact into the profile's convention directory, which
    is the source of truth the next build registers ownership from. That makes
    the profile copy — not the project copy the claim removed — the thing the
    operator is looking at from then on.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def test_claim_moves_artifact_into_the_profile(self, service, project_dir):
        """The artifact lands in the profile, and leaves the project tree."""
        result = service.scaffold_override(SAFE_ARTIFACT)

        assert result["moved_to_profile"] is True
        profile_copy = self._profile_root(project_dir) / "rules" / "safety.md"
        assert profile_copy.is_file()
        assert not (project_dir / result["output_path"]).exists()

    def test_claim_writes_no_project_ownership(self, service, project_dir):
        """Registration is the next build's job, not the gallery's.

        Writing config.yml here would be the second source of truth the profile
        model removes — and the next build would overwrite it anyway.
        """
        service.scaffold_override(SAFE_ARTIFACT)
        assert SAFE_ARTIFACT not in _get_user_owned(project_dir)

    def test_claimed_artifact_stays_visible_and_owned(self, service, project_dir):
        """A claim must not make the artifact disappear from the gallery.

        The project copy is gone until the next build, so a gallery that only
        looked at the project tree would show the operator's own artifact
        vanishing the moment they claimed it.
        """
        service.scaffold_override(SAFE_ARTIFACT)

        svc = ScaffoldGalleryService(project_dir)
        assert SAFE_ARTIFACT in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed[SAFE_ARTIFACT]["status"] == "user-owned"

    def test_save_writes_the_profile_copy(self, service, project_dir):
        """An edit has to land where the next build reads from.

        Writing the project copy would look identical in the UI and be gone at
        the next `build --force` — the same silent loss the claim path exists
        to prevent, one layer up.
        """
        service.scaffold_override(SAFE_ARTIFACT)

        svc = ScaffoldGalleryService(project_dir)
        edited = "# Facility safety rules\nTwo-person rule for magnet writes.\n"
        svc.save_override(SAFE_ARTIFACT, edited)

        profile_copy = self._profile_root(project_dir) / "rules" / "safety.md"
        assert profile_copy.read_text(encoding="utf-8") == edited
        assert ScaffoldGalleryService(project_dir).get_content(SAFE_ARTIFACT)["content"] == edited

    def test_register_untracked_moves_the_file_too(self, service, project_dir):
        """A hand-written file's durable home is the profile, same as a claim."""
        orphan = project_dir / ".claude" / "rules" / "hand-written.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Hand written\n", encoding="utf-8")

        result = service.register_untracked("rules/hand-written")

        assert result["status"] == "registered"
        assert (self._profile_root(project_dir) / "rules" / "hand-written.md").is_file()
        assert not orphan.exists()

    def test_refusal_reaches_the_caller_with_its_message(self, service, monkeypatch):
        """A CLI refusal is surfaced verbatim, not paraphrased or swallowed.

        Every refusal names what was refused and what to do instead; that text
        is the whole value of the refusal to an operator. The route family
        turns it into a 409 with the message intact — re-wrapping it here would
        throw away the only part worth reading.
        """
        from osprey.cli import scaffold_cmd

        message = "Nothing was moved.\n  Rebuild it from a profile, then claim again."

        def _refuse(project_dir, name):
            raise scaffold_cmd.ScaffoldClaimError(message)

        monkeypatch.setattr(scaffold_cmd, "claim_into_profile", _refuse)

        with pytest.raises(scaffold_cmd.ScaffoldClaimError) as raised:
            service.scaffold_override(SAFE_ARTIFACT)
        assert str(raised.value) == message


# ===========================================================================
# Volume mode — deployed container, per-user claude-config volume
# ===========================================================================


class TestVolumeMode:
    """Claims in a deployed container land on the volume, and survive it."""

    def test_claim_writes_the_store_not_config(self, container_project, volume_dir):
        """config.yml is image-baked: writing it would vanish with the container."""
        svc = ScaffoldGalleryService(container_project)
        before = _get_user_owned(container_project)

        svc.scaffold_override(SAFE_ARTIFACT)

        assert _get_user_owned(container_project) == before
        assert _store_index(volume_dir)["artifacts"][SAFE_ARTIFACT]["state"] == "claimed"

    def test_claim_survives_a_container_recreation(self, container_project, volume_dir, tmp_path):
        """The test this feature exists for.

        A recreated container gets a pristine tree from the image and the same
        volume. The operator's claim — and the content it was a claim of — has
        to still be there, or the claim silently never happened.
        """
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        edited = "# Safety rules\nOperator wrote this in the container.\n"
        svc.save_override(SAFE_ARTIFACT, edited)

        # The recreated container: fresh tree, same volume, app start.
        assert restore_scaffold_bodies(pristine) == [SAFE_ARTIFACT]
        recreated = ScaffoldGalleryService(pristine)

        assert SAFE_ARTIFACT in recreated._user_owned
        assert recreated.get_content(SAFE_ARTIFACT)["content"] == edited
        art = recreated._registry.get(SAFE_ARTIFACT)
        assert (pristine / art.output_path).read_text(encoding="utf-8") == edited, (
            "the running agent reads the project tree, so the body must be restored there too"
        )

    def test_created_artifact_survives_a_container_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """A file that exists only in the container is the easiest thing to lose."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.create_artifact("rules", "shift-handover", "# Shift handover\nCheck the orbit.\n")

        restore_scaffold_bodies(pristine)
        recreated = ScaffoldGalleryService(pristine)
        assert "rules/shift-handover" in recreated._user_owned
        assert (pristine / ".claude" / "rules" / "shift-handover.md").is_file()

    def test_release_outranks_build_derived_ownership(self, container_project, volume_dir):
        """Merge precedence: the store's record wins over config.yml.

        config.yml is what the profile owned when the image was built. The
        store is what this operator did afterwards — later, and by hand. If the
        build-derived entry won instead, unclaim would have nowhere to be
        recorded and the button would report success and change nothing.
        """
        baked = _get_user_owned(container_project)
        assert baked, "the preset build is expected to derive some ownership"
        name = baked[0]

        ScaffoldGalleryService(container_project).unoverride(name)

        svc = ScaffoldGalleryService(container_project)
        assert name not in svc._user_owned
        assert _store_index(volume_dir)["artifacts"][name]["state"] == "released"

    def test_release_survives_a_container_recreation(self, container_project, volume_dir, tmp_path):
        """A release is durable for the same reason a claim is."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        name = _get_user_owned(container_project)[0]

        ScaffoldGalleryService(container_project).unoverride(name)

        assert name not in ScaffoldGalleryService(pristine)._user_owned

    def test_claim_adds_to_build_derived_ownership(self, container_project):
        """The gallery reads the union, not one surface or the other."""
        baked = set(_get_user_owned(container_project))

        ScaffoldGalleryService(container_project).scaffold_override(SAFE_ARTIFACT)

        owned = set(ScaffoldGalleryService(container_project)._user_owned)
        assert baked <= owned
        assert SAFE_ARTIFACT in owned

    def test_corrupt_store_does_not_take_the_gallery_down(self, container_project, volume_dir):
        """The store is on a volume — truncation is a thing that happens.

        Build-derived ownership is still readable, so the gallery degrades to
        that rather than failing every request.
        """
        ScaffoldGalleryService(container_project).scaffold_override(SAFE_ARTIFACT)
        index = volume_dir / "osprey" / "scaffold" / "user_owned.json"
        index.write_text('{"artifacts": {"rules/safety": {"stat', encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        assert svc.list_artifacts()
        assert set(_get_user_owned(container_project)) <= set(svc._user_owned)

    def test_corrupt_store_is_set_aside_not_overwritten(self, container_project, volume_dir):
        """A later claim recovers the store, keeping the damaged file as evidence."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text("}{ not json", encoding="utf-8")

        ScaffoldGalleryService(container_project).scaffold_override(SAFE_ARTIFACT)

        assert (store_dir / "user_owned.json.corrupt").is_file()
        assert _store_index(volume_dir)["artifacts"][SAFE_ARTIFACT]["state"] == "claimed"

    def test_concurrent_claims_do_not_lose_records(self, container_project, volume_dir):
        """Requests are served from a thread pool; a claim must not be lost."""
        import threading

        names = [f"rules/concurrent-{i}" for i in range(8)]
        for name in names:
            path = container_project / ".claude" / "rules" / f"{name.split('/')[1]}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}\n", encoding="utf-8")

        def register(name: str) -> None:
            ScaffoldGalleryService(container_project).register_untracked(name)

        threads = [threading.Thread(target=register, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        recorded = _store_index(volume_dir)["artifacts"]
        assert set(names) <= set(recorded)

    def test_record_for_an_artifact_the_image_dropped_is_harmless(
        self, container_project, volume_dir
    ):
        """A claim outlives the image that carried it; the gallery must not care."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {
                        "rules/retired": {
                            "state": "claimed",
                            "output_path": ".claude/rules/retired.md",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        svc = ScaffoldGalleryService(container_project)
        listed = {a["name"] for a in svc.list_artifacts()}

        assert "rules/retired" in svc._user_owned
        assert "rules/retired" not in listed
        assert not (container_project / ".claude" / "rules" / "retired.md").exists()

    def test_local_edit_is_not_clobbered_by_the_durable_copy(self, container_project, volume_dir):
        """Rehydration restores an image-fresh file, never a newer local edit.

        Someone editing through the terminal rather than the gallery is doing
        the same work by another route; overwriting it on the next request
        would destroy it.
        """
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        svc.save_override(SAFE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(SAFE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        assert restore_scaffold_bodies(container_project) == []

        assert (container_project / art.output_path).read_text(
            encoding="utf-8"
        ) == edited_in_terminal


# ===========================================================================
# Degraded topology — a profile is named but cannot be reached
# ===========================================================================


class TestDegradedTopology:
    """No durable surface, so every write refuses — it does not fall back.

    Recording ownership in the project's own config.yml here would report
    success and then be erased by the next build, which regenerates config.yml
    from the profile. Refusing is the honest answer, and it is the same answer
    ``osprey scaffold claim`` gives.
    """

    def test_claim_refuses_with_the_cli_message(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError, match="profile"):
            svc.scaffold_override(SAFE_ARTIFACT)

    def test_claim_writes_nothing(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        before = _get_user_owned(degraded_project_dir)
        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override(SAFE_ARTIFACT)

        assert _get_user_owned(degraded_project_dir) == before

    def test_create_artifact_refuses(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError, match="cannot be\n *reached|cannot be reached"):
            svc.create_artifact("rules", "nowhere-to-put-this")

    def test_release_refuses(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        owned = _get_user_owned(degraded_project_dir)
        assert owned, "the preset build is expected to derive some ownership"

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.unoverride(owned[0])

    def test_the_gallery_still_opens(self, degraded_project_dir):
        """Reads must keep working — a refused write is not a broken page."""
        svc = ScaffoldGalleryService(degraded_project_dir)
        assert svc.list_artifacts()
        assert svc.get_content(SAFE_ARTIFACT)["content"]


# ===========================================================================
# Restore-path containment
# ===========================================================================


class TestRestoreContainment:
    """The store is data on a mounted volume, so its paths are not trusted.

    An entry's ``output_path`` decides where a restore writes. Left
    unconstrained it could name ``.env``, ``config.yml``, or a path climbing out
    of the project altogether, and the app would obligingly overwrite it at
    startup.
    """

    def _seed(self, volume_dir: Path, output_path: str) -> None:
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files").mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {
                        "rules/planted": {"state": "claimed", "output_path": output_path}
                    },
                }
            ),
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        "output_path",
        [
            "config.yml",
            ".env",
            "../escaped.md",
            "/etc/passwd",
            ".claude/../../escaped.md",
            # Inside the ownable tree, and still refused: the build generates
            # it, so a stored copy would shadow the regenerated one forever.
            ".claude/hooks/hook_config.json",
            ".claude/settings.json",
        ],
    )
    def test_restore_refuses_paths_outside_the_ownable_tree(
        self, container_project, volume_dir, output_path
    ):
        self._seed(volume_dir, output_path)

        assert restore_scaffold_bodies(container_project) == []

    def test_a_planted_path_does_not_overwrite_config(self, container_project, volume_dir):
        self._seed(volume_dir, "config.yml")
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files").mkdir(parents=True, exist_ok=True)
        before = (container_project / "config.yml").read_text(encoding="utf-8")

        restore_scaffold_bodies(container_project)

        assert (container_project / "config.yml").read_text(encoding="utf-8") == before


# ===========================================================================
# Artifact shapes the naming rules trip over
# ===========================================================================


class TestArtifactShapes:
    """Two shapes whose paths do not follow the ``<name>.md`` assumption."""

    def test_hook_paths_keep_their_py_suffix(self, container_project, volume_dir):
        """A hook is a ``.py`` script and is owned under a name that says so.

        Appending ``.md`` to it would record the claimed body against a path
        that does not exist, so the body would be stored and never read back.
        """
        svc = ScaffoldGalleryService(container_project)
        assert svc._canonical_to_path("hooks/osprey_cf_feedback_capture.py") == (
            ".claude/hooks/osprey_cf_feedback_capture.py"
        )
        assert svc._canonical_to_path("rules/safety") == ".claude/rules/safety.md"

    def test_a_claimed_hook_body_round_trips(self, container_project, volume_dir):
        """The body of a hook claim is retrievable, not written into the void."""
        hook = container_project / ".claude" / "hooks" / "shift-check.py"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        svc._record_claim("hooks/shift-check.py", hook.read_text(encoding="utf-8"))

        stored = _store_index(volume_dir)["artifacts"]["hooks/shift-check.py"]
        assert stored["output_path"] == ".claude/hooks/shift-check.py"
        body = volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "hooks" / "shift-check.py"
        assert body.read_text(encoding="utf-8") == hook.read_text(encoding="utf-8")

    def test_directory_artifacts_are_not_read_as_text(self, service, project_dir):
        """A skill is a directory. Reading its profile slot as text would raise.

        The gallery edits single files; a directory-shaped artifact has no body
        to open, and asking for one must not crash the page.
        """
        skills = [a["name"] for a in service.list_artifacts() if a["name"].startswith("skills/")]
        assert skills, "the control-assistant preset is expected to ship a skill"

        for name in skills:
            assert service._profile_file(name) is None


class TestVolumeSaveDurability:
    """Editing an artifact you already own is as losable as claiming one."""

    def test_editing_a_build_derived_artifact_survives_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """Ownership from the image's config.yml carries no store record.

        A save that wrote only the body would leave it on the volume with
        nothing pointing at it, and the restore — which walks the records —
        would skip it. The operator's edit would sit intact on a disk nobody
        reads and come back as the framework's original.
        """
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        name = _get_user_owned(container_project)[0]

        svc = ScaffoldGalleryService(container_project)
        edited = f"# {name}\nEdited in the container, never claimed separately.\n"
        result = svc.save_override(name, edited)

        assert restore_scaffold_bodies(pristine) == [name]
        assert (pristine / result["path"]).read_text(encoding="utf-8") == edited


# ===========================================================================
# What may be owned at all
# ===========================================================================


class TestGeneratedPathsAreNeverOwnable:
    """A generated file must not be frozen by a claim, in any mode.

    ``osprey scaffold claim`` refuses these already, so the two modes that hand
    it the claim inherit the rule. The two that record ownership themselves —
    a container writing the volume, a pre-profile project writing config.yml —
    have to apply it themselves or the gallery is a way around it.
    """

    #: The catalog name and the project path of a generated artifact that lives
    #: inside an otherwise claimable channel. This one is the write-safety
    #: layer's runtime configuration: ``osprey_writes_check.py`` reads its
    #: ``write_tools``, and the render derives that from the resolved config.
    HOOK_CONFIG = ("hooks/hook-config", ".claude/hooks/hook_config.json")

    def test_claim_of_a_generated_hook_config_is_refused_on_the_volume(
        self, container_project, volume_dir
    ):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        name, path = self.HOOK_CONFIG
        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.scaffold_override(name)

        assert _store_index(volume_dir) == {}, "a refused claim records nothing"
        assert not (volume_dir / "osprey" / "scaffold" / "files" / path).exists()

    def test_claim_of_settings_json_is_refused_on_the_volume(self, container_project, volume_dir):
        """settings.json is the permissions channel — the profile's config keys own it."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.scaffold_override("settings-json")
        assert _store_index(volume_dir) == {}

    def test_claim_of_a_generated_path_is_refused_in_config_mode(self, detached_service):
        """The legacy surface is durable, which makes freezing it worse, not better."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        name, _ = self.HOOK_CONFIG
        before = _get_user_owned(detached_service.project_dir)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            detached_service.scaffold_override(name)
        assert _get_user_owned(detached_service.project_dir) == before

    def test_the_other_spelling_is_refused_too(self, container_project, volume_dir):
        """Ownership follows the path, so naming it by filename changes nothing.

        ``register_untracked`` takes the name the file has on disk rather than
        the catalog canonical, and it reaches the store by a different route —
        a guard on one spelling only would be no guard at all.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.register_untracked("hooks/hook_config.json")
        assert _store_index(volume_dir) == {}

    def test_a_regenerated_hook_config_is_never_restored_over(
        self, container_project, volume_dir, tmp_path
    ):
        """The failure the refusal exists to prevent, asserted end to end.

        A frozen copy on the volume would be written back over the freshly
        generated file at every container start, so a rebuilt image would ship
        a new write-tool set and the container would go on running the stale
        one — silently, and for as long as the volume lived.
        """
        _name, path = self.HOOK_CONFIG
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        generated = (pristine / path).read_text(encoding="utf-8")

        # Plant the claim directly: the gallery refuses to create it, and the
        # point here is that the restore refuses to act on one either way.
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files" / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
        (store_dir / "files" / path).write_text('{"write_tools": []}', encoding="utf-8")
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {"hooks/hook-config": {"state": "claimed", "output_path": path}},
                }
            ),
            encoding="utf-8",
        )

        assert restore_scaffold_bodies(pristine) == []
        assert (pristine / path).read_text(encoding="utf-8") == generated

    def test_an_artifact_the_store_cannot_hold_is_refused(self, container_project, volume_dir):
        """A record with no body would claim the framework's own text.

        CLAUDE.md lives outside the ownable tree, so the store has nowhere to
        keep it. Recording ownership anyway would report success, keep no copy,
        and hand the operator the image's original back after the next
        recreation while still calling it theirs.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override("claude-md")

        assert _store_index(volume_dir) == {}
        assert "claude-md" not in ScaffoldGalleryService(container_project)._user_owned

    def test_a_refused_claim_leaves_no_file_behind(self, container_project):
        """The claim path renders missing artifacts to disk — a refusal must not."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        _name, path = self.HOOK_CONFIG
        (container_project / path).unlink()

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override("hooks/hook-config")
        assert not (container_project / path).exists()

    def _poison_index(self, volume_dir: Path, name: str, output_path: str | None) -> None:
        """Plant an ownership record the way an older OSPREY would have."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {"state": "claimed"}
        if output_path is not None:
            entry["output_path"] = output_path
        (store_dir / "user_owned.json").write_text(
            json.dumps({"version": 1, "artifacts": {name: entry}}), encoding="utf-8"
        )

    @pytest.mark.parametrize(
        ("name", "output_path"),
        [
            ("hooks/hook-config", ".claude/hooks/hook_config.json"),
            ("hooks/hook_config.json", ".claude/hooks/hook_config.json"),
            # Written before the guard existed, so it carries no path at all —
            # the name is the only thing left to recognize it by.
            ("hooks/hook-config", None),
            ("settings-json", ".claude/settings.json"),
        ],
    )
    def test_a_planted_record_is_not_ownership(
        self, container_project, volume_dir, name, output_path
    ):
        """Containment covers bodies; a record is not a body.

        The store outlives the image, so a record can predate the guard that
        would refuse it today. Left to flow through, it tells the operator they
        own the write-safety layer's configuration — and the gallery would show
        them the framework's own text as theirs.
        """
        self._poison_index(volume_dir, name, output_path)

        svc = ScaffoldGalleryService(container_project)

        assert name not in svc._user_owned
        assert all(a["name"] != name for a in svc.list_artifacts() if a["status"] == "user-owned")

    def test_a_planted_record_does_not_survive_a_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """Not listed, not restored, and the generated file is untouched."""
        path = ".claude/hooks/hook_config.json"
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        generated = (pristine / path).read_text(encoding="utf-8")

        self._poison_index(volume_dir, "hooks/hook-config", path)
        files = volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "hooks"
        files.mkdir(parents=True, exist_ok=True)
        (files / "hook_config.json").write_text('{"write_tools": []}', encoding="utf-8")

        assert restore_scaffold_bodies(pristine) == []
        assert (pristine / path).read_text(encoding="utf-8") == generated

    def test_saving_over_a_planted_record_is_refused_in_words(self, container_project, volume_dir):
        """The refusal names the file and its channel, like every other one.

        Without this the save reaches the store, fails as a write it cannot
        make durable, and the operator gets a bare server error where every
        comparable refusal explains itself.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        self._poison_index(volume_dir, "hooks/hook-config", ".claude/hooks/hook_config.json")
        svc = ScaffoldGalleryService(container_project)

        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.save_override("hooks/hook-config", '{"write_tools": []}')

    def test_a_generated_path_reaching_save_is_refused_by_name(self, container_project):
        """The guard on save itself, independent of ownership filtering."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        svc._user_owned = [*svc._user_owned, "hooks/hook-config"]

        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.save_override("hooks/hook-config", '{"write_tools": []}')

    def test_ordinary_artifacts_are_unaffected(self, container_project, volume_dir):
        """The guard names generated files, not whole channels.

        ``hooks/`` holds authored hooks as well as the generated config, and
        refusing the channel to protect one file in it would be the wrong
        trade — the operator would lose every hook to save one.
        """
        hook = container_project / ".claude" / "hooks" / "shift-check.py"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        svc.register_untracked("hooks/shift-check.py")
        svc.scaffold_override(SAFE_ARTIFACT)

        recorded = _store_index(volume_dir)["artifacts"]
        assert recorded["hooks/shift-check.py"]["state"] == "claimed"
        assert recorded[SAFE_ARTIFACT]["state"] == "claimed"


class TestBodilessClaimsAreRefused:
    """A record is only written once the body it is a claim of is safely down."""

    def test_a_failed_body_write_records_nothing(self, container_project, volume_dir, monkeypatch):
        """A full volume must not leave a claim on the framework's own text."""
        from osprey.interfaces.web_terminal import ownership as ownership_mod

        monkeypatch.setattr(
            ownership_mod.OwnershipStore, "write_content", lambda self, path, content: False
        )

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ownership_mod.OwnershipStoreError):
            svc.scaffold_override(SAFE_ARTIFACT)

        assert SAFE_ARTIFACT not in _store_index(volume_dir).get("artifacts", {})

    def test_a_body_is_replaced_atomically(self, container_project, volume_dir):
        """No reader ever sees half a body, so a restore cannot install one."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        store = svc._store

        store.write_content(".claude/rules/safety.md", "# One\n")
        store.write_content(".claude/rules/safety.md", "# Two, longer than the first\n")

        assert store.read_content(".claude/rules/safety.md") == "# Two, longer than the first\n"
        leftovers = list(
            (volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "rules").glob(".*tmp")
        )
        assert leftovers == [], f"atomic write left staging files behind: {leftovers}"


class TestTreeAndStoreDisagree:
    """Which copy the gallery shows when the two surfaces differ.

    ``rehydrate`` already decides this for writes — it leaves a tree copy alone
    unless it is provably still the image's. The read path has to answer the
    same way, or an edit it declined to overwrite is one the operator cannot
    see and will destroy with their next save.
    """

    def test_an_edit_made_outside_the_gallery_is_what_the_gallery_shows(
        self, container_project, volume_dir
    ):
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        svc.save_override(SAFE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(SAFE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        assert restore_scaffold_bodies(container_project) == []
        shown = ScaffoldGalleryService(container_project).get_content(SAFE_ARTIFACT)
        assert shown["content"] == edited_in_terminal, (
            "the gallery must show the copy the agent is reading, not the older durable one"
        )

    def test_saving_what_the_gallery_showed_does_not_revert_that_edit(
        self, container_project, volume_dir
    ):
        """The consequence of getting the read wrong, asserted directly."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        svc.save_override(SAFE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(SAFE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        # Open the editor, change nothing, save — the round trip a UI makes easy.
        reopened = ScaffoldGalleryService(container_project)
        reopened.save_override(SAFE_ARTIFACT, reopened.get_content(SAFE_ARTIFACT)["content"])

        assert (container_project / art.output_path).read_text(
            encoding="utf-8"
        ) == edited_in_terminal

    def test_an_untouched_image_copy_still_loses_to_the_durable_one(
        self, container_project, volume_dir, tmp_path
    ):
        """The recreation case must keep working: image-fresh tree, older claim."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)
        edited = "# The operator's version\n"
        svc.save_override(SAFE_ARTIFACT, edited)

        # Read before the restore runs: the tree is still the image's.
        assert ScaffoldGalleryService(pristine).get_content(SAFE_ARTIFACT)["content"] == edited


class TestDirectoryArtifactsHaveNoBody:
    """A skill or a service is a directory; saving text to one is a bad request."""

    def test_saving_a_profile_owned_directory_is_refused(self, project_dir):
        """Reachable by name even though the gallery does not list it.

        A profile-held skill is in ``_user_owned`` — that is what makes it show
        as owned — so ``save_override`` accepts the name and used to try to
        write text onto the directory itself, which is an unhandled OSError
        rather than an answer.
        """
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        profile_root = Path(manifest["build_args"]["profile_path_abs"]).parent
        skill = profile_root / "skills" / "orbit-check"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(
            "---\nname: orbit-check\ndescription: Check the orbit\n---\n", encoding="utf-8"
        )

        svc = ScaffoldGalleryService(project_dir)
        assert "skills/orbit-check" in svc._user_owned, (
            "a profile-held skill is owned, which is exactly why the name reaches save"
        )
        with pytest.raises(ValueError, match="whole directory"):
            svc.save_override("skills/orbit-check", "# not a directory\n")


class TestUnclaimIsHonestAboutTheProfile:
    """Releasing an artifact the profile supplies does not release it.

    Ownership of such an artifact is derived: the build's convention scan
    re-registers it from the profile every time. Dropping the project's record
    changes nothing that survives the next build, so reporting "removed" is
    simply false — and it is false in the direction that matters, because the
    operator walks away believing they gave something up.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def _claimed(self, project_dir: Path) -> str:
        """Claim SAFE_ARTIFACT into the profile and return its name."""
        ScaffoldGalleryService(project_dir).scaffold_override(SAFE_ARTIFACT)
        return SAFE_ARTIFACT

    def test_unclaim_says_the_profile_still_supplies_it(self, project_dir):
        name = self._claimed(project_dir)

        outcome = ScaffoldGalleryService(project_dir).unoverride(name)

        assert outcome["status"] == "still-supplied-by-profile"
        assert "still supplies it" in outcome["message"]
        assert "Remove it from the profile" in outcome["message"]

    def test_unclaim_changes_nothing(self, project_dir):
        """Not the profile copy, not the tree, not config.yml."""
        name = self._claimed(project_dir)
        slot = self._profile_root(project_dir) / "rules" / "safety.md"
        before = slot.read_text(encoding="utf-8")
        config_before = _get_user_owned(project_dir)

        ScaffoldGalleryService(project_dir).unoverride(name, delete_file=True)

        assert slot.read_text(encoding="utf-8") == before
        assert _get_user_owned(project_dir) == config_before

    def test_the_artifact_is_still_owned_afterwards(self, project_dir):
        """Honesty in the other direction: the gallery must not hide it either."""
        name = self._claimed(project_dir)

        ScaffoldGalleryService(project_dir).unoverride(name)

        svc = ScaffoldGalleryService(project_dir)
        assert name in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed[name]["status"] == "user-owned"

    def test_the_message_is_the_cli_s_own_words(self, project_dir):
        """One sentence, two surfaces — so they cannot drift apart."""
        from osprey.cli.scaffold_cmd import still_supplied_by_profile_message

        name = self._claimed(project_dir)
        slot = self._profile_root(project_dir) / "rules" / "safety.md"

        outcome = ScaffoldGalleryService(project_dir).unoverride(name)

        assert outcome["message"] == still_supplied_by_profile_message(str(slot))

    def test_an_artifact_the_profile_does_not_supply_still_releases(
        self, container_project, volume_dir
    ):
        """The honest path must not swallow the ordinary one."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(SAFE_ARTIFACT)

        outcome = ScaffoldGalleryService(container_project).unoverride(SAFE_ARTIFACT)

        assert outcome["status"] == "removed"
        assert SAFE_ARTIFACT not in ScaffoldGalleryService(container_project)._user_owned


# ===========================================================================
# Route-level refusals
# ===========================================================================


class TestRouteRefusals:
    """A refused write has to reach the operator as a reason, not a 500.

    The service raises in its own vocabulary; the route family decides what the
    browser sees. An exception the route does not name becomes a bare 500 with
    the message stripped, which is the one outcome that helps nobody.
    """

    def _client(self, project_dir: Path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(project_dir)
        return TestClient(app)

    def test_a_store_that_will_not_take_the_write_is_a_409(
        self, container_project, volume_dir, monkeypatch
    ):
        """A full or read-only volume, reported rather than swallowed."""
        from osprey.interfaces.web_terminal import ownership as ownership_mod

        ScaffoldGalleryService(container_project).scaffold_override(SAFE_ARTIFACT)
        monkeypatch.setattr(
            ownership_mod.OwnershipStore, "write_content", lambda self, path, content: False
        )

        response = self._client(container_project).put(
            f"/api/scaffold/{SAFE_ARTIFACT}/override", json={"content": "# edited\n"}
        )

        assert response.status_code == 409, response.text
        assert "ownership store" in response.json()["detail"]

    def test_claiming_a_generated_file_is_a_409_with_the_reason(self, container_project):
        response = self._client(container_project).post("/api/scaffold/hooks/hook-config/claim")

        assert response.status_code == 409, response.text
        assert "generated, not authored" in response.json()["detail"]

    def test_unclaiming_a_profile_held_artifact_is_a_409_not_a_success(self, project_dir):
        """The gallery's error banner is where the operator would see 'done'."""
        ScaffoldGalleryService(project_dir).scaffold_override(SAFE_ARTIFACT)

        response = self._client(project_dir).delete(
            f"/api/scaffold/{SAFE_ARTIFACT}/override?delete_file=true"
        )

        assert response.status_code == 409, response.text
        assert "still supplies it" in response.json()["detail"]

    def test_creating_a_traversing_name_is_refused(self, container_project):
        response = self._client(container_project).post(
            "/api/scaffold/create", json={"category": "rules", "name": "../../evil"}
        )

        assert response.status_code in (400, 409), response.text
        assert not (container_project.parent / "evil.md").exists()
