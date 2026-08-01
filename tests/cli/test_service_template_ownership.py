"""Tests for the unified services/ ownership regime.

Service compose templates under ``<project>/services/`` are catalog-managed
directory artifacts: refreshed from the framework by every build unless the
project claims them via ``osprey scaffold claim services/<name>`` (which
records ``services/<name>`` in config.yml's ``scaffold.user_owned``, exactly
like the Claude Code artifacts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

_SERVICE_ARTIFACTS = [
    "services/postgresql",
    "services/openobserve",
    "services/event_dispatcher",
    "services/dispatch_worker",
    "services/bluesky",
    "services/bluesky_panels",
    "services/virtual_accelerator",
]


class TestCatalogServiceEntries:
    """The catalog registers every packaged service template as a directory artifact."""

    @pytest.mark.parametrize("name", _SERVICE_ARTIFACTS)
    def test_service_artifact_registered(self, name):
        art = BuildArtifactCatalog.default().get(name)
        assert art is not None, f"Missing catalog entry: {name}"
        assert art.is_directory
        assert art.template_root == "services"
        assert art.output_path == name

    def test_no_unregistered_service_templates(self):
        """Every packaged service template directory has a catalog entry."""
        services_root = (
            Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "services"
        )
        registered = {
            a.template_path
            for a in BuildArtifactCatalog.default().all_artifacts()
            if a.template_root == "services"
        }
        on_disk = {d.name for d in services_root.iterdir() if d.is_dir()}
        assert on_disk == registered

    def test_claude_artifacts_still_default_root(self):
        art = BuildArtifactCatalog.default().get("claude-md")
        assert art.template_root == "claude_code"
        assert not art.is_directory


class TestBuildRefreshHonorsClaims:
    """_copy_service_templates refreshes unclaimed services, skips claimed ones."""

    def _write_config(self, project_path: Path, config: dict) -> None:
        (project_path / "config.yml").write_text(yaml.dump(config), encoding="utf-8")

    def _project(self, tmp_path: Path, *, user_owned: list[str] | None = None) -> Path:
        project_path = tmp_path / "project"
        project_path.mkdir()
        config = {
            "deployed_services": ["postgresql"],
            "services": {"postgresql": {"path": "./services/postgresql"}},
        }
        if user_owned:
            config["scaffold"] = {"user_owned": user_owned}
        self._write_config(project_path, config)
        return project_path

    def test_unclaimed_service_is_refreshed(self, tmp_path: Path) -> None:
        from osprey.cli.build_cmd import _copy_service_templates

        project_path = self._project(tmp_path)
        stale = project_path / "services" / "postgresql" / "docker-compose.yml.j2"
        stale.parent.mkdir(parents=True)
        stale.write_text("# locally edited\n", encoding="utf-8")

        count = _copy_service_templates(project_path)

        assert count == 1
        assert "locally edited" not in stale.read_text(encoding="utf-8")

    def test_claimed_service_is_left_untouched(self, tmp_path: Path) -> None:
        from osprey.cli.build_cmd import _copy_service_templates

        project_path = self._project(tmp_path, user_owned=["services/postgresql"])
        claimed = project_path / "services" / "postgresql" / "docker-compose.yml.j2"
        claimed.parent.mkdir(parents=True)
        claimed.write_text("# locally edited\n", encoding="utf-8")

        count = _copy_service_templates(project_path)

        assert count == 0
        assert claimed.read_text(encoding="utf-8") == "# locally edited\n"

    def test_claim_of_one_service_does_not_shield_another(self, tmp_path: Path) -> None:
        from osprey.cli.build_cmd import _copy_service_templates

        project_path = self._project(tmp_path, user_owned=["services/openobserve"])
        edited = project_path / "services" / "postgresql" / "docker-compose.yml.j2"
        edited.parent.mkdir(parents=True)
        edited.write_text("# locally edited\n", encoding="utf-8")

        count = _copy_service_templates(project_path)

        assert count == 1
        assert "locally edited" not in edited.read_text(encoding="utf-8")


class TestInjectorsHonorClaims:
    """The feature injectors (_inject_bluesky et al.) skip claimed services."""

    def test_inject_bluesky_skips_claimed_template(self, tmp_path: Path) -> None:
        from osprey.cli.build_injectors import _inject_bluesky
        from osprey.cli.build_profile import BlueskyConfig

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text(
            yaml.dump({"scaffold": {"user_owned": ["services/bluesky"]}}),
            encoding="utf-8",
        )
        claimed = project_path / "services" / "bluesky" / "docker-compose.yml.j2"
        claimed.parent.mkdir(parents=True)
        claimed.write_text("# locally edited\n", encoding="utf-8")

        _inject_bluesky(BlueskyConfig(), project_path)

        # Template untouched; config registration still happens.
        assert claimed.read_text(encoding="utf-8") == "# locally edited\n"
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert "bluesky" in config["services"]
        assert "bluesky" in config["deployed_services"]


class TestProfileServiceTemplateResolution:
    """_inject_profile_services resolves ``osprey.<name>`` to the bundled template.

    A profile may declare a service whose template is the framework's own
    (``template: osprey.openobserve``) rather than a directory shipped in the
    profile — the same form ``_copy_service_templates`` and
    ``BuildProfile.validate`` already accept.
    """

    def _project(self, tmp_path: Path) -> Path:
        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("project_name: test\n", encoding="utf-8")
        return project_path

    def test_bundled_prefix_resolves_to_package_template(self, tmp_path: Path) -> None:
        from osprey.cli.build_injectors import _inject_profile_services
        from osprey.cli.build_profile import ServiceDef

        project_path = self._project(tmp_path)
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        count = _inject_profile_services(
            profile_dir,
            project_path,
            {"openobserve": ServiceDef(template="osprey.openobserve")},
        )

        assert count == 1
        compose = project_path / "services" / "openobserve" / "docker-compose.yml.j2"
        assert compose.is_file(), "bundled template was not copied into the project"
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert config["services"]["openobserve"]["path"] == "./services/openobserve"
        assert "openobserve" in config["deployed_services"]

    def test_profile_relative_template_still_resolves(self, tmp_path: Path) -> None:
        from osprey.cli.build_injectors import _inject_profile_services
        from osprey.cli.build_profile import ServiceDef

        project_path = self._project(tmp_path)
        profile_dir = tmp_path / "profile"
        (profile_dir / "svc" / "typesense").mkdir(parents=True)
        (profile_dir / "svc" / "typesense" / "docker-compose.yml.j2").write_text(
            "# facility template\n", encoding="utf-8"
        )

        count = _inject_profile_services(
            profile_dir,
            project_path,
            {"typesense": ServiceDef(template="svc/typesense", config={"port": 8108})},
        )

        assert count == 1
        copied = project_path / "services" / "typesense" / "docker-compose.yml.j2"
        assert copied.read_text(encoding="utf-8") == "# facility template\n"
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert config["services"]["typesense"]["port"] == 8108

    def test_missing_bundled_template_warns_and_skips(self, tmp_path: Path) -> None:
        """An unknown ``osprey.<name>`` is skipped, not raised — as in _copy_service_templates."""
        from osprey.cli.build_injectors import _inject_profile_services
        from osprey.cli.build_profile import ServiceDef

        project_path = self._project(tmp_path)
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        count = _inject_profile_services(
            profile_dir,
            project_path,
            {"nope": ServiceDef(template="osprey.no_such_service")},
        )

        assert count == 0
        assert not (project_path / "services" / "nope").exists()
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert "nope" not in (config.get("services") or {})
        assert "nope" not in (config.get("deployed_services") or [])

    def test_claimed_bundled_service_is_left_untouched(self, tmp_path: Path) -> None:
        from osprey.cli.build_injectors import _inject_profile_services
        from osprey.cli.build_profile import ServiceDef

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text(
            yaml.dump({"scaffold": {"user_owned": ["services/openobserve"]}}),
            encoding="utf-8",
        )
        claimed = project_path / "services" / "openobserve" / "docker-compose.yml.j2"
        claimed.parent.mkdir(parents=True)
        claimed.write_text("# locally edited\n", encoding="utf-8")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        count = _inject_profile_services(
            profile_dir,
            project_path,
            {"openobserve": ServiceDef(template="osprey.openobserve")},
        )

        # Template untouched; config registration still happens.
        assert count == 1
        assert claimed.read_text(encoding="utf-8") == "# locally edited\n"
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert "openobserve" in config["services"]
        assert "openobserve" in config["deployed_services"]


class TestScaffoldCliDirectoryArtifacts:
    """scaffold claim/diff/unclaim work on directory (service) artifacts."""

    def _project(self, tmp_path: Path) -> Path:
        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("project_name: test\n", encoding="utf-8")
        return project_path

    def test_claim_copies_missing_directory_and_records_ownership(self, tmp_path: Path):
        from osprey.cli.scaffold_cmd import claim

        project_path = self._project(tmp_path)
        result = CliRunner().invoke(claim, ["services/postgresql", "-p", str(project_path)])

        assert result.exit_code == 0, result.output
        assert (project_path / "services" / "postgresql" / "docker-compose.yml.j2").exists()
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert "services/postgresql" in config["scaffold"]["user_owned"]

    def test_claim_existing_directory_marks_ownership_only(self, tmp_path: Path):
        from osprey.cli.scaffold_cmd import claim

        project_path = self._project(tmp_path)
        existing = project_path / "services" / "postgresql" / "docker-compose.yml.j2"
        existing.parent.mkdir(parents=True)
        existing.write_text("# mine\n", encoding="utf-8")

        result = CliRunner().invoke(claim, ["services/postgresql", "-p", str(project_path)])

        assert result.exit_code == 0, result.output
        assert existing.read_text(encoding="utf-8") == "# mine\n"
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        assert "services/postgresql" in config["scaffold"]["user_owned"]

    def test_diff_reports_local_edits_per_file(self, tmp_path: Path):
        from osprey.cli.scaffold_cmd import claim, diff

        project_path = self._project(tmp_path)
        runner = CliRunner()
        assert runner.invoke(claim, ["services/postgresql", "-p", str(project_path)]).exit_code == 0

        target = project_path / "services" / "postgresql" / "docker-compose.yml.j2"
        target.write_text(target.read_text(encoding="utf-8") + "# local tweak\n", encoding="utf-8")

        result = runner.invoke(diff, ["services/postgresql", "-p", str(project_path)])
        assert result.exit_code == 0, result.output
        assert "# local tweak" in result.output
        assert "services/postgresql/docker-compose.yml.j2" in result.output

    def test_diff_clean_after_claim(self, tmp_path: Path):
        from osprey.cli.scaffold_cmd import claim, diff

        project_path = self._project(tmp_path)
        runner = CliRunner()
        assert runner.invoke(claim, ["services/postgresql", "-p", str(project_path)]).exit_code == 0

        result = runner.invoke(diff, ["services/postgresql", "-p", str(project_path)])
        assert result.exit_code == 0, result.output
        assert "no differences" in result.output
