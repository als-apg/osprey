"""Framework-guard tests for projects with empty deployed_services.

The hello-world preset (and any future "agent-only" preset) declares no
deployed_services. Two failure modes have to stay fixed:

1. ``osprey build`` must still copy the root ``services/docker-compose.yml.j2``
   into the project, because the renderer references it unconditionally.
2. ``osprey deploy up`` must succeed (graceful no-op) instead of dying with
   ``TemplateNotFound`` mid-render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment.compose_generator import prepare_compose_files


def _write_config(project_path: Path, deployed_services: list[str]) -> Path:
    """Write a minimal config.yml into ``project_path`` and return its path."""
    config_path = project_path / "config.yml"
    yaml = YAML()
    config: dict = {
        "project_name": "hwt-fixture",
        "build_dir": str(project_path / "build"),
        "deployed_services": deployed_services,
    }
    with open(config_path, "w") as fh:
        yaml.dump(config, fh)
    return config_path


def test_copy_service_templates_copies_root_when_no_services(tmp_path: Path) -> None:
    """Empty deployed_services must still copy the root compose template."""
    _write_config(tmp_path, deployed_services=[])

    result = _copy_service_templates(tmp_path)

    assert result == 0, "no per-service templates should be copied"
    root_template = tmp_path / "services" / "docker-compose.yml.j2"
    assert root_template.is_file(), (
        "root services/docker-compose.yml.j2 must be copied even when "
        "deployed_services is empty (otherwise prepare_compose_files dies "
        "with TemplateNotFound at render time)"
    )


def test_prepare_compose_files_no_services_renders_root_only(tmp_path: Path) -> None:
    """With empty deployed_services, prepare_compose_files renders just the root."""
    config_path = _write_config(tmp_path, deployed_services=[])
    _copy_service_templates(tmp_path)

    monkey_cwd = Path.cwd()
    try:
        # render_template resolves SERVICES_DIR relative to cwd
        import os

        os.chdir(tmp_path)
        config, compose_files = prepare_compose_files(str(config_path))
    finally:
        import os

        os.chdir(monkey_cwd)

    assert len(compose_files) == 1, (
        f"expected exactly one rendered compose file (the root), got {compose_files}"
    )
    rendered = Path(compose_files[0])
    assert rendered.is_file()
    rendered_text = rendered.read_text()
    assert "services:" not in rendered_text, (
        f"empty-deployed-services preset rendered a compose with services block:\n{rendered_text}"
    )


def test_copy_service_templates_no_config_returns_zero(tmp_path: Path) -> None:
    """Missing config.yml is a no-op, not a crash."""
    result = _copy_service_templates(tmp_path)
    assert result == 0
    assert not (tmp_path / "services").exists()


@pytest.mark.parametrize("services", [[], ["does.not.exist"]])
def test_copy_service_templates_root_always_present_with_valid_pkg_services(
    tmp_path: Path,
    services: list[str],
) -> None:
    """Whether deployed_services is empty or has unknown entries, the root is copied first."""
    _write_config(tmp_path, deployed_services=services)
    _copy_service_templates(tmp_path)
    assert (tmp_path / "services" / "docker-compose.yml.j2").is_file()
