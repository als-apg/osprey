"""Tests for TemplateService."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.optimizer.services.template_service import MAX_TEMPLATE_SIZE, TemplateService

VALID_TEMPLATE_YAML = """\
name: test_template
override: als_injector
description: A test template
"""

VALID_TEMPLATE_YML = """\
name: another_template
override: als_injector
"""

INVALID_TEMPLATE_YAML = """\
not_a_valid: template
missing: required_fields
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_scan_templates_finds_yaml_files(tmp_path: Path) -> None:
    """scan_templates discovers .yaml and .yml files."""
    _write(tmp_path / "one.yaml", VALID_TEMPLATE_YAML)
    _write(tmp_path / "two.yml", VALID_TEMPLATE_YML)

    service = TemplateService(template_dir=tmp_path)
    results = service.scan_templates()

    names = {r.name for r in results}
    assert names == {"one", "two"}
    assert all(r.valid for r in results)


def test_scan_templates_missing_dir_returns_empty(tmp_path: Path) -> None:
    """Nonexistent template directory returns an empty list."""
    service = TemplateService(template_dir=tmp_path / "does_not_exist")
    assert service.scan_templates() == []


def test_scan_templates_invalid_template_marked(tmp_path: Path) -> None:
    """Invalid YAML templates are flagged as valid=False."""
    _write(tmp_path / "bad.yaml", INVALID_TEMPLATE_YAML)

    service = TemplateService(template_dir=tmp_path)
    results = service.scan_templates()

    assert len(results) == 1
    assert results[0].valid is False
    assert results[0].error is not None
    assert "override" in results[0].error  # the missing required field is named


@pytest.mark.parametrize("extension", ["yaml", "yml"])
def test_load_template_config_handles_both_extensions(tmp_path: Path, extension: str) -> None:
    """load_template_config picks up either .yaml or .yml on disk."""
    _write(tmp_path / f"my_tmpl.{extension}", VALID_TEMPLATE_YAML)

    service = TemplateService(template_dir=tmp_path)
    config = service.load_template_config("my_tmpl")

    assert config["name"] == "test_template"
    assert config["override"] == "als_injector"


def test_load_template_config_not_found_raises(tmp_path: Path) -> None:
    """load_template_config raises FileNotFoundError for missing template."""
    service = TemplateService(template_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        service.load_template_config("nonexistent")


@pytest.mark.parametrize(
    "content, expected_match",
    [
        pytest.param("- item1\n- item2\n", "must be a dictionary", id="yaml-is-list"),
        pytest.param("key: [unterminated\n", "Invalid YAML format", id="malformed-yaml"),
        pytest.param(INVALID_TEMPLATE_YAML, "Invalid template", id="missing-required-fields"),
        pytest.param("x" * (MAX_TEMPLATE_SIZE + 1), "too large", id="oversized"),
    ],
)
def test_load_template_config_invalid_content_raises(
    tmp_path: Path, content: str, expected_match: str
) -> None:
    """Non-dict YAML, malformed YAML, missing required fields, and oversized files all
    surface as ValueError with a clear message."""
    _write(tmp_path / "bad.yaml", content)

    service = TemplateService(template_dir=tmp_path)

    with pytest.raises(ValueError, match=expected_match):
        service.load_template_config("bad")


def test_default_dir_picks_docker_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When /app/templates exists, it wins over the local default."""
    docker_dir = tmp_path / "docker_templates"
    docker_dir.mkdir()

    real_path_cls = Path

    def fake_path(value: str) -> Path:
        if value == "/app/templates":
            return docker_dir
        return real_path_cls(value)

    monkeypatch.setattr("osprey.services.optimizer.services.template_service.Path", fake_path)

    service = TemplateService()
    assert service.template_dir == docker_dir


def test_scan_templates_unexpected_error_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-Validation/Value errors during loading become 'Unexpected error' entries."""
    _write(tmp_path / "doomed.yaml", VALID_TEMPLATE_YAML)
    service = TemplateService(template_dir=tmp_path)

    def boom(_self: TemplateService, _path: Path) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(TemplateService, "_load_template", boom)

    results = service.scan_templates()

    assert len(results) == 1
    assert results[0].valid is False
    assert results[0].error is not None
    assert "Unexpected error" in results[0].error


def test_load_template_missing_path_raises_filenotfound(tmp_path: Path) -> None:
    """Direct call to _load_template with a non-existent path raises FileNotFoundError."""
    service = TemplateService(template_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        service._load_template(tmp_path / "ghost.yaml")
