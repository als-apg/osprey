"""Tests for the core ``file_system`` health category.

Covers the lazy ``CORE_CATEGORIES`` registry surface, the factory/contract
shape, and the per-row status semantics of the migrated file-system checks:
``project_paths``, ``project_root_path``, ``agent_data_dir``, ``env_file``,
``registry_file`` and ``disk_space``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.health.core import CORE_CATEGORIES, get_core_category_factory
from osprey.health.core.file_system import file_system
from osprey.health.models import CheckResult, Status

_GB = 1024**3


def _run(
    config: dict[str, Any] | None,
    cwd: Path,
    context: Any = None,
) -> list[CheckResult]:
    """Build and invoke the file_system category callable for ``cwd``."""
    category = file_system(config, context, cwd=cwd)
    return category()


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


# --------------------------------------------------------------------------- #
# Registry surface / factory contract
# --------------------------------------------------------------------------- #


class TestRegistryAndContract:
    def test_registry_resolves_file_system_to_this_factory(self) -> None:
        assert CORE_CATEGORIES["file_system"] is file_system
        assert get_core_category_factory("file_system") is file_system

    def test_factory_returns_zero_arg_callable_producing_results(self, tmp_path: Path) -> None:
        category = file_system({}, None, cwd=tmp_path)
        results = category()
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)

    def test_every_row_is_categorized_file_system(self, tmp_path: Path) -> None:
        results = _run({}, tmp_path)
        assert results
        assert all(r.category == "file_system" for r in results)

    def test_context_argument_is_accepted_and_ignored(self, tmp_path: Path) -> None:
        sentinel = object()
        results = _run({}, tmp_path, context=sentinel)
        assert results  # a non-connector category simply ignores the context

    def test_none_config_degrades_gracefully(self, tmp_path: Path) -> None:
        by_name = _by_name(_run(None, tmp_path))
        # No project_root available -> single project_paths warning, not an error.
        assert by_name["project_paths"].status is Status.WARNING
        assert by_name["project_paths"].message == "No project_root configured"

    def test_cwd_defaults_to_process_cwd_when_omitted(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("X=1\n")
        results = file_system({}, None)()  # no cwd kwarg -> Path.cwd()
        assert _by_name(results)["env_file"].status is Status.OK


# --------------------------------------------------------------------------- #
# project_paths / project_root_path / agent_data_dir
# --------------------------------------------------------------------------- #


class TestProjectPaths:
    def test_no_project_root_configured_warns(self, tmp_path: Path) -> None:
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["project_paths"].status is Status.WARNING
        assert by_name["project_paths"].message == "No project_root configured"
        # The early return means no project_root_path / agent_data_dir rows.
        assert "project_root_path" not in by_name
        assert "agent_data_dir" not in by_name

    def test_project_root_exists_ok(self, tmp_path: Path) -> None:
        by_name = _by_name(_run({"project_root": str(tmp_path)}, tmp_path))
        assert by_name["project_root_path"].status is Status.OK
        assert str(tmp_path) in by_name["project_root_path"].message

    def test_project_root_missing_warns(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        by_name = _by_name(_run({"project_root": str(missing)}, tmp_path))
        assert by_name["project_root_path"].status is Status.WARNING
        assert "does not exist" in by_name["project_root_path"].message

    def test_project_root_expands_env_var(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        by_name = _by_name(_run({"project_root": "${PROJECT_ROOT}"}, tmp_path))
        assert by_name["project_root_path"].status is Status.OK
        assert str(tmp_path) in by_name["project_root_path"].message

    def test_agent_data_dir_existing_and_writable_ok(self, tmp_path: Path) -> None:
        (tmp_path / "_agent_data").mkdir()
        by_name = _by_name(_run({"project_root": str(tmp_path)}, tmp_path))
        assert by_name["agent_data_dir"].status is Status.OK
        assert "writable" in by_name["agent_data_dir"].message

    def test_agent_data_dir_can_be_created_ok(self, tmp_path: Path) -> None:
        # Directory absent but parent (project_root) exists and is writable.
        by_name = _by_name(_run({"project_root": str(tmp_path)}, tmp_path))
        assert by_name["agent_data_dir"].status is Status.OK
        assert "can be created" in by_name["agent_data_dir"].message

    def test_custom_agent_data_dir_from_config(self, tmp_path: Path) -> None:
        (tmp_path / "custom").mkdir()
        config = {"project_root": str(tmp_path), "file_paths": {"agent_data_dir": "custom"}}
        by_name = _by_name(_run(config, tmp_path))
        assert by_name["agent_data_dir"].status is Status.OK
        assert str(tmp_path / "custom") in by_name["agent_data_dir"].message

    def test_agent_data_dir_cannot_be_created_warns(self, tmp_path: Path) -> None:
        # project_root itself does not exist, so the agent-data parent is absent
        # and the directory can neither be found nor created.
        missing_root = tmp_path / "ghost"
        by_name = _by_name(_run({"project_root": str(missing_root)}, tmp_path))
        assert by_name["agent_data_dir"].status is Status.WARNING
        assert "Cannot create" in by_name["agent_data_dir"].message

    def test_agent_data_dir_not_writable_warns(self, tmp_path: Path) -> None:
        if hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0:
            pytest.skip("root bypasses filesystem write permissions")
        data_dir = tmp_path / "_agent_data"
        data_dir.mkdir()
        data_dir.chmod(0o500)  # read/execute only, not writable
        try:
            by_name = _by_name(_run({"project_root": str(tmp_path)}, tmp_path))
            assert by_name["agent_data_dir"].status is Status.WARNING
            assert "not writable" in by_name["agent_data_dir"].message
        finally:
            data_dir.chmod(0o700)  # restore so tmp cleanup can remove it

    def test_exception_path_yields_project_paths_error(self, tmp_path: Path) -> None:
        # file_paths is a str, so file_paths.get(...) raises AttributeError, which
        # the broad guard reports as a single project_paths error row.
        config = {"project_root": str(tmp_path), "file_paths": "not-a-mapping"}
        by_name = _by_name(_run(config, tmp_path))
        assert by_name["project_paths"].status is Status.ERROR
        assert "Error checking project paths" in by_name["project_paths"].message


# --------------------------------------------------------------------------- #
# env_file
# --------------------------------------------------------------------------- #


class TestEnvFile:
    def test_env_file_present_ok(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("KEY=value\n")
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["env_file"].status is Status.OK
        assert by_name["env_file"].message == ".env file found"

    def test_env_file_absent_warns(self, tmp_path: Path) -> None:
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["env_file"].status is Status.WARNING
        assert by_name["env_file"].message == ".env file not found"


# --------------------------------------------------------------------------- #
# registry_file
# --------------------------------------------------------------------------- #


class TestRegistryFile:
    def test_no_config_yml_emits_no_registry_row(self, tmp_path: Path) -> None:
        assert "registry_file" not in _by_name(_run({}, tmp_path))

    def test_config_yml_without_registry_path_emits_no_row(self, tmp_path: Path) -> None:
        (tmp_path / "config.yml").write_text("project_root: /somewhere\n")
        assert "registry_file" not in _by_name(_run({}, tmp_path))

    def test_registry_file_present_ok(self, tmp_path: Path) -> None:
        (tmp_path / "registry.yml").write_text("components: []\n")
        (tmp_path / "config.yml").write_text("registry_path: registry.yml\n")
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["registry_file"].status is Status.OK
        assert str(tmp_path / "registry.yml") in by_name["registry_file"].message

    def test_registry_file_configured_but_missing_errors(self, tmp_path: Path) -> None:
        (tmp_path / "config.yml").write_text("registry_path: registry.yml\n")
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["registry_file"].status is Status.ERROR
        assert "not found" in by_name["registry_file"].message

    def test_registry_path_expands_env_var(self, monkeypatch, tmp_path: Path) -> None:
        (tmp_path / "reg.yml").write_text("x: 1\n")
        monkeypatch.setenv("REG_FILE", "reg.yml")
        (tmp_path / "config.yml").write_text("registry_path: ${REG_FILE}\n")
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["registry_file"].status is Status.OK

    def test_empty_config_yml_is_tolerated(self, tmp_path: Path) -> None:
        # safe_load of an empty file returns None; the broad guard swallows the
        # resulting AttributeError and no registry_file row is emitted.
        (tmp_path / "config.yml").write_text("")
        by_name = _by_name(_run({}, tmp_path))
        assert "registry_file" not in by_name


# --------------------------------------------------------------------------- #
# disk_space
# --------------------------------------------------------------------------- #


class TestDiskSpace:
    def _patch_usage(self, monkeypatch, *, total: float, used: float, free: float) -> None:
        monkeypatch.setattr(
            "osprey.health.core.file_system.shutil.disk_usage",
            lambda _path: SimpleNamespace(total=total, used=used, free=free),
        )

    def test_ample_space_ok(self, monkeypatch, tmp_path: Path) -> None:
        self._patch_usage(monkeypatch, total=100 * _GB, used=50 * _GB, free=50 * _GB)
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["disk_space"].status is Status.OK
        assert "50% full" in by_name["disk_space"].message

    def test_low_free_space_warns(self, monkeypatch, tmp_path: Path) -> None:
        # Under 1 GB free even though the disk is only 20% full.
        self._patch_usage(monkeypatch, total=1000 * _GB, used=200 * _GB, free=0.5 * _GB)
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["disk_space"].status is Status.WARNING
        assert "GB free" in by_name["disk_space"].message
        assert by_name["disk_space"].details

    def test_high_percentage_full_warns(self, monkeypatch, tmp_path: Path) -> None:
        # 91% full though 9 GB remain free.
        self._patch_usage(monkeypatch, total=100 * _GB, used=91 * _GB, free=9 * _GB)
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["disk_space"].status is Status.WARNING
        assert "91% full" in by_name["disk_space"].message

    def test_zero_total_does_not_divide_by_zero(self, monkeypatch, tmp_path: Path) -> None:
        self._patch_usage(monkeypatch, total=0, used=0, free=0)
        by_name = _by_name(_run({}, tmp_path))
        # free_gb (0) < 1.0 -> warning, and pct_used is 0 rather than raising.
        assert by_name["disk_space"].status is Status.WARNING
        assert "0% full" in by_name["disk_space"].message

    def test_disk_usage_error_warns(self, monkeypatch, tmp_path: Path) -> None:
        def _boom(_path: Path) -> None:
            raise OSError("stat failed")

        monkeypatch.setattr("osprey.health.core.file_system.shutil.disk_usage", _boom)
        by_name = _by_name(_run({}, tmp_path))
        assert by_name["disk_space"].status is Status.WARNING
        assert "Could not check disk space" in by_name["disk_space"].message
