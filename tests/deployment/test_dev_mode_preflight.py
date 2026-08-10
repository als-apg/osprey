"""Tests for the ``--dev`` preflight and its hard-fail contract.

``--dev`` is a statement about *which code runs in the containers*. When the
local wheel could not be staged the deploy used to log a warning and continue,
bringing containers up on the pinned PyPI release with exit code 0 — so a
deployment meant to test local changes silently tested the released package
instead. These tests pin the replacement: refuse, loudly, before any work.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment.errors import DevModeUnavailableError
from osprey.deployment.wheel_build import (
    _build_dev_wheel_cached,
    _copy_local_framework_for_override,
    _reset_wheel_build_cache,
    preflight_dev_mode,
)


@pytest.fixture(autouse=True)
def _clean_wheel_cache():
    _reset_wheel_build_cache()
    yield
    _reset_wheel_build_cache()


@pytest.fixture
def editable_checkout(tmp_path, monkeypatch):
    """Fake an editable osprey install rooted at a writable temp checkout."""
    src = tmp_path / "checkout" / "src" / "osprey"
    src.mkdir(parents=True)
    (tmp_path / "checkout" / "pyproject.toml").write_text("[project]\nname='osprey'\n")

    import osprey

    monkeypatch.setattr(osprey, "__file__", str(src / "__init__.py"))
    return tmp_path / "checkout"


class TestPreflightDevMode:
    """Every precondition fails fast, before any deploy work happens."""

    def test_missing_build_package_raises(self, editable_checkout, monkeypatch):
        import importlib.util

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "build" else real_find_spec(name, *a, **k),
        )

        with pytest.raises(DevModeUnavailableError) as exc:
            preflight_dev_mode()

        assert "build" in str(exc.value)

    def test_missing_build_package_names_the_install_command(self, editable_checkout, monkeypatch):
        """The error is only useful if it says how to fix it."""
        import importlib.util

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "build" else real_find_spec(name, *a, **k),
        )

        with pytest.raises(DevModeUnavailableError) as exc:
            preflight_dev_mode()

        assert "uv sync --extra dev" in exc.value.remedy

    def test_non_editable_install_raises(self, monkeypatch, tmp_path):
        site = tmp_path / "site-packages" / "osprey"
        site.mkdir(parents=True)

        import osprey

        monkeypatch.setattr(osprey, "__file__", str(site / "__init__.py"))

        with pytest.raises(DevModeUnavailableError) as exc:
            preflight_dev_mode()

        assert "editable" in str(exc.value) or "editable" in exc.value.remedy

    def test_missing_pyproject_raises(self, monkeypatch, tmp_path):
        src = tmp_path / "checkout" / "src" / "osprey"
        src.mkdir(parents=True)
        # deliberately no pyproject.toml

        import osprey

        monkeypatch.setattr(osprey, "__file__", str(src / "__init__.py"))

        with pytest.raises(DevModeUnavailableError) as exc:
            preflight_dev_mode()

        assert "pyproject.toml" in str(exc.value)

    def test_healthy_environment_passes(self, editable_checkout):
        """The repo's own dev environment must satisfy the preflight."""
        preflight_dev_mode()  # must not raise

    def test_preflight_is_cheap(self, editable_checkout, monkeypatch):
        """No subprocess: the whole point is failing before any real work."""

        def _explode(*args, **kwargs):
            raise AssertionError("preflight must not spawn a subprocess")

        monkeypatch.setattr(subprocess, "run", _explode)

        preflight_dev_mode()


class TestWheelBuildFailuresRaise:
    """A failed wheel build must not degrade into a released-code deploy."""

    def test_missing_build_module_raises(self, editable_checkout, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0] if a else [], 1, "", "No module named build"
            ),
        )

        with pytest.raises(DevModeUnavailableError):
            _build_dev_wheel_cached(editable_checkout)

    def test_broken_local_source_raises_and_surfaces_stderr(self, editable_checkout, monkeypatch):
        """The most dangerous case: local code is broken, so it must not be hidden."""
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0] if a else [], 1, "", "SyntaxError: invalid syntax in osprey/foo.py"
            ),
        )

        with pytest.raises(DevModeUnavailableError) as exc:
            _build_dev_wheel_cached(editable_checkout)

        assert "SyntaxError" in exc.value.remedy

    def test_failed_build_is_not_memoized(self, editable_checkout, monkeypatch):
        """A fixed checkout must be retried, not served from a poisoned memo."""
        calls = []

        def _fail(*a, **k):
            calls.append(1)
            return subprocess.CompletedProcess(a[0] if a else [], 1, "", "boom")

        monkeypatch.setattr(subprocess, "run", _fail)

        for _ in range(2):
            with pytest.raises(DevModeUnavailableError):
                _build_dev_wheel_cached(editable_checkout)

        assert len(calls) == 2, "failed builds must not be cached"

    def test_build_producing_no_wheel_raises(self, editable_checkout, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
        )

        with pytest.raises(DevModeUnavailableError):
            _build_dev_wheel_cached(editable_checkout)


class TestStagingNeverSilentlyFallsBack:
    """``_copy_local_framework_for_override`` returns True or raises — no False."""

    def test_non_editable_install_raises_instead_of_returning_false(self, monkeypatch, tmp_path):
        site = tmp_path / "site-packages" / "osprey"
        site.mkdir(parents=True)

        import osprey

        monkeypatch.setattr(osprey, "__file__", str(site / "__init__.py"))

        with pytest.raises(DevModeUnavailableError):
            _copy_local_framework_for_override(str(tmp_path / "ctx"))

    def test_unexpected_error_is_converted_not_swallowed(self, editable_checkout, monkeypatch):
        """The old broad `except Exception` returned False, hiding the fallback."""
        import osprey.deployment.wheel_build as wb

        def _boom(*a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(wb, "_build_dev_wheel_cached", _boom)

        with pytest.raises(DevModeUnavailableError) as exc:
            _copy_local_framework_for_override(str(editable_checkout))

        assert "disk on fire" in str(exc.value)


class TestPrepareComposeFilesPreflight:
    """The guard runs before config loading, so nothing is half-prepared."""

    def test_dev_mode_preflights_before_touching_config(self, monkeypatch, tmp_path):
        import osprey.deployment.compose_generator as cg

        def _fail_preflight():
            raise DevModeUnavailableError("nope", "do the thing")

        def _config_must_not_load(*a, **k):
            raise AssertionError("config was loaded before the --dev preflight ran")

        monkeypatch.setattr(cg, "preflight_dev_mode", _fail_preflight)
        monkeypatch.setattr(cg, "load_project_config", _config_must_not_load)

        with pytest.raises(DevModeUnavailableError):
            cg.prepare_compose_files(str(tmp_path / "config.yml"), dev_mode=True)

    def test_non_dev_mode_skips_the_preflight(self, monkeypatch, tmp_path):
        """A production deploy must not require an editable checkout."""
        import osprey.deployment.compose_generator as cg

        def _must_not_run():
            raise AssertionError("preflight ran without --dev")

        monkeypatch.setattr(cg, "preflight_dev_mode", _must_not_run)

        # Fails later on the missing config, which is fine — it proves the
        # preflight was not reached.
        with pytest.raises(Exception) as exc:
            cg.prepare_compose_files(str(tmp_path / "missing.yml"), dev_mode=False)

        assert not isinstance(exc.value, AssertionError)


class TestPureImageContextsAreQuiet:
    """Routine no-Dockerfile contexts must not drown the real signal."""

    def test_no_dockerfile_context_logs_at_debug(self, tmp_path, caplog):
        import logging

        from osprey.deployment.compose_generator import _stage_dev_wheel_for_context

        ctx = tmp_path / "postgresql"
        ctx.mkdir()

        with caplog.at_level(logging.INFO):
            staged = _stage_dev_wheel_for_context(str(ctx), dev_mode=True)

        assert staged is False
        assert "no osprey wheel was staged" not in caplog.text

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            _stage_dev_wheel_for_context(str(ctx), dev_mode=True)

        assert "no osprey wheel was staged" in caplog.text


def test_build_is_a_base_dependency():
    """`--dev` is a shipped CLI flag; it must not need an optional extra."""
    import tomllib

    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    base = data["project"]["dependencies"]
    assert any(dep == "build" or dep.startswith("build") for dep in base), (
        "'build' must be a base dependency: an editable install without "
        "--extra dev otherwise passes every other --dev precondition and "
        "fails only at wheel-build time"
    )
