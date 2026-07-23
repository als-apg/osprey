"""Tests for the deploy-side project staleness advisory.

A rendered project stamps its provenance (osprey version + resolved-preset
hash) into ``.osprey-manifest.json`` at build time; ``deploy up`` and
``deploy status`` compare that against the installed framework and warn —
never fail — when the project predates the code deploying it. The check is
fail-open by design: a legacy project without a manifest deploys silently.
"""

from __future__ import annotations

import json

import pytest

from osprey.cli import build_profile
from osprey.deployment import staleness


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    d = tmp_path / "presets"
    d.mkdir()
    monkeypatch.setattr(build_profile, "_presets_dir", lambda: d)
    return d


def _write_preset(d, name, text):
    (d / f"{name}.yml").write_text(text, encoding="utf-8")


def _write_manifest(project_dir, **overrides):
    data = {
        "schema_version": "1.2.0",
        "creation": {
            "osprey_version": "2026.7.0",
            "template": "demo",
        },
        "build_args": {"source": "preset", "preset": "demo", "project_name": "proj"},
        "reproducible_command": "osprey build proj --preset demo",
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    (project_dir / ".osprey-manifest.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_no_manifest_is_silent(tmp_path):
    assert staleness.staleness_reasons(tmp_path) == []


def test_corrupt_manifest_is_silent(tmp_path):
    (tmp_path / ".osprey-manifest.json").write_text("{not json", encoding="utf-8")
    assert staleness.staleness_reasons(tmp_path) == []


def test_fresh_project_yields_no_reasons(tmp_path, presets_dir, monkeypatch):
    _write_preset(presets_dir, "demo", "name: Demo\n")
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_manifest(
        tmp_path,
        creation={"preset_hash": build_profile.compute_preset_hash("demo")},
    )
    assert staleness.staleness_reasons(tmp_path) == []


def test_version_drift_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.8.1")
    _write_manifest(tmp_path)
    reasons = staleness.staleness_reasons(tmp_path)
    assert len(reasons) == 1
    assert "2026.7.0" in reasons[0] and "2026.8.1" in reasons[0]


def test_unknown_installed_version_is_silent(tmp_path, monkeypatch):
    """A broken/partial install must not manufacture a drift warning."""
    monkeypatch.setattr(staleness, "_installed_version", lambda: "unknown")
    _write_manifest(tmp_path)
    assert staleness.staleness_reasons(tmp_path) == []


def test_preset_content_drift_is_reported(tmp_path, presets_dir, monkeypatch):
    """Same installed version, changed preset — the --dev checkout incident."""
    _write_preset(presets_dir, "demo", "name: Demo\n")
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_manifest(
        tmp_path,
        creation={"preset_hash": build_profile.compute_preset_hash("demo")},
    )
    _write_preset(presets_dir, "demo", "name: Demo\nmodules.web_terminals:\n  enabled: true\n")
    reasons = staleness.staleness_reasons(tmp_path)
    assert len(reasons) == 1
    assert "demo" in reasons[0]


def test_manifest_without_preset_hash_skips_content_check(tmp_path, presets_dir, monkeypatch):
    """Manifests from before the stamp existed only get the version check."""
    _write_preset(presets_dir, "demo", "name: Demo\n")
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_manifest(tmp_path)
    assert staleness.staleness_reasons(tmp_path) == []


def test_removed_preset_is_silent_on_content_check(tmp_path, presets_dir, monkeypatch):
    """A preset that no longer ships must not crash or false-positive."""
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_manifest(tmp_path, creation={"preset_hash": "sha256:deadbeef"})
    assert staleness.staleness_reasons(tmp_path) == []


@pytest.fixture
def _captured_warnings(monkeypatch):
    warnings: list = []
    monkeypatch.setattr(
        staleness.logger, "warning", lambda *a, **k: warnings.append(" ".join(map(str, a)))
    )
    return warnings


def test_warn_if_project_stale_logs_reasons_and_remedy(tmp_path, monkeypatch, _captured_warnings):
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.8.1")
    _write_manifest(tmp_path)
    staleness.warn_if_project_stale(tmp_path)
    assert len(_captured_warnings) == 1
    text = _captured_warnings[0]
    assert "2026.7.0" in text
    assert "osprey build proj --preset demo --force" in text


def test_warn_if_project_stale_is_quiet_when_fresh(tmp_path, monkeypatch, _captured_warnings):
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_manifest(tmp_path)
    staleness.warn_if_project_stale(tmp_path)
    assert _captured_warnings == []


def test_warn_if_project_stale_never_raises(tmp_path, monkeypatch):
    """Advisory means advisory: internal failure must not break a deploy."""

    def _boom(project_dir):
        raise RuntimeError("staleness exploded")

    monkeypatch.setattr(staleness, "staleness_reasons", _boom)
    staleness.warn_if_project_stale(tmp_path)  # must not raise
