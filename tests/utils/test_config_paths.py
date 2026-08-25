"""The one rule for resolving a config-relative path.

Every config key that names a directory or file relative to the project —
``facility_knowledge.bundle_path``, the qmd mirror, the ARIEL vocabulary file —
must land on the same directory no matter which process reads it. These tests
pin every branch of :func:`osprey.utils.config_paths.resolve_config_relative_path`
— including the one that matters on a host, where the config sits in the
``build/`` render zone and the anchor is the repo above it — and assert that the
two pre-existing resolvers really delegate to it rather than keeping private
copies of the rule that can drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.utils.config_paths import project_root_for_config_dir, resolve_config_relative_path


@pytest.fixture
def foreign_cwd(tmp_path, monkeypatch):
    """Run from a directory unrelated to the project, and return it.

    A CWD-relative resolver is only distinguishable from a config-relative one
    when the two differ, so every test here starts somewhere else.
    """
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_absolute_value_passes_through_unchanged(foreign_cwd, tmp_path):
    """An absolute value is returned as given — not re-resolved."""
    absolute = tmp_path / "shared" / "bundle"

    assert resolve_config_relative_path(absolute, tmp_path / "project") == absolute
    assert resolve_config_relative_path(str(absolute), None) == absolute


def test_tilde_value_is_expanded_against_home(foreign_cwd, tmp_path, monkeypatch):
    """``~/...`` expands to the user's home, ignoring both CWD and config dir."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert resolve_config_relative_path("~/kb", tmp_path / "project") == home / "kb"
    assert resolve_config_relative_path("~/kb", None) == home / "kb"


def test_relative_value_resolves_against_config_dir(foreign_cwd, tmp_path):
    """With *config_dir* given, a relative value hangs off it, not the CWD."""
    config_dir = tmp_path / "project"
    config_dir.mkdir()

    resolved = resolve_config_relative_path("data/facility_knowledge", config_dir)

    assert resolved == (config_dir / "data/facility_knowledge").resolve()
    assert resolved != (foreign_cwd / "data/facility_knowledge").resolve()


def test_relative_value_in_the_build_zone_resolves_against_the_repo(foreign_cwd, tmp_path):
    """A config at ``<repo>/build/config.yml`` anchors on ``<repo>``, not ``build/``.

    The render zone is disposable — every ``osprey build`` re-creates it — and
    it is not what the compose layer binds into containers: every bind source is
    resolved against the repo root (the pinned ``--project-directory``). A
    resolver anchored on ``build/`` had the qmd exporter writing a mirror the
    sidecar never mounted, and persona readers opening a baked bundle while the
    live one was bound one directory over.
    """
    repo = tmp_path / "repo"
    build = repo / "build"
    build.mkdir(parents=True)

    resolved = resolve_config_relative_path("var/ariel_mirror", build)

    assert resolved == (repo / "var/ariel_mirror").resolve()
    assert project_root_for_config_dir(build) == repo


def test_a_container_project_dir_is_its_own_root(foreign_cwd, tmp_path):
    """A config not in a ``build/`` zone anchors on its own directory.

    A container's project directory IS its render (``/app/<project>`` holds the
    config at ``build/config.yml`` only when the image carries the repo layout;
    a flat legacy project holds it at the root), so a directory not named
    ``build`` is the root — the same rule ``repo_root_for_config`` applies to
    every other runtime path.
    """
    project = tmp_path / "project"
    project.mkdir()

    assert project_root_for_config_dir(project) == project
    assert resolve_config_relative_path("data/kb", project) == (project / "data/kb").resolve()


def test_relative_value_without_config_dir_uses_the_resolved_config_file(
    foreign_cwd, tmp_path, monkeypatch
):
    """Without *config_dir*, the project of ``OSPREY_CONFIG``'s file wins."""
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text("", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_dir / "config.yml"))

    resolved = resolve_config_relative_path("data/vocab.yml", None)

    assert resolved == (config_dir / "data/vocab.yml").resolve()


def test_without_config_dir_a_build_zone_config_still_anchors_on_the_repo(
    foreign_cwd, tmp_path, monkeypatch
):
    """The ``OSPREY_CONFIG`` fallback applies the same build-zone rule."""
    repo = tmp_path / "repo"
    build = repo / "build"
    build.mkdir(parents=True)
    (build / "config.yml").write_text("", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

    resolved = resolve_config_relative_path("data/vocab.yml", None)

    assert resolved == (repo / "data/vocab.yml").resolve()


def test_relative_value_falls_back_to_the_cwd_when_config_lookup_raises(foreign_cwd, monkeypatch):
    """A raising ``resolve_config_path()`` degrades to a CWD-relative path.

    The fallback exists because callers include a launcher that swallows
    errors: a resolution failure must not take down ingestion or a UI.
    """
    import osprey_connectors.workspace as workspace

    def _boom() -> Path:
        raise RuntimeError("no workspace")

    monkeypatch.setattr(workspace, "resolve_config_path", _boom)

    resolved = resolve_config_relative_path("data/vocab.yml", None)

    assert resolved == (foreign_cwd / "data/vocab.yml").resolve()


@pytest.mark.parametrize(
    ("value", "config_dir_name"),
    [
        ("/abs/bundle", "project"),
        ("~/kb", "project"),
        ("data/rel", "project"),
        ("data/rel", None),
    ],
)
def test_both_public_resolvers_delegate_to_the_shared_helper(
    value, config_dir_name, tmp_path, monkeypatch
):
    """``resolve_bundle_path`` and ``resolve_mirror_path`` are delegations.

    They keep their own names and signatures, but the rule itself lives in one
    place — so both must hold the shared helper itself and hand it the
    arguments they were given, unaltered.
    """
    from osprey.services.ariel_search.enhancement.qmd_export import exporter
    from osprey.services.facility_knowledge import bundle_path

    assert bundle_path.resolve_config_relative_path is resolve_config_relative_path
    assert exporter.resolve_config_relative_path is resolve_config_relative_path

    config_dir = None if config_dir_name is None else tmp_path / config_dir_name
    sentinel = tmp_path / "sentinel"
    calls: list[tuple[object, object]] = []

    def _record(raw, config_dir=None):
        calls.append((raw, config_dir))
        return sentinel

    monkeypatch.setattr(bundle_path, "resolve_config_relative_path", _record)
    monkeypatch.setattr(exporter, "resolve_config_relative_path", _record)

    assert bundle_path.resolve_bundle_path(value, config_dir) is sentinel
    assert exporter.resolve_mirror_path(value, config_dir) is sentinel
    assert calls == [(value, config_dir), (value, config_dir)]
