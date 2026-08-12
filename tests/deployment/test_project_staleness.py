"""Tests for the deploy-side render staleness advisory.

A render stamps its provenance (osprey version + resolved-profile hash) into
``.osprey-manifest.json`` at build time; ``osprey status`` compares that
against the installed framework and warns — never fails — when the render
predates the code deploying it. The check is fail-open by design: a render
without a manifest is reported on silently.

Distinct from the drift REFUSAL the start verbs apply
(``staleness.check_drift``, pinned in tests/deployment/test_up_as_built.py):
this advisory is the softer version-and-content comparison that reports rather
than blocks.
"""

from __future__ import annotations

import json

import pytest

from osprey.cli import build_profile, build_profile_presets
from osprey.deployment import staleness


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    d = tmp_path / "presets"
    d.mkdir()
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: d)
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
        "reproducible_command": "osprey build",
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


def test_development_commits_do_not_read_as_drift(tmp_path, presets_dir, monkeypatch):
    """Two dev builds of the same release must not register as staleness.

    Both sides of this comparison carry the release lineage, not the running
    version. If either carried the running version, every commit in a development
    checkout would report the project as stale and the advisory would become noise.
    """
    _write_preset(presets_dir, "demo", "name: Demo\n")

    # Rendered at one dev commit — the manifest's own osprey_version is what
    # pins that side of the comparison.
    _write_manifest(
        tmp_path,
        creation={
            "osprey_version": "2026.7.0",
            "preset_hash": build_profile.compute_preset_hash("demo"),
        },
    )

    # ...deployed from another, 40 commits later on the same release.
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")

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


def test_preset_content_drift_is_reported_when_no_profile_is_recorded(
    tmp_path, presets_dir, monkeypatch
):
    """Same installed version, changed preset — the --dev checkout incident.

    The preset branch is the fallback now: it applies to a manifest carrying no
    profile path at all (one written before profile-always builds, or one whose
    profile path could not be recorded). A current build records the profile it
    rendered from and is judged by that instead — see
    :func:`test_an_edited_profile_reports_the_render_stale`.
    """
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


def test_an_edited_profile_reports_the_render_stale(tmp_path, monkeypatch):
    """The edit the advisory most needs to see, on a real deployment repo.

    A build renders ``build/`` from the repo's ``profile.yml``, so editing that
    profile is how a facility changes its deployment — and the render is stale
    from that moment. Judging the render by the bundled preset instead would
    stay silent on exactly this change, since the preset never moved.

    Materialized and built for real rather than hand-stamped: the claim is that
    the hash the build writes and the hash the deploy recomputes describe the
    same source, which a synthesized manifest could not show.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path / "proj"
    result = CliRunner().invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    assert staleness.staleness_reasons(repo / "build") == []

    profile_path = repo / "profile.yml"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8") + "\ndeploy_services: false\n",
        encoding="utf-8",
    )

    reasons = staleness.staleness_reasons(repo / "build")
    assert len(reasons) == 1
    assert "profile" in reasons[0]
    assert str(profile_path) in reasons[0]


def _write_profile_manifest(project_dir, build_args, preset_hash):
    """Write a manifest for a *profile*-sourced build.

    Not ``_write_manifest``: that one merges into a preset-sourced default, and a
    leftover ``preset`` key would route the content check down the preset branch
    instead of the profile branch under test.
    """
    (project_dir / ".osprey-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "creation": {"osprey_version": "2026.7.0", "preset_hash": preset_hash},
                "build_args": {"project_name": "proj", "source": "profile", **build_args},
                "reproducible_command": "osprey build",
            }
        ),
        encoding="utf-8",
    )


def _write_profile_project(tmp_path, *, absolute: bool):
    """A project built from a positional profile, deployed from its own directory.

    Mirrors the real shape: the profile and its ``data:`` tree live wherever the
    facility keeps them, the user typed a path relative to *that* directory, and
    a later ``osprey up`` runs from the repo root instead.

    Args:
        absolute: Whether the manifest also carries ``profile_path_abs`` — the
            difference between a manifest this build writes and a legacy one.
    """
    facility = tmp_path / "facility"
    (facility / "data").mkdir(parents=True)
    (facility / "data" / "channels.json").write_text('{"channels": []}\n', encoding="utf-8")
    profile = facility / "profile.yml"
    profile.write_text("name: Facility\ndata: data\n", encoding="utf-8")

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    build_args = {"profile_path": "profile.yml"}
    if absolute:
        build_args["profile_path_abs"] = str(profile)
    _write_profile_manifest(project_dir, build_args, build_profile.compute_profile_hash(profile))
    return profile, project_dir


def test_relative_profile_build_still_sees_a_data_edit(tmp_path, monkeypatch):
    """SC8: the advisory fires from the deploy directory, not just the build one.

    The relative ``profile.yml`` in the manifest names nothing from the project
    directory. Following it alone hashes to "cannot compare" and the project
    deploys silently against data that has since changed.
    """
    profile, project_dir = _write_profile_project(tmp_path, absolute=True)
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    monkeypatch.chdir(project_dir)

    assert staleness.staleness_reasons(project_dir) == []

    (profile.parent / "data" / "channels.json").write_text('{"channels": [1]}\n', encoding="utf-8")
    reasons = staleness.staleness_reasons(project_dir)
    assert len(reasons) == 1
    assert "changed since this project was rendered" in reasons[0]


def test_legacy_manifest_without_absolute_path_falls_back(tmp_path, monkeypatch):
    """Manifests written before ``profile_path_abs`` keep their old behavior.

    From the directory the build ran in the relative string still resolves, so
    the fallback is a real comparison rather than a permanent blind spot.
    """
    profile, project_dir = _write_profile_project(tmp_path, absolute=False)
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    monkeypatch.chdir(profile.parent)

    assert staleness.staleness_reasons(project_dir) == []

    (profile.parent / "data" / "channels.json").write_text('{"channels": [1]}\n', encoding="utf-8")
    assert len(staleness.staleness_reasons(project_dir)) == 1


def test_unresolvable_profile_path_is_silent(tmp_path, monkeypatch):
    """A profile that has moved away is "cannot compare", never drift."""
    monkeypatch.setattr(staleness, "_installed_version", lambda: "2026.7.0")
    _write_profile_manifest(
        tmp_path,
        {
            "profile_path": "profile.yml",
            "profile_path_abs": str(tmp_path / "gone" / "profile.yml"),
        },
        "sha256:deadbeef",
    )
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
    # The manifest's own command, printed verbatim — no `--force` appended (it
    # went with the legacy build surface) and no retired verb in the follow-up.
    #
    # The retired spelling on the last line is assertion DATA: it is the string
    # that must NOT appear. A sweep that rewrites verb names across the tree has
    # to skip it, or the pair collapses into `X in text` and `X not in text` and
    # the test can never pass.
    assert "osprey build" in text
    assert "--force" not in text
    assert "osprey up" in text
    assert "osprey deploy up" not in text


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
