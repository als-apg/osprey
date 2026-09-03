"""The preset-drift lint — a materialized profile compared with its preset.

``osprey init`` stamps ``provenance:`` into the profile it writes and nothing
read it back: a profile could lack a line the preset gained, and every build
stayed green. These tests pin the comparison
(:func:`~osprey.cli.build_profile_drift.preset_drift_report`) and the two
surfaces that print it — ``osprey validate`` refuses unmarked differences,
``osprey build`` names them under ``-v``.

The fixture is a real ``osprey init`` of the ``control-assistant`` preset,
made once per module and copied per test: the first assertion of the suite is
that a fresh materialization reports nothing, and every other test edits that
repo the way an operator would.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_drift import DriftReport, preset_drift_report
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.build_profile_presets import _presets_dir
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.validate_cmd import validate

PROFILE = "profile.yml"
READONLY = Path("personas") / "readonly.yml"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization of the control-assistant preset."""
    root = tmp_path_factory.mktemp("materialized") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    """A private copy of the materialization for this test to edit."""
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _report(profile_file: Path) -> DriftReport:
    """Run the lint the way the verbs do: the provenance the loader parsed."""
    root = profile_file if profile_file.name == PROFILE else profile_file.parent.parent / PROFILE
    profile, _ = resolve_build_profile(root, None)
    assert profile.provenance is not None
    return preset_drift_report(profile_file, profile.provenance)


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must occur exactly once in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _line_of(path: Path, needle: str) -> int:
    """1-based line number of the one line containing *needle*."""
    hits = [
        i
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if needle in line
    ]
    assert len(hits) == 1, f"{needle!r} must occur exactly once in {path}"
    return hits[0]


# ---------------------------------------------------------------------------
# The baseline: what init writes today does not drift
# ---------------------------------------------------------------------------


def test_fresh_materialization_reports_nothing(repo: Path) -> None:
    report = _report(repo / PROFILE)

    assert report.unmarked == []
    assert report.stale_markers == []
    assert report.note is None


def test_header_stamp_and_lint_share_one_hash(repo: Path) -> None:
    """The header comment, the `provenance:` key and the lint all read
    :func:`compute_preset_hash`, which is why a fresh materialization cannot
    report the preset as moved."""
    profile = repo / PROFILE
    header_hash = _header_hash(profile)

    assert header_hash == compute_preset_hash("control-assistant")
    assert f"preset_hash: {header_hash}" in profile.read_text(encoding="utf-8")
    assert _header_hash(repo / READONLY) == compute_preset_hash("control-assistant-readonly")


def _header_hash(path: Path) -> str:
    """The hash the emitted header comment names."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#   preset content hash: "):
            return line.removeprefix("#   preset content hash: ").strip()
    raise AssertionError(f"{path} has no emitted header")


def test_fresh_hello_world_reports_nothing(tmp_path: Path) -> None:
    root = tmp_path / "hello"
    result = CliRunner().invoke(init, [str(root), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output

    report = _report(root / PROFILE)

    assert report.unmarked == []
    assert report.note is None


# ---------------------------------------------------------------------------
# What is reported
# ---------------------------------------------------------------------------


def test_member_the_preset_selects_and_the_profile_lacks(repo: Path) -> None:
    """The failure that motivated the lint: a `web_panels` entry the preset
    gained and the materialized copy never picked up."""
    profile = repo / PROFILE
    _edit(profile, "  - system-health", "  # - system-health")

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["web_panels: system-health"]
    rendered = report.unmarked[0].render()
    assert "profile.yml" in rendered
    preset_line = _line_of(_presets_dir() / "control-assistant.yml", "- system-health")
    assert f"control-assistant.yml:{preset_line}" in rendered


def test_scalar_that_differs_names_both_values_and_both_lines(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  web.theme: light", "  web.theme: dark")

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["config.web.theme"]
    rendered = report.unmarked[0].render()
    assert "'dark'" in rendered and "'light'" in rendered
    assert f"profile.yml:{_line_of(profile, 'web.theme: dark')}" in rendered
    preset_line = _line_of(_presets_dir() / "control-assistant.yml", "web.theme: light")
    assert f"control-assistant.yml:{preset_line}" in rendered


def test_member_the_profile_adds(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  - system-health", "  - system-health\n  - artifacts")

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["web_panels: artifacts"]


def test_config_key_the_app_template_knows_is_not_drift(repo: Path) -> None:
    """A facility tuning a template knob the preset leaves alone is ordinary
    business, not a deviation from the preset."""
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  web.theme: light\n  web.docs_url: https://docs.facility.example",
    )

    assert _report(profile).unmarked == []


def test_config_key_nothing_knows_is_reported_once_at_its_root(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  web.theme: light\n  web.no_such_block:\n    depth: 1\n    other: 2",
    )

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["config.web.no_such_block"]


def test_top_level_block_the_profile_dropped(repo: Path) -> None:
    profile = repo / PROFILE
    text = profile.read_text(encoding="utf-8")
    start = text.index("\nbluesky:\n")
    end = text.index("\n", text.index("tiled_enabled", start))
    profile.write_text(text[:start] + text[end:], encoding="utf-8")

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["bluesky"]


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def test_marker_within_three_lines_above_silences_the_line(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  # DEVIATION: facility — the control room is dark\n  #\n  #\n  web.theme: dark",
    )

    report = _report(profile)

    assert report.unmarked == []
    assert [f.subject for f in report.findings] == ["config.web.theme"]
    assert report.stale_markers == []


def test_marker_four_lines_above_does_not_reach(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  # DEVIATION: facility — too far up\n  #\n  #\n  #\n  web.theme: dark",
    )

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == ["config.web.theme"]
    assert len(report.stale_markers) == 1


def test_absence_is_silenced_by_a_marker_that_names_it(repo: Path) -> None:
    """A line the profile lacks has nothing to write a marker above, so the
    marker names what is missing instead — anywhere in the file."""
    profile = repo / PROFILE
    _edit(profile, "  - system-health   # SYSTEM tab, a framework health dashboard\n", "")
    with profile.open("a", encoding="utf-8") as fh:
        fh.write("\n# DEVIATION: facility — no system-health tab; the SYSTEM view is elsewhere\n")

    report = _report(profile)

    assert report.unmarked == []
    assert report.stale_markers == []


def test_dropped_block_is_silenced_by_naming_it(repo: Path) -> None:
    profile = repo / PROFILE
    text = profile.read_text(encoding="utf-8")
    start = text.index("\nbluesky:\n")
    end = text.index("\n", text.index("tiled_enabled", start))
    text = text[:start] + "\n# DEVIATION: facility — bluesky is not run here" + text[end:]
    profile.write_text(text, encoding="utf-8")

    assert _report(profile).unmarked == []


def test_marker_tag_is_configurable(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  preset: control-assistant\n",
        "  preset: control-assistant\n  deviation_marker: ALS-DEVIATION\n",
    )
    _edit(
        profile,
        "  web.theme: light",
        "  # DEVIATION: facility — wrong tag\n  web.theme: dark",
    )

    assert [f.subject for f in _report(profile).unmarked] == ["config.web.theme"]

    _edit(profile, "  # DEVIATION: facility — wrong tag", "  # ALS-DEVIATION: facility — right tag")

    assert _report(profile).unmarked == []


def test_stale_marker_is_reported(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  # DEVIATION: facility — nothing differs here\n  web.theme: light",
    )

    report = _report(profile)

    assert report.unmarked == []
    assert len(report.stale_markers) == 1
    assert f"profile.yml:{_line_of(profile, 'nothing differs here')}" in report.stale_markers[0]


# ---------------------------------------------------------------------------
# The hash note
# ---------------------------------------------------------------------------


def test_moved_preset_hash_is_noted_with_both_versions(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  preset_hash: sha256:", "  preset_hash: sha256:0000dead")

    report = _report(profile)

    assert report.note is not None
    assert "sha256:0000dead" in report.note
    assert "control-assistant" in report.note
    assert "OSPREY" in report.note
    assert report.unmarked == []


def test_unbundled_preset_cannot_be_compared(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  preset: control-assistant\n", "  preset: retired-preset\n")

    report = _report(profile)

    assert report.findings == []
    assert report.note is not None and "retired-preset" in report.note


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


def test_persona_delta_is_compared_with_its_own_preset(repo: Path) -> None:
    delta = repo / READONLY
    _edit(delta, "  web.ui_mode: simple\n", "")

    report = _report(repo / PROFILE)

    assert [f.subject for f in report.unmarked] == ["config.web.ui_mode"]
    rendered = report.unmarked[0].render()
    assert "personas/readonly.yml" in rendered
    preset_line = _line_of(_presets_dir() / "control-assistant-readonly.yml", "web.ui_mode: simple")
    assert f"control-assistant-readonly.yml:{preset_line}" in rendered


def test_validating_a_delta_reports_that_persona_only(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")
    _edit(repo / READONLY, "  web.ui_mode: simple\n", "")

    report = _report(repo / READONLY)

    assert [f.subject for f in report.unmarked] == ["config.web.ui_mode"]


def test_persona_exclude_of_something_the_preset_keeps(repo: Path) -> None:
    delta = repo / READONLY
    with delta.open("a", encoding="utf-8") as fh:
        fh.write("\nexclude:\n  skills:\n    - diagnose\n")

    report = _report(repo / PROFILE)

    assert [f.subject for f in report.unmarked] == ["skills: diagnose"]
    assert f"personas/readonly.yml:{_line_of(delta, '- diagnose')}" in report.unmarked[0].render()

    _edit(
        delta,
        "    - diagnose",
        "    # DEVIATION: facility — read-only logins do not diagnose\n    - diagnose",
    )

    assert _report(repo / PROFILE).unmarked == []


def test_persona_the_preset_does_not_know_is_reported_once(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "      readonly:\n",
        "      operator:\n        project: demo-operator\n        project_path: build/demo-operator\n"
        "        build_profile: personas/operator.yml\n      readonly:\n",
    )
    (repo / "personas" / "operator.yml").write_text(
        "name: Operator\ndeploy_services: false\n", encoding="utf-8"
    )

    report = _report(profile)

    assert [f.subject for f in report.unmarked] == [
        "config.modules.web_terminals.personas.operator"
    ]


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------


def test_validate_refuses_unmarked_drift(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 2, result.output
    assert "web_panels: system-health" in result.output
    assert "# DEVIATION:" in result.output


def test_validate_drift_warn_demotes_to_a_warning(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")

    result = CliRunner().invoke(validate, ["--repo", str(repo), "--drift=warn"])

    assert result.exit_code == 0, result.output
    assert "web_panels: system-health" in result.output
    assert "Profile is valid" in result.output


def test_validate_passes_a_fresh_materialization(repo: Path) -> None:
    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "drift" not in result.output.lower()


def test_validate_prints_stale_markers_and_the_hash_note(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  # DEVIATION: facility — nothing differs here\n  web.theme: light",
    )
    _edit(profile, "  preset_hash: sha256:", "  preset_hash: sha256:0000dead")

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "marks no difference" in result.output
    assert "sha256:0000dead" in result.output


def test_build_names_drift_under_verbose_only(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")
    args = ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]

    quiet = CliRunner().invoke(cli, args)
    assert quiet.exit_code == 0, quiet.output
    assert "system-health" not in quiet.output

    verbose = CliRunner().invoke(cli, ["-v", *args])
    assert verbose.exit_code == 0, verbose.output
    assert "web_panels: system-health" in verbose.output
