"""The preset-drift lint — a materialized profile compared with its preset.

``osprey init`` stamps ``provenance:`` into the profile it writes and nothing
read it back: a profile could lack a line the preset gained, and every build
stayed green. These tests pin the comparison
(:func:`~osprey.cli.build_profile_drift.preset_drift_report`) and what the two
verbs print of it: ``osprey validate`` and ``osprey build`` say only when the
preset has moved on, and refuse and list no difference. Which sort a finding is
is :attr:`~osprey.cli.build_profile_drift.DriftFinding.kind`;
tests/cli/test_preset_edits_never_refuse.py runs each kind through validate.

The fixture is a real ``osprey init`` of the ``demo`` preset,
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
from osprey.cli.set_cmd import set as set_cmd
from osprey.cli.validate_cmd import validate
from osprey.utils.workspace import TIERS_DIRNAME

PROFILE = "profile.yml"
READONLY = Path(TIERS_DIRNAME) / "readonly.yml"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization of the demo preset."""
    root = tmp_path_factory.mktemp("materialized") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "demo", "--no-git"])
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


def _structural(report: DriftReport) -> list:
    """The findings where one document has a key, block or member the other has not."""
    return [f for f in report.findings if f.kind != "value"]


def _values(report: DriftReport) -> list:
    """The findings where both documents carry a key with different values."""
    return [f for f in report.findings if f.kind == "value"]


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

    assert report.findings == []
    assert report.note is None


def test_header_stamp_and_lint_share_one_hash(repo: Path) -> None:
    """The header comment, the `provenance:` key and the lint all read
    :func:`compute_preset_hash`, which is why a fresh materialization cannot
    report the preset as moved."""
    profile = repo / PROFILE
    header_hash = _header_hash(profile)

    assert header_hash == compute_preset_hash("demo")
    assert f"preset_hash: {header_hash}" in profile.read_text(encoding="utf-8")
    tier_hash = _header_hash(repo / READONLY)
    assert tier_hash == compute_preset_hash(f"{TIERS_DIRNAME}/readonly")
    assert f"preset_hash: {tier_hash}" in (repo / READONLY).read_text(encoding="utf-8")


#: The header line naming the hash: ``profile.yml`` stamps its preset's, a
#: tier or app file the hash of the bundled file it was made from.
_HEADER_HASH_PREFIXES = ("#   preset content hash: ", "#   content hash: ")


def _header_hash(path: Path) -> str:
    """The hash the emitted header comment names."""
    for line in path.read_text(encoding="utf-8").splitlines():
        for prefix in _HEADER_HASH_PREFIXES:
            if line.startswith(prefix):
                return line.removeprefix(prefix).strip()
    raise AssertionError(f"{path} has no emitted header")


def test_fresh_hello_world_reports_nothing(tmp_path: Path) -> None:
    root = tmp_path / "hello"
    result = CliRunner().invoke(init, [str(root), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output

    report = _report(root / PROFILE)

    assert report.findings == []
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

    assert [f.subject for f in _structural(report)] == ["web_panels: system-health"]
    assert report.findings[0].kind == "missing_member"
    rendered = report.findings[0].render()
    assert "profile.yml" in rendered
    preset_line = _line_of(_presets_dir() / "demo.yml", "- system-health")
    assert f"demo.yml:{preset_line}" in rendered


def test_scalar_that_differs_names_both_values_and_both_lines(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  web.theme: light", "  web.theme: dark")

    report = _report(profile)

    assert [f.subject for f in _values(report)] == ["config.web.theme"]
    assert _structural(report) == []
    assert report.findings[0].kind == "value"
    rendered = report.findings[0].render()
    assert "'dark'" in rendered and "'light'" in rendered
    assert f"profile.yml:{_line_of(profile, 'web.theme: dark')}" in rendered
    preset_line = _line_of(_presets_dir() / "demo.yml", "web.theme: light")
    assert f"demo.yml:{preset_line}" in rendered


def test_member_the_profile_adds(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  - system-health", "  - system-health\n  - artifacts")

    report = _report(profile)

    assert [f.subject for f in _structural(report)] == ["web_panels: artifacts"]
    assert report.findings[0].kind == "extra_member"


def test_config_knob_a_facility_turns_is_a_note_not_a_refusal(repo: Path) -> None:
    """The preset carries the whole configuration, so a facility changing a
    documented value differs from it on a key both documents have — the kind of
    difference the emitted profile invites, and the one that does not refuse."""
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.channel_suggestions.max_channels: 50000",
        "  web.channel_suggestions.max_channels: 100000",
    )

    report = _report(profile)

    assert [f.subject for f in _values(report)] == ["config.web.channel_suggestions.max_channels"]
    assert _structural(report) == []


def test_config_key_nothing_knows_is_reported_once_at_its_root(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(
        profile,
        "  web.theme: light",
        "  web.theme: light\n  web.no_such_block:\n    depth: 1\n    other: 2",
    )

    report = _report(profile)

    assert [f.subject for f in _structural(report)] == ["config.web.no_such_block"]
    assert report.findings[0].kind == "extra"


def test_top_level_block_the_profile_dropped(repo: Path) -> None:
    profile = repo / PROFILE
    text = profile.read_text(encoding="utf-8")
    start = text.index("\nbluesky:\n")
    end = text.index("\n", text.index("tiled_enabled", start))
    profile.write_text(text[:start] + text[end:], encoding="utf-8")

    report = _report(profile)

    assert [f.subject for f in _structural(report)] == ["bluesky"]
    assert report.findings[0].kind == "missing"


# ---------------------------------------------------------------------------
# The hash note
# ---------------------------------------------------------------------------


def test_moved_preset_hash_is_noted_with_both_versions(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  preset_hash: sha256:", "  preset_hash: sha256:0000dead")

    report = _report(profile)

    assert report.note is not None
    assert "sha256:0000dead" in report.note
    assert "demo" in report.note
    assert "OSPREY" in report.note
    assert report.findings == []


def test_unbundled_preset_cannot_be_compared(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  preset: demo\n", "  preset: retired-preset\n")

    report = _report(profile)

    assert report.findings == []
    assert report.note is not None and "retired-preset" in report.note


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


def test_layer_delta_is_compared_with_its_own_preset(repo: Path) -> None:
    delta = repo / READONLY
    _edit(delta, "  control_system.writes: never\n", "")

    report = _report(repo / PROFILE)

    assert [f.subject for f in report.findings] == ["config.control_system"]
    rendered = report.findings[0].render()
    assert f"{TIERS_DIRNAME}/readonly.yml" in rendered
    preset_line = _line_of(
        _presets_dir() / TIERS_DIRNAME / "readonly.yml", "  control_system.writes: never"
    )
    assert f"readonly.yml:{preset_line}" in rendered


def test_validating_a_delta_reports_that_render_only(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")
    _edit(repo / READONLY, "  control_system.writes: never\n", "")

    report = _report(repo / READONLY)

    assert [f.subject for f in report.findings] == ["config.control_system"]


def test_profile_exclude_of_something_the_preset_keeps(repo: Path) -> None:
    """A tier only adds, so a subtraction is spelled in ``profile.yml`` — and is a finding."""
    profile = repo / PROFILE
    with profile.open("a", encoding="utf-8") as fh:
        fh.write("\nexclude:\n  skills:\n    - diagnose\n")

    report = _report(profile)

    assert [f.subject for f in _structural(report)] == ["skills: diagnose"]
    assert report.findings[0].kind == "exclude"
    assert f"{PROFILE}:{_line_of(profile, '    - diagnose')}" in report.findings[0].render()


def test_a_tier_from_a_source_osprey_does_not_bundle_is_noted_once(repo: Path) -> None:
    """A tier whose provenance names no bundled tier has nothing to compare with.

    It is noted once, by file and name, and contributes no finding.
    """
    (repo / TIERS_DIRNAME / "operator.yml").write_text(
        "name: Operator\nprovenance:\n  preset: operator\n  preset_hash: sha256:00\n",
        encoding="utf-8",
    )

    report = _report(repo / PROFILE)

    assert report.findings == []
    assert report.note is not None
    assert report.note.count(f"{TIERS_DIRNAME}/operator.yml") == 1
    assert "'operator'" in report.note


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "named"),
    [
        ("  - system-health", "  # - system-health", "system-health"),
        ("  web.theme: light\n", "", "config.web.theme"),
        ("  web.theme: light", "  web.theme: dark", "config.web.theme"),
    ],
    ids=["commented-out-entry", "dropped-key", "changed-value"],
)
def test_validate_neither_refuses_nor_lists_an_edit(
    repo: Path, before: str, after: str, named: str
) -> None:
    """The profile is the operator's: an edit is not a fault, so the verb
    that gates CI passes it, and says nothing about it."""
    _edit(repo / PROFILE, before, after)

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output
    assert named not in result.output
    assert "DEVIATION" not in result.output


def test_set_knob_then_validate_green(repo: Path) -> None:
    """Turning a documented knob with `osprey set` leaves the repo valid."""
    setting = CliRunner().invoke(set_cmd, ["--repo", str(repo), "config.web.theme=dark"])
    assert setting.exit_code == 0, setting.output

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_validate_passes_a_fresh_materialization(repo: Path) -> None:
    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "drift" not in result.output.lower()


def test_validate_prints_the_hash_note(repo: Path) -> None:
    profile = repo / PROFILE
    _edit(profile, "  preset_hash: sha256:", "  preset_hash: sha256:0000dead")

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "sha256:0000dead" in result.output
    assert "has moved on" in result.output


def test_build_never_lists_an_edit(repo: Path) -> None:
    _edit(repo / PROFILE, "  - system-health", "  # - system-health")
    args = ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]

    for verbosity in ([], ["-v"]):
        result = CliRunner().invoke(cli, [*verbosity, *args])
        assert result.exit_code == 0, result.output
        assert "web_panels: system-health" not in result.output


# ---------------------------------------------------------------------------
# What the repo owns rather than what the preset carries
# ---------------------------------------------------------------------------


def test_two_checkouts_of_one_profile_report_the_same_thing(
    materialized: Path, tmp_path: Path
) -> None:
    """A render's render name is pinned in the profile and read from there by
    everything that builds or mounts it, so the lint's answer is a property of
    the tracked document rather than of the directory holding it."""
    first = tmp_path / "assistant"
    second = tmp_path / "assistant-wt"
    shutil.copytree(materialized, first)
    shutil.copytree(materialized, second)

    assert _report(first / PROFILE).findings == []
    assert _report(second / PROFILE).findings == []
    assert [f.render() for f in _report(first / PROFILE).findings] == [
        f.render() for f in _report(second / PROFILE).findings
    ]

    for copy in (first, second):
        _edit(copy / PROFILE, "  web.theme: light", "  web.theme: dark")

    assert [f.subject for f in _report(first / PROFILE).findings] == ["config.web.theme"]
    assert [f.subject for f in _report(second / PROFILE).findings] == ["config.web.theme"]


def test_a_deployment_name_of_its_own_is_not_preset_content(repo: Path) -> None:
    """``project_name`` names this deployment on its hosts; no preset carries one."""
    profile = repo / PROFILE
    _edit(profile, f"project_name: {repo.name}", "project_name: elsewhere")

    assert _report(profile).findings == []
