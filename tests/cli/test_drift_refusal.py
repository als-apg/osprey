"""Which drift findings refuse, which only report, and the query behind `expand`.

The preset carries the whole configuration a deployment renders, so the profile
`osprey init` writes differs from it in two very different ways. A key, a block
or a list member one document has and the other has not is the silent failure
the lint was written for: it builds green forever and nobody is told. A key both
documents carry with different values is the opposite — it is the operator doing
what the emitted header invites, and refusing it would mean a marker comment for
every knob a facility turns.

So `preset_drift_report` labels each finding with a
:attr:`~osprey.cli.build_profile_drift.DriftFinding.kind`, and one policy maps
kinds to costs: :data:`~osprey.cli.build_profile_drift.REFUSAL_KINDS` refuse
under ``--drift=error``, :data:`~osprey.cli.build_profile_drift.NOTE_KINDS`
never do. The accepted cost of that ruling is that a preset changing a default
in a newer OSPREY arrives as a note rather than as a refusal.

The same comparison answers a second question, for ``osprey profile expand``:
:func:`~osprey.cli.build_profile_drift.lacking_config_keys` — not which leaves
differ but which the profile has not got, flattened to one dotted key each.

The per-finding shape of the comparison lives in tests/cli/test_preset_drift.py;
what this file pins is the policy over it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_drift import (
    NOTE_KINDS,
    REFUSAL_KINDS,
    DriftReport,
    lacking_config_keys,
    preset_drift_report,
)
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate

PROFILE = "profile.yml"
PRESET = "control-assistant"

#: A documented value the preset carries and a facility plausibly changes.
THEME = "  web.theme: light"
#: A list member the preset selects.
PANEL = "  - system-health"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization to edit copies of."""
    root = tmp_path_factory.mktemp("drift-policy") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", PRESET, "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _report(repo: Path) -> DriftReport:
    profile, _ = resolve_build_profile(repo / PROFILE, None)
    assert profile.provenance is not None
    return preset_drift_report(repo / PROFILE, profile.provenance)


def _edit(repo: Path, old: str, new: str) -> None:
    path = repo / PROFILE
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must occur exactly once in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


# ── the policy is total ──────────────────────────────────────────────────────


def test_every_kind_has_exactly_one_cost() -> None:
    """A kind in neither set would be found and then silently discarded."""
    assert REFUSAL_KINDS & NOTE_KINDS == set()
    assert REFUSAL_KINDS | NOTE_KINDS == {
        "value",
        "missing",
        "missing_member",
        "extra",
        "extra_member",
        "exclude",
    }


def test_refusals_and_notes_partition_the_unclaimed(repo: Path) -> None:
    """A surface printing both prints everything, and neither drops a finding."""
    _edit(repo, PANEL, f"  # {PANEL.strip()}")
    _edit(repo, THEME, "  web.theme: dark")

    report = _report(repo)

    assert [f.kind for f in report.refusals] == ["missing_member"]
    assert [f.kind for f in report.notes] == ["value"]
    assert len(report.refusals) + len(report.notes) == len(report.unmarked)
    assert set(report.refusals) | set(report.notes) == set(report.unmarked)


# ── what refuses ─────────────────────────────────────────────────────────────


def test_a_key_the_profile_lost_refuses(repo: Path) -> None:
    """The silence the lint exists for: a line the preset has and this copy
    does not, which every build would otherwise render past without a word."""
    _edit(repo, f"{THEME}\n", "")

    report = _report(repo)
    assert [f.kind for f in report.refusals] == ["missing"]

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 2, result.output
    assert "config.web.theme" in result.output


def test_a_member_the_preset_selects_refuses(repo: Path) -> None:
    _edit(repo, PANEL, f"  # {PANEL.strip()}")

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 2, result.output
    assert "system-health" in result.output


def test_a_key_the_preset_does_not_carry_refuses(repo: Path) -> None:
    _edit(repo, THEME, f"{THEME}\n  web.no_such_knob: 1")

    report = _report(repo)
    assert [f.kind for f in report.refusals] == ["extra"]

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 2, result.output
    assert "web.no_such_knob" in result.output


def test_a_refusal_still_passes_under_drift_warn(repo: Path) -> None:
    """`--drift=warn` is the escape hatch for the refusing kinds, unchanged."""
    _edit(repo, f"{THEME}\n", "")

    result = CliRunner().invoke(validate, ["--repo", str(repo), "--drift=warn"])

    assert result.exit_code == 0, result.output
    assert "config.web.theme" in result.output


# ── what only reports ────────────────────────────────────────────────────────


def test_a_changed_value_is_reported_and_passes(repo: Path) -> None:
    _edit(repo, THEME, "  web.theme: dark")

    report = _report(repo)
    assert [f.kind for f in report.notes] == ["value"]
    assert report.refusals == []

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "preset drift: config.web.theme" in result.output
    assert "Profile is valid" in result.output


def test_a_changed_value_prints_under_drift_error_too(repo: Path) -> None:
    """The note is not a `--drift=warn` feature: the refusing spelling prints it
    as well, or an operator would only learn of it by relaxing the gate."""
    _edit(repo, THEME, "  web.theme: dark")

    result = CliRunner().invoke(validate, ["--repo", str(repo), "--drift=error"])

    assert result.exit_code == 0, result.output
    assert "preset drift: config.web.theme" in result.output


def test_a_marker_still_silences_a_note(repo: Path) -> None:
    _edit(repo, THEME, "  # DEVIATION: facility — the control room is dark\n  web.theme: dark")

    report = _report(repo)

    assert report.notes == []
    assert report.stale_markers == []


# ── the leaves `osprey profile expand` writes ────────────────────────────────


def test_a_fresh_materialization_lacks_nothing(repo: Path) -> None:
    raw = yaml.safe_load((repo / PROFILE).read_text(encoding="utf-8"))

    assert lacking_config_keys(raw, PRESET) == []


def test_a_dropped_branch_comes_back_as_its_leaves() -> None:
    """What expand needs: one dotted key per leaf, never the branch itself, so
    every key it writes can carry the preset's comment for that key."""
    lacking = lacking_config_keys({"name": "demo", "config": {}}, PRESET)

    assert lacking, "the preset carries a config: block to be lacking"
    assert not any(
        other != key and other.startswith(f"{key}.") for key in lacking for other in lacking
    ), "a branch and its leaves were both reported"
    assert any(key.count(".") > 1 for key in lacking), "a deep branch was not flattened"


def _nested(parts: list[str]) -> dict[str, object]:
    """``a.b.c`` written out whole: ``{"a": {"b": {"c": ...}}}``."""
    node: object = "x"
    for part in reversed(parts):
        node = {part: node}
    return node  # type: ignore[return-value]


def _split(parts: list[str]) -> dict[str, object]:
    """``a.b.c`` written the middle way: ``{"a": {"b.c": ...}}``."""
    head, *rest = parts
    return {head: {".".join(rest): "x"}}


@pytest.mark.parametrize(
    ("dots", "spell"),
    [(1, _nested), (2, _nested), (2, _split)],
    ids=["nested-pair", "nested-triple", "split-triple"],
)
def test_any_split_between_dotted_and_nested_is_not_lacking(dots: int, spell) -> None:
    """Every spelling addresses one rendered key, so a profile that nests what
    the preset dots — at any depth, including the mix of the two the emitted
    config invites — is not missing it. Expand must not write it twice."""
    dotted = lacking_config_keys({"name": "demo", "config": {}}, PRESET)
    key = next(k for k in dotted if k.count(".") == dots)

    nested = lacking_config_keys({"name": "demo", "config": spell(key.split("."))}, PRESET)

    assert key not in nested
    assert set(nested) < set(dotted)
