"""No difference from the preset refuses, and the query behind `--revalidate`.

The profile `osprey init` writes is the operator's to edit. A key, a block or a
list member one document has and the other has not, and a key both carry with
different values, are all edits a facility may make on purpose. A comparison
with today's preset alone cannot tell such an edit from a line the preset gained
since, so ``osprey validate`` refuses none of them and lists none of them: it
says only when the preset has moved on.

The same comparison answers a second question, for ``osprey init --revalidate``:
:func:`~osprey.cli.build_profile_drift.lacking_config_keys` — not which leaves
differ but which the profile has not got, flattened to one dotted key each.

The per-finding shape of the comparison lives in tests/cli/test_preset_drift.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_drift import (
    DriftReport,
    lacking_config_keys,
    preset_drift_report,
)
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate

PROFILE = "profile.yml"
PRESET = "demo"

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


# ── no edit refuses ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("old", "new", "kind", "named"),
    [
        (f"{THEME}\n", "", "missing", "config.web.theme"),
        (PANEL, f"  # {PANEL.strip()}", "missing_member", "system-health"),
        (THEME, f"{THEME}\n  web.no_such_knob: 1", "extra", "web.no_such_knob"),
        (THEME, "  web.theme: dark", "value", "config.web.theme"),
    ],
    ids=["a-key-the-profile-lost", "a-member-the-preset-selects", "an-extra-key", "a-value"],
)
def test_an_edit_is_found_but_neither_refused_nor_listed(
    repo: Path, old: str, new: str, kind: str, named: str
) -> None:
    """Each kind of difference is still found, so the comparison works, but
    validate passes and does not print it: the profile is the operator's."""
    _edit(repo, old, new)

    assert [f.kind for f in _report(repo).findings] == [kind]

    result = CliRunner().invoke(validate, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output
    assert named not in result.output
    assert "DEVIATION" not in result.output


def test_drift_is_not_an_option_any_more(repo: Path) -> None:
    result = CliRunner().invoke(validate, ["--repo", str(repo), "--drift=warn"])

    assert result.exit_code == 2
    assert "No such option" in result.output


# ── the leaves `osprey init --revalidate` writes ─────────────────────────────


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
