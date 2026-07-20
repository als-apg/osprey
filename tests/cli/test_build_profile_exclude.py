"""Tests for the ``exclude:`` profile key (list subtraction for ``extends``).

``exclude`` lets a child profile remove entries that an ``extends`` base
contributed to a string-list field (skills, rules, hooks, agents,
output_styles, web_panels, dependencies). It is applied inside
``_resolve_extends`` after each ``_deep_merge`` and consumed there, so:

* a child can drop an inherited entry;
* a *deeper* extends layer that re-adds the entry merges in afterwards and wins;
* an override file / ``--set`` re-add merges *before* extends resolution and is
  stripped again, so it cannot win.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.cli.build_profile import (
    _apply_exclude,
    resolve_build_profile,
)
from osprey.errors import BuildProfileError


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# (a) child excludes an inherited skill
# ---------------------------------------------------------------------------


def test_child_excludes_inherited_skill(tmp_path: Path) -> None:
    """A child ``exclude`` removes an entry contributed by its ``extends`` base."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [alpha, beta, gamma]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [beta]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.skills == ["alpha", "gamma"]
    # ``exclude`` is consumed during resolution and never surfaces on the model.
    assert not hasattr(resolved, "exclude")


def test_exclude_works_across_multiple_fields(tmp_path: Path) -> None:
    """Every declared excludable field is subtracted independently."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\n"
        "skills: [s1, s2]\n"
        "rules: [r1, r2]\n"
        "dependencies: [pkg-a, pkg-b]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\n"
        "exclude:\n  skills: [s2]\n  rules: [r1]\n  dependencies: [pkg-a]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.skills == ["s1"]
    assert resolved.rules == ["r2"]
    assert resolved.dependencies == ["pkg-b"]


# ---------------------------------------------------------------------------
# (b) a deeper extends layer re-adds an excluded entry and wins
# ---------------------------------------------------------------------------


def test_deeper_layer_readds_excluded_entry_and_wins(tmp_path: Path) -> None:
    """base [a,b,c] → parent excludes b → child re-adds b: b survives."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a, b, c]\n",
    )
    _write(
        tmp_path / "parent.yml",
        "extends: ./base.yml\nname: Parent\nexclude:\n  skills: [b]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./parent.yml\nname: Child\nskills: [b]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    # Parent removed b, but the child re-declared it after the exclusion applied.
    assert "b" in resolved.skills
    assert "a" in resolved.skills
    assert "c" in resolved.skills


# ---------------------------------------------------------------------------
# (c) an override file / --set re-add does NOT win
# ---------------------------------------------------------------------------


def test_override_file_readd_does_not_win(tmp_path: Path) -> None:
    """An override file merges into the top layer pre-exclusion; its re-add is stripped."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a, b, c]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )
    override = _write(tmp_path / "override.yml", "skills: [b]\n")
    resolved, _ = resolve_build_profile(
        child.resolve(), preset=None, overrides=(override.resolve(),)
    )
    assert "b" not in resolved.skills
    assert resolved.skills == ["a", "c"]


def test_set_readd_does_not_win(tmp_path: Path) -> None:
    """A ``--set`` re-add also merges pre-exclusion and is stripped."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a, b, c]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None, set_pairs=("skills=[b]",))
    assert "b" not in resolved.skills
    assert resolved.skills == ["a", "c"]


# ---------------------------------------------------------------------------
# (d) using exclude does not trigger the unknown-key warning
# ---------------------------------------------------------------------------


def test_exclude_does_not_warn_unknown_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``exclude`` is a known key — it must not fire ``_warn_unknown_keys``."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a, b]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )
    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile"):
        resolve_build_profile(child.resolve(), preset=None)
    assert not any("Unknown profile key" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# (e) excluding a non-existent entry is a silent no-op
# ---------------------------------------------------------------------------


def test_exclude_nonexistent_entry_is_silent_noop(tmp_path: Path) -> None:
    """Excluding an entry the base never declared changes nothing and raises nothing."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a, b]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [does-not-exist]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.skills == ["a", "b"]


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_exclude_without_extends_applies_to_self(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Base-less ``exclude`` only touches the file's own declarations (with a debug log)."""
    profile = _write(
        tmp_path / "p.yml",
        "name: Solo\ndata_bundle: hello_world\nskills: [a, b]\nexclude:\n  skills: [b]\n",
    )
    with caplog.at_level(logging.DEBUG, logger="osprey.cli.build_profile"):
        resolved, _ = resolve_build_profile(profile.resolve(), preset=None)
    assert resolved.skills == ["a"]
    assert any("without 'extends'" in rec.message for rec in caplog.records)


def test_exclude_unknown_field_raises(tmp_path: Path) -> None:
    """Excluding a non-list / unknown field is rejected with a clear error."""
    _write(tmp_path / "base.yml", "name: Base\ndata_bundle: hello_world\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config: [x]\n",
    )
    with pytest.raises(BuildProfileError, match="unknown or non-list field"):
        resolve_build_profile(child.resolve(), preset=None)


def test_exclude_non_list_value_raises(tmp_path: Path) -> None:
    """A field mapped to a non-list value is rejected."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata_bundle: hello_world\nskills: [a]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: not-a-list\n",
    )
    with pytest.raises(BuildProfileError, match="must be a list"):
        resolve_build_profile(child.resolve(), preset=None)


def test_apply_exclude_rejects_non_mapping() -> None:
    """``_apply_exclude`` requires a mapping ``exclude`` value."""
    with pytest.raises(BuildProfileError, match="must be a mapping"):
        _apply_exclude({"skills": ["a"]}, ["skills"])
