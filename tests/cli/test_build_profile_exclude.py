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

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import (
    _apply_exclude,
    _deep_merge,
    _merge_lists,
    _resolve_extends,
    resolve_build_profile,
)
from osprey.errors import BuildProfileError


def _write(path: Path, text: str) -> Path:
    """Write a profile and the ``data:`` tree every profile must name."""
    (path.parent / "data").mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# (a) child excludes an inherited skill
# ---------------------------------------------------------------------------


def test_child_excludes_inherited_skill(tmp_path: Path) -> None:
    """A child ``exclude`` removes an entry contributed by its ``extends`` base."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [alpha, beta, gamma]\n",
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
        "name: Base\ndata: data\nskills: [s1, s2]\nrules: [r1, r2]\ndependencies: [pkg-a, pkg-b]\n",
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
        "name: Base\ndata: data\nskills: [a, b, c]\n",
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
# (c) an overlay re-add does NOT win; a --set re-add DOES
# ---------------------------------------------------------------------------


def test_a_host_variant_overlay_readd_does_not_win(tmp_path: Path) -> None:
    """A host-variant overlay is INHERITANCE: it merges pre-exclusion and is stripped.

    An overlay is a layer like any other — a difference over the profile it
    sits above — so it is merged before ``extends`` is resolved and the child's
    ``exclude:`` still runs afterwards. Re-adding a skill from there is not the
    verb that takes an exclusion back.
    """
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [a, b, c]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )
    overlay = _write(tmp_path / "overlay.yml", "skills: [b]\n")
    resolved, _ = resolve_build_profile(child.resolve(), preset=None, overlays=(overlay.resolve(),))
    assert "b" not in resolved.skills
    assert resolved.skills == ["a", "c"]


def test_a_set_readd_wins_over_the_exclusion(tmp_path: Path) -> None:
    """The other half: ``--set`` is an EDIT, applied after the exclusion ran.

    It states the skill list rather than layering under inheritance, so an
    operator can put back what a profile excludes — and gets exactly the list
    they typed, not that list merged with what survived the exclusion.
    """
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [a, b, c]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None, set_pairs=("skills=[b]",))
    assert resolved.skills == ["b"]


# ---------------------------------------------------------------------------
# exclude.config: taking entries out of a list-valued config key
# ---------------------------------------------------------------------------


def test_exclude_config_subtracts_from_a_list_valued_key(tmp_path: Path) -> None:
    """``exclude: config:`` names a dotted config key and removes entries from its list."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nconfig:\n"
        "  deployed_services: [alpha, beta, gamma]\n  facility.name: Ring\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config:\n    deployed_services: [beta]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.config["deployed_services"] == ["alpha", "gamma"]
    assert resolved.config["facility.name"] == "Ring"


def test_a_host_variant_overlay_narrows_the_services_it_deploys(tmp_path: Path) -> None:
    """The case the verb exists for: one host deploys fewer services than the profile.

    An overlay is inheritance, so its lists union with the profile's and it
    cannot state a shorter ``deployed_services``. Taking away is explicit, and
    ``exclude: config:`` is the spelling — the same verb that drops a skill.
    """
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nconfig:\n  deployed_services: [alpha, beta, gamma]\n",
    )
    child = _write(tmp_path / "child.yml", "extends: ./base.yml\nname: Child\n")
    overlay = _write(
        tmp_path / "overlay.yml",
        "exclude:\n  config:\n    deployed_services: [beta, gamma]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None, overlays=(overlay.resolve(),))
    assert resolved.config["deployed_services"] == ["alpha"]


def test_exclude_config_of_an_absent_key_is_a_silent_noop(tmp_path: Path) -> None:
    """Nothing inherited under that key means nothing to take away."""
    _write(tmp_path / "base.yml", "name: Base\ndata: data\nconfig:\n  facility.name: Ring\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config:\n    nothing.here: [x]\n",
    )
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert "nothing.here" not in resolved.config
    assert resolved.config["facility.name"] == "Ring"


def test_exclude_config_of_a_non_list_value_raises(tmp_path: Path) -> None:
    """A scalar is stated, not subtracted from: naming one under exclude is refused."""
    _write(tmp_path / "base.yml", "name: Base\ndata: data\nconfig:\n  facility.name: Ring\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config:\n    facility.name: [Ring]\n",
    )
    with pytest.raises(BuildProfileError, match="is not a list"):
        resolve_build_profile(child.resolve(), preset=None)


def test_exclude_config_must_be_a_mapping(tmp_path: Path) -> None:
    _write(tmp_path / "base.yml", "name: Base\ndata: data\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config: [x]\n",
    )
    with pytest.raises(BuildProfileError, match="exclude.config must be a mapping"):
        resolve_build_profile(child.resolve(), preset=None)


def test_exclude_config_entry_must_be_a_list(tmp_path: Path) -> None:
    _write(tmp_path / "base.yml", "name: Base\ndata: data\nconfig:\n  deployed_services: [alpha]\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  config:\n    deployed_services: alpha\n",
    )
    with pytest.raises(BuildProfileError, match="must be a list of entries"):
        resolve_build_profile(child.resolve(), preset=None)


# ---------------------------------------------------------------------------
# (d) using exclude does not trigger the unknown-key warning
# ---------------------------------------------------------------------------


def test_exclude_is_not_an_unknown_key(tmp_path: Path) -> None:
    """``exclude`` is allowlisted — it must not trip ``_reject_unknown_keys``.

    Unknown keys are fatal, so resolving at all is the assertion.
    """
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [a, b]\n",
    )
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  skills: [b]\n",
    )

    profile, _dir = resolve_build_profile(child.resolve(), preset=None)

    assert profile.skills == ["a"]


# ---------------------------------------------------------------------------
# (e) excluding a non-existent entry is a silent no-op
# ---------------------------------------------------------------------------


def test_exclude_nonexistent_entry_is_silent_noop(tmp_path: Path) -> None:
    """Excluding an entry the base never declared changes nothing and raises nothing."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [a, b]\n",
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
        "name: Solo\ndata: data\nskills: [a, b]\nexclude:\n  skills: [b]\n",
    )
    with caplog.at_level(logging.DEBUG, logger="osprey.cli.build_profile_merge"):
        resolved, _ = resolve_build_profile(profile.resolve(), preset=None)
    assert resolved.skills == ["a"]
    assert any("without 'extends'" in rec.message for rec in caplog.records)


def test_exclude_unknown_field_raises(tmp_path: Path) -> None:
    """Excluding a non-list / unknown field is rejected with a clear error."""
    _write(tmp_path / "base.yml", "name: Base\ndata: data\n")
    child = _write(
        tmp_path / "child.yml",
        "extends: ./base.yml\nname: Child\nexclude:\n  bogus: [x]\n",
    )
    with pytest.raises(BuildProfileError, match="unknown or non-list field"):
        resolve_build_profile(child.resolve(), preset=None)


def test_exclude_non_list_value_raises(tmp_path: Path) -> None:
    """A field mapped to a non-list value is rejected."""
    _write(
        tmp_path / "base.yml",
        "name: Base\ndata: data\nskills: [a]\n",
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


# ---------------------------------------------------------------------------
# build_profile_merge coverage gaps: _merge_lists / _deep_merge internals
# ---------------------------------------------------------------------------


def test_merge_lists_both_empty_returns_empty_list() -> None:
    """Two empty lists short-circuit to ``[]`` without touching the item loop."""
    assert _merge_lists([], []) == []


def test_merge_lists_non_string_items_concatenate() -> None:
    """Non-string-only lists (e.g. lifecycle step dicts) concatenate rather than dedup."""
    base = [{"step": "a"}]
    child = [{"step": "a"}, {"step": "b"}]
    merged = _merge_lists(base, child)
    # Concatenated, not deduped: the repeated {"step": "a"} appears twice.
    assert merged == [{"step": "a"}, {"step": "a"}, {"step": "b"}]


def test_deep_merge_recurses_into_nested_dicts() -> None:
    """A dict-valued key present on both sides merges recursively (child wins per-leaf)."""
    base = {"config": {"a": 1, "b": 2}}
    child = {"config": {"b": 20, "c": 3}}
    merged = _deep_merge(base, child)
    assert merged == {"config": {"a": 1, "b": 20, "c": 3}}


def test_apply_exclude_skips_field_whose_value_is_not_a_list() -> None:
    """A field named in ``exclude`` that resolved to a non-list value is a silent no-op."""
    merged = {"skills": {"nested": "dict-not-a-list"}}
    _apply_exclude(merged, {"skills": ["does-not-matter"]})
    # Unchanged: the non-list current value is left alone rather than erroring.
    assert merged == {"skills": {"nested": "dict-not-a-list"}}


# ---------------------------------------------------------------------------
# build_profile_merge coverage gaps: _resolve_extends error/branch paths
# ---------------------------------------------------------------------------


def test_resolve_extends_invalid_base_yaml_raises(tmp_path: Path) -> None:
    """A base file with unparsable YAML raises with the offending path named."""
    _write(tmp_path / "base.yml", "name: [unclosed\n")
    child = _write(tmp_path / "child.yml", "extends: ./base.yml\nname: Child\n")
    with pytest.raises(BuildProfileError, match="Invalid YAML"):
        _resolve_extends({"extends": "./base.yml", "name": "Child"}, child.resolve())


def test_resolve_extends_base_not_a_mapping_raises(tmp_path: Path) -> None:
    """A base file whose YAML parses to a non-mapping (e.g. a list) is rejected."""
    _write(tmp_path / "base.yml", "- a\n- b\n")
    child = _write(tmp_path / "child.yml", "extends: ./base.yml\nname: Child\n")
    with pytest.raises(BuildProfileError, match="must be a YAML mapping"):
        _resolve_extends({"extends": "./base.yml", "name": "Child"}, child.resolve())


def test_resolve_extends_circular_reference_raises(tmp_path: Path) -> None:
    """A extends B extends A is detected and reported as a cycle, not infinite recursion."""
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    _write(a, "extends: ./b.yml\nname: A\n")
    _write(b, "extends: ./a.yml\nname: B\n")
    with pytest.raises(BuildProfileError, match="Circular extends detected"):
        resolve_build_profile(a.resolve(), preset=None)


def test_resolve_extends_unresolvable_value_raises(tmp_path: Path) -> None:
    """An ``extends`` value that names neither a bundled preset nor an existing file errors."""
    child = _write(tmp_path / "child.yml", "extends: no-such-preset-or-file\nname: Child\n")
    with pytest.raises(BuildProfileError, match="Cannot resolve extends"):
        resolve_build_profile(child.resolve(), preset=None)


def test_resolve_extends_by_bundled_preset_name(tmp_path: Path, monkeypatch) -> None:
    """``extends`` resolves a bundled preset by name before falling back to a sibling path."""
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    _write(presets_dir / "demo-base.yml", "name: Base\ndata: data\nskills: [a]\n")
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets_dir)

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    child = _write(profiles_dir / "child.yml", "extends: demo-base\nname: Child\nskills: [b]\n")
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.skills == ["a", "b"]
