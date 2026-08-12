"""Tests for the unknown-key hard error and the profile-schema version floor.

An unrecognized top-level key used to warn and be ignored, which shipped
deployments quietly missing whatever the profile asked for. It is now rejected,
naming every offender at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from osprey.cli import build_profile_presets
from osprey.cli.build_profile_load import (
    _KNOWN_PROFILE_KEYS,
    _PROFILE_SCHEMA_MIN_OSPREY,
    load_profile,
)
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A preset directory the test owns, replacing the bundled one."""
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets)
    return presets


# ---------------------------------------------------------------------------
# Unknown keys are rejected
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected_with_its_closest_spelling(tmp_path: Path) -> None:
    """The error names the offending key and the key it was probably meant to be."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "mcp_server": {}})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "'mcp_server'" in message
    assert "did you mean 'mcp_servers'?" in message


def test_removed_overlay_key_is_rejected(tmp_path: Path) -> None:
    """`overlay:` was removed with FR-5 — a profile still carrying it must fail loudly.

    The hash path resolves a profile without parsing it, so schema removals
    are invisible there; only the loader can pin that the key stays gone.
    """
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "overlay": {"rules/x.md": "y"}})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    assert "'overlay'" in str(excinfo.value)


def test_all_unknown_keys_are_named_in_one_error(tmp_path: Path) -> None:
    """Accumulated, not first-wins — one pass fixes the whole file."""
    profile = _write_yaml(
        tmp_path / "p.yml", {"name": "p", "skils": [], "overlays": {}, "zzz_nonsense": 1}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "'skils'" in message
    assert "'overlays'" in message
    assert "'zzz_nonsense'" in message


def test_error_lists_the_valid_keys(tmp_path: Path) -> None:
    """R5 mitigation: the message carries the fix, not just the complaint."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "zzz_nonsense": 1})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "valid keys are:" in message
    assert "data_bundle" in message
    assert "mcp_servers" in message
    # The YAML-surface spelling has to be listed even though the check itself
    # never sees it (normalization pops it first): this list is what a facility
    # is told to write, and `app_template` is the current spelling.
    assert "app_template" in message


def test_unknown_key_in_an_extends_parent_is_rejected(fake_presets: Path, tmp_path: Path) -> None:
    """Inheritance is not a laundering path — merged layers face the same check."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "sklls": ["a"]})
    child = _write_yaml(tmp_path / "child.yml", {"name": "child", "extends": "base"})

    with pytest.raises(BuildProfileError, match="'sklls'"):
        resolve_build_profile(child, None)


def test_unknown_key_from_a_set_pair_is_rejected(fake_presets: Path) -> None:
    """``--set`` cannot inject a key the schema does not define either."""
    _write_yaml(fake_presets / "base.yml", {"name": "base"})

    with pytest.raises(BuildProfileError, match="'nonsense'"):
        resolve_build_profile(None, "base", set_pairs=("nonsense=1",))


def test_extends_and_exclude_are_not_unknown(fake_presets: Path, tmp_path: Path) -> None:
    """Both are consumed during resolution but stay allowlisted for the
    pre-resolution callers that parse a raw layer."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "skills": ["a", "b"]})
    child = _write_yaml(
        tmp_path / "child.yml",
        {"name": "child", "extends": "base", "exclude": {"skills": ["b"]}},
    )

    profile, _dir = resolve_build_profile(child, None)

    assert profile.skills == ["a"]


def test_every_bundled_preset_passes_the_stricter_schema() -> None:
    """The shipped presets must not be the first casualties of the promotion."""
    for name in build_profile_presets.list_presets():
        resolve_build_profile(None, name)


# ---------------------------------------------------------------------------
# Profile-schema version floor
# ---------------------------------------------------------------------------


def test_schema_min_osprey_is_pinned() -> None:
    """Pinned deliberately: bumping it is a compatibility decision, not a rename.

    Stamping the running ``__version__`` instead would let the emitting release
    satisfy its own gate while ignoring the keys it just wrote.
    """
    assert _PROFILE_SCHEMA_MIN_OSPREY == "2026.9.0"


def test_running_osprey_satisfies_the_schema_floor() -> None:
    """The code shipping these keys must be able to build what it emits.

    ``osprey build`` compares ``max(release lineage, _PROFILE_SCHEMA_MIN_OSPREY)``
    against a profile's ``requires_osprey_version``, and emitted profiles stamp
    ``>=`` this floor. Between releases the tag lineage sits *behind* the floor
    (the floor names the next release, which does not exist yet), so it is the
    schema arm of the max() that keeps every materialized profile buildable.
    Assert the effective capability exactly as the check computes it — the
    profile round-trip itself is proven functionally by
    ``test_profile_source_flow_parity``, which builds from emitted profiles on
    whatever checkout runs the suite.
    """
    from osprey.version import get_release_version

    effective = max(Version(get_release_version()), Version(_PROFILE_SCHEMA_MIN_OSPREY))
    assert effective in SpecifierSet(f">={_PROFILE_SCHEMA_MIN_OSPREY}")


def test_schema_min_osprey_is_a_usable_version_floor() -> None:
    """It has to parse both as a version and as the ``>=`` specifier stamped."""
    floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
    spec = SpecifierSet(f">={_PROFILE_SCHEMA_MIN_OSPREY}")

    assert floor in spec
    assert Version("2026.5.0") not in spec


def test_schema_floor_covers_the_keys_it_gates() -> None:
    """The floor exists for these keys, so both must be part of the schema.

    ``app_template`` is normalized away before the unknown-key check ever runs;
    it stays a member because the same frozenset is the "valid keys are:" list
    that check prints (pinned by test_error_lists_the_valid_keys).
    """
    assert {"app_template", "data"} <= _KNOWN_PROFILE_KEYS
