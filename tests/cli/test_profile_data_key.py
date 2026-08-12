"""Tests for the profile-carried ``data:`` tree and its preset-mode rejection.

``data: <dir>`` names the facility data tree a profile carries, anchored on the
profile *root* — which for a persona delta under ``personas/`` is the directory
holding the root ``profile.yml``, not the delta's own parent, so personas share
the root's tree without spelling a traversal. It is meaningless for ``--preset``
builds, which have no profile directory to anchor against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import (
    _KNOWN_PROFILE_KEYS,
    BuildProfile,
    load_profile,
    resolve_build_profile,
)
from osprey.cli.profile_root import resolve_profile_root
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
# Declaration and validation
# ---------------------------------------------------------------------------


def test_data_is_a_recognized_profile_key() -> None:
    """``data:`` is allowlisted, so it never trips the unknown-key path."""
    assert "data" in _KNOWN_PROFILE_KEYS


def test_data_directory_beside_the_profile_validates(tmp_path: Path) -> None:
    """The materialized-profile layout: ``data/`` next to ``profile.yml``."""
    (tmp_path / "data").mkdir()
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": "data"})

    assert load_profile(profile).data == "data"


def test_persona_data_anchors_at_the_profile_root(tmp_path: Path) -> None:
    """The persona layout: a delta under ``personas/`` sharing the root's tree."""
    (tmp_path / "data").mkdir()
    _write_yaml(tmp_path / "profile.yml", {"name": "root", "data": "data"})
    profile = _write_yaml(tmp_path / "personas" / "alice.yml", {"name": "alice", "data": "data"})

    loaded = load_profile(profile)

    assert loaded.data == "data"
    root_dir, is_persona_delta = resolve_profile_root(profile)
    assert is_persona_delta is True
    assert loaded.resolved_data_root(root_dir) == (tmp_path / "data").resolve()


def test_persona_file_without_a_profile_root_is_rejected(tmp_path: Path) -> None:
    """A ``personas/`` file with no root cannot anchor, so it never loads."""
    (tmp_path / "data").mkdir()
    profile = _write_yaml(tmp_path / "personas" / "alice.yml", {"name": "alice", "data": "data"})

    with pytest.raises(BuildProfileError, match="profile root is missing"):
        load_profile(profile)


def test_missing_data_directory_is_reported_with_both_paths(tmp_path: Path) -> None:
    """The error names the declared path and where it resolved to."""
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": "data"})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "data directory not found" in message
    assert str((tmp_path / "data").resolve()) in message


def test_data_pointing_at_a_file_is_rejected(tmp_path: Path) -> None:
    """A data tree must be a directory, not a file."""
    (tmp_path / "data").write_text("not a tree", encoding="utf-8")
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": "data"})

    with pytest.raises(BuildProfileError, match="data must be a directory"):
        load_profile(profile)


@pytest.mark.parametrize("value", [42, "", "   ", ["data"]])
def test_non_path_data_values_are_rejected(tmp_path: Path, value: Any) -> None:
    """Only a non-empty string names a tree."""
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": value})

    with pytest.raises(BuildProfileError, match="data must be a non-empty directory path"):
        load_profile(profile)


def test_data_errors_accumulate_with_other_validation_errors(tmp_path: Path) -> None:
    """House convention: one raise carrying every problem, not the first."""
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "", "data": "data", "tier": 2})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "data directory not found" in message
    assert "'name' is required" in message
    assert "tier must be 1 or 3" in message


def test_resolved_data_root_is_none_without_the_key(tmp_path: Path) -> None:
    """Profiles carrying no ``data:`` keep today's bundle-sourced behavior."""
    assert BuildProfile(name="p").resolved_data_root(tmp_path) is None


# ---------------------------------------------------------------------------
# Preset-mode rejection
# ---------------------------------------------------------------------------


def test_preset_mode_rejects_data_from_set_pair(fake_presets: Path, tmp_path: Path) -> None:
    """SC5 negative matrix: ``--preset X --set data=...`` errors legibly."""
    _write_yaml(fake_presets / "base.yml", {"name": "base"})
    (tmp_path / "data").mkdir()

    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(None, "base", set_pairs=(f"data={tmp_path / 'data'}",))

    message = str(excinfo.value)
    assert "'data' is not supported with --preset" in message
    assert "osprey init" in message
    assert "--preset base" in message


def test_preset_mode_rejects_data_from_override_file(fake_presets: Path, tmp_path: Path) -> None:
    """The same rejection however the key is injected — here via ``-O``."""
    _write_yaml(fake_presets / "base.yml", {"name": "base"})
    override = _write_yaml(tmp_path / "o.yml", {"data": "data"})

    with pytest.raises(BuildProfileError, match="not supported with --preset"):
        resolve_build_profile(None, "base", overrides=(override,))


def test_preset_mode_rejects_data_carried_by_the_preset_itself(fake_presets: Path) -> None:
    """A preset that spells ``data:`` is rejected rather than silently anchored
    on the bundled package directory."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "data": "data"})

    with pytest.raises(BuildProfileError, match="not supported with --preset"):
        resolve_build_profile(None, "base")


def test_preset_mode_rejects_data_inherited_through_extends(fake_presets: Path) -> None:
    """``extends:`` is an injection path too — the guard runs post-resolution."""
    _write_yaml(fake_presets / "parent.yml", {"name": "parent", "data": "data"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "parent"})

    with pytest.raises(BuildProfileError, match="not supported with --preset"):
        resolve_build_profile(None, "child")


def test_file_mode_accepts_data(tmp_path: Path) -> None:
    """The guard is scoped to preset mode; file mode is the supported path."""
    (tmp_path / "data").mkdir()
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": "data"})

    resolved, profile_dir = resolve_build_profile(profile, None)

    assert resolved.data == "data"
    assert resolved.resolved_data_root(profile_dir) == (tmp_path / "data").resolve()
