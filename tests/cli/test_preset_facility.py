"""The preset-side ``facility:`` key and the packaged data it composes.

A preset names a bundled facility (``templates/facilities/<name>/``) with
``facility:``, beside the app template it names with ``app_template:``. Both
keys are preset-side only: the resolver follows ``extends:`` to find them, the
preset read consumes them, an emitted profile carries neither, and a repo
``profile.yml`` spelling one is refused. ``osprey init`` copies the app
template's ``data/`` plus the facility at ``data/facility/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import _load_preset_raw, resolve_build_profile
from osprey.cli.build_profile_presets import PRESET_FACILITY_KEY, preset_facility
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    """Write a profile document; a repo profile also gets the ``data:`` it must name."""
    if path.parent.name != "presets":
        body = {**body, "data": "data"}
        (path.parent / "data").mkdir(exist_ok=True)
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
# Resolver
# ---------------------------------------------------------------------------


def test_the_resolver_follows_extends(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    assert preset_facility("child") == "example"


def test_the_nearest_facility_on_the_chain_wins(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(
        fake_presets / "child.yml", {"name": "child", "extends": "base", "facility": "hello_world"}
    )

    assert preset_facility("child") == "hello_world"


def test_a_chain_naming_no_facility_resolves_to_none(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})

    assert preset_facility("base") is None


# ---------------------------------------------------------------------------
# Consumed by the preset read; never part of a profile
# ---------------------------------------------------------------------------


def test_the_preset_read_consumes_the_facility_key(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})

    raw, _path = _load_preset_raw("base")

    assert PRESET_FACILITY_KEY not in raw


def test_an_inherited_facility_key_is_consumed_too(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    profile, _dir = resolve_build_profile(None, "child")

    assert profile.name == "child"


def test_a_repo_profile_extending_a_preset_naming_a_facility_resolves(
    fake_presets: Path, tmp_path: Path
) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "extends": "base"})

    resolved, _dir = resolve_build_profile(profile, None)

    assert resolved.name == "p"


def test_a_profile_naming_a_facility_is_refused(tmp_path: Path) -> None:
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "facility": "example"})

    with pytest.raises(BuildProfileError, match="facility is a preset key"):
        resolve_build_profile(profile, None)
