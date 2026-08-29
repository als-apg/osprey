"""Tests for the strict key check inside a profile's ``bluesky:`` block.

Reading every field of the block with ``.get()`` would silently drop an unknown
key — a typo like ``tiled_enabld:``, or a knob a release removed — and the build
would ship without whatever the facility asked for. The block is closed the same
way the top level is: unknown keys hard-error, naming every offender and its
closest known spelling.

The valid set is *derived* from :class:`BlueskyConfig`'s dataclass fields, the
same declaration ``_parse_profile`` reads the block into, so the check cannot
drift from what the parser actually honors.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile_presets
from osprey.cli.build_profile_load import (
    _KNOWN_BLUESKY_KEYS,
    _parse_profile,
    load_profile,
)
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.build_profile_schema import BlueskyConfig
from osprey.cli.profile_cmd import validate as profile_validate
from osprey.errors import BuildProfileError
from osprey.port_layout import default_port

#: A port an author pinned by hand, one block above the layout's own. What these
#: tests pin is that an AUTHORED number survives parsing and merging, so the
#: value has to be provably not a default — derived from one, and clear of it.
AUTHORED_PORT = default_port("bluesky") + 1000

#: The tiled catalog's port in the same hand-pinned deployment.
AUTHORED_TILED_PORT = AUTHORED_PORT + 1


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


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


# ── the known set is derived, not hand-maintained ────────────────────────────


def test_known_keys_are_the_schema_fields() -> None:
    """One source of truth: the dataclass the block is parsed into.

    A hand-kept duplicate list would drift the moment a field is added or
    renamed — accepting a key the parser ignores, or rejecting one it reads.
    """
    assert _KNOWN_BLUESKY_KEYS == {f.name for f in fields(BlueskyConfig)}


def test_every_valid_key_is_accepted_and_read() -> None:
    """A block spelling every field parses, and each value reaches the config."""
    raw: dict[str, Any] = {
        "name": "x",
        "bluesky": {
            "port": AUTHORED_PORT,
            "tiled_enabled": True,
            "tiled_port": AUTHORED_TILED_PORT,
            "second_lane": True,
            "plan_dir": "/facility/plans",
            "excluded_plans": ["orm"],
            "devices_file": "data/facility_devices.yml",
            "device_page_size": 250,
        },
    }

    profile = _parse_profile(raw)

    assert profile.bluesky == BlueskyConfig(
        port=AUTHORED_PORT,
        tiled_enabled=True,
        tiled_port=AUTHORED_TILED_PORT,
        second_lane=True,
        plan_dir="/facility/plans",
        excluded_plans=["orm"],
        devices_file="data/facility_devices.yml",
        device_page_size=250,
    )


def test_empty_block_still_takes_the_defaults() -> None:
    """The check rejects unknown keys, it does not require any."""
    profile = _parse_profile({"name": "x", "bluesky": {}})

    assert profile.bluesky == BlueskyConfig()


# ── values are validated, not just key names ─────────────────────────────────


@pytest.mark.parametrize("value", [0, -5, True, "abc"])
def test_device_page_size_rejects_a_value_that_is_not_a_positive_int(value: Any) -> None:
    """The key is auto-accepted (it is a dataclass field), so the VALUE check is
    the only thing standing between a nonsense page size and a shipped build.

    ``True`` is in the set on purpose: it is an ``int`` to ``isinstance``, and a
    boolean page size is an authoring mistake, not a size of one.
    """
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "bluesky": {"device_page_size": value}})

    message = str(excinfo.value)
    assert "bluesky.device_page_size" in message
    assert repr(value) in message


def test_device_page_size_accepts_one() -> None:
    """The floor is inclusive — a one-device page is small, not invalid."""
    profile = _parse_profile({"name": "x", "bluesky": {"device_page_size": 1}})

    assert profile.bluesky is not None
    assert profile.bluesky.device_page_size == 1


# ── unknown keys are rejected ────────────────────────────────────────────────


def test_unknown_bluesky_key_is_rejected_naming_the_key() -> None:
    """The offender is named, so the operator knows which line to delete."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "bluesky": {"zzz_nonsense": 1}})

    message = str(excinfo.value)
    assert "'zzz_nonsense'" in message
    assert "bluesky" in message


def test_typo_gets_its_nearest_match_suggested() -> None:
    """A near-miss spelling carries the fix, not just the complaint."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "bluesky": {"tiled_enabld": True}})

    message = str(excinfo.value)
    assert "'tiled_enabld'" in message
    assert "did you mean 'tiled_enabled'?" in message


def test_all_unknown_bluesky_keys_are_named_in_one_error() -> None:
    """Accumulated, not first-wins — one pass fixes the whole block."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(
            {"name": "x", "bluesky": {"zzz_nonsense": 1, "prt": AUTHORED_PORT, "tiled": True}}
        )

    message = str(excinfo.value)
    assert "'zzz_nonsense'" in message
    assert "'prt'" in message
    assert "'tiled'" in message


def test_error_lists_the_valid_bluesky_keys() -> None:
    """The message carries the whole closed set the block may spell."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "bluesky": {"zzz_nonsense": 1}})

    message = str(excinfo.value)
    assert "valid keys are:" in message
    for key in _KNOWN_BLUESKY_KEYS:
        assert key in message


def test_error_style_matches_the_top_level_check() -> None:
    """Same exception, same wording — operators see one behavior, not two.

    Pinned against the top-level unknown-key error rather than a literal: if
    that message is ever reworded, this check must move with it.
    """
    with pytest.raises(BuildProfileError) as block_err:
        _parse_profile({"name": "x", "bluesky": {"zzz_nonsense": 1}})
    with pytest.raises(BuildProfileError) as top_err:
        _parse_profile({"name": "x", "zzz_nonsense": 1})

    block = str(block_err.value)
    top = str(top_err.value)
    assert block.startswith("Unknown bluesky key(s): 'zzz_nonsense'.")
    assert top.startswith("Unknown profile key(s): 'zzz_nonsense'.")
    # Everything after the offender list is the identical remedy sentence.
    assert (
        block.split("'zzz_nonsense'.", 1)[1].split("valid keys are:")[0]
        == (top.split("'zzz_nonsense'.", 1)[1].split("valid keys are:")[0])
    )


def test_stale_demo_runner_key_fails_loudly() -> None:
    """Regression pin for a retired knob.

    ``bluesky.demo_runner`` names a runner OSPREY does not have. A profile
    carrying it must stop the build rather than be quietly ignored — an
    unhonored key has to announce itself.
    """
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "bluesky": {"demo_runner": "mock"}})

    assert "'demo_runner'" in str(excinfo.value)


def test_unknown_key_survives_a_file_load(tmp_path: Path) -> None:
    """The check is on the parse path every profile file takes."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "bluesky": {"demo_runner": True}})

    with pytest.raises(BuildProfileError, match="'demo_runner'"):
        load_profile(profile)


def test_profile_validate_exits_non_zero(tmp_path: Path) -> None:
    """The CLI surfaces it as a usage error, not a warning in the log."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "bluesky": {"demo_runner": True}})

    result = CliRunner().invoke(profile_validate, [str(profile)])

    assert result.exit_code != 0
    assert "'demo_runner'" in _flat(result.output)


# ── inheritance sees the merged block ────────────────────────────────────────


def test_parent_declared_key_is_not_unknown_from_a_child(
    fake_presets: Path, tmp_path: Path
) -> None:
    """The check runs on the merged block, so layering is not a false positive."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "bluesky": {"port": AUTHORED_PORT}})
    child = _write_yaml(
        tmp_path / "child.yml",
        {"name": "child", "extends": "base", "bluesky": {"tiled_enabled": True}},
    )

    profile, _dir = resolve_build_profile(child, None)

    assert profile.bluesky is not None
    assert profile.bluesky.port == AUTHORED_PORT
    assert profile.bluesky.tiled_enabled is True


def test_unknown_key_in_an_extends_parent_is_rejected(fake_presets: Path, tmp_path: Path) -> None:
    """Inheritance is not a laundering path for the block either."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "bluesky": {"demo_runner": True}})
    child = _write_yaml(tmp_path / "child.yml", {"name": "child", "extends": "base"})

    with pytest.raises(BuildProfileError, match="'demo_runner'"):
        resolve_build_profile(child, None)


def test_unknown_key_from_a_set_pair_is_rejected(fake_presets: Path) -> None:
    """``--set`` cannot inject a key the block does not define either."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "bluesky": {"port": AUTHORED_PORT}})

    with pytest.raises(BuildProfileError, match="'demo_runner'"):
        resolve_build_profile(None, "base", set_pairs=("bluesky.demo_runner=true",))


# ── the shipped presets survive the promotion ────────────────────────────────


def test_every_bundled_preset_still_resolves() -> None:
    """Including the two ``extends: control-assistant`` children, whose bluesky
    block arrives entirely from the parent layer."""
    for name in build_profile_presets.list_presets():
        resolve_build_profile(None, name)


@pytest.mark.parametrize(
    "preset", ["control-assistant", "control-assistant-readonly", "control-assistant-readwrite"]
)
def test_control_assistant_family_keeps_its_bluesky_block(preset: str) -> None:
    """The preset that actually carries a bluesky block resolves it intact.

    The preset names no port — the bridge and its catalog publish inside the
    deployment's port block — so what it resolves to is the layout's own pair.
    """
    profile, _dir = resolve_build_profile(None, preset)

    assert profile.bluesky is not None
    assert profile.bluesky.port == default_port("bluesky")
    assert profile.bluesky.tiled_port == default_port("tiled")
    assert profile.bluesky.tiled_enabled is True
