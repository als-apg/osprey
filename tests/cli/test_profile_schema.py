"""Tests for the ``dispatch:`` block of the build-profile schema.

Covers the :class:`DispatchConfig` dataclass, its per-field validation in
:meth:`BuildProfile.validate`, the triggers-file resolution (profile-relative
or bundled preset name), the ``shared`` + multi-worker advisory warning, the
``_parse_profile`` round-trip, the ``osprey.``-prefixed bundled-template skip,
and the ``_KNOWN_PROFILE_KEYS`` membership.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli import build_profile as bp
from osprey.cli.build_profile import (
    BuildProfile,
    DispatchConfig,
    ServiceDef,
    _parse_profile,
)
from osprey.errors import BuildProfileError


def _write_triggers(tmp_path: Path, name: str = "trig.yml") -> str:
    """Write a minimal triggers file into ``tmp_path`` and return its name."""
    (tmp_path / name).write_text("triggers: []", encoding="utf-8")
    return name


def test_no_dispatch_validates(tmp_path: Path) -> None:
    """A profile with no dispatch block validates without raising."""
    BuildProfile(name="x").validate(tmp_path)


def test_valid_dispatch_validates(tmp_path: Path) -> None:
    """A dispatch with a profile-relative triggers file validates."""
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(name="x", dispatch=DispatchConfig(triggers=triggers))
    profile.validate(tmp_path)


def test_worker_count_below_one_raises(tmp_path: Path) -> None:
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(name="x", dispatch=DispatchConfig(triggers=triggers, worker_count=0))
    with pytest.raises(BuildProfileError, match="worker_count"):
        profile.validate(tmp_path)


def test_workspace_mode_invalid_raises(tmp_path: Path) -> None:
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(
        name="x",
        dispatch=DispatchConfig(triggers=triggers, workspace_mode="weird"),  # type: ignore[arg-type]
    )
    with pytest.raises(BuildProfileError, match="workspace_mode"):
        profile.validate(tmp_path)


def test_port_overflow_raises(tmp_path: Path) -> None:
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(
        name="x",
        dispatch=DispatchConfig(triggers=triggers, worker_port_base=65530, worker_count=10),
    )
    with pytest.raises(BuildProfileError, match="65535"):
        profile.validate(tmp_path)


def test_triggers_missing_file_raises(tmp_path: Path) -> None:
    profile = BuildProfile(name="x", dispatch=DispatchConfig(triggers="does-not-exist.yml"))
    with pytest.raises(BuildProfileError, match="triggers"):
        profile.validate(tmp_path)


def test_triggers_empty_string_raises(tmp_path: Path) -> None:
    profile = BuildProfile(name="x", dispatch=DispatchConfig(triggers=""))
    with pytest.raises(BuildProfileError, match="triggers"):
        profile.validate(tmp_path)


def test_shared_multiworker_emits_warning(tmp_path: Path) -> None:
    """shared workspace + worker_count>1 warns but does not raise."""
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(
        name="x",
        dispatch=DispatchConfig(triggers=triggers, workspace_mode="shared", worker_count=2),
    )
    with pytest.warns(UserWarning, match="shared"):
        profile.validate(tmp_path)


def test_bundled_triggers_name_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundled triggers name resolves via _triggers_dir, independent of profile_dir."""
    triggers = tmp_path / "triggers"
    triggers.mkdir()
    (triggers / "tutorial_triggers.yml").write_text("triggers: []", encoding="utf-8")
    monkeypatch.setattr(bp, "_triggers_dir", lambda: triggers)

    profile_dir = tmp_path / "empty_profile"
    profile_dir.mkdir()
    profile = BuildProfile(name="x", dispatch=DispatchConfig(triggers="tutorial_triggers.yml"))
    profile.validate(profile_dir)


def test_parse_round_trip() -> None:
    profile = _parse_profile({"name": "x", "dispatch": {"triggers": "t.yml", "worker_count": 3}})
    assert profile.dispatch is not None
    assert profile.dispatch.worker_count == 3
    assert profile.dispatch.triggers == "t.yml"
    assert profile.dispatch.dispatcher_port == 8020
    assert profile.dispatch.workspace_mode == "isolated"


def test_dispatch_not_a_mapping_raises() -> None:
    with pytest.raises(BuildProfileError, match="dispatch"):
        _parse_profile({"name": "x", "dispatch": "nope"})


def test_dispatch_is_known_key() -> None:
    assert "dispatch" in bp._KNOWN_PROFILE_KEYS


def test_servicedef_osprey_prefix_skips_filesystem_check(tmp_path: Path) -> None:
    """An ``osprey.``-prefixed template skips the profile-dir check; others error."""
    bundled = BuildProfile(
        name="x", services={"ed": ServiceDef(template="osprey.event_dispatcher")}
    )
    bundled.validate(tmp_path)  # no filesystem error despite no such dir

    missing = BuildProfile(name="x", services={"ed": ServiceDef(template="nonexistent-dir")})
    with pytest.raises(BuildProfileError, match="template dir not found"):
        missing.validate(tmp_path)


def test_control_assistant_preset_ships_events_panel() -> None:
    """The shipped control-assistant preset exposes the event-dispatcher
    dashboard as a custom ``events`` web panel, backed by a URL override so
    ``BuildProfile.validate`` accepts it (custom panels require
    ``web.panels.<id>.url``). Guards the wiring against regressions."""
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)

    profile.validate(presets_dir)  # raises BuildProfileError on any issue

    assert "events" in profile.web_panels
    assert profile.config["web.panels.events.url"]
