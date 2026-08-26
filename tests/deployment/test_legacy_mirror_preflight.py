"""``osprey up`` names a logbook mirror left under the retired ``build/`` anchor.

The mirror is derived, so nothing is moved: the preflight warns, counts the
files, and names ``osprey ariel qmd-resync`` — and stays silent when there is
nothing under the old path, when the export is off, or when the path is
absolute (no anchor to have moved).
"""

from __future__ import annotations

import logging

from osprey.deployment.container_lifecycle import _preflight_legacy_ariel_mirror


def _config(mirror_path: str = "var/ariel_mirror", *, enabled: bool = True) -> dict:
    return {
        "ariel": {
            "enhancement_modules": {
                "qmd_export": {"enabled": enabled, "mirror_path": mirror_path},
            }
        }
    }


def test_files_under_the_legacy_anchor_are_named(tmp_path, caplog):
    legacy = tmp_path / "build" / "var" / "ariel_mirror"
    legacy.mkdir(parents=True)
    (legacy / "2026-01-01_entry.md").write_text("# entry\n")
    (legacy / "2026-01-02_entry.md").write_text("# entry\n")

    with caplog.at_level(logging.WARNING):
        _preflight_legacy_ariel_mirror(_config(), tmp_path)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "2 file(s)" in message
    assert str(legacy) in message
    assert str(tmp_path / "var" / "ariel_mirror") in message
    assert "osprey ariel qmd-resync" in message


def test_an_empty_legacy_directory_is_silent(tmp_path, caplog):
    (tmp_path / "build" / "var" / "ariel_mirror").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        _preflight_legacy_ariel_mirror(_config(), tmp_path)
    assert caplog.records == []


def test_no_legacy_directory_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _preflight_legacy_ariel_mirror(_config(), tmp_path)
    assert caplog.records == []


def test_a_disabled_export_is_silent(tmp_path, caplog):
    legacy = tmp_path / "build" / "var" / "ariel_mirror"
    legacy.mkdir(parents=True)
    (legacy / "entry.md").write_text("# entry\n")
    with caplog.at_level(logging.WARNING):
        _preflight_legacy_ariel_mirror(_config(enabled=False), tmp_path)
    assert caplog.records == []


def test_an_absolute_mirror_path_is_silent(tmp_path, caplog):
    """An absolute path never had an anchor to move."""
    mirror = tmp_path / "elsewhere"
    mirror.mkdir()
    with caplog.at_level(logging.WARNING):
        _preflight_legacy_ariel_mirror(_config(str(mirror)), tmp_path)
    assert caplog.records == []
