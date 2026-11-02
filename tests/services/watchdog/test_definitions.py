"""Tests for loading watchdog definitions at run time and from a profile.

These pin the two readers of the one parser: the rendered ``config.yml`` map is
the enabled watchdogs with their files under the build directory, where a file
gone missing costs that watchdog alone, with one warning; a profile's block is
every watchdog it declares, enabled or not, so one can be replayed first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.watchdog.definitions import load_profile_watchdogs, load_watchdogs
from osprey_connectors.errors import BuildProfileError

_LOGGER = "osprey.services.watchdog.definitions"

_OPTIONS = [
    {"id": "pushing", "text": "Pushing past a guard", "flag": True},
    {"id": "routine", "text": "Routine work"},
]


def _decision(**keys: Any) -> dict[str, Any]:
    return {"engine": "decision", "options": _OPTIONS, **keys}


def _agent(**keys: Any) -> dict[str, Any]:
    return {"engine": "agent", "tools": ["mcp__controls__channel_read"], **keys}


def _files(base: Path, *names: str) -> Path:
    for name in names:
        directory = base / "watchdogs" / name
        directory.mkdir(parents=True)
        (directory / "instructions.md").write_text(f"Judge as {name}.\n", encoding="utf-8")
    return base


def _rendered() -> dict[str, Any]:
    """The ``config.yml`` shape the render writes: enabled watchdogs, ``enabled: true``."""
    return {
        "control_system": {"type": "mock"},
        "watchdogs": {
            "guard-circumvention": _decision(enabled=True, threshold=0.7),
            "second-decision": _decision(enabled=True),
            "lattice-review": _agent(enabled=True, model="fast", timeout_s=30, max_cost_usd=0.1),
            "second-agent": _agent(enabled=True),
        },
    }


def _profile(base: Path, block: Any) -> Path:
    (base / "profile.yml").write_text(
        yaml.safe_dump({"name": "wd", "watchdogs": block}, sort_keys=False), encoding="utf-8"
    )
    return base


def test_a_rendered_map_loads_every_entry_from_the_build_dir(tmp_path: Path) -> None:
    _files(tmp_path, "guard-circumvention", "second-decision", "lattice-review", "second-agent")
    reference = tmp_path / "watchdogs" / "lattice-review" / "references" / "ring.md"
    reference.parent.mkdir()
    reference.write_text("The ring.\n", encoding="utf-8")

    definitions = load_watchdogs(_rendered(), tmp_path)

    assert [d.name for d in definitions] == [
        "guard-circumvention",
        "second-decision",
        "lattice-review",
        "second-agent",
    ]
    assert all(d.enabled for d in definitions)
    by_name = {d.name: d for d in definitions}
    assert by_name["guard-circumvention"].threshold == 0.7
    assert by_name["guard-circumvention"].instructions == "Judge as guard-circumvention.\n"
    assert by_name["lattice-review"].references == (reference,)
    assert by_name["lattice-review"].model == "fast"
    assert by_name["lattice-review"].timeout_s == 30.0


def test_several_watchdogs_of_one_engine_keep_their_own_names(tmp_path: Path) -> None:
    _files(tmp_path, "guard-circumvention", "second-decision", "lattice-review", "second-agent")

    definitions = load_watchdogs(_rendered(), tmp_path)

    engines = {d.name: d.engine for d in definitions}
    assert [n for n, e in engines.items() if e == "decision"] == [
        "guard-circumvention",
        "second-decision",
    ]
    assert [n for n, e in engines.items() if e == "agent"] == ["lattice-review", "second-agent"]
    assert {d.name: d.instructions for d in definitions}["second-agent"] == (
        "Judge as second-agent.\n"
    )


@pytest.mark.parametrize(
    "config", [{}, {"watchdogs": None}, {"watchdogs": {}}], ids=["no-key", "null", "empty"]
)
def test_no_rendered_watchdogs_load_nothing_quietly(
    config: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert load_watchdogs(config, tmp_path) == ()
    assert caplog.records == []


def test_a_disabled_entry_never_loads_at_run_time(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only enabled watchdogs run, and a disabled one's files are not looked for."""
    _files(tmp_path, "on")
    config = {
        "watchdogs": {
            "on": _decision(enabled=True),
            "off": _decision(enabled=False),
            "unstated": _decision(),
        }
    }

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        definitions = load_watchdogs(config, tmp_path)

    assert [d.name for d in definitions] == ["on"]
    assert caplog.records == []


def _no_directory(base: Path) -> None:
    """The render never copied this watchdog's files."""


def _no_instructions(base: Path) -> None:
    (base / "watchdogs" / "gone").mkdir(parents=True)


def _empty_instructions(base: Path) -> None:
    (base / "watchdogs" / "gone").mkdir(parents=True)
    (base / "watchdogs" / "gone" / "instructions.md").write_text("  \n", encoding="utf-8")


def _unreadable_instructions(base: Path) -> None:
    (base / "watchdogs" / "gone").mkdir(parents=True)
    (base / "watchdogs" / "gone" / "instructions.md").write_bytes(b"\xff\xfe\xff")


@pytest.mark.parametrize(
    "lose_files",
    [_no_directory, _no_instructions, _empty_instructions, _unreadable_instructions],
)
def test_missing_files_cost_that_watchdog_and_warn_once(
    lose_files: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _files(tmp_path, "kept", "also-kept")
    lose_files(tmp_path)
    config = {
        "watchdogs": {
            "kept": _decision(enabled=True),
            "gone": _agent(enabled=True),
            "also-kept": _agent(enabled=True),
        }
    }

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        definitions = load_watchdogs(config, tmp_path)

    assert [d.name for d in definitions] == ["kept", "also-kept"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "gone" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "block",
    [
        ["not", "a", "map"],
        {"bad-threshold": _decision(enabled=True, threshold=2)},
        {"../escape": _decision(enabled=True)},
        {"bad-enabled": _decision(enabled="yes")},
        {"bad-entry": "decision"},
    ],
    ids=["not-a-map", "threshold", "unsafe-name", "enabled-type", "entry-type"],
)
def test_a_broken_rendered_block_is_refused_not_skipped(block: Any, tmp_path: Path) -> None:
    """Only absent files are a run-time condition; anything else is a broken build."""
    _files(tmp_path, "bad-threshold", "bad-enabled", "bad-entry")

    with pytest.raises(BuildProfileError):
        load_watchdogs({"watchdogs": block}, tmp_path)


def test_a_profile_loads_every_watchdog_enabled_or_not(tmp_path: Path) -> None:
    _files(tmp_path, "off", "unstated", "on", "agent-off")
    _profile(
        tmp_path,
        {
            "off": _decision(enabled=False),
            "unstated": _decision(),
            "on": _decision(enabled=True),
            "agent-off": _agent(enabled=False),
        },
    )

    definitions = load_profile_watchdogs(tmp_path)

    assert [d.name for d in definitions] == ["off", "unstated", "on", "agent-off"]
    assert {d.name: d.engine for d in definitions}["agent-off"] == "agent"
    assert {d.name: d.instructions for d in definitions}["off"] == "Judge as off.\n"


def test_disabled_watchdogs_load_only_from_the_profile(tmp_path: Path) -> None:
    block = {"off": _decision(enabled=False)}
    _files(tmp_path, "off")
    _profile(tmp_path, block)

    assert load_watchdogs({"watchdogs": block}, tmp_path) == ()
    assert [d.name for d in load_profile_watchdogs(tmp_path)] == ["off"]


def test_a_profile_is_read_for_its_block_alone(tmp_path: Path) -> None:
    """Replay comes before enabling, so nothing else about the build is demanded."""
    _files(tmp_path, "on")
    _profile(tmp_path, {"on": _decision(enabled=True)})

    assert [d.name for d in load_profile_watchdogs(tmp_path)] == ["on"]


def test_a_profile_with_no_block_has_no_watchdogs(tmp_path: Path) -> None:
    (tmp_path / "profile.yml").write_text("name: wd\n", encoding="utf-8")

    assert load_profile_watchdogs(tmp_path) == ()


def test_a_profile_watchdog_with_no_files_is_refused(tmp_path: Path) -> None:
    """Unlike at run time: the author is present to add the file."""
    _profile(tmp_path, {"gone": _decision()})

    with pytest.raises(BuildProfileError, match="gone"):
        load_profile_watchdogs(tmp_path)


def test_a_missing_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BuildProfileError, match="not found"):
        load_profile_watchdogs(tmp_path)
