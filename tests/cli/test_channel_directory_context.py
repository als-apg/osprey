"""The ``channel_directory`` predicates in the render context and on the profile card.

Both read them from :mod:`osprey.services.channel_directory.resolve`; neither
computes them. A malformed section at render is a warning and five false keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import _parse_profile
from osprey.cli.profile_card import _registry_context, _render_config
from osprey.cli.templates.claude_code import config_derived_context
from osprey.services.channel_directory.resolve import PREDICATES

_URL = {"url": "https://directory.example/api"}
_ALL_FALSE = dict.fromkeys(PREDICATES, False)


def _keys(config: dict[str, Any], tmp_path: Path) -> dict[str, bool]:
    ctx = config_derived_context(config, tmp_path)
    return {key: ctx[key] for key in PREDICATES}


@pytest.mark.parametrize("config", [{}, {"channel_directory": None}], ids=["absent", "null"])
def test_keys_are_false_without_a_section(config: dict[str, Any], tmp_path: Path) -> None:
    assert _keys(config, tmp_path) == _ALL_FALSE


def test_a_va_only_section_is_configured_and_serves_no_history(tmp_path: Path) -> None:
    assert _keys({"channel_directory": {}}, tmp_path) == {
        **_ALL_FALSE,
        "channel_directory_configured": True,
    }


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (
            {"alarm_log": {"type": "phoebus_alarm_logger", "settings": _URL}},
            {"channel_directory_alarm_log": True, "channel_directory_alarm_config": True},
        ),
        (
            {"alarm_state": {"replay_file": "replays/alarms.ndjson"}},
            {"channel_directory_alarm_state": True, "channel_directory_alarm_config": True},
        ),
        (
            {
                "alarm_state": {
                    "type": "phoebus_alarm_server",
                    "settings": {"bootstrap_servers": "kafka:9092", "topic": "Accelerator"},
                }
            },
            {"channel_directory_alarm_state": True, "channel_directory_alarm_config": True},
        ),
        (
            {"put_log": {"type": "caputlog", "settings": _URL}},
            {"channel_directory_put_log": True},
        ),
    ],
    ids=["alarm_log", "alarm_replay", "alarm_feed", "put_log"],
)
def test_each_sub_block_sets_its_own_keys(
    block: dict[str, Any], expected: dict[str, bool], tmp_path: Path
) -> None:
    section = {"type": "channelfinder", "settings": _URL, **block}

    assert _keys({"channel_directory": section}, tmp_path) == {
        **_ALL_FALSE,
        "channel_directory_configured": True,
        **expected,
    }


def test_a_malformed_section_warns_and_renders_every_key_false(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="osprey.cli.templates"):
        keys = _keys({"channel_directory": {"type": "channelfinder"}}, tmp_path)

    assert keys == _ALL_FALSE
    assert len(caplog.records) == 1
    assert "channel_directory.settings.url" in caplog.records[0].getMessage()


# ── the profile card ─────────────────────────────────────────────────────────


def _card_keys(config: dict[str, Any], delta: dict[str, Any] | None = None) -> dict[str, bool]:
    profile = _parse_profile({"name": "x", "data": "data", "config": config})
    ctx = _registry_context(profile, _render_config(profile, delta))
    return {key: ctx[key] for key in PREDICATES}


def test_the_card_reads_the_same_keys() -> None:
    assert _card_keys({}) == _ALL_FALSE
    assert _card_keys(
        {"channel_directory.alarm_log": {"type": "phoebus_alarm_logger", "settings": _URL}}
    ) == {
        **_ALL_FALSE,
        "channel_directory_configured": True,
        "channel_directory_alarm_log": True,
        "channel_directory_alarm_config": True,
    }


def test_a_layer_spelling_null_removes_the_section_from_the_card() -> None:
    delta = {"config": {"channel_directory": None}}

    assert _card_keys({"channel_directory": {}}, delta) == _ALL_FALSE
