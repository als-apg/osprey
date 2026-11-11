"""``artifact_server.example_artifact`` is retired: a profile that states it still builds.

The gallery no longer seeds an example artifact. The profile parser drops the
key in every spelling, with one warning per source and spelling.
"""

from __future__ import annotations

import logging

import pytest

from osprey.cli.build_profile_document import (
    _WARNED_RETIRED_KEYS,
    normalize_retired_example_artifact,
)


@pytest.fixture(autouse=True)
def _fresh_warning_registry():
    _WARNED_RETIRED_KEYS.clear()
    yield
    _WARNED_RETIRED_KEYS.clear()


@pytest.mark.parametrize(
    ("config", "spelled", "left"),
    [
        (
            {"artifact_server.example_artifact": True, "web.theme": "main"},
            "artifact_server.example_artifact",
            {"web.theme": "main"},
        ),
        (
            {"artifact_server": {"example_artifact": False, "host": "127.0.0.1"}},
            "artifact_server: example_artifact",
            {"artifact_server": {"host": "127.0.0.1"}},
        ),
        ({"artifact_server": {"example_artifact": True}}, "artifact_server: example_artifact", {}),
    ],
)
def test_the_key_warns_naming_its_line_and_is_dropped(config, spelled, left, caplog) -> None:
    raw = {"config": config}
    with caplog.at_level(logging.WARNING):
        normalize_retired_example_artifact(raw, "apps/logbook.yml")
    assert raw["config"] == left
    assert f"apps/logbook.yml: `{spelled}` is no longer read; delete it" in caplog.text


def test_it_warns_once_per_source_and_spelling(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            normalize_retired_example_artifact(
                {"config": {"artifact_server.example_artifact": True}}, "profile.yml"
            )
    assert caplog.text.count("no longer read") == 1


def test_other_artifact_server_keys_pass_untouched(caplog) -> None:
    raw = {"config": {"artifact_server.page_size": 50}}
    with caplog.at_level(logging.WARNING):
        normalize_retired_example_artifact(raw, "profile.yml")
    assert raw == {"config": {"artifact_server.page_size": 50}}
    assert not caplog.text
