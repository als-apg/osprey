"""``web.bar_items`` is dropped at the single parse point, with one warning naming the sidebar.

The sidebar shell replaced the header and the status bar the key arranged,
so nothing reads it. A profile emitted before the sidebar is otherwise sound:
the key is ignored and the build goes on, told once per document where the
bars went. Every other ``web.*`` key survives.
"""

from __future__ import annotations

import logging
import textwrap

import pytest

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_document import (
    _parse_profile_document,
    normalize_retired_bar_items,
)

_LOGGER_NAME = doc.__name__


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()


def _parse(text: str, source: str = "profile.yml"):
    return _parse_profile_document(textwrap.dedent(text), source)


def _bar_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and "web.bar_items" in r.getMessage()
    ]


def test_a_nested_bar_items_is_dropped_and_the_sidebar_named(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(
        """
        name: Child
        config:
          web:
            app_name: Mine
            bar_items:
              header: [clock]
              status_visible: false
        """
    )
    assert raw["config"]["web"] == {"app_name": "Mine"}
    (line,) = _bar_warnings(caplog)
    assert "profile.yml" in line
    assert "sidebar" in line
    assert "ignored" in line
    assert "osprey init --revalidate" in line
    assert "Workspace" in line and "Tools" in line and "More" in line


def test_a_dotted_spelling_is_dropped_too(caplog):
    """``osprey set config.web.bar_items=…`` writes the dotted form."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(
        """
        name: Child
        config:
          web.bar_items: [clock]
          web.bar_items.locked: true
          web.app_name: Mine
        """
    )
    assert raw["config"] == {"web.app_name": "Mine"}
    assert len(_bar_warnings(caplog)) == 1


def test_an_emptied_web_mapping_goes_with_the_key(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(
        """
        name: Child
        config:
          web:
            bar_items: {header: [clock]}
          cli:
            theme: default
        """
    )
    assert raw["config"] == {"cli": {"theme": "default"}}


def test_a_silent_document_is_untouched_and_quiet(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(
        """
        name: Child
        config:
          web:
            app_name: Mine
        """
    )
    assert raw["config"] == {"web": {"app_name": "Mine"}}
    assert _bar_warnings(caplog) == []


def test_one_warning_per_build_names_the_first_source(caplog):
    """The image staging copy parses the same profile again; it must not warn twice."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    normalize_retired_bar_items({"config": {"web": {"bar_items": {}}}}, "profile.yml")
    normalize_retired_bar_items({"config": {"web.bar_items": 1}}, "build/.tmp/x/profile.yml")
    (line,) = _bar_warnings(caplog)
    assert line.startswith("profile.yml:")
