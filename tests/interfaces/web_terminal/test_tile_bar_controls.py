"""``web.tile_bar.controls`` as the terminal reads it and stamps it on the page.

:func:`~osprey.interfaces.web_terminal.app.tile_bar_controls` reads the shared
controls of every service tile's bar. The index page carries them on
``<html>`` as ``data-tile-bar-controls``, space-separated and always present:
an empty value is a bar without shared controls, not an absent setting. The
documentation root that the Help control opens a panel's page under rides
beside it as ``data-docs-url``, which is absent when ``web.docs_url`` is blank.

``osprey build`` refuses a bad value (``tests/cli/test_build_profile_tile_bar.py``).
The terminal reads a ``config.yml`` that may be edited by hand and keeps the
page up, so its answer to a bad value is a warning and a bar it can draw.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal.app import tile_bar_controls
from tests.interfaces.web_terminal._started_app import started_client, terminal_client

#: The ids every warning and refusal names as the ones to use.
_ALLOWED = "ask_agent, reload, help"

_APP_LOGGER = "osprey.interfaces.web_terminal.app"


def _read(tmp_path: Path, text: str) -> list[str]:
    """``tile_bar_controls`` over a ``config.yml`` that holds *text*."""
    path = tmp_path / "config.yml"
    path.write_text(text, encoding="utf-8")
    return tile_bar_controls(path)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings the reader logged."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _APP_LOGGER and record.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("project_name: x\n", id="no_web_section"),
        pytest.param("web: {}\n", id="empty_web_section"),
        pytest.param("web:\n  tile_bar: {}\n", id="empty_block"),
        pytest.param("web:\n  tile_bar:\n", id="block_with_no_value"),
        pytest.param("web:\n  tile_bar:\n    controls:\n", id="key_with_no_value"),
    ],
)
def test_an_absent_setting_gives_the_default(tmp_path, caplog, text) -> None:
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, text) == ["ask_agent"]
    assert _warnings(caplog) == []


def test_the_configured_order_is_kept(tmp_path) -> None:
    text = "web:\n  tile_bar:\n    controls: [help, reload, ask_agent]\n"
    assert _read(tmp_path, text) == ["help", "reload", "ask_agent"]


def test_an_empty_list_gives_a_bar_without_shared_controls(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, "web:\n  tile_bar:\n    controls: []\n") == []
    assert _warnings(caplog) == []


def test_a_repeated_id_keeps_its_first_place(tmp_path) -> None:
    text = "web:\n  tile_bar:\n    controls: [reload, help, reload]\n"
    assert _read(tmp_path, text) == ["reload", "help"]


def test_an_unknown_id_is_left_out_with_a_warning_that_names_the_ids(tmp_path, caplog) -> None:
    text = "web:\n  tile_bar:\n    controls: [reload, relaod]\n"
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, text) == ["reload"]
    (warning,) = _warnings(caplog)
    assert "'relaod'" in warning
    assert _ALLOWED in warning


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        pytest.param("reload", "'reload'", id="string"),
        pytest.param("{reload: true}", "{'reload': True}", id="mapping"),
    ],
)
def test_a_value_that_is_not_a_list_gives_the_default_with_a_warning(
    tmp_path, caplog, value, shown
) -> None:
    text = f"web:\n  tile_bar:\n    controls: {value}\n"
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, text) == ["ask_agent"]
    (warning,) = _warnings(caplog)
    assert shown in warning
    assert _ALLOWED in warning


def test_a_block_that_is_not_a_mapping_gives_the_default_with_a_warning(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, "web:\n  tile_bar: [reload]\n") == ["ask_agent"]
    (warning,) = _warnings(caplog)
    assert "web.tile_bar" in warning


def test_a_config_that_cannot_be_read_gives_the_default_with_a_warning(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, "web: [unclosed\n") == ["ask_agent"]
    (warning,) = _warnings(caplog)
    assert "web.tile_bar.controls" in warning


# ---------------------------------------------------------------------------
# The stamp on <html>
# ---------------------------------------------------------------------------


def _html_tag(body: str) -> str:
    """The page's opening ``<html …>`` tag."""
    match = re.search(r"<html\b[^>]*>", body)
    assert match, "the page has no <html> element"
    return match.group(0)


@pytest.mark.parametrize(
    ("web", "controls", "stamped"),
    [
        pytest.param({}, ["ask_agent"], "ask_agent", id="default"),
        pytest.param(
            {"tile_bar": {"controls": ["reload", "help"]}},
            ["reload", "help"],
            "reload help",
            id="configured",
        ),
        pytest.param({"tile_bar": {"controls": []}}, [], "", id="empty"),
    ],
)
def test_the_controls_are_stamped_on_html(workspace_dir, web, controls, stamped) -> None:
    with terminal_client(
        workspace_dir,
        patch("osprey.interfaces.web_terminal.app._load_web_ui_config", return_value=web),
    ) as client:
        assert client.app.state.tile_bar_controls == controls
        tag = _html_tag(client.get("/").text)
    assert f'data-tile-bar-controls="{stamped}"' in tag


def test_the_docs_root_is_stamped_on_html(workspace_dir) -> None:
    with started_client(
        workspace_dir, config_values={"web.docs_url": "https://docs.example.org/osprey"}
    ) as client:
        tag = _html_tag(client.get("/").text)
    assert 'data-docs-url="https://docs.example.org/osprey"' in tag


def test_a_blank_docs_url_leaves_the_docs_root_out(workspace_dir) -> None:
    with started_client(workspace_dir, config_values={"web.docs_url": ""}) as client:
        tag = _html_tag(client.get("/").text)
    assert "data-docs-url" not in tag
    assert 'data-tile-bar-controls="ask_agent"' in tag
