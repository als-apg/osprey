"""The deployment default for the bar hosts: ``web.bar_items``.

One bespoke fail-open coercion, in the shape ``_load_panel_presets`` set: a
malformed entry is warned about and dropped, the good entries around it
survive, and nothing here may stop the terminal from booting. A deployment
that cannot be read renders the shipped arrangement rather than an empty page.

Two shapes are used. Most tests call :func:`_load_bar_items` directly against a
``config.yml`` on disk, because the coercion is the unit under test. The last
class runs a full ``create_app`` lifespan to prove the resolved document really
does reach ``app.state.bar_layout`` — the seam ``effective_bar_layout`` reads.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    BAR_LAYOUT_VERSION,
    DEFAULT_BAR_LAYOUT,
    MAX_BAR_ITEMS_PER_HOST,
    _load_bar_items,
    create_app,
    effective_bar_layout,
)


def _write_config(tmp_path: Path, bar_items: object) -> Path:
    """A ``config.yml`` whose ``web:`` section carries *bar_items*."""
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"web": {"bar_items": bar_items}}), encoding="utf-8")
    return path


def _types(items: list[dict]) -> list[str]:
    return [item["type"] for item in items]


class TestAbsentAndUnreadable:
    """Nothing configured, and nothing readable, both render the shipped bars."""

    def test_absent_block_yields_the_shipped_default(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(yaml.safe_dump({"web": {"theme": "main"}}), encoding="utf-8")
        assert _load_bar_items(path).layout == DEFAULT_BAR_LAYOUT

    def test_absent_config_file_yields_the_shipped_default(self, tmp_path):
        assert _load_bar_items(tmp_path / "nope.yml").layout == DEFAULT_BAR_LAYOUT

    def test_unreadable_config_never_raises(self, tmp_path):
        """A config read that blows up is a warning, not a failed boot."""
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_ui_config",
            side_effect=RuntimeError("config.yml is a directory"),
        ):
            result = _load_bar_items(tmp_path / "config.yml")
        assert result.layout == DEFAULT_BAR_LAYOUT
        assert result.locked == []

    @pytest.mark.parametrize("raw", ["not-a-mapping", 7, ["header"]])
    def test_wrong_top_level_type_yields_the_shipped_default(self, tmp_path, raw, caplog):
        path = _write_config(tmp_path, raw)
        with caplog.at_level(logging.WARNING):
            result = _load_bar_items(path)
        assert result.layout == DEFAULT_BAR_LAYOUT
        assert "web.bar_items" in caplog.text

    def test_default_carries_this_build_s_schema_version(self, tmp_path):
        assert _load_bar_items(tmp_path / "nope.yml").layout["version"] == BAR_LAYOUT_VERSION


class TestValidBlocks:
    """A well-formed block is honoured, in both entry spellings."""

    def test_string_entries_become_items(self, tmp_path):
        path = _write_config(tmp_path, {"header": ["logo", "space", "display"]})
        layout = _load_bar_items(path).layout
        assert _types(layout["header"]) == ["logo", "space", "display"]

    def test_mapping_entries_keep_their_options(self, tmp_path):
        path = _write_config(tmp_path, {"status": [{"type": "clock", "options": {"zone": "utc"}}]})
        layout = _load_bar_items(path).layout
        assert layout["status"] == [{"type": "clock", "options": {"zone": "utc"}}]

    def test_an_unconfigured_host_keeps_the_shipped_order(self, tmp_path):
        """Configuring one bar must not silently empty the other."""
        path = _write_config(tmp_path, {"header": ["logo"]})
        layout = _load_bar_items(path).layout
        assert _types(layout["header"]) == ["logo"]
        assert layout["status"] == DEFAULT_BAR_LAYOUT["status"]

    def test_an_explicitly_empty_host_is_honoured(self, tmp_path):
        path = _write_config(tmp_path, {"status": []})
        assert _load_bar_items(path).layout["status"] == []

    def test_status_visible_is_honoured(self, tmp_path):
        path = _write_config(tmp_path, {"status_visible": False})
        assert _load_bar_items(path).layout["status_visible"] is False

    def test_configured_document_is_unsaved(self, tmp_path):
        """Config is a default, not a saved document: it starts at rev 0."""
        path = _write_config(tmp_path, {"header": ["logo"]})
        assert _load_bar_items(path).layout["rev"] == 0

    def test_the_default_document_is_not_mutated(self, tmp_path):
        """Two loads of a configured header leave the shipped default intact."""
        before = [dict(item) for item in DEFAULT_BAR_LAYOUT["header"]]
        _load_bar_items(_write_config(tmp_path, {"header": ["logo"]}))
        assert DEFAULT_BAR_LAYOUT["header"] == before


class TestDropRules:
    """Malformed entries are dropped one at a time, never the whole key."""

    def test_unknown_type_is_dropped_with_a_warning(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"header": ["logo", "teleporter", "display"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert _types(layout["header"]) == ["logo", "display"]
        assert "teleporter" in caplog.text

    def test_host_mismatch_is_dropped_with_a_warning(self, tmp_path, caplog):
        """``logo`` is header-only; naming it in the status bar drops it."""
        path = _write_config(tmp_path, {"status": ["clock", "logo"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert _types(layout["status"]) == ["clock"]
        assert "logo" in caplog.text

    @pytest.mark.parametrize("entry", [42, None, [], {}, {"type": 5}, {"nope": "logo"}])
    def test_malformed_entry_is_dropped_and_neighbours_survive(self, tmp_path, entry, caplog):
        path = _write_config(tmp_path, {"header": ["logo", entry, "display"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert _types(layout["header"]) == ["logo", "display"]
        assert "web.bar_items.header" in caplog.text

    def test_non_mapping_options_are_dropped_but_the_item_survives(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"status": [{"type": "clock", "options": "utc"}]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert layout["status"] == [{"type": "clock"}]
        assert "options" in caplog.text

    def test_a_host_that_is_not_a_list_falls_back_to_the_shipped_order(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"header": "logo"})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert layout["header"] == DEFAULT_BAR_LAYOUT["header"]
        assert "web.bar_items.header" in caplog.text

    def test_a_non_boolean_status_visible_falls_back(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"status_visible": "yes"})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert layout["status_visible"] is True
        assert "web.bar_items.status_visible" in caplog.text

    def test_a_host_over_the_cap_is_truncated_with_a_warning(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"status": ["clock"] * (MAX_BAR_ITEMS_PER_HOST + 3)})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path).layout
        assert len(layout["status"]) == MAX_BAR_ITEMS_PER_HOST
        assert "web.bar_items.status" in caplog.text


class TestLocked:
    """``locked`` names types this deployment's users may never remove."""

    def test_locked_ids_are_resolved(self, tmp_path):
        path = _write_config(tmp_path, {"locked": ["clock", "docs"]})
        assert _load_bar_items(path).locked == ["clock", "docs"]

    def test_unknown_locked_id_is_dropped_with_a_warning(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"locked": ["clock", "teleporter"]})
        with caplog.at_level(logging.WARNING):
            result = _load_bar_items(path)
        assert result.locked == ["clock"]
        assert "teleporter" in caplog.text

    @pytest.mark.parametrize("raw", ["clock", 3, {"clock": True}])
    def test_a_locked_value_that_is_not_a_list_is_dropped(self, tmp_path, raw, caplog):
        path = _write_config(tmp_path, {"locked": raw})
        with caplog.at_level(logging.WARNING):
            result = _load_bar_items(path)
        assert result.locked == []
        assert "web.bar_items.locked" in caplog.text

    def test_absent_locked_is_empty(self, tmp_path):
        """The catalog's own locked types are the client's business, not config's."""
        path = _write_config(tmp_path, {"header": ["logo"]})
        assert _load_bar_items(path).locked == []


class TestLifespanWiring:
    """The resolved document reaches the seam the renderer reads."""

    @contextmanager
    def _app(self, tmp_path: Path, bar_items: object | None):
        web_section: dict = {} if bar_items is None else {"bar_items": bar_items}
        project_dir = tmp_path / "project"
        project_dir.mkdir(exist_ok=True)
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_ui_config",
            return_value=web_section,
        ):
            app = create_app(shell_command="echo", project_dir=project_dir)
            with TestClient(app):
                yield app

    def test_configured_layout_lands_on_app_state(self, tmp_path):
        with self._app(tmp_path, {"header": ["logo", "display"], "locked": ["docs"]}) as app:
            assert _types(app.state.bar_layout["header"]) == ["logo", "display"]
            assert app.state.bar_items_locked == ["docs"]
            assert effective_bar_layout(app) is app.state.bar_layout

    def test_no_config_leaves_the_shipped_default_in_place(self, tmp_path):
        with self._app(tmp_path, None) as app:
            assert effective_bar_layout(app) == DEFAULT_BAR_LAYOUT
            assert app.state.bar_items_locked == []
