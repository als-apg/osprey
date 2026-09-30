"""The deployment default for the bar hosts: ``web.bar_items``.

One bespoke fail-open coercion, in the shape ``_load_panel_presets`` set: a
malformed entry is warned about and dropped, the good entries around it
survive, and nothing here may stop the terminal from booting. A deployment
that cannot be read renders the shipped arrangement rather than an empty page.

Two shapes are used. Most tests call :func:`_load_bar_items` directly against a
``config.yml`` on disk, with a deployment context that offers every gated item,
so the coercion is the unit under test and the availability filter drops
nothing. ``TestLifespanWiring`` runs a full ``create_app`` lifespan to prove the
resolved document really does reach ``app.state.bar_layout`` — the seam
``effective_bar_layout`` reads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal.app import (
    BAR_ITEM_GATES,
    BAR_ITEM_OPTIONS,
    BAR_LAYOUT_VERSION,
    DEFAULT_BAR_LAYOUT,
    MAX_BAR_ITEMS_PER_HOST,
    _load_bar_items,
    bar_availability_context,
    effective_bar_layout,
    renderable_bar_layout,
)

#: A deployment that can show every gated item.
_OFFERS_EVERYTHING = bar_availability_context(
    identity_available=True, bluesky_available=True, system_health_available=True
)

#: A single-user deployment without the SYSTEM panel or the Bluesky bridge: the
#: shape the shipped default degrades on.
_BARE = bar_availability_context(
    identity_available=False, bluesky_available=False, system_health_available=False
)


def _write_config(tmp_path: Path, bar_items: object) -> Path:
    """A ``config.yml`` whose ``web:`` section carries *bar_items*."""
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"web": {"bar_items": bar_items}}), encoding="utf-8")
    return path


def _types(items: list[dict]) -> list[str]:
    return [item["type"] for item in items]


def _every_allowed_option_value():
    """Every value the catalog allows, one param per ``(type, option, value)``.

    An enum gives each of its values, a boolean both, a number its bounds.
    """
    for item_type, specs in BAR_ITEM_OPTIONS.items():
        for name, spec in specs.items():
            kind = spec["kind"]
            if kind == "enum":
                values: tuple = tuple(spec["values"])
            elif kind == "boolean":
                values = (True, False)
            else:
                values = (spec["min"], spec["max"])
            for value in values:
                yield pytest.param(item_type, name, value, id=f"{item_type}.{name}={value!r}")


class TestAbsentAndUnreadable:
    """Nothing configured, and nothing readable, both render the shipped bars."""

    @pytest.mark.parametrize("config_file_exists", [True, False], ids=["no-key", "no-file"])
    def test_absent_block_yields_the_shipped_default(self, tmp_path, config_file_exists):
        path = tmp_path / "config.yml"
        if config_file_exists:
            path.write_text(yaml.safe_dump({"web": {"theme": "main"}}), encoding="utf-8")
        assert _load_bar_items(path, context=_OFFERS_EVERYTHING) == DEFAULT_BAR_LAYOUT

    def test_unreadable_config_never_raises(self, tmp_path):
        """A config read that blows up is a warning, not a failed boot."""
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_ui_config",
            side_effect=RuntimeError("config.yml is a directory"),
        ):
            result = _load_bar_items(tmp_path / "config.yml", context=_OFFERS_EVERYTHING)
        assert result == DEFAULT_BAR_LAYOUT

    @pytest.mark.parametrize("raw", ["not-a-mapping", 7, ["header"]])
    def test_wrong_top_level_type_yields_the_shipped_default(self, tmp_path, raw, caplog):
        path = _write_config(tmp_path, raw)
        with caplog.at_level(logging.WARNING):
            result = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert result == DEFAULT_BAR_LAYOUT
        assert "web.bar_items" in caplog.text


class TestValidBlocks:
    """A well-formed block is honoured, in both entry spellings."""

    def test_string_entries_become_items(self, tmp_path):
        path = _write_config(tmp_path, {"header": ["logo", "space", "display"]})
        layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["header"]) == ["logo", "space", "display"]
        # Config is a default, not a saved document: it starts at rev 0.
        assert layout["rev"] == 0

    def test_mapping_entries_keep_their_options(self, tmp_path):
        path = _write_config(tmp_path, {"status": [{"type": "clock", "options": {"zone": "utc"}}]})
        layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [{"type": "clock", "options": {"zone": "utc"}}]

    def test_the_facility_zone_is_a_clock_option(self, tmp_path):
        entry = {"type": "clock", "options": {"zone": "facility"}}
        path = _write_config(tmp_path, {"status": [entry]})
        layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [entry]

    def test_an_unconfigured_host_keeps_the_shipped_order(self, tmp_path):
        """Configuring one bar must not silently empty the other."""
        path = _write_config(tmp_path, {"header": ["logo"]})
        layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["header"]) == ["logo"]
        assert layout["status"] == DEFAULT_BAR_LAYOUT["status"]

    def test_an_explicitly_empty_host_is_honoured(self, tmp_path):
        path = _write_config(tmp_path, {"status": []})
        assert _load_bar_items(path, context=_OFFERS_EVERYTHING)["status"] == []

    @pytest.mark.parametrize(("hidden", "shown"), [("status", "header"), ("header", "status")])
    def test_a_visibility_flag_is_honoured_on_its_own(self, tmp_path, hidden, shown):
        path = _write_config(tmp_path, {f"{hidden}_visible": False})
        layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout[f"{hidden}_visible"] is False
        assert layout[f"{shown}_visible"] is True

    def test_the_default_document_is_not_mutated(self, tmp_path):
        """An unconfigured bar is handed out as a copy of the shipped one, so a
        caller editing the result cannot reach the process-wide default."""
        before = [dict(item) for item in DEFAULT_BAR_LAYOUT["status"]]
        layout = _load_bar_items(
            _write_config(tmp_path, {"header": ["logo"]}), context=_OFFERS_EVERYTHING
        )
        layout["status"].append({"type": "separator"})
        assert DEFAULT_BAR_LAYOUT["status"] == before


class TestDropRules:
    """Malformed entries are dropped one at a time, never the whole key."""

    def test_unknown_type_is_dropped_with_a_warning(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"header": ["logo", "teleporter", "display"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["header"]) == ["logo", "display"]
        assert "teleporter" in caplog.text

    def test_any_type_may_sit_in_either_bar(self, tmp_path, caplog):
        """``logo`` used to be header-only; the status bar now keeps it."""
        path = _write_config(tmp_path, {"header": [], "status": ["clock", "logo"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["status"]) == ["clock", "logo"]
        assert [
            record
            for record in caplog.records
            if record.name == "osprey.interfaces.web_terminal.app"
            and record.levelno >= logging.WARNING
        ] == []

    def test_a_second_copy_of_a_single_node_type_is_dropped_with_a_warning(self, tmp_path, caplog):
        """Counted across both bars, header first: the status-bar ``docs`` is
        the copy. A type the catalog marks multi (``separator``) repeats."""
        path = _write_config(
            tmp_path,
            {"header": ["logo", "docs", "separator"], "status": ["docs", "separator", "clock"]},
        )
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["header"]) == ["logo", "docs", "separator"]
        assert _types(layout["status"]) == ["separator", "clock"]
        assert "web.bar_items.status[0] places 'docs' a second time" in caplog.text

    def test_a_configured_bar_counts_against_the_shipped_order_of_the_other(self, tmp_path, caplog):
        """Only the status bar is configured, so the header keeps the shipped
        order: the ``logo`` it places is a second copy in the status bar, the
        ``docs`` it does not place stays available, and a second ``space`` is
        fine either way."""
        path = _write_config(tmp_path, {"status": ["docs", "logo", "space", "space"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["status"]) == ["docs", "space", "space"]
        assert "web.bar_items.status[1] places 'logo' a second time" in caplog.text

    @pytest.mark.parametrize("entry", [42, None, [], {}, {"type": 5}, {"nope": "logo"}])
    def test_malformed_entry_is_dropped_and_neighbours_survive(self, tmp_path, entry, caplog):
        path = _write_config(tmp_path, {"header": ["logo", entry, "display"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["header"]) == ["logo", "display"]
        assert "web.bar_items.header" in caplog.text

    def test_non_mapping_options_are_dropped_but_the_item_survives(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"status": [{"type": "clock", "options": "utc"}]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [{"type": "clock"}]
        assert "options" in caplog.text

    @pytest.mark.parametrize(
        ("item_type", "options", "kept", "named"),
        [
            pytest.param(
                "clock",
                {"zone": "UTC", "format": "12h"},
                {"format": "12h"},
                "options.zone",
                id="clock-zone-wrong-case",
            ),
            pytest.param(
                "clock", {"seconds": "yes"}, {}, "options.seconds", id="clock-seconds-str"
            ),
            pytest.param("space", {"width": 5000}, {}, "options.width", id="space-width-over-max"),
            pytest.param("space", {"width": "120"}, {}, "options.width", id="space-width-str"),
            pytest.param("space", {"width": True}, {}, "options.width", id="space-width-bool"),
        ],
    )
    def test_an_option_value_outside_its_spec_is_dropped_with_a_warning(
        self, tmp_path, caplog, item_type, options, kept, named
    ):
        path = _write_config(tmp_path, {"status": [{"type": item_type, "options": options}]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [{"type": item_type, "options": kept}]
        assert f"web.bar_items.status[0].{named}" in caplog.text

    @pytest.mark.parametrize(
        ("item_type", "options", "kept", "named"),
        [
            pytest.param(
                "clock", {"zone": "utc", "tz": "utc"}, {"zone": "utc"}, "tz", id="clock-tz"
            ),
            # A type that takes no options at all: every key is one it does not take.
            pytest.param("logo", {"size": 3}, {}, "size", id="logo-size"),
        ],
    )
    def test_an_option_the_type_does_not_take_is_dropped_with_a_warning(
        self, tmp_path, caplog, item_type, options, kept, named
    ):
        # An empty header frees the logo the shipped header places.
        path = _write_config(
            tmp_path, {"header": [], "status": [{"type": item_type, "options": options}]}
        )
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [{"type": item_type, "options": kept}]
        assert f"web.bar_items.status[0].options.{named}" in caplog.text

    @pytest.mark.parametrize(("item_type", "name", "value"), _every_allowed_option_value())
    def test_every_value_the_catalog_allows_is_kept(self, tmp_path, caplog, item_type, name, value):
        item = {"type": item_type, "options": {name: value}}
        path = _write_config(tmp_path, {"status": [item]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["status"] == [item]
        assert [
            record
            for record in caplog.records
            if record.name == "osprey.interfaces.web_terminal.app"
            and record.levelno >= logging.WARNING
        ] == []

    def test_a_host_that_is_not_a_list_falls_back_to_the_shipped_order(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"header": "logo"})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout["header"] == DEFAULT_BAR_LAYOUT["header"]
        assert "web.bar_items.header" in caplog.text

    @pytest.mark.parametrize("flag", ["status_visible", "header_visible"])
    def test_a_non_boolean_visibility_flag_falls_back(self, tmp_path, caplog, flag):
        path = _write_config(tmp_path, {flag: "yes"})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert layout[flag] is True
        assert f"web.bar_items.{flag}" in caplog.text

    def test_a_host_over_the_cap_is_truncated_with_a_warning(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"status": ["clock"] * (MAX_BAR_ITEMS_PER_HOST + 3)})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert len(layout["status"]) == MAX_BAR_ITEMS_PER_HOST
        assert "web.bar_items.status" in caplog.text


class TestLifespanWiring:
    """The resolved document reaches the seam the renderer reads."""

    def test_configured_layout_lands_on_app_state(self, bar_items_app):
        with bar_items_app(web={"bar_items": {"header": ["logo", "display"]}}) as client:
            app = client.app
            assert _types(app.state.bar_layout["header"]) == ["logo", "display"]
            assert effective_bar_layout(app) is app.state.bar_layout

    def test_the_default_is_filtered_by_what_the_deployment_renders(self, bar_items_app):
        """No SYSTEM panel and no user: the rev-0 document the lifespan leaves
        on state names neither ``system-health`` nor ``identity``, so the
        browser's normalizer has nothing to drop and nothing to latch on."""
        with bar_items_app(
            web={}, env={"OSPREY_TERMINAL_USER": "", "OSPREY_WEB_APP_NAME": ""}
        ) as client:
            app = client.app
            layout = app.state.bar_layout
            assert _types(layout["status"]) == ["space", "clock"]
            assert "identity" not in _types(layout["header"])
            assert effective_bar_layout(app) is layout


class TestUnrenderableItemsLeaveTheDefault:
    """The deployment default is renderable by construction.

    ``bar_render_plan`` already drops a gated item the deployment cannot show;
    the document behind the paint must drop it too, or the browser reads the
    difference as lost content and latches Customize read-only (#863). An
    authored item is warned about by position, because an operator wrote it;
    an item the shipped order supplied is dropped quietly, because nobody did.
    """

    def test_a_deployment_that_renders_everything_gets_the_shipped_default_itself(self, tmp_path):
        assert _load_bar_items(tmp_path / "nope.yml", context=_OFFERS_EVERYTHING) == (
            DEFAULT_BAR_LAYOUT
        )

    def test_a_bare_deployment_gets_the_default_less_the_gated_items(self, tmp_path):
        layout = _load_bar_items(tmp_path / "nope.yml", context=_BARE)
        assert _types(layout["status"]) == ["space", "clock"]
        assert _types(layout["header"]) == ["logo", "space", "control-target", "search", "display"]
        assert layout["rev"] == 0 and layout["version"] == BAR_LAYOUT_VERSION
        assert layout["header_visible"] is True and layout["status_visible"] is True

    def test_the_shipped_constant_is_not_mutated_by_the_filter(self, tmp_path):
        before = {host: list(DEFAULT_BAR_LAYOUT[host]) for host in ("header", "status")}
        _load_bar_items(tmp_path / "nope.yml", context=_BARE)
        assert {host: list(DEFAULT_BAR_LAYOUT[host]) for host in ("header", "status")} == before

    def test_an_authored_unrenderable_item_is_dropped_with_a_warning_by_position(
        self, tmp_path, caplog
    ):
        path = _write_config(tmp_path, {"status": ["space", "system-health", "clock"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_BARE)
        assert _types(layout["status"]) == ["space", "clock"]
        assert "web.bar_items.status[1]" in caplog.text
        assert "system-health" in caplog.text

    def test_the_same_authored_item_survives_where_the_deployment_renders_it(
        self, tmp_path, caplog
    ):
        path = _write_config(tmp_path, {"status": ["space", "system-health", "clock"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_OFFERS_EVERYTHING)
        assert _types(layout["status"]) == ["space", "system-health", "clock"]
        assert caplog.text == ""

    def test_an_unconfigured_host_is_filtered_without_a_warning(self, tmp_path, caplog):
        """The status bar came from the shipped order, not from the operator:
        ``system-health`` leaves it, and no line blames ``web.bar_items`` for
        an item the operator never wrote."""
        path = _write_config(tmp_path, {"header": ["logo", "display"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_BARE)
        assert _types(layout["status"]) == ["space", "clock"]
        assert _types(layout["header"]) == ["logo", "display"]
        assert caplog.text == ""

    def test_the_warning_names_the_gate_the_item_depends_on(self, tmp_path, caplog):
        path = _write_config(tmp_path, {"header": ["logo", "identity", "bluesky-queue"]})
        with caplog.at_level(logging.WARNING):
            layout = _load_bar_items(path, context=_BARE)
        assert _types(layout["header"]) == ["logo"]
        assert "web.bar_items.header[1]" in caplog.text
        assert "web.bar_items.header[2]" in caplog.text
        assert "identity" in caplog.text and "bluesky-queue" in caplog.text
        # What each item needs, in the words an operator acts on.
        assert BAR_ITEM_GATES["identity"] in caplog.text
        assert BAR_ITEM_GATES["bluesky-queue"] in caplog.text

    def test_renderable_bar_layout_copies_and_keeps_the_envelope(self):
        source = {**DEFAULT_BAR_LAYOUT, "rev": 0, "status_visible": False}
        layout = renderable_bar_layout(source, context=_BARE)
        assert layout is not source
        assert _types(layout["status"]) == ["space", "clock"]
        assert layout["status_visible"] is False
        assert layout["rev"] == 0
        assert layout["version"] == BAR_LAYOUT_VERSION
        assert _types(source["status"]) == ["space", "system-health", "clock"]
