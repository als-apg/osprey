"""A deployment's own name and glyph for a panel: ``web.panels.<id>.label`` / ``.icon``.

Both keys are read through ``osprey.profiles.web_panels``, for a built-in and
a declared panel alike, so every reader of a label answers with the same word.
"""

from __future__ import annotations

import pytest

from osprey.interfaces.design_system.icons import icon_names
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    SIDECAR_PANELS,
    authored_panel_labels,
    panel_icons,
    panel_label,
    panel_labels,
    sidecar_panel_descriptors,
)

_SIDECAR_ID = next(iter(SIDECAR_PANELS))


@pytest.mark.parametrize("panels", [None, {}, [], {"ariel": True}, {"ariel": {"enabled": True}}])
def test_an_unrenamed_deployment_reads_the_registry_labels(panels) -> None:
    assert panel_labels(panels) == BUILTIN_PANEL_LABELS


def test_a_block_label_renames_a_builtin_panel_and_no_other() -> None:
    labels = panel_labels({"ariel": {"label": "Logbook"}, "grafana": {"label": "GRAFANA"}})

    assert labels == {**BUILTIN_PANEL_LABELS, "ariel": "Logbook"}
    assert list(labels) == list(BUILTIN_PANEL_LABELS)


@pytest.mark.parametrize("label", ["", "   ", None, 7, ["Logbook"]])
def test_a_label_that_is_not_text_says_nothing(label) -> None:
    assert panel_labels({"ariel": {"label": label}}) == BUILTIN_PANEL_LABELS


def test_one_label_answers_authored_then_registry_then_the_id() -> None:
    panels = {"ariel": {"label": "Logbook"}, "grafana": {"label": "Dashboards"}, "bare": {}}

    assert panel_label("ariel", panels) == "Logbook"
    assert panel_label("grafana", panels) == "Dashboards"
    assert panel_label("okf", panels) == BUILTIN_PANEL_LABELS["okf"]
    assert panel_label("bare", panels) == "BARE"
    assert panel_label("ariel") == BUILTIN_PANEL_LABELS["ariel"]


def test_icons_are_read_for_builtin_and_declared_panels_alike() -> None:
    panels = {
        "ariel": {"icon": "book-open"},
        "grafana": {"label": "GRAFANA", "url": "http://x:3000", "icon": "chart-column"},
        "okf": True,
        "events": {"label": "EVENTS", "icon": ""},
    }

    assert panel_icons(panels) == {"ariel": "book-open", "grafana": "chart-column"}
    assert panel_icons(None) == {}


def test_a_sidecar_descriptor_carries_the_deployments_label_and_icon() -> None:
    own = {d["id"]: d for d in sidecar_panel_descriptors()}[_SIDECAR_ID]
    named = {
        d["id"]: d
        for d in sidecar_panel_descriptors({_SIDECAR_ID: {"label": "Notes", "icon": "monitor"}})
    }[_SIDECAR_ID]

    assert own["label"] == BUILTIN_PANEL_LABELS[_SIDECAR_ID]
    assert own["icon"] == SIDECAR_PANELS[_SIDECAR_ID].icon
    assert named == {**own, "label": "Notes", "icon": "monitor"}


def test_the_python_reader_names_the_glyphs_the_design_system_draws() -> None:
    names = icon_names()

    assert {"workspace", "monitor", "camera", "chart-column", "users", "radiation", "cpu"} <= names
    assert all(name == name.strip() and name for name in names)


def test_the_labels_a_deployment_wrote_are_listed_once_each() -> None:
    panels = {
        "ascc": {"label": "ASCC", "url": "http://x"},
        "ariel": {"label": "Logbook"},
        "twin": {"label": "ASCC", "url": "http://y"},
        "bare": {"url": "http://z"},
        "blank": {"label": "  "},
        "on": True,
    }
    assert authored_panel_labels(panels) == ["ASCC", "Logbook"]


@pytest.mark.parametrize("panels", [None, {}, "nope"])
def test_a_deployment_that_writes_no_label_lists_none(panels) -> None:
    assert authored_panel_labels(panels) == []
