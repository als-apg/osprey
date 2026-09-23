"""The ``web_panels`` selection, projected onto every render's ``web.panels``.

A render's tab strip is read from ``web.panels.<id>`` alone, so the profile's
``web_panels:`` selection has to reach every block there — including the
blocks a persona *inherits* through ``config:`` for a panel it excluded, and
the blocks an injector writes into a deploying render for a tab only its
personas select. ``panel_selection_overrides`` is that projection;
``panel_selection_errors`` refuses an authored ``enabled`` that contradicts
the selection; ``panel_id_errors`` refuses an id no request for that panel
could be routed with; ``panel_spec_enabled`` is the one predicate every reader
of a block shares.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile_panels import (
    bar_items_selection_warnings,
    panel_id_errors,
    panel_selection_errors,
    panel_selection_overrides,
)
from osprey.profiles.web_panels import panel_id_refusal, panel_spec_enabled
from osprey.utils.config_writer import config_update_fields, load_config_document

# ---------------------------------------------------------------------------
# panel_spec_enabled — the shared predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(True, True, id="bare_true"),
        pytest.param({}, True, id="empty_mapping_defaults_on"),
        pytest.param({"label": "X", "url": "http://x"}, True, id="custom_block_without_enabled"),
        pytest.param({"enabled": True}, True, id="enabled_true"),
        pytest.param({"enabled": False}, False, id="enabled_false"),
        pytest.param({"url": "http://x", "enabled": False}, False, id="custom_disabled"),
        pytest.param(False, False, id="bare_false"),
        pytest.param(None, False, id="bare_key_no_value"),
        pytest.param("yes", False, id="string_is_not_a_block"),
    ],
)
def test_panel_spec_enabled(spec, expected) -> None:
    """One answer for builtin and custom blocks alike; anything that is not a
    block (or ``true``) is off — fail closed."""
    assert panel_spec_enabled(spec) is expected


# ---------------------------------------------------------------------------
# panel_selection_overrides — the projection
# ---------------------------------------------------------------------------


def test_every_block_is_told_whether_it_is_selected() -> None:
    """Selected → ``enabled: true``; every other block → ``enabled: false``.

    The inherited ``lattice`` label and the url-backed ``events`` /
    ``beam-viewer`` blocks are exactly the residue a persona that excludes
    those tabs keeps carrying — and, before the projection, exactly the tabs
    it kept showing.
    """
    rendered = {
        "web": {
            "panels": {
                "okf": {"enabled": True},
                "lattice": {"label": "LATTICE"},
                "events": {
                    "label": "EVENTS",
                    "url": "http://localhost:10010",
                    "path": "/dashboard",
                },
                "beam-viewer": {"label": "BEAM", "url": "http://localhost:10920"},
            }
        }
    }
    assert panel_selection_overrides(["okf"], rendered) == {
        "web.panels.okf.enabled": True,
        "web.panels.lattice.enabled": False,
        "web.panels.events.enabled": False,
        "web.panels.beam-viewer.enabled": False,
    }


def test_a_selected_custom_block_is_switched_on_explicitly() -> None:
    """A custom panel has no template line writing its ``enabled``; the
    projection writes it so the render says what it shows."""
    rendered = {"web": {"panels": {"grafana": {"url": "http://grafana:3000"}}}}
    assert panel_selection_overrides(["grafana"], rendered) == {"web.panels.grafana.enabled": True}


def test_universal_panels_are_always_on() -> None:
    """``artifacts`` is served regardless of selection; a block for it (which
    nothing writes, but nothing forbids) is never switched off."""
    rendered = {"web": {"panels": {"artifacts": {"label": "WORKSPACE"}}}}
    assert panel_selection_overrides([], rendered) == {"web.panels.artifacts.enabled": True}


def test_a_selection_with_no_block_projects_nothing_for_it() -> None:
    """The projection only annotates blocks that exist: a selected panel with
    no block is the template's or the injector's job (and, for a Reach tab,
    ``selected_panel_errors``'s refusal), not this function's."""
    assert panel_selection_overrides(["okf", "events"], {}) == {}
    assert panel_selection_overrides(["okf"], {"web": {"ui_mode": "expert"}}) == {}


def test_a_non_mapping_block_is_projected_too() -> None:
    """``okf: true`` is a legal spelling of an enabled builtin; the projection
    still states the selection for it rather than skipping it."""
    rendered = {"web": {"panels": {"okf": True, "ariel": True}}}
    assert panel_selection_overrides(["okf"], rendered) == {
        "web.panels.okf.enabled": True,
        "web.panels.ariel.enabled": False,
    }


# ---------------------------------------------------------------------------
# panel_selection_errors — an authored `enabled` that contradicts the selection
# ---------------------------------------------------------------------------


def test_an_authored_enabled_true_for_an_unselected_panel_is_refused() -> None:
    (error,) = panel_selection_errors({"web.panels.lattice.enabled": True}, ["okf"])
    assert "web.panels.lattice.enabled: True" in error
    assert "not in web_panels" in error
    assert "lattice" in error


def test_an_authored_enabled_false_for_a_selected_panel_is_refused() -> None:
    (error,) = panel_selection_errors({"web.panels.okf.enabled": False}, ["okf"])
    assert "web.panels.okf.enabled: False" in error
    assert "hidden: true" in error, "the way to keep a selected tab off-screen is named"


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param({"web.panels.lattice.enabled": True}, id="dotted"),
        pytest.param({"web.panels.lattice": {"enabled": True}}, id="prefix_over_mapping"),
        pytest.param({"web": {"panels": {"lattice": {"enabled": True}}}}, id="nested"),
        pytest.param({"web": {"panels.lattice.enabled": True}}, id="mixed"),
    ],
)
def test_every_spelling_of_a_contradiction_is_found(spelling) -> None:
    """Each is legal YAML for the same rendered leaf; a spelling missed here is
    a contradiction the projection would silently overwrite."""
    (error,) = panel_selection_errors(spelling, [])
    assert "lattice" in error


def test_an_agreeing_or_absent_enabled_is_allowed() -> None:
    assert panel_selection_errors({"web.panels.okf.enabled": True}, ["okf"]) == []
    assert panel_selection_errors({"web.panels.lattice.enabled": False}, ["okf"]) == []
    assert panel_selection_errors({"web.panels.lattice.label": "LATTICE"}, ["okf"]) == []
    assert panel_selection_errors({}, ["okf"]) == []
    assert panel_selection_errors(None, ["okf"]) == []


def test_universal_panels_are_never_a_contradiction() -> None:
    assert panel_selection_errors({"web.panels.artifacts.enabled": True}, []) == []


# ---------------------------------------------------------------------------
# panel_id_errors — an id the served terminal could never route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel_id", ["überblick", "beam viewer", "beam/viewer", "-x"])
def test_an_id_the_terminal_cannot_serve_is_refused(panel_id) -> None:
    """The terminal refuses to start on such an id, so the build refuses first.

    A profile that renders one builds a deployment whose every request for that
    panel fails, and a build is where the id is still a line in a file the
    operator has open.
    """
    (error,) = panel_id_errors({f"web.panels.{panel_id}.url": "http://localhost:9000"})
    assert repr(panel_id) in error


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param({"web.panels.überblick.url": "http://x:1"}, id="dotted"),
        pytest.param({"web.panels.überblick": {"url": "http://x:1"}}, id="prefix_over_mapping"),
        pytest.param({"web": {"panels": {"überblick": {"url": "http://x:1"}}}}, id="nested"),
        pytest.param({"web": {"panels.überblick.url": "http://x:1"}}, id="mixed"),
    ],
)
def test_every_spelling_of_the_block_reaches_the_check(spelling) -> None:
    (error,) = panel_id_errors(spelling)
    assert "überblick" in error


def test_a_dot_segment_block_is_refused_as_the_empty_id_it_reads_as() -> None:
    """A dot-segment cannot survive a dotted key, and is refused either way.

    ``web.panels..url`` is the same key path as a panel named ``.``, and the
    key walk splits on the dots, so the id it recovers is empty rather than a
    dot. Both are ids no request could be routed with, so the reading that
    reaches the check is refused on its own account; the terminal, which reads
    the mapping key itself, names the dot.
    """
    (error,) = panel_id_errors({"web.panels..url": "http://x:1"})
    assert "[A-Za-z0-9][A-Za-z0-9._-]*" in error


def _prefix_over_mapping(panel_id):
    return {"web.panels": {panel_id: {"url": "http://x:1"}}}


def _nested(panel_id):
    return {"web": {"panels": {panel_id: {"url": "http://x:1"}}}}


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_prefix_over_mapping, id="prefix_over_mapping"),
        pytest.param(_nested, id="nested"),
    ],
)
@pytest.mark.parametrize(
    "panel_id",
    [
        pytest.param("beam.view er", id="space_after_dot"),
        pytest.param("beam./viewer", id="slash_after_dot"),
        pytest.param(".beam", id="leading_dot"),
    ],
)
def test_a_dotted_id_written_as_a_mapping_key_is_checked_whole(panel_id, spelling) -> None:
    """The render keeps a mapping key whole and the terminal refuses what it keeps.

    So the build reads that key whole too, and refuses it before it deploys.
    """
    (error,) = panel_id_errors(spelling(panel_id))
    assert repr(panel_id) in error


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_prefix_over_mapping, id="prefix_over_mapping"),
        pytest.param(_nested, id="nested"),
    ],
)
def test_a_dotted_id_the_terminal_can_serve_passes(spelling) -> None:
    """A dot alone is inside the id class, so a dotted id the terminal serves passes."""
    assert panel_id_errors(spelling("beam.viewer")) == []


def test_a_whole_dotted_key_splits_at_every_dot() -> None:
    """The render splits a top-level key at every dot.

    This key renders panel ``beam`` with a child ``view er``, and ``beam`` is an
    id the terminal serves.
    """
    assert panel_id_errors({"web.panels.beam.view er.url": "http://x:1"}) == []


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_prefix_over_mapping("beam.view er"), id="prefix_over_mapping_refused"),
        pytest.param(_nested("beam.view er"), id="nested_refused"),
        pytest.param(_prefix_over_mapping("beam.viewer"), id="prefix_over_mapping_served"),
        pytest.param(_nested("beam.viewer"), id="nested_served"),
        pytest.param({"web.panels.beam.view er.url": "http://x:1"}, id="dotted_split"),
        pytest.param({"web.panels.überblick.url": "http://x:1"}, id="dotted_refused"),
    ],
)
def test_the_lint_reads_each_id_the_render_writes(tmp_path, spelling) -> None:
    """The build lint and the terminal share one verdict and the ids it is asked about.

    ``config_update_fields`` is what ``osprey build`` applies a profile's
    ``config:`` through, so the ids are read off its output rather than restated.
    """
    path = tmp_path / "config.yml"
    path.write_text("web:\n  panels:\n    okf:\n      enabled: true\n")
    config_update_fields(path, spelling)
    rendered = load_config_document(path)["web"]["panels"]
    assert panel_id_errors(spelling) == [
        r for pid in sorted(map(str, rendered)) if (r := panel_id_refusal(pid)) is not None
    ]


def test_an_authored_enabled_on_a_dotted_id_is_judged_under_that_id() -> None:
    """An ``enabled`` on a dotted id under a mapping is judged under the whole id."""
    config = {"web.panels": {"beam.viewer": {"url": "http://x:1", "enabled": True}}}
    (error,) = panel_selection_errors(config, [])
    assert "'beam.viewer' is not in web_panels" in error
    assert panel_selection_errors(config, ["beam.viewer"]) == []


def test_an_ordinary_id_passes() -> None:
    assert panel_id_errors({"web.panels.my-grafana.url": "http://x:1"}) == []
    assert panel_id_errors({"web.panels.okf.enabled": True}) == []
    assert panel_id_errors({}) == []
    assert panel_id_errors(None) == []


# ---------------------------------------------------------------------------
# bar_items_selection_warnings — an authored bar item the render cannot show
# ---------------------------------------------------------------------------


def test_a_panel_gated_item_without_its_panel_is_warned_about_by_position() -> None:
    """The server drops ``system-health`` from the default it serves when the
    SYSTEM panel is not selected (#863); the build says so first, naming the
    entry the operator wrote and the panel it needs."""
    config = {"web": {"bar_items": {"status": ["space", "system-health", "clock"]}}}

    warnings = bar_items_selection_warnings(config, ["ariel", "artifacts"])

    assert len(warnings) == 1
    assert warnings[0].startswith("web: bar_items: status[1] places 'system-health'")
    assert "'system-health' is not in web_panels" in warnings[0]


def test_a_selected_panel_keeps_its_item_quiet() -> None:
    config = {"web.bar_items": {"status": ["space", "system-health", "clock"]}}
    assert bar_items_selection_warnings(config, ["system-health"]) == []


def test_a_mapping_entry_is_judged_by_its_type() -> None:
    config = {
        "web.bar_items.header": [
            "logo",
            {"type": "bluesky-queue", "options": {"controls": "full"}},
        ]
    }

    warnings = bar_items_selection_warnings(config, ["ariel"])

    assert len(warnings) == 1
    assert warnings[0].startswith("web.bar_items.header[1] places 'bluesky-queue'")
    assert "needs the 'bluesky' panel" in warnings[0]


def test_both_bars_and_both_gated_items_in_bar_then_list_order() -> None:
    config = {
        "web": {
            "bar_items": {
                "header": ["logo", "bluesky-queue"],
                "status": ["system-health", "clock", "bluesky-queue"],
            }
        }
    }

    warnings = bar_items_selection_warnings(config, [])

    assert [w.split(" places ")[0] for w in warnings] == [
        "web: bar_items: header[1]",
        "web: bar_items: status[0]",
        "web: bar_items: status[2]",
    ]


def test_identity_is_not_a_build_time_judgment() -> None:
    """``identity`` is gated on a runtime fact (a terminal user or a deployment
    name) the build cannot see, so an authored ``identity`` never warns here."""
    config = {"web": {"bar_items": {"header": ["logo", "identity"]}}}
    assert bar_items_selection_warnings(config, []) == []


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="no_config"),
        pytest.param({"web": {"bar_items": None}}, id="bar_items_null"),
        pytest.param({"web": {"bar_items": {"status": "system-health"}}}, id="not_a_list"),
        pytest.param({"web": {"bar_items": {"status": [None, 3, {}]}}}, id="unreadable_entries"),
        pytest.param({"web": {"bar_items": {"status": ["space", "clock"]}}}, id="ungated_items"),
    ],
)
def test_nothing_to_judge_warns_nothing(config) -> None:
    """Shape problems are the server loader's to report; this only speaks to a
    readable entry naming a gated item."""
    assert bar_items_selection_warnings(config, []) == []


# ---------------------------------------------------------------------------
# artifact_menu_catalog — what the install skill offers
# ---------------------------------------------------------------------------


def test_every_offered_panel_carries_a_description() -> None:
    """An entry with an empty description is a menu line the reader cannot judge.

    Descriptions came from the companion-server registry alone, so a panel
    served any other way — a sidecar the web terminal starts itself — was
    offered as a bare id with a trailing blank comment.
    """
    from osprey.cli.build_profile_emit import artifact_menu_catalog
    from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS

    entries = artifact_menu_catalog()["web_panels"]

    assert {name for name, _ in entries} == BUILTIN_PANELS - UNIVERSAL_PANELS
    assert [name for name, description in entries if not description] == []
