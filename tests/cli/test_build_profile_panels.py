"""The ``web_panels`` selection, projected onto every render's ``web.panels``.

A render's tab strip is read from ``web.panels.<id>`` alone, so the profile's
``web_panels:`` selection has to reach every block there — including the
blocks a persona *inherits* through ``config:`` for a panel it excluded, and
the blocks an injector writes into a deploying render for a tab only its
personas select. ``panel_selection_overrides`` is that projection;
``panel_selection_errors`` refuses an authored ``enabled`` that contradicts
the selection; ``panel_spec_enabled`` is the one predicate every reader of a
block shares.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile_panels import (
    panel_selection_errors,
    panel_selection_overrides,
)
from osprey.profiles.web_panels import panel_spec_enabled

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
