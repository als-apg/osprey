"""Tree navigation behaviour of the hierarchical query mixin.

Drives a real :class:`HierarchicalChannelDatabase` -- never a mock -- through
``get_options_at_level`` and the ``_navigate_to_node`` /
``_extract_tree_options`` helpers it delegates to, all in
``osprey.services.channel_finder.databases._hierarchical_query``.

Four navigation behaviours are characterised here:

* a selection that names nothing in the tree aborts navigation, and the public
  entry point turns that into an empty option list rather than an error;
* an ``instances`` level does not consume a selection -- navigation descends
  into the container whose key matches the *level name*, compared
  case-insensitively -- and simply stays put when no such container exists;
* a tree selection that is an *expanded* instance (``CH-1`` against a tree
  holding ``CH`` with an ``_expansion``) resolves to the container that
  generates it;
* a tree child carrying an ``_expansion`` is expanded inline, so the option
  list offers the generated instances instead of the container name.

Every database is built from JSON under ``tmp_path`` via the shared fixtures in
``tests/services/channel_finder/conftest.py``; the one shape not in that family
is built locally below. Nothing here touches the network, a container, or the
repository tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


def _names(options: list[dict[str, str]]) -> list[str]:
    """Return just the ``name`` field of each option, in order."""
    return [option["name"] for option in options]


def _mixed_case_container_doc() -> dict:
    """An ``instances`` level whose container is spelled ``Device``, not ``DEVICE``.

    The conftest family always spells instance containers in upper case, which
    leaves the ``key.upper() == prev_level.upper()`` comparison in
    ``_navigate_to_node`` indistinguishable from a plain ``key == level.upper()``
    check. This shape separates the two: the level is named ``device`` and the
    container ``Device``, so navigation only reaches ``X`` if both sides are
    folded. ``_hierarchical_loading._find_level_container`` folds the same way,
    so the document loads.
    """
    return {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}:{device}:{signal}",
        },
        "tree": {
            "SR": {
                "_description": "Storage Ring",
                "Device": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {"_description": "Horizontal position"},
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Tree descent by selection
# ---------------------------------------------------------------------------


class TestTreeDescent:
    """``_navigate_to_node`` following ordinary ``tree``-level selections."""

    def test_first_level_needs_no_selections(self, tree_only_db: HierarchicalChannelDatabase):
        """At the first level the loop breaks immediately and the root is used."""
        options = tree_only_db.get_options_at_level("system", {})

        assert _names(options) == ["SR", "BR"]
        assert options[0]["description"] == "Storage Ring"

    def test_selection_descends_one_level(self, tree_only_db: HierarchicalChannelDatabase):
        """A matching selection moves the cursor into that child node."""
        options = tree_only_db.get_options_at_level("device", {"system": "SR"})

        assert _names(options) == ["BPM", "QUAD"]
        assert options[0]["description"] == "Beam Position Monitor"

    def test_missing_description_becomes_empty_string(
        self, tree_only_db: HierarchicalChannelDatabase
    ):
        """A child without ``_description`` still yields an option, with ``""``."""
        options = tree_only_db.get_options_at_level("signal", {"system": "SR", "device": "QUAD"})

        assert options == [{"name": "Setpoint", "description": ""}]

    def test_list_valued_selection_uses_its_first_entry(
        self, tree_only_db: HierarchicalChannelDatabase
    ):
        """Selections may arrive as lists; navigation follows the first element."""
        options = tree_only_db.get_options_at_level("device", {"system": ["SR", "BR"]})

        assert _names(options) == ["BPM", "QUAD"]


# ---------------------------------------------------------------------------
# Navigation failure -> empty option list
# ---------------------------------------------------------------------------


class TestInvalidPath:
    """A selection matching neither a child nor an expansion aborts the walk."""

    def test_unknown_selection_navigates_to_none(self, tree_only_db: HierarchicalChannelDatabase):
        """No child ``NOPE`` and no ``_expansion`` to fall back on -> ``None``."""
        assert tree_only_db._navigate_to_node("device", {"system": "NOPE"}) is None

    def test_unknown_selection_yields_no_options(self, tree_only_db: HierarchicalChannelDatabase):
        """The public entry point reports the dead end as an empty list."""
        assert tree_only_db.get_options_at_level("device", {"system": "NOPE"}) == []
        # ... and the same call with a real system does return options, so the
        # empty list above is the navigation failure and not an empty fixture.
        assert tree_only_db.get_options_at_level("device", {"system": "SR"}) != []

    def test_empty_selection_list_yields_no_options(
        self, tree_only_db: HierarchicalChannelDatabase
    ):
        """An empty list collapses to ``None``, which matches no child."""
        assert tree_only_db.get_options_at_level("device", {"system": []}) == []


# ---------------------------------------------------------------------------
# Instance levels: descend into the container, ignore the selection
# ---------------------------------------------------------------------------


class TestInstanceLevelDescent:
    """``instances`` levels move the cursor into their container, not past it."""

    def test_descends_into_container_named_after_the_level(
        self, range_instances_db: HierarchicalChannelDatabase
    ):
        """``device`` (instances) enters ``DEVICE`` so ``signal`` sees its children."""
        options = range_instances_db.get_options_at_level(
            "signal", {"system": "SR", "family": "BPM"}
        )

        assert _names(options) == ["X", "Y"]
        assert options[0]["description"] == "Horizontal position"

    def test_instance_selection_does_not_move_the_cursor(
        self, range_instances_db: HierarchicalChannelDatabase
    ):
        """Passing the selected instance changes nothing: the container is entered."""
        without_selection = range_instances_db.get_options_at_level(
            "signal", {"system": "SR", "family": "QUAD"}
        )
        with_selection = range_instances_db.get_options_at_level(
            "signal", {"system": "SR", "family": "QUAD", "device": "Q2"}
        )

        assert _names(without_selection) == ["Setpoint", "Readback"]
        assert with_selection == without_selection

    def test_container_lookup_is_case_insensitive(self, hier_db_factory):
        """A ``Device`` container is found for a ``device`` level."""
        db = hier_db_factory(_mixed_case_container_doc(), "mixed_case_container.json")

        options = db.get_options_at_level("signal", {"system": "SR"})

        assert options == [{"name": "X", "description": "Horizontal position"}]

    def test_absent_container_leaves_the_cursor_in_place(
        self, optional_instance_level_db: HierarchicalChannelDatabase
    ):
        """``BR`` has no ``DEVICE`` container, so ``signal`` sees ``BR``'s own children."""
        options = optional_instance_level_db.get_options_at_level("signal", {"system": "BR"})

        assert options == [{"name": "Status", "description": ""}]

    def test_sibling_branch_with_a_container_still_descends(
        self, optional_instance_level_db: HierarchicalChannelDatabase
    ):
        """The same optional level does descend under ``SR``, which has the container."""
        options = optional_instance_level_db.get_options_at_level("signal", {"system": "SR"})

        assert options == [{"name": "X", "description": "Horizontal position"}]


# ---------------------------------------------------------------------------
# Tree selection that is an expanded instance
# ---------------------------------------------------------------------------


class TestExpandedInstanceSelection:
    """A ``tree``-level selection may name an instance generated by ``_expansion``."""

    def test_expanded_instance_resolves_to_its_container(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """``device="CH-1"`` is not a tree key; the ``CH`` container generates it."""
        options = hybrid_expansion_db.get_options_at_level(
            "signal", {"system": "SR", "device": "CH-1"}
        )

        assert _names(options) == ["Voltage", "Current"]

    def test_every_generated_instance_resolves_alike(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """The fallback is not anchored to the first instance of the range."""
        for instance in ("CH-1", "CH-2", "CH-3"):
            node = hybrid_expansion_db._navigate_to_node(
                "signal", {"system": "SR", "device": instance}
            )

            assert node is not None
            assert node["_description"] == "Corrector channels"

    def test_direct_child_selection_skips_the_fallback(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """A sibling that *is* a tree key is matched directly, expansions untouched."""
        options = hybrid_expansion_db.get_options_at_level(
            "signal", {"system": "SR", "device": "PS"}
        )

        assert options == [{"name": "Status", "description": ""}]

    def test_instance_outside_the_range_is_still_invalid(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """``CH-9`` looks like an instance but the expansion never generates it."""
        assert (
            hybrid_expansion_db._navigate_to_node("signal", {"system": "SR", "device": "CH-9"})
            is None
        )
        assert (
            hybrid_expansion_db.get_options_at_level("signal", {"system": "SR", "device": "CH-9"})
            == []
        )


# ---------------------------------------------------------------------------
# Inline expansion of tree children
# ---------------------------------------------------------------------------


class TestInlineExpansionOptions:
    """``_extract_tree_options`` expands children that carry an ``_expansion``."""

    def test_expanding_child_is_offered_as_its_instances(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """``CH`` is replaced by ``CH-1``..``CH-3``; the plain sibling stays as-is."""
        options = hybrid_expansion_db.get_options_at_level("device", {"system": "SR"})

        assert _names(options) == ["CH-1", "CH-2", "CH-3", "PS"]
        assert "CH" not in _names(options)

    def test_expanded_options_are_described_by_index(
        self, hybrid_expansion_db: HierarchicalChannelDatabase
    ):
        """Generated instances carry a positional description, not the container's."""
        options = hybrid_expansion_db.get_options_at_level("device", {"system": "SR"})

        assert options[0] == {"name": "CH-1", "description": "Instance 1"}
        assert options[-1] == {"name": "PS", "description": "Power supply"}

    def test_metadata_and_scalar_keys_are_never_offered(
        self, scalar_leaves_db: HierarchicalChannelDatabase
    ):
        """Underscore keys and non-dict children are both skipped."""
        options = scalar_leaves_db.get_options_at_level("device", {"system": "SR"})

        assert _names(options) == ["BPM", "QUAD"]

        scalar_children = scalar_leaves_db.get_options_at_level(
            "signal", {"system": "SR", "device": "BPM"}
        )

        assert scalar_children == []
