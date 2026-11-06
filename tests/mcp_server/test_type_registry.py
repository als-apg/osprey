"""Tests for the OSPREY type registry — single source of truth for type metadata."""

import json
import re

import pytest

from osprey.stores.type_registry import (
    ARTIFACT_TYPES,
    CATEGORIES,
    TOOL_TYPES,
    get_artifact_types,
    get_categories,
    get_tool_types,
    registry_to_api_dict,
)

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class TestTypeDefs:
    """Every TypeDef must have a non-empty label and a valid 6-digit hex colour."""

    @pytest.mark.parametrize("key,td", list(ARTIFACT_TYPES.items()), ids=list(ARTIFACT_TYPES))
    def test_artifact_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"artifact type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"artifact type {key!r} has invalid colour {td.color!r}"

    @pytest.mark.parametrize("key,td", list(TOOL_TYPES.items()), ids=list(TOOL_TYPES))
    def test_tool_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"tool type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"tool type {key!r} has invalid colour {td.color!r}"


class TestSerializerCoverage:
    """Every type ``serialize_object`` can stamp is a registered one.

    The gallery reads its badge label and colour from this registry, so an
    artifact stamped with a type the registry does not carry renders with the
    raw key as its label and no colour at all — silently, in the UI only.
    """

    def test_every_emitted_artifact_type_is_registered(self):
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "src" / "osprey" / "stores" / "artifact_store.py"
        ).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "serialize_object"
        )

        # (content, artifact_type, filename, mime) — the second element of every
        # tuple this function returns.
        emitted = {
            node.value.elts[1].value
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Tuple)
            and len(node.value.elts) == 4
            and isinstance(node.value.elts[1], ast.Constant)
        }

        assert emitted, "the AST scan found no artifact types — the scan is broken"
        assert emitted <= set(ARTIFACT_TYPES)


class TestPublicAPI:
    def test_get_artifact_types_returns_copy(self):
        a = get_artifact_types()
        a["bogus"] = None  # mutate
        assert "bogus" not in ARTIFACT_TYPES

    def test_get_tool_types_returns_copy(self):
        t = get_tool_types()
        t["bogus"] = None
        assert "bogus" not in TOOL_TYPES

    def test_get_categories_returns_copy(self):
        c = get_categories()
        c["bogus"] = None
        assert "bogus" not in CATEGORIES


class TestRegistryToAPIDict:
    def test_json_serialisable(self):
        d = registry_to_api_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_structure(self):
        d = registry_to_api_dict()
        assert set(d.keys()) == {
            "artifact_types",
            "tool_types",
            "categories",
            "base_forms",
            "extension_types",
        }
        for domain in ("artifact_types", "tool_types", "categories"):
            for _key, info in d[domain].items():
                assert "label" in info
                assert "color" in info
        for form in d["base_forms"]:
            assert form["key"]
            assert form["label"]

    def test_all_artifact_types_present(self):
        d = registry_to_api_dict()
        assert set(d["artifact_types"]) == set(ARTIFACT_TYPES)

    def test_all_tool_types_present(self):
        d = registry_to_api_dict()
        assert set(d["tool_types"]) == set(TOOL_TYPES)


class TestLegacyCategoryLabels:
    """Categories are retired: nothing writes one, but an entry saved before the
    retirement still carries its ``category`` and the gallery badge labels it
    from the read-only table."""

    LEGACY_CATEGORIES = [
        "channel_values",
        "write_results",
        "code_output",
        "visualization",
        "dashboard",
        "document",
        "screenshot",
        "agent_response",
        "logbook_research",
        "search_results",
    ]

    @pytest.mark.parametrize("cat", LEGACY_CATEGORIES)
    def test_legacy_category_still_labelled(self, cat):
        assert cat in get_categories(), f"{cat!r} lost its legacy badge label"

    @pytest.mark.parametrize("key,td", list(CATEGORIES.items()), ids=list(CATEGORIES))
    def test_legacy_category_fields(self, key, td):
        assert td.key == key
        assert td.label
        assert HEX_RE.match(td.color)

    def test_channel_list_not_in_categories(self):
        """Phantom type — no producer ever existed."""
        assert "channel_list" not in CATEGORIES

    def test_category_write_api_is_gone(self):
        from osprey.stores import type_registry

        for name in ("register_category", "load_categories_from_config", "valid_category_keys"):
            assert not hasattr(type_registry, name), name
