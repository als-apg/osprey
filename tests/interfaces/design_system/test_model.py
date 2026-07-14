"""Tests for osprey.interfaces.design_system.generator.model.

Exercises the DTCG loading/flattening/alias-resolution pipeline against
small, hermetic fixture token trees under
``tests/interfaces/design_system/fixtures/`` — never against the real
``src/osprey/interfaces/design_system/tokens/`` sources, which are authored
by a separate task and may not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.interfaces.design_system.generator.model import (
    AliasStatus,
    RawToken,
    TokenModelError,
    flatten_document,
    load_json_document,
    load_token_tree,
    resolve_document,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOKENS_DIR = FIXTURES_DIR / "tokens"
ALIAS_CASES_DIR = FIXTURES_DIR / "tokens_alias_cases"


def _raw(value: object, *, path: str = "x", type_: str | None = "color") -> RawToken:
    """Build a minimal RawToken for resolve_document unit tests."""
    return RawToken(
        path=path,
        value=value,
        type=type_,
        description=None,
        extensions={},
        source_file=Path("<test>"),
    )


# --- load_json_document -----------------------------------------------------


def test_load_json_document_parses_object(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text('{"a": {"$value": "1"}}', encoding="utf-8")

    document = load_json_document(path)

    assert document == {"a": {"$value": "1"}}


def test_load_json_document_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TokenModelError, match="could not read"):
        load_json_document(tmp_path / "missing.json")


def test_load_json_document_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(TokenModelError, match="invalid JSON"):
        load_json_document(path)


def test_load_json_document_non_object_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TokenModelError, match="must be a JSON object"):
        load_json_document(path)


# --- flatten_document --------------------------------------------------------


def test_flatten_document_produces_dot_paths() -> None:
    document = {
        "color": {
            "teal": {
                "300": {"$value": "#2dd4bf", "$type": "color"},
                "500": {"$value": "#14b8a6", "$type": "color"},
            }
        }
    }

    tokens = flatten_document(document, source_file=Path("core.json"))

    assert set(tokens) == {"color.teal.300", "color.teal.500"}
    assert tokens["color.teal.300"].value == "#2dd4bf"
    assert tokens["color.teal.300"].type == "color"
    assert tokens["color.teal.300"].source_file == Path("core.json")


def test_flatten_document_preserves_source_order() -> None:
    document = {
        "b": {"$value": "2", "$type": "number"},
        "a": {"$value": "1", "$type": "number"},
    }

    tokens = flatten_document(document, source_file=Path("x.json"))

    assert list(tokens) == ["b", "a"]


def test_flatten_document_group_type_is_inherited() -> None:
    document = {
        "bg": {
            "$type": "color",
            "primary": {"$value": "#000000"},
            "panel": {"$value": "#111111", "$type": "dimension"},
        }
    }

    tokens = flatten_document(document, source_file=Path("x.json"))

    assert tokens["bg.primary"].type == "color"
    # Own $type overrides the inherited group $type.
    assert tokens["bg.panel"].type == "dimension"


def test_flatten_document_captures_description_and_extensions() -> None:
    document = {
        "text": {
            "primary": {
                "$value": "#e6edf3",
                "$type": "color",
                "$description": "Primary body text.",
                "$extensions": {"osprey.wcag": "aa"},
            }
        }
    }

    tokens = flatten_document(document, source_file=Path("x.json"))

    token = tokens["text.primary"]
    assert token.description == "Primary body text."
    assert token.extensions == {"osprey.wcag": "aa"}


def test_flatten_document_ignores_root_extensions() -> None:
    document = {
        "$extensions": {"id": "dark", "mode": "dark"},
        "bg": {"primary": {"$value": "#000", "$type": "color"}},
    }

    tokens = flatten_document(document, source_file=Path("x.json"))

    assert set(tokens) == {"bg.primary"}


def test_flatten_document_rejects_bare_value_at_root() -> None:
    with pytest.raises(TokenModelError, match="document root"):
        flatten_document({"$value": "1"}, source_file=Path("x.json"))


def test_flatten_document_rejects_value_and_children_mixed() -> None:
    document = {"bg": {"$value": "#000", "primary": {"$value": "#111"}}}

    with pytest.raises(TokenModelError, match="both \\$value and nested group"):
        flatten_document(document, source_file=Path("x.json"))


def test_flatten_document_rejects_non_object_child() -> None:
    document = {"bg": "not-an-object"}

    with pytest.raises(TokenModelError, match="expected an object"):
        flatten_document(document, source_file=Path("x.json"))


def test_flatten_document_rejects_non_string_type() -> None:
    document = {"bg": {"$value": "#000", "$type": 5}}

    with pytest.raises(TokenModelError, match="\\$type must be a string"):
        flatten_document(document, source_file=Path("x.json"))


def test_flatten_document_rejects_non_string_description() -> None:
    document = {"bg": {"$value": "#000", "$description": 5}}

    with pytest.raises(TokenModelError, match="\\$description must be a string"):
        flatten_document(document, source_file=Path("x.json"))


def test_flatten_document_rejects_non_object_extensions() -> None:
    document = {"bg": {"$value": "#000", "$extensions": "nope"}}

    with pytest.raises(TokenModelError, match="\\$extensions must be an object"):
        flatten_document(document, source_file=Path("x.json"))


def test_token_model_error_message_includes_file_and_path() -> None:
    try:
        flatten_document({"bg": "oops"}, source_file=Path("tokens/core.json"))
    except TokenModelError as exc:
        assert "tokens/core.json" in str(exc)
        assert exc.source_file == Path("tokens/core.json")
        assert exc.path == "bg"
    else:
        pytest.fail("expected TokenModelError")


# --- resolve_document ---------------------------------------------------------


def test_resolve_document_literal_is_not_alias() -> None:
    primitives = {"color.teal.500": _raw("#14b8a6")}
    tokens = {"accent.base": _raw("#ffffff")}

    resolved = resolve_document(tokens, primitives, all_known_paths=set(primitives) | set(tokens))

    token = resolved["accent.base"]
    assert token.value == "#ffffff"
    assert token.alias_status == AliasStatus.NOT_ALIAS
    assert token.alias_target is None


def test_resolve_document_resolves_one_hop_alias() -> None:
    primitives = {"color.teal.500": _raw("#14b8a6", path="color.teal.500")}
    tokens = {"accent.base": _raw("{color.teal.500}", path="accent.base")}
    all_known = set(primitives) | set(tokens)

    resolved = resolve_document(tokens, primitives, all_known)

    token = resolved["accent.base"]
    assert token.value == "#14b8a6"
    assert token.alias_status == AliasStatus.RESOLVED
    assert token.alias_target == "color.teal.500"


def test_resolve_document_flags_multi_hop_alias() -> None:
    primitives = {
        "color.teal.500": _raw("#14b8a6", path="color.teal.500"),
        "chained.one": _raw("{color.teal.500}", path="chained.one"),
    }
    tokens = {"accent.base": _raw("{chained.one}", path="accent.base")}
    all_known = set(primitives) | set(tokens)

    resolved = resolve_document(tokens, primitives, all_known)

    token = resolved["accent.base"]
    assert token.alias_status == AliasStatus.MULTI_HOP
    assert token.alias_target == "chained.one"
    # Left unresolved: the raw alias string is preserved, not a partial value.
    assert token.value == "{chained.one}"


def test_resolve_document_flags_dangling_alias() -> None:
    primitives = {"color.teal.500": _raw("#14b8a6", path="color.teal.500")}
    tokens = {"accent.base": _raw("{color.nonexistent}", path="accent.base")}
    all_known = set(primitives) | set(tokens)

    resolved = resolve_document(tokens, primitives, all_known)

    token = resolved["accent.base"]
    assert token.alias_status == AliasStatus.DANGLING
    assert token.alias_target == "color.nonexistent"


def test_resolve_document_flags_non_primitive_alias_target() -> None:
    primitives = {"color.teal.500": _raw("#14b8a6", path="color.teal.500")}
    tokens = {
        "bg.primary": _raw("#0a0f1a", path="bg.primary"),
        "accent.base": _raw("{bg.primary}", path="accent.base"),
    }
    all_known = set(primitives) | set(tokens)

    resolved = resolve_document(tokens, primitives, all_known)

    token = resolved["accent.base"]
    assert token.alias_status == AliasStatus.NOT_PRIMITIVE
    assert token.alias_target == "bg.primary"


def test_resolve_document_self_referencing_primitives_one_hop() -> None:
    # When resolving the primitives document itself, pass raw_tokens as
    # both raw_tokens and primitives (per the documented contract).
    primitives = {
        "color.teal.500": _raw("#14b8a6", path="color.teal.500"),
        "alias.brand": _raw("{color.teal.500}", path="alias.brand"),
    }

    resolved = resolve_document(primitives, primitives, all_known_paths=set(primitives))

    assert resolved["alias.brand"].value == "#14b8a6"
    assert resolved["alias.brand"].alias_status == AliasStatus.RESOLVED
    assert resolved["color.teal.500"].alias_status == AliasStatus.NOT_ALIAS


@pytest.mark.parametrize(
    "malformed",
    ["{no.closing.brace", "no.opening.brace}", "{}", "{has space}", "plain string"],
)
def test_resolve_document_rejects_malformed_alias_syntax(malformed: str) -> None:
    tokens = {"x": _raw(malformed, path="x")}

    resolved = resolve_document(tokens, primitives={}, all_known_paths=set())

    # Anything not matching {dotted.path} exactly is treated as a literal,
    # never partially parsed.
    assert resolved["x"].alias_status == AliasStatus.NOT_ALIAS
    assert resolved["x"].value == malformed


# --- load_token_tree (fixture-file integration) ------------------------------


def test_load_token_tree_happy_path() -> None:
    tree = load_token_tree(TOKENS_DIR)

    # Primitives loaded and flattened.
    assert "color.teal.500" in tree.primitives
    assert tree.primitives["color.teal.500"].value == "#14b8a6"

    # A primitive that is itself a one-hop alias to another primitive
    # resolves cleanly (core.json's alias.brand -> color.teal.500).
    assert tree.primitives["alias.brand"].value == "#14b8a6"
    assert tree.primitives["alias.brand"].alias_status == AliasStatus.RESOLVED

    # Both themes present, each with the semantic vocabulary flattened.
    assert set(tree.themes) == {"dark", "light"}
    dark = tree.themes["dark"]
    light = tree.themes["light"]
    assert dark["bg.primary"].alias_status == AliasStatus.RESOLVED
    assert dark["bg.primary"].value == "#0a0f1a"
    assert light["bg.primary"].value == "#f8fafc"

    # Literal (non-alias) semantic token passes through unchanged, with
    # its group-inherited $type and its own $description intact.
    assert dark["bg.panel"].value == "#0d1420"
    assert dark["bg.panel"].type == "color"
    assert dark["text.primary"].description == "Primary body text color."

    # Theme document root metadata is captured separately from tokens.
    assert tree.theme_metadata["dark"] == {"id": "dark", "label": "Dark", "mode": "dark"}
    assert tree.theme_metadata["light"]["mode"] == "light"

    # Interface extension tokens loaded, namespaced by file stem, with
    # their own per-token dark/light structure resolved against primitives.
    assert set(tree.interfaces) == {"demo"}
    demo = tree.interfaces["demo"]
    assert demo["badge.accent.dark"].value == "#2dd4bf"
    assert demo["badge.accent.light"].value == "#14b8a6"
    assert tree.interface_metadata["demo"] == {}


def test_load_token_tree_missing_primitives_raises(tmp_path: Path) -> None:
    (tmp_path / "themes").mkdir()

    with pytest.raises(TokenModelError, match="primitives file not found"):
        load_token_tree(tmp_path)


def test_load_token_tree_tolerates_missing_optional_dirs(tmp_path: Path) -> None:
    (tmp_path / "core.json").write_text(
        '{"color": {"a": {"$value": "#000", "$type": "color"}}}', encoding="utf-8"
    )

    tree = load_token_tree(tmp_path)

    assert tree.themes == {}
    assert tree.interfaces == {}
    assert tree.theme_metadata == {}
    assert tree.interface_metadata == {}
    assert tree.primitives["color.a"].value == "#000"


def test_load_token_tree_classifies_every_alias_status() -> None:
    tree = load_token_tree(ALIAS_CASES_DIR)

    accent = tree.themes["dark"]
    assert accent["accent.resolved"].alias_status == AliasStatus.RESOLVED
    assert accent["accent.resolved"].value == "#14b8a6"
    assert accent["accent.multihop"].alias_status == AliasStatus.MULTI_HOP
    assert accent["accent.dangling"].alias_status == AliasStatus.DANGLING
    assert accent["accent.not-primitive"].alias_status == AliasStatus.NOT_PRIMITIVE

    # Unresolved aliases keep their raw {alias.path} string for error
    # reporting rather than a silently wrong value.
    assert accent["accent.multihop"].value == "{chained.one}"
    assert accent["accent.dangling"].value == "{color.nonexistent}"
    assert accent["accent.not-primitive"].value == "{bg.primary}"
