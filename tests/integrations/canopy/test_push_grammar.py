"""The push module's own copy of CANOPY's push grammar, driven from the shared vectors.

``fixtures/grammar-vectors.json`` is authored in CANOPY and copied here byte for
byte. The push module runs on the host, where CANOPY is not installed, so it
keeps its own stdlib implementation of the grammar and of the forbidden-character
predicate; these cases are the ones CANOPY runs against its own. Nothing here
needs CANOPY installed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from osprey.integrations.canopy import push
from osprey.integrations.canopy.library import DEPLOYMENT_IGNORED_NAMES, IMAGE_SUFFIXES
from osprey.integrations.canopy.push import (
    ASSET_SEGMENT,
    ID_PATTERN,
    PUSH_IMAGE_SUFFIXES,
    RESERVED_ENDINGS,
    RESERVED_PREFIXES,
    RESERVED_SEGMENTS,
    casefold_collisions,
    first_forbidden_pointer,
    has_forbidden_char,
    is_forbidden_char,
    is_reserved_segment,
)

VECTORS_PATH = Path(__file__).parent / "fixtures" / "grammar-vectors.json"
VECTORS_BYTES = VECTORS_PATH.read_bytes()
VECTORS = json.loads(VECTORS_BYTES)
RESERVED = VECTORS["reserved"]
FORBIDDEN = VECTORS["characters"]["forbidden"]
FORBIDDEN_CHARS = [(name, char) for name, chars in FORBIDDEN.items() for char in chars]
POINTER_CLASSES = {case["class"] for case in VECTORS["pointers"]}


def _code(char: str) -> str:
    return f"U+{ord(char):04X}"


# --- the module -----------------------------------------------------------------------------


#: Every module of the push package.
PACKAGE_FILES = sorted(Path(push.__file__).parent.glob("*.py"))


def _imports(module_path: Path) -> set[str]:
    """The absolute imports of ``module_path``; a relative one may only name a sibling module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level == 1, ast.unparse(node)
                siblings = [node.module] if node.module else [a.name for a in node.names]
                assert all((module_path.parent / f"{name}.py").is_file() for name in siblings)
                continue
            imported.add(node.module or "")
    return imported


def test_the_module_imports_only_the_standard_library_and_osprey() -> None:
    """CANOPY runs only inside the container: the module never imports it, nor any third party.

    OSPREY's own code spans two top-level packages: ``osprey`` and the connector
    workspace package ``osprey_connectors``.
    """
    assert len(PACKAGE_FILES) > 1, PACKAGE_FILES
    imported = set().union(*(_imports(path) for path in PACKAGE_FILES))
    tops = {name.split(".")[0] for name in imported}
    own = {"osprey", "osprey_connectors"}
    assert tops <= sys.stdlib_module_names | own, sorted(tops - sys.stdlib_module_names - own)
    assert not any(name.startswith(("canopy", "canopy_mcp")) for name in imported)


def test_the_module_runs_no_git_command() -> None:
    """The push tool runs no git command; the admin commits. Nothing in the source spawns one."""
    source = "".join(path.read_text(encoding="utf-8") for path in PACKAGE_FILES)
    for node in (n for path in PACKAGE_FILES for n in ast.walk(ast.parse(path.read_text("utf-8")))):
        if isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if (
                isinstance(first, ast.List)
                and first.elts
                and isinstance(first.elts[0], ast.Constant)
            ):
                assert first.elts[0].value != "git", ast.unparse(node)
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                assert not first.value.startswith("git "), ast.unparse(node)
    assert "GitPython" not in source and "dulwich" not in source


# --- the vector file ---------------------------------------------------------------------------


def test_the_vector_file_is_ascii_only() -> None:
    assert VECTORS_BYTES.isascii()


def test_the_vector_file_has_no_trailing_whitespace() -> None:
    assert b"\r" not in VECTORS_BYTES and b"\t" not in VECTORS_BYTES
    assert [
        n for n, line in enumerate(VECTORS_BYTES.split(b"\n"), 1) if line != line.rstrip()
    ] == []


def test_the_vector_file_ends_in_exactly_one_newline() -> None:
    assert VECTORS_BYTES.endswith(b"\n") and not VECTORS_BYTES.endswith(b"\n\n")


# --- patterns and suffix sets --------------------------------------------------------------------


def test_the_patterns_are_the_ones_the_vectors_name() -> None:
    assert ID_PATTERN.pattern == "{0}(/{0})*".format(VECTORS["patterns"]["idSegment"])
    assert ASSET_SEGMENT.pattern == VECTORS["patterns"]["assetSegment"]


def test_the_suffix_sets() -> None:
    assert sorted(IMAGE_SUFFIXES) == VECTORS["suffixes"]["image"]
    assert sorted(PUSH_IMAGE_SUFFIXES) == VECTORS["suffixes"]["pushImage"]
    assert PUSH_IMAGE_SUFFIXES == IMAGE_SUFFIXES - {".svg"}


# --- reserved segments ---------------------------------------------------------------------------


def test_the_reserved_tables() -> None:
    assert sorted(RESERVED_SEGMENTS) == RESERVED["names"]
    assert sorted(DEPLOYMENT_IGNORED_NAMES) == RESERVED["configuredNames"]
    assert sorted(RESERVED_PREFIXES) == sorted(RESERVED["prefixes"])
    assert sorted(RESERVED_ENDINGS) == RESERVED["endings"]
    assert all(
        name == name.casefold()
        for name in (*RESERVED_SEGMENTS, *RESERVED_PREFIXES, *RESERVED_ENDINGS)
    )


def test_every_reserved_rule_has_a_case() -> None:
    cases = RESERVED["cases"]
    assert {c["segment"].casefold() for c in cases if c["rule"] == "name"} == set(RESERVED["names"])
    assert {c["segment"].casefold() for c in cases if c["rule"] == "configured"} == set(
        RESERVED["configuredNames"]
    )
    for prefix in RESERVED["prefixes"]:
        assert any(c["rule"] == "prefix" and c["segment"].startswith(prefix) for c in cases)
    for ending in RESERVED["endings"]:
        assert any(c["rule"] == "ending" and c["segment"].endswith(ending) for c in cases)


@pytest.mark.parametrize("case", RESERVED["cases"], ids=lambda c: f"{c['rule']}:{c['segment']}")
def test_a_reserved_segment_is_reserved_by_its_rule(case: dict) -> None:
    segment = case["segment"]
    folded = segment.casefold()
    assert is_reserved_segment(segment)
    if case["rule"] == "name":
        assert folded in RESERVED_SEGMENTS
    elif case["rule"] == "configured":
        assert folded in DEPLOYMENT_IGNORED_NAMES
    elif case["rule"] == "prefix":
        assert folded.startswith(RESERVED_PREFIXES) and folded not in RESERVED_SEGMENTS
    else:
        assert case["rule"] == "ending"
        assert folded.endswith(RESERVED_ENDINGS) and not folded.startswith(RESERVED_PREFIXES)


@pytest.mark.parametrize("segment", RESERVED["free"])
def test_a_near_miss_is_not_reserved(segment: str) -> None:
    assert not is_reserved_segment(segment)


# --- id and asset segments -------------------------------------------------------------------------


def _id_segment(segment: str) -> bool:
    return "/" not in segment and ID_PATTERN.fullmatch(segment) is not None


def _asset_segment(segment: str) -> bool:
    return ASSET_SEGMENT.fullmatch(segment) is not None


SEGMENT_KINDS = {"idSegments": _id_segment, "assetSegments": _asset_segment}


def _accepted() -> list:
    return [
        pytest.param(kind, s, id=f"{kind}:{s}")
        for kind in SEGMENT_KINDS
        for s in VECTORS[kind]["accepted"]
    ]


def _refused() -> list:
    return [
        pytest.param(kind, r["segment"], r["why"], id=f"{kind}:{r['why']}:{r['segment']!r}")
        for kind in SEGMENT_KINDS
        for r in VECTORS[kind]["refused"]
    ]


@pytest.mark.parametrize(("kind", "segment"), _accepted())
def test_a_segment_is_accepted(kind: str, segment: str) -> None:
    assert SEGMENT_KINDS[kind](segment)
    assert not is_reserved_segment(segment)


@pytest.mark.parametrize(("kind", "segment", "why"), _refused())
def test_a_segment_is_refused(kind: str, segment: str, why: str) -> None:
    assert why in ("pattern", "reserved")
    assert SEGMENT_KINDS[kind](segment) is (why == "reserved")
    if why == "reserved":
        assert is_reserved_segment(segment)


@pytest.mark.parametrize("kind", SEGMENT_KINDS)
def test_the_segment_patterns_are_the_stored_ones(kind: str) -> None:
    pattern = re.compile(VECTORS["patterns"][kind.removesuffix("s")])
    cases = [*VECTORS[kind]["accepted"], *(r["segment"] for r in VECTORS[kind]["refused"])]
    for segment in cases:
        assert (pattern.fullmatch(segment) is not None) is SEGMENT_KINDS[kind](segment)


# --- the forbidden-character predicate ---------------------------------------------------------------


def test_every_character_class_is_covered() -> None:
    assert set(FORBIDDEN) == {"c0", "del", "c1", "bidi", "format", "surrogate"}
    assert all(FORBIDDEN.values())
    assert set(FORBIDDEN) <= POINTER_CLASSES


@pytest.mark.parametrize(
    ("name", "char"), FORBIDDEN_CHARS, ids=lambda v: v if len(v) > 1 else _code(v)
)
def test_a_forbidden_character(name: str, char: str) -> None:  # noqa: ARG001 - the id
    assert is_forbidden_char(char)
    assert has_forbidden_char(f"ab{char}cd")


@pytest.mark.parametrize("char", VECTORS["characters"]["allowed"], ids=_code)
def test_an_allowed_character(char: str) -> None:
    assert not is_forbidden_char(char)
    assert not has_forbidden_char(f"ab{char}cd")


def test_the_c0_and_c1_blocks_in_full() -> None:
    allowed = {"\t", "\n", "\r"}
    for code in [*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)]:
        assert is_forbidden_char(chr(code)) is (chr(code) not in allowed), _code(chr(code))
    assert not any(is_forbidden_char(chr(code)) for code in range(0x20, 0x7F))


def test_every_surrogate_is_forbidden() -> None:
    assert all(is_forbidden_char(chr(code)) for code in range(0xD800, 0xE000))


def test_an_empty_string_holds_no_forbidden_character() -> None:
    assert not has_forbidden_char("")


# --- the document walk -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["pointers"], ids=lambda c: c["name"])
def test_the_first_offending_pointer(case: dict) -> None:
    assert first_forbidden_pointer(case["doc"]) == case["pointer"]


def test_the_walk_does_not_change_the_document() -> None:
    for case in VECTORS["pointers"]:
        before = json.dumps(case["doc"])
        first_forbidden_pointer(case["doc"])
        assert json.dumps(case["doc"]) == before


# --- casefold uniqueness -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["casefold"], ids=lambda c: c["name"])
def test_casefold_collisions(case: dict) -> None:
    assert casefold_collisions(case["paths"]) == case["collisions"]
    assert casefold_collisions(reversed(case["paths"])) == case["collisions"]


# --- the printer uses the same predicate --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "char"), FORBIDDEN_CHARS, ids=lambda v: v if len(v) > 1 else _code(v)
)
def test_safe_text_strips_every_forbidden_character(name: str, char: str) -> None:  # noqa: ARG001
    assert push.safe_text(f"ab{char}cd") == "abcd"


def test_safe_text_escapes_the_three_line_controls_and_the_backslash() -> None:
    assert push.safe_text("a\rb\nc\td\\e") == "a\\rb\\nc\\td\\\\e"


def test_safe_text_writes_non_ascii_as_u_escapes() -> None:
    assert push.safe_text("café \U0001f600") == "caf\\u00e9 \\ud83d\\ude00"
    assert push.safe_text({"b": 1, "a": "x"}) == '{"a": "x", "b": 1}'
