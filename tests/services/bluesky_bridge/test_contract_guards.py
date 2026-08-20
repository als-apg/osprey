"""Greppable acceptance criteria for the declared-contract work, as tests.

Two of the criteria are stated as greps over the bridge's source, because what
they assert is the *absence* of a way of writing code: SC-1 says the bridge no
longer guesses a channel's role or a channel's data column from its name, and
SC-12 says bluesky's wire words are not used as identifiers on authored
surfaces. Absence has no call site to unit-test, so it is pinned here by
reading the source tree the same way the criteria are written.

A grep test earns its keep only if it would actually fail, so every scan below
asserts it read a non-empty set of files first: a scan that silently matched
nothing -- a moved package, a typo in the glob -- would otherwise pass forever
while the rule it guards rots.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel

import osprey.services.bluesky_bridge as bluesky_bridge
from osprey.services.bluesky_bridge.plans_core import grid_scan, orm

BRIDGE_ROOT = Path(bluesky_bridge.__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = (
    Path(bluesky_bridge.__file__).parents[2] / "templates" / "claude_code" / "claude" / "skills"
)

# The name-guessing rules the declared contract replaces, spelled exactly as
# SC-1 greps for them. Each one inferred something the plan author is now
# required to declare: a plural-to-singular device-name guess, a role guessed
# from the substring "detect", and two hard-coded key lists that guessed which
# params carried read-only devices and which carried the point count.
SC1_GUESSING_RULES = (
    'removesuffix("s")',
    '"detect" in',
    "_READ_ONLY_DEVICE_KEYS",
    "_POINT_COUNT_KEYS",
)

# The per-module column-matching helpers SC-1 replaces with the one shared
# `plan_fields.resolve_column`. Two lived in `orm_analysis`, one in
# `plans_core/grid_scan.py`, and all three carried their own copy of the
# exact-then-prefix rule.
SC1_DELETED_HELPERS = (
    "_match_value",
    "_resolve_key",
    "_resolve_column",
)


def _bridge_sources() -> list[Path]:
    """Every Python file under the bridge package, tests excluded by location."""
    return sorted(path for path in BRIDGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _files_containing(needle: str) -> list[str]:
    """Bridge-relative paths of every source file whose text contains *needle*."""
    return [
        str(path.relative_to(BRIDGE_ROOT))
        for path in _bridge_sources()
        if needle in path.read_text(encoding="utf-8")
    ]


def test_the_bridge_source_scan_reads_a_real_file_set() -> None:
    """The scans below are only meaningful if they read the bridge's sources.

    Non-vacuity control for every grep in this module: it pins that the tree
    being scanned exists, is substantial, and contains the modules the criteria
    name. Without it a wrong `BRIDGE_ROOT` would turn every assertion in this
    file into a tautology.
    """
    sources = _bridge_sources()
    assert len(sources) > 10, f"only {len(sources)} source files under {BRIDGE_ROOT}"

    names = {str(path.relative_to(BRIDGE_ROOT)) for path in sources}
    assert "plan_fields.py" in names
    assert "orm_analysis.py" in names
    assert "plans_core/grid_scan.py" in names


def test_the_scan_would_notice_a_planted_token() -> None:
    """Negative control: the scanner really does find a token when one is there.

    `plan_fields.resolve_column` exists in the tree, so searching for it must
    return the file that defines it. If this fails, the two absence assertions
    below are passing for the wrong reason.
    """
    assert "plan_fields.py" in _files_containing("def resolve_column")


@pytest.mark.parametrize("rule", SC1_GUESSING_RULES)
def test_no_name_guessing_rule_survives_in_the_bridge(rule: str) -> None:
    """SC-1, first clause: the bridge infers no role or count from a name.

    Every one of these guessed something a plan now declares outright through
    the role-typed fields in `plan_fields`. A hit here means a consumer went
    back to reading intent out of a string.
    """
    hits = _files_containing(rule)
    assert hits == [], f"name-guessing rule {rule!r} is back in: {hits}"


@pytest.mark.parametrize("helper", SC1_DELETED_HELPERS)
def test_column_resolution_has_exactly_one_implementation(helper: str) -> None:
    """SC-1, second clause: one shared column resolver, no private copies.

    `plan_fields.resolve_column` is the single place the exact-then-prefix
    column rule is written down. These three names were its per-module
    duplicates; a hit means a fourth copy of the rule is drifting into being.
    """
    hits = _files_containing(helper)
    assert hits == [], f"deleted column-matching helper {helper!r} is back in: {hits}"


def test_the_shared_column_resolver_is_the_one_consumers_import() -> None:
    """The positive half of SC-1's second clause.

    Deleting the copies is only half the criterion -- the modules that used to
    carry them must now reach for the shared helper instead, or they resolve
    nothing at all.
    """
    for module in ("orm_analysis.py", "plans_core/grid_scan.py"):
        source = (BRIDGE_ROOT / module).read_text(encoding="utf-8")
        assert "resolve_column" in source, f"{module} no longer resolves channels to columns"
        assert "from osprey.services.bluesky_bridge.plan_fields import" in source, (
            f"{module} resolves columns without importing the shared helper"
        )


# ---------------------------------------------------------------------------
# SC-12: bluesky's wire words are not identifiers on any authored surface
# ---------------------------------------------------------------------------

# The two start-document keys bluesky itself uses for the channels a run moved
# and read. They are the WIRE's vocabulary, not the author's: a plan declares
# `movable` / `readable` channels, and `plan_fields.scan_metadata()` is the one
# place that translates the declaration into these keys.
WIRE_WORDS: Final = ("motors", "detectors")

# What "used as an identifier" means, spelled out, because the rule has to be
# decidable by a machine and defensible to the next person editing a skill:
#
#   * a wire word inside a code span or code block is ALWAYS a violation --
#     that is a field name, a kwarg, or a dict key an author would copy;
#   * a wire word in prose is a violation only when it TOUCHES code
#     punctuation (`detectors=`, `"detectors"`, `.detectors`, `_detectors`),
#     which is how an identifier reads when it is written mid-sentence;
#   * a wire word as an ordinary English plural in a sentence -- prose ABOUT
#     the wire, or about a room full of detectors -- is NOT a violation. SC-12
#     governs the vocabulary an author is taught to write, not the words the
#     documentation is allowed to use to explain it.
#
# `plan_fields.py` is exempt by construction: it is the translator, and its own
# test (`test_run_metadata_keys_are_spelled_only_inside_scan_metadata`) already
# pins that the keys appear nowhere in it but inside `scan_metadata`.
_CODE_BEFORE: Final = frozenset("=\"'[(_`")
_CODE_AFTER: Final = frozenset("=:\"'[()_`")

_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_RST_CODE_BLOCK = re.compile(
    r"^[ \t]*\.\. (?:code-block|literalinclude)::.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE
)
_INLINE_LITERAL = re.compile(r"``[^`]+``|`[^`\n]+`")


def _code_regions(text: str) -> list[str]:
    """Every span of *text* that is code rather than prose.

    Markdown fences, reStructuredText code directives, and inline literal
    spans in either syntax -- the three places an authored surface shows an
    author something to type.
    """
    regions = _FENCED_BLOCK.findall(text)
    regions += _RST_CODE_BLOCK.findall(text)
    regions += _INLINE_LITERAL.findall(text)
    return regions


def _reads_as_identifier(text: str, start: int, end: int) -> bool:
    """Whether the match at [*start*, *end*) is punctuated like code.

    A dot counts only when it joins the word to another name (`params.detectors`,
    `detectors.tolist()`); a sentence-final period does not, or every honest
    sentence ending in the word would be a violation.
    """
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if before in _CODE_BEFORE and before:
        return True
    if after in _CODE_AFTER and after:
        return True
    if before == "." and start >= 2 and (text[start - 2].isalnum() or text[start - 2] == "_"):
        return True
    return (
        after == "." and end + 1 < len(text) and (text[end + 1].isalnum() or text[end + 1] == "_")
    )


def _identifier_hits(text: str, word: str) -> list[str]:
    """Lines of *text* where *word* reads as an identifier (see the rule above).

    Returns the offending lines, trimmed, so a failure names what to fix rather
    than only that something is wrong.
    """
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])")
    lines = text.splitlines()

    def _line_at(offset: int, haystack: str) -> str:
        start = haystack.rfind("\n", 0, offset) + 1
        end = haystack.find("\n", offset)
        return haystack[start : end if end != -1 else len(haystack)].strip()

    hits: list[str] = []
    for region in _code_regions(text):
        for match in pattern.finditer(region):
            hits.append(f"in code: {_line_at(match.start(), region)}")

    for match in pattern.finditer(text):
        if _reads_as_identifier(text, match.start(), match.end()):
            line_no = text[: match.start()].count("\n")
            hits.append(f"line {line_no + 1}: {lines[line_no].strip()}")

    return sorted(set(hits))


def _docstring_of(path: Path, function_name: str) -> str:
    """The docstring of *function_name* in *path*, read without importing it.

    `ast` rather than an import: these are MCP tool functions behind a decorator
    that builds a server object, and a vocabulary grep has no business starting
    one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == function_name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{function_name} is not defined in {path}")


# `bluesky-plans` is the first authored surface that is a Jinja template rather
# than a static file, so it needs a decision the other five did not: scan the
# raw `.j2`, or scan a rendered form? This scans the RAW file. Three reasons,
# each checked against the template rather than assumed:
#
#   1. A render is only ever ONE branch. The template opens with
#      `{% if "bluesky" in enabled_servers %}` and its plan table has three
#      arms (`index is none` / no rows / rows). Rendering without "bluesky" in
#      `enabled_servers` returns the empty string, which would make every
#      assertion below vacuous; rendering WITH it still discards the two arms
#      the fabricated context did not take. The raw file is 5944 characters of
#      prose once Jinja markup is stripped, against 5217-5418 for a render --
#      the raw scan is the only one that reads every branch an operator could
#      be shown.
#   2. A render substitutes DATA, not authored prose. `{{ row.name }}` and
#      `{{ row.description }}` come from the plan files a deployment happens to
#      ship, so a render would police build output that varies per site. SC-12
#      governs the vocabulary an author is taught to write. Plan-side spellings
#      are already pinned by
#      `test_no_shipped_plan_names_a_params_field_after_a_wire_word`.
#   3. Jinja markup does not fool the scanner. `{`, `}` and `%` are in neither
#      `_CODE_BEFORE` nor `_CODE_AFTER`, so a delimiter cannot manufacture an
#      identifier hit; and a wire word written as a template variable
#      (`{{ detectors }}`) would still be caught, correctly. The markup adds
#      text, it never removes any, so it cannot mask a violation either.
#
# The cost of scanning raw is that Jinja control flow inflates the character
# count, so the non-vacuity floor could in principle be cleared on markup
# alone. `_TEMPLATED_SURFACES` closes that: templated surfaces must clear the
# floor on prose left after the markup is stripped.
_JINJA_MARKUP = re.compile(r"{%-?.*?-?%}|{{.*?}}", re.DOTALL)

# Authored surfaces whose text is a Jinja template. Scanned raw (see above),
# and held to the non-vacuity floor on their prose rather than their markup.
_TEMPLATED_SURFACES: Final = frozenset({"bluesky-plans/SKILL.md.j2"})


def _authored_surfaces() -> dict[str, str]:
    """Label -> text for every surface SC-12 governs."""
    tools = Path(bluesky_bridge.__file__).parents[2] / "mcp_server" / "bluesky" / "tools"
    return {
        "bluesky-plans/SKILL.md.j2": (SKILLS_ROOT / "bluesky-plans" / "SKILL.md.j2").read_text(
            encoding="utf-8"
        ),
        "writing-bluesky-plans/SKILL.md": (
            SKILLS_ROOT / "writing-bluesky-plans" / "SKILL.md"
        ).read_text(encoding="utf-8"),
        "operating-bluesky-plans/SKILL.md": (
            SKILLS_ROOT / "operating-bluesky-plans" / "SKILL.md"
        ).read_text(encoding="utf-8"),
        "docs/source/how-to/bluesky/write-plans.rst": (
            REPO_ROOT / "docs" / "source" / "how-to" / "bluesky" / "write-plans.rst"
        ).read_text(encoding="utf-8"),
        "list_plans docstring": _docstring_of(tools / "read_tools.py", "list_plans"),
        "write_plan docstring": _docstring_of(tools / "authoring.py", "write_plan"),
    }


def _params_field_names(model_cls: type[BaseModel]) -> set[str]:
    """Every field name a plan's `PARAMS` exposes, nested models included.

    Read off the JSON schema rather than `model_fields` alone because that is
    what a nested model contributes to: `grid_scan` carries its movable channel
    as `axes[].setpoint`, under `$defs`, and an author copies the schema's
    spelling.
    """
    schema = model_cls.model_json_schema()
    names = set(schema.get("properties", {}))
    for definition in schema.get("$defs", {}).values():
        names |= set(definition.get("properties", {}))
    return names


def test_the_authored_surface_scan_reads_every_surface() -> None:
    """Non-vacuity control for the SC-12 scans: six real, substantial surfaces.

    A renamed skill directory or a moved doc page would otherwise turn the
    absence assertions below into tautologies over an empty string. Templated
    surfaces are measured on their prose, not their Jinja, so a template that
    lost its body could not clear the floor on control flow alone.
    """
    surfaces = _authored_surfaces()
    assert set(surfaces) == {
        "bluesky-plans/SKILL.md.j2",
        "writing-bluesky-plans/SKILL.md",
        "operating-bluesky-plans/SKILL.md",
        "docs/source/how-to/bluesky/write-plans.rst",
        "list_plans docstring",
        "write_plan docstring",
    }
    assert _TEMPLATED_SURFACES <= set(surfaces), (
        "a templated surface is labelled here but not scanned -- its prose floor is off"
    )
    for label, text in surfaces.items():
        body = _JINJA_MARKUP.sub("", text) if label in _TEMPLATED_SURFACES else text
        assert len(body) > 400, f"{label} is only {len(body)} characters of prose -- did it move?"


def test_the_identifier_scan_separates_code_from_prose() -> None:
    """Negative control on the scanner itself, both directions.

    Planted identifiers must be found in each of the three code shapes and in
    punctuation-adjacent prose; a plain English sentence using the same word
    must not be. Without this, a scanner that found nothing at all would let
    every SC-12 assertion pass forever.
    """
    found = '```\nmd = {"detectors": names}\n```'
    assert _identifier_hits(found, "detectors")
    assert _identifier_hits("stamp it with ``detectors``", "detectors")
    assert _identifier_hits(".. code-block:: python\n\n   detectors = []\n", "detectors")
    assert _identifier_hits("pass detectors= to the helper", "detectors")
    assert _identifier_hits("the params.detectors field", "detectors")

    assert _identifier_hits("the run records what its detectors saw", "detectors") == []
    assert _identifier_hits("a hall full of detectors, and motors too", "motors") == []
    assert _identifier_hits("the start document names its detectors.", "detectors") == []


@pytest.mark.parametrize("word", WIRE_WORDS)
@pytest.mark.parametrize("label", sorted(_authored_surfaces()))
def test_no_wire_word_is_an_identifier_on_an_authored_surface(label: str, word: str) -> None:
    """SC-12: authored surfaces teach capability vocabulary, not wire keys.

    An author copies what these surfaces show. A wire word shown as a field
    name, a kwarg or a dict key teaches the one spelling the declared contract
    removed -- and the plan built from it would declare nothing at all.
    """
    hits = _identifier_hits(_authored_surfaces()[label], word)
    assert hits == [], f"{label} uses {word!r} as an identifier:\n" + "\n".join(hits)


@pytest.mark.parametrize("plan_module", [orm, grid_scan], ids=["orm", "grid_scan"])
def test_no_shipped_plan_names_a_params_field_after_a_wire_word(plan_module: object) -> None:
    """SC-12, the half that reaches the wire: `PARAMS` field names.

    A field name is the operator-facing spelling -- it is the queue item's
    kwarg, the draft's `plan_args` key and the form label -- so a shipped plan
    naming one after a start-document key would put the wire's vocabulary in
    front of every operator.
    """
    names = _params_field_names(plan_module.PARAMS)  # type: ignore[attr-defined]
    assert names, "PARAMS exposes no fields at all"
    leaked = sorted(names & set(WIRE_WORDS))
    assert leaked == [], f"{plan_module.__name__} names PARAMS fields after wire keys: {leaked}"


def test_the_shipped_plans_declare_the_fields_this_scan_expects() -> None:
    """Non-vacuity control for the field-name scan.

    Pins that both shipped models really were introspected -- an empty or
    unrelated schema would make the assertion above vacuous -- and that the
    role-carrying fields are spelled in capability terms.
    """
    assert {"correctors", "readbacks"} <= _params_field_names(orm.PARAMS)
    assert {"readbacks", "axes", "setpoint"} <= _params_field_names(grid_scan.PARAMS)
