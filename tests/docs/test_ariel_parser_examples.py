"""The ARIEL search guides' query, vocabulary and caption examples run through the code.

``docs/source/how-to/ariel/search-modes.rst`` shows keyword queries, glob and
``/…/`` pattern tokens with the names they match, a sample ``vocabulary.yml``
and the ``expanded_terms`` a query produces. Each example is read out of the
page and handed to the code that handles it: :func:`parse_keyword_query`, the
vocabulary loader and ``osprey ariel vocab-check``, and :func:`expand_query`.
A change on either side fails here.

Glob matches are checked by running the parser's PostgreSQL ARE in Python's
``re`` with the word anchors ``\\m``/``\\M`` translated; every other construct a
translated glob emits (``\\S``, escaped literals) means the same in both.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group
from osprey.services.ariel_search.config import VocabularyConfig
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search.keyword import MAX_QUERY_LENGTH, parse_keyword_query
from osprey.services.ariel_search.search.patterns import EMPTY_PATTERN_NOTICE
from osprey.services.ariel_search.vocabulary import load_vocabulary
from osprey.services.ariel_search.vocabulary.expand import expand_query
from osprey.services.ariel_search.vocabulary.model import Vocabulary

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARIEL_DOCS = _REPO_ROOT / "docs" / "source" / "how-to" / "ariel"
_SEARCH_MODES = _ARIEL_DOCS / "search-modes.rst"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """*text* with every whitespace run collapsed to one space."""
    return re.sub(r"\s+", " ", text)


def _code_blocks(text: str) -> list[str]:
    """The dedented bodies of every ``.. code-block::`` in *text*."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"^(\s*)\.\. code-block::", lines[i])
        if not match:
            i += 1
            continue
        indent = len(match.group(1))
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            i += 1
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


def _block(path: Path, marker: str) -> str:
    """The one code block of *path* containing *marker*."""
    found = [b for b in _code_blocks(_text(path)) if marker in b]
    assert len(found) == 1, f"{len(found)} code blocks in {path.name} contain {marker!r}"
    return found[0]


def _parse(query: str) -> tuple[str, dict[str, str], tuple[str, ...], tuple[str, ...]]:
    """``(search_text, field_filters, phrases, pattern sources)`` of *query*."""
    parsed = parse_keyword_query(query)
    sources = tuple(span.source for span in parsed.pattern_spans)
    return parsed.search_text, parsed.field_filters, parsed.phrases, sources


def _are_to_python(are: str) -> re.Pattern[str]:
    r"""Compile a translated glob's ARE for Python, as ``~*`` runs it.

    PostgreSQL's ``\m`` (word start) and ``\M`` (word end) have no Python
    spelling, so they become lookarounds; ``~*`` is case-insensitive.
    """
    python = are.replace(r"\m", r"\b(?=\w)").replace(r"\M", r"\b(?<=\w)")
    return re.compile(python, re.IGNORECASE)


def _page_vocabulary_path(tmp_path: Path) -> Path:
    path = tmp_path / "vocabulary.yml"
    path.write_text(_block(_SEARCH_MODES, "concepts:") + "\n", encoding="utf-8")
    return path


def _page_vocabulary(tmp_path: Path) -> Vocabulary:
    vocabulary, errors, warnings = load_vocabulary(_page_vocabulary_path(tmp_path))
    assert errors == []
    assert warnings == []
    assert vocabulary is not None
    return vocabulary


def _expand_with_defaults(text: str, vocabulary: Vocabulary) -> list[dict]:
    """*text*'s expansion groups under the configuration defaults."""
    defaults = VocabularyConfig()
    expansion = expand_query(
        text,
        vocabulary,
        canonical_to_acronym=defaults.canonical_to_acronym,
        canonical_to_shorthand=defaults.canonical_to_shorthand,
    )
    return [group.to_dict() for group in expansion.groups]


# ---------------------------------------------------------------------------
# Keyword query syntax
# ---------------------------------------------------------------------------

#: Every query line of the Keyword Search tab's syntax block, and what the
#: parser makes of it: (search_text, field_filters, phrases, pattern sources).
_SYNTAX_BLOCK_PARSES: dict[str, tuple[str, dict[str, str], tuple[str, ...], tuple[str, ...]]] = {
    "RF cavity fault": ("RF cavity fault", {}, (), ()),
    "RF AND cavity": ("RF AND cavity", {}, (), ()),
    "vacuum OR pressure": ("vacuum OR pressure", {}, (), ()),
    "beam NOT injection": ("beam NOT injection", {}, (), ()),
    '"RF cavity trip"': ("", {}, ("RF cavity trip",), ()),
    "author:smith": ("", {"author": "smith"}, (), ()),
    "date:2024-06": ("", {"date": "2024-06"}, (), ()),
    'author:jones "beam loss" date:2024-01': (
        "",
        {"author": "jones", "date": "2024-01"},
        ("beam loss",),
        (),
    ),
    "SR01C___BPM*": ("", {}, (), ("glob",)),
    r"/SR0[1-4]C___BPM\d+/": ("", {}, (), ("regex",)),
}


def _syntax_block_queries() -> list[str]:
    block = _block(_SEARCH_MODES, "# Simple terms")
    return [line.strip() for line in block.splitlines() if line.strip() and line[0] != "#"]


def test_syntax_block_lists_exactly_the_checked_queries() -> None:
    assert sorted(_syntax_block_queries()) == sorted(_SYNTAX_BLOCK_PARSES)


@pytest.mark.parametrize("query", sorted(_SYNTAX_BLOCK_PARSES))
def test_syntax_block_query_parses_as_documented(query: str) -> None:
    assert _parse(query) == _SYNTAX_BLOCK_PARSES[query]


def test_documented_truncation_length_is_the_parser_limit() -> None:
    """Both sentences that state the limit: the How-it-works step and the expansion note."""
    stated = re.findall(
        r"truncated at ([\d,]+) characters|longer than ([\d,]+) characters are truncated",
        _flat(_text(_SEARCH_MODES)),
    )
    assert len(stated) == 2
    assert {int((a or b).replace(",", "")) for a, b in stated} == {MAX_QUERY_LENGTH}


# ---------------------------------------------------------------------------
# Pattern tokens
# ---------------------------------------------------------------------------

_GLOB_LINE = re.compile(r"^(?P<glob>\S+)\s+matches (?P<hits>.+), not (?P<miss>\S+)$")


def _glob_lines() -> list[tuple[str, list[str], str]]:
    block = _block(_SEARCH_MODES, "SR01C___BPM*     matches")
    rows: list[tuple[str, list[str], str]] = []
    for line in block.splitlines():
        match = _GLOB_LINE.match(line.strip())
        assert match, f"glob line not in 'GLOB matches A and B, not C' form: {line!r}"
        hits = [hit.removeprefix("bare ") for hit in match["hits"].split(" and ")]
        rows.append((match["glob"], hits, match["miss"]))
    assert rows
    return rows


@pytest.mark.parametrize(("glob", "hits", "miss"), _glob_lines())
def test_glob_line_matches_and_misses_as_documented(glob: str, hits: list[str], miss: str) -> None:
    parsed = parse_keyword_query(glob)
    assert [span.source for span in parsed.pattern_spans] == ["glob"]
    pattern = _are_to_python(parsed.pattern_spans[0].body)
    for hit in hits:
        assert pattern.search(hit), (glob, hit)
        assert pattern.search(f"checked {hit.lower()} today"), (glob, hit)
    assert not pattern.search(miss), (glob, miss)


def test_glob_translation_negative_control() -> None:
    """The ARE translation rejects a name that differs by one character."""
    pattern = _are_to_python(parse_keyword_query("SR01C___BPM*").pattern_spans[0].body)
    assert not pattern.search("SR02C___BPM3")


def test_regex_block_lines_parse_as_documented(tmp_path: Path) -> None:
    vocabulary = _page_vocabulary(tmp_path)
    block = _block(_SEARCH_MODES, "one pattern, plus the t/s expansion")
    rows = [re.split(r"\s{2,}", line.strip()) for line in block.splitlines()]
    assert len(rows) == 2
    for query, claim in rows:
        parsed = parse_keyword_query(query)
        assert [span.source for span in parsed.pattern_spans] == ["regex"], query
        if claim == "one pattern":
            assert parsed.search_text == ""
        else:
            assert claim == "one pattern, plus the t/s expansion"
            assert parsed.search_text == "t/s"
            assert _expand_with_defaults(parsed.search_text, vocabulary) == [
                {"original": "t/s", "alternatives": ["troubleshoot"]}
            ]


#: Queries the prose names, and how many patterns each yields.
_PROSE_PATTERN_COUNTS = {
    "did the BPM trip?": 0,
    "SR01C___BPM?": 0,
    "t/s": 0,
    "i/o": 0,
    "/beam loss/": 1,
    r"/SR0[1-4]C___BPM\d+/?": 1,
    "/BPMs?/": 1,
    "test UNION SELECT * FROM users --": 0,
    "5 * 3": 0,
    "flow rate 5 / 6 / 7": 0,
    "//": 0,
}


@pytest.mark.parametrize(("query", "count"), _PROSE_PATTERN_COUNTS.items())
def test_prose_query_yields_the_documented_pattern_count(query: str, count: int) -> None:
    assert f"``{query}``" in _text(_SEARCH_MODES)
    assert len(parse_keyword_query(query).pattern_spans) == count


def test_question_mark_inside_a_pattern_body_is_kept() -> None:
    assert parse_keyword_query("/BPMs?/").pattern_spans[0].body == "BPMs?"
    assert parse_keyword_query(r"/SR0[1-4]C___BPM\d+/?").pattern_spans[0].body == (
        r"SR0[1-4]C___BPM\d+"
    )


@pytest.mark.parametrize(
    "query", ["test UNION SELECT * FROM users --", "5 * 3", "flow rate 5 / 6 / 7"]
)
def test_too_generic_pattern_is_searched_as_text_with_a_diagnostic(query: str) -> None:
    parsed = parse_keyword_query(query)
    assert parsed.pattern_spans == ()
    assert parsed.search_text
    assert [d.category for d in parsed.diagnostics] == ["pattern"]


def test_rejected_regex_contributes_its_body_words() -> None:
    words = parse_keyword_query("flow rate 5 / 6 / 7").search_text.split()
    assert sorted(words) == sorted(["flow", "rate", "5", "6", "7"])


def test_empty_pattern_is_ignored_with_a_notice() -> None:
    parsed = parse_keyword_query("//")
    assert parsed.search_text == ""
    assert [(d.level, d.message) for d in parsed.diagnostics] == [
        (DiagnosticLevel.INFO, EMPTY_PATTERN_NOTICE)
    ]


# ---------------------------------------------------------------------------
# Vocabulary file and expansion
# ---------------------------------------------------------------------------


def test_page_vocabulary_loads_without_errors_or_warnings(tmp_path: Path) -> None:
    vocabulary = _page_vocabulary(tmp_path)
    assert [concept.canonical for concept in vocabulary.concepts] == [
        "beam position monitor",
        "troubleshoot",
        "beam loss",
    ]


def _vocab_check(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, dict]:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(ariel_group, ["vocab-check", "--json", str(path)])
    return result.exit_code, json.loads(result.stdout)


def test_page_vocabulary_passes_vocab_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report = _vocab_check(_page_vocabulary_path(tmp_path), tmp_path, monkeypatch)
    assert code == 0
    assert report["status"] == "ok"
    assert report["concepts"] == 3
    assert report["errors"] == []
    assert report["warnings"] == []


def test_vocab_check_negative_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's own rule: an unknown key is refused, not ignored."""
    path = _page_vocabulary_path(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "    note: extra\n", encoding="utf-8")
    code, report = _vocab_check(path, tmp_path, monkeypatch)
    assert code == 1
    assert report["status"] == "invalid"
    assert report["errors"]


def test_expanded_terms_example_is_what_ts_expands_to(tmp_path: Path) -> None:
    shown = json.loads(_block(_SEARCH_MODES, '"original"'))
    assert _expand_with_defaults("ts", _page_vocabulary(tmp_path)) == shown


@pytest.mark.parametrize(
    ("text", "groups"),
    [
        # Form to canonical is always on.
        ("bpm", [{"original": "bpm", "alternatives": ["beam position monitor"]}]),
        # canonical_to_shorthand defaults off.
        ("troubleshoot", []),
        # A synonym reaches every other member.
        ("beam dump", [{"original": "beam dump", "alternatives": ["beam loss", "lost beam"]}]),
        # Edge punctuation is stripped; a hyphen is a space.
        ("(BPM),", [{"original": "bpm", "alternatives": ["beam position monitor"]}]),
        (
            "Beam-Position Monitor",
            [{"original": "beam position monitor", "alternatives": ["bpm", "bpms"]}],
        ),
        # A phrase matches only where the text breaks the same way.
        ("beam. position monitor", []),
    ],
)
def test_page_vocabulary_expands_as_the_prose_says(
    text: str, groups: list[dict], tmp_path: Path
) -> None:
    assert _expand_with_defaults(text, _page_vocabulary(tmp_path)) == groups
