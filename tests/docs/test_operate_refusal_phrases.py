"""The refusal phrases ``operate.rst`` quotes are the ones a refused write speaks.

The "What the chip cannot lift" table on the operate page lists, in italics, the
phrase the chip shows and the agent is refused with for each lock. An operator
searches a log or a transcript for those words, so each one must be a literal
part of a phrase in ``osprey_connectors.refusals`` (the contract) and in the
stdlib-only hook twin ``osprey_target_state.py`` (which restates it).

The page also quotes whole refusals with the machine suffix that
``refusal_phrase`` appends, ``(Concerns <kind words>.)``. The suffix is
templated, so it is matched by a regex whose words must be one of
``KIND_WORDS``; the rest of the quote must equal a phrase exactly.

Both directions: every lock reason the contract defines appears in the table,
except the ones :data:`NOT_IN_TABLE` names with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest

from osprey_connectors import refusals
from tests._by_path import load_module

_REPO = Path(__file__).resolve().parents[2]
_PAGE = _REPO / "docs" / "source" / "how-to" / "web-terminal" / "operate.rst"
_HOOK = _REPO / "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py"

#: Reasons the lock table leaves out on purpose, each with why.
NOT_IN_TABLE = {
    "session_held": "a hold the operator answers in the approval card, not a lock",
    "auth_unreadable": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_holder": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_channel": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_range": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_exhausted": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_lease_lapsed": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_expired": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_revoked": "a program write authorization refusal; see 'Approve a program's writes'",
    "auth_overlap": "refuses an approval, not a write; see 'Approve a program's writes'",
}

_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])")
_QUOTED_REFUSAL = re.compile(r'\*"([^"]+)"\*')
_CONCERNS = re.compile(r"\s*\(Concerns (?P<words>[^)]+)\.\)$")


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    return load_module("_operate_hook_twin", _HOOK)


def lock_table_cells(text: str) -> list[str]:
    """The first-column cells of the list-table whose header is ``Phrase``, whitespace joined."""
    lines = text.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == ".. list-table::"
        and any(lines[j].strip() == "* - Phrase" for j in range(i, min(i + 6, len(lines))))
    )
    cells: list[str] = []
    current: list[str] | None = None
    column = 0
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped.startswith("* - "):
            column = 0
            current = [stripped[4:]]
            cells.append("")
        elif stripped.startswith("- "):
            column += 1
            current = None
        elif current is not None and column == 0 and stripped:
            current.append(stripped)
        if current is not None and column == 0:
            cells[-1] = " ".join(current)
    return cells[1:]  # drop the header row


def phrases_in(cell: str) -> list[str]:
    return _ITALIC.findall(cell)


def reasons_for(fragment: str, table: dict[str, str]) -> set[str]:
    """Every reason whose phrase contains *fragment* word for word."""
    return {reason for reason, phrase in table.items() if fragment in phrase}


def exact_reason(quote: str, table: dict[str, str], kind_words: dict[str, str]) -> str | None:
    """The reason whose phrase *quote* is, once a ``(Concerns …)`` suffix of a known kind is cut."""
    match = _CONCERNS.search(quote)
    if match is not None:
        if match["words"] not in kind_words.values():
            return None
        quote = quote[: match.start()]
    return next((reason for reason, phrase in table.items() if phrase == quote), None)


def _table_phrases() -> list[str]:
    cells = lock_table_cells(_PAGE.read_text(encoding="utf-8"))
    found = [phrase for cell in cells for phrase in phrases_in(cell)]
    assert len(found) >= 11, f"lock table parsed to too few phrases: {found}"
    return found


@pytest.mark.parametrize("phrase", _table_phrases())
def test_each_table_phrase_is_part_of_a_contract_phrase(phrase: str, hook: ModuleType) -> None:
    assert reasons_for(phrase, refusals._PHRASES), f"not in refusals._PHRASES: {phrase!r}"
    assert reasons_for(phrase, hook._PHRASES), f"not in the hook twin's _PHRASES: {phrase!r}"


def test_every_lock_reason_is_in_the_table() -> None:
    covered = set().union(*(reasons_for(p, refusals._PHRASES) for p in _table_phrases()))
    missing = set(refusals.REASONS) - covered - set(NOT_IN_TABLE)
    assert not missing, f"lock reasons the operate table does not list: {sorted(missing)}"


def test_exclusions_are_live() -> None:
    stale = set(NOT_IN_TABLE) - set(refusals.REASONS)
    assert not stale, f"NOT_IN_TABLE names reasons the contract no longer has: {sorted(stale)}"
    listed = set().union(*(reasons_for(p, refusals._PHRASES) for p in _table_phrases()))
    both = listed & set(NOT_IN_TABLE)
    assert not both, f"NOT_IN_TABLE names reasons the table does list: {sorted(both)}"


def test_quoted_refusals_are_whole_phrases_with_a_known_machine() -> None:
    text = " ".join(_PAGE.read_text(encoding="utf-8").split())
    quotes = _QUOTED_REFUSAL.findall(text)
    assert quotes, "the page no longer quotes a whole refusal"
    for quote in quotes:
        assert exact_reason(quote, refusals._PHRASES, refusals.KIND_WORDS), (
            f"quoted refusal is not a contract phrase with a known machine suffix: {quote!r}"
        )


def test_suffix_examples_name_a_known_machine() -> None:
    text = " ".join(_PAGE.read_text(encoding="utf-8").split())
    suffixes = re.findall(r"\(Concerns ([^)]+)\.\)", text)
    assert suffixes
    for words in suffixes:
        assert words in refusals.KIND_WORDS.values(), f"unknown machine words: {words!r}"


def test_negative_controls() -> None:
    phrases = refusals._PHRASES
    kinds = refusals.KIND_WORDS
    # A reworded table phrase matches no reason.
    assert not reasons_for("the deployment is running read only", phrases)
    # A whole refusal with an unknown machine, or a changed word, is not a phrase.
    assert exact_reason(phrases["switch_off"] + " (Concerns the live machine.)", phrases, kinds)
    assert not exact_reason(phrases["switch_off"] + " (Concerns the real machine.)", phrases, kinds)
    assert not exact_reason(phrases["switch_off"].replace("chip", "badge"), phrases, kinds)
    # The table parser reads first-column cells only, joined across lines.
    sample = "\n".join(
        [
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Phrase",
            "     - Who lifts it",
            "   * - *one*, *two",
            "       three*",
            "     - *not a phrase*",
        ]
    )
    cells = lock_table_cells(sample)
    assert [p for c in cells for p in phrases_in(c)] == ["one", "two three"]
