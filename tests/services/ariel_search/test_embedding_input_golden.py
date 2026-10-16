"""Golden and property tests for the expanded embedding input.

``build_embedding_input`` with an entry expander decides which bytes of a logbook entry
reach the embedding model. Those bytes are part of every stored vector's build key, so the
golden cases below pin them exactly, and the property cases check the fitting rules on
generated entries:

* the input never exceeds the byte room;
* an entry that does not fit keeps at least three quarters of the room of its own text;
* the cut sits on a whitespace word end unless no word end lies in the window above the
  three-quarter floor, in which case it is the plain byte cut;
* the output is ``raw[:cut]`` plus the last scan's insertions in text order, the trailing
  ones that do not fit dropped whole;
* the insertions of a cut entry take at most a quarter of the room;
* one more word would overflow, or none remains;
* the cut is never shorter than a naive extend-while-it-fits walk;
* the fitter scans the entry at most four times.

The vocabulary is the unit-test vocabulary of the ``unit_vocabulary_file`` fixture.
"""

from __future__ import annotations

import random
import re

import pytest

from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    RESERVED_TOKENS,
    build_embedding_input,
    fit_to_input_limit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SpyExpander:
    """Wraps an expander and records the ``limit`` of every ``expand`` call."""

    def __init__(self, inner):
        self.inner = inner
        self.limits = []

    def expand(self, text, *, limit=None, tokens=None):
        self.limits.append(limit)
        return self.inner.expand(text, limit=limit, tokens=tokens)


@pytest.fixture
def expander(unit_vocabulary_file):
    """The unit vocabulary's entry expander, wrapped in a scan-counting spy."""
    from osprey.services.ariel_search.vocabulary import EntryExpander
    from osprey.services.ariel_search.vocabulary.loader import load_vocabulary

    loaded, errors, _warnings = load_vocabulary(
        unit_vocabulary_file, canonical_to_acronym=True, canonical_to_shorthand=False
    )
    assert errors == []
    return _SpyExpander(
        EntryExpander(loaded, canonical_to_acronym=True, canonical_to_shorthand=False)
    )


def _tokens_for(room):
    """``max_input_tokens`` whose byte room for an entry without picture text is ``room``."""
    return room + RESERVED_TOKENS


def _nbytes(text):
    return len(text.encode("utf-8"))


def _byte_prefix(text, budget):
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def _word_ends(raw):
    return [m.end() for m in re.finditer(r"\S+", raw)]


def _ins_bytes(insertion):
    return _nbytes(" " + insertion.text)


def _expanded_bytes(raw, cut, insertions):
    return _nbytes(raw[:cut]) + sum(_ins_bytes(i) for i in insertions if i.anchor <= cut)


def _render_greedy(raw, cut, insertions, room):
    """``raw[:cut]`` with insertions kept in text order until the first that overflows."""
    used = _nbytes(raw[:cut])
    out, pos = [], 0
    for ins in insertions:
        if ins.anchor > cut or used + _ins_bytes(ins) > room:
            break
        used += _ins_bytes(ins)
        out.append(raw[pos : ins.anchor] + " " + ins.text)
        pos = ins.anchor
    out.append(raw[pos:cut])
    return "".join(out)


def _naive_cut(raw, inner, room):
    """Extend word by word while the prefix plus its own scan's insertions fits the room."""
    best = 0
    for end in _word_ends(raw):
        scan = inner.expand(raw, limit=end)
        if _expanded_bytes(raw, end, scan.insertions) > room:
            break
        best = end
    return best


def _check_invariants(raw, att, max_tokens, spy):
    """Build the input for ``raw`` and assert every FR3 fitting invariant; return it."""
    inner = spy.inner
    spy.limits.clear()
    built = build_embedding_input(raw, att, max_tokens, expander=spy)
    budget = max_tokens - RESERVED_TOKENS

    # Picture text keeps its quarter and its separator; the entry gets the room left.
    if att and att.strip():
        att_part = _byte_prefix(att, budget // 4)
        room = budget - _nbytes(att_part) - 1
        assert built.text.endswith("\n" + att_part)
        entry = built.text[: len(built.text) - len(att_part) - 1]
    else:
        room = budget
        entry = built.text
    assert _nbytes(built.text) <= budget

    cut = built.kept_chars
    raw_room = _byte_prefix(raw, room)
    least = room * 3 // 4
    kept_bytes = _nbytes(raw[:cut])
    whole = cut == len(raw)

    # Output bytes never exceed the room.
    assert _nbytes(entry) <= room

    # The kept prefix is the whole entry or at least min(3/4 room, raw bytes).
    assert whole or kept_bytes >= min(least, _nbytes(raw))
    assert cut <= len(raw_room) or whole

    # Word-aligned unless the byte cut was taken: either the room scan had nothing to
    # insert (today's bytes), or no word end lies between the floor and the room.
    room_scan = inner.expand(raw, limit=len(raw_room))
    window = [e for e in _word_ends(raw) if e <= len(raw_room) and _nbytes(raw[:e]) >= least]
    word_aligned = whole or cut in _word_ends(raw)
    if not word_aligned:
        assert cut == len(raw_room)
        if room_scan.insertions:
            assert window == []  # the no-boundary fallback fired
    if not room_scan.insertions:
        assert entry == raw_room

    # Output == raw[:cut] + the last scan's insertions, trailing ones dropped whole.
    last = inner.expand(raw, limit=spy.limits[-1])
    assert entry == _render_greedy(raw, cut, last.insertions, room)
    # Every raw character of the kept prefix survives in order.
    assert _strip_insertions(entry, raw, cut, last.insertions) == raw[:cut]

    # A cut entry's insertions take at most a quarter of the room.
    inserted = _nbytes(entry) - kept_bytes
    if not whole:
        assert inserted <= room - least

    # One more word overflows, or none remains (byte cuts: one more character overflows).
    if not whole:
        if word_aligned:
            later = [e for e in _word_ends(raw) if e > cut]
            if later:
                nxt = later[0]
                walk_measure = _expanded_bytes(raw, nxt, room_scan.insertions)
                own_measure = _expanded_bytes(raw, nxt, inner.expand(raw, limit=nxt).insertions)
                assert nxt > len(raw_room) or walk_measure > room or own_measure > room
        else:
            assert _nbytes(raw[: cut + 1]) > room

    # The naive extend-while-fits walk is a lower bound.
    assert cut >= _naive_cut(raw, inner, room)

    # At most four scans: one bounded by the room, then at most three confirming re-scans.
    assert 1 <= len(spy.limits) <= 4
    return built


def _strip_insertions(entry, raw, cut, insertions):
    """Remove the rendered insertions from ``entry``, recovering ``raw[:cut]``."""
    out, pos, epos = [], 0, 0
    for ins in insertions:
        if ins.anchor > cut:
            break
        seg = raw[pos : ins.anchor]
        if entry[epos : epos + len(seg) + 1 + len(ins.text)] != seg + " " + ins.text:
            break
        out.append(seg)
        epos += len(seg) + 1 + len(ins.text)
        pos = ins.anchor
    out.append(entry[epos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Golden cases
# ---------------------------------------------------------------------------

#: Long entries and the exact embedding input each builds at ``max_input_tokens``.
_GOLDEN = (
    (
        # Cut at the floor: three quarters of the room is the entry's own text, and only
        # the first insertion fits what is left; the rest are dropped whole.
        "rf-bpm-mixed",
        "RF trip at 03:12, BPM 3 in sector 7 reads noisy; quad QF4 reset by the operator. "
        "Vertical orbit drifted by 40 um after the RF cavity came back. Checked BPMs in "
        "sectors 6 and 8, no change; ts continues with the i/o crate tomorrow morning.",
        128,
        "RF radio frequency trip at 03:12, BPM 3 in sector 7 reads noisy; quad QF4 reset by "
        "the operator. Vertical orbit",
    ),
    (
        # " beam position monitor" overflows after " radio frequency": it and every later
        # insertion are dropped, never a later one that would still fit.
        "dense-acronyms",
        "RF BPM quad " * 30,
        96,
        "RF radio frequency BPM quad RF BPM quad RF BPM quad RF BPM quad RF BPM quad RF BPM",
    ),
    (
        "repeated-entry",
        "BPM 12 lost signal after the RF interlock; quad power supply tripped twice. " * 4,
        160,
        "BPM beam position monitor 12 lost signal after the RF interlock; quad power supply "
        "tripped twice. BPM 12 lost signal after the RF interlock;",
    ),
)


class TestGolden:
    """Exact embedding inputs for fixed long entries.

    These bytes are what the embedding model reads, so they are part of every stored
    vector's build key. Updating an expected value here changes what the expander embeds
    and must be paired with an ``EXPANSION_RULE`` bump in
    ``osprey.services.ariel_search.vocabulary.entry``; otherwise vectors built before and
    after the change carry the same key but differ.
    """

    @pytest.mark.parametrize(("name", "raw", "max_tokens", "expected"), _GOLDEN)
    def test_golden_embedding_input(self, expander, name, raw, max_tokens, expected):
        built = build_embedding_input(raw, None, max_tokens, expander=expander)
        assert built.text == expected, name
        _check_invariants(raw, None, max_tokens, expander)

    def test_golden_picture_text_suffix(self, expander):
        raw = "RF trip at BPM 3, quad reset and the vertical orbit drifted " * 6
        built = build_embedding_input(raw, "scope trace: RF phase", 128, expander=expander)
        assert built.text == (
            "RF radio frequency trip at BPM 3, quad reset and the vertical orbit drifted RF "
            "trip at BPM\nscope trace: RF phase"
        )
        _check_invariants(raw, "scope trace: RF phase", 128, expander)


# ---------------------------------------------------------------------------
# Named edge cases
# ---------------------------------------------------------------------------


class TestNamedCases:
    """The cases FR3 names, each checked against every invariant plus its own point."""

    def test_cjk_entry_counts_bytes_not_characters(self, expander):
        # Each CJK character is three UTF-8 bytes: the 110-byte room holds 32 code points.
        raw = "RF " + "加速器のビーム位置 " * 20 + "BPM"
        room = 110
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert built.text == "RF radio frequency " + " ".join(["加速器のビーム位置"] * 3)
        assert built.kept_chars == 32

    def test_space_free_entry_takes_the_byte_cut(self, expander):
        raw = "RF-" + "x" * 300
        room = 64
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert built.text == fit_to_input_limit(raw, _tokens_for(room))
        assert built.kept_chars == room

    def test_torn_RFQ_is_not_matched_as_RF(self, expander):
        raw = "alpha beta gamma RFQ conditioning resumed"
        room = len("alpha beta gamma RF")
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert "radio frequency" not in built.text
        assert built.text == fit_to_input_limit(raw, _tokens_for(room))

    def test_200_byte_token_inside_the_entry(self, expander):
        raw = "RF trip " + "z" * 200 + " BPM noisy quad reset"
        room = 300
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert built.text.startswith("RF radio frequency trip " + "z" * 200)

    def test_straddling_400_byte_token_falls_back_to_byte_cut(self, expander):
        room = 200
        # The 400-byte token starts below three quarters of the room and ends past the room,
        # so no word end lies between the floor and the room: the byte cut is final, and it
        # fills the room, leaving no byte for an insertion.
        head = "RF trip at BPM 3, quad reset; " + "w " * 45
        raw = head + "y" * 400 + " quad"
        assert _nbytes(head) < room * 3 // 4
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert built.text == fit_to_input_limit(raw, _tokens_for(room))
        assert built.kept_chars == room
        assert expander.limits == [room, room]

    def test_cut_between_BPM_and_its_spelled_out_form_dedupe_is_rechecked(self, expander):
        # "BPM | beam position monitor": scanned up to the room, the spelled-out form follows
        # the acronym, so BPM gains nothing and the walk picks the cut after "beam". At that
        # cut "position monitor" is gone, BPM's insertion returns and overflows, and the
        # confirming re-scans move the cut back before BPM.
        raw = "RF " + "w " * 65 + "BPM beam position monitor " + "w " * 200
        room = 160
        bpm = raw.index("BPM")
        room_scan = expander.inner.expand(raw, limit=room)
        assert [i.text for i in room_scan.insertions] == ["radio frequency"]
        assert expander.inner.expand(raw, limit=bpm + len("BPM beam")).insertions[-1].text == (
            "beam position monitor"
        )

        built = _check_invariants(raw, None, _tokens_for(room), expander)

        assert expander.limits == [room, bpm + len("BPM beam"), bpm + 3, bpm - 1]
        assert built.kept_chars == bpm - 1
        assert built.text == "RF radio frequency " + raw[3 : bpm - 1]

    def test_15_acronyms_short_entry_is_never_cut(self, expander):
        raw = " ".join(["RF trip"] * 15)
        room = len(raw) + 40
        built = _check_invariants(raw, None, _tokens_for(room), expander)
        assert built.kept_chars == len(raw)
        assert built.text.count("radio frequency") == 40 // len(" radio frequency")


# ---------------------------------------------------------------------------
# Property tests on seeded random entries
# ---------------------------------------------------------------------------

_WORDS = (
    "RF",
    "BPM",
    "BPMs",
    "quad",
    "ts",
    "t/s",
    "i/o",
    "vert.",
    "orbit",
    "vertical",
    "beam",
    "position",
    "monitor",
    "radio",
    "frequency",
    "|",
    "RFQ",
    "trip",
    "ünïcödé",
    "加速器",
    "x" * 9,
    "z" * 60,
    "y" * 200,
    "RF-trip",
    "(quadrupole)",
    # Spelled-out neighbours: a cut between the acronym and its canonical re-enables the
    # insertion the room scan deduplicated, which is what drives the confirming re-scans.
    "BPM beam position monitor",
    "BPMs beam position monitor",
    "RF radio frequency",
    "quad quadrupole",
)


def _random_entry(rng):
    words = [rng.choice(_WORDS) for _ in range(rng.randint(1, 90))]
    seps = [rng.choice((" ", " ", " ", "  ", "\n", ", ")) for _ in words]
    return "".join(w + s for w, s in zip(words, seps, strict=True)).rstrip()


def _random_cases(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        raw = _random_entry(rng)
        max_tokens = rng.randint(40, 520)
        att = rng.choice((None, None, "", "scope trace: RF phase", "图" * rng.randint(1, 80)))
        yield raw, att, max_tokens


class TestProperties:
    """FR3's invariants hold on seeded random entries."""

    def test_random_entries_satisfy_every_invariant(self, expander):
        for raw, att, max_tokens in _random_cases(seed=20261010, count=500):
            _check_invariants(raw, att, max_tokens, expander)

    def test_fitter_runs_at_most_four_scans(self, expander):
        for raw, att, max_tokens in _random_cases(seed=7, count=500):
            expander.limits.clear()
            build_embedding_input(raw, att, max_tokens, expander=expander)
            assert 1 <= len(expander.limits) <= 4, (raw, att, max_tokens, expander.limits)
