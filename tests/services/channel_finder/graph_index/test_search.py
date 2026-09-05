"""Tests for the faceted search the graph finder's page is drawn from.

Every index here is a real one: the corpora are parsed, the rows derived and
the file written by the builder, then reopened read-only exactly as a
deployment opens it. What is asserted is therefore the answer an operator
would get, not what an in-memory table would have said.

The corpus most of the module reads is several of the fixture corpora glued
together, so that one index carries every corner the search has to get right at
once: two sections, a device placed nowhere, one address bound under two
devices, one binding node hung under two devices, and bindings that read, that
write, that do both and that do neither. The expected numbers are small enough
to state by hand, which is the point — a facet count that drifts is meant to be
read off the fixture, not recomputed by the test.
"""

from __future__ import annotations

import time
from importlib.resources import as_file, files
from pathlib import Path
from statistics import median

import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    ParsedCorpus,
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.reader import GraphIndex, open_graph_index

#: The prefixes and ontology every fixture corpus opens with, stripped when
#: several of them are glued into one.
_HEAD = corpora.PREFIXES + corpora.SHARED_ONTOLOGY

SEM = corpora.NARAD_SEM
DEVICE = corpora.DEVICE

#: The keys one search answers with, as ``graph-finder-render.js`` reads them.
PAYLOAD_KEYS = {
    "total",
    "devices",
    "page",
    "pages",
    "page_size",
    "truncated",
    "rows",
    "facets",
    "empty",
    "suggestions",
}

#: The keys one result row carries. Direction is not among them: the table
#: derives it from ``edges``.
ROW_KEYS = {
    "fullPv",
    "description",
    "device",
    "device_uri",
    "section",
    "system",
    "edges",
    "signals",
}


def _combined(*sources: str) -> str:
    """Glue fixture corpora into one, keeping a single copy of the ontology."""
    return _HEAD + "".join(source.removeprefix(_HEAD) for source in sources)


def _meta(parsed: ParsedCorpus, **overrides: object) -> dict:
    """The ``meta`` mapping a corpus build states for *parsed*."""
    values = {
        "corpus_sha256": "b" * 64,
        "corpus_filename": "demo_machine.ttl",
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    values.update(overrides)
    assert set(values) == set(CALLER_META_KEYS)
    return values


def _write(parsed: ParsedCorpus, index_path: Path, **overrides: object) -> Path:
    """Write *parsed* to *index_path* and hand the path back."""
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        _meta(parsed, **overrides),
    )
    return index_path


def _open(index_path: Path) -> GraphIndex:
    """Open a built index, failing loudly on an absence."""
    index = open_graph_index(index_path)
    assert isinstance(index, GraphIndex), index
    return index


def _values(payload: dict, facet: str) -> list[str]:
    """The values of one facet, in the order the rail would draw them."""
    return [entry["value"] for entry in payload["facets"][facet]]


def _counts(payload: dict, facet: str) -> dict[str, int]:
    """One facet as a value-to-count mapping, for order-free assertions."""
    return {entry["value"]: entry["count"] for entry in payload["facets"][facet]}


def _addresses(payload: dict) -> list[str]:
    """The addresses of a page, in the order the table would draw them."""
    return [row["fullPv"] for row in payload["rows"]]


@pytest.fixture(scope="module")
def mixed_index(tmp_path_factory: pytest.TempPathFactory) -> GraphIndex:
    """One index over every corner at once: ten bindings under seven devices.

    ``SR`` holds eight of them and ``BR`` one; ``NOWHERE`` is placed in no
    section and no system at all. ``SR:MAG:SHARED:CURRENT`` is one address
    bound under two devices, and ``SR:MAG:TWICE:CURRENT`` is one binding node
    hung under two devices — the two different ways a ``fullPv`` answers twice.
    """
    parsed = parse_corpus(
        _combined(
            corpora.SUBCLASS_CHAIN,
            corpora.BOTH_EDGES,
            corpora.SHARED_FULL_PV,
            corpora.BINDING_UNDER_TWO_DEVICES,
            corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM,
        )
    )
    path = _write(parsed, tmp_path_factory.mktemp("mixed") / "graph.duckdb")
    with _open(path) as index:
        yield index


@pytest.fixture(scope="module")
def empty_index(tmp_path_factory: pytest.TempPathFactory) -> GraphIndex:
    """An index over a corpus that declares an ontology and binds nothing."""
    parsed = parse_corpus(corpora.NO_BINDINGS)
    path = _write(parsed, tmp_path_factory.mktemp("empty") / "graph.duckdb")
    with _open(path) as index:
        yield index


class TestPayloadShape:
    def test_the_answer_carries_exactly_the_keys_the_finder_reads(self, mixed_index: GraphIndex):
        assert set(mixed_index.search()) == PAYLOAD_KEYS

    def test_a_row_carries_exactly_the_keys_the_table_reads(self, mixed_index: GraphIndex):
        rows = mixed_index.search()["rows"]
        assert rows
        assert all(set(row) == ROW_KEYS for row in rows)

    def test_every_facet_is_present_even_when_it_counted_nothing(self, mixed_index: GraphIndex):
        payload = mixed_index.search(tokens=["no-such-channel"])

        assert payload["total"] == 0
        assert set(payload["facets"]) == {"section", "system", "class", "signal", "dir"}
        assert all(entries == [] for entries in payload["facets"].values())

    def test_signals_are_the_uri_and_name_pairs_the_card_and_table_share(
        self, mixed_index: GraphIndex
    ):
        (row,) = mixed_index.search(tokens=["qf2:current"])["rows"]

        assert row["signals"] == [
            {"uri": f"{SEM}quad_current_rb", "name": "quad_current_rb"},
            {"uri": f"{SEM}quad_current_sp", "name": "quad_current_sp"},
        ]
        assert row["edges"] == ["READSSIGNAL", "WRITESSIGNAL"]

    def test_a_device_placed_nowhere_answers_with_nulls_not_blanks(self, mixed_index: GraphIndex):
        (row,) = mixed_index.search(tokens=["nowhere"])["rows"]

        assert row["section"] is None
        assert row["system"] is None
        assert row["device"] is None
        assert row["description"] == "Unplaced readback"


class TestTotals:
    def test_total_counts_bindings_and_devices_counts_the_distinct_ones(
        self, mixed_index: GraphIndex
    ):
        payload = mixed_index.search()

        assert payload["total"] == 10
        assert payload["devices"] == 7

    def test_one_binding_under_two_devices_is_two_rows_and_two_devices(
        self, mixed_index: GraphIndex
    ):
        payload = mixed_index.search(tokens=["twice"])

        assert payload["total"] == 2
        assert payload["devices"] == 2
        assert _addresses(payload) == ["SR:MAG:TWICE:CURRENT"] * 2


class TestTokens:
    def test_every_token_must_match_the_same_binding(self, mixed_index: GraphIndex):
        assert mixed_index.search(tokens=["qf1"])["total"] == 3
        assert mixed_index.search(tokens=["readback"])["total"] == 2
        assert _addresses(mixed_index.search(tokens=["qf1", "readback"])) == [
            "SR:MAG:QF1:CURRENT:RB"
        ]

    def test_a_token_no_binding_carries_matches_nothing(self, mixed_index: GraphIndex):
        assert mixed_index.search(tokens=["qf1", "sextupole"])["total"] == 0

    def test_tokens_are_folded_before_they_are_matched(self, mixed_index: GraphIndex):
        assert mixed_index.search(tokens=["QF1"])["total"] == 3
        assert mixed_index.search(tokens=["ReadBack"])["total"] == 2

    def test_a_token_matches_a_class_name_the_device_rolls_up_to(self, mixed_index: GraphIndex):
        # Nothing is typed Magnet; every device rolls up to it, and the class
        # names and alt labels are in the haystack the build wrote.
        assert mixed_index.search(tokens=["magnet"])["total"] == 10
        assert mixed_index.search(tokens=["focusing magnet"])["total"] == 10

    def test_no_tokens_match_everything(self, mixed_index: GraphIndex):
        assert mixed_index.search(tokens=[])["total"] == 10


class TestFilters:
    def test_sections_are_ored_within_the_facet(self, mixed_index: GraphIndex):
        assert mixed_index.search(sections=["SR"])["total"] == 8
        assert mixed_index.search(sections=["BR"])["total"] == 1
        assert mixed_index.search(sections=["SR", "BR"])["total"] == 9

    def test_a_device_without_a_section_is_never_kept_by_a_section_filter(
        self, mixed_index: GraphIndex
    ):
        unfiltered = mixed_index.search()
        assert "NOWHERE:RB" in _addresses(unfiltered)

        every_section = mixed_index.search(sections=["SR", "BR"])
        assert "NOWHERE:RB" not in _addresses(every_section)
        assert every_section["total"] == unfiltered["total"] - 1

    def test_a_device_without_a_system_is_never_kept_by_a_system_filter(
        self, mixed_index: GraphIndex
    ):
        payload = mixed_index.search(systems=["MAG"])

        assert payload["total"] == 9
        assert "NOWHERE:RB" not in _addresses(payload)

    def test_filters_are_anded_across_facets(self, mixed_index: GraphIndex):
        # Six bindings carry the readback signal; four of those are in SR, and
        # the intersection is smaller than either selection alone.
        assert mixed_index.search(signals=["quad_current_rb"])["total"] == 6
        assert mixed_index.search(sections=["SR"])["total"] == 8
        assert mixed_index.search(sections=["SR"], signals=["quad_current_rb"])["total"] == 4

    def test_a_class_filter_rolls_its_subclasses_up(self, mixed_index: GraphIndex):
        assert mixed_index.search(cls=f"{SEM}Magnet")["total"] == 10
        assert mixed_index.search(cls=f"{SEM}Quadrupole")["total"] == 10
        assert mixed_index.search(cls=f"{SEM}Sextupole")["total"] == 0

    def test_a_signal_filter_keeps_a_binding_carrying_any_of_the_names(
        self, mixed_index: GraphIndex
    ):
        assert mixed_index.search(signals=["quad_current_sp"])["total"] == 4
        assert mixed_index.search(signals=["quad_current_rb"])["total"] == 6
        assert mixed_index.search(signals=["quad_current_sp", "quad_current_rb"])["total"] == 9

    def test_a_signal_filter_matches_no_name_at_all(self, mixed_index: GraphIndex):
        assert mixed_index.search(signals=["no_such_signal"])["total"] == 0


class TestDirection:
    def test_direction_is_derived_from_the_edges_a_binding_carries(self, mixed_index: GraphIndex):
        assert _addresses(mixed_index.search(dirs=["RW"])) == [
            "SR:MAG:QF2:CURRENT",
            "SR:MAG:QF2:SAME",
        ]
        assert _addresses(mixed_index.search(dirs=["none"])) == ["SR:MAG:QF1:NOTE"]

    def test_a_binding_with_both_edges_matches_r_and_w_as_well_as_rw(self, mixed_index: GraphIndex):
        both = "SR:MAG:QF2:CURRENT"

        assert both in _addresses(mixed_index.search(dirs=["R"]))
        assert both in _addresses(mixed_index.search(dirs=["W"]))
        assert both in _addresses(mixed_index.search(dirs=["RW"]))

    def test_directions_are_ored_within_the_facet(self, mixed_index: GraphIndex):
        assert mixed_index.search(dirs=["R"])["total"] == 7
        assert mixed_index.search(dirs=["W"])["total"] == 4
        assert mixed_index.search(dirs=["R", "W"])["total"] == 9

    def test_the_dir_facet_counts_a_binding_under_every_value_matching_it(
        self, mixed_index: GraphIndex
    ):
        assert _counts(mixed_index.search(), "dir") == {"R": 7, "W": 4, "RW": 2, "none": 1}


class TestFacets:
    def test_a_null_section_is_no_section_the_rail_can_offer(self, mixed_index: GraphIndex):
        payload = mixed_index.search()

        assert _counts(payload, "section") == {"SR": 8, "BR": 1}
        assert _counts(payload, "system") == {"MAG": 9}

    def test_section_and_system_count_bindings(self, mixed_index: GraphIndex):
        assert sum(_counts(mixed_index.search(), "section").values()) == 9

    def test_the_class_facet_counts_devices_not_bindings(self, mixed_index: GraphIndex):
        counts = _counts(mixed_index.search(), "class")

        assert counts == {
            f"{SEM}AcceleratorDevice": 7,
            f"{SEM}Magnet": 7,
            f"{SEM}Quadrupole": 7,
        }

    def test_the_signal_facet_counts_bindings_per_name(self, mixed_index: GraphIndex):
        assert _counts(mixed_index.search(), "signal") == {
            "quad_current_rb": 6,
            "quad_current_sp": 4,
        }

    def test_each_facet_is_counted_with_its_own_filter_lifted(self, mixed_index: GraphIndex):
        payload = mixed_index.search(sections=["BR"])

        assert payload["total"] == 1
        # The section facet still offers what a different section would add …
        assert _counts(payload, "section") == {"SR": 8, "BR": 1}
        # … while every other facet is counted inside the BR selection.
        assert _counts(payload, "system") == {"MAG": 1}
        assert _counts(payload, "dir") == {"R": 1}
        assert _counts(payload, "class") == {
            f"{SEM}AcceleratorDevice": 1,
            f"{SEM}Magnet": 1,
            f"{SEM}Quadrupole": 1,
        }

    def test_the_class_facet_lifts_only_the_class_filter(self, mixed_index: GraphIndex):
        payload = mixed_index.search(cls=f"{SEM}Sextupole")

        assert payload["total"] == 0
        assert _counts(payload, "class") == {
            f"{SEM}AcceleratorDevice": 7,
            f"{SEM}Magnet": 7,
            f"{SEM}Quadrupole": 7,
        }
        assert payload["facets"]["section"] == []

    def test_entries_are_ordered_by_count_then_by_value(self, mixed_index: GraphIndex):
        payload = mixed_index.search()

        assert _values(payload, "section") == ["SR", "BR"]
        assert _values(payload, "signal") == ["quad_current_rb", "quad_current_sp"]
        assert _values(payload, "dir") == ["R", "W", "RW", "none"]
        # Every class holds all seven devices, so the tie breaks on the URI.
        assert _values(payload, "class") == [
            f"{SEM}AcceleratorDevice",
            f"{SEM}Magnet",
            f"{SEM}Quadrupole",
        ]

    def test_a_facet_with_more_values_than_the_cap_is_clipped_and_reported(
        self, mixed_index: GraphIndex
    ):
        payload = mixed_index.search(facet_cap=2)

        assert payload["truncated"] is True
        assert _values(payload, "class") == [f"{SEM}AcceleratorDevice", f"{SEM}Magnet"]
        assert _values(payload, "dir") == ["R", "W"]
        # A facet that fits is not clipped along with the ones that did not.
        assert _values(payload, "section") == ["SR", "BR"]

    def test_a_facet_exactly_at_the_cap_is_not_reported_as_clipped(self, mixed_index: GraphIndex):
        payload = mixed_index.search(facet_cap=4)

        assert payload["truncated"] is False
        assert len(payload["facets"]["dir"]) == 4

    def test_nothing_is_clipped_at_the_default_cap(self, mixed_index: GraphIndex):
        assert mixed_index.search()["truncated"] is False


class TestRowsAndPaging:
    def test_rows_are_ordered_by_address_then_by_device(self, mixed_index: GraphIndex):
        payload = mixed_index.search(page_size=50)

        assert _addresses(payload) == [
            "NOWHERE:RB",
            "SR:MAG:QF1:CURRENT:RB",
            "SR:MAG:QF1:CURRENT:SP",
            "SR:MAG:QF1:NOTE",
            "SR:MAG:QF2:CURRENT",
            "SR:MAG:QF2:SAME",
            "SR:MAG:SHARED:CURRENT",
            "SR:MAG:SHARED:CURRENT",
            "SR:MAG:TWICE:CURRENT",
            "SR:MAG:TWICE:CURRENT",
        ]

    def test_one_address_bound_under_two_devices_breaks_the_tie_on_the_device(
        self, mixed_index: GraphIndex
    ):
        rows = mixed_index.search(tokens=["shared"])["rows"]

        assert [row["device_uri"] for row in rows] == [
            f"{DEVICE}demo_SR_QF3",
            f"{DEVICE}demo_SR_QF4",
        ]

    def test_a_page_is_the_slice_the_offset_and_size_name(self, mixed_index: GraphIndex):
        first = mixed_index.search(skip=0, page_size=3)
        second = mixed_index.search(skip=3, page_size=3)

        assert _addresses(first) == [
            "NOWHERE:RB",
            "SR:MAG:QF1:CURRENT:RB",
            "SR:MAG:QF1:CURRENT:SP",
        ]
        assert _addresses(second) == [
            "SR:MAG:QF1:NOTE",
            "SR:MAG:QF2:CURRENT",
            "SR:MAG:QF2:SAME",
        ]
        assert first["page"] == 1
        assert second["page"] == 2
        assert first["pages"] == second["pages"] == 4
        assert first["page_size"] == 3

    def test_the_last_page_holds_what_is_left(self, mixed_index: GraphIndex):
        payload = mixed_index.search(skip=9, page_size=3)

        assert len(payload["rows"]) == 1
        assert payload["page"] == 4

    def test_a_page_beyond_the_last_is_empty_and_still_reports_the_counts(
        self, mixed_index: GraphIndex
    ):
        payload = mixed_index.search(skip=300, page_size=3)

        assert payload["rows"] == []
        assert payload["page"] == 101
        assert payload["pages"] == 4
        assert payload["total"] == 10
        assert _counts(payload, "section") == {"SR": 8, "BR": 1}

    def test_no_matches_is_no_pages(self, mixed_index: GraphIndex):
        payload = mixed_index.search(tokens=["no-such-channel"])

        assert payload["total"] == 0
        assert payload["pages"] == 0
        assert payload["page"] == 1

    @pytest.mark.parametrize(
        ("kwargs", "fault"),
        [
            ({"page_size": 0}, "page_size"),
            ({"skip": -1}, "skip"),
            ({"facet_cap": -1}, "facet_cap"),
        ],
    )
    def test_a_page_that_cannot_exist_is_refused(
        self, mixed_index: GraphIndex, kwargs: dict, fault: str
    ):
        with pytest.raises(ValueError, match=fault):
            mixed_index.search(**kwargs)


class TestEmptyIndex:
    def test_an_index_that_binds_nothing_says_so_and_names_the_corpus(
        self, empty_index: GraphIndex
    ):
        payload = empty_index.search()

        assert payload["empty"] is True
        assert payload["total"] == 0
        assert payload["devices"] == 0
        assert payload["rows"] == []
        assert all(entries == [] for entries in payload["facets"].values())
        (suggestion,) = payload["suggestions"]
        assert "demo_machine.ttl" in suggestion
        assert "osprey knowledge build-ttl" in suggestion

    def test_an_index_that_binds_something_carries_no_suggestion(self, mixed_index: GraphIndex):
        payload = mixed_index.search(tokens=["no-such-channel"])

        assert payload["empty"] is False
        assert payload["suggestions"] == []


class TestClosedIndex:
    def test_a_search_on_a_closed_index_is_refused(self, tmp_path: Path):
        parsed = parse_corpus(corpora.SUBCLASS_CHAIN)
        index = _open(_write(parsed, tmp_path / "graph.duckdb"))
        index.close()

        with pytest.raises(RuntimeError, match="closed"):
            index.search()


class TestDemoCorpusTiming:
    """The shipped corpus, on the budget the flat index exists to hold."""

    @pytest.fixture(scope="class")
    def demo_index_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        resource = (
            files("osprey.templates")
            .joinpath("apps")
            .joinpath("control_assistant")
            .joinpath("data")
            .joinpath("demo_machine.ttl")
        )
        with as_file(resource) as path:
            parsed = parse_corpus(path.read_text(encoding="utf-8"))
        return _write(parsed, tmp_path_factory.mktemp("demo") / "graph.duckdb")

    def test_the_first_search_after_opening_answers_at_once(self, demo_index_path: Path):
        with _open(demo_index_path) as index:
            started = time.perf_counter()
            payload = index.search()
            elapsed = time.perf_counter() - started

        assert payload["total"] == 2908
        # Budgeted at 100 ms; asserted well above it, because a shared CI
        # machine is slower than a workstation by more than the margin.
        print(f"first search on the demo index: {elapsed * 1000:.1f} ms")
        assert elapsed < 0.5, f"the first search took {elapsed * 1000:.1f} ms"

    def test_a_warm_search_stays_inside_the_interactive_budget(self, demo_index_path: Path):
        shapes: list[dict] = [
            {},
            {"tokens": ["sr"]},
            {"tokens": ["current"]},
            {"tokens": ["quad", "current"]},
            {"tokens": ["bpm"]},
            {"sections": ["SR"]},
            {"systems": ["MAG"]},
            {"dirs": ["R"]},
            {"dirs": ["RW"]},
            {"dirs": ["none"]},
            {"dirs": ["R", "W"]},
            {"tokens": ["sr"], "dirs": ["W"]},
            {"skip": 500},
            {"skip": 2500},
            {"page_size": 200},
            {"facet_cap": 5},
            {"tokens": ["magnet"]},
            {"tokens": ["setpoint"]},
            {"tokens": ["readback"], "dirs": ["R"]},
            {"tokens": ["sr"], "page_size": 100, "facet_cap": 20},
        ]

        with _open(demo_index_path) as index:
            index.search()  # Warm the page cache; the first call is timed above.
            timings = []
            for shape in shapes:
                started = time.perf_counter()
                index.search(**shape)
                timings.append(time.perf_counter() - started)

        p50 = median(timings)
        print(
            f"warm search over {len(shapes)} shapes on the demo index: "
            f"p50 {p50 * 1000:.1f} ms, max {max(timings) * 1000:.1f} ms"
        )
        # Budgeted at 30 ms; asserted well above it, for the same reason.
        assert p50 < 0.2, f"the warm median was {p50 * 1000:.1f} ms"
