"""Tests for ``search_channels``, the keyword tool over the graph search index.

Every index here is a real one: a fixture corpus is parsed, its rows derived and
the file written by the builder, then found the way a deployment finds it — a
``config.yml`` naming a relative ``services.graphdb.index_path``, reached through
``OSPREY_CONFIG``. So what is asserted is the answer the deployed agent would
get, resolution included, not what a hand-built payload would have said.

The tool holds its index for the process's lifetime, which is exactly the state
tests must not share. ``_reset_index`` runs around every test below; the one
test that asserts the holding does its own second call inside a single test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from osprey.mcp_server.channel_finder_graph.tools import search_channels as tool_module
from osprey.mcp_server.channel_finder_graph.tools.search_channels import (
    DIRECTIONS,
    PAGE_SIZE,
    Direction,
    search_channels,
)
from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    ParsedCorpus,
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.reader import (
    GraphIndex,
    GraphIndexAbsence,
)
from tests.services.channel_finder.graph_index import corpora

from .conftest import get_tool_fn

pytestmark = pytest.mark.unit

#: The prefixes and ontology every fixture corpus opens with, stripped when
#: several of them are glued into one.
_HEAD = corpora.PREFIXES + corpora.SHARED_ONTOLOGY

#: The keys one answer carries, as the tool's docstring states them.
PAYLOAD_KEYS = {"total", "devices", "page", "pages", "rows", "facets", "truncated"}

#: The keys one row carries. ``device_uri`` and ``edges`` are not among them:
#: the tool hands out no device URIs, and direction says what the edges meant.
ROW_KEYS = {
    "fullPv",
    "description",
    "device",
    "section",
    "system",
    "direction",
    "signals",
}

#: The tool's raw callable, unwrapped from the FastMCP registration.
_search = get_tool_fn(search_channels)


def _combined(*sources: str) -> str:
    """Glue fixture corpora into one, keeping a single copy of the ontology."""
    return _HEAD + "".join(source.removeprefix(_HEAD) for source in sources)


def _meta(parsed: ParsedCorpus) -> dict:
    """The ``meta`` mapping a corpus build states for *parsed*."""
    values = {
        "corpus_sha256": "c" * 64,
        "corpus_filename": "fixture.ttl",
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    assert set(values) == set(CALLER_META_KEYS)
    return values


def _build_index(corpus: str, index_path: Path) -> None:
    """Write the index a corpus parses to, at *index_path*."""
    parsed = parse_corpus(corpus)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        _meta(parsed),
    )


def _write_config(render: Path, index_path: str | None = "data/graph.duckdb") -> Path:
    """Write a config naming the index relative to its own directory."""
    lines = ["facility:", "  name: Test Facility"]
    if index_path is not None:
        lines += ["services:", "  graphdb:", f"    index_path: {index_path}"]
    config = render / "config.yml"
    config.write_text("\n".join(lines) + "\n")
    return config


@pytest.fixture(autouse=True)
def _reset_tool_index():
    """Make every test open its own index, and close it again afterwards."""
    tool_module._reset_index()
    yield
    tool_module._reset_index()


@pytest.fixture
def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A render directory whose config points at a built fixture index.

    The corpus is several fixture corpora at once: two sections, a device
    placed nowhere, and bindings that read, that write, that do both and that
    do neither — every direction the tool reports, in one index.
    """
    _build_index(
        _combined(
            corpora.SUBCLASS_CHAIN,
            corpora.BOTH_EDGES,
            corpora.SHARED_FULL_PV,
            corpora.BINDING_UNDER_TWO_DEVICES,
            corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM,
        ),
        tmp_path / "data" / "graph.duckdb",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_config(tmp_path)))
    return tmp_path


def _envelope(exc: pytest.ExceptionInfo[ToolError]) -> dict:
    """The structured envelope a raised ``ToolError`` carries."""
    return json.loads(str(exc.value))


class TestKeywordSearch:
    def test_a_keyword_answers_the_addresses_that_carry_it(self, render: Path):
        payload = json.loads(_search(query="qf1"))

        assert set(payload) == PAYLOAD_KEYS
        assert payload["total"] == 3
        assert payload["devices"] == 1
        assert {row["fullPv"] for row in payload["rows"]} == {
            "SR:MAG:QF1:CURRENT:SP",
            "SR:MAG:QF1:CURRENT:RB",
            "SR:MAG:QF1:NOTE",
        }

    def test_a_row_carries_exactly_the_documented_keys(self, render: Path):
        rows = json.loads(_search(query="qf1"))["rows"]

        assert rows
        assert all(set(row) == ROW_KEYS for row in rows)

    def test_direction_is_derived_from_the_rows_edges(self, render: Path):
        payload = json.loads(_search(query="qf"))
        directions = {row["fullPv"]: row["direction"] for row in payload["rows"]}

        assert directions["SR:MAG:QF1:CURRENT:SP"] == "W"
        assert directions["SR:MAG:QF1:CURRENT:RB"] == "R"
        assert directions["SR:MAG:QF1:NOTE"] == "none"
        assert directions["SR:MAG:QF2:CURRENT"] == "RW"

    def test_signals_are_uri_and_name_pairs(self, render: Path):
        (row,) = json.loads(_search(query="qf1:current:rb"))["rows"]

        assert row["signals"] == [
            {"uri": f"{corpora.NARAD_SEM}quad_current_rb", "name": "quad_current_rb"}
        ]

    def test_every_token_must_match(self, render: Path):
        assert json.loads(_search(query="qf1 setpoint"))["total"] == 1
        assert json.loads(_search(query="qf1 nosuchword"))["total"] == 0

    def test_an_unplaced_device_answers_with_null_section_and_system(self, render: Path):
        (row,) = json.loads(_search(query="unplaced"))["rows"]

        assert row["fullPv"] == "NOWHERE:RB"
        assert row["section"] is None
        assert row["system"] is None


class TestFilters:
    def test_a_section_filter_keeps_only_that_section(self, render: Path):
        payload = json.loads(_search(section="BR"))

        assert payload["total"] == 1
        assert [row["fullPv"] for row in payload["rows"]] == ["SR:MAG:TWICE:CURRENT"]
        assert all(row["section"] == "BR" for row in payload["rows"])

    def test_a_direction_filter_keeps_only_that_direction(self, render: Path):
        payload = json.loads(_search(direction="W"))

        assert payload["total"]
        assert all(row["direction"] in {"W", "RW"} for row in payload["rows"])

    def test_a_signal_filter_keeps_only_channels_bound_to_it(self, render: Path):
        payload = json.loads(_search(signal="quad_current_sp"))

        assert payload["total"]
        assert all(
            any(entry["name"] == "quad_current_sp" for entry in row["signals"])
            for row in payload["rows"]
        )

    def test_a_class_filter_rolls_subclasses_up(self, render: Path):
        under_magnet = json.loads(_search(class_uri=f"{corpora.NARAD_SEM}Magnet"))
        everything = json.loads(_search())

        assert under_magnet["total"] == everything["total"]

    def test_filters_are_anded_with_the_query(self, render: Path):
        assert json.loads(_search(query="qf1", section="BR"))["total"] == 0


class TestFacets:
    def test_the_five_facets_are_always_present(self, render: Path):
        payload = json.loads(_search())

        assert set(payload["facets"]) == {"section", "system", "class", "signal", "dir"}

    def test_a_facet_entry_is_a_value_and_a_count(self, render: Path):
        sections = {
            entry["value"]: entry["count"] for entry in json.loads(_search())["facets"]["section"]
        }

        assert sections == {"SR": 8, "BR": 1}

    def test_no_facet_carries_more_than_ten_values(self, render: Path):
        facets = json.loads(_search())["facets"]

        assert all(len(entries) <= 10 for entries in facets.values())

    def test_truncated_says_whether_a_facet_was_cut_off(self, render: Path):
        """The reader's own flag, handed through rather than dropped.

        Ten values is a short list, so an agent that reads a facet and sees
        nothing it wants has to be told whether an eleventh value exists.
        """
        payload = json.loads(_search())

        assert payload["truncated"] is False, (
            "no facet of this fixture index has more than ten values"
        )


class TestPaging:
    def test_the_first_page_holds_every_row_of_a_small_index(self, render: Path):
        payload = json.loads(_search())

        assert payload["page"] == 1
        assert payload["pages"] == 1
        assert payload["total"] == 10
        assert len(payload["rows"]) == payload["total"]

    def test_a_page_beyond_the_end_answers_no_rows_but_the_same_totals(self, render: Path):
        payload = json.loads(_search(page=3))

        assert payload["rows"] == []
        assert payload["page"] == 3
        assert payload["total"] == 10
        assert payload["pages"] == 1

    def test_the_page_size_is_the_finders_own(self):
        """One page width across the three places that spell it.

        The tool spells the number rather than importing the route's, which
        would pull FastAPI into an MCP process. Importing it here instead
        keeps the duplication honest without costing the server anything.
        """
        import inspect

        from osprey.interfaces.channel_finder import database_api

        reader_default = inspect.signature(GraphIndex.search).parameters["page_size"].default

        assert PAGE_SIZE == database_api._GRAPH_SEARCH_PAGE_SIZE
        assert PAGE_SIZE == reader_default


class TestValidation:
    def test_the_four_directions_are_the_annotation(self):
        """The schema an agent reads and the tuple the check uses are one."""
        from typing import get_args

        assert DIRECTIONS == get_args(Direction)
        assert set(DIRECTIONS) == {"R", "W", "RW", "none"}

    def test_an_unknown_direction_is_refused(self, render: Path):
        with pytest.raises(ToolError) as exc:
            _search(direction="read")

        envelope = _envelope(exc)
        assert envelope["error_type"] == "validation_error"
        assert "direction" in envelope["error_message"]
        assert envelope["suggestions"]

    def test_page_zero_is_refused(self, render: Path):
        with pytest.raises(ToolError) as exc:
            _search(page=0)

        envelope = _envelope(exc)
        assert envelope["error_type"] == "validation_error"
        assert "page" in envelope["error_message"]

    def test_a_refusal_never_opens_the_index(self, render: Path):
        with pytest.raises(ToolError):
            _search(page=0)

        assert tool_module._INDEX is None


class TestAbsence:
    def test_a_missing_index_is_a_service_unavailable_naming_the_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        render = tmp_path / "render"
        render.mkdir()
        (render / "config.yml").write_text(
            "facility:\n  name: Test Facility\n"
            "services:\n  graphdb:\n"
            "    ttl_path: data/machine.ttl\n"
            "    index_path: data/graph.duckdb\n"
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(render / "config.yml"))

        with pytest.raises(ToolError) as exc:
            _search(query="qf1")

        envelope = _envelope(exc)
        assert envelope["error_type"] == "service_unavailable"
        assert str(render / "data" / "graph.duckdb") in envelope["error_message"]
        assert envelope["details"]["reason"] == "missing"
        assert any("osprey knowledge build-index" in s for s in envelope["suggestions"])
        assert any("osprey build" in s for s in envelope["suggestions"])

    def test_no_corpus_configured_names_the_corpus_key_instead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OSPREY_CONFIG", str(_write_config(tmp_path)))

        with pytest.raises(ToolError) as exc:
            _search()

        suggestions = _envelope(exc)["suggestions"]
        assert any("services.graphdb.ttl_path" in s for s in suggestions)
        assert not any("osprey knowledge build-index" in s for s in suggestions)

    def test_a_stale_schema_asks_for_a_rebuild_not_a_first_build(
        self, render: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An index this build cannot read is a stale file, not a missing one.

        Telling that project to configure a corpus or to run its first build
        names a step it already took.
        """
        stale = GraphIndexAbsence(
            reason="schema_mismatch",
            path=render / "data" / "graph.duckdb",
            detail="built for schema version 99",
        )
        monkeypatch.setattr(tool_module, "open_graph_index", lambda path: stale)

        with pytest.raises(ToolError) as exc:
            _search()

        envelope = _envelope(exc)
        assert envelope["details"]["reason"] == "schema_mismatch"
        suggestions = envelope["suggestions"]
        assert any("rebuilt" in s for s in suggestions)
        assert any("osprey knowledge build-index" in s for s in suggestions)
        assert not any("osprey build` renders" in s for s in suggestions)

    def test_an_absence_is_not_remembered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A build that runs after the server started must still be picked up."""
        monkeypatch.setenv("OSPREY_CONFIG", str(_write_config(tmp_path)))

        with pytest.raises(ToolError):
            _search()

        _build_index(corpora.SUBCLASS_CHAIN, tmp_path / "data" / "graph.duckdb")

        assert json.loads(_search(query="qf1"))["total"] == 3


class TestHeldIndex:
    def test_a_second_call_reuses_the_index_the_first_one_opened(self, render: Path):
        json.loads(_search(query="qf1"))
        opened = tool_module._INDEX
        assert opened is not None

        json.loads(_search(query="qf2"))

        assert tool_module._INDEX is opened
        assert not opened.closed

    def test_the_index_is_opened_once_however_many_searches_run(
        self, render: Path, monkeypatch: pytest.MonkeyPatch
    ):
        opens = []
        real_open = tool_module.open_graph_index

        def _counting_open(path):
            opens.append(path)
            return real_open(path)

        monkeypatch.setattr(tool_module, "open_graph_index", _counting_open)

        for _ in range(3):
            _search(query="qf1")

        assert len(opens) == 1
