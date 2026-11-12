"""The ChannelFinder backend sends the measured grammar and reads the measured shapes."""

from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors._service_http import ReadOnlyTransport
from osprey_connectors.directory import ROW_FIELD_PROPERTIES, DirectoryQuery, DirectoryRecord
from osprey_connectors.directory._channelfinder import ChannelFinderDirectory
from osprey_connectors.service_answers import BackendSpec, Outcome, Source, Unavailable

DEADLINE = 6.0
FIXTURES = Path(__file__).parent / "fixtures" / "channelfinder"
CHANNELS = "/resources/channels"
COUNT = "/resources/channels/count"
PROPERTIES = "/resources/properties"
DOWN = Unavailable("no answer within 6 s")


def recorded(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class Replay:
    """A transport answering each request from *answer*, and recording what was sent."""

    def __init__(self, answer: Callable[[str, list[tuple[str, Any]]], Any]) -> None:
        self.answer = answer
        self.sent: list[tuple[str, str, list[tuple[str, Any]]]] = []
        self.deadlines: list[float] = []

    async def request(self, method, path, *, params=None, deadline, **_body):
        pairs = list(params or [])
        self.sent.append((method, path, pairs))
        self.deadlines.append(deadline)
        return self.answer(path, pairs)

    def params_to(self, path: str) -> list[list[tuple[str, Any]]]:
        return [pairs for _, sent_path, pairs in self.sent if sent_path == path]


def backend(answer, **settings) -> tuple[ChannelFinderDirectory, Replay]:
    replay = Replay(answer)
    spec = BackendSpec(
        kind="directory",
        type="channelfinder",
        target=settings.pop("target", "live"),
        settings={"url": "https://cf.demo.lab/ChannelFinder", **settings},
        rehearsal=False,
    )
    return ChannelFinderDirectory(spec, transport=replay), replay


def hits_on(key: str, *, rows: Any = None, total: Any = 412) -> Callable:
    """Answer rows and *total* to the requests naming *key*, nothing to any other."""
    rows = recorded("channels") if rows is None else rows

    def answer(path: str, pairs: list[tuple[str, Any]]) -> Any:
        hit = any(name == key for name, _ in pairs)
        return (total if hit else 0) if path == COUNT else (rows if hit else [])

    return answer


def always(*, rows: Any = None, total: Any = 412) -> Callable:
    rows = recorded("channels") if rows is None else rows
    return lambda path, _pairs: total if path == COUNT else rows


class TestWordSearch:
    async def test_words_search_the_description_first_and_stop_there_on_a_hit(self):
        directory, replay = backend(hits_on("recordDesc"), description_property="recordDesc")

        page = await directory.search(DirectoryQuery(words="cooling water"), DEADLINE)

        assert page.outcome is Outcome.FOUND
        assert replay.params_to(CHANNELS) == [
            [("recordDesc", "*cooling*water*"), ("~size", 50), ("~from", 0)]
        ]
        assert replay.params_to(COUNT) == [[("recordDesc", "*cooling*water*")]]
        assert page.total == 412
        assert page.truncated is True
        assert [row.name for row in page.rows] == ["DEMO:RF1:CAV:Temp-I", "DEMO:RF2:CAV:Temp-I"]
        assert page.request == "GET /resources/channels?recordDesc=*cooling*water*&~size=50&~from=0"
        assert page.searched == "name or description matches cooling water"

    async def test_words_fall_to_names_when_no_description_matches(self):
        directory, replay = backend(hits_on("~name"), description_property="recordDesc")

        page = await directory.search(DirectoryQuery(words="temp cav"), DEADLINE)

        assert page.outcome is Outcome.FOUND
        assert replay.params_to(CHANNELS) == [
            [("recordDesc", "*temp*cav*"), ("~size", 50), ("~from", 0)],
            [("~name", "*temp*"), ("~name", "*cav*"), ("~size", 50), ("~from", 0)],
        ]
        assert page.request == (
            "GET /resources/channels?recordDesc=*temp*cav*&~size=50&~from=0"
            " · GET /resources/channels?~name=*temp*&~name=*cav*&~size=50&~from=0"
        )

    async def test_the_configured_description_property_is_the_one_searched(self):
        directory, replay = backend(hits_on("Comment"), description_property="Comment")

        page = await directory.search(DirectoryQuery(words="flow"), DEADLINE)

        assert replay.params_to(CHANNELS)[0][0] == ("Comment", "*flow*")
        assert "Comment=*flow*" in page.request

    async def test_recsyncs_own_description_property_is_the_default(self):
        directory, replay = backend(hits_on("recordDesc"))

        page = await directory.search(DirectoryQuery(words="flow"), DEADLINE)

        assert replay.params_to(CHANNELS)[0][0] == ("recordDesc", "*flow*")
        assert page.rows[0].description == "Cavity 1 cooling water temperature"

    async def test_the_second_search_spends_only_what_the_first_left(self):
        spent = 0.01
        hits = hits_on("~name")

        def slow(path, pairs):
            time.sleep(spent)
            return hits(path, pairs)

        directory, replay = backend(slow)

        await directory.search(DirectoryQuery(words="temp"), DEADLINE)

        assert replay.deadlines[-1] <= DEADLINE - spent

    async def test_a_miss_on_both_searches_is_no_match(self):
        directory, _ = backend(always(rows=[], total=0))

        page = await directory.search(DirectoryQuery(words="cooling"), DEADLINE)

        assert page.outcome is Outcome.NO_MATCH
        assert page.total == 0
        assert page.rows == []
        assert page.truncated is False


class TestNameTerms:
    @pytest.mark.parametrize(
        ("name", "sent"),
        [
            ("RF1", ["*RF1*"]),
            ("DEMO:RF*", ["DEMO:RF*"]),
            ("*Temp-I", ["*Temp-I"]),
            ("DEMO:?F1", ["*DEMO:?F1*"]),
            ("=DEMO:RF1:CAV:Temp-I", ["DEMO:RF1:CAV:Temp-I"]),
            ("RF1 Temp", ["*RF1*", "*Temp*"]),
        ],
    )
    async def test_name_terms_are_globs_wrapped_unless_anchored(self, name, sent):
        directory, replay = backend(always())

        await directory.search(DirectoryQuery(name=name), DEADLINE)

        terms = [value for key, value in replay.params_to(CHANNELS)[0] if key == "~name"]
        assert terms == sent

    async def test_a_word_with_a_glob_is_a_name_term_on_both_searches(self):
        directory, replay = backend(hits_on("nothing"))

        await directory.search(DirectoryQuery(words="cooling DEMO:RF*"), DEADLINE)

        description, names = replay.params_to(CHANNELS)
        assert ("recordDesc", "*cooling*") in description
        assert ("~name", "DEMO:RF*") in description
        assert ("~name", "*cooling*") in names
        assert ("~name", "DEMO:RF*") in names

    async def test_a_query_of_only_name_terms_sends_one_search(self):
        directory, replay = backend(hits_on("nothing"))

        await directory.search(DirectoryQuery(words="=DEMO:RF1"), DEADLINE)

        assert len(replay.params_to(CHANNELS)) == 1

    @pytest.mark.parametrize("separator", [",", "|", ";"])
    async def test_a_separator_in_a_term_cannot_split_it_into_an_or(self, separator):
        directory, replay = backend(always())

        await directory.search(DirectoryQuery(name=f"=DEMO{separator}A"), DEADLINE)

        assert ("~name", "DEMO?A") in replay.params_to(CHANNELS)[0]

    async def test_a_glob_miss_is_no_match(self):
        directory, _ = backend(always(rows=[], total=0))

        page = await directory.search(DirectoryQuery(name="DEMO:NOPE*"), DEADLINE)

        assert page.outcome is Outcome.NO_MATCH

    async def test_an_empty_query_asks_for_every_name(self):
        directory, replay = backend(always())

        page = await directory.search(DirectoryQuery(), DEADLINE)

        assert replay.params_to(CHANNELS)[0][0] == ("~name", "*")
        assert replay.params_to(COUNT)[0] == [("~name", "*")]
        assert page.searched == "every PV"


class TestFilters:
    async def test_every_filter_goes_to_the_server_and_origin_does_not(self):
        directory, replay = backend(always())

        await directory.search(
            DirectoryQuery(
                name="DEMO:*",
                status="Inactive",
                ioc=("rf1-cav", "rf2-cav"),
                record_type="ai",
                origin=("curated",),
            ),
            DEADLINE,
        )

        sent = dict(replay.params_to(CHANNELS)[0])
        assert sent["pvStatus"] == "Inactive"
        assert sent["iocName"] == "rf1-cav|rf2-cav"
        assert sent["recordType"] == "ai"
        assert all("origin" not in key for key in sent)
        assert dict(replay.params_to(COUNT)[0]) == {
            key: value for key, value in sent.items() if not key.startswith(("~size", "~from"))
        }

    async def test_an_or_comes_only_from_a_typed_list(self):
        directory, replay = backend(always())

        await directory.search(DirectoryQuery(ioc=("a,b", "c")), DEADLINE)

        assert dict(replay.params_to(CHANNELS)[0])["iocName"] == "a?b|c"

    async def test_exact_properties_are_never_wildcarded(self):
        directory, replay = backend(always(), exact_properties=["recordType", "iocName"])

        await directory.search(
            DirectoryQuery(properties={"iocName": "rf1-cav", "Engineer": "rf"}), DEADLINE
        )

        sent = dict(replay.params_to(CHANNELS)[0])
        assert sent["iocName"] == "rf1-cav"
        assert sent["Engineer"] == "*rf*"

    async def test_a_malformed_exact_list_names_no_property(self):
        directory, replay = backend(always(), exact_properties=[{"not": "a name"}])

        await directory.search(DirectoryQuery(properties={"Engineer": "rf"}), DEADLINE)

        assert dict(replay.params_to(CHANNELS)[0])["Engineer"] == "*rf*"

    async def test_status_and_record_type_are_exact_by_default(self):
        directory, replay = backend(always())

        await directory.search(
            DirectoryQuery(properties={"pvStatus": "Active", "recordType": "ai"}), DEADLINE
        )

        sent = dict(replay.params_to(CHANNELS)[0])
        assert (sent["pvStatus"], sent["recordType"]) == ("Active", "ai")

    @pytest.mark.parametrize(
        ("value", "sent"),
        [("=rf-group", "rf-group"), ("rf*", "rf*"), ("*group", "*group"), ("rf", "*rf*")],
    )
    async def test_a_property_value_reads_like_a_name_term(self, value, sent):
        directory, replay = backend(always())

        await directory.search(DirectoryQuery(properties={"Engineer": value}), DEADLINE)

        assert dict(replay.params_to(CHANNELS)[0])["Engineer"] == sent

    async def test_the_plain_words_name_the_filters(self):
        directory, _ = backend(always())

        page = await directory.search(
            DirectoryQuery(words="flow", status="Active", ioc=("a", "b"), record_type="ai"),
            DEADLINE,
        )

        assert page.searched == (
            "name or description matches flow · status Active · IOC a or b · record type ai"
        )


class TestPaging:
    async def test_a_page_is_an_offset_in_rows(self):
        directory, replay = backend(always())

        await directory.search(DirectoryQuery(name="DEMO*", page=2, page_size=25), DEADLINE)

        sent = replay.params_to(CHANNELS)[0]
        assert ("~size", 25) in sent
        assert ("~from", 50) in sent

    async def test_the_last_page_is_not_truncated(self):
        directory, _ = backend(always(total=2))

        page = await directory.search(DirectoryQuery(name="DEMO*"), DEADLINE)

        assert page.truncated is False

    async def test_past_the_end_of_a_description_search_stays_on_it(self):
        directory, replay = backend(always(rows=[], total=3))

        page = await directory.search(DirectoryQuery(words="flow", page=1), DEADLINE)

        assert len(replay.params_to(CHANNELS)) == 1
        assert page.outcome is Outcome.FOUND
        assert page.rows == []
        assert page.total == 3

    async def test_a_later_page_of_a_name_search_reaches_the_name_search(self):
        directory, replay = backend(hits_on("~name"))

        page = await directory.search(DirectoryQuery(words="temp", page=1), DEADLINE)

        assert len(replay.params_to(CHANNELS)) == 2
        assert page.rows


class TestPartial:
    async def test_a_page_whose_count_failed_is_partial(self):
        def answer(path, _pairs):
            return DOWN if path == COUNT else recorded("channels")

        directory, _ = backend(answer)

        page = await directory.search(DirectoryQuery(words="cooling"), DEADLINE)

        assert page.outcome is Outcome.FOUND
        assert page.total is None
        assert page.partial == [{"part": "count", "reason": DOWN.reason}]
        assert len(page.rows) == 2
        assert page.truncated is False

    async def test_a_full_page_whose_count_failed_may_have_more(self):
        def answer(path, _pairs):
            return DOWN if path == COUNT else recorded("channels")

        directory, _ = backend(answer)

        page = await directory.search(DirectoryQuery(name="DEMO*", page_size=2), DEADLINE)

        assert page.truncated is True

    async def test_a_page_that_failed_is_unavailable_with_its_reason(self):
        def answer(path, _pairs):
            return 412 if path == COUNT else DOWN

        directory, replay = backend(answer)

        page = await directory.search(DirectoryQuery(words="cooling"), DEADLINE)

        assert page.outcome is Outcome.UNAVAILABLE
        assert page.reason == DOWN.reason
        assert page.total == 412
        assert page.request.startswith("GET /resources/channels?recordDesc=")
        assert len(replay.params_to(CHANNELS)) == 1

    async def test_a_first_page_with_no_rows_and_no_count_still_tries_the_names(self):
        def answer(path, pairs):
            if path == COUNT:
                return DOWN
            return recorded("channels") if pairs[0][0] == "~name" else []

        directory, _ = backend(answer)

        page = await directory.search(DirectoryQuery(words="temp"), DEADLINE)

        assert page.outcome is Outcome.FOUND
        assert page.partial == [{"part": "count", "reason": DOWN.reason}]

    async def test_a_later_page_with_no_rows_and_no_count_says_it_cannot_tell(self):
        def answer(path, _pairs):
            return DOWN if path == COUNT else []

        directory, replay = backend(answer)

        page = await directory.search(DirectoryQuery(words="temp", page=1), DEADLINE)

        assert page.outcome is Outcome.UNAVAILABLE
        assert DOWN.reason in page.reason
        assert len(replay.params_to(CHANNELS)) == 1

    async def test_no_rows_and_no_count_on_the_last_search_is_no_match_with_the_reason(self):
        def answer(path, _pairs):
            return DOWN if path == COUNT else []

        directory, _ = backend(answer)

        page = await directory.search(DirectoryQuery(name="DEMO:NOPE*"), DEADLINE)

        assert page.outcome is Outcome.NO_MATCH
        assert page.total is None
        assert page.partial == [{"part": "count", "reason": DOWN.reason}]


class TestRows:
    async def test_the_listed_fields_become_the_row(self):
        directory, _ = backend(always())

        page = await directory.search(DirectoryQuery(name="DEMO*"), DEADLINE)

        row = page.rows[0]
        assert row.origin == "unknown"
        assert row.status == "Active"
        assert row.description == "Cavity 1 cooling water temperature"
        assert row.record_type == "ai"
        assert row.ioc == "rf1-cav"
        assert row.host == "ioc-rf1.demo.lab"
        assert row.last_announced == "2026-10-30 08:12:44.501223"

    async def test_an_inactive_channel_reads_inactive(self):
        directory, _ = backend(always())

        page = await directory.search(DirectoryQuery(status="Inactive"), DEADLINE)

        assert page.rows[1].status == "Inactive"
        assert page.outcome is Outcome.FOUND

    async def test_every_property_is_optional(self):
        directory, _ = backend(always(rows=[{"name": "DEMO:BARE"}], total=1))

        page = await directory.search(DirectoryQuery(name="DEMO*"), DEADLINE)

        row = page.rows[0]
        assert (row.name, row.status, row.ioc, row.description) == ("DEMO:BARE", None, None, None)

    @pytest.mark.parametrize(
        "body",
        [{"channels": []}, "nope", [{"owner": "recceiver"}], ["DEMO:A"], [{"name": 7}]],
    )
    async def test_a_body_that_is_not_channels_is_malformed(self, body):
        directory, _ = backend(lambda path, _pairs: 1 if path == COUNT else body)

        page = await directory.search(DirectoryQuery(name="DEMO*"), DEADLINE)

        assert page.outcome is Outcome.UNAVAILABLE
        assert page.reason == "malformed response"


class TestCount:
    async def test_the_bare_integer_is_the_count(self):
        directory, replay = backend(always(total=recorded("count")))

        assert await directory.count(DirectoryQuery(words="cooling"), DEADLINE) == 412
        assert replay.params_to(COUNT) == [[("recordDesc", "*cooling*")]]
        assert replay.params_to(CHANNELS) == []

    async def test_it_counts_the_names_when_no_description_matches(self):
        directory, replay = backend(hits_on("~name", total=9))

        assert await directory.count(DirectoryQuery(words="temp"), DEADLINE) == 9
        assert len(replay.params_to(COUNT)) == 2

    async def test_no_search_matching_counts_zero(self):
        directory, _ = backend(always(total=0))

        assert await directory.count(DirectoryQuery(words="temp"), DEADLINE) == 0

    async def test_a_failure_is_the_reason(self):
        directory, _ = backend(always(total=DOWN))

        assert await directory.count(DirectoryQuery(words="temp"), DEADLINE) == DOWN

    @pytest.mark.parametrize("body", ["412", 4.5, True, -1, None, [412]])
    async def test_anything_but_a_count_is_malformed(self, body):
        directory, _ = backend(always(total=body))

        assert await directory.count(DirectoryQuery(name="DEMO*"), DEADLINE) == Unavailable(
            "malformed response"
        )


class TestDescribe:
    async def test_a_record_holds_the_rest_of_the_properties(self):
        directory, replay = backend(always())

        record = await directory.describe("DEMO:RF1:CAV:Temp-I", DEADLINE)

        assert isinstance(record, DirectoryRecord)
        assert record.row.status == "Active"
        assert record.row.origin == "unknown"
        assert record.properties["caPort"] == "5064"
        assert record.properties["Engineer"] == "rf-group"
        assert record.tags == []
        for mapped in ("pvStatus", "time", "iocName", "hostName", "recordType", "recordDesc"):
            assert mapped not in record.properties
        assert ("~name", "DEMO:RF1:CAV:Temp-I") in replay.params_to(CHANNELS)[0]

    async def test_each_mapped_property_fills_its_row_field(self):
        directory, _ = backend(always())
        fixture = recorded("channels")[0]
        values = {prop["name"]: prop["value"] for prop in fixture["properties"]}

        record = await directory.describe(fixture["name"], DEADLINE)

        assert isinstance(record, DirectoryRecord)
        for field, prop in ROW_FIELD_PROPERTIES.items():
            assert getattr(record.row, field) == values[prop]
            assert prop not in record.properties
        assert record.row.description == values["recordDesc"]

    async def test_a_record_carries_its_tag_names(self):
        directory, _ = backend(always())

        record = await directory.describe("DEMO:RF2:CAV:Temp-I", DEADLINE)

        assert record.tags == ["archived"]
        assert record.row.status == "Inactive"
        assert record.properties["alias"] == "DEMO:RF2:Temp"

    async def test_an_exact_miss_is_not_listed(self):
        directory, _ = backend(always(rows=[]))

        assert await directory.describe("DEMO:NOPE", DEADLINE) is Outcome.NOT_LISTED

    async def test_an_answer_that_holds_other_names_is_not_listed(self):
        directory, _ = backend(always())

        assert await directory.describe("DEMO:*", DEADLINE) is Outcome.NOT_LISTED

    async def test_a_comma_in_the_name_still_finds_that_name(self):
        directory, replay = backend(always(rows=[{"name": "DEMO:A.B"}, {"name": "DEMO:A,B"}]))

        record = await directory.describe("DEMO:A,B", DEADLINE)

        assert record.row.name == "DEMO:A,B"
        assert ("~name", "DEMO:A?B") in replay.params_to(CHANNELS)[0]

    async def test_a_failure_is_the_reason_not_a_miss(self):
        directory, _ = backend(always(rows=DOWN))

        assert await directory.describe("DEMO:A", DEADLINE) == DOWN


class TestProperties:
    async def test_the_names_come_from_the_properties_route(self):
        directory, replay = backend(lambda _path, _pairs: recorded("properties"))

        assert await directory.property_names(DEADLINE) == ["Engineer", "iocName", "pvStatus"]
        assert replay.sent[0][:2] == ("GET", PROPERTIES)

    async def test_a_failed_properties_route_is_the_reason(self):
        directory, _ = backend(lambda _path, _pairs: DOWN)

        assert await directory.property_names(DEADLINE) == DOWN

    async def test_values_are_read_off_the_channels_that_have_the_property(self):
        directory, replay = backend(always())

        assert await directory.property_values("iocName", None, DEADLINE) == [
            "rf1-cav",
            "rf2-cav",
        ]
        assert replay.params_to(CHANNELS)[0][0] == ("iocName", "*")

    async def test_a_prefix_is_sent_and_holds_on_the_answer(self):
        directory, replay = backend(always())

        assert await directory.property_values("iocName", "rf1", DEADLINE) == ["rf1-cav"]
        assert replay.params_to(CHANNELS)[0][0] == ("iocName", "rf1*")

    async def test_a_failed_read_of_values_is_the_reason(self):
        directory, _ = backend(always(rows=DOWN))

        assert await directory.property_values("iocName", None, DEADLINE) == DOWN


class TestProbe:
    async def test_the_evidence_names_the_properties_and_the_newest_announcement(self):
        def answer(path, _pairs):
            return recorded("properties") if path == PROPERTIES else recorded("channels")

        directory, _ = backend(answer)

        evidence = await directory.probe(DEADLINE)

        assert "3 property names" in evidence
        assert "2026-10-30 08:12:44.501223" in evidence

    async def test_a_silent_properties_route_is_the_reason(self):
        directory, _ = backend(lambda _path, _pairs: DOWN)

        assert await directory.probe(DEADLINE) == DOWN

    async def test_a_failed_sample_is_named_beside_the_answer(self):
        def answer(path, _pairs):
            return recorded("properties") if path == PROPERTIES else DOWN

        directory, _ = backend(answer)

        evidence = await directory.probe(DEADLINE)

        assert "3 property names" in evidence
        assert DOWN.reason in evidence


class TestConstruction:
    def test_it_takes_the_contracts_seams_and_does_no_io(self):
        spec = BackendSpec(
            kind="directory",
            type="channelfinder",
            target="standin",
            settings={},
            rehearsal=True,
        )

        directory = ChannelFinderDirectory(spec, project_dir=Path("."), consumer_factory=object())

        assert directory.source == Source(
            target="standin", service="channelfinder", label="All PVs", rehearsal=True
        )

    def test_it_may_ask_for_reads_only(self):
        assert set(ChannelFinderDirectory.ALLOWED_REQUESTS) == {
            ("GET", CHANNELS),
            ("GET", COUNT),
            ("GET", PROPERTIES),
            ("GET", "/resources/tags"),
        }

    async def test_a_login_page_reaches_the_page_as_its_reason(self):
        login = Unavailable(
            "answered with a login page (an access proxy is in front of the service)"
        )
        directory, _ = backend(lambda _path, _pairs: login)

        page = await directory.search(DirectoryQuery(name="DEMO*"), DEADLINE)

        assert page.outcome is Outcome.UNAVAILABLE
        assert page.reason == login.reason

    async def test_a_block_with_no_url_answers_a_reason(self):
        spec = BackendSpec(
            kind="directory",
            type="channelfinder",
            target="live",
            settings={"timeout_s": "soon"},
            rehearsal=False,
        )

        page = await ChannelFinderDirectory(spec).search(DirectoryQuery(), DEADLINE)

        assert page.outcome is Outcome.UNAVAILABLE
        assert page.reason

    async def test_every_request_passes_the_real_transports_allowlist(self, monkeypatch):
        urls: list[str] = []

        def exchange(_self, request, _bound):
            urls.append(request.full_url)
            path = urllib.parse.urlsplit(request.full_url).path
            if path.endswith("/count"):
                return 412
            return recorded("properties" if path.endswith("/properties") else "channels")

        monkeypatch.setattr(ReadOnlyTransport, "_exchange", exchange)
        spec = BackendSpec(
            kind="directory",
            type="channelfinder",
            target="live",
            settings={
                "url": "https://cf.demo.lab/ChannelFinder",
                "description_property": "recordDesc",
                "exact_properties": ["pvStatus"],
                "caveats": ["a note"],
            },
            rehearsal=False,
        )
        directory = ChannelFinderDirectory(spec)

        page = await directory.search(DirectoryQuery(words="cooling"), DEADLINE)
        record = await directory.describe("DEMO:RF1:CAV:Temp-I", DEADLINE)
        evidence = await directory.probe(DEADLINE)

        assert page.outcome is Outcome.FOUND
        assert isinstance(record, DirectoryRecord)
        assert isinstance(evidence, str)
        assert (
            "https://cf.demo.lab/ChannelFinder/resources/channels"
            "?recordDesc=%2Acooling%2A&~size=50&~from=0"
        ) in urls


def test_recorded_shapes_use_demo_names_only():
    channels: list[Mapping[str, Any]] = recorded("channels")
    assert all(channel["name"].startswith("DEMO:") for channel in channels)
