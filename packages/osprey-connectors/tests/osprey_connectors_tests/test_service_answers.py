"""The answers every read-only service backend shares, and the contract of each kind."""

from __future__ import annotations

import ast
import importlib
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import osprey_connectors
from osprey_connectors import service_answers
from osprey_connectors.alarm_log import AlarmLog, AlarmLogEntry
from osprey_connectors.alarm_state import (
    AlarmCondition,
    AlarmConfiguration,
    AlarmEntry,
    AlarmSeverity,
    AlarmState,
    AlarmStates,
)
from osprey_connectors.directory import (
    ChannelDirectory,
    DirectoryDetail,
    DirectoryEntry,
    DirectoryPage,
    DirectoryQuery,
)
from osprey_connectors.model import evolve
from osprey_connectors.put_log import PutLog, PutLogEntry
from osprey_connectors.service_answers import (
    BackendSpec,
    LogPage,
    NotListed,
    Page,
    PageOutcome,
    ServiceKind,
    Source,
    Unavailable,
)

PACKAGE = Path(osprey_connectors.__file__).parent
KINDS = ("directory", "alarm_log", "alarm_state", "put_log")
SOURCE = Source(kind="directory", target="va", service="simulator_view", rehearsal=False)
ABCS = {
    ChannelDirectory: {"search", "count", "describe", "property_names", "property_values", "probe"},
    AlarmLog: {"history", "config_for", "probe"},
    AlarmState: {"start", "aclose", "state_for", "states", "config_for", "probe"},
    PutLog: {"history", "probe"},
}
RECORDS = (
    Unavailable,
    NotListed,
    Source,
    BackendSpec,
    DirectoryEntry,
    DirectoryDetail,
    DirectoryQuery,
    DirectoryPage,
    AlarmLogEntry,
    PutLogEntry,
    Page,
    LogPage,
    AlarmCondition,
    AlarmEntry,
    AlarmStates,
    AlarmConfiguration,
)


def _spec(**changes) -> BackendSpec:
    fields = {
        "kind": "directory",
        "type": "channelfinder",
        "target": "standin",
        "settings": {"url": "https://cf.example.org"},
        "rehearsal": True,
    }
    return BackendSpec(**(fields | changes))


class _Directory(ChannelDirectory):
    async def search(self, query, *, timeout_s):
        raise NotImplementedError

    async def count(self, query, *, timeout_s):
        raise NotImplementedError

    async def describe(self, address, *, timeout_s):
        raise NotImplementedError

    async def property_names(self, *, timeout_s):
        raise NotImplementedError

    async def property_values(self, property, *, prefix=None, timeout_s):
        raise NotImplementedError

    async def probe(self, *, timeout_s):
        raise NotImplementedError


class _Log(PutLog):
    async def history(self, address, *, start_at=None, limit, timeout_s):
        raise NotImplementedError

    async def probe(self, *, timeout_s):
        raise NotImplementedError


# -- The shapes ---------------------------------------------------------------


def test_a_page_ends_found_or_no_match_and_never_unavailable():
    assert get_args(PageOutcome) == ("found", "no_match")
    with pytest.raises(ValidationError, match="outcome"):
        Page(outcome="unavailable", source=SOURCE)  # type: ignore[arg-type]


def test_the_archive_is_a_service_kind_and_the_logbook_is_not():
    assert get_args(ServiceKind) == ("directory", "alarm_log", "alarm_state", "put_log", "archive")


@pytest.mark.parametrize(
    ("record", "fields"),
    [
        (Source, ["kind", "target", "service", "rehearsal"]),
        (BackendSpec, ["kind", "type", "target", "settings", "rehearsal"]),
        (
            AlarmLogEntry,
            ["event_at", "severity", "acknowledged", "status", "message", "user", "kind"],
        ),
        (
            PutLogEntry,
            ["put_at", "recorded_stamp", "address", "field", "old", "new", "user", "client"],
        ),
        (Unavailable, ["cause", "reason"]),
        (NotListed, ["address"]),
        (Page, ["outcome", "source", "rows", "truncated", "note"]),
        (LogPage, ["outcome", "source", "rows", "truncated", "note", "retention_known"]),
        (
            DirectoryPage,
            ["outcome", "source", "rows", "truncated", "note", "searched", "request", "total"],
        ),
        (
            AlarmCondition,
            ["severity", "status", "latched", "acknowledged", "message", "event_at", "stale"],
        ),
        (AlarmEntry, ["address", "path", "condition"]),
        (AlarmStates, ["rows", "stale", "last_contact_at"]),
        (
            AlarmConfiguration,
            ["in_tree", "path", "guidance", "displays", "enabled", "latching"],
        ),
    ],
)
def test_a_record_has_the_contract_fields_in_order(record, fields):
    assert [*record.model_fields] == fields


def test_the_alarm_list_dumps_its_counts_after_its_fields():
    states = AlarmStates(rows=(), stale=False, last_contact_at=None)

    assert [*states.model_dump()] == ["rows", "stale", "last_contact_at", "counts"]


@pytest.mark.parametrize("record", RECORDS)
def test_every_record_is_frozen(record):
    assert record.model_config["frozen"]


def test_unavailable_takes_its_cause_and_reason_by_keyword_and_requires_both():
    answer = Unavailable(cause="not_ready", reason="alarm feed still loading")

    assert (answer.cause, answer.reason) == ("not_ready", "alarm feed still loading")
    with pytest.raises(ValidationError, match="cause"):
        Unavailable(reason="alarm feed still loading")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="reason"):
        Unavailable(cause="not_ready")  # type: ignore[call-arg]


def test_an_unavailable_cause_is_one_of_the_closed_words():
    with pytest.raises(ValidationError, match="cause"):
        Unavailable(cause="loading", reason="alarm feed still loading")  # type: ignore[arg-type]


def _row(channel: str, severity: str | None) -> AlarmEntry:
    return AlarmEntry(address=channel, path=None, condition=AlarmCondition(severity=severity))


def test_the_counts_are_derived_from_the_rows_and_cannot_be_passed():
    rows = (_row("A", "MAJOR"), _row("B", "OK"), _row("C", "MAJOR"), _row("D", None))
    states = AlarmStates(rows=rows, stale=False, last_contact_at=None)

    assert dict(states.counts) == {"MAJOR": 2, "OK": 1}
    assert dict(AlarmStates(rows=(), stale=True, last_contact_at=None).counts) == {}
    with pytest.raises(ValidationError, match="counts"):
        AlarmStates(rows=rows, stale=False, last_contact_at=None, counts={"OK": 9})  # type: ignore[call-arg]


def test_the_list_converts_to_the_json_shape_with_its_counts():
    contact = datetime(2026, 11, 6, 12, tzinfo=UTC)
    states = AlarmStates(rows=(_row("A", "MINOR"),), stale=False, last_contact_at=contact)

    assert states.model_dump()["counts"] == {"MINOR": 1}
    assert '"counts":{"MINOR":1}' in states.model_dump_json()
    assert '"last_contact_at":"2026-11-06T12:00:00Z"' in states.model_dump_json()


@pytest.mark.parametrize(
    "build",
    [
        lambda moment: AlarmStates(rows=(), stale=False, last_contact_at=moment),
        lambda moment: AlarmCondition(event_at=moment),
        lambda moment: AlarmLogEntry(event_at=moment),
        lambda moment: PutLogEntry(put_at=moment, address="A"),
        lambda moment: DirectoryEntry(address="A", announced_at=moment),
    ],
)
def test_a_record_refuses_an_instant_with_no_zone_or_as_text(build):
    assert build(datetime(2026, 11, 6, 12, tzinfo=UTC)) is not None
    with pytest.raises(ValueError, match="timezone"):
        build(datetime(2026, 11, 6, 12))
    with pytest.raises(ValueError):
        build("2026-11-06T12:00:00Z")


def test_a_page_holds_its_rows_as_a_tuple():
    assert LogPage(outcome="found", source=SOURCE).rows == ()
    page = LogPage(outcome="found", source=SOURCE, rows=["a", "b"])

    assert page.rows == ("a", "b")


def test_a_directory_page_total_is_a_count_or_why_it_is_unknown():
    unknown = Unavailable(cause="timeout", reason="no answer within 6 s")
    page = DirectoryPage(outcome="no_match", searched="q", request="r", source=SOURCE, total=0)

    assert page.total == 0
    assert evolve(page, total=unknown).total == unknown
    with pytest.raises(ValidationError, match="total"):
        DirectoryPage(outcome="no_match", searched="q", request="r", source=SOURCE)  # type: ignore[call-arg]


#: The severity ranks the alarm contract kept in a private table before ``AlarmSeverity``.
_OLD_RANKS = {"OK": 0, "MINOR": 1, "MAJOR": 2, "INVALID": 3, "UNDEFINED": 4}


def test_a_severity_ranks_as_the_old_table_did():
    assert {severity.value: severity.rank for severity in AlarmSeverity} == _OLD_RANKS
    assert sorted(AlarmSeverity, key=lambda severity: severity.rank) == list(AlarmSeverity)


@pytest.mark.parametrize(
    ("recorded", "read"),
    [
        (None, (None, False)),
        ("OK", (AlarmSeverity.OK, False)),
        ("MAJOR", (AlarmSeverity.MAJOR, False)),
        ("MAJOR_ACK", (AlarmSeverity.MAJOR, True)),
        ("UNDEFINED_ACK", (AlarmSeverity.UNDEFINED, True)),
        ("BOGUS", (AlarmSeverity.UNDEFINED, False)),
        ("BOGUS_ACK", (AlarmSeverity.UNDEFINED, True)),
        (3, (AlarmSeverity.UNDEFINED, False)),
    ],
)
def test_a_recorded_severity_word_reads_as_its_member_and_acknowledgement(recorded, read):
    assert AlarmSeverity.of_recorded(recorded) == read


@pytest.mark.parametrize("record", [AlarmCondition, AlarmLogEntry])
def test_a_record_reads_a_severity_word_as_its_member_and_refuses_another(record):
    assert record(severity="MAJOR").severity is AlarmSeverity.MAJOR
    assert record(severity=AlarmSeverity.MINOR).severity is AlarmSeverity.MINOR
    assert record().severity is None
    with pytest.raises(ValidationError, match="severity"):
        record(severity="MAJOR_ACK")


def test_a_severity_dumps_as_its_word():
    record = AlarmCondition(severity="MAJOR")

    assert record.model_dump(mode="json")["severity"] == "MAJOR"
    assert '"severity":"MAJOR"' in record.model_dump_json()


def test_not_listed_names_the_address_it_was_asked_for():
    assert NotListed(address="SR:BPM1").address == "SR:BPM1"


async def test_a_backend_without_a_lifecycle_starts_and_closes_as_a_no_op():
    backend = _Directory(_spec())

    assert await backend.start() is None
    assert await backend.aclose() is None
    assert await backend.aclose() is None


# -- The ABCs -----------------------------------------------------------------


@pytest.mark.parametrize(("abc", "methods"), ABCS.items(), ids=lambda v: getattr(v, "__name__", ""))
def test_an_abc_declares_exactly_its_methods(abc, methods):
    assert abc.__abstractmethods__ == methods
    with pytest.raises(TypeError, match="abstract"):
        abc(_spec())


@pytest.mark.parametrize(("abc", "methods"), ABCS.items(), ids=lambda v: getattr(v, "__name__", ""))
def test_every_method_is_async_and_a_read_takes_a_keyword_only_timeout_s_last(abc, methods):
    for method in methods:
        func = getattr(abc, method)
        assert inspect.iscoroutinefunction(func), f"{abc.__name__}.{method}"
        if method in {"start", "aclose"}:
            continue
        last = list(inspect.signature(func).parameters.values())[-1]
        assert (last.name, last.kind) == ("timeout_s", inspect.Parameter.KEYWORD_ONLY), method


@pytest.mark.parametrize(
    ("abc", "method"),
    [
        (AlarmLog, "history"),
        (AlarmLog, "config_for"),
        (AlarmState, "state_for"),
        (AlarmState, "config_for"),
        (PutLog, "history"),
    ],
)
def test_a_read_of_one_channel_names_it_address(abc, method):
    assert list(inspect.signature(getattr(abc, method)).parameters)[1] == "address"


@pytest.mark.parametrize("abc", ABCS)
def test_construction_has_the_contract_signature(abc):
    parameters = inspect.signature(abc).parameters

    assert list(parameters) == ["spec", "project_dir", "seams"]
    assert parameters["project_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["project_dir"].default is None
    assert parameters["seams"].kind is inspect.Parameter.VAR_KEYWORD


def test_constructing_a_backend_keeps_its_arguments_and_ignores_other_seams(tmp_path):
    spec = _spec()

    backend = _Directory(spec, project_dir=tmp_path, consumer_factory=object())

    assert backend.spec is spec
    assert backend.project_dir == tmp_path


def test_a_source_names_the_kind_of_its_backend():
    assert _Directory(_spec()).source == Source(
        kind="directory", target="standin", service="channelfinder", rehearsal=True
    )
    source = _Log(_spec(kind="put_log", type="caputlog", target="live", rehearsal=False)).source

    assert source == Source(kind="put_log", target="live", service="caputlog", rehearsal=False)


def test_a_directory_answers_no_field_properties_unless_it_has_them():
    properties = _Directory(_spec()).field_properties

    assert dict(properties) == {}
    with pytest.raises(TypeError):
        properties["server"] = "iocName"  # type: ignore[index]


@pytest.mark.parametrize("record", [Source, AlarmConfiguration, DirectoryEntry])
def test_no_answer_carries_a_display_label(record):
    assert "label" not in record.model_fields
    assert not hasattr(ChannelDirectory, "label")


# -- The module surface -------------------------------------------------------


def _public_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = (
        {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        }
        | {
            node.target.id
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        | {
            # A named closed set is a plain alias assignment.
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
    )
    return {name for name in defined if not name.startswith("_")}


def test_every_public_name_of_the_shared_answers_is_listed():
    assert _public_definitions(Path(service_answers.__file__)) == set(service_answers.__all__)


@pytest.mark.parametrize("kind", KINDS)
def test_a_kind_namespace_lists_exactly_its_contract(kind):
    namespace = importlib.import_module(f"osprey_connectors.{kind}")

    assert _public_definitions(PACKAGE / kind / "_contract.py") == set(namespace.__all__)
