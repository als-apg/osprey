"""The answers every read-only service backend shares, and the contract of each kind."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

import osprey_connectors
from osprey_connectors import service_answers
from osprey_connectors.alarm_log import AlarmHistoryRow, AlarmLog
from osprey_connectors.alarm_state import (
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
    AlarmStateRow,
    AlarmStates,
)
from osprey_connectors.directory import (
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
)
from osprey_connectors.put_log import PutLog, PutRow
from osprey_connectors.service_answers import (
    BackendSpec,
    HistoryPage,
    Outcome,
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
    Source,
    BackendSpec,
    DirectoryRow,
    DirectoryRecord,
    DirectoryQuery,
    DirectoryPage,
    AlarmHistoryRow,
    PutRow,
    HistoryPage,
    AlarmStateRecord,
    AlarmStateRow,
    AlarmStates,
    AlarmConfigRecord,
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
    async def history(self, address, since, limit, *, timeout_s):
        raise NotImplementedError

    async def probe(self, *, timeout_s):
        raise NotImplementedError


# -- The shapes ---------------------------------------------------------------


def test_outcome_is_the_five_words_and_serialises_as_them():
    assert {outcome.value for outcome in Outcome} == {
        "found",
        "inactive",
        "not_listed",
        "no_match",
        "unavailable",
    }
    assert Outcome.NO_MATCH == "no_match"


@pytest.mark.parametrize(
    ("record", "fields"),
    [
        (Source, ["kind", "target", "service", "rehearsal"]),
        (BackendSpec, ["kind", "type", "target", "settings", "rehearsal"]),
        (AlarmHistoryRow, ["time", "severity", "status", "message", "user", "kind"]),
        (PutRow, ["at", "address", "field", "old", "new", "user", "client"]),
        (
            HistoryPage,
            ["outcome", "source", "rows", "truncated", "retention_known", "reason"],
        ),
        (
            AlarmStateRecord,
            ["severity", "status", "latched", "acknowledged", "message", "time", "stale"],
        ),
        (AlarmStateRow, ["address", "path", "state"]),
        (AlarmStates, ["rows", "stale", "last_contact"]),
        (
            AlarmConfigRecord,
            ["in_tree", "path", "guidance", "displays", "enabled", "latching"],
        ),
    ],
)
def test_a_record_has_the_contract_fields_in_order(record, fields):
    assert [*record.model_fields] == fields


def test_the_alarm_list_dumps_its_counts_after_its_fields():
    states = AlarmStates(rows=(), stale=False, last_contact=None)

    assert [*states.model_dump()] == ["rows", "stale", "last_contact", "counts"]


@pytest.mark.parametrize("record", RECORDS)
def test_every_record_is_frozen(record):
    assert record.model_config["frozen"]


def test_unavailable_takes_its_reason_by_keyword_and_requires_it():
    assert Unavailable(reason="alarm feed still loading").reason == "alarm feed still loading"
    with pytest.raises(ValidationError, match="reason"):
        Unavailable()  # type: ignore[call-arg]


def _row(channel: str, severity: str | None) -> AlarmStateRow:
    return AlarmStateRow(address=channel, path=None, state=AlarmStateRecord(severity=severity))


def test_the_counts_are_derived_from_the_rows_and_cannot_be_passed():
    rows = (_row("A", "MAJOR"), _row("B", "OK"), _row("C", "MAJOR"), _row("D", None))
    states = AlarmStates(rows=rows, stale=False, last_contact=None)

    assert dict(states.counts) == {"MAJOR": 2, "OK": 1}
    assert dict(AlarmStates(rows=(), stale=True, last_contact=None).counts) == {}
    with pytest.raises(ValidationError, match="counts"):
        AlarmStates(rows=rows, stale=False, last_contact=None, counts={"OK": 9})  # type: ignore[call-arg]


def test_the_list_converts_to_the_json_shape_with_its_counts():
    states = AlarmStates(rows=(_row("A", "MINOR"),), stale=False, last_contact="t")

    assert states.model_dump()["counts"] == {"MINOR": 1}
    assert '"counts":{"MINOR":1}' in states.model_dump_json()


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: DirectoryPage(
            outcome=Outcome.UNAVAILABLE, searched="", request="", source=SOURCE, **kw
        ),
        lambda **kw: HistoryPage(outcome=Outcome.UNAVAILABLE, source=SOURCE, **kw),
    ],
)
def test_an_unavailable_page_carries_its_reason(build):
    assert build(reason="not available for this target").reason
    with pytest.raises(ValueError, match="reason"):
        build()
    with pytest.raises(ValueError, match="reason"):
        build(reason="")


def test_a_page_holds_its_rows_as_a_tuple():
    assert HistoryPage(outcome=Outcome.FOUND, source=SOURCE).rows == ()
    page = HistoryPage(outcome=Outcome.FOUND, source=SOURCE, rows=["a", "b"])

    assert page.rows == ("a", "b")


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


@pytest.mark.parametrize("record", [Source, AlarmConfigRecord, DirectoryRow])
def test_no_answer_carries_a_display_label(record):
    assert "label" not in record.model_fields
    assert not hasattr(ChannelDirectory, "label")


# -- The module surface -------------------------------------------------------


def _public_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    } | {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    return {name for name in defined if not name.startswith("_")}


def test_every_public_name_of_the_shared_answers_is_listed():
    assert _public_definitions(Path(service_answers.__file__)) == set(service_answers.__all__)


@pytest.mark.parametrize("kind", KINDS)
def test_a_kind_namespace_lists_exactly_its_contract(kind):
    namespace = importlib.import_module(f"osprey_connectors.{kind}")

    assert _public_definitions(PACKAGE / kind / "_contract.py") == set(namespace.__all__)
