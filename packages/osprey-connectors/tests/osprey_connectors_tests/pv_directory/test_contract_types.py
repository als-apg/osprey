"""The contract every directory-kind backend and its consumers share."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

import osprey_connectors
from osprey_connectors.pv_directory import contract
from osprey_connectors.pv_directory.contract import (
    ALARM_STALE_WINDOW_S,
    AlarmConfigRecord,
    AlarmHistoryRow,
    AlarmLog,
    AlarmState,
    AlarmStateRecord,
    BackendSpec,
    DirectoryPage,
    DirectoryRecord,
    DirectoryRow,
    HistoryPage,
    Outcome,
    PutLog,
    PutRow,
    PVDirectory,
    SearchQuery,
    Source,
    Unavailable,
)

PACKAGE = Path(osprey_connectors.__file__).parent
SOURCE = Source(target="va", service="simulator_view", label="All PVs", rehearsal=False)
ABCS = {
    PVDirectory: {"search", "count", "describe", "property_names", "property_values", "probe"},
    AlarmLog: {"history", "config_for", "probe"},
    AlarmState: {"start", "aclose", "state_for", "config_for", "probe"},
    PutLog: {"history", "probe"},
}
RECORDS = (
    Unavailable,
    Source,
    BackendSpec,
    DirectoryRow,
    DirectoryRecord,
    SearchQuery,
    DirectoryPage,
    AlarmHistoryRow,
    PutRow,
    HistoryPage,
    AlarmStateRecord,
    AlarmConfigRecord,
)


def _spec(**changes) -> BackendSpec:
    fields = {
        "kind": "pv_directory",
        "type": "channelfinder",
        "target": "standin",
        "settings": {"url": "https://cf.example.org"},
        "rehearsal": True,
    }
    return BackendSpec(**(fields | changes))


class _Directory(PVDirectory):
    async def search(self, query, deadline):
        raise NotImplementedError

    async def count(self, query, deadline):
        raise NotImplementedError

    async def describe(self, name, deadline):
        raise NotImplementedError

    async def property_names(self, deadline):
        raise NotImplementedError

    async def property_values(self, name, prefix, deadline):
        raise NotImplementedError

    async def probe(self, deadline):
        raise NotImplementedError


class _Log(PutLog):
    async def history(self, pv, since, limit, deadline):
        raise NotImplementedError

    async def probe(self, deadline):
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
        (Source, ["target", "service", "label", "rehearsal"]),
        (BackendSpec, ["kind", "type", "target", "settings", "rehearsal"]),
        (
            DirectoryRow,
            [
                "name",
                "origin",
                "status",
                "description",
                "record_type",
                "ioc",
                "host",
                "last_announced",
            ],
        ),
        (DirectoryRecord, ["row", "properties", "tags"]),
        (
            SearchQuery,
            [
                "words",
                "name",
                "properties",
                "status",
                "ioc",
                "record_type",
                "origin",
                "page",
                "page_size",
            ],
        ),
        (
            DirectoryPage,
            [
                "outcome",
                "searched",
                "request",
                "source",
                "total",
                "rows",
                "truncated",
                "partial",
                "facets",
                "facets_reason",
                "reason",
            ],
        ),
        (AlarmHistoryRow, ["time", "severity", "status", "message", "user", "kind"]),
        (PutRow, ["at", "pv", "field", "old", "new", "user", "client"]),
        (
            HistoryPage,
            ["outcome", "source", "rows", "truncated", "retention_known", "reason"],
        ),
        (
            AlarmStateRecord,
            ["severity", "status", "latched", "acknowledged", "message", "time", "stale"],
        ),
        (
            AlarmConfigRecord,
            ["in_tree", "path", "guidance", "displays", "enabled", "latching", "label"],
        ),
    ],
)
def test_a_record_has_the_contract_fields_in_order(record, fields):
    assert [f.name for f in dataclasses.fields(record)] == fields


@pytest.mark.parametrize("record", RECORDS)
def test_every_record_is_frozen(record):
    assert record.__dataclass_params__.frozen


def test_unavailable_takes_its_reason_positionally_and_requires_it():
    assert Unavailable("alarm feed still loading").reason == "alarm feed still loading"
    with pytest.raises(TypeError):
        Unavailable()  # type: ignore[call-arg]


def test_a_row_starts_with_an_unknown_origin():
    assert DirectoryRow(name="SR:BPM:01").origin == "unknown"


def test_the_stale_window_is_a_minute():
    assert ALARM_STALE_WINDOW_S == 60


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


def test_a_page_of_any_other_outcome_needs_no_reason():
    page = DirectoryPage(outcome=Outcome.NO_MATCH, searched="q", request="r", source=SOURCE)

    assert (page.rows, page.partial, page.total, page.facets) == ([], [], None, None)


def test_pages_do_not_share_their_default_lists():
    first = HistoryPage(outcome=Outcome.FOUND, source=SOURCE)
    second = HistoryPage(outcome=Outcome.FOUND, source=SOURCE)

    assert first.rows is not second.rows


def test_a_record_converts_to_the_json_shape():
    record = DirectoryRecord(row=DirectoryRow(name="A"), properties={"caPort": "5064"})

    assert dataclasses.asdict(record)["row"]["name"] == "A"
    assert dataclasses.asdict(record)["properties"] == {"caPort": "5064"}


# -- SearchQuery keys a cache -------------------------------------------------


def test_equal_queries_hash_alike_whatever_the_filter_spelling():
    first = SearchQuery(words="bpm", ioc=["a", "b"], properties={"x": "1"})
    second = SearchQuery(words="bpm", ioc=("a", "b"), properties={"x": "1"})

    assert first == second
    assert hash(first) == hash(second)


def test_queries_that_differ_only_in_properties_are_different_keys():
    first = SearchQuery(properties={"x": "1"})
    second = SearchQuery(properties={"x": "2"})

    assert first != second
    assert len({first, second}) == 2


@pytest.mark.parametrize("name", ["ioc", "record_type", "origin"])
def test_one_filter_value_is_one_value_not_its_letters(name):
    assert getattr(SearchQuery(**{name: "abc"}), name) == ("abc",)
    assert getattr(SearchQuery(**{name: None}), name) == ()


# -- The ABCs -----------------------------------------------------------------


@pytest.mark.parametrize(("abc", "methods"), ABCS.items(), ids=lambda v: getattr(v, "__name__", ""))
def test_an_abc_declares_exactly_its_methods(abc, methods):
    assert abc.__abstractmethods__ == methods
    with pytest.raises(TypeError, match="abstract"):
        abc(_spec())


@pytest.mark.parametrize(("abc", "methods"), ABCS.items(), ids=lambda v: getattr(v, "__name__", ""))
def test_every_method_is_async_and_a_read_takes_a_deadline_last(abc, methods):
    for method in methods:
        func = getattr(abc, method)
        assert inspect.iscoroutinefunction(func), f"{abc.__name__}.{method}"
        if method in {"start", "aclose"}:
            continue
        assert list(inspect.signature(func).parameters)[-1] == "deadline", method


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


def test_a_directory_source_carries_its_class_label():
    class Named(_Directory):
        label = "Site directory"

    assert _Directory(_spec()).label == "All PVs"
    assert Named(_spec()).source == Source(
        target="standin", service="channelfinder", label="Site directory", rehearsal=True
    )


def test_another_kind_labels_its_source_with_the_service_name():
    source = _Log(_spec(kind="put_log", type="caputlog", target="live", rehearsal=False)).source

    assert source == Source(target="live", service="caputlog", label="caputlog", rehearsal=False)


# -- The module surface -------------------------------------------------------


def test_every_public_name_of_the_contract_is_listed():
    tree = ast.parse(inspect.getsource(contract))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    } | {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert {name for name in defined if not name.startswith("_")} == set(contract.__all__)


@pytest.mark.parametrize("package", ["pv_directory", "alarm_log", "alarm_state", "put_log"])
def test_a_subpackage_root_re_exports_nothing(package):
    tree = ast.parse((PACKAGE / package / "__init__.py").read_text(encoding="utf-8"))

    assert [type(node) for node in tree.body] == [ast.Expr], "only a docstring"
