"""Tests for the roster source contract: rows, the loader, the merge, conformance.

A roster source is deployment code that states, per IOC, its host, class, whether
it is expected to run and optionally a group. The merge yields one row per IOC
name naming every source that stated it; a source that raises is reported
unavailable with the exception's class as the reason and the others are kept.
The documented example source (a static list read from a YAML file) is checked
with the same conformance helper a deployment calls on its own source.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

from osprey.services.channel_directory import roster
from osprey.services.channel_directory.roster import (
    MergedRosterRow,
    RosterMerge,
    RosterRow,
    RosterSource,
    RosterSourceError,
    SourceOutcome,
    check_roster_source,
    load_roster_source,
    load_roster_sources,
    merge_rosters,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "docs/source/how-to/control-systems/examples/roster"
EXAMPLE_SOURCE = EXAMPLES / "roster_static.py"

PUBLIC_NAMES = {
    "MergedRosterRow",
    "RosterMerge",
    "RosterRow",
    "RosterSource",
    "RosterSourceError",
    "SourceOutcome",
    "check_roster_source",
    "load_roster_source",
    "load_roster_sources",
    "merge_rosters",
}


class _Static:
    def __init__(self, name: str, rows: list[RosterRow]) -> None:
        self.name = name
        self._rows = rows
        self.deadlines: list[float] = []

    def read(self, deadline: float) -> list[RosterRow]:
        self.deadlines.append(deadline)
        return list(self._rows)


class _Raising:
    def __init__(self, name: str, exc: BaseException) -> None:
        self.name = name
        self._exc = exc

    def read(self, deadline: float) -> list[RosterRow]:
        del deadline
        raise self._exc


class _Returns:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def read(self, deadline: float) -> object:
        del deadline
        return self._value


# --- public names ---------------------------------------------------------------------


def test_public_names_are_the_contract() -> None:
    assert set(roster.__all__) == PUBLIC_NAMES
    for name in roster.__all__:
        assert getattr(roster, name) is not None


def test_package_root_reexports_nothing_of_the_roster() -> None:
    from osprey.services import channel_directory

    assert not PUBLIC_NAMES & set(getattr(channel_directory, "__all__", ()))


# --- RosterRow ------------------------------------------------------------------------


def test_row_positional_with_optional_group() -> None:
    row = RosterRow("SR01-IOC", "srv1", "softioc", "running")
    assert (row.name, row.host, row.class_name, row.expected, row.group) == (
        "SR01-IOC",
        "srv1",
        "softioc",
        "running",
        None,
    )
    assert RosterRow("B", None, None, None, group="booster").group == "booster"


@pytest.mark.parametrize("expected", ["running", "stopped", None])
def test_row_accepts_each_expectation(expected: str | None) -> None:
    assert RosterRow("A", "h", "c", expected).expected == expected


@pytest.mark.parametrize("expected", ["Running", "up", "", True])
def test_row_refuses_other_expectations(expected: object) -> None:
    with pytest.raises(ValueError, match="expected"):
        RosterRow("A", "h", "c", expected)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "   ", None, 3])
def test_row_refuses_a_blank_name(name: object) -> None:
    with pytest.raises(ValueError, match="name"):
        RosterRow(name, "h", "c", None)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["host", "class_name", "group"])
@pytest.mark.parametrize("value", [10, ["a", "b"], {"x": 1}, 1.5])
def test_row_refuses_a_non_string_field(field: str, value: object) -> None:
    fields: dict[str, object] = {"host": None, "class_name": None, "group": None, field: value}
    with pytest.raises(ValueError, match=rf"'A'.*{field}"):
        RosterRow("A", fields["host"], fields["class_name"], None, group=fields["group"])  # type: ignore[arg-type]


def test_row_is_frozen() -> None:
    row = RosterRow("A", "h", "c", None)
    with pytest.raises(AttributeError):
        row.name = "B"  # type: ignore[misc]


# --- RosterSource protocol ------------------------------------------------------------


def test_protocol_recognises_a_source_by_its_members() -> None:
    assert isinstance(_Static("s", []), RosterSource)
    assert not isinstance(object(), RosterSource)


# --- merge ----------------------------------------------------------------------------


def test_merge_yields_one_row_per_name_naming_every_source() -> None:
    a = _Static(
        "cmdb", [RosterRow("IOC1", "h1", "soft", "running"), RosterRow("IOC2", "h2", None, None)]
    )
    b = _Static("siocmgr", [RosterRow("IOC1", "h9", "hard", "stopped", group="sr")])
    merged = merge_rosters([a, b], deadline=2.0)

    assert isinstance(merged, RosterMerge)
    assert [row.name for row in merged.rows] == ["IOC1", "IOC2"]
    ioc1 = merged.rows[0]
    assert isinstance(ioc1, MergedRosterRow)
    assert ioc1.sources == ("cmdb", "siocmgr")
    # the first source in configured order wins each field it states
    assert (ioc1.host, ioc1.class_name, ioc1.expected) == ("h1", "soft", "running")
    # a field the first source leaves empty is filled from a later one
    assert ioc1.group == "sr"
    assert merged.rows[1].sources == ("cmdb",)
    assert merged.sources == (
        SourceOutcome("cmdb", "ok", None),
        SourceOutcome("siocmgr", "ok", None),
    )
    assert a.deadlines == [2.0] and b.deadlines == [2.0]


def test_merge_fills_an_empty_field_from_a_later_source() -> None:
    a = _Static("a", [RosterRow("X", None, None, None)])
    b = _Static("b", [RosterRow("X", "hb", "cb", "stopped", group="g")])
    (row,) = merge_rosters([a, b], deadline=1.0).rows
    assert (row.host, row.class_name, row.expected, row.group) == ("hb", "cb", "stopped", "g")


def test_a_name_twice_in_one_source_names_that_source_once() -> None:
    a = _Static("a", [RosterRow("X", "h1", None, None), RosterRow("X", "h2", "c", None)])
    (row,) = merge_rosters([a], deadline=1.0).rows
    assert row.sources == ("a",)
    assert (row.host, row.class_name) == ("h1", "c")


def test_failing_source_is_unavailable_with_its_exception_class_and_others_kept() -> None:
    good = _Static("good", [RosterRow("IOC1", "h", "c", "running")])
    bad = _Raising("bad", ConnectionRefusedError("refused: secret-host:5432"))
    merged = merge_rosters([bad, good], deadline=1.0)

    assert [row.name for row in merged.rows] == ["IOC1"]
    assert merged.rows[0].sources == ("good",)
    assert merged.sources == (
        SourceOutcome("bad", "unavailable", "ConnectionRefusedError"),
        SourceOutcome("good", "ok", None),
    )


def test_a_source_returning_something_else_is_unavailable() -> None:
    merged = merge_rosters(
        [_Returns("not-list", {"IOC": 1}), _Returns("bad-item", [("IOC", "h")])], deadline=1.0
    )
    assert merged.rows == ()
    assert [(s.name, s.outcome, s.reason) for s in merged.sources] == [
        ("not-list", "unavailable", "TypeError"),
        ("bad-item", "unavailable", "TypeError"),
    ]


def test_a_tuple_of_rows_is_unavailable_as_conformance_says() -> None:
    src = _Returns("t", (RosterRow("A", None, None, None),))
    merged = merge_rosters([src], deadline=1.0)
    assert merged.rows == ()
    assert merged.sources == (SourceOutcome("t", "unavailable", "TypeError"),)
    assert any("must return a list" in p for p in check_roster_source(src))  # type: ignore[arg-type]


def test_merge_with_no_sources_is_empty() -> None:
    assert merge_rosters([], deadline=1.0) == RosterMerge(rows=(), sources=())


def test_keyboard_interrupt_is_not_swallowed() -> None:
    with pytest.raises(KeyboardInterrupt):
        merge_rosters([_Raising("k", KeyboardInterrupt())], deadline=1.0)


# --- loader ---------------------------------------------------------------------------


@pytest.fixture
def plugin_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("_roster_plugin_under_test")
    module.instance = _Static("inst", [RosterRow("A", None, None, None)])  # type: ignore[attr-defined]
    module.factory = lambda: _Static("made", [])  # type: ignore[attr-defined]
    module.Klass = type(  # type: ignore[attr-defined]
        "Klass", (), {"name": "klass", "read": lambda self, deadline: []}
    )
    module.bad_factory = lambda: 42  # type: ignore[attr-defined]
    module.raising_factory = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
    module.not_a_source = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_loader_returns_an_instance_as_is(plugin_module: types.ModuleType) -> None:
    assert load_roster_source("_roster_plugin_under_test:instance") is plugin_module.instance


@pytest.mark.usefixtures("plugin_module")
def test_loader_calls_a_zero_argument_factory() -> None:
    assert load_roster_source("_roster_plugin_under_test:factory").name == "made"


def test_loader_instantiates_a_class(plugin_module: types.ModuleType) -> None:
    source = load_roster_source("_roster_plugin_under_test:Klass")
    assert isinstance(source, plugin_module.Klass)
    assert source.name == "klass"


@pytest.mark.usefixtures("plugin_module")
def test_loader_keeps_configured_order() -> None:
    sources = load_roster_sources(
        ["_roster_plugin_under_test:factory", "_roster_plugin_under_test:instance"]
    )
    assert [s.name for s in sources] == ["made", "inst"]


@pytest.mark.parametrize(
    ("ref", "fragment"),
    [
        ("no_colon_here", "module:attribute"),
        (":attr", "module:attribute"),
        ("mod:", "module:attribute"),
        ("_roster_no_such_module_xyz:thing", "cannot import"),
        ("_roster_plugin_under_test:missing", "no attribute"),
        ("_roster_plugin_under_test:not_a_source", "not a roster source"),
        ("_roster_plugin_under_test:bad_factory", "not a roster source"),
        ("_roster_plugin_under_test:raising_factory", "factory raised"),
    ],
)
@pytest.mark.usefixtures("plugin_module")
def test_loader_refuses_with_the_reference_named(ref: str, fragment: str) -> None:
    with pytest.raises(RosterSourceError, match=fragment) as info:
        load_roster_source(ref)
    assert ref in str(info.value)


def test_loader_wraps_any_exception_raised_at_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "_roster_raises_on_import.py").write_text("raise RuntimeError('setup failed')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "_roster_raises_on_import", raising=False)
    ref = "_roster_raises_on_import:roster"
    with pytest.raises(RosterSourceError, match="raised RuntimeError") as info:
        load_roster_source(ref)
    assert ref in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_loader_refuses_two_sources_with_one_name(plugin_module: types.ModuleType) -> None:
    plugin_module.twin = _Static("inst", [])  # type: ignore[attr-defined]
    refs = ["_roster_plugin_under_test:instance", "_roster_plugin_under_test:twin"]
    with pytest.raises(RosterSourceError, match="share the name 'inst'") as info:
        load_roster_sources(refs)
    assert all(ref in str(info.value) for ref in refs)


def test_loader_error_is_a_value_error() -> None:
    assert issubclass(RosterSourceError, ValueError)


# --- conformance helper ---------------------------------------------------------------


def test_a_conforming_source_has_no_problems() -> None:
    src = _Static("ok", [RosterRow("A", "h", "c", "running"), RosterRow("B", None, None, None)])
    assert check_roster_source(src) == []


def test_conformance_reports_a_missing_member() -> None:
    problems = check_roster_source(object())  # type: ignore[arg-type]
    assert problems and "RosterSource" in problems[0]


@pytest.mark.parametrize("name", ["", 7])
def test_conformance_reports_a_bad_name(name: object) -> None:
    src = _Static("x", [])
    src.name = name  # type: ignore[assignment]
    assert any("name" in p for p in check_roster_source(src))


def test_conformance_reports_a_raising_read() -> None:
    problems = check_roster_source(_Raising("r", OSError("down")))
    assert any("OSError" in p for p in problems)


def test_conformance_reports_wrong_return_types() -> None:
    assert any("list" in p for p in check_roster_source(_Returns("r", "IOC1")))
    assert any("RosterRow" in p for p in check_roster_source(_Returns("r", [("IOC1",)])))


def test_conformance_reports_duplicate_names() -> None:
    src = _Static("d", [RosterRow("A", None, None, None), RosterRow("A", None, None, None)])
    assert any("A" in p and "more than once" in p for p in check_roster_source(src))


def test_conformance_reports_an_overrun_deadline() -> None:
    class Slow:
        name = "slow"

        def read(self, deadline: float) -> list[RosterRow]:
            time.sleep(deadline + 0.05)
            return []

    assert any("deadline" in p for p in check_roster_source(Slow(), deadline=0.05))


# --- the documented example source ----------------------------------------------------


def _example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("roster_static", EXAMPLE_SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "roster_static", module)
    spec.loader.exec_module(module)
    return module


def test_example_source_conforms(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _example_module(monkeypatch)
    source = load_roster_source("roster_static:roster")
    assert check_roster_source(source) == []
    rows = source.read(1.0)
    assert {row.expected for row in rows} == {"running", "stopped", None}
    assert any(row.group for row in rows)
    assert module.StaticRoster is type(source)


def test_example_source_reads_any_yaml_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _example_module(monkeypatch)
    path = tmp_path / "servers.yml"
    path.write_text("servers:\n  - {name: X1, host: h, class: soft, expected: running}\n")
    source = module.StaticRoster(path)
    assert source.name == "servers"
    assert source.read(1.0) == [RosterRow("X1", "h", "soft", "running")]


def test_example_source_with_a_missing_file_fails_and_merge_reports_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _example_module(monkeypatch)
    source = module.StaticRoster(tmp_path / "absent.yml", name="static")
    merged = merge_rosters([source], deadline=1.0)
    assert merged.sources == (SourceOutcome("static", "unavailable", "FileNotFoundError"),)
