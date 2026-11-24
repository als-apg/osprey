"""The channel-finder delegation judge, exercised on traces the product makes.

``tests/e2e/channel_finder_judge.py`` decides whether a live channel-finder run
passes: no tool erred save the hierarchical finder's level-order refusal, that
refusal is followed by the repaired request, and the final answer names every
storage-ring horizontal BPM address and no other facility address. Each tool
result below is the text the real tool produces over a small real index, so the
verdicts are pinned without an agent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import PropertyMock, patch

import pytest
from fastmcp.exceptions import ToolError

from osprey.facility.views.channel_finder import hierarchical_document
from osprey.mcp_server.channel_finder_hierarchical.server_context import (
    initialize_cf_hier_context,
    reset_cf_hier_context,
)
from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
    UnselectedLevelError,
)
from osprey_facility.records import FacilityRecords
from tests.e2e.channel_finder_judge import (
    BUILD_CHANNELS,
    GET_OPTIONS,
    designed_level_refusal,
    expected_horizontal_bpms,
    judge_channel_finder_run,
)
from tests.mcp_server.conftest import get_tool_fn

_DATABASE = (
    "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database"
)

DELEGATION = "toolu_channel_finder"

#: Two storage-ring sectors of one BPM each, a second ring, and the BPM signals
#: a wrong answer would name instead: the other plane and the golden reference.
FACILITY: dict[str, Any] = {
    "places": [
        {"id": "SR", "level": "machine", "names": ["storage ring"]},
        {"id": "SR/S1", "level": "sector"},
        {"id": "SR/S2", "level": "sector"},
        {"id": "BR", "level": "machine"},
    ],
    "devices": [
        {"id": "SR/BPM1", "class": "BeamPositionMonitor", "place": "SR/S1"},
        {"id": "SR/BPM2", "class": "BeamPositionMonitor", "place": "SR/S2"},
        {"id": "BR/BPM1", "class": "BeamPositionMonitor", "place": "BR"},
    ],
    "channels": [
        {"id": "SR:BPM1:X", "on": {"device": "SR/BPM1"}, "signal": "position_x_readback"},
        {"id": "SR:BPM1:Y", "on": {"device": "SR/BPM1"}, "signal": "position_y_readback"},
        {
            "id": "SR:BPM1:GOLDEN:X",
            "on": {"device": "SR/BPM1"},
            "signal": "position_x_golden_readback",
        },
        {"id": "SR:BPM2:X", "on": {"device": "SR/BPM2"}, "signal": "position_x_readback"},
        {"id": "BR:BPM1:X", "on": {"device": "BR/BPM1"}, "signal": "position_x_readback"},
    ],
}

MACHINE = {"machine": "SR"}
SECTORS = {**MACHINE, "sector": ["S1", "S2"]}
CLASS = {**SECTORS, "class": "BeamPositionMonitor"}
DEVICES = {**CLASS, "device": ["SR/BPM1", "SR/BPM2"]}
LEAF = {**DEVICES, "leaf": "position_x_readback"}


@pytest.fixture
def records() -> FacilityRecords:
    return FacilityRecords(FACILITY)


@pytest.fixture
def database(tmp_path, monkeypatch, records) -> HierarchicalChannelDatabase:
    """The index a build writes for :data:`FACILITY`, served by the hierarchical tools."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    path = tmp_path / "hierarchical.json"
    path.write_text(json.dumps(hierarchical_document(records)), encoding="utf-8")
    db = HierarchicalChannelDatabase(str(path))
    reset_cf_hier_context()
    initialize_cf_hier_context()
    with patch(_DATABASE, new_callable=PropertyMock, return_value=db):
        yield db
    reset_cf_hier_context()


def _trace(name: str, **fields: Any) -> SimpleNamespace:
    base = {
        "input": {},
        "tool_use_id": None,
        "parent_tool_use_id": DELEGATION,
        "result": None,
        "is_error": False,
    }
    return SimpleNamespace(name=name, **{**base, **fields})


def _delegation() -> SimpleNamespace:
    return _trace(
        "Agent",
        input={"subagent_type": "channel-finder"},
        tool_use_id=DELEGATION,
        parent_tool_use_id=None,
    )


def _call(name: str, **arguments: Any) -> SimpleNamespace:
    """The trace of one channel-finder call, with the result the real tool answers."""
    if name == GET_OPTIONS:
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options as tool,
        )
    else:
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels as tool,
        )
    try:
        answer = get_tool_fn(tool)(**arguments)
    except ToolError as exc:
        return _trace(name, input=arguments, result=str(exc), is_error=True)
    return _trace(name, input=arguments, result=answer)


def _options(level: str, selections: dict[str, Any]) -> SimpleNamespace:
    return _call(GET_OPTIONS, level=level, selections=selections)


def _build(selections: dict[str, Any] = LEAF) -> SimpleNamespace:
    return _call(BUILD_CHANNELS, selections=selections)


def _clean() -> list[SimpleNamespace]:
    return [
        _options("machine", {}),
        _options("sector", MACHINE),
        _options("class", SECTORS),
        _options("device", CLASS),
        _options("leaf", DEVICES),
        _build(),
    ]


def _refused() -> SimpleNamespace:
    trace = _options("class", MACHINE)
    assert trace.is_error, trace.result
    return trace


def _run(traces: list[SimpleNamespace], answer: str | None) -> SimpleNamespace:
    final = None if answer is None else SimpleNamespace(result=answer, is_error=False)
    return SimpleNamespace(tool_traces=[_delegation(), *traces], result=final)


def _answer(addresses: set[str] | frozenset[str]) -> str:
    return "\n".join(f"- `{address}`" for address in sorted(addresses))


def _judge(run, database, records) -> list[str]:
    return judge_channel_finder_run(run, database, records, expected_horizontal_bpms(records))


def test_the_expected_set_is_the_storage_ring_horizontal_bpm_readbacks(records):
    assert expected_horizontal_bpms(records) == {"SR:BPM1:X", "SR:BPM2:X"}


def test_a_clean_run_with_a_correct_answer_passes(database, records):
    expected = expected_horizontal_bpms(records)

    assert _judge(_run(_clean(), _answer(expected)), database, records) == []


def test_a_level_order_refusal_then_a_repaired_request_and_a_correct_answer_passes(
    database, records
):
    expected = expected_horizontal_bpms(records)
    refused = _refused()
    traces = [
        refused,
        _options("sector", MACHINE),
        _options("class", SECTORS),
        _options("device", CLASS),
        _options("leaf", DEVICES),
        _build(),
    ]

    assert designed_level_refusal(database)(refused)
    assert _judge(_run(traces, _answer(expected)), database, records) == []


def test_any_other_tool_error_fails(database, records):
    expected = expected_horizontal_bpms(records)
    no_node = _options("sector", {"machine": "NOPE"})
    bad_build = _build({**LEAF, "nope": "x"})
    assert no_node.is_error and bad_build.is_error

    for erring in (no_node, bad_build):
        failures = _judge(_run([erring, *_clean()], _answer(expected)), database, records)

        assert failures, erring.result
        assert any(erring.name in failure for failure in failures), failures
        assert not designed_level_refusal(database)(erring)


def test_a_wrong_final_answer_fails(database, records):
    expected = expected_horizontal_bpms(records)
    one_short = set(sorted(expected)[1:])
    with_golden = {*expected, "SR:BPM1:GOLDEN:X"}

    for answer in (_answer(one_short), _answer(with_golden), None, ""):
        failures = _judge(_run(_clean(), answer), database, records)

        assert failures, answer
    assert any(
        "SR:BPM1:GOLDEN:X" in failure
        for failure in _judge(_run(_clean(), _answer(with_golden)), database, records)
    )
    assert any(
        sorted(expected)[0] in failure
        for failure in _judge(_run(_clean(), _answer(one_short)), database, records)
    )


def test_an_address_inside_a_longer_token_is_not_named(database, records):
    expected = expected_horizontal_bpms(records)
    answer = " ".join(f"{address}:SUFFIX" for address in sorted(expected))

    assert _judge(_run(_clean(), answer), database, records)


def test_a_refusal_with_no_repaired_request_fails(database, records):
    expected = expected_horizontal_bpms(records)
    unrepaired = [_refused(), _options("sector", MACHINE), _build()]
    unbuilt = [_refused(), _options("sector", MACHINE), _options("class", SECTORS)]

    for traces in (unrepaired, unbuilt):
        assert _judge(_run(traces, _answer(expected)), database, records)


def test_a_refusal_the_product_does_not_make_fails(database, records):
    expected = expected_horizontal_bpms(records)
    refused = _refused()
    accepted_input = _trace(
        GET_OPTIONS,
        input={"level": "class", "selections": SECTORS},
        result=refused.result,
        is_error=True,
    )
    other_tool = _trace(BUILD_CHANNELS, input=refused.input, result=refused.result, is_error=True)

    for forged in (accepted_input, other_tool):
        assert not designed_level_refusal(database)(forged)
        assert _judge(_run([forged, *_clean()], _answer(expected)), database, records)


def test_the_refusal_is_the_products_exception(database):
    with pytest.raises(UnselectedLevelError) as refused:
        database.get_options_at_level("class", MACHINE)

    assert isinstance(refused.value, ValueError)
    assert (refused.value.level, refused.value.requested) == ("sector", "class")
    with pytest.raises(ValueError) as no_node:
        database.get_options_at_level("sector", {"machine": "NOPE"})
    assert not isinstance(no_node.value, UnselectedLevelError)
