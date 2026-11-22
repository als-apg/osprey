"""A command states how long one act may take; its status states what it reads when idle.

A command channel states ``timeout_s``; the readback it pairs with, its status,
states ``idle``. The build holds the two together and stops with
``command-invalid`` where one is missing, misplaced or of the wrong type, and
where a limits record states a re-read the wait for the status replaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.pairing import NOT_FOLLOWED, pair_follows
from osprey.facility.schema import core
from osprey.simulation.engines import pyat
from tests.facility._pyaml_trees import built_document, measured_tree
from tests.facility._synthetic_trees import plain_tree, write_tree

#: A command on the plain tree's quadrupole, and the status it pairs with.
COMMAND = "Q1:ON"
STATUS = "Q1:STATE"
#: A second command on the same device that shares the status.
RESET = "Q1:RESET"
SWITCH = {"value_type": "bool", "options": ["OFF", "ON"]}


def _tree(**changes: dict[str, Any]) -> dict[str, Any]:
    """The plain tree with a bounded command and its idle status; ``changes`` edit by address."""
    tree = plain_tree()
    channels = [
        {
            "id": COMMAND,
            "role": "setpoint",
            "signal": "power_on_command",
            "timeout_s": 5.0,
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
        {
            "id": STATUS,
            "signal": "power_on_status",
            "idle": "OFF",
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
    ]
    for channel in channels:
        channel.update(changes.get(channel["id"], {}))
        for key in [key for key, value in channel.items() if value is None]:
            del channel[key]
    tree["records/channels.yaml"] += channels
    return tree


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    return build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")


def _channels(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(channel["id"]): channel for channel in document["channels"]}


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, tree)
    return stopped.value.format_message()


def test_a_command_and_its_status_state_bound_and_idle(tmp_path: Path) -> None:
    channels = _channels(_build(tmp_path, _tree()))

    assert channels[COMMAND]["command"] is True
    assert channels[COMMAND]["pair"] == STATUS
    assert channels[COMMAND]["timeout_s"] == 5.0
    assert channels[STATUS]["idle"] == "OFF"


def test_a_bound_on_a_channel_that_is_no_command_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"][0]["timeout_s"] = 1.0

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:SP — `timeout_s` on a channel that is no "
        "command; fix: remove `timeout_s`; only a command states how long one act may take"
    )


@pytest.mark.parametrize("bound", [0, -1.0, float("inf"), float("nan")])
def test_a_bound_that_is_not_positive_stops_the_build(tmp_path: Path, bound: float) -> None:
    assert _stop(tmp_path, _tree(**{COMMAND: {"timeout_s": bound}})) == (
        f"facility: command-invalid: channel Q1:ON — `timeout_s` {bound!r} is not a finite "
        "number > 0; fix: state `timeout_s` as the longest one act may take, in seconds"
    )


def test_a_bound_without_an_idle_status_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"idle": None}})) == (
        "facility: command-invalid: channel Q1:ON — `timeout_s` on a command whose status "
        "Q1:STATE states no `idle`; fix: state `idle` on Q1:STATE"
    )


def test_a_bound_on_a_command_that_is_its_own_pair_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{COMMAND: {"pair": COMMAND}, STATUS: {"idle": None}})) == (
        "facility: command-invalid: channel Q1:ON — `timeout_s` on a command that is its own "
        "pair; fix: name the readback that reads the act as `pair`, and state `idle` on it"
    )


def test_an_idle_status_whose_command_states_no_bound_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{COMMAND: {"timeout_s": None}})) == (
        "facility: command-invalid: channel Q1:ON — a command whose status Q1:STATE states "
        "`idle` states no `timeout_s`; fix: state `timeout_s` on Q1:ON"
    )


def test_idle_on_a_setpoint_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"][0]["idle"] = 0.0

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:SP — `idle` on a setpoint channel; fix: remove "
        "`idle`; only a readback a command pairs with states it"
    )


@pytest.mark.parametrize(
    ("status", "idle", "problem"),
    [
        pytest.param(SWITCH, "MAYBE", "'MAYBE', not a label", id="bool"),
        pytest.param(
            {"value_type": "enum", "options": ["IDLE", "MOVING"]}, 0, "0, not a label", id="enum"
        ),
        pytest.param({"value_type": "int", "options": None}, 0.5, "0.5, not an integer", id="int"),
        pytest.param(
            {"value_type": "float", "options": None}, "IDLE", "'IDLE', not a number", id="float"
        ),
        pytest.param({"value_type": "string", "options": None}, 0, "0, not text", id="string"),
    ],
)
def test_idle_of_the_wrong_type_stops_the_build(
    tmp_path: Path, status: dict[str, Any], idle: Any, problem: str
) -> None:
    tree = _tree(**{STATUS: {**status, "idle": idle}})
    value_type = status["value_type"]

    assert _stop(tmp_path, tree) == (
        f"facility: command-invalid: channel Q1:STATE — `idle` of {value_type} channel Q1:STATE "
        f"is {problem}; fix: type `idle` by the channel: a label on a bool or enum channel, an "
        "integer on an int, a number on a float, text on a string; a waveform channel is never a "
        "status"
    )


def test_a_string_status_states_its_idle_text(tmp_path: Path) -> None:
    tree = _tree(**{STATUS: {"value_type": "string", "options": None, "idle": "IDLE"}})

    assert _channels(_build(tmp_path, tree))[STATUS]["idle"] == "IDLE"


#: The status as a waveform channel.
WAVEFORM = {"value_type": "waveform", "options": None, "shape": [3]}


@pytest.mark.parametrize(
    "idle", [pytest.param([0, 0, 0], id="trace"), pytest.param("IDLE", id="text")]
)
def test_idle_on_a_waveform_status_stops_the_build_by_name(tmp_path: Path, idle: Any) -> None:
    tree = _tree(**{COMMAND: {"timeout_s": None}, STATUS: {**WAVEFORM, "idle": idle}})

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:STATE — `idle` on waveform channel Q1:STATE; a "
        "waveform channel is never a status; fix: type `idle` by the channel: a label on a bool "
        "or enum channel, an integer on an int, a number on a float, text on a string; a "
        "waveform channel is never a status"
    )


def test_a_bound_on_a_command_whose_status_is_a_waveform_stops_the_build(tmp_path: Path) -> None:
    tree = _tree(**{STATUS: {**WAVEFORM, "idle": None}})

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:ON — `timeout_s` on a command whose status "
        "Q1:STATE is a waveform channel; fix: pair Q1:ON with a status that is no waveform, or "
        "remove `timeout_s`"
    )


#: Where the build's sentence on how ``idle`` is typed is also stated.
_REPO = Path(__file__).parents[2]
_CORE_YAML = _REPO / "src" / "osprey" / "facility" / "schema" / "core.yaml"
_DOCS_PAGE = _REPO / "docs" / "source" / "how-to" / "describe-your-facility.rst"
_FIX = "type `idle` by the channel: "


def _idle_row(page: str) -> str:
    """The text of the ``idle`` row of the channel slot table, up to the next row."""
    row = page.split("   * - ``idle``", 1)[1]
    return row.split("   * - ", 1)[0]


def _spaced(text: str) -> str:
    return " ".join(text.split())


def test_idle_is_typed_by_one_sentence(tmp_path: Path) -> None:
    tree = _tree(**{COMMAND: {"timeout_s": None}, STATUS: {**WAVEFORM, "idle": 0}})
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, tree)
    remedy = stopped.value.remedy
    assert remedy.startswith(_FIX)
    stated = "typed by the channel: " + remedy.removeprefix(_FIX)

    schema = yaml.safe_load(_CORE_YAML.read_text(encoding="utf-8"))
    assert stated in _spaced(schema["classes"]["Channel"]["attributes"]["idle"]["description"])
    assert stated in _spaced(core.Channel.model_fields["idle"].description or "")
    assert stated in _spaced(_idle_row(_DOCS_PAGE.read_text(encoding="utf-8")))


def test_idle_on_a_status_no_command_pairs_with_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"][1]["idle"] = 0.0

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:RB — `idle` on a readback no command pairs "
        "with; fix: remove `idle`, or name Q1:RB as a command's `pair`"
    )


def test_two_commands_may_share_one_idle_status(tmp_path: Path) -> None:
    tree = _tree(**{COMMAND: {"pair": STATUS}})
    tree["records/channels.yaml"].append(
        {
            "id": RESET,
            "role": "setpoint",
            "signal": "reset_command",
            "pair": STATUS,
            "timeout_s": 1.0,
            "on": {"device": "SR/Q1"},
            **SWITCH,
        }
    )

    channels = _channels(_build(tmp_path, tree))

    assert channels[RESET]["command"] is True
    assert channels[RESET]["pair"] == channels[COMMAND]["pair"] == STATUS


def _marking(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    """The pyat engine marks *address* ``command``, and describes it as it would otherwise."""
    describe = pyat.describe

    def marked(record: Any) -> dict[str, Any]:
        description = dict(describe(record))
        if pyat.field(record, "address") == address:
            description["marks"] = ["command"]
        return description

    monkeypatch.setattr(pyat, "describe", marked)


def test_an_engine_marked_command_takes_a_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marking(monkeypatch, "QD:SP")
    tree = measured_tree()
    (qd,) = [c for c in tree["records/channels.yaml"] if c["id"] == "QD:SP"]
    qd.update(pair="QD:BUSY", timeout_s=2.0)
    tree["records/channels.yaml"].append(
        {"id": "QD:BUSY", "value_type": "int", "idle": 0, "on": {"device": "SR/QD"}}
    )

    channels = _channels(built_document(tmp_path, tree)[0])

    assert channels["QD:SP"]["command"] is True
    assert channels["QD:SP"]["timeout_s"] == 2.0
    assert channels["QD:BUSY"]["idle"] == 0


def test_confirm_true_on_a_command_with_a_bound_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["limits.yaml"] = {"records": [{"address": COMMAND, "writable": True, "confirm": True}]}

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: limit Q1:ON — `confirm: true` on a command that states "
        "`timeout_s`; fix: remove `confirm`; the wait for Q1:STATE confirms the write"
    )


def test_a_status_that_states_idle_is_not_followed(tmp_path: Path) -> None:
    channels = _channels(_build(tmp_path, _tree()))

    follows = {(f.setpoint, f.readback): (f.follow, f.why) for f in pair_follows(channels)}

    assert follows[(COMMAND, STATUS)] == (NOT_FOLLOWED, "the readback states `idle`")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param({}, id="same-type"),
        pytest.param({"value_type": "int", "options": None}, id="by-index"),
    ],
)
def test_follows_naming_a_setpoint_on_an_idle_status_stops_the_build(
    tmp_path: Path, command: dict[str, Any]
) -> None:
    tree = _tree(**{COMMAND: command, STATUS: {"simulation": {"follows": COMMAND}}})

    assert _stop(tmp_path, tree) == (
        "facility: command-invalid: channel Q1:STATE — `follows` names Q1:ON on a status that "
        "states `idle`; a followed status holds each write to Q1:ON and never returns to `idle` "
        "by itself; fix: remove `follows`; a status that states `idle` is never followed"
    )


def test_follows_none_on_an_idle_status_builds(tmp_path: Path) -> None:
    channels = _channels(_build(tmp_path, _tree(**{STATUS: {"simulation": {"follows": "none"}}})))

    assert channels[STATUS]["idle"] == "OFF"


#: The measured tree's QD setpoint, the further address its wiring record states, and its status.
SETPOINT, FURTHER, BUSY = "QD:SP", "RACK3:UNIT2:SP", "QD:BUSY"


def _further_tree(monkeypatch: pytest.MonkeyPatch, **further: Any) -> dict[str, Any]:
    """The measured tree with ``SETPOINT`` a bounded command also served as ``FURTHER``."""
    from tests.facility.test_wiring_also import _tree as also_tree

    _marking(monkeypatch, SETPOINT)
    tree = also_tree()
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == SETPOINT:
            channel.update(pair=BUSY, timeout_s=2.0)
        elif channel["id"] == FURTHER:
            channel.update(further)
    tree["records/channels.yaml"].append(
        {"id": BUSY, "value_type": "int", "idle": 0, "on": {"device": "SR/QD"}}
    )
    return tree


def _further_stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)
    return stopped.value.format_message()


@pytest.mark.parametrize(("key", "value"), [("timeout_s", 2.0), ("idle", 0)])
def test_a_bound_or_idle_on_a_further_address_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    tree = _further_tree(monkeypatch, **{key: value})

    assert _further_stop(tmp_path, tree) == (
        f"facility: command-invalid: channel {FURTHER} — `{key}` on {FURTHER}, which wiring "
        f"record SR/QD:SP states under `also`; fix: state `{key}` on {SETPOINT} only; it holds "
        "under each of its addresses"
    )


def test_confirm_true_on_a_further_address_of_a_bounded_command_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _further_tree(monkeypatch)
    limit = {"writable": True, "min_value": -1.0, "max_value": 1.0, "confirm": True}
    tree["limits.yaml"] = {"records": [{"address": FURTHER, **limit}]}

    assert _further_stop(tmp_path, tree) == (
        f"facility: command-invalid: limit {FURTHER} — `confirm: true` on a further address of a "
        f"command that states `timeout_s`; fix: remove `confirm`; the wait for {BUSY} confirms "
        "the write"
    )
