"""A command is a channel the rendered facility definition marks ``command``.

The build marks a channel ``command: true`` when it is a setpoint whose
vocabulary signal is a command word or a model's engine marks its wiring record
``command``, and stops when an engine describes a command-signal setpoint
without the mark or marks a readback. Whatever fires or judges a command reads the mark alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines import pyat
from tests.facility._pyaml_trees import built_document, measured_tree

#: An unwired setpoint whose signal is a command word.
FIRE = "SR:SCREEN:IN"


def _tree() -> dict[str, Any]:
    tree = measured_tree()
    tree["records/channels.yaml"] += [
        {"id": FIRE, "role": "setpoint", "signal": "screen_in_command", "on": {"place": "SR"}},
        {"id": "SR:SCREEN:ECHO", "signal": "screen_in_command", "on": {"place": "SR"}},
    ]
    return tree


def _channels(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(channel["id"]): channel for channel in document["channels"]}


def _marking(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    """The pyat engine marks *address* ``command``, and describes it as it would otherwise."""
    describe = pyat.describe

    def marked(record: Any) -> dict[str, Any]:
        description = dict(describe(record))
        if pyat.field(record, "address") == address:
            description["marks"] = ["command"]
        return description

    monkeypatch.setattr(pyat, "describe", marked)


def test_a_setpoint_whose_signal_is_a_command_word_is_marked(tmp_path: Path) -> None:
    channels = _channels(built_document(tmp_path, _tree())[0])

    assert channels[FIRE]["command"] is True
    assert "command" in channels[FIRE]["provenance"]["defaults"]
    assert "command" not in channels["SR:SCREEN:ECHO"]
    assert not any("command" in channels[a] for a in ("QF:SP", "QD:SP", "BPM1:X"))


def test_a_record_its_engine_marks_is_a_command_whatever_its_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marking(monkeypatch, "QD:SP")

    channels = _channels(built_document(tmp_path, _tree())[0])

    assert channels["QD:SP"]["command"] is True
    assert "command" not in channels["QF:SP"]


def test_a_command_signal_its_engine_describes_unmarked_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    (qd,) = [c for c in tree["records/channels.yaml"] if c["id"] == "QD:SP"]
    qd["signal"] = "pulsing_command"

    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)

    assert stopped.value.format_message() == (
        "facility: engine-invalid: model SR — QD:SP's signal pulsing_command makes it a "
        "command; the engine describes it without the command mark; fix: make the engine's "
        "describe() mark QD:SP command, or give QD:SP a signal that is no command word"
    )


def test_a_source_states_no_command_mark(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"][0]["command"] = True

    with pytest.raises(FacilityBuildError, match="`command` is computed by the build"):
        built_document(tmp_path, tree)


def test_a_command_mark_on_a_readback_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marking(monkeypatch, "BPM1:X")

    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, _tree())

    assert stopped.value.format_message() == (
        "facility: engine-invalid: model SR — the engine marks BPM1:X command; its direction "
        "is read; fix: make the engine's describe() mark write records alone command"
    )
