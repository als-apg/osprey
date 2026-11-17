"""A scenario's events, noise and logbook, and a seed's gate and linear inputs, as view records.

An archiver event is one record per ``shape`` that holds the rules between its
keys, so a view stating an event that breaks one is refused when it is read,
in the rule's own words. A noise record is the one term it states. The authored
shorthands of a ``gate`` input and a ``linear`` input are written back as
stated and read through one accessor. A calendar ``time`` written ``HH:MM`` is
written back ``HH:MM:SS``.
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import TypeAdapter, ValidationError

from osprey.facility.scenarios import check_scenario_events
from osprey.facility.views.simulator import withdraw_scenarios
from osprey_connectors.simulation import envelope, series, texture
from osprey_connectors.simulation.records import (
    AbsoluteNoise,
    ArchiverEvent,
    EventRuleError,
    Noise,
    RelativeNoise,
    ScenarioRecord,
    ScenariosDocument,
    SeedRecord,
    TopoffRecord,
)
from osprey_connectors.simulation.view import SimulatorView, ViewSchemaError
from tests._simulator_view import write_scenarios_view

EVENT: TypeAdapter[ArchiverEvent] = TypeAdapter(ArchiverEvent)
NOISE: TypeAdapter[Noise] = TypeAdapter(Noise)
UTC_ZONE = ZoneInfo("UTC")
ANCHOR_S = 1_762_300_000.0


def _refusal(raw: dict) -> EventRuleError:
    with pytest.raises(ValidationError) as raised:
        EVENT.validate_python(raw)
    cause = raised.value.errors()[0]["ctx"]["error"]
    assert isinstance(cause, EventRuleError)
    return cause


# --- the HH:MM -> HH:MM:SS write-back ------------------------------------------------


def test_an_hh_mm_calendar_time_is_written_back_with_seconds() -> None:
    stated = {
        "name": "s",
        "archiver": [
            {
                "channel": "A",
                "events": [
                    {"shape": "step", "at_when": {"days_ago": 1, "time": "08:00"}, "to": 1.0},
                    {"shape": "step", "at_time": "09:15", "to": 2.0},
                ],
            }
        ],
        "logbook": [
            {
                "entry_id": "E-1",
                "when": {"days_ago": 1, "time": "08:00"},
                "author": "a",
                "title": "t",
                "text": "x",
            }
        ],
    }

    written = ScenarioRecord.model_validate(stated).model_dump(mode="json", exclude_unset=True)

    first, second = written["archiver"][0]["events"]
    assert first["at_when"] == {"days_ago": 1, "time": "08:00:00"}
    assert second["at_time"] == "09:15:00"
    assert written["logbook"][0]["when"] == {"days_ago": 1, "time": "08:00:00"}
    # Every other key is written back as it was stated.
    assert {**written, "archiver": stated["archiver"], "logbook": stated["logbook"]} == stated


def test_an_hh_mm_ss_calendar_time_is_written_back_as_stated() -> None:
    stated = {
        "name": "s",
        "archiver": [
            {
                "channel": "A",
                "events": [
                    {"shape": "step", "at_when": {"days_ago": 1, "time": "08:00:30"}, "to": 1},
                    {"shape": "spike", "at_time": "23:59:59", "amplitude": 1, "width": 2.5},
                ],
            }
        ],
    }

    assert ScenarioRecord.model_validate(stated).model_dump(mode="json", exclude_unset=True) == (
        stated
    )


def test_withdrawing_a_scenario_rewrites_an_hh_mm_time_with_seconds(tmp_path: Path) -> None:
    kept = {
        "archiver": [{"channel": "A", "events": [{"shape": "step", "at_time": "08:00", "to": 1}]}]
    }
    path = write_scenarios_view(tmp_path, {"kept": kept, "gone": {}})
    assert '"at_time": "08:00"' in path.read_text(encoding="utf-8")

    withdraw_scenarios(path.parent, ["gone"])

    (scenario,) = json.loads((path.parent / ScenariosDocument.FILE).read_text(encoding="utf-8"))[
        "scenarios"
    ]
    assert scenario["archiver"][0]["events"] == [{"shape": "step", "at_time": "08:00:00", "to": 1}]


# --- events ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "detail", "remedy"),
    [
        (
            {"shape": "spike", "at_offset": -60, "amplitude": 1.0},
            "lacks `width`",
            "give the `spike` event `width`",
        ),
        (
            {"shape": "step", "at": 0.5, "at_offset": -60, "to": 1.0},
            "states 2 position keys (`at`, `at_offset`)",
            "state exactly one of `at`, `at_offset`, `at_time`, `at_when`",
        ),
        (
            {"shape": "step", "at": 0.5, "until": 0.25, "to": 1.0},
            "ends at `until` 0.25, not after `at` 0.5",
            "write `until` later than `at`",
        ),
        (
            {"shape": "step", "at_offset": -60, "until_when": {"days_ago": 0, "time": "01:00"}},
            "ends a `step` placed by `at_offset` with `until_when`",
            "end a step placed by `at_offset` with `until_offset` only",
        ),
        (
            {"shape": "loss", "at_offset": None, "amount": 1.0},
            "has `at_offset` None, not a number",
            "write `at_offset` as a number",
        ),
        (
            {"shape": "burst", "at_offset": -60, "amplitude": True, "decay_s": 5},
            "has `amplitude` True, not a number",
            "write `amplitude` as a number",
        ),
    ],
    ids=["missing-key", "two-positions", "end-not-after", "end-unpaired", "null", "bool"],
)
def test_an_event_breaking_a_rule_is_refused_in_its_words(
    raw: dict, detail: str, remedy: str
) -> None:
    refusal = _refusal(raw)

    assert (refusal.detail, refusal.remedy) == (detail, remedy)
    assert str(refusal) == detail


def test_a_ramp_ends_by_the_key_its_position_pairs_with() -> None:
    refusal = _refusal({"shape": "ramp", "at": 0.1, "until": 0.5, "until_offset": 60, "to": 1.0})

    assert refusal.detail == "ends a `ramp` placed by `at` with `until`, `until_offset`"
    assert refusal.remedy == "end a ramp placed by `at` with `until` only"


def test_an_event_stating_another_shapes_key_stops_the_build() -> None:
    document = {
        "channels": [{"id": "SR:X", "value_type": "float"}],
        "scenarios": [
            {
                "name": "warm",
                "archiver": [
                    {
                        "channel": "SR:X",
                        "events": [{"shape": "step", "at": 0.5, "to": 1, "width": 2}],
                    }
                ],
            }
        ],
    }

    (error,) = check_scenario_events(document)

    assert error.format_message() == (
        "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 of "
        "channel SR:X states `width`, which a `step` event does not carry; fix: drop `width`"
    )


def test_a_view_event_breaking_a_rule_refuses_the_view(tmp_path: Path) -> None:
    broken = {"archiver": [{"channel": "A", "events": [{"shape": "spike", "at_offset": -60}]}]}
    path = write_scenarios_view(tmp_path, {"broken": broken})

    with pytest.raises(ViewSchemaError, match="rebuild") as raised:
        SimulatorView.open(path.parent).scenarios()

    assert "lacks `amplitude`, `width`" in str(raised.value)


@pytest.mark.parametrize(
    ("raw", "span"),
    [
        ({"shape": "spike", "at_offset": -600, "amplitude": 1, "width": 10}, (-40.0, 40.0)),
        ({"shape": "burst", "at_offset": -600, "amplitude": 1, "decay_s": 5}, (0.0, 50.0)),
        (
            {
                "shape": "burst",
                "at_offset": -600,
                "amplitude": 1,
                "decay_s": 5,
                "tail_share": 0.1,
                "tail_decay_s": 25,
            },
            (0.0, 250.0),
        ),
        ({"shape": "step", "at_offset": -600, "until_offset": -300, "to": 1}, (0.0, 300.0)),
        ({"shape": "step", "at_offset": -600, "to": 1}, (0.0, 600.0)),
        ({"shape": "ramp", "at_offset": -600, "until_offset": -300, "to": 1}, (0.0, 600.0)),
        ({"shape": "loss", "at_offset": -600, "amount": 5}, (0.0, 600.0)),
    ],
    ids=["spike", "burst", "burst-tail", "bounded-step", "step", "ramp", "loss-without-topoff"],
)
def test_one_occurrence_spans_what_its_shape_changes(raw: dict, span: tuple) -> None:
    at = ANCHOR_S - 600.0
    start, end = series.occurrence_span(
        EVENT.validate_python(raw), at, anchor_s=ANCHOR_S, tz=UTC_ZONE, topoff=None
    )

    assert (start - at, end - at) == pytest.approx(span)


def test_a_loss_spans_until_top_off_has_refilled_it() -> None:
    loss = EVENT.validate_python({"shape": "loss", "at_offset": -600, "amount": 18.0})
    topoff = TopoffRecord(ceiling=500.0, lifetime_h=7.0, period_s=31.0)
    at = ANCHOR_S - 600.0

    _start, end = series.occurrence_span(loss, at, anchor_s=ANCHOR_S, tz=UTC_ZONE, topoff=topoff)

    # One period to the first refill, then 24 shots of 0.75 mA at one a second.
    assert end - at == pytest.approx(31.0 + 24.0)


def test_a_daily_event_occurs_on_every_date_of_the_span() -> None:
    daily = EVENT.validate_python({"shape": "step", "at_time": "12:00", "to": 1.0})
    day = 86_400.0
    start = 1_767_225_600.0  # 2026-01-01T00:00:00Z

    assert series.occurrences(
        daily, start_s=start, end_s=start + 2.5 * day, anchor_s=ANCHOR_S, tz=UTC_ZONE
    ) == [start + 0.5 * day, start + 1.5 * day, start + 2.5 * day]


def test_a_window_fraction_event_occurs_at_no_instant() -> None:
    fraction = EVENT.validate_python({"shape": "step", "at": 0.5, "to": 1.0})

    with pytest.raises(ValueError, match="window fraction"):
        series.occurrences(fraction, start_s=0.0, end_s=1.0, anchor_s=ANCHOR_S, tz=UTC_ZONE)


# --- noise ----------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [{}, {"absolute": 0.1, "relative": 0.01}, {"sigma": 0.1}])
def test_a_noise_record_states_exactly_one_term(raw: dict) -> None:
    with pytest.raises(ValidationError):
        NOISE.validate_python(raw)


def test_the_texture_and_the_envelope_read_one_term() -> None:
    relative = NOISE.validate_python({"relative": 0.01})
    absolute = NOISE.validate_python({"absolute": 0.5})

    assert isinstance(relative, RelativeNoise)
    assert isinstance(absolute, AbsoluteNoise)
    assert texture._noise_term(relative) == ("relative", 0.01)
    assert texture._noise_term(absolute) == ("absolute", 0.5)
    assert envelope._noise_sigma(relative, 200.0) == pytest.approx(2.0)
    assert envelope._noise_sigma(absolute, 200.0) == pytest.approx(0.5)


# --- gate and linear shorthands -------------------------------------------------------


def test_a_seeds_linear_inputs_read_as_coefficients_and_write_back_as_stated() -> None:
    stated = {"linear": {"B": {"coefficient": 0.5}, "A": 2}}
    seed = SeedRecord.model_validate(stated)

    assert seed.linear_terms() == (("A", 2.0), ("B", 0.5))
    assert SeedRecord().linear_terms() == ()
    assert seed.model_dump(mode="json", exclude_unset=True) == stated


def test_a_gates_inputs_read_as_labels_and_write_back_as_stated() -> None:
    stated = {"gate": {"inputs": {"Y": {"blocking_label": "OUT"}, "X": True}, "gated": 0.1}}
    seed = SeedRecord.model_validate(stated)

    assert seed.gate is not None
    assert seed.gate.blocking_labels() == (("X", True), ("Y", "OUT"))
    assert seed.model_dump(mode="json", exclude_unset=True) == stated


@pytest.mark.parametrize("label", [["ON"], {"label": "ON"}, None])
def test_a_gate_input_that_is_no_label_is_refused(label: object) -> None:
    with pytest.raises(ValidationError):
        SeedRecord.model_validate({"gate": {"inputs": {"X": label}, "gated": 0.1}})
