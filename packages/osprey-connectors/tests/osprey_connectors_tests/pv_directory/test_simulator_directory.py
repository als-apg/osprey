"""The simulator directory answers from a build's simulator view, never from a service."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.pv_directory._simulator import SimulatorDirectory
from osprey_connectors.pv_directory.contract import (
    SERVED_SIGNAL_PROPERTY,
    BackendSpec,
    DirectoryRecord,
    Outcome,
    SearchQuery,
    Source,
    Unavailable,
)
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    ENGINEER,
    HEARTBEAT,
    IOC_HEALTH_SIGNALS,
    SERVED_MODELS_FILE,
    SERVED_MODELS_SCHEMA,
    STATUS_SIGNAL,
    UPTIME,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    VIEW_RELPATH,
)

DEADLINE = 6.0

SETPOINT = "T:SR:QF1:SP"
READBACK = "T:SR:QF1:RB"
COOLING = "T:RF:CW:FLOW"
SOLO = "T:RF:CW:ENABLE"
ORPHAN = "T:LN:GUN:SP"


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    return {
        "address": address,
        "role": "readback",
        "pair": None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": "texture",
        "on": None,
        **fields,
    }


def _health(model: str) -> list[str]:
    return [f"T:SIM:{model}:{signal}" for signal in IOC_HEALTH_SIGNALS]


_CHANNELS = [
    _channel(
        SETPOINT,
        role="setpoint",
        pair=READBACK,
        unit="A",
        owner="SR",
        description="Quadrupole current setting. Trimmed by the optics group.",
    ),
    _channel(READBACK, owner="SR", unit="A", description="Quadrupole current readback."),
    _channel(COOLING, description="Cavity cooling water flow."),
    _channel(SOLO, role="monitor", value_type="bool", description="Water interlock OK"),
    _channel(ORPHAN, role="setpoint", owner="LN", description="Gun voltage setting."),
]


def _write_view(
    project: Path,
    *,
    ioc_health: list[str] | None = None,
    channels: list[dict[str, Any]] | None = None,
) -> None:
    """Write a view serving physics models ``SR`` and ``BARE`` (no deck) and the texture.

    ``LN`` is in the facility file and not served.
    """
    listed = _CHANNELS if channels is None else channels
    addresses: dict[str, Any] = {
        "schema": ADDRESSES_SCHEMA,
        "channels": sorted(entry["address"] for entry in listed),
        "status": ["T:SIM:BARE:STATUS", "T:SIM:SR:STATUS"],
    }
    if ioc_health is not None:
        addresses["ioc_health"] = ioc_health
    documents = {
        ADDRESSES_FILE: addresses,
        SERVED_MODELS_FILE: {
            "schema": SERVED_MODELS_SCHEMA,
            "models": ["BARE", "SR", "texture"],
        },
        VARIABLES_FILE: {
            "schema": VARIABLES_SCHEMA,
            "code": "T",
            "models": [
                {"name": "BARE", "engine": "pyat", "served": True},
                {"name": "LN", "engine": "pyat", "served": False},
                {"name": "SR", "engine": "pyat", "served": True},
                {"name": "texture", "engine": "texture", "served": True},
            ],
            "channels": listed,
        },
    }
    view_dir = project / VIEW_RELPATH
    view_dir.mkdir(parents=True)
    for name, document in documents.items():
        (view_dir / name).write_text(json.dumps(document), encoding="utf-8")


def _spec() -> BackendSpec:
    return BackendSpec(
        kind="pv_directory", type="simulator_view", target="va", settings={}, rehearsal=False
    )


@pytest.fixture
def directory(tmp_path: Path) -> SimulatorDirectory:
    _write_view(tmp_path, ioc_health=_health("BARE") + _health("SR") + _health("texture"))
    return SimulatorDirectory(_spec(), project_dir=tmp_path)


async def _names(directory: SimulatorDirectory, **fields: Any) -> list[str]:
    page = await directory.search(SearchQuery(**fields), DEADLINE)
    return [row.name for row in page.rows]


async def test_a_word_search_finds_the_description_and_no_name_carries_the_word(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(SearchQuery(words="cooling water"), DEADLINE)

    assert page.outcome is Outcome.FOUND
    assert [row.name for row in page.rows] == [COOLING]
    assert page.total == 1


async def test_words_match_in_any_order_and_any_case(directory: SimulatorDirectory) -> None:
    assert await _names(directory, words="WATER cooling") == [COOLING]


async def test_descriptions_answer_first_and_names_only_when_none_matched(
    tmp_path: Path,
) -> None:
    _write_view(
        tmp_path,
        channels=[
            _channel("T:FLOW:A", description="Cavity temperature."),
            _channel("T:OTHER:B", description="Cooling flow."),
            _channel("T:CAVITY:C", description="Nothing of note."),
        ],
    )
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    assert await _names(directory, words="flow") == ["T:OTHER:B"]
    assert await _names(directory, words="cavity") == ["T:FLOW:A"]
    assert await _names(directory, words="OTHER") == ["T:OTHER:B"]


@pytest.mark.parametrize(
    ("words", "address"),
    [
        ("heartbeat", "T:SIM:BARE:HEARTBEAT"),
        ("T:SIM:SR:HEARTBEAT", "T:SIM:SR:HEARTBEAT"),
        ("heartbeat counter sim-SR", "T:SIM:SR:HEARTBEAT"),
        ("seconds since sim-texture started", "T:SIM:texture:UPTIME"),
        ("T:SIM:texture:ENGINEER", "T:SIM:texture:ENGINEER"),
    ],
)
async def test_the_ioc_health_pvs_are_found_by_name_and_by_description(
    directory: SimulatorDirectory, words: str, address: str
) -> None:
    assert address in await _names(directory, words=words)


async def test_each_health_and_status_pv_says_what_it_is_in_plain_words(
    directory: SimulatorDirectory,
) -> None:
    rows = {
        suffix: (await directory.describe(f"T:SIM:SR:{suffix}", DEADLINE)).row  # type: ignore[union-attr]
        for suffix in ("HEARTBEAT", "UPTIME", "ENGINEER", "STATUS")
    }

    assert rows["HEARTBEAT"].description == "heartbeat counter of IOC sim-SR"
    assert rows["UPTIME"].description == "seconds since IOC sim-SR started"
    assert rows["ENGINEER"].description == "responsible engineer of IOC sim-SR"
    assert rows["STATUS"].description == "simulation status of IOC sim-SR"
    assert {suffix: row.record_type for suffix, row in rows.items()} == {
        "HEARTBEAT": "longin",
        "UPTIME": "ai",
        "ENGINEER": "stringin",
        "STATUS": "stringin",
    }
    assert {row.ioc for row in rows.values()} == {"sim-SR"}


async def test_each_status_and_health_row_carries_its_signal(
    directory: SimulatorDirectory,
) -> None:
    for signal in (STATUS_SIGNAL, HEARTBEAT, UPTIME, ENGINEER):
        record = await directory.describe(f"T:SIM:SR:{signal}", DEADLINE)
        assert isinstance(record, DirectoryRecord)
        assert record.properties == {SERVED_SIGNAL_PROPERTY: signal}
    channel = await directory.describe(SETPOINT, DEADLINE)
    assert isinstance(channel, DirectoryRecord)
    assert SERVED_SIGNAL_PROPERTY not in channel.properties


async def test_a_search_by_the_signal_mark_finds_only_those_rows(
    directory: SimulatorDirectory,
) -> None:
    assert await _names(
        directory, ioc=("sim-SR",), properties={SERVED_SIGNAL_PROPERTY: ENGINEER}
    ) == ["T:SIM:SR:ENGINEER"]
    assert await _names(directory, properties={SERVED_SIGNAL_PROPERTY: HEARTBEAT}) == [
        "T:SIM:BARE:HEARTBEAT",
        "T:SIM:SR:HEARTBEAT",
        "T:SIM:texture:HEARTBEAT",
    ]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"name": "T:SR:QF1:*"}, [READBACK, SETPOINT]),
        ({"name": "t:sr:qf1:*"}, [READBACK, SETPOINT]),
        ({"name": "*:RB"}, [READBACK]),
        ({"name": "T:SR"}, [READBACK, SETPOINT]),
        ({"name": "QF?"}, [READBACK, SETPOINT]),
        ({"name": "SR QF1 RB"}, [READBACK]),
        ({"name": f"={READBACK}"}, [READBACK]),
        ({"words": "T:SR:*:RB"}, [READBACK]),
        ({"words": f"={READBACK}"}, [READBACK]),
        ({"words": "current T:SR:*:SP"}, [SETPOINT]),
    ],
)
async def test_globs_and_exact_names_follow_the_channelfinder_convention(
    directory: SimulatorDirectory, fields: dict[str, Any], expected: list[str]
) -> None:
    assert await _names(directory, **fields) == expected


async def test_a_leading_equals_adds_no_wildcard(directory: SimulatorDirectory) -> None:
    assert await _names(directory, name="=T:SR") == []
    assert await _names(directory, name=f"={READBACK.lower()}") == []


async def test_regex_characters_in_a_pattern_are_literal(directory: SimulatorDirectory) -> None:
    assert await _names(directory, name="T.SR") == []


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"ioc": "sim-SR"}, [READBACK, SETPOINT]),
        ({"record_type": ("ao", "bi")}, [ORPHAN, SOLO, SETPOINT]),
        ({"properties": {"unit": "A"}}, [READBACK, SETPOINT]),
        ({"properties": {"unit": "V"}}, []),
        ({"status": "Inactive"}, []),
        ({"status": "Active", "ioc": "sim-SR", "words": "setting"}, [SETPOINT]),
    ],
)
async def test_filters_narrow_the_result(
    directory: SimulatorDirectory, fields: dict[str, Any], expected: list[str]
) -> None:
    query = {"name": "T:*:*:*", **fields}
    names = await _names(directory, **query)

    assert [name for name in names if not name.startswith("T:SIM:")] == expected


async def test_origin_is_left_to_the_service(directory: SimulatorDirectory) -> None:
    query = SearchQuery(words="cooling", origin="curated")

    page = await directory.search(query, DEADLINE)

    assert [row.origin for row in page.rows] == ["unknown"]
    assert await directory.count(query, DEADLINE) == 1


async def test_a_channel_row_carries_the_view_as_a_directory_states_it(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(SearchQuery(name=f"={SETPOINT}"), DEADLINE)

    assert asdict(page.rows[0]) == {
        "name": SETPOINT,
        "origin": "unknown",
        "status": "Active",
        "description": "Quadrupole current setting.",
        "record_type": "ao",
        "ioc": "sim-SR",
        "host": "virtual-accelerator",
        "last_announced": None,
    }


@pytest.mark.parametrize(
    ("role", "value_type", "record_type"),
    [
        ("setpoint", "float", "ao"),
        ("setpoint", "int", "longout"),
        ("setpoint", "bool", "bo"),
        ("setpoint", "enum", "mbbo"),
        ("setpoint", "string", "stringout"),
        ("setpoint", "waveform", "waveform"),
        ("readback", "float", "ai"),
        ("readback", "int", "longin"),
        ("monitor", "bool", "bi"),
        ("monitor", "enum", "mbbi"),
        ("output", "string", "stringin"),
        ("output", "waveform", "waveform"),
    ],
)
async def test_the_record_type_comes_from_role_and_value_type(
    tmp_path: Path, role: str, value_type: str, record_type: str
) -> None:
    _write_view(tmp_path, channels=[_channel("T:X", role=role, value_type=value_type)])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    record = await directory.describe("T:X", DEADLINE)

    assert isinstance(record, DirectoryRecord)
    assert record.row.record_type == record_type


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("One sentence.", "One sentence."),
        ("No full stop", "No full stop"),
        ("First part. Second part. Third.", "First part."),
        ("  Wrapped\n  over lines.  Then more.", "Wrapped over lines."),
        ("v1.5 of the setting", "v1.5 of the setting"),
    ],
)
async def test_the_description_is_the_first_sentence(tmp_path: Path, text: str, first: str) -> None:
    _write_view(tmp_path, channels=[_channel("T:X", description=text)])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    record = await directory.describe("T:X", DEADLINE)

    assert isinstance(record, DirectoryRecord)
    assert record.row.description == first


@pytest.mark.parametrize("description", [None, "", "   "])
async def test_a_channel_with_no_description_has_none(
    tmp_path: Path, description: str | None
) -> None:
    _write_view(tmp_path, channels=[_channel("T:X", description=description)])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    record = await directory.describe("T:X", DEADLINE)

    assert isinstance(record, DirectoryRecord)
    assert record.row.description is None
    assert await _names(directory, words="anything") == []


async def test_an_exact_miss_is_not_listed(directory: SimulatorDirectory) -> None:
    assert await directory.describe("T:NOT:THERE", DEADLINE) is Outcome.NOT_LISTED
    assert await directory.describe(READBACK.lower(), DEADLINE) is Outcome.NOT_LISTED


@pytest.mark.parametrize(
    "fields", [{"words": "nothing-here"}, {"name": "NOPE*"}, {"name": "=T:NOT:THERE"}]
)
async def test_a_search_with_no_rows_is_no_match_never_not_listed(
    directory: SimulatorDirectory, fields: dict[str, Any]
) -> None:
    page = await directory.search(SearchQuery(**fields), DEADLINE)

    assert page.outcome is Outcome.NO_MATCH
    assert page.rows == []
    assert page.total == 0
    assert await directory.count(SearchQuery(**fields), DEADLINE) == 0


async def test_a_served_no_deck_model_lists_status_and_health_rows_and_no_channels(
    directory: SimulatorDirectory,
) -> None:
    names = await _names(directory, ioc="sim-BARE")

    assert names == sorted([*_health("BARE"), "T:SIM:BARE:STATUS"])


async def test_a_channel_of_an_unserved_model_is_the_textures_as_the_composite_serves_it(
    directory: SimulatorDirectory,
) -> None:
    record = await directory.describe(ORPHAN, DEADLINE)

    assert isinstance(record, DirectoryRecord)
    assert record.row.ioc == "sim-texture"
    assert await _names(directory, ioc="sim-LN") == []


async def test_a_record_carries_no_ports_and_no_engineer_property(
    directory: SimulatorDirectory,
) -> None:
    record = await directory.describe(SETPOINT, DEADLINE)
    health = await directory.describe("T:SIM:SR:ENGINEER", DEADLINE)

    assert isinstance(record, DirectoryRecord)
    assert record.properties == {"role": "setpoint", "unit": "A", "valueType": "float"}
    assert isinstance(health, DirectoryRecord)
    assert health.properties == {SERVED_SIGNAL_PROPERTY: ENGINEER}
    names = await directory.property_names(DEADLINE)
    assert isinstance(names, list)
    assert not {"caPort", "pvaPort", "Engineer"} & set(names)


async def test_a_view_from_a_build_without_ioc_health_still_answers(tmp_path: Path) -> None:
    _write_view(tmp_path)
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    assert await _names(directory, ioc="sim-BARE") == ["T:SIM:BARE:STATUS"]
    assert await directory.describe("T:SIM:SR:HEARTBEAT", DEADLINE) is Outcome.NOT_LISTED


async def test_a_view_listing_an_unknown_health_signal_is_unavailable_and_says_rebuild(
    tmp_path: Path,
) -> None:
    _write_view(tmp_path, ioc_health=["T:SIM:SR:CPU_LOAD"])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    page = await directory.search(SearchQuery(), DEADLINE)

    assert page.outcome is Outcome.UNAVAILABLE
    assert "rebuild" in (page.reason or "")


async def test_the_request_is_the_search_in_words_and_never_a_url(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(SearchQuery(words="cooling water"), DEADLINE)

    assert page.request == "simulator data · name or description matches cooling water"
    assert page.searched == "name or description matches cooling water"


async def test_the_request_names_every_filter(directory: SimulatorDirectory) -> None:
    query = SearchQuery(
        name="T:SR:*",
        status="Active",
        ioc=("sim-SR", "sim-BARE"),
        record_type="ao",
        properties={"unit": "A"},
    )

    page = await directory.search(query, DEADLINE)

    assert page.searched == (
        "name matches T:SR:* · status Active · IOC sim-SR or sim-BARE · record type ao · unit A"
    )
    assert page.request == f"simulator data · {page.searched}"


async def test_an_empty_search_lists_every_pv(directory: SimulatorDirectory) -> None:
    page = await directory.search(SearchQuery(page_size=100), DEADLINE)

    assert page.searched == "every PV"
    assert page.outcome is Outcome.FOUND
    # Five channels, two status addresses, three health addresses for each of three IOCs.
    assert page.total == 5 + 2 + 9
    assert [row.name for row in page.rows] == sorted(row.name for row in page.rows)
    assert await directory.count(SearchQuery(), DEADLINE) == 16


async def test_pages_slice_the_whole_result_and_say_when_more_follows(
    directory: SimulatorDirectory,
) -> None:
    every = await _names(directory, page_size=100)

    first = await directory.search(SearchQuery(page=0, page_size=10), DEADLINE)
    last = await directory.search(SearchQuery(page=1, page_size=10), DEADLINE)
    beyond = await directory.search(SearchQuery(page=9, page_size=10), DEADLINE)

    assert [row.name for row in (*first.rows, *last.rows)] == every
    assert (first.total, first.truncated) == (16, True)
    assert (last.total, last.truncated) == (16, False)
    assert (beyond.outcome, beyond.rows, beyond.total) == (Outcome.FOUND, [], 16)


async def test_the_source_is_the_specs(directory: SimulatorDirectory) -> None:
    page = await directory.search(SearchQuery(), DEADLINE)

    assert page.source == Source(
        target="va", service="simulator_view", label="All PVs", rehearsal=False
    )


async def test_properties_are_listed_by_name_and_value(directory: SimulatorDirectory) -> None:
    assert await directory.property_names(DEADLINE) == ["role", "signal", "unit", "valueType"]
    assert await directory.property_values("signal", None, DEADLINE) == [
        "ENGINEER",
        "HEARTBEAT",
        "STATUS",
        "UPTIME",
    ]
    assert await directory.property_values("unit", None, DEADLINE) == ["A"]
    assert await directory.property_values("role", "set", DEADLINE) == ["setpoint"]
    assert await directory.property_values("role", "nope", DEADLINE) == []
    assert await directory.property_values("nosuch", None, DEADLINE) == []


async def test_the_probe_states_its_evidence(tmp_path: Path) -> None:
    _write_view(tmp_path)
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    answer = await directory.probe(DEADLINE)

    assert isinstance(answer, str)
    assert "7 PVs" in answer


def test_building_one_reads_nothing(tmp_path: Path) -> None:
    SimulatorDirectory(_spec(), project_dir=tmp_path / "does-not-exist")


def test_a_seam_it_does_not_take_is_ignored(tmp_path: Path) -> None:
    SimulatorDirectory(_spec(), project_dir=tmp_path, transport=object())


async def _every_answer(directory: SimulatorDirectory) -> list[Any]:
    page = await directory.search(SearchQuery(words="x"), DEADLINE)
    assert page.outcome is Outcome.UNAVAILABLE
    return [
        Unavailable(page.reason or ""),
        await directory.count(SearchQuery(), DEADLINE),
        await directory.describe("T:X", DEADLINE),
        await directory.property_names(DEADLINE),
        await directory.property_values("unit", None, DEADLINE),
        await directory.probe(DEADLINE),
    ]


async def test_a_missing_view_is_unavailable_with_the_way_out(tmp_path: Path) -> None:
    answers = await _every_answer(SimulatorDirectory(_spec(), project_dir=tmp_path))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert "run osprey build" in answer.reason


async def test_no_project_directory_is_unavailable() -> None:
    answers = await _every_answer(SimulatorDirectory(_spec()))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert "project directory" in answer.reason


async def test_a_view_of_another_schema_is_unavailable_and_says_rebuild(tmp_path: Path) -> None:
    _write_view(tmp_path)
    (tmp_path / VIEW_RELPATH / ADDRESSES_FILE).write_text(
        json.dumps({"schema": "osprey.facility.addresses/0"}), encoding="utf-8"
    )

    answers = await _every_answer(SimulatorDirectory(_spec(), project_dir=tmp_path))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert "rebuild with osprey build" in answer.reason


@pytest.mark.parametrize("broken", ["{ not json", '{"schema": "osprey.facility.simulator/2"}'])
async def test_a_broken_variables_file_is_unavailable(tmp_path: Path, broken: str) -> None:
    _write_view(tmp_path)
    (tmp_path / VIEW_RELPATH / VARIABLES_FILE).write_text(broken, encoding="utf-8")

    answers = await _every_answer(SimulatorDirectory(_spec(), project_dir=tmp_path))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert answer.reason


async def test_a_view_built_after_a_miss_is_found(tmp_path: Path) -> None:
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)
    assert isinstance(await directory.probe(DEADLINE), Unavailable)

    _write_view(tmp_path)

    assert isinstance(await directory.probe(DEADLINE), str)
