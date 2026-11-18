"""The simulator directory answers from a build's simulator view, never from a service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.directory import DirectoryDetail, DirectoryQuery
from osprey_connectors.directory._simulator import SimulatorDirectory
from osprey_connectors.service_answers import BackendSpec, NotListed, Source, Unavailable
from osprey_connectors.simulation.records import (
    AddressesDocument,
    ServedModelsDocument,
    VariablesDocument,
)
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    IOC_HEALTH_SIGNALS,
    SERVED_SIGNAL_PROPERTY,
    STATUS_SIGNAL,
    UPTIME,
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
    _channel(SOLO, role="none", value_type="bool", description="Water interlock OK"),
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
        "schema": AddressesDocument.SCHEMA,
        "channels": sorted(entry["address"] for entry in listed),
        "status": ["T:SIM:BARE:STATUS", "T:SIM:SR:STATUS"],
    }
    if ioc_health is not None:
        addresses["ioc_health"] = ioc_health
    documents = {
        AddressesDocument.FILE: addresses,
        ServedModelsDocument.FILE: {
            "schema": ServedModelsDocument.SCHEMA,
            "models": ["BARE", "SR", "texture"],
        },
        VariablesDocument.FILE: {
            "schema": VariablesDocument.SCHEMA,
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
        kind="directory", type="simulator_view", target="va", settings={}, rehearsal=False
    )


@pytest.fixture
def directory(tmp_path: Path) -> SimulatorDirectory:
    _write_view(tmp_path, ioc_health=_health("BARE") + _health("SR") + _health("texture"))
    return SimulatorDirectory(_spec(), project_dir=tmp_path)


async def _names(directory: SimulatorDirectory, **fields: Any) -> list[str]:
    page = await directory.search(DirectoryQuery(**fields), timeout_s=DEADLINE)
    return [row.address for row in page.rows]


async def test_a_word_search_finds_the_description_and_no_name_carries_the_word(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(DirectoryQuery(words="cooling water"), timeout_s=DEADLINE)

    assert page.outcome == "found"
    assert [row.address for row in page.rows] == [COOLING]
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
        suffix: (await directory.describe(f"T:SIM:SR:{suffix}", timeout_s=DEADLINE)).entry  # type: ignore[union-attr]
        for suffix in ("HEARTBEAT", "UPTIME", "ENGINEER", "STATUS")
    }

    assert rows["HEARTBEAT"].description == "heartbeat counter of IOC sim-SR"
    assert rows["UPTIME"].description == "seconds since IOC sim-SR started"
    assert rows["ENGINEER"].description == "responsible engineer of IOC sim-SR"
    assert rows["STATUS"].description == "simulation status of IOC sim-SR"
    assert {suffix: row.channel_type for suffix, row in rows.items()} == {
        "HEARTBEAT": "longin",
        "UPTIME": "ai",
        "ENGINEER": "stringin",
        "STATUS": "stringin",
    }
    assert {row.server for row in rows.values()} == {"sim-SR"}


async def test_each_status_and_health_row_carries_its_signal(
    directory: SimulatorDirectory,
) -> None:
    for signal in (STATUS_SIGNAL, HEARTBEAT, UPTIME, ENGINEER):
        record = await directory.describe(f"T:SIM:SR:{signal}", timeout_s=DEADLINE)
        assert isinstance(record, DirectoryDetail)
        assert record.properties == {SERVED_SIGNAL_PROPERTY: signal}
    channel = await directory.describe(SETPOINT, timeout_s=DEADLINE)
    assert isinstance(channel, DirectoryDetail)
    assert SERVED_SIGNAL_PROPERTY not in channel.properties


async def test_a_search_by_the_signal_mark_finds_only_those_rows(
    directory: SimulatorDirectory,
) -> None:
    assert await _names(
        directory, server=("sim-SR",), properties={SERVED_SIGNAL_PROPERTY: ENGINEER}
    ) == ["T:SIM:SR:ENGINEER"]
    assert await _names(directory, properties={SERVED_SIGNAL_PROPERTY: HEARTBEAT}) == [
        "T:SIM:BARE:HEARTBEAT",
        "T:SIM:SR:HEARTBEAT",
        "T:SIM:texture:HEARTBEAT",
    ]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"address_pattern": "T:SR:QF1:*"}, [READBACK, SETPOINT]),
        ({"address_pattern": "t:sr:qf1:*"}, [READBACK, SETPOINT]),
        ({"address_pattern": "*:RB"}, [READBACK]),
        ({"address_pattern": "T:SR"}, [READBACK, SETPOINT]),
        ({"address_pattern": "QF?"}, [READBACK, SETPOINT]),
        ({"address_pattern": "SR QF1 RB"}, [READBACK]),
        ({"address_pattern": f"={READBACK}"}, [READBACK]),
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
    assert await _names(directory, address_pattern="=T:SR") == []
    assert await _names(directory, address_pattern=f"={READBACK.lower()}") == []


async def test_regex_characters_in_a_pattern_are_literal(directory: SimulatorDirectory) -> None:
    assert await _names(directory, address_pattern="T.SR") == []


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"server": "sim-SR"}, [READBACK, SETPOINT]),
        ({"channel_type": ("ao", "bi")}, [ORPHAN, SOLO, SETPOINT]),
        ({"properties": {"unit": "A"}}, [READBACK, SETPOINT]),
        ({"properties": {"unit": "V"}}, []),
        ({"status": "Inactive"}, []),
        ({"status": "Active", "server": "sim-SR", "words": "setting"}, [SETPOINT]),
    ],
)
async def test_filters_narrow_the_result(
    directory: SimulatorDirectory, fields: dict[str, Any], expected: list[str]
) -> None:
    query = {"address_pattern": "T:*:*:*", **fields}
    names = await _names(directory, **query)

    assert [name for name in names if not name.startswith("T:SIM:")] == expected


async def test_a_channel_row_carries_the_view_as_a_directory_states_it(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(
        DirectoryQuery(address_pattern=f"={SETPOINT}"), timeout_s=DEADLINE
    )

    assert page.rows[0].model_dump() == {
        "address": SETPOINT,
        "status": "Active",
        "description": "Quadrupole current setting.",
        "channel_type": "ao",
        "server": "sim-SR",
        "host": "virtual-accelerator",
        "announced_at": None,
        "recorded_stamp": None,
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
        ("none", "bool", "bi"),
        ("none", "enum", "mbbi"),
        ("none", "string", "stringin"),
        ("readback", "waveform", "waveform"),
    ],
)
async def test_the_record_type_comes_from_role_and_value_type(
    tmp_path: Path, role: str, value_type: str, record_type: str
) -> None:
    # A waveform channel states its shape, as the build writes it.
    shape = {"shape": [4]} if value_type == "waveform" else {}
    _write_view(tmp_path, channels=[_channel("T:X", role=role, value_type=value_type, **shape)])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    record = await directory.describe("T:X", timeout_s=DEADLINE)

    assert isinstance(record, DirectoryDetail)
    assert record.entry.channel_type == record_type


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

    record = await directory.describe("T:X", timeout_s=DEADLINE)

    assert isinstance(record, DirectoryDetail)
    assert record.entry.description == first


@pytest.mark.parametrize("description", [None, "", "   "])
async def test_a_channel_with_no_description_has_none(
    tmp_path: Path, description: str | None
) -> None:
    _write_view(tmp_path, channels=[_channel("T:X", description=description)])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    record = await directory.describe("T:X", timeout_s=DEADLINE)

    assert isinstance(record, DirectoryDetail)
    assert record.entry.description is None
    assert await _names(directory, words="anything") == []


async def test_an_exact_miss_is_not_listed(directory: SimulatorDirectory) -> None:
    assert await directory.describe("T:NOT:THERE", timeout_s=DEADLINE) == NotListed(
        address="T:NOT:THERE"
    )
    assert await directory.describe(READBACK.lower(), timeout_s=DEADLINE) == NotListed(
        address=READBACK.lower()
    )


@pytest.mark.parametrize(
    "fields",
    [{"words": "nothing-here"}, {"address_pattern": "NOPE*"}, {"address_pattern": "=T:NOT:THERE"}],
)
async def test_a_search_with_no_rows_is_no_match_never_not_listed(
    directory: SimulatorDirectory, fields: dict[str, Any]
) -> None:
    page = await directory.search(DirectoryQuery(**fields), timeout_s=DEADLINE)

    assert page.outcome == "no_match"
    assert page.rows == ()
    assert page.total == 0
    assert await directory.count(DirectoryQuery(**fields), timeout_s=DEADLINE) == 0


async def test_a_served_no_deck_model_lists_status_and_health_rows_and_no_channels(
    directory: SimulatorDirectory,
) -> None:
    names = await _names(directory, server="sim-BARE")

    assert names == sorted([*_health("BARE"), "T:SIM:BARE:STATUS"])


async def test_a_channel_of_an_unserved_model_is_the_textures_as_the_composite_serves_it(
    directory: SimulatorDirectory,
) -> None:
    record = await directory.describe(ORPHAN, timeout_s=DEADLINE)

    assert isinstance(record, DirectoryDetail)
    assert record.entry.server == "sim-texture"
    assert await _names(directory, server="sim-LN") == []


async def test_a_record_carries_no_ports_and_no_engineer_property(
    directory: SimulatorDirectory,
) -> None:
    record = await directory.describe(SETPOINT, timeout_s=DEADLINE)
    health = await directory.describe("T:SIM:SR:ENGINEER", timeout_s=DEADLINE)

    assert isinstance(record, DirectoryDetail)
    assert record.properties == {"role": "setpoint", "unit": "A", "valueType": "float"}
    assert isinstance(health, DirectoryDetail)
    assert health.properties == {SERVED_SIGNAL_PROPERTY: ENGINEER}
    names = await directory.property_names(timeout_s=DEADLINE)
    assert isinstance(names, list)
    assert not {"caPort", "pvaPort", "Engineer"} & set(names)


async def test_a_view_from_a_build_without_ioc_health_still_answers(tmp_path: Path) -> None:
    _write_view(tmp_path)
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    assert await _names(directory, server="sim-BARE") == ["T:SIM:BARE:STATUS"]
    assert await directory.describe("T:SIM:SR:HEARTBEAT", timeout_s=DEADLINE) == NotListed(
        address="T:SIM:SR:HEARTBEAT"
    )


async def test_a_view_listing_an_unknown_health_signal_is_unavailable_and_says_rebuild(
    tmp_path: Path,
) -> None:
    _write_view(tmp_path, ioc_health=["T:SIM:SR:CPU_LOAD"])
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    page = await directory.search(DirectoryQuery(), timeout_s=DEADLINE)

    assert isinstance(page, Unavailable)
    assert "rebuild" in page.reason


async def test_the_request_is_the_search_in_words_and_never_a_url(
    directory: SimulatorDirectory,
) -> None:
    page = await directory.search(DirectoryQuery(words="cooling water"), timeout_s=DEADLINE)

    assert page.request == "simulator data · name or description matches cooling water"
    assert page.searched == "name or description matches cooling water"


async def test_the_request_names_every_filter(directory: SimulatorDirectory) -> None:
    query = DirectoryQuery(
        address_pattern="T:SR:*",
        status="Active",
        server=("sim-SR", "sim-BARE"),
        channel_type="ao",
        properties={"unit": "A"},
    )

    page = await directory.search(query, timeout_s=DEADLINE)

    assert page.searched == (
        "name matches T:SR:* · status Active · IOC sim-SR or sim-BARE · record type ao · unit A"
    )
    assert page.request == f"simulator data · {page.searched}"


async def test_an_empty_search_lists_every_pv(directory: SimulatorDirectory) -> None:
    page = await directory.search(DirectoryQuery(page_size=100), timeout_s=DEADLINE)

    assert page.searched == "every PV"
    assert page.outcome == "found"
    # Five channels, two status addresses, three health addresses for each of three IOCs.
    assert page.total == 5 + 2 + 9
    assert [row.address for row in page.rows] == sorted(row.address for row in page.rows)
    assert await directory.count(DirectoryQuery(), timeout_s=DEADLINE) == 16


async def test_pages_slice_the_whole_result_and_say_when_more_follows(
    directory: SimulatorDirectory,
) -> None:
    every = await _names(directory, page_size=100)

    first = await directory.search(DirectoryQuery(page=0, page_size=10), timeout_s=DEADLINE)
    last = await directory.search(DirectoryQuery(page=1, page_size=10), timeout_s=DEADLINE)
    beyond = await directory.search(DirectoryQuery(page=9, page_size=10), timeout_s=DEADLINE)

    assert [row.address for row in (*first.rows, *last.rows)] == every
    assert (first.total, first.truncated) == (16, True)
    assert (last.total, last.truncated) == (16, False)
    assert (beyond.outcome, beyond.rows, beyond.total) == ("found", (), 16)


async def test_the_source_is_the_specs(directory: SimulatorDirectory) -> None:
    page = await directory.search(DirectoryQuery(), timeout_s=DEADLINE)

    assert page.source == Source(
        kind="directory", target="va", service="simulator_view", rehearsal=False
    )


async def test_properties_are_listed_by_name_and_value(directory: SimulatorDirectory) -> None:
    assert await directory.property_names(timeout_s=DEADLINE) == [
        "role",
        "signal",
        "unit",
        "valueType",
    ]
    assert await directory.property_values("signal", timeout_s=DEADLINE) == [
        "ENGINEER",
        "HEARTBEAT",
        "STATUS",
        "UPTIME",
    ]
    assert await directory.property_values("unit", timeout_s=DEADLINE) == ["A"]
    assert await directory.property_values("role", prefix="set", timeout_s=DEADLINE) == ["setpoint"]
    assert await directory.property_values("role", prefix="nope", timeout_s=DEADLINE) == []
    assert await directory.property_values("nosuch", timeout_s=DEADLINE) == []


async def test_the_probe_states_its_evidence(tmp_path: Path) -> None:
    _write_view(tmp_path)
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)

    answer = await directory.probe(timeout_s=DEADLINE)

    assert isinstance(answer, str)
    assert "7 PVs" in answer


def test_building_one_reads_nothing(tmp_path: Path) -> None:
    SimulatorDirectory(_spec(), project_dir=tmp_path / "does-not-exist")


def test_a_seam_it_does_not_take_is_ignored(tmp_path: Path) -> None:
    SimulatorDirectory(_spec(), project_dir=tmp_path, transport=object())


async def _every_answer(directory: SimulatorDirectory) -> list[Any]:
    return [
        await directory.search(DirectoryQuery(words="x"), timeout_s=DEADLINE),
        await directory.count(DirectoryQuery(), timeout_s=DEADLINE),
        await directory.describe("T:X", timeout_s=DEADLINE),
        await directory.property_names(timeout_s=DEADLINE),
        await directory.property_values("unit", timeout_s=DEADLINE),
        await directory.probe(timeout_s=DEADLINE),
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
    (tmp_path / VIEW_RELPATH / AddressesDocument.FILE).write_text(
        json.dumps({"schema": "osprey.facility.addresses/0"}), encoding="utf-8"
    )

    answers = await _every_answer(SimulatorDirectory(_spec(), project_dir=tmp_path))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert "rebuild with osprey build" in answer.reason


@pytest.mark.parametrize("broken", ["{ not json", '{"schema": "osprey.facility.simulator/2"}'])
async def test_a_broken_variables_file_is_unavailable(tmp_path: Path, broken: str) -> None:
    _write_view(tmp_path)
    (tmp_path / VIEW_RELPATH / VariablesDocument.FILE).write_text(broken, encoding="utf-8")

    answers = await _every_answer(SimulatorDirectory(_spec(), project_dir=tmp_path))

    for answer in answers:
        assert isinstance(answer, Unavailable)
        assert answer.reason


async def test_a_view_built_after_a_miss_is_found(tmp_path: Path) -> None:
    directory = SimulatorDirectory(_spec(), project_dir=tmp_path)
    assert isinstance(await directory.probe(timeout_s=DEADLINE), Unavailable)

    _write_view(tmp_path)

    assert isinstance(await directory.probe(timeout_s=DEADLINE), str)
