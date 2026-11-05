"""The demo facility's operation mode, an enum, end to end on the write side.

``SR:OPS:MODE:RB`` is a four-option enum. One world holds it three ways, and
all three must agree on the option index:

* the **virtual accelerator** serves it as an enum variable whose labels are
  the record's four options, in the record's order — the order is what gives
  each label its Channel Access index;
* the **archive seed** stores each sample as the option index of the label the
  composite holds, an integer, moved by a scenario's ``step`` events;
* the **recorder** samples the served channel over Channel Access and stores
  the index the read answers with, never the label.

The view is the example facility's own, built as a deployment builds it, so the
record's options reach every half through the real build path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from lume.variables import EnumVariable

from osprey.services.archiver_recorder import Recorder, resolve_channel_addresses
from osprey.services.archiver_recorder.recorder import channel_access_reader
from osprey_connectors.simulation.archive import ArchiveComposite, build
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import VARIABLES_FILE, SimulatorView
from osprey_connectors_tests._epics_fakes import (
    DBR_ENUM,
    FakeChannel,
    ca_reading,
    install_fake_aioca,
)
from tests._simulator_view import write_scenarios_view
from tests.services.test_archiver_recorder import _FakeWriter, _settings, _write_config

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"

#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}

MODE = "SR:OPS:MODE:RB"
MODES = ["User operation", "Injection", "Machine studies", "Off"]

#: The instant the scenario set is applied at; its steps sit before it.
ANCHOR_S = 1_760_000_000.0
INJECTION_AT_S = -3600.0
OFF_AT_S = -600.0

#: A scenario that walks the mode through two later options, by label.
MODE_CHANGE = {
    "description": "The ring goes to injection, then off.",
    "archiver": [
        {
            "channel": MODE,
            "events": [
                {"shape": "step", "at_offset": INJECTION_AT_S, "to": "Injection"},
                {"shape": "step", "at_offset": OFF_AT_S, "to": "Off"},
            ],
        }
    ],
}


@pytest.fixture(scope="module")
def view_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The example facility's simulator view, as a build writes it, plus one mode scenario."""
    from osprey.facility.build import build_facility
    from osprey.facility.records import FacilityRecords
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("project")
    view = root / "data" / "simulator"
    records = FacilityRecords(build_facility(TEMPLATE_FACILITY, project_name="example"))
    write_simulator_view(
        view,
        ViewInputs(
            records=records,
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, records),
            reported=None,
        ),
    )
    write_scenarios_view(root, {"nominal": {}, "mode-change": MODE_CHANGE})
    return view


@pytest.fixture(scope="module")
def archive(view_dir: Path) -> ArchiveComposite:
    return build(view_dir, ["mode-change"], anchor_s=ANCHOR_S)


def test_the_view_carries_the_mode_as_an_enum_of_four_options(view_dir: Path) -> None:
    view = SimulatorView.open(view_dir)
    records = {channel["address"]: channel for channel in view.document(VARIABLES_FILE)["channels"]}
    assert records[MODE]["value_type"] == "enum"
    assert list(records[MODE]["options"]) == MODES


def test_the_va_serves_an_enum_variable_with_the_four_labels_in_order(
    view_dir: Path, tmp_path: Path
) -> None:
    composite = Composite(
        SimulatorView.open(view_dir),
        state_dir=tmp_path,
        instance="virtual_accelerator",
        model_log=False,
    )

    variable = composite.supported_variables[MODE]

    assert isinstance(variable, EnumVariable)
    assert list(variable.options) == MODES
    assert variable.read_only is True
    assert composite.get(MODE) == "Machine studies"


def test_the_seed_stores_the_option_index_of_each_label(archive: ArchiveComposite) -> None:
    times = [
        ANCHOR_S + INJECTION_AT_S - 60.0,
        ANCHOR_S + INJECTION_AT_S + 60.0,
        ANCHOR_S + OFF_AT_S + 60.0,
    ]

    samples = archive.series(MODE, times)

    assert samples == [
        MODES.index("Machine studies"),
        MODES.index("Injection"),
        MODES.index("Off"),
    ]
    assert all(type(sample) is int for sample in samples)


def test_the_recorder_records_the_mode_as_a_scalar(view_dir: Path) -> None:
    channels = resolve_channel_addresses(view_dir)

    assert MODE in channels.scalars
    assert MODE not in channels.waveforms


async def test_the_recorder_stores_the_ca_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = MODES.index("Machine studies")
    install_fake_aioca(
        monkeypatch,
        {
            MODE: FakeChannel(
                ca_reading(index, datatype=DBR_ENUM),
                ctrl=ca_reading(0, datatype=DBR_ENUM, enums=MODES),
            )
        },
    )
    writer = _FakeWriter()
    settings = _settings()
    recorder = Recorder(
        settings=settings,
        addresses=[MODE],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=channel_access_reader(settings),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is True

    _, values = writer.samples[0]
    assert values == {MODE: index}
    assert float(values[MODE]).is_integer()
