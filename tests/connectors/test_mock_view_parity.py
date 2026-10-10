"""The mock's metadata is the simulator view's, read without the facility file.

A composite built from a copy of the control-assistant build's ``data/``, with
``facility.json`` left out, serves every address the mock serves, and each
address's unit and description as the mock reports them.
"""

from __future__ import annotations

import json
import shutil

from osprey.connectors.control_system.va_in_process_connector import VAInProcessConnector
from tests.facility.served_tree import in_process_config

FACILITY_FILE = "facility.json"


async def test_a_composite_without_the_facility_file_matches_the_in_process_metadata(
    built_control_assistant, tmp_path
):
    from osprey_connectors.simulation.composite import Composite

    data = tmp_path / "build" / "data"
    shutil.copytree(
        built_control_assistant.build_dir / "data",
        data,
        ignore=shutil.ignore_patterns(FACILITY_FILE),
    )
    assert not (data / FACILITY_FILE).exists()
    view = data / "simulator"
    composite = Composite(view, model_log=False)
    records = {
        channel["address"]: channel
        for channel in json.loads((view / "variables.json").read_text())["channels"]
    }

    connector = VAInProcessConnector()
    await connector.connect(
        in_process_config(built_control_assistant.build_dir / "data" / "simulator")
    )
    try:
        addresses = sorted(connector._served)
        assert addresses == sorted(composite.supported_variables)
        for address in addresses:
            metadata = await connector.get_metadata(address)
            variable = composite.supported_variables[address]
            record = records.get(address, {})
            unit = getattr(variable, "unit", None) or record.get("unit") or ""
            assert metadata.units == unit, address
            assert metadata.description == record.get("description"), address
    finally:
        await connector.disconnect()

    assert addresses, "the build serves no address"
    assert any(records[a].get("description") for a in addresses if a in records)
