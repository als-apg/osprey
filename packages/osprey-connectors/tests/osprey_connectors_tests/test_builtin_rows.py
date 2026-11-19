"""Every list of built-in types is read off the built-in rows, and reads as it always did.

``connector_types.BUILTIN_CONNECTORS`` holds one row per built-in type, and the
public lists (what ``osprey init`` offers, what ``osprey set`` accepts, which
types are simulated, which invent their history, ...) are derived from it. The
values below are the literal tables those lists were before the rows existed,
so a derivation that drifts — a row reordered, a flag flipped — fails here
rather than in whatever reads the list.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from osprey_connectors import call_timeout, config_sections, connector_types, factory, targets
from osprey_connectors.connector_types import BuiltinConnector


def test_the_public_lists_keep_their_values_and_order() -> None:
    assert connector_types.CLI_CONTROL_SYSTEM_TYPES == [
        "epics",
        "virtual_accelerator",
        "doocs",
        "tango",
    ]
    assert connector_types.SET_CONTROL_SYSTEM_TYPES == [
        "epics",
        "virtual_accelerator",
        "doocs",
        "tango",
        "live_standin",
    ]
    assert connector_types.SIMULATED_TYPES == ("virtual_accelerator",)
    assert connector_types.STANDIN_TYPES == ("live_standin",)
    assert connector_types.INVENTED_HISTORY_TYPES == ("virtual_accelerator", "live_standin")
    assert connector_types.DEFAULT_VA_PORT == 5064
    assert connector_types.RETIRED_CONTROL_SYSTEM_TYPES == {"mock": "virtual_accelerator"}
    assert connector_types.CONTROL_TARGETS == ["live", "va", "standin"]


def test_the_private_tables_keep_their_values() -> None:
    assert connector_types._TRANSPORTS == {
        "epics": "ca",
        "virtual_accelerator": "ca",
        "live_standin": "ca",
        "doocs": "doocs",
        "tango": "tango",
    }
    assert targets._BASELINE_TARGETS == {"virtual_accelerator": "va", "live_standin": "standin"}
    assert list(config_sections._ARCHIVER_TYPES) == [
        "mock_archiver",
        "epics_archiver",
        "mongodb_archiver",
        "doocs_archiver",
        "mya_archiver",
    ]
    assert call_timeout._RENAMED_PATHS == frozenset(
        {
            "control_system.connector.epics.timeout",
            "control_system.connector.virtual_accelerator.timeout",
            "control_system.connector.live_standin.timeout",
            "control_system.connector.tango.timeout",
            "control_system.connector.doocs.timeout",
        }
    )


def test_the_factory_registers_the_rows_in_their_order() -> None:
    assert factory._BUILTIN_CONTROL_SYSTEMS == (
        "epics",
        "virtual_accelerator",
        "doocs",
        "tango",
        "live_standin",
    )
    assert factory._BUILTIN_ARCHIVERS == (
        "mock_archiver",
        "epics_archiver",
        "mongodb_archiver",
        "doocs_archiver",
        "mya_archiver",
    )


def test_the_type_name_alias_names_every_control_system_row() -> None:
    names = {row.name for row in connector_types.BUILTIN_CONNECTORS if row.kind == "control_system"}

    assert set(get_args(connector_types.ControlSystemTypeName)) == names


def _row(**fields: object) -> BuiltinConnector:
    base: dict[str, object] = {
        "name": "epics",
        "kind": "control_system",
        "module_path": "m",
        "class_name": "C",
        "description": "d",
        "transport": "ca",
        "machine": "real",
        "baseline_target": "live",
    }
    return BuiltinConnector.model_validate({**base, **fields})


def test_a_control_system_row_states_its_machine() -> None:
    assert _row().machine == "real"
    with pytest.raises(ValidationError, match="transport, machine and target"):
        _row(transport=None)


def test_a_control_system_row_is_named_by_the_alias() -> None:
    with pytest.raises(ValidationError, match="not a ControlSystemTypeName"):
        _row(name="site.Connector")


def test_an_archiver_row_states_no_machine() -> None:
    with pytest.raises(ValidationError, match="states a control-system field"):
        _row(name="mock_archiver", kind="archiver")
    with pytest.raises(ValidationError, match="states a control-system field"):
        _row(
            name="mock_archiver",
            kind="archiver",
            transport=None,
            machine=None,
            baseline_target=None,
            invents_history=True,
        )
