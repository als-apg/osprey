"""The protocol-agnostic connector layer must not name protocol primitives.

Two families sit behind one abstraction: control-system connectors speak
``channel_address`` everywhere, while the archiver contract used to speak
``pv_list``/``pv_name`` — an EPICS word that a DOOCS or MongoDB backend has no
use for. The same leak reached the public reason codes (``CAPUT_FAILED`` named a
Channel Access primitive that a DOOCS write never runs) and the simulation
helpers. The write reason codes are now the ``WriteOutcome`` words themselves,
which describe what became of the write rather than how it was sent.

These are signature-level assertions on purpose: a rename that misses one
implementation leaves the abstraction split, and only the contract shows it.
"""

import inspect

import pytest

from osprey.connectors.archiver import ArchiverConnector, ArchiverMetadata
from osprey.connectors.archiver.doocs_archiver_connector import DOOCSArchiverConnector
from osprey.connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector
from osprey.connectors.control_system.base import ChannelMetadata
from osprey.errors import ChannelWriteFailedError

#: Every implementation of the archiver contract, plus the contract itself.
ARCHIVERS = [
    ArchiverConnector,
    EPICSArchiverConnector,
    MockArchiverConnector,
    MongoDBArchiverConnector,
    DOOCSArchiverConnector,
]


def _params(cls, method: str) -> list[str]:
    return list(inspect.signature(getattr(cls, method)).parameters)


@pytest.mark.unit
@pytest.mark.parametrize("cls", ARCHIVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("get_data", "channels"),
        ("get_metadata", "channel"),
        ("check_availability", "channels"),
    ],
)
def test_archiver_contract_speaks_channels(cls, method, expected):
    """The archiver's own parameter names must match the columns it returns."""
    params = _params(cls, method)
    assert expected in params, f"{cls.__name__}.{method} params: {params}"
    assert not [p for p in params if p.startswith("pv")], (
        f"{cls.__name__}.{method} still names a PV: {params}"
    )


@pytest.mark.unit
def test_archiver_metadata_identifies_a_channel():
    """ArchiverMetadata names what it describes the same way get_data does."""
    fields = ArchiverMetadata.__dataclass_fields__
    assert "channel" in fields
    assert "pv_name" not in fields


@pytest.mark.unit
def test_simulation_engine_and_taxonomy_speak_channels():
    """The simulation helpers are wrapped by channel-named callers one line deep."""
    from osprey.connectors.channel_taxonomy import classify_channel
    from osprey.simulation.engine import SimulationEngine

    assert _params(SimulationEngine, "has_channel") == ["self", "channel"]
    assert list(inspect.signature(classify_channel).parameters) == ["channel"]


@pytest.mark.unit
def test_write_failure_reasons_are_protocol_neutral():
    """A DOOCS write failure must not be reported as a Channel Access failure."""
    assert ChannelWriteFailedError._VALID_REASONS == ("FAILED", "MISMATCH", "UNCONFIRMED")


@pytest.mark.unit
def test_channel_metadata_display_range_is_not_named_like_a_write_bound():
    """ChannelMetadata reports what the control system displays.

    ``ChannelLimitsConfig.min_value``/``max_value`` are the bounds OSPREY refuses
    to write past. Sharing those identifiers made a display range look enforced.
    """
    fields = ChannelMetadata.__dataclass_fields__
    assert "display_low" in fields
    assert "display_high" in fields
    assert "min_value" not in fields
    assert "max_value" not in fields
