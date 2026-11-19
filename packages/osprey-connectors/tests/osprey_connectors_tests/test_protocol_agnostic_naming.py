"""The protocol-agnostic connector layer must not name protocol primitives.

Two families sit behind one abstraction: control-system connectors speak
``address`` everywhere, while the archiver contract used to speak
``pv_list``/``pv_name`` — an EPICS word that a DOOCS or MongoDB backend has no
use for. The archive now speaks ``address`` too: in its parameters, its
coverage record and the column of the frame it returns. The same leak reached
the public reason codes (``CAPUT_FAILED`` named a Channel Access primitive that
a DOOCS write never runs) and the simulation helpers. The write reason codes are now the ``WriteOutcome`` words themselves,
which describe what became of the write rather than how it was sent.

These are contract-level assertions on purpose: a rename that misses one
implementation leaves the abstraction split, and only the contract shows it.
Every backend builds its frame with ``long_frame``, so checking that builder's
columns checks every backend's.
"""

import inspect
from datetime import UTC, datetime

import pandas as pd
import pytest

from osprey_connectors.archiver import ArchiveCoverage, ChannelArchive
from osprey_connectors.archiver._doocs_archiver_connector import DOOCSArchiverConnector
from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector
from osprey_connectors.archiver._mock_archiver_connector import MockArchiverConnector
from osprey_connectors.archiver._mongodb_archiver_connector import MongoDBArchiverConnector
from osprey_connectors.archiver._mya_archiver_connector import MYAArchiverConnector
from osprey_connectors.archiver._timerange import LONG_COLUMNS, long_frame
from osprey_connectors.control_system._base import ChannelMetadata
from osprey_connectors.errors import ChannelWriteFailedError

#: Every implementation of the archiver contract, plus the contract itself.
ARCHIVERS = [
    ChannelArchive,
    EPICSArchiverConnector,
    MockArchiverConnector,
    MongoDBArchiverConnector,
    DOOCSArchiverConnector,
    MYAArchiverConnector,
]


def _params(cls, method: str) -> list[str]:
    return list(inspect.signature(getattr(cls, method)).parameters)


@pytest.mark.parametrize("cls", ARCHIVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("history", "addresses"),
        ("coverage", "addresses"),
    ],
)
def test_archiver_contract_speaks_addresses(cls, method, expected):
    """The archiver's own parameter names must match the columns it returns."""
    params = _params(cls, method)
    assert expected in params, f"{cls.__name__}.{method} params: {params}"
    assert not [p for p in params if p.startswith("pv")], (
        f"{cls.__name__}.{method} still names a PV: {params}"
    )


def test_archive_coverage_identifies_a_channel_by_address():
    """ArchiveCoverage names what it describes the same way history does."""
    fields = ArchiveCoverage.model_fields
    assert "address" in fields
    assert not [f for f in fields if f.startswith("pv")], f"ArchiveCoverage fields: {list(fields)}"


@pytest.mark.parametrize("empty", [False, True], ids=["samples", "empty"])
def test_archiver_frame_names_its_rows_by_address(empty):
    """The frame history returns names each row's channel the way its parameters do."""
    index = pd.DatetimeIndex([] if empty else [datetime(2026, 1, 1, tzinfo=UTC)], tz=UTC)
    frame = long_frame({"A:RB": pd.Series([] if empty else [1.0], index=index, dtype=float)})
    assert LONG_COLUMNS == ("timestamp", "address", "value")
    assert list(frame.columns) == list(LONG_COLUMNS)


def test_write_failure_reasons_are_protocol_neutral():
    """A DOOCS write failure must not be reported as a Channel Access failure."""
    for outcome in ("failed", "mismatch", "unconfirmed"):
        assert ChannelWriteFailedError("SR:Q", outcome).outcome == outcome


def test_channel_metadata_display_range_is_not_named_like_a_write_bound():
    """ChannelMetadata reports what the control system displays.

    ``ChannelLimitsConfig.min_value``/``max_value`` are the bounds OSPREY refuses
    to write past. Sharing those identifiers made a display range look enforced.
    """
    fields = ChannelMetadata.model_fields
    assert "display_low" in fields
    assert "display_high" in fields
    assert "min_value" not in fields
    assert "max_value" not in fields
