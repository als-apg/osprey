"""What every archive is, whatever its store.

Every built-in is a service backend of kind ``archive``: it is built from a
``BackendSpec`` with the constructor every backend shares, and answers the
contract's ``history`` and ``coverage`` with the contract's own parameters, so
a caller holding any archive asks it the same way.
"""

import inspect

import pytest

from osprey_connectors.archiver import ChannelArchive
from osprey_connectors.archiver._doocs_archiver_connector import DOOCSArchiverConnector
from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector
from osprey_connectors.archiver._mock_archiver_connector import MockArchiverConnector
from osprey_connectors.archiver._mongodb_archiver_connector import MongoDBArchiverConnector
from osprey_connectors.archiver._mya_archiver_connector import MYAArchiverConnector
from osprey_connectors.connector_types import BUILTIN_CONNECTORS
from osprey_connectors.factory import ConnectorFactory

#: Every built-in implementation of the archive contract.
ARCHIVERS = [
    EPICSArchiverConnector,
    MockArchiverConnector,
    MongoDBArchiverConnector,
    DOOCSArchiverConnector,
    MYAArchiverConnector,
]


@pytest.mark.parametrize("cls", ARCHIVERS, ids=lambda c: c.__name__)
def test_every_archiver_is_an_archive_backend(cls) -> None:
    assert issubclass(cls, ChannelArchive)
    assert list(inspect.signature(cls.__init__).parameters) == [
        "self",
        "spec",
        "project_dir",
        "seams",
    ]


@pytest.mark.parametrize("cls", ARCHIVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", ["history", "coverage", "extent", "probe"])
def test_every_archiver_takes_the_contracts_parameters(cls, method) -> None:
    contract = inspect.signature(getattr(ChannelArchive, method)).parameters
    own = inspect.signature(getattr(cls, method)).parameters
    assert list(own) == list(contract)
    assert [p.kind for p in own.values()] == [p.kind for p in contract.values()]


def test_every_builtin_archiver_row_registers_as_an_archive() -> None:
    rows = [row.name for row in BUILTIN_CONNECTORS if row.kind == "archiver"]
    assert rows
    assert set(rows) <= set(ConnectorFactory.list_services("archive"))
