"""What every archiver answers the same way, whatever its store.

An archiver that is not connected refuses ``check_availability`` with
``RuntimeError``: a disconnected connector has no store to ask, so it can say
neither yes, no nor unknown about a channel.
"""

import pytest

from osprey_connectors.archiver._doocs_archiver_connector import DOOCSArchiverConnector
from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector
from osprey_connectors.archiver._mock_archiver_connector import MockArchiverConnector
from osprey_connectors.archiver._mongodb_archiver_connector import MongoDBArchiverConnector
from osprey_connectors.archiver._mya_archiver_connector import MYAArchiverConnector

#: Every built-in implementation of the archiver contract.
ARCHIVERS = [
    EPICSArchiverConnector,
    MockArchiverConnector,
    MongoDBArchiverConnector,
    DOOCSArchiverConnector,
    MYAArchiverConnector,
]


@pytest.mark.parametrize("cls", ARCHIVERS, ids=lambda c: c.__name__)
async def test_check_availability_refuses_when_not_connected(cls) -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        await cls().check_availability(["SR:CURRENT:RB"])
