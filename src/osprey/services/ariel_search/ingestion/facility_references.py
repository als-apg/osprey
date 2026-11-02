"""The facility records an ARIEL entry writer checks entries against.

The entry writer checks each entry against the facility file of the project it
runs in; without one, entries are stored unchecked (``NULL``) until re-ingested.
"""

from __future__ import annotations

from osprey.facility.records import FacilityRecords
from osprey.utils.facility import FacilityFileError
from osprey_connectors.logger import get_logger

logger = get_logger("ariel")


def load_facility_records() -> FacilityRecords | None:
    """Read the facility file of this project.

    Returns:
        Its records, or ``None`` when it is absent or cannot be read.
    """
    from osprey.channel_roster.sources import facility_file_path
    from osprey_connectors.workspace import load_osprey_config

    path = facility_file_path(load_osprey_config())
    try:
        records = FacilityRecords.from_file(path)
    except FacilityFileError as exc:
        logger.warning(f"Facility file {path} not used, entries are stored unchecked: {exc}")
        return None
    if records is None:
        logger.info(
            "facility file not built; logbook entries are stored without facility references"
        )
    return records
