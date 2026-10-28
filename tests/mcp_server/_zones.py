"""Time-zone helpers for tests that must tell a facility zone from the host clock."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def zone_away_from_host() -> ZoneInfo:
    """Return a zone whose wall-clock differs from the host clock by hours.

    A stamp shown to follow the facility zone and not the host clock needs the two
    to differ by more than the test tolerance on any host. UTC+14 is used unless the
    host already runs at +14 h; the closest real host offset to +14 h is +13:45,
    fifteen minutes away.
    """
    if datetime.now().astimezone().utcoffset() == timedelta(hours=14):
        return ZoneInfo("Etc/GMT+12")
    return ZoneInfo("Etc/GMT-14")
