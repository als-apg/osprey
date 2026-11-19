"""Channel-value and config stand-ins shared by the channel_read tool suites."""

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import MagicMock

from osprey_connectors_tests._config_trees import installed, with_values


def make_channel_value(value, sampled_at=datetime(2024, 1, 15, 10, 30, tzinfo=UTC)):
    """A non-enum channel reading of ``value`` with fixed metadata."""
    cv = MagicMock()
    cv.value = value
    cv.sampled_at = sampled_at
    cv.metadata = MagicMock()
    cv.metadata.units = "counts"
    cv.metadata.precision = 3
    cv.alarm.status = "NO_ALARM"
    cv.alarm.severity = 0
    cv.metadata.description = "Test channel"
    # Not an enum channel: a bare MagicMock attribute would be truthy, and the
    # read tool ships the enum keys only when they are not None.
    cv.enum_label = None
    cv.metadata.enum_labels = None
    return cv


def config_patch(overrides: dict):
    """Serve the process config with the named keys (dotted) set over it."""
    if not overrides:
        return nullcontext()
    return installed(with_values(overrides))
