"""Channel-value and config stand-ins shared by the channel_read tool suites."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch


def make_channel_value(value, timestamp="2024-01-15T10:30:00"):
    """A non-enum channel reading of ``value`` with fixed metadata."""
    cv = MagicMock()
    cv.value = value
    cv.timestamp = timestamp
    cv.metadata = MagicMock()
    cv.metadata.units = "counts"
    cv.metadata.precision = 3
    cv.metadata.alarm_status = "NO_ALARM"
    cv.metadata.alarm_severity = 0
    cv.metadata.description = "Test channel"
    # Not an enum channel: a bare MagicMock attribute would be truthy, and the
    # read tool ships the enum keys only when they are not None.
    cv.metadata.enum_label = None
    cv.metadata.enum_labels = None
    return cv


def config_patch(overrides: dict):
    """Patch only the named config keys, delegating everything else."""
    if not overrides:
        return nullcontext()

    from osprey_connectors import config as config_module

    real = config_module.get_config_value

    def fake(path, default=None, *args, **kwargs):
        if path in overrides:
            return overrides[path]
        return real(path, default, *args, **kwargs)

    return patch("osprey_connectors.config.get_config_value", side_effect=fake)
