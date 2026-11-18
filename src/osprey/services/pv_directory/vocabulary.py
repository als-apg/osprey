"""The words the PV directory service serves for its sources: one producer of each.

The connectors' answers name only the kind and the type of the service that
answered; what the panel and the agent read as a source's label, the label of
an alarm configuration and the reason a raising service gives are made here.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "CONFIG_LABELS",
    "DIRECTORY_LABEL",
    "LOGBOOK_SERVICE",
    "ServedOutcome",
    "service_failed",
    "source_label",
]

#: How a served answer ended, as the pages and the tools read its ``outcome``: the
#: connectors' page outcomes (``found``, ``no_match``), and the framework's own
#: words for an exact lookup that finds nothing (``not_listed``), a channel the
#: directory lists as inactive (``inactive``), a PV no section knows (``unknown``)
#: and a source that gave no answer (``unavailable``).
ServedOutcome = Literal["found", "no_match", "not_listed", "inactive", "unknown", "unavailable"]

#: The service type of the logbook source.
LOGBOOK_SERVICE = "ariel"

#: The label of the directory, on every target and whatever answers it.
DIRECTORY_LABEL = "All PVs"

#: The label of an alarm configuration, by the kind of service that answered it.
CONFIG_LABELS = {
    "alarm_state": "live configuration",
    "alarm_log": "last logged configuration",
}


def source_label(kind: str, service: str) -> str:
    """The served label of a source: the directory's, else the service's type name."""
    return DIRECTORY_LABEL if kind == "directory" else service


def service_failed(failure: BaseException) -> str:
    """The served reason of a service that raised: its exception's class, never its text.

    A backend's exception can carry its URL, a login included; the caller logs
    the whole exception, and only the class name is served.
    """
    return f"the service failed: {type(failure).__name__}"
