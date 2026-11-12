"""The words the PV directory service serves for its sources: one producer of each.

The connectors' answers name only the kind and the type of the service that
answered; what the panel and the agent read as a source's label, and the label
of an alarm configuration, are made here.
"""

from __future__ import annotations

__all__ = ["CONFIG_LABELS", "DIRECTORY_LABEL", "source_label"]

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
