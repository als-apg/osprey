"""The channel-finder index views: each selected pipeline's database.

The in_context index is written to ``<render>/data/channel_finder/in_context.json``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "channels": [
        {"channel": <name>, "address": <address>, "description": <text>},
        ...
      ]
    }

One row per channel tagged ``in_context``, sorted by address. A row's
``channel`` is the channel's first ``names`` entry, else its address; its
``description`` is the channel's own. A render carries the index when its
``channel_finder.pipeline_mode`` is ``in_context``; a facility with no tagged
channel then stops the build with ``view-unsupported``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs

__all__ = [
    "CHANNEL_FINDER_SCHEMA",
    "IN_CONTEXT_FILE",
    "IN_CONTEXT_TAG",
    "in_context_document",
    "in_context_selected",
    "write_in_context",
]

CHANNEL_FINDER_SCHEMA = "osprey.facility.channel_finder/1"
IN_CONTEXT_FILE = "in_context.json"
IN_CONTEXT_TAG = "in_context"

#: The rendered config key that selects a pipeline.
PIPELINE_MODE_KEY = "channel_finder.pipeline_mode"


def _row(channel: Mapping[str, Any]) -> dict[str, Any]:
    address = str(channel["id"])
    names = channel.get("names")
    name = names[0] if isinstance(names, list) and names else address
    return {"channel": name, "address": address, "description": channel.get("description")}


def in_context_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The in_context index of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{schema, channels}``, one row per tagged channel, sorted by address.
    """
    rows = [
        _row(channel)
        for channel in doc.get("channels", [])
        if IN_CONTEXT_TAG in (channel.get("tags") or [])
    ]
    rows.sort(key=lambda row: row["address"])
    return {"schema": CHANNEL_FINDER_SCHEMA, "channels": rows}


def in_context_selected(inputs: ViewInputs) -> bool:
    """Whether the render selects the in_context pipeline.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when the rendered ``channel_finder.pipeline_mode`` is ``in_context``.
    """
    channel_finder = inputs.rendered_config.get("channel_finder")
    if not isinstance(channel_finder, Mapping):
        return False
    return channel_finder.get("pipeline_mode") == IN_CONTEXT_TAG


def write_in_context(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the in_context index into ``root``.

    Args:
        root: The render's ``data/channel_finder`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.

    Raises:
        FacilityBuildError: ``view-unsupported`` when no channel is tagged
            ``in_context``.
    """
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.served import CONFIG_SOURCE
    from osprey.facility.views import view_bytes

    document = in_context_document(inputs.doc)
    if not document["channels"]:
        raise FacilityBuildError(
            "view-unsupported",
            PIPELINE_MODE_KEY,
            [CONFIG_SOURCE],
            f"tag at least one channel `{IN_CONTEXT_TAG}`, or select another channel_finder_mode",
            record_kind="path",
            detail=f"selects {IN_CONTEXT_TAG} and no channel is tagged `{IN_CONTEXT_TAG}`",
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / IN_CONTEXT_FILE
    target.write_bytes(view_bytes(document))
    return [target]
