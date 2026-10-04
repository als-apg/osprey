"""Whether the ARIEL agent may look at logbook pictures: one switch.

``ariel.attachments.view.enabled`` decides whether ``attachment_view``,
``attachment_to_artifact`` and the attachment summaries are offered to agents.
It defaults to on: viewing a stored picture makes no model call and needs no
server.

This leaf module imports nothing from ``osprey``. ``AttachmentsConfig`` parses
the key through it at runtime and the build reads the raw config through it,
so both apply one rule without the build importing the ARIEL service.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ATTACHMENTS_KEY = "ariel.attachments"
VIEW_KEY = "ariel.attachments.view"
VIEW_ENABLED_KEY = "ariel.attachments.view.enabled"
DEFAULT_VIEW_ENABLED = True


def _block(parent: Mapping[str, Any], name: str, key: str) -> Mapping[str, Any]:
    """Return a nested block; absent or null is empty, any other non-mapping is refused."""
    value = parent.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping, got {value!r}")
    return value


def attachment_view_enabled(ariel_section: Mapping[str, Any]) -> bool:
    """Return the configured ``ariel.attachments.view.enabled`` value.

    Args:
        ariel_section: The ``ariel`` section of the loaded configuration.

    Returns:
        The switch; an absent key or block gives ``True``.

    Raises:
        ValueError: The value is present but not a boolean (``"no"`` and ``1``
            are refused, not coerced), or a parent block is not a mapping.
    """
    attachments = _block(ariel_section, "attachments", ATTACHMENTS_KEY)
    view = _block(attachments, "view", VIEW_KEY)
    value = view.get("enabled", DEFAULT_VIEW_ENABLED)
    if not isinstance(value, bool):
        raise ValueError(f"{VIEW_ENABLED_KEY} must be a boolean, got {value!r}")
    return value
