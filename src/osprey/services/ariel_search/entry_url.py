"""The canonical logbook link for an ARIEL entry.

One producer for both egress surfaces (the MCP tools and the web API):

- :func:`is_ariel_only` — whether an entry exists only in ARIEL, so the
  facility logbook has no page for it;
- :func:`build_entry_url` — the config-driven ``ariel.entry_url_template``
  rendered with the entry id, or ``None``.

``get_config_value`` resolves the default config, which the MCP server and the
web lifespan (through ``get_registry(config_path=...)``) both set to the
deployment's ``config.yml``, so the two processes read the same template.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ARIEL_NATIVE_SOURCE_SYSTEM", "build_entry_url", "is_ariel_only"]

# The source_system value ARIEL stamps on its own natively-created entries
# (see mcp_server/ariel/tools/entry.py entry_create direct mode). Such entries
# are not (yet) in the facility logbook, so no canonical entry_url exists.
ARIEL_NATIVE_SOURCE_SYSTEM = "ARIEL MCP"

# One-time guard so a malformed template does not spam the per-entry hot path.
_entry_url_template_warned = False


def is_ariel_only(source_system: str | None, metadata: Mapping[str, Any] | None) -> bool:
    """Whether the entry lives only in ARIEL, with no facility logbook page.

    True for an entry saved local-only (``metadata.sync_status`` is
    ``local_only``, e.g. an ``ARIEL Web`` save) and for an ARIEL-native entry
    (``source_system`` is :data:`ARIEL_NATIVE_SOURCE_SYSTEM`).
    """
    from osprey.services.ariel_search.models import SyncStatus

    if source_system == ARIEL_NATIVE_SOURCE_SYSTEM:
        return True
    if isinstance(metadata, Mapping):
        return metadata.get("sync_status") == SyncStatus.LOCAL_ONLY.value
    return False


def build_entry_url(
    entry_id: str | None,
    source_system: str | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Render the config-driven canonical logbook entry URL, or ``None``.

    An egress transform mirroring ``to_facility_iso``: it reads the
    facility-supplied ``ariel.entry_url_template`` from the merged config and
    renders it with the URL-encoded ``entry_id`` so the agent links entries
    verbatim instead of inventing a URL.

    Returns ``None`` (emit no URL) when:

    - the entry is ARIEL-only (:func:`is_ariel_only`): an ARIEL-native entry or
      a local-only save is not in the facility logbook;
    - ``entry_id`` is empty/blank;
    - no ``ariel.entry_url_template`` is configured (a deployment that
      configures no template emits no ``entry_url`` and shows plain IDs);
    - the template is malformed (fail-safe — this runs per-entry on the search
      hot path, so a one-character typo in the template must degrade to "no URL",
      never crash a read).
    """
    from urllib.parse import quote

    # Imported here, not at module level: tests patch the source module.
    from osprey_connectors.config import get_config_value

    if is_ariel_only(source_system, metadata):
        return None
    if not entry_id or not str(entry_id).strip():
        return None

    try:
        template = get_config_value("ariel.entry_url_template", None)
    except Exception:
        # Config not loaded/resolvable (e.g. no config.yml): treat as an
        # unconfigured deployment — emit no URL rather than crash the read.
        return None
    if not template:
        return None

    try:
        if not isinstance(template, str):
            raise TypeError(f"expected a string, got {type(template).__name__}")
        return template.format(entry_id=quote(str(entry_id), safe=""))
    except Exception:
        global _entry_url_template_warned
        if not _entry_url_template_warned:
            _entry_url_template_warned = True
            logger.warning(
                "Malformed ariel.entry_url_template %r (needs a single {entry_id} "
                "placeholder); emitting no entry_url.",
                template,
            )
        return None
