"""MCP tool: capabilities — report ARIEL service capabilities."""

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import make_error, mcp
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.services.ariel_search.capabilities import get_capabilities
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.entry_fields import (
    EntryFieldDeclarationError,
    entry_field_descriptors,
)

logger = logging.getLogger("osprey.mcp_server.ariel.tools.capabilities")


def _entry_fields_block(config: ARIELConfig) -> dict[str, Any]:
    """The ``entry_fields`` payload keys for the configured adapter.

    A misdeclared field is reported as ``entry_fields_error`` beside an empty
    ``entry_fields`` list, so the rest of the capabilities still answer.

    Args:
        config: ARIEL configuration naming the ingestion adapter.

    Returns:
        ``{"entry_fields": [...]}``, plus ``entry_fields_error`` on a
        declaration error.
    """
    try:
        descriptors = entry_field_descriptors(config)
    except EntryFieldDeclarationError as exc:
        logger.warning("entry field declarations refused: %s", exc.message)
        return {"entry_fields": [], "entry_fields_error": exc.message}

    fields = []
    for descriptor in descriptors:
        item = descriptor.to_dict()
        item.pop("options_endpoint", None)
        fields.append(item)
    return {"entry_fields": fields}


@mcp.tool()
async def capabilities() -> str:
    """Report available ARIEL search capabilities.

    Returns enabled search modules, search modes, the embedding provider, the
    facility vocabulary, and the parameters every search mode accepts.

    ``vocabulary`` reports whether this deployment ships a facility vocabulary,
    how many concepts it holds, and whether expansion is applied by default.
    When it is enabled, the search tools' ``expand_query`` argument overrides
    that default per call; when it is disabled the argument is a no-op and
    ``shared_parameters`` carries no ``expand_query`` entry.

    ``search_modes`` lists the search modules that are both registered and
    enabled — the modes a ``search`` call may actually route to. It is derived
    from the same registry-backed source the capabilities API uses, so the two
    can never drift apart. ``sql_query`` is a separate MCP tool rather than a
    search mode: it bypasses mode dispatch by design and is therefore
    intentionally absent from this list.

    ``attachments`` describes what this deployment does with logbook pictures
    and other attachment files, read from configuration only:

    - ``copy_on_ingest``: which attachment files ingest copies into the store
      (``"images"``, ``"all"`` or ``"none"``).
    - ``formats``: ``viewable`` lists the formats a stored picture can be shown
      in; ``reserved`` lists formats that are recognised but never shown.
    - ``view``: whether agents may look at stored pictures with
      ``attachment_view`` and keep them with ``attachment_to_artifact``; when
      false, attachments are not offered to agents.
    - ``captions``: whether model captions are generated for pictures.
    - ``picture_search``: whether searches can match pictures, which needs both
      image embeddings and the ``hybrid`` search mode enabled.
    - ``picture_search_unavailable``: why this server last saw picture search
      fail -- ``"unreachable"`` (the embedding server did not answer),
      ``"model"`` (it serves another model), ``"auth"`` (it refused the
      credentials) or ``"config"`` (the ``image_embedding`` block is unusable)
      -- or null when picture search has not failed or its last attempt
      succeeded. It is kept until a later search tries pictures again, so it
      can name a fault that has since been fixed; while it is set, hybrid
      searches may answer from text alone. Read from this process's memory,
      not probed.

    The web ``/api/capabilities`` endpoint reports the same block.

    ``entry_fields`` lists the extra fields this facility's logbook asks for
    when an entry is written, in form order; it is an empty list when the
    facility declares none or no ingestion adapter is configured. Each item has
    ``name``, ``label``, ``description``, ``type`` (``text``, ``int``,
    ``float``, ``bool``, ``date``, ``select`` or ``dynamic_select``),
    ``default`` and ``section``; ``options`` lists the allowed
    ``{value, label}`` choices of a ``select``; ``min``/``max`` bound a
    number; ``required`` appears (true) when the field must be filled before
    the entry is published; ``depends_on`` names the fields whose values decide
    a ``dynamic_select``'s choices, which are read live from the facility when
    the entry is checked. Pass values for these fields, keyed by ``name``, as
    the ``fields`` argument of ``entry_create`` and ``entry_publish``; a wrong
    value is refused naming the field. When the facility declares its fields
    wrongly, ``entry_fields`` is empty and ``entry_fields_error`` says what is
    wrong; that key is absent otherwise. The web ``/api/capabilities``
    endpoint does not report this block.

    Does NOT require database connectivity, so this is *not* a health check: a
    successful response says nothing about whether the database is reachable.
    For live database/health status (connectivity, entry counts) call the
    ``status`` tool, or run ``osprey ariel status``.

    Returns:
        JSON with capabilities information.
    """
    try:
        context = get_ariel_context()
        config = context.config
        caps = get_capabilities(config)
        modes = caps["categories"]["direct"]["modes"]

        return json.dumps(
            {
                "search_modes": [mode["name"] for mode in modes],
                "enabled_search_modules": config.get_enabled_search_modules(),
                "embedding": {
                    "provider": config.embedding.provider,
                },
                "vocabulary": caps["vocabulary"],
                "shared_parameters": caps["shared_parameters"],
                "attachments": caps["attachments"],
                **_entry_fields_block(config),
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("capabilities failed")
        return make_error(
            "internal_error",
            f"Failed to get capabilities: {exc}",
            ["Check ARIEL configuration in config.yml."],
        )
