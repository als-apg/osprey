"""MCP tool: hybrid_search — hybrid ranked search of the ARIEL logbook.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: guidance on when to prefer this over keyword_search
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import (
    make_error,
    mcp,
    parse_date_filters,
)
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.mcp_server.ariel.tools.search_envelope import (
    ResultWindow,
    advanced_params,
    raise_for_fault_exception,
    raise_for_vocabulary_error,
    raise_on_statement_fault,
    success_envelope,
)
from osprey.services.ariel_search.exceptions import (
    ConfigurationError,
    PatternError,
    SearchTimeoutError,
    VocabularyError,
)

logger = logging.getLogger("osprey.mcp_server.ariel.tools.hybrid_search")

# Diagnostic source the search service stamps on a failure of this mode.
_QMD_DIAGNOSTIC_SOURCE = "service.hybrid"


@mcp.tool()
async def hybrid_search(
    query: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
    exclude_entry_ids: list[str] | None = None,
    expand_query: bool | None = None,
) -> str:
    """Search the ARIEL logbook using hybrid keyword + semantic ranking.

    Runs both a keyword and a vector query over the mirrored logbook corpus and
    returns one merged ranking. Best when a question mixes specific terms with a
    described situation, or when keyword_search returned too little.

    Filtering here is best-effort, and that matters for how you read a short
    result set. The corpus is ranked first and the date, author and
    source_system filters are applied afterwards to the top of that ranking —
    they are not applied inside the database. A selective filter can therefore
    return fewer than max_results entries even when more matching entries exist
    in the corpus. Read a short result set as "the ranked window ran out", never
    as "no more matching entries exist". When a filter has to be exhaustive —
    every entry by one author, every entry in a date range — use keyword_search
    or sql_query, which filter in the database rather than after ranking.

    Args:
        query: Natural language description or keywords describing what to find.
        max_results: Maximum number of results (1-100, default 10).
        start_date: Filter entries after this ISO-8601 date (e.g. "2024-01-15").
        end_date: Filter entries before this ISO-8601 date.
        author: Filter by author name (partial match).
        source_system: Filter by source system (exact match).
        exclude_entry_ids: Entry IDs to exclude from results (for iterative search).
        expand_query: Apply the facility vocabulary (shorthand/acronym expansion).
            None = the configured default; see capabilities().vocabulary.expand_by_default.

    Returns:
        JSON with matching entries and relevance scores, plus the vocabulary
        expansion applied. Scores order the results and are not comparable
        across queries.
    """
    if not query or not query.strip():
        return make_error(
            "validation_error",
            "Empty search query.",
            ["Provide a description or keywords for what you are looking for."],
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        parsed_start, parsed_end = parse_date_filters(start_date, end_date)
        time_range = (parsed_start, parsed_end) if parsed_start or parsed_end else None

        window = ResultWindow.build(max_results, exclude_entry_ids)

        result = await service.search(
            query,
            max_results=window.fetch_count,
            time_range=time_range,
            mode="hybrid",
            advanced_params=advanced_params(
                author=author, source_system=source_system, expand_query=expand_query
            ),
        )

        # Category-specific faults first: a rejected pattern or a cancelled
        # statement is not a sidecar outage, and _sidecar_fault matches on the
        # source alone.
        raise_on_statement_fault(result, "hybrid")

        fault = _sidecar_fault(result)
        if fault is not None:
            return make_error(
                "service_unavailable", f"ARIEL hybrid search is unavailable: {fault}", _hints()
            )

        response = success_envelope(query, "hybrid", result, window.select(result.entries))

        return json.dumps(response, default=str)

    except ToolError:
        raise
    except (PatternError, SearchTimeoutError) as exc:
        raise_for_fault_exception(exc, "hybrid")
    except VocabularyError as exc:
        raise_for_vocabulary_error(exc, "hybrid")
    except ConfigurationError as exc:
        # Two different operator states reach this handler and they need
        # different advice. The service raises with config_key
        # "search_modules.hybrid.enabled" when the mode is registered but switched
        # off, and with "modes" when the mode is not registered at all -- the
        # module never imported. Telling the second case to set the enable key
        # sends the operator to a key that is very likely already set.
        return make_error(
            "service_unavailable",
            f"ARIEL hybrid search is unavailable: {exc}",
            [
                *_configuration_hints(getattr(exc, "config_key", "")),
                "Use keyword_search or semantic_search meanwhile.",
            ],
        )
    except Exception as exc:
        logger.exception("hybrid_search failed")
        return make_error(
            "internal_error",
            f"ARIEL hybrid search failed: {exc}",
            [
                "Check ARIEL service configuration in config.yml.",
                "Verify the ARIEL database is reachable.",
            ],
        )


def _configuration_hints(config_key: str) -> list[str]:
    """Advise on a configuration refusal according to which one it was.

    Args:
        config_key: The ``config_key`` the service attached to the error.

    Returns:
        The suggestion that fits. ``search_modules.hybrid.enabled`` means the mode
        exists and is switched off, so naming the key is actionable. ``modes``
        means the mode resolved against nothing — the search module is not in
        the registry, usually because it failed to import — and there the
        enable key is very likely already set, so the useful place to look is
        the server's startup log.
    """
    if config_key == "search_modules.hybrid.enabled":
        return ["Enable it with ariel.search_modules.hybrid.enabled: true in config.yml."]
    if config_key == "modes":
        return [
            "The hybrid search module is not registered — check the ARIEL server's "
            "startup log for an import error in osprey.services.ariel_search.search.qmd."
        ]
    return ["Check the ARIEL search module configuration in config.yml."]


def _sidecar_fault(result: object) -> str | None:
    """Return the message of a qmd-mode error diagnostic, or ``None``.

    The search service catches a module's exception and returns an empty result
    carrying one ERROR diagnostic instead of propagating it, so an unreachable
    sidecar reaches this tool looking exactly like a query that matched nothing.
    Surfacing it as an error envelope is what keeps the agent from concluding
    the corpus holds no answer when in fact nothing was searched.

    Args:
        result: The service's search result.

    Returns:
        The diagnostic message, or ``None`` when the mode answered normally.
    """
    from osprey.services.ariel_search.models import DiagnosticLevel

    for diagnostic in getattr(result, "diagnostics", None) or ():
        if (
            getattr(diagnostic, "level", None) is DiagnosticLevel.ERROR
            and getattr(diagnostic, "source", "") == _QMD_DIAGNOSTIC_SOURCE
        ):
            return str(getattr(diagnostic, "message", "") or "no detail reported")
    return None


def _hints() -> list[str]:
    """Build operator suggestions naming this deployment's health endpoint.

    Args:
        None.

    Returns:
        Suggestions for the error envelope, most specific first. The exact URL
        to probe is included when a sidecar is configured, so an operator
        reading the agent's error has the command rather than a description of
        it.
    """
    base_url = None
    try:
        from osprey.deployment.qmd_service import resolve_qmd_service_config
        from osprey.utils.workspace import load_osprey_config

        qmd_config = resolve_qmd_service_config(load_osprey_config())
        base_url = qmd_config.base_url if qmd_config is not None else None
    except Exception:  # noqa: BLE001 — a config fault must not replace the real error.
        logger.debug("could not resolve services.qmd while building hybrid_search hints")

    if base_url is None:
        first = (
            "No services.qmd block is configured — add one in the build profile "
            "(profile.yml on the host), then rebuild and redeploy to get the sidecar."
        )
    else:
        first = f"Check the sidecar is answering: curl {base_url}/health"

    return [
        first,
        "Verify the qmd sidecar container is running and finished its startup index pass.",
        "Use keyword_search or semantic_search, which do not depend on the sidecar.",
    ]
