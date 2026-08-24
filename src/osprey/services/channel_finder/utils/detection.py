"""Pipeline configuration detection utilities."""

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.services.channel_finder.core.exceptions import PipelineModeError


def detect_pipeline_config(config: dict) -> tuple[str | None, dict | None]:
    """Detect which channel finder pipeline is configured.

    Checks pipeline_mode first (explicit selection), then falls back
    to probing which pipelines have a database path configured.

    The ``graph`` paradigm is answered on the mode alone: its store is a graph
    database reached over the network, so there is no ``database.path`` to
    probe and detection returns ``("graph", None)`` --- callers read the
    connection details from the ``graph`` config block instead. That also means
    auto-detection never selects graph: with no ``pipeline_mode`` set,
    detection only ever picks among the paradigms backed by a database file.

    An explicit ``pipeline_mode`` naming a paradigm that does not exist is
    rejected outright rather than quietly falling through to auto-detection,
    which would otherwise hand back whichever *other* pipeline happens to have
    a database path. A known mode whose own database path is unset keeps the
    auto-detect fallback.

    Args:
        config: Full application configuration dictionary.

    Returns:
        Tuple of (pipeline_type, db_config), ``("graph", None)`` for the graph
        paradigm, or (None, None) if unconfigured.

    Raises:
        PipelineModeError: ``channel_finder.pipeline_mode`` is set to a value
            that is not a known channel-finder paradigm.
    """
    cf_config = config.get("channel_finder", {})
    pipelines = cf_config.get("pipelines", {})

    pipeline_mode = cf_config.get("pipeline_mode")

    # Graph has no database file to probe, so the mode alone selects it.
    if pipeline_mode == "graph":
        return "graph", None

    if pipeline_mode is not None and pipeline_mode not in VALID_CHANNEL_FINDER_MODES:
        raise PipelineModeError(
            f"Unknown channel_finder.pipeline_mode: {pipeline_mode!r}. "
            f"Valid modes are: {', '.join(VALID_CHANNEL_FINDER_MODES)}."
        )

    hierarchical_config = pipelines.get("hierarchical", {})
    in_context_config = pipelines.get("in_context", {})
    middle_layer_config = pipelines.get("middle_layer", {})

    # Explicit pipeline_mode takes priority
    if pipeline_mode == "in_context" and in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    elif pipeline_mode == "hierarchical" and hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})
    elif pipeline_mode == "middle_layer" and middle_layer_config.get("database", {}).get("path"):
        return "middle_layer", middle_layer_config.get("database", {})

    # Auto-detect from available pipeline configs
    if middle_layer_config.get("database", {}).get("path"):
        return "middle_layer", middle_layer_config.get("database", {})
    elif hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})
    elif in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    else:
        return None, None
