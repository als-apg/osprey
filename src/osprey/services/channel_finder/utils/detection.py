"""Pipeline configuration detection utilities."""


def detect_pipeline_config(config: dict) -> tuple[str | None, dict | None]:
    """Detect which channel finder pipeline is configured.

    Checks pipeline_mode first (explicit selection), then falls back
    to probing which pipelines have a database path configured.

    Args:
        config: Full application configuration dictionary.

    Returns:
        Tuple of (pipeline_type, db_config) or (None, None) if unconfigured.
    """
    cf_config = config.get("channel_finder", {})
    pipelines = cf_config.get("pipelines", {})

    pipeline_mode = cf_config.get("pipeline_mode")

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
