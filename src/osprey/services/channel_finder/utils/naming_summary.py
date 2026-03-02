"""Generate compact naming pattern summaries from curated channel databases.

Produces a markdown summary (~500-1000 tokens) describing PV naming
conventions, device types, and useful search examples. Used by:

1. The ``get_naming_patterns`` MCP tool at runtime
2. Template context builder at ``osprey claude regen`` time (embedded in agent prompt)
"""

from __future__ import annotations

import logging

from osprey.services.channel_finder.utils.detection import detect_pipeline_config

logger = logging.getLogger("osprey.services.channel_finder.utils.naming_summary")


def generate_naming_summary(config: dict) -> str:
    """Generate a compact naming summary from the configured curated database.

    Probes the config for pipeline configurations and loads the appropriate
    database to extract naming patterns.

    Args:
        config: Full application configuration dictionary (from config.yml).

    Returns:
        Markdown string summarizing naming conventions, or a fallback message
        if no curated database is configured.
    """
    pipeline_type, db_config = detect_pipeline_config(config)

    if pipeline_type is None or db_config is None:
        return _fallback_summary()

    try:
        if pipeline_type == "hierarchical":
            return _summary_from_hierarchical(db_config, config)
        elif pipeline_type == "middle_layer":
            return _summary_from_middle_layer(db_config, config)
        else:
            return _summary_from_flat(db_config, config)
    except Exception:
        logger.warning(
            "Could not generate naming summary from %s database",
            pipeline_type,
            exc_info=True,
        )
        return _fallback_summary()


def _summary_from_hierarchical(db_config: dict, config: dict) -> str:
    """Generate summary from a hierarchical channel database."""
    db_path = db_config.get("path", "")
    if not db_path:
        return _fallback_summary()

    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    db = HierarchicalChannelDatabase(db_path)
    levels = db.hierarchy_levels
    pattern = db.naming_pattern

    lines = [
        "## PV Naming Conventions",
        "",
        f"**Naming pattern**: `{pattern}`" if pattern else "",
        "",
        "**Hierarchy levels**:",
    ]
    for level in levels:
        lines.append(f"- `{level}`")

    # Extract some sample options from the first level
    try:
        first_options = db.get_options(levels[0], {})
        if first_options:
            sample = first_options[:8]
            lines.append("")
            lines.append(f"**Example `{levels[0]}` values**: {', '.join(sample)}")
    except Exception:
        pass

    lines.append("")
    lines.append("**Search tip**: Use the hierarchy level names as pattern segments.")
    lines.append(
        "For example, if the pattern is `{area}:{device}:{signal}`, "
        "search with `SR:BPM:*` to find all BPM signals in the storage ring."
    )

    return "\n".join(line for line in lines if line is not None)


def _summary_from_middle_layer(db_config: dict, config: dict) -> str:
    """Generate summary from a middle-layer channel database."""
    db_path = db_config.get("path", "")
    if not db_path:
        return _fallback_summary()

    from osprey.services.channel_finder.databases.middle_layer import (
        MiddleLayerDatabase,
    )

    db = MiddleLayerDatabase(db_path)
    systems = db.list_systems()

    lines = [
        "## PV Naming Conventions",
        "",
        "**Structure**: MML functional hierarchy (System → Family → Field → Channel)",
        "",
        f"**Systems**: {', '.join(systems[:10])}",
    ]

    # Get families for the first system
    if systems:
        try:
            families = db.list_families(systems[0])
            if families:
                lines.append(f"**Example families** (in {systems[0]}): {', '.join(families[:8])}")
        except Exception:
            pass

    lines.append("")
    lines.append(
        "**Common device types**: BPM (position), HCM/VCM (correctors), "
        "QF/QD (quadrupoles), DCCT (beam current)"
    )
    lines.append(
        "**Search tip**: PV names typically follow `{System}:{Family}:{Field}:{Instance}` patterns."
    )

    return "\n".join(lines)


def _summary_from_flat(db_config: dict, config: dict) -> str:
    """Generate summary from a flat or template channel database."""
    db_path = db_config.get("path", "")
    if not db_path:
        return _fallback_summary()

    db_type = db_config.get("type", "template")
    if db_type == "template":
        from osprey.services.channel_finder.databases.template import ChannelDatabase
    else:
        from osprey.services.channel_finder.databases.flat import ChannelDatabase

    db = ChannelDatabase(db_path)
    channels = db.get_all_channels()

    # Infer patterns from channel names
    prefixes: dict[str, int] = {}
    for ch in channels[:200]:
        name = ch.get("name", ch) if isinstance(ch, dict) else str(ch)
        parts = name.split(":")
        if len(parts) >= 2:
            prefix = f"{parts[0]}:{parts[1]}"
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

    sorted_prefixes = sorted(prefixes.items(), key=lambda x: -x[1])[:10]

    lines = [
        "## PV Naming Conventions",
        "",
        f"**Total channels**: {len(channels)}",
        "",
        "**Common prefixes**:",
    ]
    for prefix, count in sorted_prefixes:
        lines.append(f"- `{prefix}:*` ({count} channels)")

    lines.append("")
    lines.append(
        "**Search tip**: Use the common prefixes above as starting points "
        "for glob pattern searches."
    )

    return "\n".join(lines)


def _fallback_summary() -> str:
    """Return a generic summary when no curated database is available."""
    return (
        "## PV Naming Conventions\n\n"
        "No curated channel database is configured. Use `search_pvs` with broad\n"
        "glob patterns (e.g., `*BPM*`, `SR:*`) to discover the naming structure.\n\n"
        "**Common accelerator PV patterns**:\n"
        "- `{Area}:{DeviceType}:{Instance}:{Signal}` (e.g., `SR:BPM:01:X`)\n"
        "- `{System}:{Subsystem}:{Parameter}` (e.g., `SR:RF:Frequency`)\n"
    )
