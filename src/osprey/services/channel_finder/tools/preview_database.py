"""
Preview Database Presentation

Shows how the channel database will be presented to the LLM.
Auto-detects pipeline type (hierarchical vs in_context vs middle_layer)
and displays accordingly.
"""

import json
import random
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from osprey.services.channel_finder.databases import (
    HierarchicalChannelDatabase,
    MiddleLayerDatabase,
    TemplateChannelDatabase,
)


def _resolve_path(path_str: str) -> Path:
    """Resolve a database path. Absolute paths returned as-is; relative paths resolved via config."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    try:
        from osprey.services.channel_finder.utils.config import resolve_path

        return resolve_path(path_str)
    except Exception:
        return Path.cwd() / path


try:
    from osprey.cli.styles import console as osprey_console
    from osprey.cli.styles import get_active_theme

    console = osprey_console
    theme = get_active_theme()
except ImportError:
    console = Console()
    theme = None


def detect_pipeline_config(config) -> tuple[str | None, dict | None]:
    """Detect which pipeline is configured.

    Returns:
        tuple: (pipeline_type, db_config)
    """
    cf_config = config.get("channel_finder", {})
    pipelines = cf_config.get("pipelines", {})

    pipeline_mode = cf_config.get("pipeline_mode")

    hierarchical_config = pipelines.get("hierarchical", {})
    in_context_config = pipelines.get("in_context", {})
    middle_layer_config = pipelines.get("middle_layer", {})

    if pipeline_mode == "in_context" and in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    elif pipeline_mode == "hierarchical" and hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})
    elif pipeline_mode == "middle_layer" and middle_layer_config.get("database", {}).get("path"):
        return "middle_layer", middle_layer_config.get("database", {})

    if middle_layer_config.get("database", {}).get("path"):
        return "middle_layer", middle_layer_config.get("database", {})
    elif hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})
    elif in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    else:
        return None, None


# ============================================================================
# Middle Layer Preview
# ============================================================================


def preview_middle_layer(
    db_path: str,
    depth: int = 3,
    max_items: int = 3,
    sections: list = None,
    focus: str = None,
    show_full: bool = False,
) -> None:
    """Preview middle layer database with tree structure."""
    if sections is None:
        sections = ["tree"]

    console.print()
    console.print(
        Panel.fit(
            "[bold primary]Middle Layer Database Preview[/bold primary]\n"
            "[dim]Shows the functional hierarchy (System \u2192 Family \u2192 Field \u2192 Channels)[/dim]",
            border_style="primary",
            padding=(1, 2),
        )
    )

    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_column("Property", style="label")
    config_table.add_column("Value", style="value")

    resolved_path = _resolve_path(db_path)
    config_table.add_row("Database Path", db_path)
    config_table.add_row("Database Type", "middle_layer")
    config_table.add_row("Resolved Path", str(resolved_path))

    console.print()
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="info"))

    with console.status("[bold info]Loading database...", spinner="dots"):
        database = MiddleLayerDatabase(str(resolved_path))
        all_channels = database.get_all_channels()

    total_channels = len(all_channels)
    console.print(
        f"\n[success]\u2713 Successfully loaded [bold]{total_channels:,}[/bold] channels[/success]\n"
    )

    if "stats" in sections or "tree" in sections:
        systems = set()
        families = set()
        fields = set()

        for channel in all_channels:
            if "system" in channel:
                systems.add(channel["system"])
            if "family" in channel:
                families.add(channel["family"])
            if "field" in channel:
                fields.add(channel["field"])

        stats_table = Table(show_header=True, box=box.ROUNDED, padding=(0, 2))
        stats_table.add_column("Metric", style="label", no_wrap=True)
        stats_table.add_column("Count", justify="right", style="value")

        stats_table.add_row("Total Channels", f"{total_channels:,}")
        stats_table.add_row("Systems", f"{len(systems)}")
        stats_table.add_row("Families", f"{len(families)}")
        stats_table.add_row("Fields", f"{len(fields)}")

        console.print(
            Panel(
                stats_table,
                title="[bold]Database Statistics[/bold]",
                border_style="accent",
                padding=(1, 2),
            )
        )

    if "tree" in sections:
        _render_middle_layer_tree(
            database, depth if not show_full else -1, max_items if not show_full else -1, focus
        )

    if "samples" in sections:
        _render_samples_section(database)

    console.print()
    console.print(
        Panel.fit(
            f"[success]\u2713 Preview complete! [bold]{total_channels:,}[/bold] total channels[/success]",
            border_style="success",
        )
    )
    console.print()


def _render_middle_layer_tree(database, depth, max_items, focus) -> None:
    """Render the middle layer hierarchy tree."""
    console.print()

    tree_data = _build_middle_layer_tree(database)

    if focus:
        focus_parts = focus.split(":")
        tree_root, tree_title = _navigate_middle_layer_focus(tree_data, focus_parts)
        if tree_root is None:
            console.print(f"[error]\u2717 Focus path '{focus}' not found in database[/error]\n")
            return
    else:
        tree_root = tree_data
        tree_title = "[bold primary]Middle Layer Hierarchy[/bold primary]"

    tree = Tree(tree_title, guide_style="info")

    max_depth = depth if depth > 0 else 10
    max_items_val = None if max_items == -1 else max_items

    _add_middle_layer_nodes(tree, tree_root, 0, max_depth, max_items_val)

    console.print(
        Panel(tree, title="[bold]Hierarchy Tree[/bold]", border_style="primary", padding=(1, 2))
    )

    if depth > 0 or max_items > 0:
        console.print()
        console.print(
            "[info]\U0001f4a1 Tip:[/info] Use [bold]--full[/bold] flag to see complete hierarchy"
        )


def _build_middle_layer_tree(database) -> dict:
    """Build a hierarchical tree structure from middle layer database."""
    tree = {}

    descriptions = {}
    for system_name, system_data in database.data.items():
        if not isinstance(system_data, dict):
            continue

        system_desc = system_data.get("_description", "")
        descriptions[system_name] = {"_description": system_desc, "families": {}}

        for family_name, family_data in system_data.items():
            if not isinstance(family_data, dict) or family_name.startswith("_"):
                continue

            family_desc = family_data.get("_description", "")
            descriptions[system_name]["families"][family_name] = family_desc

    for channel in database.get_all_channels():
        system = channel.get("system", "Unknown")
        family = channel.get("family", "Unknown")
        field = channel.get("field", "Unknown")

        if system not in tree:
            tree[system] = {
                "_channels": 0,
                "_families": {},
                "_description": descriptions.get(system, {}).get("_description", ""),
            }

        tree[system]["_channels"] += 1

        if family not in tree[system]["_families"]:
            tree[system]["_families"][family] = {
                "_channels": 0,
                "_fields": {},
                "_description": descriptions.get(system, {}).get("families", {}).get(family, ""),
            }

        tree[system]["_families"][family]["_channels"] += 1

        if field not in tree[system]["_families"][family]["_fields"]:
            tree[system]["_families"][family]["_fields"][field] = 0

        tree[system]["_families"][family]["_fields"][field] += 1

    return tree


def _navigate_middle_layer_focus(tree, focus_parts) -> tuple[dict | None, str | None]:
    """Navigate to focus in middle layer tree."""
    current = tree
    title_parts = []

    for i, part in enumerate(focus_parts):
        title_parts.append(part)
        if i == 0:
            if part not in current:
                return None, None
            current = {"_families": current[part].get("_families", {})}
        elif i == 1:
            if part not in current.get("_families", {}):
                return None, None
            current = {"_fields": current["_families"][part].get("_fields", {})}
        else:
            return None, None

    title = f"[bold primary]{':'.join(title_parts)}[/bold primary]"
    return current, title


def _add_middle_layer_nodes(parent, data, level, max_depth, max_items) -> None:
    """Recursively add middle layer nodes to tree."""
    if level >= max_depth:
        return

    if level == 0:
        items = {k: v for k, v in data.items() if not k.startswith("_")}
        item_type = "system"
    elif "_families" in data:
        items = data["_families"]
        item_type = "family"
    elif "_fields" in data:
        items = data["_fields"]
        item_type = "field"
    else:
        return

    count = 0
    for name, node_data in sorted(items.items()):
        if max_items and count >= max_items:
            remaining = len(items) - max_items
            parent.add(f"[dim]... {remaining} more {item_type}s[/dim]")
            break

        count += 1

        if isinstance(node_data, dict):
            ch_count = node_data.get("_channels", 0)
        else:
            ch_count = node_data

        if level == 0:
            style = "cyan bold"
            desc_text = node_data.get("_description", "") if isinstance(node_data, dict) else ""
            if len(desc_text) > 80:
                desc_text = desc_text[:77] + "..."
            desc = f" [dim]-[/dim] {desc_text}" if desc_text else ""
        elif level == 1:
            style = "yellow"
            desc_text = node_data.get("_description", "") if isinstance(node_data, dict) else ""
            if len(desc_text) > 80:
                desc_text = desc_text[:77] + "..."
            desc = f" [dim]-[/dim] {desc_text}" if desc_text else ""
        else:
            style = "green"
            desc = ""

        label = f"[{style}]{name}[/{style}] [dim]({ch_count} channels)[/dim]{desc}"
        branch = parent.add(label)

        if isinstance(node_data, dict) and level + 1 < max_depth:
            _add_middle_layer_nodes(branch, node_data, level + 1, max_depth, max_items)


# ============================================================================
# Hierarchical Preview
# ============================================================================


def preview_hierarchical(
    db_path: str,
    depth: int = 3,
    max_items: int = 3,
    sections: list = None,
    focus: str = None,
) -> None:
    """Preview hierarchical database with tree structure."""
    if sections is None:
        sections = ["tree"]

    console.print()
    console.print(
        Panel.fit(
            "[bold primary]Hierarchical Database Preview[/bold primary]\n"
            "[dim]Shows the tree structure of the hierarchical channel database[/dim]",
            border_style="primary",
            padding=(1, 2),
        )
    )

    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_column("Property", style="label")
    config_table.add_column("Value", style="value")

    resolved_path = _resolve_path(db_path)
    config_table.add_row("Database Path", db_path)
    config_table.add_row("Resolved Path", str(resolved_path))

    with console.status("[bold info]Loading database...", spinner="dots"):
        database = HierarchicalChannelDatabase(str(resolved_path))
        stats = database.get_statistics()
        hierarchy_levels = database.hierarchy_levels

    config_table.add_row("Hierarchy Levels", " \u2192 ".join(hierarchy_levels))

    depth_str = "unlimited" if depth == -1 else str(depth)
    max_items_str = "unlimited" if max_items == -1 else str(max_items)
    config_table.add_row("Display Depth", depth_str)
    config_table.add_row("Max Items/Level", max_items_str)
    if focus:
        config_table.add_row("Focus Path", focus)

    console.print()
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="info"))

    total_channels = stats.get("total_channels", 0)
    console.print(
        f"\n[success]\u2713 Successfully loaded [bold]{total_channels}[/bold] channels[/success]\n"
    )

    if "tree" in sections:
        _render_tree_section(database, hierarchy_levels, depth, max_items, focus)

    if "stats" in sections:
        _render_stats_section(database, hierarchy_levels)

    if "breakdown" in sections:
        _render_breakdown_section(database, hierarchy_levels, focus)

    if "samples" in sections:
        _render_samples_section(database)

    console.print()
    footer_msg = f"[success]\u2713 Preview complete! [bold]{total_channels}[/bold] total channels"
    if focus:
        footer_msg += f" (focused on: {focus})"
    footer_msg += "[/success]"

    console.print(Panel.fit(footer_msg, border_style="success"))
    console.print()


def _render_tree_section(database, hierarchy_levels, depth, max_items, focus):
    """Render the hierarchy tree section."""
    console.print()

    db_tree = database.tree

    if focus:
        focus_parts = focus.split(":")
        tree_root, start_level = _navigate_to_focus(
            db_tree, focus_parts, hierarchy_levels, database
        )
        if tree_root is None:
            console.print(f"[error]\u2717 Focus path '{focus}' not found in database[/error]\n")
            return
        tree_title = f"[bold primary]{focus}[/bold primary]"
    else:
        tree_root = db_tree
        start_level = 0
        tree_title = "[bold primary]Channel Database Hierarchy[/bold primary]"

    tree = Tree(tree_title, guide_style="info")

    max_depth = depth if depth > 0 else len(hierarchy_levels)
    max_items_val = None if max_items == -1 else max_items

    item_count = 0
    for node_name, node_data in tree_root.items():
        if node_name.startswith("_"):
            continue

        if max_items_val and item_count >= max_items_val:
            remaining = len([k for k in tree_root if not k.startswith("_")]) - max_items_val
            tree.add(f"[dim]... {remaining} more {hierarchy_levels[start_level]}s[/dim]")
            break

        item_count += 1

        if start_level == 0:
            node_channels = [
                ch
                for ch in database.channel_map.values()
                if ch.get("path", {}).get(hierarchy_levels[0]) == node_name
            ]
        else:
            node_channels = _count_channels_matching_focus(
                database, focus_parts + [node_name], hierarchy_levels
            )

        node_branch = tree.add(
            f"[cyan bold]{node_name}[/cyan bold] [dim]({len(node_channels)} channels)[/dim]"
        )

        if start_level + 1 < max_depth and isinstance(node_data, dict):
            _add_hierarchy_level_new(
                node_branch,
                node_data,
                database,
                hierarchy_levels,
                start_level + 1,
                max_items_val,
                max_depth,
                start_level,
                [node_name],
            )

    console.print(
        Panel(tree, title="[bold]Hierarchy Tree[/bold]", border_style="primary", padding=(1, 2))
    )

    if depth > 0 or max_items > 0:
        console.print()
        console.print(
            "[info]\U0001f4a1 Tip:[/info] Use [bold]--depth -1 --max-items -1[/bold] to see complete hierarchy"
        )


def _render_stats_section(database, hierarchy_levels):
    """Render the level statistics section."""
    console.print()

    level_stats = _calculate_level_statistics(database, hierarchy_levels)

    stats_table = Table(show_header=True, box=box.ROUNDED, padding=(0, 2))
    stats_table.add_column("Level", style="label", no_wrap=True)
    stats_table.add_column("Name", style="accent", no_wrap=True)
    stats_table.add_column("Unique Values", justify="right", style="value")

    for idx, (level_name, unique_count) in enumerate(level_stats, 1):
        stats_table.add_row(str(idx), level_name, f"{unique_count:,}")

    console.print(
        Panel(
            stats_table,
            title="[bold]Hierarchy Level Statistics[/bold]",
            border_style="accent",
            padding=(1, 2),
        )
    )


def _render_breakdown_section(database, hierarchy_levels, focus):
    """Render channel count breakdown by path."""
    console.print()

    breakdown = _calculate_breakdown(database, hierarchy_levels, focus)

    breakdown_table = Table(show_header=True, box=box.ROUNDED, padding=(0, 2))
    breakdown_table.add_column("Path", style="label", no_wrap=False)
    breakdown_table.add_column("Channels", justify="right", style="value")

    for path, count in breakdown[:20]:
        breakdown_table.add_row(path, f"{count:,}")

    if len(breakdown) > 20:
        breakdown_table.add_row("[dim]...[/dim]", f"[dim]{len(breakdown) - 20} more paths[/dim]")

    console.print(
        Panel(
            breakdown_table,
            title="[bold]Channel Count Breakdown[/bold]",
            border_style="accent",
            padding=(1, 2),
        )
    )


def _render_samples_section(database, num_samples=5):
    """Render sample channel names."""
    console.print()

    all_channels = database.get_all_channels()

    if len(all_channels) <= num_samples:
        samples = all_channels
    else:
        samples = random.sample(all_channels, num_samples)

    sample_text = "\n".join([f"  {ch['channel']}" for ch in samples])
    sample_text += (
        f"\n\n[dim]({len(samples)} random samples from {len(all_channels):,} total channels)[/dim]"
    )

    console.print(
        Panel(
            sample_text,
            title="[bold]Sample Channels[/bold]",
            border_style="accent",
            padding=(1, 2),
        )
    )


def _navigate_to_focus(tree, focus_parts, hierarchy_levels, database):
    """Navigate to a focus path in the tree.

    Returns:
        tuple: (subtree_dict, start_level_index) or (None, None) if not found
    """
    current = tree
    level_idx = 0

    for part in focus_parts:
        if not isinstance(current, dict) or part not in current:
            return None, None
        current = current[part]
        level_idx += 1

    return current, level_idx


def _count_channels_matching_focus(database, path_parts, hierarchy_levels):
    """Count channels matching a focus path."""
    matching = []
    for _ch_name, ch_data in database.channel_map.items():
        ch_path = ch_data.get("path", {})
        matches = True
        for idx, part in enumerate(path_parts):
            if idx < len(hierarchy_levels):
                if ch_path.get(hierarchy_levels[idx]) != part:
                    matches = False
                    break
        if matches:
            matching.append(ch_data)
    return matching


def _calculate_level_statistics(database, hierarchy_levels):
    """Calculate unique value counts for each hierarchy level."""
    level_counts = []

    for level_name in hierarchy_levels:
        unique_values = set()
        for ch_data in database.channel_map.values():
            path = ch_data.get("path", {})
            if level_name in path:
                unique_values.add(path[level_name])
        level_counts.append((level_name, len(unique_values)))

    return level_counts


def _calculate_breakdown(database, hierarchy_levels, focus):
    """Calculate channel count breakdown by path."""
    path_counts = {}

    for ch_data in database.channel_map.values():
        path = ch_data.get("path", {})

        for depth in range(1, len(hierarchy_levels) + 1):
            parts = []
            for i in range(depth):
                if i < len(hierarchy_levels):
                    level_name = hierarchy_levels[i]
                    if level_name in path:
                        parts.append(path[level_name])

            if parts:
                path_str = ":".join(parts)
                path_counts[path_str] = path_counts.get(path_str, 0) + 1

    sorted_breakdown = sorted(path_counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_breakdown


def _add_hierarchy_level_new(
    parent_branch,
    data,
    database,
    hierarchy_levels,
    level_idx,
    max_items,
    max_depth,
    start_level,
    parent_path=None,
):
    """Recursively add hierarchy levels to the tree (with depth limit)."""
    if level_idx >= len(hierarchy_levels) or level_idx >= max_depth:
        return

    current_level = hierarchy_levels[level_idx]
    children = _get_children_at_level(data, current_level, hierarchy_levels, level_idx)

    if not children:
        return

    branch_count = 0
    for child_name, child_data in children.items():
        if max_items and branch_count >= max_items:
            parent_branch.add(f"[dim]... {len(children) - max_items} more {current_level}s[/dim]")
            break

        branch_count += 1

        child_path = (parent_path or []) + [child_name]

        child_channel_count = _count_channels_at_path(
            database, hierarchy_levels, child_path, level_idx
        )

        if level_idx == max_depth - 1 or level_idx == len(hierarchy_levels) - 1:
            child_branch = parent_branch.add(
                f"[green]{child_name}[/green] [dim]({child_channel_count} channels)[/dim]"
            )
        else:
            child_branch = parent_branch.add(
                f"[yellow]{child_name}[/yellow] [dim]({child_channel_count} channels)[/dim]"
            )

        if level_idx + 1 < max_depth:
            _add_hierarchy_level_new(
                child_branch,
                child_data,
                database,
                hierarchy_levels,
                level_idx + 1,
                max_items,
                max_depth,
                start_level,
                child_path,
            )


def _get_children_at_level(data, current_level, hierarchy_levels, level_idx):
    """Get children at a specific hierarchy level."""
    if not isinstance(data, dict):
        return {}

    if "_expansion" in data:
        expansion = data["_expansion"]
        exp_type = expansion.get("_type")

        if exp_type == "range":
            pattern = expansion.get("_pattern", "{}")
            start, end = expansion.get("_range", [1, 1])
            return {pattern.format(i): data for i in range(start, end + 1)}
        elif exp_type == "list":
            instances = expansion.get("_instances", [])
            return dict.fromkeys(instances, data)

    if current_level == "device" and "devices" in data:
        device_config = data["devices"]
        device_type = device_config.get("_type")

        if device_type == "range":
            pattern = device_config.get("_pattern", "{}")
            start, end = device_config.get("_range", [1, 1])
            return {pattern.format(i): data for i in range(start, end + 1)}
        elif device_type == "list":
            instances = device_config.get("_instances", [])
            return dict.fromkeys(instances, data)

    if current_level == "field" and "fields" in data:
        return {
            k: v for k, v in data["fields"].items() if not k.startswith("_") and isinstance(v, dict)
        }

    if current_level == "subfield" and "subfields" in data:
        return {
            k: v
            for k, v in data["subfields"].items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def _count_channels_at_path(database, hierarchy_levels, path_values, current_level_idx):
    """Count channels matching a specific path through the hierarchy."""
    count = 0
    for ch_data in database.channel_map.values():
        ch_path = ch_data.get("path", {})
        matches = True

        for idx, value in enumerate(path_values):
            if idx <= current_level_idx and idx < len(hierarchy_levels):
                level_name = hierarchy_levels[idx]
                if ch_path.get(level_name) != value:
                    matches = False
                    break

        if matches:
            count += 1

    return count


# ============================================================================
# In-Context Preview
# ============================================================================


def preview_in_context(db_path: str, presentation_mode: str, show_full: bool = False):
    """Preview in-context database with formatted channel list."""

    console.print()
    console.print(
        Panel.fit(
            "[bold primary]In-Context Database Preview[/bold primary]\n"
            "[dim]Shows how the channel database will be presented to the LLM[/dim]",
            border_style="primary",
            padding=(1, 2),
        )
    )

    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_column("Property", style="label")
    config_table.add_column("Value", style="value")

    config_table.add_row("Database Path", db_path)
    config_table.add_row("Presentation Mode", f"[success]{presentation_mode}[/success]")

    resolved_path = _resolve_path(db_path)
    config_table.add_row("Resolved Path", str(resolved_path))

    console.print()
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="info"))

    with console.status("[bold info]Loading database...", spinner="dots"):
        database = TemplateChannelDatabase(str(resolved_path), presentation_mode=presentation_mode)
        all_channels = database.get_all_channels()
        stats = database.get_statistics()

    console.print(
        f"\n[success]\u2713 Successfully loaded [bold]{len(all_channels)}[/bold] channels[/success]\n"
    )

    stats_table = Table(show_header=True, box=box.ROUNDED, padding=(0, 2))
    stats_table.add_column("Metric", style="label", no_wrap=True)
    stats_table.add_column("Count", justify="right", style="value")

    stats_table.add_row("Total Channels", str(len(all_channels)))
    if stats:
        template_entries = stats.get("template_entries", 0)
        standalone_entries = stats.get("standalone_entries", 0)
        if template_entries > 0 or standalone_entries > 0:
            stats_table.add_row("Template Entries", str(template_entries))
            stats_table.add_row("Standalone Entries", str(standalone_entries))

    console.print(
        Panel(stats_table, title="[bold]Database Statistics[/bold]", border_style="accent")
    )

    console.print()
    if show_full:
        title = f"[bold]LLM Presentation[/bold] [dim](all {len(all_channels)} channels)[/dim]"
    else:
        title = "[bold]LLM Presentation[/bold] [dim](first 20 channels)[/dim]"

    if show_full:
        sample_channels = all_channels
    else:
        sample_channels = all_channels[:20]

    formatted = database.format_chunk_for_prompt(sample_channels, include_addresses=False)

    console.print(Panel(formatted, title=title, border_style="primary", padding=(1, 2)))

    if not show_full and len(all_channels) > 20:
        console.print()
        console.print(
            f"[dim]... {len(all_channels) - 20} more channels not shown[/dim]\n"
            "[info]\U0001f4a1 Tip:[/info] Use [bold]--full[/bold] flag to see all channels"
        )

    console.print()
    console.print(
        Panel.fit(
            f"[success]\u2713 Preview complete! [bold]{len(all_channels)}[/bold] total channels in database[/success]",
            border_style="success",
        )
    )
    console.print()


# ============================================================================
# Main Entry Point
# ============================================================================


def preview_database(
    depth: int = 3,
    max_items: int = 3,
    sections: str = "tree",
    focus: str = None,
    show_full: bool = False,
    db_path: str = None,
) -> None:
    """Preview database based on configured pipeline type.

    Args:
        depth: Maximum depth to display (default: 3, -1 for unlimited)
        max_items: Maximum items per level (default: 3, -1 for unlimited)
        sections: Comma-separated list of sections to display
        focus: Path to focus on (hierarchical/middle_layer only)
        show_full: Sets depth and max_items to -1 for complete view
        db_path: Direct path to database file (overrides config)
    """

    if db_path:
        resolved_path = _resolve_path(db_path)

        try:
            with open(resolved_path) as f:
                data = json.load(f)

            if "hierarchy" in data or "tree" in data:
                pipeline_type = "hierarchical"
                db_config = {}
            elif isinstance(data, dict) and any(
                key in data for key in ["SR", "BR", "BTS", "VAC", "Scraper"]
            ):
                pipeline_type = "middle_layer"
                db_config = {}
            else:
                pipeline_type = "in_context"
                db_config = {}
        except Exception as e:
            console.print(f"[error]\u2717 Error loading database from {db_path}: {e}[/error]")
            return
    else:
        from osprey.services.channel_finder.utils.config import get_config

        config = get_config()
        pipeline_type, db_config = detect_pipeline_config(config)

        if not pipeline_type:
            console.print()
            console.print(
                Panel(
                    "[bold error]Error:[/bold error] No database configured\n\n"
                    "[warning]Check config.yml:[/warning] Configure either:\n"
                    "  \u2022 channel_finder.pipelines.middle_layer.database.path\n"
                    "  \u2022 channel_finder.pipelines.hierarchical.database.path\n"
                    "  \u2022 channel_finder.pipelines.in_context.database.path\n\n"
                    "Or use [bold]--database[/bold] to specify a database file directly",
                    border_style="error",
                    title="\u274c Configuration Error",
                )
            )
            return

        db_path = db_config.get("path")

    # Handle --full flag (backwards compatibility)
    if show_full:
        depth = -1
        max_items = -1

    # Parse sections
    section_list = [s.strip() for s in sections.split(",")]
    if "all" in section_list:
        section_list = ["tree", "stats", "breakdown", "samples"]

    if pipeline_type == "hierarchical":
        preview_hierarchical(
            db_path=db_path, depth=depth, max_items=max_items, sections=section_list, focus=focus
        )
    elif pipeline_type == "middle_layer":
        preview_middle_layer(
            db_path=db_path,
            depth=depth,
            max_items=max_items,
            sections=section_list,
            focus=focus,
            show_full=show_full,
        )
    else:
        presentation_mode = db_config.get("presentation_mode", "template")
        preview_in_context(db_path, presentation_mode, show_full)
