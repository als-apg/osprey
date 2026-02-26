"""
Database Validation Tool

Validates channel database JSON files for correctness and compatibility with the system.
Auto-detects pipeline type (hierarchical vs in_context) and validates accordingly.
"""

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from osprey.services.channel_finder.databases import (
    HierarchicalChannelDatabase,
    TemplateChannelDatabase,
)

try:
    from osprey.cli.styles import console as osprey_console
    from osprey.cli.styles import get_active_theme

    console = osprey_console
    theme = get_active_theme()
except ImportError:
    console = Console()
    theme = None


def detect_pipeline_config(config):
    """Detect which pipeline is configured.

    Returns:
        tuple: (pipeline_type, db_config) where pipeline_type is 'hierarchical' or 'in_context'
    """
    cf_config = config.get("channel_finder", {})
    pipelines = cf_config.get("pipelines", {})

    pipeline_mode = cf_config.get("pipeline_mode")

    hierarchical_config = pipelines.get("hierarchical", {})
    in_context_config = pipelines.get("in_context", {})

    if pipeline_mode == "in_context" and in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    elif pipeline_mode == "hierarchical" and hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})

    if hierarchical_config.get("database", {}).get("path"):
        return "hierarchical", hierarchical_config.get("database", {})
    elif in_context_config.get("database", {}).get("path"):
        return "in_context", in_context_config.get("database", {})
    else:
        return None, None


def validate_json_structure(db_path: Path) -> tuple[bool, list[str], list[str]]:
    """Validate JSON file structure and schema.

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    if not db_path.exists():
        errors.append(f"Database file not found: {db_path}")
        return False, errors, warnings

    try:
        with open(db_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {e}")
        return False, errors, warnings
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors, warnings

    if isinstance(data, list):
        warnings.append("Using legacy array format. Consider using dict format with metadata.")
        channels = data
    elif isinstance(data, dict):
        if "channels" not in data:
            errors.append("Missing 'channels' key in database dict")
            return False, errors, warnings
        channels = data["channels"]

        if "presentation_mode" in data:
            valid_modes = ["explicit", "template"]
            if data["presentation_mode"] not in valid_modes:
                warnings.append(
                    f"Unknown presentation_mode: {data['presentation_mode']}. Valid: {valid_modes}"
                )
    else:
        errors.append(f"Invalid top-level type: {type(data)}. Expected list or dict.")
        return False, errors, warnings

    if not isinstance(channels, list):
        errors.append(f"'channels' must be a list, got {type(channels)}")
        return False, errors, warnings

    if len(channels) == 0:
        errors.append("Database contains no channels")
        return False, errors, warnings

    for i, entry in enumerate(channels):
        if not isinstance(entry, dict):
            errors.append(f"Channel {i}: must be a dict, got {type(entry)}")
            continue

        is_template = entry.get("template", False)

        if is_template:
            required = ["base_name", "instances", "description"]
            for field in required:
                if field not in entry:
                    errors.append(f"Template {i}: missing required field '{field}'")

            if "instances" in entry:
                instances = entry["instances"]
                if not isinstance(instances, list) or len(instances) != 2:
                    errors.append(
                        f"Template {i}: 'instances' must be [start, end], got {instances}"
                    )
                elif instances[0] > instances[1]:
                    errors.append(
                        f"Template {i}: instance start ({instances[0]}) > end ({instances[1]})"
                    )

            if "sub_channels" in entry:
                if not isinstance(entry["sub_channels"], list):
                    errors.append(f"Template {i}: 'sub_channels' must be a list")
                elif len(entry["sub_channels"]) == 0:
                    warnings.append(f"Template {i}: 'sub_channels' is empty")

            if "axes" in entry:
                if not isinstance(entry["axes"], list):
                    errors.append(f"Template {i}: 'axes' must be a list")

            if "address_pattern" not in entry:
                warnings.append(
                    f"Template {i}: missing 'address_pattern'. Will use default pattern."
                )

            if "channel_descriptions" not in entry:
                warnings.append(
                    f"Template {i}: missing 'channel_descriptions'. Will use generic descriptions."
                )
        else:
            required = ["channel", "address", "description"]
            for field in required:
                if field not in entry:
                    errors.append(f"Channel {i}: missing required field '{field}'")

            for field in ["channel", "address", "description"]:
                if field in entry and not entry[field]:
                    warnings.append(f"Channel {i}: field '{field}' is empty")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_database_loading(db_path: Path, pipeline_type: str) -> tuple[bool, list[str], dict]:
    """Test loading database through the actual database class.

    Args:
        db_path: Path to database file
        pipeline_type: Either 'hierarchical' or 'in_context'

    Returns:
        (success, errors, stats)
    """
    errors = []
    stats = {}

    try:
        if pipeline_type == "hierarchical":
            db = HierarchicalChannelDatabase(str(db_path))
        else:
            db = TemplateChannelDatabase(str(db_path), presentation_mode="explicit")

        stats = db.get_statistics()

        all_channels = db.get_all_channels()
        if not all_channels:
            errors.append("Database loaded but get_all_channels() returned empty list")

        if all_channels:
            first_channel = all_channels[0]["channel"]
            lookup_result = db.get_channel(first_channel)
            if not lookup_result:
                errors.append(f"Channel lookup failed for: {first_channel}")

        return True, errors, stats

    except Exception as e:
        errors.append(f"Failed to load database: {e}")
        import traceback

        errors.append(traceback.format_exc())
        return False, errors, {}


def print_validation_results(
    is_valid: bool,
    errors: list[str],
    warnings: list[str],
    stats: dict = None,
    verbose: bool = False,
    pipeline_type: str = None,
):
    """Print formatted validation results using rich console and osprey theme."""
    console.print()

    if is_valid and not errors:
        header_text = (
            "[bold success]\u2705 VALID[/bold success]\n[dim]Database passed all checks[/dim]"
        )
        border_style = "success"
    else:
        header_text = "[bold error]\u274c INVALID[/bold error]\n[dim]Database has errors[/dim]"
        border_style = "error"

    console.print(Panel.fit(header_text, border_style=border_style, padding=(1, 2)))

    if errors:
        console.print()
        error_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        error_table.add_column("", style="error", no_wrap=False)

        for error in errors:
            error_table.add_row(f"\u2022 {error}")

        console.print(
            Panel(
                error_table,
                title=f"[bold error]\U0001f534 ERRORS ({len(errors)})[/bold error]",
                border_style="error",
                padding=(1, 2),
            )
        )

    if warnings:
        console.print()
        warning_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        warning_table.add_column("", style="warning", no_wrap=False)

        for warning in warnings:
            warning_table.add_row(f"\u2022 {warning}")

        console.print(
            Panel(
                warning_table,
                title=f"[bold warning]\u26a0\ufe0f  WARNINGS ({len(warnings)})[/bold warning]",
                border_style="warning",
                padding=(1, 2),
            )
        )

    if stats:
        console.print()
        stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        stats_table.add_column("Metric", style="label", no_wrap=True)
        stats_table.add_column("Value", style="value")

        if pipeline_type:
            stats_table.add_row("Pipeline Type", pipeline_type.replace("_", " ").title())

        stats_table.add_row("Format", stats.get("format", "unknown"))
        stats_table.add_row("Total Channels", str(stats.get("total_channels", 0)))

        if "template_entries" in stats:
            stats_table.add_row("Template Entries", str(stats["template_entries"]))
        if "standalone_entries" in stats:
            stats_table.add_row("Standalone Entries", str(stats["standalone_entries"]))
        if "compressed_ratio" in stats:
            stats_table.add_row("Compression Ratio", f"{stats['compressed_ratio']:.1f}x")
        if "systems" in stats:
            system_count = len(stats["systems"])
            stats_table.add_row("Systems", str(system_count))

        console.print(
            Panel(
                stats_table,
                title="[bold]\U0001f4ca DATABASE STATISTICS[/bold]",
                border_style="info",
                padding=(1, 2),
            )
        )

        if verbose:
            console.print()
            detailed_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
            detailed_table.add_column("Key", style="dim", no_wrap=True)
            detailed_table.add_column("Value", style="dim")

            for key, value in sorted(stats.items()):
                if key not in [
                    "format",
                    "total_channels",
                    "template_entries",
                    "standalone_entries",
                    "compressed_ratio",
                    "systems",
                ]:
                    detailed_table.add_row(key, str(value))

            if detailed_table.row_count > 0:
                console.print(
                    Panel(
                        detailed_table,
                        title="[bold]\U0001f50d DETAILED STATISTICS[/bold]",
                        border_style="dim",
                        padding=(1, 2),
                    )
                )

    if not errors and not warnings:
        console.print()
        console.print("[success]\u2728 No issues found![/success]")

    console.print()


def run_validation(
    database: str | None = None,
    pipeline: str | None = None,
    verbose: bool = False,
) -> int:
    """Run database validation.

    Args:
        database: Path to database file (default: from config).
        pipeline: Override pipeline type ('hierarchical' or 'in_context').
        verbose: Show detailed statistics.

    Returns:
        0 if valid, 1 if invalid.
    """
    from osprey.services.channel_finder.utils.config import get_config, resolve_path

    pipeline_type = pipeline

    if database:
        db_path = Path(database)
        if not pipeline_type:
            try:
                config = get_config()
                detected_type, _ = detect_pipeline_config(config)
                pipeline_type = detected_type if detected_type else "in_context"
            except Exception:
                pipeline_type = "in_context"
    else:
        try:
            config = get_config()
            detected_type, db_config = detect_pipeline_config(config)

            if not detected_type:
                console.print()
                console.print(
                    Panel(
                        "[bold error]Error:[/bold error] No database configured\n\n"
                        "[warning]Check config.yml:[/warning] Configure either:\n"
                        "  \u2022 channel_finder.pipelines.hierarchical.database.path\n"
                        "  \u2022 channel_finder.pipelines.in_context.database.path",
                        border_style="error",
                        title="\u274c Configuration Error",
                    )
                )
                return 1

            pipeline_type = detected_type
            db_path_str = db_config.get("path")
            if not db_path_str:
                console.print()
                console.print(
                    Panel(
                        "[bold error]Error:[/bold error] No database path in config",
                        border_style="error",
                        title="\u274c Configuration Error",
                    )
                )
                return 1
            db_path = resolve_path(db_path_str)
        except Exception as e:
            console.print()
            console.print(
                Panel(
                    f"[bold error]Error reading config:[/bold error] {e}",
                    border_style="error",
                    title="\u274c Configuration Error",
                )
            )
            return 1

    # Header
    console.print()
    console.print(
        Panel.fit(
            "[bold primary]Database Validation Tool[/bold primary]\n"
            f"[dim]Pipeline: {pipeline_type.replace('_', ' ').title()}[/dim]",
            border_style="primary",
            padding=(1, 2),
        )
    )

    # Configuration info
    console.print()
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_column("Property", style="label")
    config_table.add_column("Value", style="value")
    config_table.add_row("Database Path", str(db_path))
    config_table.add_row("Pipeline Type", pipeline_type.replace("_", " ").title())

    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="info"))

    # Step 1: Validate JSON structure (only for in_context databases)
    if pipeline_type == "in_context":
        is_valid, errors, warnings = validate_json_structure(db_path)
        if not is_valid:
            print_validation_results(
                False, errors, warnings, verbose=verbose, pipeline_type=pipeline_type
            )
            return 1
    else:
        errors = []
        warnings = []
        is_valid = True

    # Step 2: Validate database loading
    load_success, load_errors, stats = validate_database_loading(db_path, pipeline_type)
    errors.extend(load_errors)
    is_valid = is_valid and load_success

    print_validation_results(
        is_valid, errors, warnings, stats, verbose=verbose, pipeline_type=pipeline_type
    )

    return 0 if is_valid else 1
