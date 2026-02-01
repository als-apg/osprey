"""Config auto-update helper for configuration management.

This module is the single source of truth for all configuration file modifications.
Provides utilities for updating config.yml programmatically:
- Comment-preserving YAML updates via ruamel.yaml
- MCP capability react model configuration
- EPICS gateway configuration for production deployment
- Control system type switching

For read-only config access, use utils/config.py (ConfigBuilder, get_config_value).
"""

from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# COMMENT-PRESERVING YAML UTILITIES
# =============================================================================


def update_yaml_file(
    file_path: Path,
    updates: dict[str, Any],
    create_backup: bool = True,
    section_comments: dict[str, str] | None = None,
) -> Path | None:
    """Update a YAML file while preserving comments and formatting.

    Uses ruamel.yaml to maintain comments, blank lines, and original formatting
    when modifying YAML configuration files.

    Args:
        file_path: Path to the YAML file to update
        updates: Dictionary of updates to apply. Supports nested paths using
                dot notation in keys (e.g., {"control_system.type": "epics"})
                or nested dictionaries that will be merged.
        create_backup: If True, creates a .bak file before modifying
        section_comments: Optional dict mapping top-level keys to comment headers.
                         Comments are added with a blank line before them.
                         Example: {"simulation": "Simulation Configuration"}

    Returns:
        Path to backup file if created, None otherwise

    Examples:
        >>> # Simple update
        >>> update_yaml_file(Path("config.yml"), {"control_system.type": "epics"})

        >>> # Nested update
        >>> update_yaml_file(Path("config.yml"), {
        ...     "control_system": {
        ...         "type": "epics",
        ...         "connector": {"epics": {"gateways": {"read_only": {"port": 5064}}}}
        ...     }
        ... })

        >>> # With section comment
        >>> update_yaml_file(Path("config.yml"), {"simulation": {...}},
        ...     section_comments={"simulation": "Simulation Configuration"})
    """
    from ruamel.yaml import YAML, CommentedMap

    yaml_handler = YAML()
    yaml_handler.preserve_quotes = True
    yaml_handler.width = 4096  # Prevent line wrapping

    # Read existing content
    with open(file_path, encoding="utf-8") as f:
        data = yaml_handler.load(f)

    if data is None:
        data = CommentedMap()

    # Create backup before modifying
    backup_path = None
    if create_backup:
        backup_path = file_path.with_suffix(".yml.bak")
        backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Track which keys are being added (for comment placement)
    new_keys = set()
    for key in updates:
        top_key = key.split(".")[0] if "." in key else key
        if top_key not in data:
            new_keys.add(top_key)

    # Apply updates
    _apply_nested_updates(data, updates)

    # Add section comments for new top-level keys
    if section_comments:
        for key, comment in section_comments.items():
            if key in new_keys and key in data:
                # Add boxed section header to match existing config style:
                # # ============================================================
                # # SECTION NAME
                # # ============================================================
                separator = "=" * 60
                comment_text = f"\n{separator}\n{comment}\n{separator}"
                data.yaml_set_comment_before_after_key(key, before=comment_text)

    # Write back with preserved formatting
    with open(file_path, "w", encoding="utf-8") as f:
        yaml_handler.dump(data, f)

    return backup_path


def _apply_nested_updates(data: dict, updates: dict) -> None:
    """Apply nested updates to a dictionary, supporting dot notation keys.

    Args:
        data: Target dictionary to update (modified in place)
        updates: Updates to apply
    """
    for key, value in updates.items():
        if "." in key:
            # Dot notation path (e.g., "control_system.type")
            _set_nested_value(data, key, value)
        elif isinstance(value, dict) and key in data and isinstance(data[key], dict):
            # Recursively merge nested dictionaries
            _apply_nested_updates(data[key], value)
        else:
            # Direct assignment
            data[key] = value


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """Set a value at a nested path using dot notation.

    Args:
        data: Target dictionary
        path: Dot-separated path (e.g., "control_system.connector.epics.port")
        value: Value to set
    """
    keys = path.split(".")
    current = data

    # Navigate/create intermediate dictionaries
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


# =============================================================================
# CONFIG FILE DISCOVERY
# =============================================================================


def find_config_file() -> Path | None:
    """Find the config.yml file in current directory.

    Returns:
        Path to config.yml or None if not found
    """
    config_path = Path.cwd() / "config.yml"
    return config_path if config_path.exists() else None


def has_capability_react_model(config_path: Path, capability_name: str) -> bool:
    """Check if config already has capability-specific react model configured.

    Args:
        config_path: Path to config.yml
        capability_name: Capability name (e.g., 'weather_demo')

    Returns:
        True if {capability_name}_react is already configured
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_key = f"{capability_name}_react"
        if "models" in config and model_key in config["models"]:
            return True
    except Exception:
        pass

    return False


def get_orchestrator_model_config(config_path: Path) -> dict | None:
    """Get the orchestrator model configuration to use as template.

    Args:
        config_path: Path to config.yml

    Returns:
        Dict with provider, model_id, max_tokens or None
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "models" in config and "orchestrator" in config["models"]:
            orch_config = config["models"]["orchestrator"]
            return {
                "provider": orch_config.get("provider", "anthropic"),
                "model_id": orch_config.get("model_id", "claude-sonnet-4"),
                "max_tokens": orch_config.get("max_tokens", 4096),
            }
    except Exception:
        pass

    return None


def generate_capability_react_yaml(
    capability_name: str, template_config: dict | None = None
) -> str:
    """Generate YAML snippet for capability-specific react model.

    Args:
        capability_name: Capability name (e.g., 'weather_demo')
        template_config: Optional dict with provider, model_id, max_tokens
                        If None, uses sensible defaults

    Returns:
        Formatted YAML string for {capability_name}_react model
    """
    if template_config:
        provider = template_config["provider"]
        model_id = template_config["model_id"]
        max_tokens = template_config["max_tokens"]
    else:
        provider = "anthropic"
        model_id = "claude-sonnet-4"
        max_tokens = 4096

    return f"""  {capability_name}_react:
    provider: {provider}
    model_id: {model_id}
    max_tokens: {max_tokens}"""


def add_capability_react_to_config(
    config_path: Path,
    capability_name: str,
    template_config: dict | None = None,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Add capability-specific react model to config.yml.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        capability_name: Capability name (e.g., 'weather_demo')
        template_config: Optional model config to use as template
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content
    """
    # Build model config
    if template_config:
        provider = template_config["provider"]
        model_id_val = template_config["model_id"]
        max_tokens = template_config["max_tokens"]
    else:
        provider = "anthropic"
        model_id_val = "claude-sonnet-4"
        max_tokens = 4096

    model_key = f"{capability_name}_react"
    model_config = {
        "provider": provider,
        "model_id": model_id_val,
        "max_tokens": max_tokens,
    }

    # Apply updates using comment-preserving YAML
    update_yaml_file(
        config_path,
        {f"models.{model_key}": model_config},
        create_backup=create_backup,
    )

    # Read back the updated content
    updated_content = config_path.read_text()

    # Create preview
    capability_react_yaml = generate_capability_react_yaml(capability_name, template_config)
    preview = f"""
[bold]{capability_name.title()} ReAct Model Configuration:[/bold]
{capability_react_yaml}

[dim]Added to the 'models' section of your config.yml[/dim]
"""

    return updated_content, preview


def get_config_preview(capability_name: str, template_config: dict | None = None) -> str:
    """Get a preview of what will be added to config.

    Args:
        capability_name: Capability name (e.g., 'weather_demo')
        template_config: Optional model config to use as template

    Returns:
        Formatted preview string
    """
    capability_react_yaml = generate_capability_react_yaml(capability_name, template_config)

    return f"""
[bold]{capability_name.title()} ReAct Model Configuration:[/bold]

{capability_react_yaml}

[dim]Note: This model will be used by {capability_name} for autonomous tool selection.
If not configured, the capability falls back to using the 'orchestrator' model.[/dim]
"""


def remove_capability_react_from_config(
    config_path: Path,
    capability_name: str,
    create_backup: bool = True,
) -> tuple[Path | None, str, bool]:
    """Remove capability-specific react model from config.yml.

    Uses ruamel.yaml to preserve comments while removing the model entry.

    Args:
        config_path: Path to config.yml
        capability_name: Capability name (e.g., 'weather_demo')
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (backup_path, preview, found) where:
        - backup_path: Path to backup file, None if no backup created
        - preview: Human-readable description of what was removed
        - found: True if model was found and removed
    """
    from ruamel.yaml import YAML

    model_key = f"{capability_name}_react"

    yaml_handler = YAML()
    yaml_handler.preserve_quotes = True
    yaml_handler.width = 4096

    # Read existing content
    with open(config_path, encoding="utf-8") as f:
        data = yaml_handler.load(f)

    if data is None or "models" not in data or model_key not in data["models"]:
        preview = f"\n[dim]No config entry found for '{model_key}'[/dim]"
        return None, preview, False

    # Get the config being removed for preview
    removed_config = data["models"][model_key]
    removed_section = f"  {model_key}:\n"
    for key, value in removed_config.items():
        removed_section += f"    {key}: {value}\n"
    removed_section = removed_section.rstrip()

    # Create backup before modifying
    backup_path = None
    if create_backup:
        backup_path = config_path.with_suffix(".yml.bak")
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Remove the model entry
    del data["models"][model_key]

    # Write back with preserved formatting
    with open(config_path, "w", encoding="utf-8") as f:
        yaml_handler.dump(data, f)

    # Generate preview
    preview = f"""
[bold]{capability_name.title()} ReAct Model Configuration:[/bold]
[red]- REMOVED:[/red]
{removed_section}
"""

    return backup_path, preview, True


def get_capability_react_config(config_path: Path, capability_name: str) -> dict | None:
    """Get the capability-specific react model configuration.

    Args:
        config_path: Path to config.yml
        capability_name: Capability name

    Returns:
        Dict with model configuration or None if not found
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_key = f"{capability_name}_react"
        if "models" in config and model_key in config["models"]:
            return config["models"][model_key]
    except Exception:
        pass

    return None


# ============================================================================
# Control System Type Configuration
# ============================================================================


def get_control_system_type(config_path: Path, key: str = "control_system.type") -> str | None:
    """Get control system or archiver type from config.yml.

    Args:
        config_path: Path to config.yml
        key: Config key to read (e.g., 'control_system.type' or 'archiver.type')

    Returns:
        Type string or None if not found
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Navigate nested keys
        keys = key.split(".")
        value = config
        for k in keys:
            value = value.get(k)
            if value is None:
                return None

        return value
    except Exception:
        pass

    return None


def set_control_system_type(
    config_path: Path,
    control_type: str,
    archiver_type: str | None = None,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Update control system and optionally archiver type in config.yml.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        control_type: 'mock' or 'epics'
        archiver_type: Optional archiver type ('mock_archiver', 'epics_archiver')
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content
    """
    # Build updates dict
    updates: dict[str, Any] = {"control_system.type": control_type}

    if archiver_type:
        updates["archiver.type"] = archiver_type

    # Apply updates using comment-preserving YAML
    update_yaml_file(config_path, updates, create_backup=create_backup)

    # Read back the updated content
    updated_content = config_path.read_text()

    # Create preview
    preview_lines = [
        "[bold]Control System Configuration[/bold]\n",
        f"control_system.type: {control_type}",
    ]

    if archiver_type:
        preview_lines.append(f"archiver.type: {archiver_type}")

    preview_lines.append("\n[dim]Updated config.yml[/dim]")

    preview = "\n".join(preview_lines)

    return updated_content, preview


# ============================================================================
# EPICS Gateway Configuration
# ============================================================================


def set_epics_gateway_config(
    config_path: Path,
    facility: str,
    custom_config: dict | None = None,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Update EPICS gateway configuration in config.yml.

    Updates the control_system.connector.epics.gateways section with
    facility-specific or custom gateway settings.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        facility: 'aps', 'als', or 'custom'
        custom_config: For 'custom', dict with 'read_only' and 'write_access' gateways
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content

    Example:
        >>> new_content, preview = set_epics_gateway_config(config_path, 'aps')
    """
    from osprey.templates.data import get_facility_config

    if facility == "custom":
        if not custom_config:
            raise ValueError("custom_config required when facility='custom'")
        gateway_config = custom_config
        facility_name = "Custom"
    else:
        preset = get_facility_config(facility)
        if not preset:
            raise ValueError(f"Unknown facility: {facility}")
        gateway_config = preset["gateways"]
        facility_name = preset["name"]

    # Apply updates using comment-preserving YAML
    update_yaml_file(
        config_path,
        {"control_system.connector.epics.gateways": gateway_config},
        create_backup=create_backup,
    )

    # Read back the updated content
    updated_content = config_path.read_text()

    # Build preview string
    gateway_yaml = _format_gateway_yaml(gateway_config)
    preview = f"""
[bold]EPICS Gateway Configuration - {facility_name}[/bold]

{gateway_yaml.rstrip()}

[dim]Updated control_system.connector.epics.gateways in config.yml[/dim]
"""

    return updated_content, preview


def _format_gateway_yaml(gateway_config: dict) -> str:
    """Format gateway configuration as YAML string.

    Args:
        gateway_config: Dict with 'read_only' and 'write_access' gateway configs

    Returns:
        Formatted YAML string with proper indentation
    """
    lines = []

    for gateway_type in ["read_only", "write_access"]:
        if gateway_type in gateway_config:
            gw = gateway_config[gateway_type]
            lines.append(f"        {gateway_type}:")
            lines.append(f"          address: {gw['address']}")
            lines.append(f"          port: {gw['port']}")
            lines.append(
                f"          use_name_server: {str(gw.get('use_name_server', False)).lower()}"
            )

    return "\n".join(lines) + "\n"


def get_epics_gateway_config(config_path: Path) -> dict | None:
    """Get current EPICS gateway configuration from config.yml.

    Args:
        config_path: Path to config.yml

    Returns:
        Dict with gateway configuration or None if not found
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gateways = (
            config.get("control_system", {}).get("connector", {}).get("epics", {}).get("gateways")
        )
        return gateways
    except Exception:
        pass

    return None


def get_facility_from_gateway_config(config_path: Path) -> str | None:
    """Detect which facility preset is configured (if any).

    Compares current gateway configuration against known facility presets.

    Args:
        config_path: Path to config.yml

    Returns:
        Facility name ('APS', 'ALS', 'Custom') or None if using defaults
    """
    from osprey.templates.data import FACILITY_PRESETS

    current_gateways = get_epics_gateway_config(config_path)
    if not current_gateways:
        return None

    # Check if current config matches any preset
    for _facility_id, preset in FACILITY_PRESETS.items():
        preset_gateways = preset["gateways"]

        # Compare read_only gateway
        if "read_only" in current_gateways and "read_only" in preset_gateways:
            current_read = current_gateways["read_only"]
            preset_read = preset_gateways["read_only"]

            if (
                current_read.get("address") == preset_read["address"]
                and current_read.get("port") == preset_read["port"]
            ):
                return preset["name"]

    # Check if it looks like a custom configuration (not default ALS)
    if "read_only" in current_gateways:
        read_addr = current_gateways["read_only"].get("address", "")
        if read_addr and read_addr != "cagw-alsdmz.als.lbl.gov":
            return "Custom"

    return None


# ============================================================================
# Model Configuration
# ============================================================================


def get_all_model_configs(config_path: Path) -> dict | None:
    """Get all model configurations from config.yml.

    Args:
        config_path: Path to config.yml

    Returns:
        Dict with all model configurations or None if not found
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config.get("models", {})
    except Exception:
        pass

    return None


def update_all_models(
    config_path: Path,
    provider: str,
    model_id: str,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Update all model configurations in config.yml with new provider/model.

    This updates ALL model entries in the models section to use the same
    provider and model_id, while preserving any custom max_tokens settings.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        provider: Provider name (e.g., 'openai', 'anthropic', 'cborg')
        model_id: Model ID (e.g., 'gpt-4', 'claude-sonnet-4')
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content

    Example:
        >>> new_content, preview = update_all_models(config_path, 'openai', 'gpt-4')
    """
    # Get current models to build updates and preview
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        current_models = config.get("models", {})
    except Exception:
        current_models = {}

    # Build updates for each model, preserving max_tokens
    updates: dict[str, Any] = {}
    for model_name in current_models:
        updates[f"models.{model_name}.provider"] = provider
        updates[f"models.{model_name}.model_id"] = model_id

    # Apply updates using comment-preserving YAML
    update_yaml_file(config_path, updates, create_backup=create_backup)

    # Read back the updated content
    updated_content = config_path.read_text()

    # Create preview showing changes
    model_count = len(current_models)

    preview_lines = [
        "[bold]Model Configuration Update[/bold]\n",
        f"Provider: [value]{provider}[/value]",
        f"Model ID: [value]{model_id}[/value]",
        f"\nUpdated [bold]{model_count}[/bold] model configuration(s):",
    ]

    # List the models that were updated
    for model_name in sorted(current_models.keys()):
        current = current_models[model_name]
        current_provider = current.get("provider", "unknown")
        current_model = current.get("model_id", "unknown")

        if current_provider != provider or current_model != model_id:
            preview_lines.append(
                f"  • {model_name}: {current_provider}/{current_model} → {provider}/{model_id}"
            )
        else:
            preview_lines.append(f"  • {model_name}: [dim](no change)[/dim]")

    preview_lines.append("\n[dim]Note: max_tokens settings were preserved[/dim]")

    preview = "\n".join(preview_lines)

    return updated_content, preview
