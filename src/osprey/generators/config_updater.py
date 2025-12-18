"""Config auto-update helper for configuration management.

Provides utilities for updating config.yml programmatically:
- MCP capability react model configuration
- EPICS gateway configuration for production deployment
"""

import re
from pathlib import Path

import yaml


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
    config_path: Path, capability_name: str, template_config: dict | None = None
) -> tuple[str, str]:
    """Add capability-specific react model to config.yml.

    Args:
        config_path: Path to config.yml
        capability_name: Capability name (e.g., 'weather_demo')
        template_config: Optional model config to use as template

    Returns:
        Tuple of (new_content, preview) where preview shows what was added
    """
    content = config_path.read_text()

    # Generate the capability-specific react configuration
    capability_react_yaml = generate_capability_react_yaml(capability_name, template_config)

    # Find the models section and add capability-specific react model
    # Look for pattern: models:\n  <existing_models>
    # We want to add after the last model entry

    # Strategy: Find the models: section, then find where the next top-level key starts
    # Insert capability_react before that next section

    models_pattern = r"(models:\s*\n(?:  \w+:.*\n(?:    .*\n)*)+)"

    def add_capability_react(match):
        models_section = match.group(1)
        # Add capability_react at the end of the models section
        return f"{models_section}{capability_react_yaml}\n"

    new_content = re.sub(models_pattern, add_capability_react, content, count=1)

    # Create preview
    preview = f"""
[bold]{capability_name.title()} ReAct Model Configuration:[/bold]
{capability_react_yaml}

[dim]This will be added to the 'models' section of your config.yml[/dim]
"""

    return new_content, preview


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
    config_path: Path, capability_name: str
) -> tuple[str, str, bool]:
    """Remove capability-specific react model from config.yml.

    Args:
        config_path: Path to config.yml
        capability_name: Capability name (e.g., 'weather_demo')

    Returns:
        Tuple of (new_content, preview, found) where:
        - new_content: Updated config content
        - preview: Human-readable description of what was removed
        - found: True if model was found and removed
    """
    content = config_path.read_text()
    model_key = f"{capability_name}_react"

    # Pattern to match the entire model entry including all its properties
    # Matches:
    #   capability_name_react:
    #     provider: ...
    #     model_id: ...
    #     max_tokens: ...
    model_pattern = rf"^  {re.escape(model_key)}:\s*\n(?:    .*\n)*"

    match = re.search(model_pattern, content, flags=re.MULTILINE)

    if not match:
        preview = f"\n[dim]No config entry found for '{model_key}'[/dim]"
        return content, preview, False

    # Extract what we're removing for preview
    removed_section = match.group(0).rstrip()

    # Remove the section
    new_content = re.sub(model_pattern, "", content, flags=re.MULTILINE)

    # Generate preview
    preview = f"""
[bold]{capability_name.title()} ReAct Model Configuration:[/bold]
[red]- REMOVE:[/red]
{removed_section}
"""

    return new_content, preview, True


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
    config_path: Path, control_type: str, archiver_type: str | None = None
) -> tuple[str, str]:
    """Update control system and optionally archiver type in config.yml.

    Args:
        config_path: Path to config.yml
        control_type: 'mock' or 'epics'
        archiver_type: Optional archiver type ('mock_archiver', 'epics_archiver')

    Returns:
        Tuple of (new_content, preview)
    """
    content = config_path.read_text()

    # Update control_system.type
    control_pattern = r"(control_system:\s*\n\s*type:\s*)\w+"
    control_replacement = rf"\1{control_type}"
    new_content = re.sub(control_pattern, control_replacement, content, flags=re.MULTILINE)

    # Update archiver.type if specified
    if archiver_type:
        archiver_pattern = r"(archiver:\s*\n(?:.*\n)*?\s*type:\s*)\w+"
        archiver_replacement = rf"\1{archiver_type}"
        new_content = re.sub(
            archiver_pattern, archiver_replacement, new_content, flags=re.MULTILINE
        )

    # Create preview
    preview_lines = [
        "[bold]Control System Configuration[/bold]\n",
        f"control_system.type: {control_type}",
    ]

    if archiver_type:
        preview_lines.append(f"archiver.type: {archiver_type}")

    preview_lines.append("\n[dim]This will update your config.yml[/dim]")

    preview = "\n".join(preview_lines)

    return new_content, preview


# ============================================================================
# EPICS Gateway Configuration
# ============================================================================


def set_epics_gateway_config(
    config_path: Path, facility: str, custom_config: dict | None = None
) -> tuple[str, str]:
    """Update EPICS gateway configuration in config.yml.

    Updates the control_system.connector.epics.gateways section with
    facility-specific or custom gateway settings.

    Args:
        config_path: Path to config.yml
        facility: 'aps', 'als', or 'custom'
        custom_config: For 'custom', dict with 'read_only' and 'write_access' gateways

    Returns:
        Tuple of (new_content, preview) where preview shows what will be changed

    Example:
        >>> new_content, preview = set_epics_gateway_config(config_path, 'aps')
        >>> config_path.write_text(new_content)
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

    content = config_path.read_text()

    # Build replacement gateway section
    gateway_yaml = _format_gateway_yaml(gateway_config)

    # Find and replace the gateways section within control_system.connector.epics
    # Pattern: Look for the gateways: section under epics:
    pattern = r"(epics:\s*\n\s*timeout:.*?\n\s*gateways:\s*\n)((?:\s+\w+:\s*\n(?:\s+.*\n)*)*)"

    def replace_gateways(match):
        header = match.group(1)  # Keep "epics:\n  timeout: X\n  gateways:\n"
        return header + gateway_yaml

    new_content = re.sub(pattern, replace_gateways, content, flags=re.MULTILINE)

    # Create preview
    preview = f"""
[bold]EPICS Gateway Configuration - {facility_name}[/bold]

{gateway_yaml.rstrip()}

[dim]This will update control_system.connector.epics.gateways in config.yml[/dim]
"""

    return new_content, preview


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


def update_all_models(config_path: Path, provider: str, model_id: str) -> tuple[str, str]:
    """Update all model configurations in config.yml with new provider/model.

    This updates ALL model entries in the models section to use the same
    provider and model_id, while preserving any custom max_tokens settings.

    Args:
        config_path: Path to config.yml
        provider: Provider name (e.g., 'openai', 'anthropic', 'cborg')
        model_id: Model ID (e.g., 'gpt-4', 'claude-sonnet-4')

    Returns:
        Tuple of (new_content, preview) where preview shows what will be changed

    Example:
        >>> new_content, preview = update_all_models(config_path, 'openai', 'gpt-4')
        >>> config_path.write_text(new_content)
    """
    content = config_path.read_text()

    # Get current models to show what's changing
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        current_models = config.get("models", {})
    except Exception:
        current_models = {}

    # Pattern to match provider and model_id in any model entry
    # Matches:   provider: <value>
    #            model_id: <value>
    # Under the models: section (with 2-space indentation for role names)

    # Update all provider entries under models section
    provider_pattern = r"(models:(?:\s*\n  \w+:(?:\s*\n    (?:provider|model_id|max_tokens):.*)*)*)"

    def update_models_section(match):
        section = match.group(0)
        # Replace provider: <anything>
        section = re.sub(r"(\n    provider:\s*)\S+", rf"\1{provider}", section)
        # Replace model_id: <anything>
        section = re.sub(r"(\n    model_id:\s*)\S+", rf"\1{model_id}", section)
        return section

    new_content = re.sub(provider_pattern, update_models_section, content, flags=re.MULTILINE)

    # Create preview showing changes
    model_count = len(current_models)

    preview_lines = [
        "[bold]Model Configuration Update[/bold]\n",
        f"Provider: [value]{provider}[/value]",
        f"Model ID: [value]{model_id}[/value]",
        f"\nThis will update [bold]{model_count}[/bold] model configuration(s):",
    ]

    # List the models that will be updated
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

    preview_lines.append("\n[dim]Note: max_tokens settings will be preserved[/dim]")

    preview = "\n".join(preview_lines)

    return new_content, preview
