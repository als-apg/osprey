"""Registry auto-update helper for MCP capability generation.

Automatically adds generated capabilities to project registry with user confirmation.
"""

import re
from pathlib import Path


def find_registry_file() -> Path | None:
    """Find the registry.py file from config.yml.

    Reads registry_path from config.yml in current directory.

    Returns:
        Path to registry.py or None if not found or config missing
    """
    from osprey.utils.config import get_config_value

    try:
        # Get registry path from config
        registry_path = get_config_value("registry_path")
        if not registry_path:
            return None

        # Convert to Path and resolve relative to current directory
        registry_file = Path(registry_path)
        if not registry_file.is_absolute():
            registry_file = Path.cwd() / registry_file

        if registry_file.exists():
            return registry_file

    except Exception:
        pass

    return None


def get_project_module_name(registry_path: Path) -> str:
    """Extract the project module name from registry path.

    Args:
        registry_path: Path to registry.py

    Returns:
        Module name (e.g., 'my_control_assistant')
    """
    return registry_path.parent.name


def generate_capability_registration(
    capability_name: str,
    class_name: str,
    module_name: str,
    context_type: str,
    description: str = ""
) -> str:
    """Generate CapabilityRegistration code.

    Args:
        capability_name: Name of the capability (e.g., 'slack_mcp')
        class_name: Class name (e.g., 'SlackMcpCapability')
        module_name: Project module name
        context_type: Context type produced
        description: Human-readable description

    Returns:
        Formatted registration code
    """
    if not description:
        description = f"{capability_name} operations via MCP server"

    return f'''                CapabilityRegistration(
                    name="{capability_name}",
                    module_path="{module_name}.capabilities.{capability_name}",
                    class_name="{class_name}",
                    description="{description}",
                    provides=["{context_type}"],
                    requires=[]
                ),'''


def generate_context_registration(
    context_type: str,
    context_class_name: str,
    module_name: str,
    capability_name: str
) -> str:
    """Generate ContextClassRegistration code.

    Args:
        context_type: Context type (e.g., 'SLACK_RESULTS')
        context_class_name: Context class name
        module_name: Project module name
        capability_name: Capability name (for module path)

    Returns:
        Formatted registration code
    """
    return f'''                ContextClassRegistration(
                    context_type="{context_type}",
                    module_path="{module_name}.capabilities.{capability_name}",
                    class_name="{context_class_name}"
                ),'''


def _find_last_registration_entry(content: str, registration_type: str) -> int | None:
    """Find the position after the last registration entry of a given type.

    Uses a robust approach: finds all complete registration blocks and returns
    the position after the last one (including its trailing comma and newline).

    Args:
        content: File content to search
        registration_type: Either 'CapabilityRegistration' or 'ContextClassRegistration'

    Returns:
        Position after the last entry, or None if no entries found
    """
    # Match complete registration blocks: TypeName(...),
    # We need to handle nested parentheses (e.g., provides=["TYPE"])
    # Strategy: find all occurrences and track balanced parentheses

    pattern = rf'{registration_type}\s*\('
    matches = list(re.finditer(pattern, content))

    if not matches:
        return None

    # Find the end of the last registration (balanced parentheses)
    last_match = matches[-1]
    start = last_match.start()
    pos = last_match.end()  # Position after opening (
    depth = 1

    while pos < len(content) and depth > 0:
        char = content[pos]
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        pos += 1

    # Now pos is right after the closing )
    # Skip trailing comma and whitespace to find the insertion point
    while pos < len(content) and content[pos] in ' \t':
        pos += 1
    if pos < len(content) and content[pos] == ',':
        pos += 1

    return pos


def add_to_registry(
    registry_path: Path,
    capability_name: str,
    class_name: str,
    context_type: str,
    context_class_name: str,
    description: str = ""
) -> tuple[str, str]:
    """Add capability to registry file.

    Uses a robust insertion strategy: finds the last entry of each registration
    type and inserts after it. This approach doesn't depend on what follows
    the list (context_classes, framework_prompt_providers, or closing paren).

    Args:
        registry_path: Path to registry.py
        capability_name: Capability name
        class_name: Capability class name
        context_type: Context type
        context_class_name: Context class name
        description: Capability description

    Returns:
        Tuple of (new_content, preview) where preview shows what was added
    """
    module_name = get_project_module_name(registry_path)
    content = registry_path.read_text()

    # Generate registration code
    cap_reg = generate_capability_registration(
        capability_name, class_name, module_name, context_type, description
    )
    ctx_reg = generate_context_registration(
        context_type, context_class_name, module_name, capability_name
    )

    # Find insertion point after last CapabilityRegistration
    cap_insert_pos = _find_last_registration_entry(content, 'CapabilityRegistration')
    if cap_insert_pos is not None:
        content = content[:cap_insert_pos] + "\n" + cap_reg + content[cap_insert_pos:]

    # Find insertion point after last ContextClassRegistration
    # (search in updated content to account for capability insertion)
    ctx_insert_pos = _find_last_registration_entry(content, 'ContextClassRegistration')
    if ctx_insert_pos is not None:
        content = content[:ctx_insert_pos] + "\n" + ctx_reg + content[ctx_insert_pos:]

    # Create preview showing what was added
    preview = f"""
[bold]Capability Registration:[/bold]
{cap_reg}

[bold]Context Class Registration:[/bold]
{ctx_reg}
"""

    return content, preview


def is_already_registered(registry_path: Path, capability_name: str) -> bool:
    """Check if capability is already registered.

    Args:
        registry_path: Path to registry.py
        capability_name: Capability name to check

    Returns:
        True if already registered
    """
    content = registry_path.read_text()
    # Look for name="{capability_name}"
    pattern = r'name\s*=\s*["\']'+ re.escape(capability_name) + r'["\']'
    return bool(re.search(pattern, content))


def remove_from_registry(
    registry_path: Path,
    capability_name: str
) -> tuple[str, str, bool]:
    """Remove capability and context class from registry file.

    Args:
        registry_path: Path to registry.py
        capability_name: Capability name to remove

    Returns:
        Tuple of (new_content, preview, found) where:
        - new_content: Updated file content
        - preview: Human-readable description of what was removed
        - found: True if capability was found and removed
    """
    content = registry_path.read_text()

    # Track what we found
    found_capability = False
    found_context = False
    capability_preview = ""
    context_preview = ""

    # Remove CapabilityRegistration entry
    # Match the entire CapabilityRegistration(...), block
    # Looking for name="{capability_name}" within the block
    capability_pattern = r'CapabilityRegistration\s*\(\s*name\s*=\s*["\']' + re.escape(capability_name) + r'["\'][^)]*\)\s*,\s*\n'

    capability_match = re.search(capability_pattern, content, flags=re.DOTALL)
    if capability_match:
        found_capability = True
        capability_preview = capability_match.group(0).strip()
        content = re.sub(capability_pattern, '', content, flags=re.DOTALL)

    # Remove ContextClassRegistration entry
    # We need to find the context_type that was associated with this capability
    # First, extract context_type from the capability registration (if we found it)
    context_type = None
    if capability_match:
        provides_match = re.search(r'provides\s*=\s*\[\s*["\']([^"\']+)["\']', capability_preview)
        if provides_match:
            context_type = provides_match.group(1)

    # If we found a context_type, remove its registration
    if context_type:
        context_pattern = r'ContextClassRegistration\s*\(\s*context_type\s*=\s*["\']' + re.escape(context_type) + r'["\'][^)]*\)\s*,\s*\n'
        context_match = re.search(context_pattern, content, flags=re.DOTALL)
        if context_match:
            found_context = True
            context_preview = context_match.group(0).strip()
            content = re.sub(context_pattern, '', content, flags=re.DOTALL)

    # Generate preview
    found = found_capability or found_context

    if not found:
        preview = f"\n[dim]No registry entries found for '{capability_name}'[/dim]"
    else:
        preview = "\n[bold]Capability Registration:[/bold]\n"
        if found_capability:
            preview += f"[red]- REMOVE:[/red]\n{capability_preview}\n"
        else:
            preview += "[dim]Not found[/dim]\n"

        preview += "\n[bold]Context Class Registration:[/bold]\n"
        if found_context:
            preview += f"[red]- REMOVE:[/red]\n{context_preview}\n"
        else:
            preview += "[dim]Not found[/dim]\n"

    return content, preview, found


def get_capability_info(registry_path: Path, capability_name: str) -> dict | None:
    """Extract capability information from registry.

    Args:
        registry_path: Path to registry.py
        capability_name: Capability name

    Returns:
        Dict with class_name, context_type, context_class_name or None if not found
    """
    content = registry_path.read_text()

    # Find the CapabilityRegistration block
    capability_pattern = r'CapabilityRegistration\s*\(\s*name\s*=\s*["\']' + re.escape(capability_name) + r'["\'][^)]*\)'
    match = re.search(capability_pattern, content, flags=re.DOTALL)

    if not match:
        return None

    block = match.group(0)

    # Extract fields
    info = {}

    # Extract class_name
    class_match = re.search(r'class_name\s*=\s*["\']([^"\']+)["\']', block)
    if class_match:
        info['class_name'] = class_match.group(1)

    # Extract context_type from provides list
    provides_match = re.search(r'provides\s*=\s*\[\s*["\']([^"\']+)["\']', block)
    if provides_match:
        info['context_type'] = provides_match.group(1)

    # Extract module_path
    module_match = re.search(r'module_path\s*=\s*["\']([^"\']+)["\']', block)
    if module_match:
        info['module_path'] = module_match.group(1)

    # Find context class name from ContextClassRegistration
    if 'context_type' in info:
        context_pattern = r'ContextClassRegistration\s*\(\s*context_type\s*=\s*["\']' + re.escape(info['context_type']) + r'["\'][^)]*\)'
        context_match = re.search(context_pattern, content, flags=re.DOTALL)
        if context_match:
            context_block = context_match.group(0)
            context_class_match = re.search(r'class_name\s*=\s*["\']([^"\']+)["\']', context_block)
            if context_class_match:
                info['context_class_name'] = context_class_match.group(1)

    return info if info else None

