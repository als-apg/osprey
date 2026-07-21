"""Service status display with Rich output.

Shows detailed container status using the container runtime's ps command,
separating project containers from other Osprey containers.
"""

import json
import os
import subprocess

from rich.table import Table

from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.facility_config import normalize_facility_config
from osprey.deployment.runtime_helper import get_ps_command, get_runtime_command
from osprey.deployment.web_terminals.naming import web_container_name
from osprey.utils.config import ConfigBuilder
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.status")


class _DefaultStyles:
    """Fallback styles when cli.styles is not available."""

    BOLD_PRIMARY = "bold"
    ACCENT = "cyan"
    SUCCESS = "green"
    PRIMARY = "bold"
    INFO = "cyan"
    DIM = "dim"
    ERROR = "red"
    WARNING = "yellow"


def _format_state(state, styles):
    """Format a container ``State`` value as a colored Rich status label.

    Shared by the project/other container tables (``_add_container_to_table``)
    and the per-user web terminal table (``_format_container_health``) so the
    state->label mapping can't drift between the two.

    :param state: Raw ``State`` value from the container runtime's ps output
    :type state: str
    :param styles: Style constants object with SUCCESS/ERROR/WARNING/DIM attributes
    :type styles: object
    :return: Rich markup string, e.g. ``"[green]● Running[/green]"``
    :rtype: str
    """
    if state == "running":
        return f"[{styles.SUCCESS}]● Running[/{styles.SUCCESS}]"
    elif state == "exited":
        return f"[{styles.ERROR}]● Stopped[/{styles.ERROR}]"
    elif state == "restarting":
        return f"[{styles.WARNING}]● Restarting[/{styles.WARNING}]"
    else:
        return f"[{styles.DIM}]● {state}[/{styles.DIM}]"


def _extract_web_terminal_user_names(users_raw):
    """Extract bare usernames from ``modules.web_terminals.users``.

    Entries may be legacy bare strings (``"alice"``) or explicit object form
    (``{"name": "alice", "index": 0}``); both are handled defensively here
    rather than importing a shared normalizer, since that normalizer lives in
    a module edited in parallel by another task.

    :param users_raw: Raw value of ``modules.web_terminals.users``
    :type users_raw: object
    :return: Bare usernames in declaration order; malformed entries are dropped
    :rtype: list[str]
    """
    if not isinstance(users_raw, list):
        return []
    names = []
    for entry in users_raw:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def show_status(config_path, *, console=None, styles=None):
    """Show detailed status of services with formatted output.

    Uses direct container runtime ps to show actual container state, independent of compose files.
    Displays containers for this project separately from other running containers.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param console: Rich Console instance for output (default: creates new Console)
    :type console: rich.console.Console, optional
    :param styles: Style constants object with attributes like SUCCESS, ERROR, etc.
        (default: uses _DefaultStyles with standard Rich markup)
    :type styles: object, optional
    """
    if console is None:
        from rich.console import Console

        console = Console()
    if styles is None:
        styles = _DefaultStyles

    try:
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = normalize_facility_config(config.raw_config)
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    # Get deployed services and current project name
    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # Determine current project name (same logic as _inject_project_metadata)
    current_project = config.get("project_name")
    if not current_project:
        project_root = config.get("project_root", "")
        if project_root:
            current_project = os.path.basename(project_root.rstrip("/"))
    if not current_project:
        current_project = "unnamed-project"

    # Get all containers using direct runtime ps (not compose-dependent)
    try:
        result = subprocess.run(
            get_ps_command(config, all_containers=True), capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            console.print("\n[red]Error: Could not query container status[/red]")
            console.print(f"[dim]Command failed with return code {result.returncode}[/dim]\n")
            return

        # Parse newline-separated JSON objects (Docker format) or JSON array (Podman format)
        all_containers = []
        if result.stdout.strip():
            try:
                # Try parsing as JSON array first (Podman format)
                all_containers = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fall back to newline-separated JSON objects (Docker format)
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        all_containers.append(json.loads(line))

    except subprocess.TimeoutExpired:
        console.print("\n[red]Error: Container query timed out[/red]\n")
        return
    except json.JSONDecodeError as e:
        console.print("\n[red]Error: Could not parse container data[/red]")
        console.print(f"[dim]{e}[/dim]\n")
        return
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        return

    # Separate containers into project and non-project
    project_containers = []
    other_containers = []

    for container in all_containers:
        # Extract project label
        labels = container.get("Labels", {})
        container_project = "unknown"

        if isinstance(labels, dict):
            container_project = labels.get("osprey.project.name", "unknown")
        elif isinstance(labels, str):
            for label in labels.split(","):
                if "=" in label:
                    key, value = label.split("=", 1)
                    if key.strip() == "osprey.project.name":
                        container_project = value.strip()
                        break

        # Check if container belongs to this project
        belongs_to_project = container_project == current_project

        # Also check if container name matches any deployed service (for backward compatibility)
        names = container.get("Names", [])
        if isinstance(names, list):
            names_str = " ".join(str(n) for n in names).lower()
        else:
            names_str = str(names).lower()

        matches_service = any(
            service.split(".")[-1].lower() in names_str for service in deployed_service_names
        )

        if belongs_to_project or matches_service:
            project_containers.append(container)
        else:
            # Only include containers with osprey labels in "other"
            if container_project != "unknown":
                other_containers.append(container)

    # Helper functions for status display
    def _create_status_table():
        """Create a status table with consistent styling."""
        table = Table(show_header=True, header_style=styles.BOLD_PRIMARY)
        table.add_column("Service", style=styles.ACCENT, no_wrap=True)
        table.add_column("Project", style=styles.SUCCESS, no_wrap=True)
        table.add_column("Status", style=styles.PRIMARY)
        table.add_column("Ports", style=styles.INFO)
        table.add_column("Image", style=styles.DIM)
        return table

    def _add_container_to_table(table, container):
        """Add a container as a row in the status table."""
        # Extract container name
        names = container.get("Names", [])
        if isinstance(names, list) and names:
            container_name = names[0]
        else:
            container_name = str(names) if names else "unknown"

        # Extract project label
        labels = container.get("Labels", {})
        project_name = "unknown"
        if isinstance(labels, dict):
            project_name = labels.get("osprey.project.name", "unknown")
        elif isinstance(labels, str):
            for label in labels.split(","):
                if "=" in label:
                    key, value = label.split("=", 1)
                    if key.strip() == "osprey.project.name":
                        project_name = value.strip()
                        break

        # Truncate long project names
        if len(project_name) > 12:
            project_name = project_name[:9] + "..."

        # Format status
        status = _format_state(container.get("State", "unknown"), styles)

        # Format ports
        ports_raw = container.get("Ports", [])
        port_list = []
        if ports_raw:
            for port in ports_raw:
                if isinstance(port, dict):
                    # Handle different port format variations
                    # podman ps format: host_port, container_port
                    # compose ps format: PublishedPort, TargetPort
                    published = (
                        port.get("host_port")
                        or port.get("PublishedPort")
                        or port.get("published", "")
                    )
                    target = (
                        port.get("container_port")
                        or port.get("TargetPort")
                        or port.get("target", "")
                    )
                    if published and target:
                        port_list.append(f"{published}→{target}")
        ports = ", ".join(port_list) if port_list else "-"

        # Get image
        image = container.get("Image", "unknown")
        if len(image) > 40:
            image = "..." + image[-37:]

        table.add_row(container_name, project_name, status, ports, image)

    # Display project containers
    console.print("\n[bold]Service Status:[/bold]")

    if project_containers:
        table = _create_status_table()
        for container in project_containers:
            _add_container_to_table(table, container)
        console.print(table)
    else:
        console.print(
            f"\n[warning]ℹ️  No services running for project '{current_project}'[/warning]"
        )
        if deployed_service_names:
            console.print(f"[dim]Configured services: {', '.join(deployed_service_names)}[/dim]")
        console.print("\n[info]Start services with:[/info]")
        console.print("  • [command]osprey deploy up[/command]")

    # Display other osprey containers
    if other_containers:
        console.print("\n[bold]Other Osprey Containers:[/bold]")
        other_table = _create_status_table()
        for container in other_containers:
            _add_container_to_table(other_table, container)
        console.print(other_table)

    # Display per-user web terminal status
    modules_cfg = config.get("modules", {})
    web_terminals_cfg = (
        modules_cfg.get("web_terminals", {}) if isinstance(modules_cfg, dict) else {}
    )
    if not isinstance(web_terminals_cfg, dict):
        web_terminals_cfg = {}
    user_names = _extract_web_terminal_user_names(web_terminals_cfg.get("users"))

    if web_terminals_cfg.get("enabled") and user_names:
        facility_cfg = config.get("facility", {})
        if not isinstance(facility_cfg, dict):
            facility_cfg = {}
        facility_prefix = facility_cfg.get("prefix") or ""

        def _find_container_by_name(target_name):
            """Look up a container by exact name in the already-fetched all_containers."""
            for container in all_containers:
                names = container.get("Names", [])
                if isinstance(names, list):
                    name_candidates = [str(n) for n in names]
                else:
                    name_candidates = [str(names)]
                # Docker sometimes prefixes container names with "/"
                name_candidates = [n.lstrip("/") for n in name_candidates]
                if target_name in name_candidates:
                    return container
            return None

        def _format_container_health(container):
            """Format a Running/Stopped/Restarting/Not created label, matching the
            styling used by _add_container_to_table's state formatting."""
            if container is None:
                return f"[{styles.DIM}]● Not created[/{styles.DIM}]"
            return _format_state(container.get("State", "unknown"), styles)

        # Query existing volumes once via the runtime (read-only; never created/removed).
        existing_volumes = None
        try:
            runtime_bin = get_runtime_command(config)[0]
            vol_result = subprocess.run(
                [runtime_bin, "volume", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if vol_result.returncode == 0:
                existing_volumes = {
                    line.strip() for line in vol_result.stdout.splitlines() if line.strip()
                }
            else:
                console.print("[dim]Warning: could not query volumes for user status[/dim]")
        except subprocess.TimeoutExpired:
            console.print("[dim]Warning: volume query timed out[/dim]")
        except Exception as e:
            console.print(f"[dim]Warning: could not query volumes ({e})[/dim]")

        def _format_volume_status(volume_name):
            if existing_volumes is None:
                return f"[{styles.DIM}]? {volume_name} (unknown)[/{styles.DIM}]"
            if volume_name in existing_volumes:
                return f"[{styles.SUCCESS}]✓ {volume_name}[/{styles.SUCCESS}]"
            return f"[{styles.ERROR}]✗ {volume_name} (missing)[/{styles.ERROR}]"

        def _create_user_status_table():
            table = Table(show_header=True, header_style=styles.BOLD_PRIMARY)
            table.add_column("User", style=styles.ACCENT, no_wrap=True)
            table.add_column("Container", style=styles.PRIMARY)
            table.add_column("Claude Config Volume", style=styles.INFO)
            table.add_column("Agent Data Volume", style=styles.INFO)
            return table

        console.print("\n[bold]Web Terminal Users:[/bold]")
        user_table = _create_user_status_table()
        for user in user_names:
            container_name = web_container_name(facility_prefix, user)
            container = _find_container_by_name(container_name)
            claude_config_volume, agent_data_volume = resolve_user_volume_names(config, user)
            user_table.add_row(
                user,
                _format_container_health(container),
                _format_volume_status(claude_config_volume),
                _format_volume_status(agent_data_volume),
            )
        console.print(user_table)

    console.print()
