"""Service status display with Rich output.

Shows detailed container status using the container runtime's ps command,
separating project containers from other Osprey containers.
"""

import json
import os
import subprocess

from rich.table import Table

from osprey.deployment.runtime_helper import get_ps_command
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
            config = config.raw_config
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
        state = container.get("State", "unknown")
        if state == "running":
            status = f"[{styles.SUCCESS}]● Running[/{styles.SUCCESS}]"
        elif state == "exited":
            status = f"[{styles.ERROR}]● Stopped[/{styles.ERROR}]"
        elif state == "restarting":
            status = f"[{styles.WARNING}]● Restarting[/{styles.WARNING}]"
        else:
            status = f"[{styles.DIM}]● {state}[/{styles.DIM}]"

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

    console.print()
