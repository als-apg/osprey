"""CLI commands for the OSPREY monitoring stack.

Usage:
    osprey monitoring install [--force]
    osprey monitoring status
    osprey monitoring start
    osprey monitoring stop
"""

from __future__ import annotations

import click


@click.group("monitoring")
def monitoring():
    """Manage the OTEL monitoring stack (Collector + Prometheus + Grafana)."""


@monitoring.command()
@click.option("--force", is_flag=True, help="Re-download binaries even if already installed.")
def install(force: bool) -> None:
    """Download and install monitoring stack binaries."""
    from osprey.interfaces.monitoring.installer import install_monitoring_binaries

    install_monitoring_binaries(force=force)


@monitoring.command()
def status() -> None:
    """Check health of monitoring services."""
    from osprey.interfaces.monitoring.launcher import _monitoring_launcher

    statuses = _monitoring_launcher.status()

    for name, healthy in statuses.items():
        symbol = click.style("OK", fg="green") if healthy else click.style("DOWN", fg="red")
        click.echo(f"  {name}: {symbol}")

    if all(statuses.values()):
        click.echo("\nAll monitoring services are running.")
    elif any(statuses.values()):
        click.echo("\nSome monitoring services are not running.")
    else:
        click.echo("\nNo monitoring services are running. Run: osprey monitoring start")


@monitoring.command()
def start() -> None:
    """Start the monitoring stack."""
    from osprey.interfaces.monitoring.launcher import _monitoring_config, _monitoring_launcher

    config = _monitoring_config()
    if not config:
        click.echo("No monitoring: section found in config.yml")
        click.echo("Add a monitoring section or run: osprey monitoring install")
        return

    click.echo("Starting monitoring stack...")
    _monitoring_launcher.ensure_running(config)

    statuses = _monitoring_launcher.status()
    for name, healthy in statuses.items():
        symbol = click.style("OK", fg="green") if healthy else click.style("FAILED", fg="red")
        click.echo(f"  {name}: {symbol}")


@monitoring.command()
def stop() -> None:
    """Stop all monitoring services."""
    from osprey.interfaces.monitoring.launcher import stop_monitoring_stack

    click.echo("Stopping monitoring stack...")
    stop_monitoring_stack()
    click.echo("Monitoring stack stopped.")
