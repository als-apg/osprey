"""CLI commands for the OSPREY Artifact Gallery."""

import click

from .retired import retired_web_command


@click.group("artifacts")
def artifacts():
    """Artifact Gallery commands.

    Manage the OSPREY Artifact Gallery — a local web gallery that displays
    interactive plots, tables, and other outputs produced by Claude during
    analysis sessions.
    """


artifacts.add_command(retired_web_command("artifacts", "the Artifact Gallery"))
