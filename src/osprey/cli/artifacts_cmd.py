"""The retired ``osprey artifacts`` group: the gallery runs inside the web terminal."""

import click

from .retired import retired_web_command


@click.group("artifacts", hidden=True)
def artifacts():
    """Retired: the Artifact Gallery runs inside the web terminal.

    The group holds only the retired ``web`` stub, which points an old habit
    at ``osprey web``. It is hidden from ``osprey --help``.
    """


artifacts.add_command(retired_web_command("artifacts", "the Artifact Gallery"))
