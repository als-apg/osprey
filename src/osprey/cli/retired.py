"""Verbs the CLI has retired but still answers, so an old habit gets a pointer."""

from __future__ import annotations

import click

from osprey.cli.output import fail


def retired_web_command(verb: str, panel: str) -> click.Command:
    """Build the hidden ``web`` stub of a panel whose server left the CLI.

    It takes any arguments the old verb took, ``--help`` included, so whatever
    the operator typed gets the pointer rather than a usage error.

    Args:
        verb: The command group's name, e.g. ``"ariel"``.
        panel: The panel's name as the operator knows it, e.g. ``"ARIEL"``.

    Returns:
        A command that prints the pointer to stderr and exits 2.
    """

    @click.pass_context
    def stub(ctx: click.Context) -> None:
        fail(
            f"`osprey {verb} web` is retired: {panel} runs inside the web terminal (`osprey web`)."
        )
        ctx.exit(2)

    return click.Command(
        "web",
        callback=stub,
        hidden=True,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
