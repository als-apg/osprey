"""New verbs' docstrings name only commands that actually exist.

``osprey scaffold pull``, ``osprey scaffold personas`` and ``osprey profile
card`` each carry an Examples block that names ``osprey`` invocations. A
docstring that names a chain the live CLI does not resolve misleads whoever
reads ``--help`` before running it, so each verb's help text is pinned against
the real command tree the same way the emitted CI files are in
``test_scaffold_ci.py`` — this file reuses that check rather than
duplicating it.
"""

from __future__ import annotations

import click
import pytest

from osprey.cli.main import cli
from tests.cli.test_scaffold_ci import named_commands, unresolvable

#: Each verb, named by its full invocation, and the chain that reaches it from
#: the root group.
VERB_CHAINS: dict[str, tuple[str, ...]] = {
    "osprey scaffold pull": ("scaffold", "pull"),
    "osprey scaffold personas": ("scaffold", "personas"),
    "osprey profile card": ("profile", "card"),
}


def _command_help(chain: tuple[str, ...]) -> str:
    """The ``.help`` text of the command *chain* reaches, walking from ``cli``."""
    context = click.Context(cli)
    command: click.Command = cli
    for name in chain:
        assert isinstance(command, click.Group), f"{name!r} has no parent group"
        resolved = command.get_command(context, name)
        assert resolved is not None, f"osprey {' '.join(chain)} is not registered"
        command = resolved
    assert command.help is not None, f"osprey {' '.join(chain)} has no docstring"
    return command.help


@pytest.mark.parametrize("verb,chain", VERB_CHAINS.items(), ids=list(VERB_CHAINS))
def test_examples_name_only_live_commands(verb: str, chain: tuple[str, ...]) -> None:
    help_text = _command_help(chain)

    named = named_commands(help_text)
    assert named, f"{verb} --help names no osprey command in its Examples block"

    for named_chain in named:
        reason = unresolvable(named_chain)
        assert reason is None, reason
