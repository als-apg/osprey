"""``osprey ariel web`` is retired: it says so and starts nothing."""

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group

MESSAGE = "`osprey ariel web` is retired: ARIEL runs inside the web terminal (`osprey web`)."


@pytest.mark.parametrize(
    "args",
    [["web"], ["web", "--port", "8080", "--host", "0.0.0.0", "--reload"], ["web", "--help"]],
    ids=["bare", "old-flags", "help"],
)
def test_stub_exits_2_with_the_message(args):
    result = CliRunner().invoke(ariel_group, args)
    assert result.exit_code == 2
    assert MESSAGE in result.output


def test_stub_is_hidden_from_the_group_help():
    assert ariel_group.commands["web"].hidden
