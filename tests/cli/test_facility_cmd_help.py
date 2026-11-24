"""``osprey facility`` no longer carries the panels verbs.

The operator-screen commands moved to the ``osprey-panels`` package's own
console script; the framework's facility group must not list a ``panels``
command, or a user would reach for a verb that no longer exists there.
"""

from __future__ import annotations

from click.testing import CliRunner

from osprey.cli.main import cli


def test_facility_help_lists_no_panels_command() -> None:
    result = CliRunner().invoke(cli, ["facility", "--help"])

    assert result.exit_code == 0, result.output
    commands = result.stdout.split("Commands:", 1)[1]
    listed = {line.split()[0] for line in commands.splitlines() if line.strip()}
    assert "import" in listed
    assert "panels" not in listed


def test_facility_panels_is_not_a_command() -> None:
    result = CliRunner().invoke(cli, ["facility", "panels", "--help"])

    assert result.exit_code != 0
