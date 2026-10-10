"""The ARIEL commands' help text names code by what it does, never by line.

A ``file.py:123`` reference in ``--help`` is wrong after the next edit to that
file, and nothing fails when it goes stale.
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group

_SOURCE_LINE = re.compile(r"\.py:\d+")


@pytest.mark.parametrize("name", sorted(ariel_group.commands))
def test_no_ariel_command_help_cites_a_source_line(name: str) -> None:
    result = CliRunner().invoke(ariel_group, [name, "--help"])

    assert result.exit_code == 0, result.output
    assert not _SOURCE_LINE.search(result.output), result.output
