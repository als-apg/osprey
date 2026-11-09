"""The ARIEL guide's command table lists the ``osprey ariel`` group as it is.

``docs/source/how-to/ariel/index.rst`` ends with a table that claims to cover
all ARIEL functionality. The table's command column must equal the group's
registered commands, and the ``--mode`` values it names for ``search`` must
equal the choices the CLI accepts, which come from the search-module registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest

from osprey.cli.main import cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX = _REPO_ROOT / "docs" / "source" / "how-to" / "ariel" / "index.rst"


def _table() -> dict[str, str]:
    """Map each command in the CLI Commands table to its description cell."""
    text = _INDEX.read_text(encoding="utf-8")
    section = text[text.index("CLI Commands\n============") :]
    section = section[: section.index(".. toctree::")]
    rows = re.findall(r"^   \* - ``([a-z-]+)``\n     - (.+)$", section, re.M)
    assert rows, "the CLI Commands table has no command rows"
    return dict(rows)


def _ariel_group() -> tuple[click.Context, click.Group]:
    ctx = click.Context(cli)
    group = cli.get_command(ctx, "ariel")
    assert isinstance(group, click.Group)
    return ctx, group


def test_table_lists_every_ariel_command() -> None:
    ctx, group = _ariel_group()
    visible = {
        name
        for name in group.list_commands(ctx)
        if not getattr(group.get_command(ctx, name), "hidden", False)
    }
    assert set(_table()) == visible


def test_search_mode_cell_names_the_registered_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Outside a project the framework's baseline registrations answer, which is
    # what the guide documents.
    monkeypatch.chdir(tmp_path)
    ctx, group = _ariel_group()
    search = group.get_command(ctx, "search")
    assert search is not None
    (mode,) = [p for p in search.params if p.name == "mode"]
    assert isinstance(mode.type, click.Choice)
    cell = re.search(r"``--mode ([a-z|]+)``", _table()["search"])
    assert cell is not None
    assert set(cell.group(1).split("|")) == set(mode.type.choices)
