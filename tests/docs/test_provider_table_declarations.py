"""The provider guide's table restates what each adapter class declares.

These tests hold every column of the table to that declaration, so a provider
that changes its key variable, its protocol or what its route carries cannot
leave the guide saying otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.infrastructure.proxy.translator import _not_carried
from osprey.models.provider_registry import get_provider_registry
from osprey.models.providers.base import BaseProvider

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "source"
    / "how-to"
    / "llm-providers"
    / "configure-providers.rst"
)

_ROW = "   * - "
_CELL = "     - "


def _table_rows() -> tuple[list[str], dict[str, list[str]]]:
    """The Available Providers table: its header cells, and name → remaining cells."""
    lines = GUIDE.read_text(encoding="utf-8").splitlines()
    start = lines.index("Available Providers")
    start = next(i for i in range(start, len(lines)) if lines[i].startswith(".. list-table::"))
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" "):
            break
        if line.startswith(_ROW):
            rows.append([line[len(_ROW) :]])
        elif line.startswith(_CELL) and rows:
            rows[-1].append(line[len(_CELL) :])
    rows = [[cell.replace("``", "").strip() for cell in row] for row in rows]
    header, body = rows[0], {row[0]: row[1:] for row in rows[1:]}
    assert header and body
    return header, body


def _adapters() -> dict[str, type[BaseProvider]]:
    """Every registered built-in provider, loaded to its adapter class."""
    registry = get_provider_registry()
    loaded = {name: registry.get_provider(name) for name in registry.list_providers()}
    missing = sorted(name for name, cls in loaded.items() if cls is None)
    assert not missing, f"no provider adapter resolves for: {missing}"
    return {name: cls for name, cls in loaded.items() if cls is not None}


def _names() -> list[str]:
    return sorted(get_provider_registry().list_providers())


def _cell(name: str, column: str) -> str:
    header, rows = _table_rows()
    return rows[name][header.index(column) - 1]


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def test_the_table_has_the_declared_columns():
    header, _ = _table_rows()
    assert header == ["Name", "Description", "API Key Env Var", "Protocol", "Images", "Thinking"]


def test_the_table_lists_every_registered_provider():
    _, rows = _table_rows()
    assert set(rows) == set(_names())


@pytest.mark.parametrize("name", _names())
def test_the_key_column_is_each_adapters_key_variable(name):
    declared = _adapters()[name].api_key_env_var
    assert _cell(name, "API Key Env Var") == (declared if declared else "*(none)*")


@pytest.mark.parametrize("name", _names())
def test_the_protocol_column_is_each_adapters_protocol(name):
    native = _adapters()[name].api_protocol == "anthropic"
    assert _cell(name, "Protocol") == ("Anthropic (native)" if native else "OpenAI (proxied)")


@pytest.mark.parametrize("name", _names())
def test_the_images_column_is_what_reaches_the_model(name):
    cls = _adapters()[name]
    native = cls.api_protocol == "anthropic"
    assert _cell(name, "Images") == _yes_no(native or cls.supports_images)


@pytest.mark.parametrize("name", _names())
def test_the_thinking_column_is_what_reaches_the_model(name):
    cls = _adapters()[name]
    native = cls.api_protocol == "anthropic"
    assert _cell(name, "Thinking") == _yes_no(native or cls.supports_thinking)


def test_the_guide_quotes_the_note_the_model_reads_for_an_image():
    text = " ".join(GUIDE.read_text(encoding="utf-8").split())
    assert f"``{_not_carried('image')}``" in text
