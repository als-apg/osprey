"""The ``sql_query`` offer on the ARIEL MCP surface.

The agent's SQL tool runs only as the store's SELECT-only login. A deployment
that authors ``ariel.database.uri`` without ``ariel.database.readonly_uri`` has
no such login, so:

* ``tools/list`` has no ``sql_query`` and loses nothing else;
* a call that reaches the tool anyway is refused with ``not_supported`` naming
  the key, before any service is built.

On a derived store, or with ``readonly_uri`` authored, the tool is offered.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context, reset_ariel_context
from osprey.services.ariel_search.config import READONLY_URI_KEY
from osprey_connectors.workspace import reset_config_cache
from tests.mcp_server.ariel.conftest import get_tool_fn
from tests.mcp_server.conftest import assert_raises_error

AUTHORED_URI = "postgresql://ariel@logbook-db.example.org:5432/ariel"
AUTHORED_READONLY_URI = "postgresql://ariel_ro@logbook-db.example.org:5432/ariel"

WITHOUT_LOGIN = {"uri": AUTHORED_URI}
WITH_LOGIN = {"uri": AUTHORED_URI, "readonly_uri": AUTHORED_READONLY_URI}
DERIVED: dict[str, Any] = {}


def _context(tmp_path, monkeypatch, database: dict[str, Any]) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": {"database": database}}))
    initialize_ariel_context()


def _register_tools() -> None:
    import importlib
    import pkgutil

    from osprey.mcp_server.ariel import tools

    for found in pkgutil.iter_modules(tools.__path__):
        importlib.import_module(f"{tools.__name__}.{found.name}")


async def _listed_names() -> set[str]:
    from osprey.mcp_server.ariel.server import mcp

    _register_tools()
    return {tool.name for tool in await mcp.list_tools()}


async def test_without_a_readonly_login_tools_list_lacks_sql_query_and_nothing_else(
    tmp_path, monkeypatch
):
    _context(tmp_path / "derived", monkeypatch, DERIVED)
    offered = await _listed_names()
    reset_ariel_context()
    reset_config_cache()

    _context(tmp_path / "authored", monkeypatch, WITHOUT_LOGIN)
    withheld = await _listed_names()

    assert "sql_query" in offered
    assert offered - withheld == {"sql_query"}
    assert withheld <= offered


async def test_without_a_readonly_login_the_wire_lacks_sql_query(tmp_path, monkeypatch):
    """An MCP client's ``tools/list`` goes through the same filter."""
    from fastmcp import Client

    from osprey.mcp_server.ariel.server import mcp

    _context(tmp_path, monkeypatch, WITHOUT_LOGIN)
    _register_tools()
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert "sql_query" not in names
    assert "keyword_search" in names


@pytest.mark.parametrize("database", [DERIVED, WITH_LOGIN], ids=["derived", "authored-readonly"])
async def test_a_resolved_readonly_login_lists_sql_query(tmp_path, monkeypatch, database):
    _context(tmp_path, monkeypatch, database)
    assert "sql_query" in await _listed_names()


async def test_without_a_readonly_login_a_call_is_refused_naming_the_key(tmp_path, monkeypatch):
    """Refused before validation and before any ``registry.service()`` call."""
    from osprey.mcp_server.ariel.tools.sql_query import sql_query

    _context(tmp_path, monkeypatch, WITHOUT_LOGIN)
    service = AsyncMock(side_effect=AssertionError("no service may be built"))
    with (
        patch("osprey.mcp_server.ariel.server_context.ARIELContext.service", new=service),
        assert_raises_error(error_type="not_supported") as ctx,
    ):
        await get_tool_fn(sql_query)(sql="")

    envelope = ctx["envelope"]
    assert envelope["details"] == {"key": READONLY_URI_KEY}
    assert READONLY_URI_KEY in envelope["error_message"]
    assert READONLY_URI_KEY in " ".join(envelope["suggestions"])
    service.assert_not_awaited()


def test_offered_without_a_context():
    """An import-only walker (no context at all) lists the tool."""
    from osprey.mcp_server.ariel.server import mcp, sql_tool_offered_here

    reset_ariel_context()
    assert sql_tool_offered_here() is True
    _register_tools()
    assert "sql_query" in {tool.name for tool in asyncio.run(mcp.list_tools())}
