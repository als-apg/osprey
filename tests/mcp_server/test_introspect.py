"""Tests for MCP server introspection module."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.introspect import (
    _categorize_server,
    _resolve_env,
    _resolve_env_value,
    clear_cache,
    get_mcp_servers_cached,
    introspect_all_servers,
    introspect_server,
)

# ---------------------------------------------------------------------------
# _resolve_env_value
# ---------------------------------------------------------------------------


class TestResolveEnvValue:
    """Test bash-style ${VAR:-default} resolution."""

    def test_plain_string(self):
        assert _resolve_env_value("hello") == "hello"

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            pytest.param("${MISSING_VAR:-fallback}", "fallback", id="var-with-default"),
            pytest.param("${MISSING_VAR}", "", id="var-without-default"),
            pytest.param("${MISSING_VAR:-}", "", id="empty-default"),
        ],
    )
    def test_missing_var_resolves_to_its_default_or_empty(self, monkeypatch, template, expected):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert _resolve_env_value(template) == expected

    def test_var_with_default_present(self):
        with patch.dict(os.environ, {"MY_VAR": "real_value"}):
            assert _resolve_env_value("${MY_VAR:-fallback}") == "real_value"

    def test_var_without_default_present(self):
        with patch.dict(os.environ, {"MY_VAR": "value"}):
            assert _resolve_env_value("${MY_VAR}") == "value"

    def test_multiple_vars(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            result = _resolve_env_value("prefix_${A:-x}_${B:-y}_suffix")
            assert result == "prefix_1_2_suffix"

    def test_no_substitution_needed(self):
        assert _resolve_env_value("/path/to/config.yml") == "/path/to/config.yml"


# ---------------------------------------------------------------------------
# _resolve_env
# ---------------------------------------------------------------------------


class TestResolveEnv:
    """Test full env resolution with parent env merging."""

    def test_none_env_returns_parent(self):
        result = _resolve_env(None)
        assert "PATH" in result  # Should inherit parent env

    def test_overrides_merged(self):
        result = _resolve_env({"CUSTOM_KEY": "custom_value"})
        assert result["CUSTOM_KEY"] == "custom_value"
        assert "PATH" in result  # Parent env still present

    def test_var_resolution_in_values(self):
        with patch.dict(os.environ, {"REAL_KEY": "real_val"}):
            result = _resolve_env({"MY_SETTING": "${REAL_KEY:-default}"})
            assert result["MY_SETTING"] == "real_val"


# ---------------------------------------------------------------------------
# _categorize_server
# ---------------------------------------------------------------------------


class TestCategorizeServer:
    """Test server categorization based on args."""

    def test_osprey_module(self):
        assert _categorize_server(["-m", "osprey.mcp_server.workspace"]) == "osprey"

    def test_external_command(self):
        assert _categorize_server(["--python=3.12", "mcp-atlassian"]) == "external"

    def test_empty_args(self):
        assert _categorize_server([]) == "external"


# ---------------------------------------------------------------------------
# introspect_server
# ---------------------------------------------------------------------------


class TestIntrospectServer:
    """Test individual server introspection."""

    @pytest.mark.asyncio
    async def test_returns_base_info_on_timeout(self):
        """Server that times out should return tools=None gracefully."""
        result = await introspect_server(
            "test-server",
            {"command": "sleep", "args": ["100"], "env": {}},
            "/tmp",
            timeout=0.1,
        )
        assert result["name"] == "test-server"
        assert result["tools"] is None
        assert result["tool_count"] is None
        assert result["category"] == "external"

    @pytest.mark.asyncio
    async def test_returns_base_info_on_bad_command(self):
        """Nonexistent command should return tools=None gracefully."""
        result = await introspect_server(
            "bad-server",
            {"command": "nonexistent_binary_xyz", "args": [], "env": {}},
            "/tmp",
            timeout=2.0,
        )
        assert result["name"] == "bad-server"
        assert result["tools"] is None
        assert result["tool_count"] is None

    @pytest.mark.asyncio
    async def test_osprey_category_detection(self):
        """Server with osprey module path should be categorized as osprey."""
        result = await introspect_server(
            "controls",
            {
                "command": "python3",
                "args": ["-m", "osprey.mcp_server.control_system"],
                "env": {},
            },
            "/tmp",
            timeout=0.1,
        )
        assert result["category"] == "osprey"

    @pytest.mark.asyncio
    async def test_preserves_raw_env_in_output(self):
        """Output should contain the raw env from config, not resolved values."""
        raw_env = {"API_KEY": "${SECRET:-default}", "PLAIN": "value"}
        result = await introspect_server(
            "test",
            {"command": "nonexistent_xyz", "args": [], "env": raw_env},
            "/tmp",
            timeout=0.1,
        )
        assert result["env"] == raw_env

    @pytest.mark.asyncio
    async def test_description_field_present(self):
        """Output should always include a description field (empty on failure)."""
        result = await introspect_server(
            "test",
            {"command": "nonexistent_xyz", "args": [], "env": {}},
            "/tmp",
            timeout=0.1,
        )
        assert "description" in result
        assert result["description"] == ""


# ---------------------------------------------------------------------------
# introspect_all_servers
# ---------------------------------------------------------------------------


class TestIntrospectAllServers:
    """Test parallel introspection of multiple servers."""

    @pytest.mark.asyncio
    async def test_empty_mcp_json(self):
        result = await introspect_all_servers({}, "/tmp")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_servers_key(self):
        result = await introspect_all_servers({"other": "data"}, "/tmp")
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_servers_all_fail_gracefully(self):
        """Multiple failing servers should all return gracefully."""
        mcp_json = {
            "mcpServers": {
                "a": {"command": "nonexistent_a", "args": [], "env": {}},
                "b": {"command": "nonexistent_b", "args": [], "env": {}},
            }
        }
        result = await introspect_all_servers(mcp_json, "/tmp", timeout=0.5)
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"a", "b"}
        assert all(s["tools"] is None for s in result)


# ---------------------------------------------------------------------------
# get_mcp_servers_cached
# ---------------------------------------------------------------------------


class TestGetMcpServersCached:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path):
        clear_cache()
        result = await get_mcp_servers_cached(tmp_path / ".mcp.json", str(tmp_path))
        assert result == []

    @pytest.mark.asyncio
    async def test_cache_hit_on_same_mtime(self, tmp_path):
        """Second call with same file should use cache."""
        clear_cache()
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {}}))

        with patch(
            "osprey.mcp_server.introspect.introspect_all_servers",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_introspect:
            await get_mcp_servers_cached(mcp_file, str(tmp_path))
            await get_mcp_servers_cached(mcp_file, str(tmp_path))
            assert mock_introspect.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_mtime_change(self, tmp_path):
        """Modified file should trigger re-introspection."""
        clear_cache()
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {}}))

        with patch(
            "osprey.mcp_server.introspect.introspect_all_servers",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_introspect:
            await get_mcp_servers_cached(mcp_file, str(tmp_path))

            # Modify the file, and move its mtime past the cached one: two writes
            # inside one filesystem timestamp tick would otherwise share it.
            mcp_file.write_text(json.dumps({"mcpServers": {"new": {}}}))
            stamp = mcp_file.stat().st_mtime_ns + 10_000_000_000
            os.utime(mcp_file, ns=(stamp, stamp))

            await get_mcp_servers_cached(mcp_file, str(tmp_path))
            assert mock_introspect.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self, tmp_path):
        """clear_cache() should force re-introspection."""
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {}}))

        with patch(
            "osprey.mcp_server.introspect.introspect_all_servers",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_introspect:
            await get_mcp_servers_cached(mcp_file, str(tmp_path))
            clear_cache()
            await get_mcp_servers_cached(mcp_file, str(tmp_path))
            assert mock_introspect.call_count == 2


# --------------------------------------------------------------------------- #
# A real stdio server: what initialize and tools/list report reaches the result
# --------------------------------------------------------------------------- #

_TINY_SERVER = """
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("tiny", instructions="Answers one question")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_value(channel: str, count: int = 1) -> str:
    '''Read a value.'''
    return channel


mcp.run()
"""


def _tiny_entry(tmp_path):
    import sys

    script = tmp_path / "tiny_server.py"
    script.write_text(_TINY_SERVER)
    return {"command": sys.executable, "args": [str(script)]}


class TestRealServer:
    @pytest.mark.asyncio
    async def test_initialize_and_tools_list_facts(self, tmp_path):
        info = await introspect_server("tiny", _tiny_entry(tmp_path), str(tmp_path), timeout=30)

        assert info["error"] is None
        assert info["transport"] == "stdio"
        assert info["server_name"] == "tiny"
        assert info["instructions"] == "Answers one question"
        assert info["protocol"]
        assert info["init_ms"] >= 0 and info["list_ms"] >= 0
        (tool,) = info["tools"]
        assert tool["name"] == "read_value"
        assert tool["description"] == "Read a value."
        assert tool["inputSchema"]["required"] == ["channel"]
        assert tool["annotations"]["readOnlyHint"] is True

    @pytest.mark.asyncio
    async def test_an_http_entry_is_listed_not_spawned(self, tmp_path):
        info = await introspect_server(
            "canopy", {"type": "http", "url": "http://h:1/mcp"}, str(tmp_path)
        )
        assert info["transport"] == "http"
        assert info["url"] == "http://h:1/mcp"
        assert info["tools"] is None
        assert info["error"] is None

    @pytest.mark.asyncio
    async def test_refresh_one_server_keeps_the_others_and_peek_never_spawns(self, tmp_path):
        import json

        from osprey.mcp_server.introspect import peek_mcp_servers

        clear_cache()
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({"mcpServers": {"a": {}, "b": {}}}))
        assert peek_mcp_servers(mcp_json) == (None, None)

        calls: list[str] = []

        async def fake(name, _config, _cwd, *, timeout):  # noqa: ARG001 - introspect_server's signature
            calls.append(name)
            return {"name": name, "tools": [], "n": len(calls)}

        with patch("osprey.mcp_server.introspect.introspect_server", side_effect=fake):
            first = await get_mcp_servers_cached(mcp_json, str(tmp_path))
            again = await get_mcp_servers_cached(mcp_json, str(tmp_path), refresh=True, server="b")

        assert sorted(calls[:2]) == ["a", "b"]
        assert calls[2:] == ["b"]
        assert [s["name"] for s in again] == ["a", "b"]
        assert again[0] is first[0]
        servers, probed_at = peek_mcp_servers(mcp_json)
        assert servers == again and probed_at is not None
        clear_cache()
