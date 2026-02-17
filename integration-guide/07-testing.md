# Recipe 7: Adding Tests

## When You Need This

Always. Every MCP tool and web endpoint needs tests. OSPREY uses pytest with pytest-asyncio for async testing.

## Test Directory Structure

```
tests/
├── interfaces/
│   └── {name}/
│       └── mcp/
│           ├── conftest.py          # Shared fixtures, utilities
│           ├── test_registry.py     # Registry init, config loading, service caching
│           ├── test_{tool_group}.py  # One file per tool group
│           └── ...
├── services/
│   └── {name}/
│       ├── test_service.py          # Service layer tests
│       └── test_repository.py       # Database tests (if applicable)
└── conftest.py                      # Project-wide fixtures
```

## MCP Tool Testing Pattern

### `conftest.py` — Shared Fixtures

```python
"""Test fixtures for {your tool} MCP server."""

import pytest

from osprey.interfaces.my_tool.mcp.registry import (
    initialize_my_registry,
    reset_my_registry,
)


def get_tool_fn(tool_or_fn):
    """Extract the raw async function from a FastMCP FunctionTool.

    FastMCP wraps @mcp.tool() decorated functions into FunctionTool objects.
    The original function is available via the .fn attribute.
    """
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton between tests to prevent state leaks.

    This is autouse=True — it runs for EVERY test automatically.
    """
    yield
    reset_my_registry()


def _setup_registry(tmp_path, monkeypatch):
    """Initialize registry with test config in a temporary directory.

    Call this at the start of each test that needs the registry.
    """
    monkeypatch.chdir(tmp_path)
    config_content = """
my_tool:
  database:
    uri: postgresql://localhost/test_db
  analysis:
    default_mode: fft
"""
    (tmp_path / "config.yml").write_text(config_content)
    initialize_my_registry()


def make_mock_entry(
    entry_id="test-001",
    title="Test Entry",
    data=None,
    timestamp=None,
) -> dict:
    """Create a plain dict matching your data model.

    Returns a dict (not MagicMock) so it works with:
    - isinstance checks
    - JSON serialization in tools
    - Dictionary access patterns
    """
    from datetime import datetime, timezone

    return {
        "entry_id": entry_id,
        "title": title,
        "data": data or {},
        "timestamp": (timestamp or datetime.now(tz=timezone.utc)).isoformat(),
    }


@pytest.fixture
def mock_service():
    """Create AsyncMock with the shape of your real service."""
    from unittest.mock import AsyncMock, MagicMock

    service = AsyncMock()
    service.repository = AsyncMock()
    service.config = MagicMock()
    return service
```

### Tool Tests

```python
"""Tests for {tool_group} MCP tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.interfaces.my_tool.mcp.tools.analysis import my_analyze_tool

from .conftest import _setup_registry, get_tool_fn, make_mock_entry


class TestMyAnalyzeTool:
    """Tests for the my_analyze_tool MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_basic_analysis(self, tmp_path, monkeypatch):
        """Successful analysis returns expected JSON structure."""
        _setup_registry(tmp_path, monkeypatch)

        mock_result = MagicMock()
        mock_result.items = [make_mock_entry(entry_id="e1")]
        mock_result.tunes = {"h": 0.234, "v": 0.317}

        mock_service = AsyncMock()
        mock_service.analyze.return_value = mock_result

        with patch(
            "osprey.interfaces.my_tool.mcp.registry.MyRegistry.service",
            new=AsyncMock(return_value=mock_service),
        ):
            fn = get_tool_fn(my_analyze_tool)
            result = await fn(query="horizontal tune", mode="fft")

        data = json.loads(result)
        assert not data.get("error", False)
        assert "tunes" in data
        assert data["tunes"]["h"] == 0.234

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_mode_returns_error(self, tmp_path, monkeypatch):
        """Invalid mode produces a validation error, not an exception."""
        _setup_registry(tmp_path, monkeypatch)

        mock_service = AsyncMock()
        mock_service.analyze.side_effect = ValueError("Unknown mode: invalid")

        with patch(
            "osprey.interfaces.my_tool.mcp.registry.MyRegistry.service",
            new=AsyncMock(return_value=mock_service),
        ):
            fn = get_tool_fn(my_analyze_tool)
            result = await fn(query="test", mode="invalid")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"
        assert "suggestions" in data

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_service_failure_returns_internal_error(self, tmp_path, monkeypatch):
        """Unexpected exception is caught and returned as internal_error."""
        _setup_registry(tmp_path, monkeypatch)

        mock_service = AsyncMock()
        mock_service.analyze.side_effect = RuntimeError("DB connection lost")

        with patch(
            "osprey.interfaces.my_tool.mcp.registry.MyRegistry.service",
            new=AsyncMock(return_value=mock_service),
        ):
            fn = get_tool_fn(my_analyze_tool)
            result = await fn(query="test")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "internal_error"
```

### Registry Tests

```python
"""Tests for {your tool} registry."""

import pytest

from osprey.interfaces.my_tool.mcp.registry import (
    get_my_registry,
    initialize_my_registry,
)

from .conftest import _setup_registry


class TestRegistry:
    @pytest.mark.unit
    def test_initialize_loads_config(self, tmp_path, monkeypatch):
        _setup_registry(tmp_path, monkeypatch)
        registry = get_my_registry()
        assert registry.config is not None

    @pytest.mark.unit
    def test_get_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_my_registry()

    @pytest.mark.unit
    def test_missing_config_section_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("other_section: {}")
        with pytest.raises(RuntimeError, match="Missing"):
            initialize_my_registry()
```

## Web Interface Testing Pattern

```python
"""Tests for {your tool} web API."""

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.my_tool.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with temporary config."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text("my_tool:\n  database:\n    uri: sqlite:///test.db\n")
    app = create_app(str(config))
    return TestClient(app)


class TestHealthEndpoint:
    @pytest.mark.unit
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAnalyzeEndpoint:
    @pytest.mark.unit
    def test_analyze_success(self, client):
        resp = client.post("/api/analyze", json={
            "query": "horizontal tune",
            "mode": "fft",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    @pytest.mark.unit
    def test_analyze_invalid_mode(self, client):
        resp = client.post("/api/analyze", json={
            "query": "test",
            "mode": "nonexistent",
        })
        assert resp.status_code == 422  # Pydantic validation error
```

## Mocking Strategy

### What to Mock

| Layer | Mock? | How |
|-------|-------|-----|
| MCP tool functions | No — test these directly | Call via `get_tool_fn()` |
| Registry | Partially — mock the service it returns | `patch("...Registry.service")` |
| Service | Yes — mock for tool tests | `AsyncMock()` with set attributes |
| Repository | Yes — mock for service tests | `AsyncMock()` for all DB methods |
| Database | Yes — never hit real DB in unit tests | Mock repository or use test DB |
| Config | No — use real config files in `tmp_path` | `monkeypatch.chdir(tmp_path)` |

### Key Mocking Rules

1. **Use plain dicts for data entries**, not `MagicMock` — dicts serialize to JSON correctly and work with `isinstance` checks
2. **Always reset the registry** — the `autouse` fixture handles this automatically
3. **Use `tmp_path` + `monkeypatch.chdir`** for config — creates isolated test environments
4. **Patch at the registry level** — don't patch individual service methods deep in the import chain

## Running Tests

```bash
# All unit tests (fast, no API keys)
pytest tests/ --ignore=tests/e2e -v

# Just your tool's tests
pytest tests/interfaces/my_tool/ -v

# Single test file
pytest tests/interfaces/my_tool/mcp/test_analysis.py -v

# Single test
pytest tests/interfaces/my_tool/mcp/test_analysis.py::TestMyAnalyzeTool::test_basic_analysis -v
```

**Critical**: Never run bare `pytest` — always use `pytest tests/ --ignore=tests/e2e -v` to avoid E2E registry state leaks.

## Concrete Reference

- `tests/interfaces/ariel/mcp/conftest.py` — `get_tool_fn()`, `_reset_registry`, `make_mock_entry()`
- `tests/interfaces/ariel/mcp/test_search.py` — Full tool test suite with success/error/edge cases
- `tests/interfaces/ariel/mcp/test_registry.py` — Registry init, config loading, service caching
- `tests/interfaces/ariel/mcp/test_entry.py` — Entry CRUD with draft vs. direct modes
- `tests/hooks/` — Hook script tests (33 tests)
- `tests/mcp_server/` — OSPREY MCP server tests (76 tests)

## Checklist

- [ ] `conftest.py` with `get_tool_fn()`, `_reset_registry` (autouse), `_setup_registry`, `make_mock_entry`
- [ ] One test file per tool group
- [ ] Tests marked with `@pytest.mark.unit` or `@pytest.mark.integration`
- [ ] Async tests marked with `@pytest.mark.asyncio`
- [ ] Mock service at registry level, not deep in imports
- [ ] Plain dicts for data entries (not MagicMock)
- [ ] Test both success and error paths for every tool
- [ ] Verify JSON response structure (not just status codes)
- [ ] Registry reset between tests (autouse fixture)
