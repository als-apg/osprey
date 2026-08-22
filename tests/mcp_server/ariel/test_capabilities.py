"""Tests for the capabilities MCP tool."""

import json

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.registry import get_registry
from tests.mcp_server.ariel.conftest import get_tool_fn


def _get_capabilities():
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    return get_tool_fn(capabilities)


def _setup_registry(tmp_path, monkeypatch, search_modules=None, vocabulary=None):
    """Write a config, initialize the framework registry and the ARIEL context.

    The framework registry must be initialized because ``capabilities`` now
    advertises the modes the registry actually carries, not a hardcoded list.

    Args:
        tmp_path: Temporary working directory fixture.
        monkeypatch: Pytest monkeypatch fixture, used to chdir.
        search_modules: Optional ``search_modules`` config block. Defaults to
            keyword and semantic both enabled.
        vocabulary: Optional ``vocabulary`` config block. Omitted entirely by
            default, which is the no-vocabulary deployment.
    """
    monkeypatch.chdir(tmp_path)
    if search_modules is None:
        search_modules = {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": "nomic-embed-text"},
        }
    ariel: dict = {
        "database": {"uri": "postgresql://localhost/test"},
        "search_modules": search_modules,
    }
    if vocabulary is not None:
        ariel["vocabulary"] = vocabulary
    config = json.dumps({"ariel": ariel})
    (tmp_path / "config.yml").write_text(config)
    get_registry().initialize()
    initialize_ariel_context()


@pytest.mark.unit
async def test_capabilities_returns_modules(tmp_path, monkeypatch):
    """Capabilities returns enabled search modules."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert not data.get("error", False)
    assert "keyword" in data["enabled_search_modules"]
    assert "semantic" in data["enabled_search_modules"]


@pytest.mark.unit
async def test_capabilities_includes_search_modes(tmp_path, monkeypatch):
    """Capabilities advertises every registered, enabled search module."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_omits_sql_query_mode(tmp_path, monkeypatch):
    """``sql_query`` is a tool, not a mode, so it never appears in the mode list."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "sql_query" not in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_omits_disabled_modes(tmp_path, monkeypatch):
    """A registered module that config disables is not advertised as a mode."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        search_modules={
            "keyword": {"enabled": True},
            "semantic": {"enabled": False},
        },
    )

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" not in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_no_registry_import():
    """Capabilities does NOT import from osprey.registry (main framework)."""
    import ast
    import inspect

    from osprey.mcp_server.ariel.tools import capabilities

    # Read the source file directly rather than via inspect.getsource, whose
    # linecache/bytecode-lineno slicing can drift under a transient .py/.pyc skew.
    source_path = inspect.getsourcefile(capabilities) or inspect.getfile(capabilities)
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("osprey.registry"), (
                    "capabilities must NOT import from osprey.registry"
                )


# --- vocabulary ---------------------------------------------------------------

VOCABULARY_YML = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
  - canonical: radio frequency
    kind: acronym
    forms:
      - rf
"""


def _write_vocabulary(tmp_path):
    """Write a three-concept vocabulary file and return its absolute path."""
    path = tmp_path / "vocabulary.yml"
    path.write_text(VOCABULARY_YML)
    return str(path)


def _shared_parameter_names(data):
    """The names of the shared parameters the payload advertises."""
    return [parameter["name"] for parameter in data["shared_parameters"]]


@pytest.mark.unit
async def test_capabilities_reports_the_vocabulary(tmp_path, monkeypatch):
    """An agent learns the vocabulary exists and how big it is."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        vocabulary={"enabled": True, "path": _write_vocabulary(tmp_path)},
    )

    fn = _get_capabilities()
    data = json.loads(await fn())

    assert data["vocabulary"] == {
        "enabled": True,
        "concepts": 3,
        "expand_by_default": True,
    }


@pytest.mark.unit
async def test_capabilities_advertises_expand_query_when_enabled(tmp_path, monkeypatch):
    """The per-call toggle is advertised only where it does something."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        vocabulary={"enabled": True, "path": _write_vocabulary(tmp_path)},
    )

    fn = _get_capabilities()
    data = json.loads(await fn())

    assert "expand_query" in _shared_parameter_names(data)


@pytest.mark.unit
async def test_capabilities_omits_expand_query_when_disabled(tmp_path, monkeypatch):
    """No vocabulary means no toggle to click and no capability to explain."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    data = json.loads(await fn())

    assert data["vocabulary"] == {
        "enabled": False,
        "concepts": 0,
        "expand_by_default": False,
    }
    assert "expand_query" not in _shared_parameter_names(data)
    assert "max_results" in _shared_parameter_names(data)


@pytest.mark.unit
async def test_capabilities_reports_expand_by_default_off(tmp_path, monkeypatch):
    """A deployment that ships the vocabulary switched off says so."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        vocabulary={
            "enabled": True,
            "path": _write_vocabulary(tmp_path),
            "expand_by_default": False,
        },
    )

    fn = _get_capabilities()
    data = json.loads(await fn())

    assert data["vocabulary"]["expand_by_default"] is False
    expand = next(p for p in data["shared_parameters"] if p["name"] == "expand_query")
    assert expand["default"] is False


@pytest.mark.unit
async def test_capabilities_docstring_explains_the_vocabulary_block(tmp_path, monkeypatch):
    """The docstring is a prompt surface: it must name what it now returns."""
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    doc = get_tool_fn(capabilities).__doc__ or ""

    assert "vocabulary" in doc
    assert "expand_query" in doc
