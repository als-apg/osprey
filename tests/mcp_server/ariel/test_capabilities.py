"""Tests for the capabilities MCP tool."""

import json

from osprey.mcp_server.ariel.server_context import initialize_ariel_context, reset_ariel_context
from osprey.registry import get_registry
from osprey.utils.workspace import reset_config_cache
from tests.mcp_server.ariel.conftest import get_tool_fn


def _get_capabilities():
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    return get_tool_fn(capabilities)


def _setup_registry(tmp_path, monkeypatch, search_modules=None, vocabulary=None, extra=None):
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
        extra: Optional further ``ariel`` blocks (``enhancement_modules``,
            ``attachments``) merged into the section as given.
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
    if extra:
        ariel.update(extra)
    config = json.dumps({"ariel": ariel})
    (tmp_path / "config.yml").write_text(config)
    get_registry().initialize()
    initialize_ariel_context()


async def test_capabilities_returns_modules(tmp_path, monkeypatch):
    """Capabilities returns enabled search modules."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert not data.get("error", False)
    assert "keyword" in data["enabled_search_modules"]
    assert "semantic" in data["enabled_search_modules"]


async def test_capabilities_includes_search_modes(tmp_path, monkeypatch):
    """Capabilities advertises every registered, enabled search module."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" in data["search_modes"]


async def test_capabilities_omits_sql_query_mode(tmp_path, monkeypatch):
    """``sql_query`` is a tool, not a mode, so it never appears in the mode list."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "sql_query" not in data["search_modes"]


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


async def test_capabilities_docstring_explains_the_vocabulary_block():
    """The docstring is a prompt surface: it must name what it now returns."""
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    doc = get_tool_fn(capabilities).__doc__ or ""

    assert "vocabulary" in doc
    assert "expand_query" in doc


# ---------------------------------------------------------------------------
# attachments block
# ---------------------------------------------------------------------------

_ALL_SEARCH_MODULES = {
    "keyword": {"enabled": True},
    "semantic": {"enabled": True, "model": "nomic-embed-text"},
    "hybrid": {"enabled": True},
}
_ATTACHMENT_KEYS = {
    "copy_on_ingest",
    "formats",
    "view",
    "captions",
    "picture_search",
    "picture_search_unavailable",
}


def _picture_config(*, image_embedding=True, hybrid=True, view=None):
    """Search modules and ``extra`` blocks for one attachments scenario."""
    search_modules = {
        **_ALL_SEARCH_MODULES,
        "hybrid": {"enabled": hybrid},
    }
    extra: dict = {
        "enhancement_modules": {
            "image_embedding": {"enabled": image_embedding},
            "image_caption": {"enabled": True},
        },
    }
    if view is not None:
        extra["attachments"] = {"view": {"enabled": view}}
    return search_modules, extra


def _route_capabilities(config):
    """``GET /api/capabilities`` against a routes-only app carrying ``config``."""
    from unittest.mock import MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from osprey.interfaces.ariel.api import routes

    app = FastAPI()
    app.include_router(routes.router)
    service = MagicMock()
    service.config = config
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    response = TestClient(app).get("/api/capabilities")
    assert response.status_code == 200
    return response.json()


async def test_capabilities_reports_the_attachments_block(tmp_path, monkeypatch):
    """The tool forwards the attachments block with exactly its six fields."""
    from osprey.services.ariel_search.search import image_lane

    image_lane._reset_state()
    search_modules, extra = _picture_config()
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)

    data = json.loads(await _get_capabilities()())

    block = data["attachments"]
    assert set(block) == _ATTACHMENT_KEYS
    assert block["view"] is True
    assert block["captions"] is True
    assert block["picture_search"] is True
    assert block["picture_search_unavailable"] is None
    assert "png" in block["formats"]["viewable"]
    assert "pdf" in block["formats"]["reserved"]


async def test_capabilities_attachments_match_the_web_surface(tmp_path, monkeypatch):
    """The MCP tool and ``/api/capabilities`` report one block for one config."""
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    search_modules, extra = _picture_config()
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)

    data = json.loads(await _get_capabilities()())
    web = _route_capabilities(get_ariel_context().config)

    assert data["attachments"] == web["attachments"]


async def test_capabilities_picture_search_needs_image_embedding(tmp_path, monkeypatch):
    """No image embeddings means no picture search, on both surfaces."""
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    search_modules, extra = _picture_config(image_embedding=False)
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)

    data = json.loads(await _get_capabilities()())

    assert data["attachments"]["picture_search"] is False
    assert _route_capabilities(get_ariel_context().config)["attachments"] == data["attachments"]


async def test_capabilities_picture_search_needs_hybrid(tmp_path, monkeypatch):
    """Picture search routes through hybrid, so a disabled hybrid module disables it."""
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    search_modules, extra = _picture_config(hybrid=False)
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)

    data = json.loads(await _get_capabilities()())

    assert data["attachments"]["picture_search"] is False
    assert "hybrid" not in data["search_modes"]
    assert _route_capabilities(get_ariel_context().config)["attachments"] == data["attachments"]


async def test_capabilities_reports_view_disabled(tmp_path, monkeypatch):
    """``view.enabled: false`` flips only ``view``; the rest of the block is unchanged."""
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    search_modules, extra = _picture_config()
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)
    enabled = json.loads(await _get_capabilities()())["attachments"]

    reset_ariel_context()
    reset_config_cache()
    search_modules, extra = _picture_config(view=False)
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)
    disabled = json.loads(await _get_capabilities()())["attachments"]

    assert enabled["view"] is True
    assert disabled["view"] is False
    assert {k: v for k, v in disabled.items() if k != "view"} == {
        k: v for k, v in enabled.items() if k != "view"
    }
    assert _route_capabilities(get_ariel_context().config)["attachments"] == disabled


async def test_capabilities_docstring_defines_the_attachments_block():
    """Every attachments field an agent reads is defined in the tool docstring."""
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    doc = get_tool_fn(capabilities).__doc__ or ""

    for field in ("attachments", *_ATTACHMENT_KEYS, "viewable", "reserved"):
        assert field in doc, field


async def test_capabilities_keyset_differs_from_b1_only_by_attachments(keyset_harness):
    """Against the B1 golden, every new path sits under ``attachments`` and none is lost."""
    from tests.mcp_server.ariel.test_tool_keysets import KEYSET_DIR, keyset, run_tool

    actual = keyset(json.loads(await run_tool("capabilities", keyset_harness)))
    golden = json.loads((KEYSET_DIR / "capabilities.json").read_text())

    added = set(actual) - set(golden)
    assert added, "the attachments block must reach the payload"
    assert all(p == "attachments" or p.startswith("attachments.") for p in added), added
    assert set(golden) <= set(actual)
    assert {p: actual[p] for p in golden} == golden
    assert {"attachments.view", "attachments.picture_search"} <= added


async def test_capabilities_reports_the_picture_lanes_last_failure(tmp_path, monkeypatch):
    """``picture_search_unavailable`` forwards the lane's last reason on both surfaces."""
    from osprey.mcp_server.ariel.server_context import get_ariel_context
    from osprey.services.ariel_search.search import image_lane

    monkeypatch.setattr(image_lane, "_last_reason", "model")
    search_modules, extra = _picture_config()
    _setup_registry(tmp_path, monkeypatch, search_modules=search_modules, extra=extra)

    data = json.loads(await _get_capabilities()())

    assert data["attachments"]["picture_search"] is True
    assert data["attachments"]["picture_search_unavailable"] == "model"
    assert _route_capabilities(get_ariel_context().config)["attachments"] == data["attachments"]
