"""Tests for the config-driven ARIEL entry_url egress transform.

Covers ``build_entry_url`` and the ``entry_url`` injection into ``serialize_entry``.
Facility-neutral: uses a GENERIC template and generic ids only — never ALS
hosts/strings (OSPREY core stays facility-agnostic; the ALS value lives in
als-profiles config).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel import server
from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from tests.mcp_server.ariel.conftest import get_tool_fn, make_mock_entry

# A generic, non-ALS template. The real ALS value lives in als-profiles config.
GENERIC_TEMPLATE = "https://logbook.example/olog.php?id={entry_id}"


def _fake_config(template):
    """Return a get_config_value side_effect that supplies the entry_url template.

    Only ``ariel.entry_url_template`` is answered with ``template``; every other
    key (e.g. ``system.timezone`` used by to_facility_iso) falls through to the
    caller's default so unrelated config reads keep working.
    """

    def _side_effect(path, default=None, config_path=None):
        if path == "ariel.entry_url_template":
            return template
        return default

    return _side_effect


# ---------------------------------------------------------------------------
# build_entry_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_native_source_system_constant():
    """The module names the ARIEL-native source_system sentinel."""
    assert server.ARIEL_NATIVE_SOURCE_SYSTEM == "ARIEL MCP"


@pytest.mark.unit
def test_build_entry_url_renders_with_template():
    """FR1/FR2: a facility entry renders the template with its id."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        url = server.build_entry_url("175353", "Facility eLog")
    assert url == "https://logbook.example/olog.php?id=175353"


@pytest.mark.unit
def test_build_entry_url_source_system_defaults_to_facility():
    """A facility entry with no explicit source_system still renders (default None)."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        url = server.build_entry_url("42")
    assert url == "https://logbook.example/olog.php?id=42"


@pytest.mark.unit
def test_build_entry_url_url_encodes_id():
    """FR4: entry_id is URL-encoded with safe='' when substituted."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        url = server.build_entry_url("a b/c?d&e", "Facility eLog")
    assert url == "https://logbook.example/olog.php?id=a%20b%2Fc%3Fd%26e"


@pytest.mark.unit
def test_build_entry_url_none_when_template_unset():
    """FR3: no template configured -> None."""
    with patch("osprey.utils.config.get_config_value", _fake_config(None)):
        assert server.build_entry_url("175353", "Facility eLog") is None


@pytest.mark.unit
def test_build_entry_url_none_for_native_source():
    """FR7: ARIEL-native entries emit no url even with a template set."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        assert server.build_entry_url("ariel-deadbeef", "ARIEL MCP") is None


@pytest.mark.unit
@pytest.mark.parametrize("empty", ["", "   ", None])
def test_build_entry_url_none_for_empty_id(empty):
    """An empty/blank entry_id yields no url."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        assert server.build_entry_url(empty, "Facility eLog") is None


@pytest.mark.unit
def test_build_entry_url_fails_safe_when_config_unavailable():
    """FR6: if config resolution itself raises (e.g. no config loaded), degrade
    to None instead of crashing the per-entry read hot path."""

    def _boom(path, default=None, config_path=None):
        raise FileNotFoundError("No config.yml found in current directory")

    with patch("osprey.utils.config.get_config_value", _boom):
        assert server.build_entry_url("175353", "Facility eLog") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_template",
    [
        "https://logbook.example/olog.php?id={wrong}",  # renamed placeholder -> KeyError
        "https://logbook.example/olog.php?id={}",  # positional field -> IndexError
    ],
)
def test_build_entry_url_malformed_template_fails_safe(bad_template):
    """FR6: a malformed template returns None and never raises."""
    with patch("osprey.utils.config.get_config_value", _fake_config(bad_template)):
        assert server.build_entry_url("175353", "Facility eLog") is None


# ---------------------------------------------------------------------------
# serialize_entry injection
# ---------------------------------------------------------------------------


def _entry(entry_id="e1", source_system="Facility eLog"):
    from datetime import datetime

    return {
        "entry_id": entry_id,
        "source_system": source_system,
        "timestamp": datetime(2024, 1, 15, 10, 30, 0),
        "author": "Alice",
        "raw_text": "beam loss on the north arc",
        "summary": None,
    }


@pytest.mark.unit
def test_serialize_entry_includes_entry_url_when_configured():
    """FR2: serialize_entry carries entry_url for a facility entry."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(_entry(entry_id="175353"))
    assert result["entry_url"] == "https://logbook.example/olog.php?id=175353"


@pytest.mark.unit
def test_serialize_entry_omits_entry_url_when_unset():
    """FR3: no template -> no entry_url key at all."""
    with patch("osprey.utils.config.get_config_value", _fake_config(None)):
        result = server.serialize_entry(_entry())
    assert "entry_url" not in result


@pytest.mark.unit
def test_serialize_entry_omits_entry_url_for_native():
    """FR7: an ARIEL-native entry carries no entry_url even when configured."""
    with patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(_entry(source_system="ARIEL MCP"))
    assert "entry_url" not in result


# ---------------------------------------------------------------------------
# server-wide guardrail (FR5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_server_instructions_forbid_fabricated_urls():
    """FR5: the server-wide instructions tell the agent to use entry_url and
    never construct logbook URLs itself."""
    instructions = server.mcp.instructions.lower()
    assert "entry_url" in instructions
    assert "never" in instructions


# ---------------------------------------------------------------------------
# Remaining egress surfaces: entry_get, sql_query rows, entry_publish,
# and the entry_create exclusion guard. (Task 1.3)
# ---------------------------------------------------------------------------


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()


def _patch_service(mock_service):
    return patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    )


@pytest.mark.unit
async def test_entry_get_includes_entry_url(tmp_path, monkeypatch):
    """FR1: entry_get's inline dict carries entry_url for a facility entry."""
    _setup_registry(tmp_path, monkeypatch)
    entry = make_mock_entry(entry_id="175353", source_system="Facility eLog")

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry

    with (
        _patch_service(mock_service),
        patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_get())
        result = await fn(entry_id="175353")

    data = json.loads(result)
    assert data["entry_url"] == "https://logbook.example/olog.php?id=175353"


@pytest.mark.unit
async def test_entry_get_omits_entry_url_for_native(tmp_path, monkeypatch):
    """FR7: entry_get on an ARIEL-native entry emits no entry_url."""
    _setup_registry(tmp_path, monkeypatch)
    entry = make_mock_entry(entry_id="ariel-deadbeef", source_system="ARIEL MCP")

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry

    with (
        _patch_service(mock_service),
        patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_get())
        result = await fn(entry_id="ariel-deadbeef")

    assert "entry_url" not in json.loads(result)


@pytest.mark.unit
async def test_sql_query_rows_gain_entry_url(tmp_path, monkeypatch):
    """FR8: sql_query rows selecting entry_id gain entry_url; aggregate rows
    without an entry_id column are unchanged."""
    _setup_registry(tmp_path, monkeypatch)

    mock_rows = [
        {"entry_id": "e1", "source_system": "Facility eLog", "author": "Alice"},
        {"entry_id": "n1", "source_system": "ARIEL MCP", "author": "Bot"},  # native
        {"count": 5},  # aggregate row — no entry_id column
    ]
    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        _patch_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(return_value=mock_rows),
        ),
        patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_sql_query())
        result = await fn(query="SELECT entry_id, source_system, author FROM enhanced_entries")

    rows = json.loads(result)["rows"]
    assert rows[0]["entry_url"] == "https://logbook.example/olog.php?id=e1"
    assert "entry_url" not in rows[1]  # native entry
    assert "entry_url" not in rows[2]  # aggregate row


@pytest.mark.unit
async def test_entry_publish_includes_entry_url(tmp_path, monkeypatch):
    """FR8/CF-4: entry_publish result carries entry_url for the facility id."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="Facility eLog",
        sync_status=SyncStatus.SYNCED,
        message="Published successfully",
    )

    with (
        _patch_service(mock_service),
        patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_publish())
        result = await fn(entry_id="e1", logbook="Operations")

    assert json.loads(result)["entry_url"] == "https://logbook.example/olog.php?id=published-001"


@pytest.mark.unit
async def test_entry_create_emits_no_entry_url(tmp_path, monkeypatch):
    """MI-1: entry_create (native/draft ids) never emits entry_url, even when a
    template is configured — its entries are not in the facility logbook."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with (
        _patch_service(mock_service),
        patch("osprey.utils.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_create())
        result = await fn(subject="s", details="d", draft=False)

    assert "entry_url" not in json.loads(result)


def _import_entry_get():
    from osprey.mcp_server.ariel.tools.entry import entry_get

    return entry_get


def _import_entry_create():
    from osprey.mcp_server.ariel.tools.entry import entry_create

    return entry_create


def _import_sql_query():
    from osprey.mcp_server.ariel.tools.sql_query import sql_query

    return sql_query


def _import_entry_publish():
    from osprey.mcp_server.ariel.tools.publish import entry_publish

    return entry_publish
