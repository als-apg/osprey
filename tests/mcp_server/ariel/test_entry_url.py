"""Tests for the config-driven ARIEL entry_url egress transform.

Covers ``build_entry_url`` and the ``entry_url`` injection into ``serialize_entry``.
The template is a configuration value with no shipped default, so these cases
use a generic one and generic ids: a host baked in here would be a host no
deployment has.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel import server
from osprey.services.ariel_search.models import DEFAULT_LISTING_TEXT_CHARS
from tests.mcp_server.ariel.conftest import (
    attach_fake_attachment_reader,
    get_tool_fn,
    make_mock_entry,
    patch_ariel_service,
)

# A generic template; each deployment supplies its own in configuration.
GENERIC_TEMPLATE = "https://logbook.example/olog.php?id={entry_id}"


def _fake_config(template):
    """Return a get_config_value side_effect that supplies the entry_url template.

    Only ``ariel.entry_url_template`` is answered with ``template``; every other
    key (e.g. ``system.timezone`` used by to_facility_iso) falls through to the
    caller's default so unrelated config reads keep working.
    """

    def _side_effect(path, default=None, _config_path=None):
        if path == "ariel.entry_url_template":
            return template
        return default

    return _side_effect


# ---------------------------------------------------------------------------
# build_entry_url
# ---------------------------------------------------------------------------


def test_native_source_system_constant():
    """The module names the ARIEL-native source_system sentinel."""
    assert server.ARIEL_NATIVE_SOURCE_SYSTEM == "ARIEL MCP"


@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [
        # FR1/FR2: a facility entry renders the template with its id.
        pytest.param("175353", "https://logbook.example/olog.php?id=175353", id="renders-id"),
        # FR4: entry_id is URL-encoded with safe='' when substituted.
        pytest.param(
            "a b/c?d&e",
            "https://logbook.example/olog.php?id=a%20b%2Fc%3Fd%26e",
            id="url-encodes-id",
        ),
    ],
)
def test_build_entry_url_substitutes_the_id_into_the_template(entry_id, expected):
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        url = server.build_entry_url(entry_id, "Facility eLog")
    assert url == expected


def test_build_entry_url_source_system_defaults_to_facility():
    """A facility entry with no explicit source_system still renders (default None)."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        url = server.build_entry_url("42")
    assert url == "https://logbook.example/olog.php?id=42"


def test_build_entry_url_none_when_template_unset():
    """FR3: no template configured -> None."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(None)):
        assert server.build_entry_url("175353", "Facility eLog") is None


def test_build_entry_url_none_for_native_source():
    """FR7: ARIEL-native entries emit no url even with a template set."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        assert server.build_entry_url("ariel-deadbeef", "ARIEL MCP") is None


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_build_entry_url_none_for_empty_id(empty):
    """An empty/blank entry_id yields no url."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        assert server.build_entry_url(empty, "Facility eLog") is None


def test_build_entry_url_fails_safe_when_config_unavailable():
    """FR6: if config resolution itself raises (e.g. no config loaded), degrade
    to None instead of crashing the per-entry read hot path."""

    def _boom(_path, _default=None, _config_path=None):
        raise FileNotFoundError("No config.yml found in current directory")

    with patch("osprey_connectors.config.get_config_value", _boom):
        assert server.build_entry_url("175353", "Facility eLog") is None


@pytest.mark.parametrize(
    "bad_template",
    [
        "https://logbook.example/olog.php?id={wrong}",  # renamed placeholder -> KeyError
        "https://logbook.example/olog.php?id={}",  # positional field -> IndexError
    ],
)
def test_build_entry_url_malformed_template_fails_safe(bad_template):
    """FR6: a malformed template returns None and never raises."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(bad_template)):
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


def test_serialize_entry_includes_entry_url_when_configured():
    """FR2: serialize_entry carries entry_url for a facility entry."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(
            _entry(entry_id="175353"),
            text_limit=DEFAULT_LISTING_TEXT_CHARS,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert result["entry_url"] == "https://logbook.example/olog.php?id=175353"


def test_serialize_entry_omits_entry_url_when_unset():
    """FR3: no template -> no entry_url key at all."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(None)):
        result = server.serialize_entry(
            _entry(),
            text_limit=DEFAULT_LISTING_TEXT_CHARS,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert "entry_url" not in result


def test_serialize_entry_omits_entry_url_for_native():
    """FR7: an ARIEL-native entry carries no entry_url even when configured."""
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(
            _entry(source_system="ARIEL MCP"),
            text_limit=DEFAULT_LISTING_TEXT_CHARS,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert "entry_url" not in result


def test_serialize_entry_omits_entry_url_for_local_only_web_entry():
    """C-URL: a local-only ``OSPREY Logbook Web`` save is not in the facility logbook, so
    serialize_entry emits no entry_url even when a template is configured."""
    entry = {**_entry(entry_id="ariel-web-1", source_system="OSPREY Logbook Web")}
    entry["metadata"] = {"sync_status": "local_only"}
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(
            entry,
            text_limit=DEFAULT_LISTING_TEXT_CHARS,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert "entry_url" not in result


def test_serialize_entry_keeps_entry_url_for_synced_web_entry():
    """A published (synced) ``OSPREY Logbook Web`` entry is in the facility logbook."""
    entry = {**_entry(entry_id="175353", source_system="OSPREY Logbook Web")}
    entry["metadata"] = {"sync_status": "synced"}
    with patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)):
        result = server.serialize_entry(
            entry,
            text_limit=DEFAULT_LISTING_TEXT_CHARS,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert result["entry_url"] == "https://logbook.example/olog.php?id=175353"


# ---------------------------------------------------------------------------
# services.ariel_search.entry_url (C-URL): the shared producer
# ---------------------------------------------------------------------------


def test_server_re_exports_the_shared_producer():
    """server.py re-exports the service-layer producer, not a copy of it."""
    from osprey.services.ariel_search import entry_url

    assert server.build_entry_url is entry_url.build_entry_url
    assert server.is_ariel_only is entry_url.is_ariel_only
    assert server.ARIEL_NATIVE_SOURCE_SYSTEM == entry_url.ARIEL_NATIVE_SOURCE_SYSTEM


@pytest.mark.parametrize(
    ("source_system", "metadata", "expected"),
    [
        ("ARIEL MCP", None, True),  # MCP-native entry
        ("ARIEL MCP", {"sync_status": "synced"}, True),
        ("OSPREY Logbook Web", {"sync_status": "local_only"}, True),  # local-only web save
        ("Generic JSON", {"sync_status": "local_only"}, True),
        ("OSPREY Logbook Web", {"sync_status": "synced"}, False),
        ("OSPREY Logbook Web", {"sync_status": "pending_sync"}, False),
        ("Facility eLog", None, False),
        ("Facility eLog", {}, False),
        (None, None, False),
        ("Facility eLog", "not-a-mapping", False),
    ],
)
def test_is_ariel_only(source_system, metadata, expected):
    """One predicate: local-only sync status, or the ARIEL-native source."""
    from osprey.services.ariel_search.entry_url import is_ariel_only

    assert is_ariel_only(source_system, metadata) is expected


@pytest.mark.parametrize(
    ("template", "source_system", "metadata", "expected"),
    [
        (GENERIC_TEMPLATE, "Facility eLog", None, "https://logbook.example/olog.php?id=7"),
        (GENERIC_TEMPLATE, "ARIEL MCP", None, None),  # MCP native
        (GENERIC_TEMPLATE, "OSPREY Logbook Web", {"sync_status": "local_only"}, None),  # local-only
        (None, "Facility eLog", None, None),  # no template
        ("", "Facility eLog", None, None),  # blank template
    ],
)
def test_service_build_entry_url(template, source_system, metadata, expected):
    """The service producer: template set gives a URL; MCP native, a local-only
    ``OSPREY Logbook Web`` entry and no template give None."""
    from osprey.services.ariel_search.entry_url import build_entry_url

    with patch("osprey_connectors.config.get_config_value", _fake_config(template)):
        assert build_entry_url("7", source_system, metadata=metadata) == expected


def test_service_build_entry_url_metadata_is_keyword_only():
    """``metadata`` is keyword-only so the positional two-arg callers stay valid."""
    from osprey.services.ariel_search.entry_url import build_entry_url

    with pytest.raises(TypeError):
        build_entry_url("7", "OSPREY Logbook Web", {"sync_status": "local_only"})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# server-wide guardrail (FR5)
# ---------------------------------------------------------------------------


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


async def test_entry_get_includes_entry_url(ariel_registry):
    """FR1: entry_get's inline dict carries entry_url for a facility entry."""
    ariel_registry()
    entry = make_mock_entry(entry_id="175353", source_system="Facility eLog")

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry
    attach_fake_attachment_reader(mock_service)

    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_get())
        result = await fn(entry_id="175353")

    data = json.loads(result)
    assert data["entry_url"] == "https://logbook.example/olog.php?id=175353"


async def test_entry_get_omits_entry_url_for_native(ariel_registry):
    """FR7: entry_get on an ARIEL-native entry emits no entry_url."""
    ariel_registry()
    entry = make_mock_entry(entry_id="ariel-deadbeef", source_system="ARIEL MCP")

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry
    attach_fake_attachment_reader(mock_service)

    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_get())
        result = await fn(entry_id="ariel-deadbeef")

    assert "entry_url" not in json.loads(result)


async def test_sql_query_rows_gain_entry_url(ariel_registry):
    """FR8: sql_query rows selecting entry_id gain entry_url; aggregate rows
    without an entry_id column are unchanged."""
    ariel_registry()

    mock_rows = [
        {"entry_id": "e1", "source_system": "Facility eLog", "author": "Alice"},
        {"entry_id": "n1", "source_system": "ARIEL MCP", "author": "Bot"},  # native
        {"count": 5},  # aggregate row — no entry_id column
    ]
    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(return_value=mock_rows),
        ),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_sql_query())
        result = await fn(sql="SELECT entry_id, source_system, author FROM enhanced_entries")

    rows = json.loads(result)["rows"]
    assert rows[0]["entry_url"] == "https://logbook.example/olog.php?id=e1"
    assert "entry_url" not in rows[1]  # native entry
    assert "entry_url" not in rows[2]  # aggregate row


async def test_entry_publish_includes_entry_url(ariel_registry):
    """FR8/CF-4: entry_publish result carries entry_url for the facility id."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    ariel_registry()

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="Facility eLog",
        sync_status=SyncStatus.SYNCED,
        message="Published successfully",
    )

    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_publish())
        result = await fn(entry_id="e1", logbook="Operations")

    assert json.loads(result)["entry_url"] == "https://logbook.example/olog.php?id=published-001"


async def test_entry_create_emits_no_entry_url(ariel_registry):
    """MI-1: entry_create (native/draft ids) never emits entry_url, even when a
    template is configured — its entries are not in the facility logbook."""
    ariel_registry()

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        fn = get_tool_fn(_import_entry_create())
        result = await fn(subject="s", details="d", draft=False)

    assert "entry_url" not in json.loads(result)


# ---------------------------------------------------------------------------
# Local-only entries get no URL from any tool (A15): entry_get and
# entry_publish pass the sync status, sql_query resolves it in one batch.
# ---------------------------------------------------------------------------


async def _entry_get(entry):
    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry
    attach_fake_attachment_reader(mock_service)
    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        return json.loads(await get_tool_fn(_import_entry_get())(entry_id=entry["entry_id"]))


async def test_entry_get_omits_entry_url_for_local_only_web_entry(ariel_registry):
    """A local-only OSPREY Logbook Web save has no facility page, so entry_get emits no URL."""
    ariel_registry()
    entry = make_mock_entry(entry_id="w-local", source_system="OSPREY Logbook Web")
    entry["metadata"] = {"sync_status": "local_only"}

    assert "entry_url" not in await _entry_get(entry)


async def test_entry_get_keeps_entry_url_for_synced_web_entry(ariel_registry):
    """A synced OSPREY Logbook Web entry is in the facility logbook and keeps its URL."""
    ariel_registry()
    entry = make_mock_entry(entry_id="w-synced", source_system="OSPREY Logbook Web")
    entry["metadata"] = {"sync_status": "synced"}

    data = await _entry_get(entry)
    assert data["entry_url"] == "https://logbook.example/olog.php?id=w-synced"


async def _sql_rows(mock_rows, resolved, sql="SELECT entry_id FROM enhanced_entries"):
    """Run sql_query over ``mock_rows`` with the batch lookup returning ``resolved``."""
    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()
    mock_service.readonly_pool = None
    if isinstance(resolved, Exception):
        mock_service.repository.get_entries_by_ids.side_effect = resolved
    else:
        mock_service.repository.get_entries_by_ids.return_value = resolved
    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(return_value=mock_rows),
        ),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        result = await get_tool_fn(_import_sql_query())(sql=sql)
    return json.loads(result)["rows"], mock_service.repository.get_entries_by_ids


async def test_sql_query_id_only_rows_resolve_sync_status_in_one_batch(ariel_registry):
    """A query selecting neither metadata nor source_system: the local-only row
    gets no URL, the facility row keeps one, and both resolve in one lookup."""
    ariel_registry()
    rows, lookup = await _sql_rows(
        [{"entry_id": "w-local"}, {"entry_id": "f-1"}, {"count": 2}],
        [
            {
                "entry_id": "w-local",
                "source_system": "OSPREY Logbook Web",
                "metadata": {"sync_status": "local_only"},
            },
            {"entry_id": "f-1", "source_system": "Facility eLog", "metadata": {}},
        ],
    )

    assert "entry_url" not in rows[0]
    assert rows[1]["entry_url"] == "https://logbook.example/olog.php?id=f-1"
    assert "entry_url" not in rows[2]
    lookup.assert_awaited_once()
    assert sorted(lookup.await_args.args[0]) == ["f-1", "w-local"]
    # The lookup only feeds the URL; the rows the query selected are unchanged.
    assert set(rows[0]) == {"entry_id"}


async def test_sql_query_source_system_without_metadata_resolves_metadata(ariel_registry):
    """A row selecting source_system but not metadata still learns it is local-only."""
    ariel_registry()
    rows, lookup = await _sql_rows(
        [{"entry_id": "w-local", "source_system": "OSPREY Logbook Web"}],
        [
            {
                "entry_id": "w-local",
                "source_system": "OSPREY Logbook Web",
                "metadata": {"sync_status": "local_only"},
            }
        ],
    )

    assert "entry_url" not in rows[0]
    lookup.assert_awaited_once()


async def test_sql_query_rows_with_both_columns_skip_the_lookup(ariel_registry):
    """Rows that already carry metadata and source_system need no lookup."""
    ariel_registry()
    rows, lookup = await _sql_rows(
        [
            {
                "entry_id": "w-local",
                "source_system": "OSPREY Logbook Web",
                "metadata": {"sync_status": "local_only"},
            },
            {"entry_id": "f-1", "source_system": "Facility eLog", "metadata": {}},
        ],
        [],
    )

    assert "entry_url" not in rows[0]
    assert rows[1]["entry_url"] == "https://logbook.example/olog.php?id=f-1"
    lookup.assert_not_awaited()


async def test_sql_query_failed_lookup_omits_unresolved_urls(ariel_registry):
    """When the batch lookup fails the tool still answers, and rows whose sync
    status is unknown get no URL rather than a possibly wrong one."""
    ariel_registry()
    rows, _ = await _sql_rows(
        [
            {"entry_id": "x-1"},
            {"entry_id": "f-1", "source_system": "Facility eLog", "metadata": {}},
        ],
        RuntimeError("db down"),
    )

    assert "entry_url" not in rows[0]
    assert rows[1]["entry_url"] == "https://logbook.example/olog.php?id=f-1"


async def test_entry_publish_omits_entry_url_when_local_only(ariel_registry):
    """A publish that ends local-only has no facility page, so no URL."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    ariel_registry()
    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="w-local",
        source_system="OSPREY Logbook Web",
        sync_status=SyncStatus.LOCAL_ONLY,
        message="Saved locally",
    )

    with (
        patch_ariel_service(mock_service),
        patch("osprey_connectors.config.get_config_value", _fake_config(GENERIC_TEMPLATE)),
    ):
        result = await get_tool_fn(_import_entry_publish())(entry_id="w-local", logbook="Ops")

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
