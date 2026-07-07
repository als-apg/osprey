"""Unit tests for the `read_scan_data` MCP tool's bounded-read semantics (task 2.12).

Complements `test_scan_read_tools.py`'s basic success/error-envelope coverage
by focusing specifically on the bounded-read shape the bridge's
`GET /runs/{id}/data` route (task 2.2, backed by `live_rows.py`) produces:
`max_rows` caps, `row_count` as the true total, `truncated` flipping
correctly, `offset`/`tail` pagination, the minimal empty-stream shape, and
`partial: true` mid-run. The HTTP boundary (`_http_get_json`) is patched here
(phoebus pattern) — this file exercises the tool's own param-passing and
response-surfacing, not the network; the real end-to-end route is covered by
the non-patched integration test in `test_read_bounded.py`.
"""

from unittest.mock import patch

import pytest

from osprey.mcp_server.scan.tools import read_tools
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.scan.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


# =========================================================================
# max_rows caps the returned rows
# =========================================================================


async def test_max_rows_caps_the_returned_row_window():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[0.0], [1.0]],
        "row_count": 5,
        "truncated": True,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=2)

    assert "max_rows=2" in m.call_args.args[0]
    data = extract_response_dict(result)
    assert len(data["rows"]) == 2


# =========================================================================
# row_count reflects the true total, independent of the returned window size
# =========================================================================


async def test_row_count_reflects_true_total_not_window_size():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[0.0], [1.0]],
        "row_count": 500,
        "truncated": True,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=2)

    data = extract_response_dict(result)
    assert data["row_count"] == 500
    assert len(data["rows"]) == 2


# =========================================================================
# truncated flips: False when the window covers everything, True otherwise
# =========================================================================


async def test_truncated_false_when_window_covers_every_row():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[0.0], [1.0]],
        "row_count": 2,
        "truncated": False,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=100)

    assert extract_response_dict(result)["truncated"] is False


async def test_truncated_true_when_window_omits_rows():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[0.0], [1.0]],
        "row_count": 5,
        "truncated": True,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=2)

    assert extract_response_dict(result)["truncated"] is True


# =========================================================================
# offset / tail pagination: the tool must build the right query string
# =========================================================================


async def test_offset_is_passed_through_as_a_forward_page():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("read_scan_data")(run_id="abc123", max_rows=10, offset=20)

    url = m.call_args.args[0]
    assert "max_rows=10" in url
    assert "offset=20" in url
    assert "tail=true" not in url


async def test_tail_is_passed_through_without_an_offset():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("read_scan_data")(run_id="abc123", max_rows=10, tail=True)

    url = m.call_args.args[0]
    assert "tail=true" in url
    assert "offset=" not in url


async def test_tail_with_offset_are_both_passed_through():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("read_scan_data")(run_id="abc123", max_rows=10, offset=5, tail=True)

    url = m.call_args.args[0]
    assert "max_rows=10" in url
    assert "offset=5" in url
    assert "tail=true" in url


async def test_offset_defaults_to_omitted_from_the_query_string():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("read_scan_data")(run_id="abc123")

    url = m.call_args.args[0]
    assert "offset=" not in url
    assert "tail=" not in url
    assert "max_rows=100" in url  # the tool's own default


async def test_paginated_rows_surface_unmodified_from_the_bridge_body():
    """The tool must not re-slice or otherwise alter the bridge's own window."""
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[2.0], [3.0]],
        "row_count": 5,
        "truncated": True,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=2, offset=2)

    data = extract_response_dict(result)
    assert data["rows"] == [[2.0], [3.0]]


# =========================================================================
# Empty stream: minimal shape, no extra keys
# =========================================================================


async def test_empty_stream_returns_the_minimal_shape():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})):
        result = await _fn("read_scan_data")(run_id="abc123")

    assert extract_response_dict(result) == {"columns": [], "rows": []}


# =========================================================================
# Mid-run: partial: true surfaces through
# =========================================================================


async def test_mid_run_surfaces_partial_true():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
        "partial": True,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123")

    assert extract_response_dict(result)["partial"] is True


async def test_completed_run_omits_partial_key():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("read_scan_data")(run_id="abc123")

    assert "partial" not in extract_response_dict(result)


# =========================================================================
# 409: no run_uid yet -> a distinct, clear error (not a generic bridge error)
# =========================================================================


async def test_no_run_uid_yet_returns_a_distinct_error_type():
    with patch(
        f"{_MOD}._http_get_json",
        return_value=(409, {"detail": "run 'abc123' has not started; no data yet"}),
    ):
        with assert_raises_error(error_type="scan_data_not_ready") as ctx:
            await _fn("read_scan_data")(run_id="abc123")

    assert "has not started" in ctx["envelope"]["error_message"]
