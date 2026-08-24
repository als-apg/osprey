"""Tests for the ``get_plan_source`` Bluesky read tool.

Three properties are load-bearing here and are asserted on the request path the
tool builds, not on the response it relays:

1. **``max_chars`` is clamped before the request.** The bridge route caps
   ``max_chars`` server-side and answers an out-of-range ask with a 422. The
   tool clamps into [1, 200000] first, so an agent asking for a whole file by
   sending a huge number gets truncated source rather than a refusal it cannot
   act on.
2. **The plan name is URL-quoted.** A name carrying ``/`` or a space must not
   be injected into the path — it is quoted with ``safe=''`` so it stays one
   path segment.
3. **A 404 names the legacy ``bluesky.plan_module`` case.** Plans supplied
   through that contract DO appear in ``list_plans`` while this endpoint's
   layer scan skips them, so a bare "unknown plan" message would be actively
   misleading.

The HTTP boundary (``_http_get_json``, imported from
``osprey.mcp_server.bluesky.server_context``) is patched, so these run with no
Bluesky bridge process and no network.
"""

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from osprey.mcp_server.bluesky.tools import read_tools
from osprey.services.bluesky_bridge.app import (
    _SOURCE_TRUNCATE_CHARS,
    _SOURCE_TRUNCATE_CHARS_MAX,
)
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.read_tools"

_SOURCE = "from bluesky.plans import count\n\n\ndef sniff(detectors):\n    yield from count(...)\n"


def _fn():
    return get_tool_fn(read_tools.get_plan_source)


def _body(**overrides):
    body = {
        "name": "sniff",
        "provenance": "shipped",
        "validated": True,
        "truncated": False,
        "source": _SOURCE,
    }
    body.update(overrides)
    return body


def _requested_max_chars(mock) -> int:
    """The ``max_chars`` the tool actually put on the wire."""
    query = parse_qs(urlparse(mock.call_args.args[0]).query)
    return int(query["max_chars"][0])


# ── success ──────────────────────────────────────────────────────────────
async def test_returns_source_provenance_validated_and_truncated():
    with patch(f"{_MOD}._http_get_json", return_value=(200, _body())) as m:
        result = await _fn()(name="sniff")
    assert m.call_args.args[0] == "/plans/sniff/source?max_chars=4000"
    data = extract_response_dict(result)
    assert data["name"] == "sniff"
    assert data["source"] == _SOURCE
    assert data["provenance"] == "shipped"
    assert data["validated"] is True
    assert data["truncated"] is False


async def test_relays_session_tier_validated_false():
    """A session-tier plan that has not passed validation must read false, not
    be smoothed into the shipped tiers' unconditional true."""
    body = _body(provenance="session", validated=False)
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn()(name="sniff")
    data = extract_response_dict(result)
    assert data["provenance"] == "session"
    assert data["validated"] is False


# ── truncation ───────────────────────────────────────────────────────────
async def test_small_max_chars_is_forwarded_and_truncation_relayed():
    body = _body(source=_SOURCE[:10], truncated=True)
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn()(name="sniff", max_chars=10)
    assert _requested_max_chars(m) == 10
    data = extract_response_dict(result)
    assert len(data["source"]) == 10
    assert data["truncated"] is True


# ── mirrored bounds ──────────────────────────────────────────────────────
def test_clamp_bounds_track_the_bridges_own_bounds():
    """The tool's bounds are hand-copied from the bridge route's. Nothing else
    pins them together: if the bridge lowered its ceiling, the tool would keep
    clamping to the old one and every large ask would start returning the very
    422 the clamp exists to prevent, silently."""
    assert read_tools._SOURCE_DEFAULT_MAX_CHARS == _SOURCE_TRUNCATE_CHARS
    assert read_tools._SOURCE_MAX_MAX_CHARS == _SOURCE_TRUNCATE_CHARS_MAX


# ── clamping ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        (10**9, 200_000),
        (200_001, 200_000),
        (200_000, 200_000),
        (1, 1),
        (0, 1),
        (-5, 1),
    ],
)
async def test_max_chars_is_clamped_into_the_bridges_range(asked, expected):
    """Out-of-range asks are clamped BEFORE the request, so the route's 422
    never reaches the agent."""
    with patch(f"{_MOD}._http_get_json", return_value=(200, _body())) as m:
        await _fn()(name="sniff", max_chars=asked)
    assert _requested_max_chars(m) == expected


def _route_like(path: str, *_args, **_kwargs):
    """Stand in for the bridge route's own bounds check on ``max_chars``.

    The real route answers an out-of-range ask with a 422, so this stub must
    too — otherwise a test asserting "no 422 reaches the agent" would pass with
    the clamp deleted.
    """
    asked = int(parse_qs(urlparse(path).query)["max_chars"][0])
    if not 1 <= asked <= 200_000:
        return (422, {"detail": f"max_chars must be between 1 and 200000, got {asked}"})
    return (200, _body(truncated=True))


async def test_huge_max_chars_surfaces_no_422():
    """The clamp is what keeps the 422 branch unreachable — the patched
    boundary enforces the route's real bounds, so this fails if the clamp
    goes away."""
    with patch(f"{_MOD}._http_get_json", side_effect=_route_like):
        result = await _fn()(name="sniff", max_chars=10**9)
    assert extract_response_dict(result)["source"] == _SOURCE


# ── name quoting ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "quoted"),
    [
        ("../etc/passwd", "..%2Fetc%2Fpasswd"),
        ("dir/plan", "dir%2Fplan"),
        ("plan with spaces", "plan%20with%20spaces"),
        ("plan?max_chars=99", "plan%3Fmax_chars%3D99"),
        ("plan&x=1", "plan%26x%3D1"),
    ],
)
async def test_plan_name_is_url_quoted_into_a_single_path_segment(name, quoted):
    with patch(f"{_MOD}._http_get_json", return_value=(200, _body(name=name))) as m:
        await _fn()(name=name)
    path = m.call_args.args[0]
    assert path == f"/plans/{quoted}/source?max_chars=4000"
    # The raw name never reaches the path unencoded, and the query still holds
    # exactly the one parameter this tool sets.
    assert path.count("/") == 3
    assert parse_qs(urlparse(path).query) == {"max_chars": ["4000"]}


# ── refusals ─────────────────────────────────────────────────────────────
async def test_404_names_the_legacy_plan_module_case():
    with patch(
        f"{_MOD}._http_get_json",
        return_value=(404, {"detail": "no source file found for plan 'sniff'"}),
    ):
        with assert_raises_error(error_type="plan_source_unavailable") as ctx:
            await _fn()(name="sniff")
    envelope = ctx["envelope"]
    text = envelope["error_message"] + " " + " ".join(envelope.get("suggestions", []))
    assert "plan_module" in text
    assert "list_plans" in text


async def test_bridge_error_is_relayed():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn()(name="sniff")
    assert "boom" in ctx["envelope"]["error_message"]
