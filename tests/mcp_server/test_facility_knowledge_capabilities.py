"""Tests for the facility knowledge MCP tool: capabilities.

Tests run against an in-process OKFBundle backed by a tmp_path fixture bundle
— no live MCP transport needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

# ---------------------------------------------------------------------------
# Bundle fixtures (mirrors test_facility_knowledge_tools.py conventions)
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture
def fixture_bundle(tmp_path: Path) -> Path:
    """Minimal OKF bundle with two distinct concept types."""
    root = tmp_path / "bundle"
    _write(
        root / "index.md",
        """\
        # Facility Bundle
        * [Accelerator Overview](/accelerator_overview.md) - High-level accelerator overview.
        """,
    )
    _write(
        root / "accelerator_overview.md",
        """\
        ---
        type: facility_overview
        title: Accelerator Overview
        description: Overview of the accelerator complex.
        ---

        The facility uses a synchrotron ring to produce bright X-ray light.
        """,
    )
    _write(
        root / "tables" / "beam_params.md",
        """\
        ---
        type: data_table
        title: Beam Parameters
        description: Live beam parameter PVs from the EPICS control system.
        tags: [epics, beam]
        ---

        Contains PV names for all beam parameters.
        """,
    )
    return root


@pytest.fixture(autouse=True)
def _patch_bundle(fixture_bundle: Path, monkeypatch):
    """Inject the fixture bundle into the server module's ``_bundle`` global."""
    import osprey.mcp_server.facility_knowledge.server as srv
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    monkeypatch.setattr(srv, "_bundle", OKFBundle(fixture_bundle))


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    """capabilities() returns bundle metadata without reading document bodies."""

    @pytest.mark.asyncio
    async def test_returns_required_fields(self):
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())

        assert "bundle_path" in result
        assert "count" in result
        assert "types" in result
        assert "write_enabled" in result

    @pytest.mark.asyncio
    async def test_count_matches_concepts(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import capabilities, list_concepts

        cap = json.loads(await get_tool_fn(capabilities)())
        listing = json.loads(await get_tool_fn(list_concepts)())

        assert cap["count"] == listing["count"]

    @pytest.mark.asyncio
    async def test_types_are_sorted(self):
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())
        types = result["types"]

        assert types == sorted(types)

    @pytest.mark.asyncio
    async def test_types_contain_fixture_concept_types(self):
        """The fixture bundle has facility_overview and data_table — both must appear."""
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())
        types = set(result["types"])

        # data_table < facility_overview alphabetically, both must be present
        assert "data_table" in types
        assert "facility_overview" in types

    @pytest.mark.asyncio
    async def test_types_are_unique(self):
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())
        types = result["types"]

        assert len(types) == len(set(types))

    @pytest.mark.asyncio
    async def test_write_enabled_is_true(self):
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())

        assert result["write_enabled"] is True

    @pytest.mark.asyncio
    async def test_bundle_path_is_string(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())

        assert isinstance(result["bundle_path"], str)
        assert result["bundle_path"] == str(fixture_bundle)

    @pytest.mark.asyncio
    async def test_empty_bundle_returns_zero_count_and_empty_types(
        self, tmp_path: Path, monkeypatch
    ):
        """An empty bundle directory must return count=0 and types=[] — NOT an error."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import capabilities
        from osprey.services.facility_knowledge.okf.bundle import OKFBundle

        empty_root = tmp_path / "empty_bundle"
        empty_root.mkdir()
        monkeypatch.setattr(srv, "_bundle", OKFBundle(empty_root))

        result = json.loads(await get_tool_fn(capabilities)())

        assert result["count"] == 0
        assert result["types"] == []

    @pytest.mark.asyncio
    async def test_returns_error_when_bundle_not_initialised(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import capabilities

        monkeypatch.setattr(srv, "_bundle", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(capabilities)()

    @pytest.mark.asyncio
    async def test_internal_error_when_list_concepts_raises(self, monkeypatch):
        """capabilities returns internal_error when bundle.list_concepts raises unexpectedly."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import capabilities

        monkeypatch.setattr(
            srv._bundle, "list_concepts", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        with assert_raises_error(error_type="internal_error"):
            await get_tool_fn(capabilities)()


# ---------------------------------------------------------------------------
# Registry permission check
# ---------------------------------------------------------------------------


class TestRegistryPermissions:
    """capabilities must be listed in the osprey_facility_knowledge permissions_allow."""

    def test_capabilities_in_permissions_allow(self):
        from osprey.registry.mcp import FRAMEWORK_SERVERS

        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert "capabilities" in sdef.permissions_allow, (
            "'capabilities' is missing from osprey_facility_knowledge.permissions_allow "
            f"(current: {sdef.permissions_allow!r})"
        )
