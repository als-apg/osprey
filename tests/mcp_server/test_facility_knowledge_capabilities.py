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
    async def test_no_write_enabled_field(self):
        """The field was a constant `True` that no consumer read.

        It described nothing: the shipped knowledge persona denies
        `draft_concept` outright, so on that deployment the manifest said writes
        were enabled while every write was refused.
        """
        from osprey.mcp_server.facility_knowledge.server import capabilities

        result = json.loads(await get_tool_fn(capabilities)())

        assert "write_enabled" not in result

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
        monkeypatch.setattr(srv, "_bundle_error", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(capabilities)()

    @pytest.mark.asyncio
    async def test_an_unconfigured_bundle_names_the_config_key(self, monkeypatch):
        """ "Start the server" is the wrong remedy for a config that names no bundle.

        The server IS started; what is missing is `facility_knowledge.bundle_path`.
        One error code for three causes sent the operator to restart something
        that was already running.
        """
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import capabilities

        monkeypatch.setattr(srv, "_bundle", None)
        monkeypatch.setattr(
            srv,
            "_bundle_error",
            (
                "bundle_not_configured",
                "This deployment names no facility knowledge bundle.",
                ["Set `facility_knowledge.bundle_path` ..."],
            ),
        )

        with assert_raises_error(error_type="bundle_not_configured"):
            await get_tool_fn(capabilities)()

    @pytest.mark.asyncio
    async def test_a_failed_load_names_the_resolved_path(self, monkeypatch):
        """A bundle that would not load is a third fault with a third fix."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import capabilities

        monkeypatch.setattr(srv, "_bundle", None)
        monkeypatch.setattr(
            srv,
            "_bundle_error",
            (
                "bundle_load_failed",
                "The facility knowledge bundle at /srv/okf could not be loaded: no index.md",
                ["Check that /srv/okf exists ..."],
            ),
        )

        with assert_raises_error(error_type="bundle_load_failed"):
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


# ---------------------------------------------------------------------------
# create_server records WHICH fault left the bundle unavailable
# ---------------------------------------------------------------------------


class TestBundleFailureCause:
    """Three faults, three fixes — so three error codes, decided at startup.

    ``_get_bundle`` runs long after the config was read and cannot tell an
    absent ``bundle_path`` from one that would not load. ``create_server`` can,
    and is the only place that can, so it records the cause there.
    """

    def _run_create_server(self, monkeypatch, tmp_path: Path, config: dict):
        import osprey.mcp_server.facility_knowledge.server as srv
        import osprey.utils.workspace as workspace

        monkeypatch.setattr(workspace, "resolve_config_path", lambda: tmp_path / "config.yml")
        monkeypatch.setattr(workspace, "load_osprey_config", lambda: config)
        monkeypatch.setattr(srv, "_bundle", None)
        monkeypatch.setattr(srv, "_bundle_error", None)

        srv.create_server()
        return srv

    def test_absent_bundle_path_is_recorded_as_not_configured(self, monkeypatch, tmp_path):
        srv = self._run_create_server(monkeypatch, tmp_path, {})

        assert srv._bundle is None
        assert srv._bundle_error is not None
        code, message, remedies = srv._bundle_error
        assert code == "bundle_not_configured"
        assert any("facility_knowledge.bundle_path" in r for r in remedies), remedies
        assert "start" not in message.lower()

    def test_unloadable_bundle_is_recorded_with_its_resolved_path(self, monkeypatch, tmp_path):
        missing = tmp_path / "not-a-bundle"
        srv = self._run_create_server(
            monkeypatch, tmp_path, {"facility_knowledge": {"bundle_path": str(missing)}}
        )

        assert srv._bundle is None
        assert srv._bundle_error is not None
        code, message, _remedies = srv._bundle_error
        assert code == "bundle_load_failed"
        assert str(missing) in message

    def test_a_loadable_bundle_records_no_cause(self, monkeypatch, tmp_path, fixture_bundle):
        srv = self._run_create_server(
            monkeypatch, tmp_path, {"facility_knowledge": {"bundle_path": str(fixture_bundle)}}
        )

        assert srv._bundle is not None
        assert srv._bundle_error is None
