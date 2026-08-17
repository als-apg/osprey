"""``qmd-resync`` against the mutation paths that bypass the enhancers.

The markdown mirror is normally written by the ``qmd_export`` enhancement as
entries flow through ingestion. Several code paths write straight to
``enhanced_entries`` and never reach an enhancer, so a mirror that only ever saw
enhanced rows would silently omit them — an operator's own web entry would be
invisible to the search the operator just used. ``run_qmd_resync`` is what closes
that hole, and each class below drives one real bypassing site and then proves
the resync re-exported it.

The sites driven here, each verified against the tree rather than trusted from
the plan — these are the entry-creation paths a *user* reaches, through the web
API, the service, and the agent's MCP tool:

* ``interfaces/ariel/api/routes.py`` ``_publish_or_local`` — the read-only-adapter
  fallback that saves a web entry locally.
* ``interfaces/ariel/api/routes.py`` ``_store_and_link_attachments`` — the
  re-upsert that links uploaded attachments onto an existing row.
* ``services/ariel_search/service.py`` ``create_entry`` — the optimistic local
  upsert after a facility write, and the re-ingestion upsert that follows it for
  a non-local adapter. This one no longer *needs* the resync: it mirrors its own
  write inline, best-effort, and its tests prove that instead — the resync stays
  its backstop for the failure paths.
* ``mcp_server/ariel/tools/entry.py`` ``entry_create`` in direct mode — the
  agent's own write, plus its attachment re-upsert.

**This list is not exhaustive, and the coverage does not depend on it being so.**
``run_qmd_resync`` is generic over ``enhanced_entries.updated_at`` — it never
enumerates call sites — so any bypassing write is picked up by construction.
``cli_operations.py`` ``apply_simulation`` (the simulation seeder) is one more
such site, deliberately left undriven because it writes through the same
``upsert_entry`` the classes below already exercise. What these tests protect is
that the resync *reaches* a row a user could have created without an enhancer
running; a new bypassing site needs a new test only when it writes by some route
other than ``upsert_entry``.

These call the real functions rather than driving HTTP, because the HTTP layer is
already covered in ``tests/interfaces/ariel/test_routes.py`` and what is under
test here is the database mutation, not the routing.

No sidecar runs in this module: everything asserted is about Postgres and the
mirror on disk. The container lane lives in
``test_qmd_ariel_search_sidecar.py``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.integration._qmd_ariel_support import open_migrated_repository, qmd_ariel_config_dict

pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.xdist_group("docker"),
]


@pytest.fixture(scope="session")
def resync_database_url(qmd_ariel_database_factory) -> str:
    """A database of this module's own, because one test purges everything."""
    return qmd_ariel_database_factory("qmd_ariel_resync")


@pytest.fixture(scope="session")
def _resync_migrated(resync_database_url: str) -> None:
    """Apply migrations once for the module, off the tests' event loop."""
    from osprey.services.ariel_search.config import ARIELConfig

    async def run() -> None:
        config = ARIELConfig.from_dict(qmd_ariel_config_dict(resync_database_url, "/tmp/unused"))
        pool, _ = await open_migrated_repository(config)
        await pool.close()

    asyncio.run(run())


@pytest.fixture
async def lane(tmp_path: Path, resync_database_url: str, _resync_migrated):
    """A clean database and an empty mirror for one test.

    ``enhanced_entries`` is truncated up front rather than at teardown so that a
    failing test leaves its rows behind to inspect.
    """
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import ARIELRepository, create_connection_pool

    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()
    config_dict = qmd_ariel_config_dict(resync_database_url, mirror_root)
    config = ARIELConfig.from_dict(config_dict)

    pool = await create_connection_pool(config.database)
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE enhanced_entries CASCADE")

    class _Lane:
        def __init__(self) -> None:
            self.config_dict = config_dict
            self.config = config
            self.mirror_root = mirror_root
            self.repository = ARIELRepository(pool, config)

        def mirror_file(self, entry: dict[str, Any]) -> Path:
            from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

            return mirror_path(self.mirror_root, entry)

        def mirrored_ids(self) -> set[str]:
            from osprey.services.ariel_search.enhancement.qmd_export.writer import (
                entry_id_from_path,
            )

            return {entry_id_from_path(path) for path in self.mirror_root.rglob("*.md")}

        async def resync(self, *, rebuild: bool = False):
            from osprey.services.ariel_search.cli_operations import run_qmd_resync

            result = await run_qmd_resync(self.config_dict, rebuild=rebuild)
            assert result is not None, "qmd_export must be enabled for this lane"
            return result

    try:
        yield _Lane()
    finally:
        await pool.close()


def _row(entry_id: str, *, raw_text: str, author: str = "Ada Lovelace") -> dict[str, Any]:
    """A minimal ``enhanced_entries`` row."""
    return {
        "entry_id": entry_id,
        "source_system": "ALS OLOG",
        "timestamp": datetime(2005, 6, 1, 12, 0, tzinfo=UTC),
        "author": author,
        "raw_text": raw_text,
        "attachments": [],
        "metadata": {},
        "enhancement_status": {},
    }


# ---------------------------------------------------------------------------
# The baseline the mutation tests build on
# ---------------------------------------------------------------------------


class TestResyncBaseline:
    """The incremental pass itself, before any bypassing site is involved."""

    async def test_resync_exports_a_row_written_behind_the_enhancers(self, lane):
        """A direct upsert is invisible to the mirror until a resync runs."""
        row = _row("baseline-0001", raw_text="Baseline entry for the mirror.")
        await lane.repository.upsert_entry(row)

        assert lane.mirrored_ids() == set(), "the enhancer never ran, so nothing is mirrored yet"

        result = await lane.resync()

        assert result.written == 1
        assert lane.mirrored_ids() == {"baseline-0001"}
        assert "Baseline entry" in lane.mirror_file(row).read_text(encoding="utf-8")

    async def test_resync_touches_the_freshness_marker_when_it_writes(self, lane):
        """The sidecar's primary re-index trigger has to advance, not merely exist.

        ``cli_operations`` touches the marker under ``if written:``. Asserting only
        that the file is present would hold just as well if the touch had moved
        outside that guard — and that guard is what stops a timer-driven resync
        making the sidecar rescan the whole corpus on every tick.
        """
        from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME

        marker = lane.mirror_root / TOUCH_MARKER_NAME
        await lane.repository.upsert_entry(_row("marker-0001", raw_text="Marker entry."))
        await lane.resync()

        assert marker.exists()
        first = marker.stat().st_mtime_ns

        await lane.repository.upsert_entry(_row("marker-0002", raw_text="Second marker entry."))
        result = await lane.resync()

        assert result.written == 1
        assert marker.stat().st_mtime_ns > first, "the marker did not advance on a write"


# ---------------------------------------------------------------------------
# One class per enhancer-bypassing mutation path
# ---------------------------------------------------------------------------


class TestWebLocalEntryPath:
    """``routes._publish_or_local`` — the read-only-adapter fallback."""

    async def test_local_only_web_entry_is_re_exported(self, lane):
        """A web entry saved locally reaches the mirror on the next resync.

        Drives the real fallback branch by giving the service an adapter that
        raises ``NotImplementedError`` from ``create_entry``, which is exactly
        what a read-only facility adapter does.
        """
        from unittest.mock import AsyncMock, MagicMock

        from osprey.interfaces.ariel.api.routes import _publish_or_local
        from osprey.services.ariel_search.models import FacilityEntryCreateRequest

        service = MagicMock()
        service.repository = lane.repository
        service.create_entry = AsyncMock(side_effect=NotImplementedError("adapter is read-only"))

        entry_id, source_system, sync_status, _message = await _publish_or_local(
            service,
            FacilityEntryCreateRequest(subject="Vacuum trip", details="Sector 7 gauge spiked."),
            fallback_metadata={
                "author": "Grace Hopper",
                "raw_text": "Sector 7 gauge spiked.",
                "metadata": {},
            },
        )

        assert source_system == "ARIEL Web"
        assert sync_status == "local_only"
        assert lane.mirrored_ids() == set(), "the fallback upsert must bypass the enhancers"

        result = await lane.resync()

        assert result.written == 1
        assert entry_id in lane.mirrored_ids()
        row = await lane.repository.get_entry(entry_id)
        assert "Sector 7 gauge spiked" in lane.mirror_file(row).read_text(encoding="utf-8")


class TestAttachmentReUpsertPath:
    """``routes._store_and_link_attachments`` — the attachment re-upsert."""

    async def test_attachment_link_exports_an_entry_the_mirror_never_saw(self, lane):
        """The re-upsert is what puts an unmirrored entry in front of the resync.

        This is the operationally interesting order: an entry is created and its
        attachments uploaded before any resync has run, so the row the pass finds
        is the re-upserted one.
        """
        from osprey.interfaces.ariel.api.routes import _store_and_link_attachments

        row = _row("attach-0001", raw_text="Scope trace attached below.")
        await lane.repository.upsert_entry(row)

        linked = await _store_and_link_attachments(
            _ServiceStub(lane.repository),
            "attach-0001",
            [("trace.png", "image/png", b"\x89PNG\r\n\x1a\n fake")],
        )

        assert linked == 1
        stored = await lane.repository.get_entry("attach-0001")
        assert stored["attachments"], "the re-upsert did not land"
        assert lane.mirrored_ids() == set(), "the re-upsert must bypass the enhancers"

        result = await lane.resync()

        assert result.written == 1
        assert lane.mirrored_ids() == {"attach-0001"}
        assert "Scope trace attached" in lane.mirror_file(row).read_text(encoding="utf-8")

    async def test_attachment_link_alone_does_not_rewrite_an_existing_file(self, lane):
        """Linking an attachment to an already-mirrored entry is free.

        ``render_entry`` does not read ``attachments`` — the mirror carries prose
        for qmd to index, and an attachment is a blob in ARIEL's own store that
        is never pushed upstream. So the re-upsert moves ``updated_at``, the
        resync re-scans the row, the bytes come out identical and the file is
        left alone. Asserting the *scan* separately from the *write* is what
        distinguishes "correctly skipped" from "never looked at".

        The freshness marker must not move either — that is the complement of
        :meth:`TestResyncBaseline.test_resync_touches_the_freshness_marker_when_it_writes`,
        and together they pin the ``if written:`` guard from both sides. A marker
        that advanced on a no-op pass would make a timer-driven resync rescan the
        whole corpus on every tick.
        """
        from osprey.interfaces.ariel.api.routes import _store_and_link_attachments
        from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME

        row = _row("attach-0002", raw_text="Already mirrored before the upload.")
        await lane.repository.upsert_entry(row)
        await lane.resync()
        path = lane.mirror_file(row)
        before = path.stat().st_mtime_ns
        marker_before = (lane.mirror_root / TOUCH_MARKER_NAME).stat().st_mtime_ns

        await _store_and_link_attachments(
            _ServiceStub(lane.repository),
            "attach-0002",
            [("trace.png", "image/png", b"\x89PNG\r\n\x1a\n fake")],
        )
        result = await lane.resync()

        assert result.scanned == 1, "the mutated row was never re-scanned"
        assert result.written == 0
        assert result.unchanged == 1
        assert path.stat().st_mtime_ns == before
        assert (lane.mirror_root / TOUCH_MARKER_NAME).stat().st_mtime_ns == marker_before, (
            "a no-op pass advanced the freshness marker"
        )


class TestMcpEntryCreatePath:
    """``mcp_server/ariel/tools/entry.py`` ``entry_create`` in direct mode.

    The agent's own write. Distinct from :class:`TestEntryCreateUpsertPaths`,
    which drives ``service.create_entry``: this one never touches a facility
    adapter, stamps ``source_system: 'ARIEL MCP'``, and upserts straight into
    ``enhanced_entries``. Same generic resync catches it, but it is a real
    user-facing path and deserves its own driver.

    ``draft=False`` is the branch under test. The default ``draft=True`` writes a
    JSON file for a human to submit through the panel and never touches the
    database, so it is not a bypassing mutation at all.
    """

    @staticmethod
    def _install(lane, monkeypatch) -> None:
        """Point the tool's context at the lane's repository.

        ``get_ariel_context`` and ``notify_agent_activity_async`` are both bound
        at module import, so the names are patched on the tool module rather
        than at their definition sites.
        """
        import osprey.mcp_server.ariel.tools.entry as entry_tool

        class _Registry:
            async def service(self):
                return _ServiceStub(lane.repository)

        monkeypatch.setattr(entry_tool, "get_ariel_context", lambda: _Registry())
        # The panel highlight would POST at a web terminal that is not running.
        # tests/mcp_server/ has a conftest stub for this; tests/integration/ does
        # not inherit it, so the seam is stubbed here.
        monkeypatch.setattr(entry_tool, "notify_agent_activity_async", _anoop)

    async def test_direct_mode_entry_is_re_exported(self, lane, monkeypatch):
        """An agent-written entry reaches the mirror on the next resync."""
        import osprey.mcp_server.ariel.tools.entry as entry_tool

        self._install(lane, monkeypatch)

        payload = json.loads(
            await _tool_fn(entry_tool.entry_create)(
                subject="Septum conditioning",
                details="Ran the conditioning sequence to 8 kV.",
                author="Agent",
                draft=False,
            )
        )
        entry_id = payload["entry_id"]

        assert payload["source_system"] == "ARIEL MCP"
        assert lane.mirrored_ids() == set(), "the MCP write must bypass the enhancers"

        result = await lane.resync()

        assert result.written == 1
        assert lane.mirrored_ids() == {entry_id}
        row = await lane.repository.get_entry(entry_id)
        assert "conditioning sequence" in lane.mirror_file(row).read_text(encoding="utf-8")

    async def test_draft_mode_writes_no_row_for_the_resync_to_find(self, lane, monkeypatch):
        """The control: the default branch is not a bypassing mutation.

        Without this, ``draft=False`` above could be passing for the wrong reason
        — the test would look identical if both branches wrote to the database.
        """
        import osprey.mcp_server.ariel.tools.entry as entry_tool

        self._install(lane, monkeypatch)
        monkeypatch.chdir(lane.mirror_root.parent)

        payload = json.loads(
            await _tool_fn(entry_tool.entry_create)(
                subject="Draft only",
                details="Should never reach Postgres.",
                draft=True,
            )
        )

        # Assert the draft branch actually succeeded before concluding anything
        # from an empty table: a tool that errored would leave Postgres empty too.
        assert "draft_id" in payload, payload
        assert "entry_id" not in payload, payload
        assert await lane.repository.count_entries() == 0
        result = await lane.resync()
        assert result.written == 0
        assert lane.mirrored_ids() == set()


class TestEntryCreateUpsertPaths:
    """``service.create_entry`` — the optimistic upsert and the re-ingest upsert."""

    async def test_optimistic_local_upsert_is_mirrored_inline(self, lane, monkeypatch):
        """``create_entry`` mirrors its optimistic upsert at creation time.

        The enhancement pipeline never sees this write, so the inline export is
        what makes an operator- or agent-created entry hybrid-searchable within
        one sidecar poll. A resync straight afterwards finds nothing left to do.
        """
        service = _service_with_adapter(
            lane, _WriteAdapter("Generic JSON", refetch=None), monkeypatch
        )

        result = await service.create_entry(
            _facility_request("Beam dump", "Interlock chain opened at 03:12.")
        )

        assert lane.mirrored_ids() == {result.entry_id}
        resync = await lane.resync()

        assert resync.written == 0

    async def test_reingestion_upsert_is_what_gets_mirrored(self, lane, monkeypatch):
        """A non-local adapter re-ingests, overwriting the optimistic row.

        The inline mirror write runs after the re-ingestion, so the mirror must
        carry the *re-ingested* text, not the optimistic one — publishing the
        optimistic text would put words in the mirror that the facility logbook
        has already superseded.
        """
        refetched = "Canonical text as the facility logbook stored it."
        service = _service_with_adapter(
            lane, _WriteAdapter("ALS OLOG", refetch=refetched), monkeypatch
        )

        result = await service.create_entry(
            _facility_request("Beam dump", "Interlock chain opened at 03:12.")
        )

        stored = await lane.repository.get_entry(result.entry_id)
        assert stored["raw_text"] == refetched, "the re-ingestion upsert did not run"

        row = await lane.repository.get_entry(result.entry_id)
        assert refetched in lane.mirror_file(row).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The mtime pin
# ---------------------------------------------------------------------------


class TestUnchangedEntriesAreNotRewritten:
    """What makes a resync cheap, and what makes it safe to run on a timer."""

    async def test_unchanged_entry_keeps_its_mtime_across_a_resync(self, lane):
        """An unchanged row's file is not touched at all.

        This is the property the whole design rests on. If a resync rewrote every
        file, the sidecar would see the entire corpus as changed and re-embed it,
        and any watcher would read a routine resync as a full rewrite.
        """
        row = _row("stable-0001", raw_text="Nothing about this entry will change.")
        await lane.repository.upsert_entry(row)
        await lane.resync(rebuild=True)

        path = lane.mirror_file(row)
        before = path.stat().st_mtime_ns

        # Bump updated_at without changing anything the renderer reads, which is
        # the shape of bookkeeping churn: a re-ingestion poll re-upserting a row
        # whose content is identical.
        await lane.repository.upsert_entry(row)
        result = await lane.resync()

        assert result.unchanged >= 1, "the row was not even re-scanned"
        assert result.written == 0, "an unchanged row was rewritten"
        assert path.stat().st_mtime_ns == before, "the mirror file's mtime moved"

    async def test_a_changed_entry_does_get_rewritten(self, lane):
        """The pin must not be a writer that never writes.

        Without this, ``test_unchanged_entry_keeps_its_mtime_across_a_resync``
        would pass just as well against an export that had silently stopped
        working.
        """
        row = _row("changing-0001", raw_text="First version of the text.")
        await lane.repository.upsert_entry(row)
        await lane.resync(rebuild=True)

        path = lane.mirror_file(row)
        before = path.stat().st_mtime_ns

        await lane.repository.upsert_entry(
            _row("changing-0001", raw_text="Second version, materially different.")
        )
        result = await lane.resync()

        assert result.written == 1
        assert path.stat().st_mtime_ns != before
        assert "Second version" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# purge -> re-ingest -> rebuild
# ---------------------------------------------------------------------------


class TestPurgeReingestRebuildCycle:
    """``ariel purge`` empties Postgres; only a rebuild can empty the mirror."""

    async def test_rebuild_after_purge_and_reingest_leaves_a_consistent_index(self, lane):
        """The mirror ends up holding exactly the rows Postgres holds.

        A purge does not touch the filesystem, so after it the mirror still
        describes entries that no longer exist. An incremental resync cannot fix
        that — it only adds and updates. ``--rebuild`` is the only pass that
        removes, which is why it is the documented recovery.
        """
        from osprey.services.ariel_search.cli_operations import execute_purge

        old = [
            _row(f"pre-purge-{i:04d}", raw_text=f"Entry {i} before the purge.") for i in range(3)
        ]
        for row in old:
            await lane.repository.upsert_entry(row)
        await lane.resync()
        assert lane.mirrored_ids() == {row["entry_id"] for row in old}

        await execute_purge(lane.config_dict, embeddings_only=False)

        assert await lane.repository.count_entries() == 0
        assert lane.mirrored_ids() == {row["entry_id"] for row in old}, (
            "a purge must not silently delete mirror files; only a rebuild removes"
        )

        # Re-ingest a different set, as a fresh ingestion run would.
        new = [
            _row(f"post-purge-{i:04d}", raw_text=f"Entry {i} after the purge.") for i in range(2)
        ]
        for row in new:
            await lane.repository.upsert_entry(row)

        result = await lane.resync(rebuild=True)

        assert result.rebuild is True
        assert result.removed >= len(old)
        assert result.written == len(new)
        assert lane.mirrored_ids() == {row["entry_id"] for row in new}

    async def test_incremental_resync_after_a_purge_would_not_have_recovered(self, lane):
        """Names the failure the rebuild exists to prevent.

        The watermark survives a purge, so an incremental pass after one finds
        nothing to do and leaves every stale file in place. Asserting that here
        is what stops someone 'simplifying' the recovery to a plain resync.
        """
        from osprey.services.ariel_search.cli_operations import execute_purge

        row = _row("stale-0001", raw_text="This row is about to be purged.")
        await lane.repository.upsert_entry(row)
        await lane.resync()
        assert lane.mirrored_ids() == {"stale-0001"}

        await execute_purge(lane.config_dict, embeddings_only=False)
        result = await lane.resync()

        assert result.removed == 0
        assert lane.mirrored_ids() == {"stale-0001"}, "an incremental pass removed something"

    async def test_rebuild_after_a_purge_with_no_reingest_empties_the_mirror(self, lane):
        """The rebuild also has to survive finding nothing at all."""
        from osprey.services.ariel_search.cli_operations import execute_purge

        await lane.repository.upsert_entry(_row("doomed-0001", raw_text="Doomed entry."))
        await lane.resync()

        await execute_purge(lane.config_dict, embeddings_only=False)
        result = await lane.resync(rebuild=True)

        assert result.written == 0
        assert result.removed >= 1
        assert lane.mirrored_ids() == set()


# ---------------------------------------------------------------------------
# Doubles for the mutation-path drivers
# ---------------------------------------------------------------------------


class _ServiceStub:
    """The attribute ``_store_and_link_attachments`` and the MCP tool reach for."""

    def __init__(self, repository) -> None:
        self.repository = repository


async def _anoop(*args: Any, **kwargs: Any) -> None:
    """Stand in for an async fire-and-forget notifier."""
    return None


def _tool_fn(tool):
    """Unwrap a FastMCP ``FunctionTool`` to the async function it decorates."""
    return getattr(tool, "fn", tool)


def _facility_request(subject: str, details: str):
    """Build the request object ``service.create_entry`` takes."""
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

    return FacilityEntryCreateRequest(subject=subject, details=details, author="Grace Hopper")


class _WriteAdapter:
    """A facility adapter that accepts a write, and optionally re-serves it.

    Args:
        source_system: Adapter's reported system name. ``"Generic JSON"`` is the
            value ``create_entry`` treats as local-only, which is what selects
            between the two upsert sites.
        refetch: ``raw_text`` the re-ingestion poll finds, or ``None`` for an
            adapter whose ``fetch_entries`` yields nothing.
    """

    supports_write = True

    def __init__(self, source_system: str, *, refetch: str | None) -> None:
        self.source_system_name = source_system
        self._refetch = refetch
        self.created_id = "facility-9001"

    async def create_entry(self, request) -> str:
        return self.created_id

    async def fetch_entries(self, since=None):
        if self._refetch is None:
            return
        yield {
            "entry_id": self.created_id,
            "source_system": self.source_system_name,
            "timestamp": datetime.now(UTC) - timedelta(seconds=1),
            "author": "Facility Logbook",
            "raw_text": self._refetch,
            "attachments": [],
            "metadata": {},
            "enhancement_status": {},
        }


class _Service:
    """The real ``create_entry``, bound to the lane's repository and config.

    ``ARIELSearchService.create_entry`` reads exactly two attributes off ``self``
    — ``config`` and ``repository`` — so borrowing the method onto this object
    runs the real code path without standing up a whole service. Anything the
    method starts reaching for beyond those two will fail loudly here rather than
    be silently absorbed by a mock.
    """

    from osprey.services.ariel_search.service import ARIELSearchService as _Real

    create_entry = _Real.create_entry
    del _Real

    def __init__(self, repository, config) -> None:
        self.repository = repository
        self.config = config


def _service_with_adapter(lane, adapter, monkeypatch: pytest.MonkeyPatch) -> _Service:
    """Bind the real ``create_entry`` to *adapter*.

    ``create_entry`` resolves its adapter through ``ingestion.get_adapter`` at
    call time, so that name is what gets patched — patching a copy imported into
    the service module would not be seen.
    """
    import osprey.services.ariel_search.ingestion as ingestion

    monkeypatch.setattr(ingestion, "get_adapter", lambda config: adapter)
    return _Service(lane.repository, lane.config)
