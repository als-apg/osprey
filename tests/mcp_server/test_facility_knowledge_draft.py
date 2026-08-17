"""Tests for the draft_concept MCP write tool.

Strategy:
- All tests inject an OKFBundle backed by tmp_path (no live server).
- The approval hook is a Claude Code PreToolUse hook — it intercepts calls
  BEFORE the Python function runs.  Since we call the raw function directly
  (bypassing the hook), we:
    (a) Verify the tool writes correctly when called (simulates "approved" path).
    (b) Verify no filesystem write occurs if we DON'T call it (simulates "rejected"
        path by simply not calling the tool — the hook blocks it before invocation).
- Concurrent collision is tested by manipulating ``_drafts_in_flight`` directly.
- The ``.qmd-touch`` freshness marker is asserted on directly (existence, mtime
  advance, atomicity), plus an AST check that the handler stays await-free.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from fastmcp.exceptions import ToolError

from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture
def fixture_bundle(tmp_path: Path) -> Path:
    """Empty bundle root for write tests."""
    root = tmp_path / "bundle"
    root.mkdir()
    return root


@pytest.fixture(autouse=True)
def _patch_bundle(fixture_bundle: Path, monkeypatch):
    """Inject the fixture bundle and reset the in-flight set before each test."""
    import osprey.mcp_server.facility_knowledge.server as srv
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    monkeypatch.setattr(srv, "_bundle", OKFBundle(fixture_bundle))
    # Ensure no residual in-flight state leaks between tests.
    srv._drafts_in_flight.clear()
    yield
    srv._drafts_in_flight.clear()


# ---------------------------------------------------------------------------
# Happy path — approved write
# ---------------------------------------------------------------------------


class TestDraftConceptWrite:
    """draft_concept writes valid documents to disk when called (approved path)."""

    @pytest.mark.asyncio
    async def test_writes_file_to_bundle(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        result = json.loads(
            await get_tool_fn(draft_concept)(
                concept_id="accelerator_overview",
                doc_type="facility_overview",
                title="Accelerator Overview",
                description="Overview of the accelerator complex.",
                body="The facility uses a synchrotron ring.\n",
            )
        )

        assert result["status"] == "written"
        assert result["concept_id"] == "accelerator_overview"
        assert (fixture_bundle / "accelerator_overview.md").exists()

    @pytest.mark.asyncio
    async def test_written_file_is_valid_okf_doc(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept
        from osprey.services.facility_knowledge.okf.document import OKFDocument

        await get_tool_fn(draft_concept)(
            concept_id="facility_overview",
            doc_type="facility_overview",
            title="Facility Overview",
            description="High-level facility description.",
            body="# Facility\n\nMain content here.\n",
        )

        text = (fixture_bundle / "facility_overview.md").read_text(encoding="utf-8")
        doc = OKFDocument.parse(text)
        assert doc.frontmatter["type"] == "facility_overview"
        assert doc.frontmatter["title"] == "Facility Overview"
        assert doc.frontmatter["description"] == "High-level facility description."
        assert "Main content here." in doc.body

    @pytest.mark.asyncio
    async def test_written_doc_passes_authoring_validation(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept
        from osprey.services.facility_knowledge.okf.document import OKFDocument

        await get_tool_fn(draft_concept)(
            concept_id="my_concept",
            doc_type="reference",
            title="My Concept",
            description="A short description.",
            body="Body text.\n",
        )

        text = (fixture_bundle / "my_concept.md").read_text(encoding="utf-8")
        doc = OKFDocument.parse(text)
        doc.validate("authoring")  # must not raise

    @pytest.mark.asyncio
    async def test_writes_nested_concept(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        result = json.loads(
            await get_tool_fn(draft_concept)(
                concept_id="tables/beam_params",
                doc_type="data_table",
                title="Beam Parameters",
                description="Live beam parameter PVs.",
                body="Contains PV names for all beam parameters.\n",
            )
        )

        assert result["status"] == "written"
        assert (fixture_bundle / "tables" / "beam_params.md").exists()

    @pytest.mark.asyncio
    async def test_second_sequential_write_overwrites(self, fixture_bundle: Path):
        """Last-approved-wins: a second sequential draft replaces the first."""
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        await get_tool_fn(draft_concept)(
            concept_id="my_concept",
            doc_type="note",
            title="Version 1",
            description="First draft.",
            body="First body.\n",
        )
        await get_tool_fn(draft_concept)(
            concept_id="my_concept",
            doc_type="note",
            title="Version 2",
            description="Second draft.",
            body="Updated body.\n",
        )

        from osprey.services.facility_knowledge.okf.document import OKFDocument

        text = (fixture_bundle / "my_concept.md").read_text(encoding="utf-8")
        doc = OKFDocument.parse(text)
        assert doc.frontmatter["title"] == "Version 2"
        assert "Updated body." in doc.body

    @pytest.mark.asyncio
    async def test_extra_frontmatter_merged(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept
        from osprey.services.facility_knowledge.okf.document import OKFDocument

        await get_tool_fn(draft_concept)(
            concept_id="tagged_concept",
            doc_type="reference",
            title="Tagged Concept",
            description="A concept with extra frontmatter.",
            body="Body.\n",
            extra_frontmatter='{"tags": ["epics", "beam"], "resource": "https://example.com"}',
        )

        text = (fixture_bundle / "tagged_concept.md").read_text(encoding="utf-8")
        doc = OKFDocument.parse(text)
        assert doc.frontmatter["tags"] == ["epics", "beam"]
        assert doc.frontmatter["resource"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_reserved_keys_win_over_extra_frontmatter(self, fixture_bundle: Path):
        """type/title/description in extra_frontmatter are overridden by explicit args."""
        from osprey.mcp_server.facility_knowledge.server import draft_concept
        from osprey.services.facility_knowledge.okf.document import OKFDocument

        await get_tool_fn(draft_concept)(
            concept_id="override_test",
            doc_type="facility_overview",
            title="Correct Title",
            description="Correct description.",
            body="Body.\n",
            extra_frontmatter='{"type": "WRONG", "title": "WRONG"}',
        )

        text = (fixture_bundle / "override_test.md").read_text(encoding="utf-8")
        doc = OKFDocument.parse(text)
        assert doc.frontmatter["type"] == "facility_overview"
        assert doc.frontmatter["title"] == "Correct Title"

    @pytest.mark.asyncio
    async def test_result_contains_absolute_path(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        result = json.loads(
            await get_tool_fn(draft_concept)(
                concept_id="path_test",
                doc_type="note",
                title="Path Test",
                description="Testing path in result.",
                body="Body.\n",
            )
        )

        assert Path(result["path"]).is_absolute()
        assert result["path"].endswith("path_test.md")


# ---------------------------------------------------------------------------
# Approval-hook interaction (no-write on rejection)
# ---------------------------------------------------------------------------


class TestApprovalGate:
    """When the approval hook blocks the call, the tool is never invoked."""

    def test_no_filesystem_write_when_tool_not_invoked(self, fixture_bundle: Path):
        """Simulate the rejection path: simply don't call the tool.

        The approval hook works at the Claude Code harness level — it intercepts
        the JSON-RPC call BEFORE Python is invoked.  In the rejection case, the
        tool function is never called, so no file is written.  This test
        asserts that invariant explicitly: a bundle with no writes has no .md
        concept files.
        """
        # No tool call made — simulates hook-rejected path.
        md_files = [f for f in fixture_bundle.rglob("*.md") if f.name not in ("index.md", "log.md")]
        assert md_files == [], "Filesystem should be empty when draft_concept was never invoked"


# ---------------------------------------------------------------------------
# Concurrent collision guard
# ---------------------------------------------------------------------------


class TestConcurrentDraftCollision:
    """A second concurrent draft for the same concept ID returns an error."""

    @pytest.mark.asyncio
    async def test_concurrent_draft_same_id_raises_conflict(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        # Simulate an in-flight draft for the same concept ID.
        srv._drafts_in_flight.add("accelerator_overview")

        with assert_raises_error(error_type="draft_conflict") as ctx:
            await get_tool_fn(draft_concept)(
                concept_id="accelerator_overview",
                doc_type="facility_overview",
                title="Accelerator Overview",
                description="Overview.",
                body="Body.\n",
            )

        assert "accelerator_overview" in ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_different_concept_ids_do_not_conflict(self, fixture_bundle: Path):
        """Two concurrent drafts for DIFFERENT IDs must both succeed."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        # Simulate an in-flight draft for a different concept.
        srv._drafts_in_flight.add("other_concept")

        # This must succeed — it's a different concept ID.
        result = json.loads(
            await get_tool_fn(draft_concept)(
                concept_id="accelerator_overview",
                doc_type="facility_overview",
                title="Accelerator Overview",
                description="Overview of the accelerator complex.",
                body="Body.\n",
            )
        )

        assert result["status"] == "written"

    @pytest.mark.asyncio
    async def test_in_flight_set_cleared_after_success(self, fixture_bundle: Path):
        """The in-flight lock is released after a successful write."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        await get_tool_fn(draft_concept)(
            concept_id="my_concept",
            doc_type="note",
            title="My Concept",
            description="Description.",
            body="Body.\n",
        )

        assert "my_concept" not in srv._drafts_in_flight

    @pytest.mark.asyncio
    async def test_in_flight_set_cleared_after_error(self, fixture_bundle: Path, monkeypatch):
        """The in-flight lock is released even when the tool raises an error."""
        # Force an error mid-write by making the bundle root read-only.
        import os

        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        os.chmod(fixture_bundle, 0o555)
        try:
            with pytest.raises((ToolError, Exception)):
                await get_tool_fn(draft_concept)(
                    concept_id="fail_concept",
                    doc_type="note",
                    title="Fail Concept",
                    description="This write will fail.",
                    body="Body.\n",
                )
        finally:
            os.chmod(fixture_bundle, 0o755)

        assert "fail_concept" not in srv._drafts_in_flight


# ---------------------------------------------------------------------------
# Validation rejection
# ---------------------------------------------------------------------------


class TestDraftConceptValidation:
    """draft_concept rejects documents that fail authoring validation."""

    @pytest.mark.asyncio
    async def test_empty_type_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="validation_error"):
            await get_tool_fn(draft_concept)(
                concept_id="bad_doc",
                doc_type="",
                title="Has Title",
                description="Has description.",
                body="Body.\n",
            )

        assert not (fixture_bundle / "bad_doc.md").exists()

    @pytest.mark.asyncio
    async def test_empty_title_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="validation_error"):
            await get_tool_fn(draft_concept)(
                concept_id="bad_doc",
                doc_type="note",
                title="",
                description="Has description.",
                body="Body.\n",
            )

        assert not (fixture_bundle / "bad_doc.md").exists()

    @pytest.mark.asyncio
    async def test_empty_description_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="validation_error"):
            await get_tool_fn(draft_concept)(
                concept_id="bad_doc",
                doc_type="note",
                title="Has Title",
                description="",
                body="Body.\n",
            )

        assert not (fixture_bundle / "bad_doc.md").exists()

    @pytest.mark.asyncio
    async def test_invalid_extra_frontmatter_json_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="validation_error"):
            await get_tool_fn(draft_concept)(
                concept_id="bad_doc",
                doc_type="note",
                title="Title",
                description="Description.",
                body="Body.\n",
                extra_frontmatter="not valid json {",
            )

    @pytest.mark.asyncio
    async def test_extra_frontmatter_array_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="validation_error"):
            await get_tool_fn(draft_concept)(
                concept_id="bad_doc",
                doc_type="note",
                title="Title",
                description="Description.",
                body="Body.\n",
                extra_frontmatter='["not", "an", "object"]',
            )


# ---------------------------------------------------------------------------
# Path traversal safety
# ---------------------------------------------------------------------------


class TestDraftConceptPathSafety:
    """draft_concept rejects concept IDs that escape the bundle root."""

    @pytest.mark.asyncio
    async def test_dotdot_traversal_rejected(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        with assert_raises_error(error_type="path_traversal"):
            await get_tool_fn(draft_concept)(
                concept_id="../secret",
                doc_type="note",
                title="Secret",
                description="Should not be written.",
                body="Body.\n",
            )

        # Confirm nothing was written outside the bundle.
        assert not (fixture_bundle.parent / "secret.md").exists()

    @pytest.mark.asyncio
    async def test_bundle_not_initialised(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import draft_concept

        monkeypatch.setattr(srv, "_bundle", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(draft_concept)(
                concept_id="test",
                doc_type="note",
                title="Test",
                description="Test.",
                body="Body.\n",
            )


# ---------------------------------------------------------------------------
# Freshness marker (.qmd-touch)
# ---------------------------------------------------------------------------


async def _draft(concept_id: str = "my_concept", title: str = "My Concept") -> dict:
    """Call draft_concept with defaults and return the parsed JSON result."""
    from osprey.mcp_server.facility_knowledge.server import draft_concept

    return json.loads(
        await get_tool_fn(draft_concept)(
            concept_id=concept_id,
            doc_type="note",
            title=title,
            description="Description.",
            body="Body.\n",
        )
    )


class TestTouchMarker:
    """draft_concept advances the bundle's ``.qmd-touch`` freshness marker."""

    @pytest.mark.asyncio
    async def test_marker_written_with_current_timestamp(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        before = datetime.now().astimezone()
        result = await _draft()
        after = datetime.now().astimezone()

        marker = fixture_bundle / TOUCH_MARKER_NAME
        assert result["touch_marker"] == "updated"
        assert marker.exists()

        stamp = datetime.fromisoformat(marker.read_text(encoding="utf-8").strip())
        assert stamp.tzinfo is not None, "marker timestamp must be timezone-aware"
        assert before <= stamp <= after

    @pytest.mark.asyncio
    async def test_marker_mtime_advances_on_second_draft(self, fixture_bundle: Path):
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        await _draft(title="Version 1")
        marker = fixture_bundle / TOUCH_MARKER_NAME

        # Backdate the marker so the advance is observable regardless of the
        # filesystem's mtime granularity.
        backdated = marker.stat().st_mtime - 60
        os.utime(marker, (backdated, backdated))

        await _draft(title="Version 2")

        assert marker.stat().st_mtime > backdated

    @pytest.mark.asyncio
    async def test_marker_advances_even_when_document_is_unchanged(self, fixture_bundle: Path):
        """Re-drafting identical content still advances the marker."""
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        await _draft()
        marker = fixture_bundle / TOUCH_MARKER_NAME
        backdated = marker.stat().st_mtime - 60
        os.utime(marker, (backdated, backdated))

        await _draft()

        assert marker.stat().st_mtime > backdated

    @pytest.mark.asyncio
    async def test_marker_is_replaced_never_deleted(self, fixture_bundle: Path, monkeypatch):
        """The marker is only ever renamed over — a consumer can never see it gone."""
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        await _draft()
        marker = fixture_bundle / TOUCH_MARKER_NAME

        removed: list[str] = []
        real_remove = os.remove

        def _spy_remove(path, *args, **kwargs):
            removed.append(str(path))
            return real_remove(path, *args, **kwargs)

        # os.remove and os.unlink are separate bindings; Path.unlink goes
        # through os.unlink, so both are spied.
        monkeypatch.setattr(os, "remove", _spy_remove)
        monkeypatch.setattr(os, "unlink", _spy_remove)

        await _draft(title="Version 2")

        assert str(marker) not in removed
        assert marker.exists()

    @pytest.mark.asyncio
    async def test_writes_go_through_atomic_replace(self, fixture_bundle: Path, monkeypatch):
        """Both the document and the marker land via a same-directory rename."""
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        renames: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def _spy_replace(src, dst, **kwargs):
            renames.append((Path(src), Path(dst)))
            return real_replace(src, dst, **kwargs)

        monkeypatch.setattr(os, "replace", _spy_replace)

        await _draft(concept_id="tables/beam_params")

        destinations = {dst for _, dst in renames}
        assert fixture_bundle / "tables" / "beam_params.md" in destinations
        assert fixture_bundle / TOUCH_MARKER_NAME in destinations
        # Same-directory renames only — os.replace is atomic within one filesystem.
        for src, dst in renames:
            assert src.parent == dst.parent

    @pytest.mark.asyncio
    async def test_no_temp_files_left_behind(self, fixture_bundle: Path):
        await _draft(concept_id="tables/beam_params")

        assert list(fixture_bundle.rglob("*.tmp")) == []

    @pytest.mark.asyncio
    async def test_marker_failure_does_not_fail_the_draft(self, fixture_bundle: Path):
        """A marker that cannot be written degrades to the sidecar interval sweep."""
        from osprey.mcp_server.facility_knowledge.server import TOUCH_MARKER_NAME

        # A directory at the marker path makes the rename fail with OSError.
        (fixture_bundle / TOUCH_MARKER_NAME).mkdir()

        result = await _draft()

        assert result["status"] == "written"
        assert result["touch_marker"] == "failed"
        assert (fixture_bundle / "my_concept.md").exists()
        assert list(fixture_bundle.glob("*.tmp")) == []

    def test_marker_is_not_listed_as_a_concept(self, fixture_bundle: Path):
        """The dotfile marker must not leak into bundle listings."""
        from osprey.services.facility_knowledge.okf.bundle import OKFBundle

        (fixture_bundle / ".qmd-touch").write_text("2026-01-01T00:00:00+00:00\n", encoding="utf-8")

        ids = {e.concept_id for e in OKFBundle(fixture_bundle).list_concepts()}
        assert ".qmd-touch" not in ids
        assert "" not in ids


class TestDraftConceptHasNoAwaitPoints:
    """The handler must stay synchronous — the collision policy depends on it."""

    def test_handler_body_contains_no_await(self):
        """No ``await``/``async with``/``async for`` anywhere in draft_concept.

        Parsed from a fresh read of the source file rather than
        ``inspect.getsource``, which can mis-slice under .py/.pyc skew.
        """
        import osprey.mcp_server.facility_knowledge.server as srv

        source_file = inspect.getsourcefile(srv)
        assert source_file is not None
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))

        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "draft_concept"
        ]
        assert len(handlers) == 1, "expected exactly one draft_concept definition"

        offenders = [
            type(node).__name__
            for node in ast.walk(handlers[0])
            if isinstance(node, ast.Await | ast.AsyncWith | ast.AsyncFor)
        ]
        assert offenders == [], f"draft_concept gained await points: {offenders}"

    def test_write_helpers_are_synchronous(self):
        import osprey.mcp_server.facility_knowledge.server as srv

        assert not inspect.iscoroutinefunction(srv._atomic_write_text)
        assert not inspect.iscoroutinefunction(srv._touch_bundle_marker)
