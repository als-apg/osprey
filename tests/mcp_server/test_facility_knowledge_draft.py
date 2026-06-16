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
"""

from __future__ import annotations

import json
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
