"""Integration tests for ARIEL ingestion pipeline.

Tests the full pipeline from adapter to storage (TEST-M004 / INT-001).

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestIngestionPipeline:
    """Test full ingestion pipeline from adapter to storage."""

    @pytest.fixture
    def sample_entries_path(self) -> Path:
        """Path to sample ALS entries fixture file."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "fixtures"
            / "ariel"
            / "sample_als_entries.jsonl"
        )

    async def test_full_pipeline_adapter_to_storage(
        self, repository, sample_entries_path, integration_ariel_config
    ):
        """Test complete ingestion flow: load -> adapt -> store -> retrieve.

        Steps:
        1. Load sample entry from fixtures/ariel/sample_als_entries.jsonl
        2. Run through ALSAdapter to normalize
        3. Store via repository.upsert_entry()
        4. Verify entry retrievable with all metadata
        """
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        # Skip if fixture file doesn't exist
        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        # Create config with ingestion settings pointing to fixture
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        # Create adapter
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_system_name == "ALS eLog"

        # Fetch entries from adapter
        entries_fetched = []
        async for entry in adapter.fetch_entries(limit=3):
            entries_fetched.append(entry)
            # Store in repository
            await repository.upsert_entry(entry)

        assert len(entries_fetched) > 0, "No entries fetched from adapter"

        # Verify entries are retrievable
        for entry in entries_fetched:
            retrieved = await repository.get_entry(entry["entry_id"])
            assert retrieved is not None
            assert retrieved["source_system"] == "ALS eLog"
            assert retrieved["raw_text"]  # Has content
            assert retrieved["author"]  # Has author

    async def test_adapter_normalizes_als_entry_format(
        self, sample_entries_path, integration_ariel_config
    ):
        """ALSAdapter correctly normalizes ALS JSON to EnhancedLogbookEntry."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Get first entry
        entry = None
        async for e in adapter.fetch_entries(limit=1):
            entry = e
            break

        assert entry is not None

        # Verify normalized structure
        assert "entry_id" in entry
        assert "source_system" in entry
        assert "timestamp" in entry
        assert "author" in entry
        assert "raw_text" in entry
        assert "attachments" in entry
        assert "metadata" in entry

        # ALS-specific metadata should be extracted
        metadata = entry.get("metadata", {})
        # subject should be in metadata (ALS-specific)
        assert "subject" in metadata or entry["raw_text"]  # Either in metadata or merged

    async def test_adapter_transforms_attachments(
        self, sample_entries_path, integration_ariel_config
    ):
        """ALSAdapter transforms relative attachment URLs to full URLs."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Find an entry with attachments
        entry_with_attachments = None
        async for entry in adapter.fetch_entries(limit=20):
            if entry.get("attachments"):
                entry_with_attachments = entry
                break

        if entry_with_attachments is None:
            pytest.skip("No entries with attachments in fixture")

        # Verify attachment URLs are fully qualified
        for attachment in entry_with_attachments["attachments"]:
            assert "url" in attachment
            url = attachment["url"]
            assert url.startswith("https://") or url.startswith("http://")

    async def test_adapter_parses_als_categories(
        self, sample_entries_path, integration_ariel_config
    ):
        """ALSAdapter parses comma-separated categories into list."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Find an entry with categories
        entry_with_categories = None
        async for entry in adapter.fetch_entries(limit=20):
            categories = entry.get("metadata", {}).get("categories", [])
            if categories:
                entry_with_categories = entry
                break

        if entry_with_categories is None:
            pytest.skip("No entries with categories in fixture")

        categories = entry_with_categories["metadata"]["categories"]
        assert isinstance(categories, list)
        assert len(categories) > 0

    async def test_stored_entry_has_enhancement_status(
        self, repository, sample_entries_path, integration_ariel_config
    ):
        """Stored entries have enhancement_status field."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Fetch and store one entry
        entry = None
        async for e in adapter.fetch_entries(limit=1):
            entry = e
            break

        if entry is None:
            pytest.skip("No entries in fixture")

        await repository.upsert_entry(entry)

        # Retrieve and check enhancement_status
        retrieved = await repository.get_entry(entry["entry_id"])
        assert retrieved is not None
        assert "enhancement_status" in retrieved
        assert isinstance(retrieved["enhancement_status"], dict)


class TestIngestionWithTimeFilters:
    """Test ingestion adapter time filtering."""

    @pytest.fixture
    def sample_entries_path(self) -> Path:
        """Path to sample ALS entries fixture file."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "fixtures"
            / "ariel"
            / "sample_als_entries.jsonl"
        )

    async def test_adapter_respects_since_filter(
        self, sample_entries_path, integration_ariel_config
    ):
        """Adapter respects since (after) time filter."""

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Get all entries first
        all_entries = []
        async for entry in adapter.fetch_entries():
            all_entries.append(entry)

        if len(all_entries) < 2:
            pytest.skip("Need at least 2 entries for time filter test")

        # Sort by timestamp
        all_entries.sort(key=lambda e: e["timestamp"])

        # Use median timestamp as filter
        mid_idx = len(all_entries) // 2
        since = all_entries[mid_idx]["timestamp"]

        # Fetch with since filter
        filtered_entries = []
        async for entry in adapter.fetch_entries(since=since):
            filtered_entries.append(entry)

        # Should have fewer entries
        assert len(filtered_entries) < len(all_entries)

        # All filtered entries should be after since
        for entry in filtered_entries:
            assert entry["timestamp"] > since

    async def test_adapter_respects_limit(self, sample_entries_path, integration_ariel_config):
        """Adapter respects limit parameter."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": integration_ariel_config.database.uri},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": str(sample_entries_path),
                },
            }
        )

        adapter = ALSLogbookAdapter(config)

        # Fetch with limit
        limited_entries = []
        async for entry in adapter.fetch_entries(limit=3):
            limited_entries.append(entry)

        assert len(limited_entries) <= 3


class TestIngestionCleanup:
    """Clean up ingestion test data."""

    async def test_cleanup(self, migrated_pool):
        """Clean up test entries created during ingestion tests."""
        async with migrated_pool.connection() as conn:
            # Delete entries created by ALSAdapter (source_system = 'ALS eLog')
            # Only delete test entries (IDs from fixture are numeric strings)
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE source_system = 'ALS eLog'
                AND entry_id ~ '^[0-9]+$'
            """)
