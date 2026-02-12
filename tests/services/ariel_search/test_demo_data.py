"""Tests for the bundled demo logbook seed data.

Validates that the demo logbook JSON is well-formed and compatible
with GenericJSONAdapter for tutorial use.
"""

import json
from pathlib import Path

DEMO_DATA_PATH = Path(
    "src/osprey/templates/apps/control_assistant/data/logbook_seed/demo_logbook.json"
)


class TestDemoLogbookData:
    """Tests for the bundled demo logbook seed data."""

    def test_demo_file_exists(self):
        """Demo logbook JSON file exists in template."""
        assert DEMO_DATA_PATH.exists(), "Demo logbook file missing from template"

    def test_demo_file_is_valid_json(self):
        """Demo logbook file is valid JSON with sufficient entries."""
        data = json.loads(DEMO_DATA_PATH.read_text())
        assert "entries" in data
        assert len(data["entries"]) >= 15, "Need enough entries for meaningful search"

    def test_demo_entries_have_required_fields(self):
        """Each demo entry has fields required by GenericJSONAdapter."""
        data = json.loads(DEMO_DATA_PATH.read_text())
        for entry in data["entries"]:
            assert "id" in entry, f"Entry missing 'id': {entry}"
            assert "timestamp" in entry, f"Entry missing 'timestamp': {entry}"
            assert "text" in entry or "title" in entry, (
                f"Entry missing text content: {entry.get('id')}"
            )

    def test_demo_entries_parseable_by_adapter(self):
        """Demo entries can be parsed by GenericJSONAdapter._convert_entry."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.ingestion.adapters.generic import (
            GenericJSONAdapter,
        )

        data = json.loads(DEMO_DATA_PATH.read_text())

        # Create minimal config for adapter
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://mock/test"},
                "ingestion": {
                    "adapter": "generic_json",
                    "source_url": str(DEMO_DATA_PATH),
                },
            }
        )
        adapter = GenericJSONAdapter(config)

        for entry_data in data["entries"]:
            result = adapter._convert_entry(entry_data)
            assert result["entry_id"], f"Missing entry_id for {entry_data['id']}"
            assert result["raw_text"], f"Missing raw_text for {entry_data['id']}"
            assert result["timestamp"], f"Missing timestamp for {entry_data['id']}"
