"""Tests for the facility_description MCP tool.

Verifies that the tool reads .claude/rules/facility.md from the project root
and returns its content, or a proper error envelope if the file is missing.
"""

import json
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict

import pytest
import yaml


@pytest.fixture
def facility_config(tmp_path):
    """Create minimal config.yml and .claude/rules/facility.md for testing."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.dump({"control_system": {"type": "mock", "writes_enabled": False}}))

    rules_dir = tmp_path / ".claude" / "rules"
    rules_dir.mkdir(parents=True)
    facility_file = rules_dir / "facility.md"
    facility_file.write_text(
        "# Facility Description\n\n"
        "## Facility Identity\n\n"
        "- **Name:** Advanced Light Source\n"
        "- **Type:** Synchrotron light source\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def missing_facility_config(tmp_path):
    """Create config.yml but no facility.md."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.dump({"control_system": {"type": "mock", "writes_enabled": False}}))
    return tmp_path


class TestFacilityDescription:
    """Tests for the facility_description tool."""

    @pytest.mark.asyncio
    async def test_success(self, facility_config, monkeypatch):
        """File exists -> returns content."""
        monkeypatch.chdir(facility_config)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.mcp_server.workspace.tools.facility_description import facility_description
        from tests.mcp_server.conftest import get_tool_fn

        fn = get_tool_fn(facility_description)
        result = extract_response_dict(await fn())

        assert "error" not in result
        assert "facility_description" in result
        assert "Advanced Light Source" in result["facility_description"]
        assert "source" in result

    @pytest.mark.asyncio
    async def test_not_found(self, missing_facility_config, monkeypatch):
        """File missing -> error envelope."""
        monkeypatch.chdir(missing_facility_config)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.mcp_server.workspace.tools.facility_description import facility_description
        from tests.mcp_server.conftest import get_tool_fn

        fn = get_tool_fn(facility_description)
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await fn()
        result = _exc_ctx["envelope"]
        assert "facility.md" in result["error_message"]
        assert len(result["suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_utf8_content(self, tmp_path, monkeypatch):
        """Handles UTF-8 special characters correctly."""
        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"control_system": {"type": "mock"}}))

        rules_dir = tmp_path / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        facility_file = rules_dir / "facility.md"
        facility_file.write_text(
            "# Facility\n\n"
            "- Beam energy: 6 GeV\n"
            "- Emittance: \u03b5 = 3.5 nm\u00b7rad\n"
            "- Circumference: 2\u03c0R \u2248 780 m\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.mcp_server.workspace.tools.facility_description import facility_description
        from tests.mcp_server.conftest import get_tool_fn

        fn = get_tool_fn(facility_description)
        result = json.loads(await fn())

        assert "error" not in result
        assert "\u03b5" in result["facility_description"]
        assert "\u03c0" in result["facility_description"]
