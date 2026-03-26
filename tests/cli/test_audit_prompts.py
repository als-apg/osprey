"""Tests for audit prompt construction and safety content."""

from pathlib import Path

from osprey.cli.audit_prompts import (
    AUDIT_SYSTEM_INSTRUCTIONS,
    build_audit_prompt,
)


class TestAuditSystemInstructions:
    """Verify the system instructions encode OSPREY's safety model."""

    def test_mentions_channel_write(self):
        assert "channel_write" in AUDIT_SYSTEM_INSTRUCTIONS

    def test_mentions_framework_servers(self):
        assert "FRAMEWORK_SERVERS" in AUDIT_SYSTEM_INSTRUCTIONS

    def test_mentions_approval(self):
        assert "approval" in AUDIT_SYSTEM_INSTRUCTIONS.lower()

    def test_mentions_hook_chain(self):
        assert "writes_check" in AUDIT_SYSTEM_INSTRUCTIONS
        assert "limits" in AUDIT_SYSTEM_INSTRUCTIONS


class TestBuildAuditPrompt:
    """Test prompt assembly."""

    def test_contains_system_instructions(self):
        prompt = build_audit_prompt("project", Path("/tmp/test"), "file1.py\nfile2.py")
        assert "channel_write" in prompt

    def test_contains_file_listing(self):
        prompt = build_audit_prompt("project", Path("/tmp/test"), "foo.py\nbar.yml")
        assert "foo.py" in prompt
        assert "bar.yml" in prompt

    def test_contains_json_schema(self):
        prompt = build_audit_prompt("project", Path("/tmp/test"), "a.py")
        # The schema's title should appear in the prompt
        assert "AuditReport" in prompt

    def test_target_type_in_prompt(self):
        p1 = build_audit_prompt("profile", Path("/tmp/test"), "a.py")
        p2 = build_audit_prompt("project", Path("/tmp/test"), "a.py")
        assert "profile" in p1
        assert "project" in p2

    def test_schema_matches_model(self):
        """The JSON schema in instructions matches the actual Pydantic model."""
        prompt = build_audit_prompt("project", Path("/tmp/test"), "a.py")
        # Verify key fields from the schema are present
        assert "overall_risk" in prompt
        assert "findings" in prompt
        assert "severity" in prompt
