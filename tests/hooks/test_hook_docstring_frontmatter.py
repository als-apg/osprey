"""Tests for hook docstring YAML front matter.

Validates that each OSPREY hook template has a properly structured docstring
with YAML front matter (name, description, event) and an ASCII flow diagram.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

HOOK_FILES = [
    "osprey_writes_check.py",
    "osprey_limits.py",
    "osprey_approval.py",
    "osprey_error_guidance.py",
    "osprey_notebook_update.py",
    "osprey_memory_guard.py",
]

# Required fields for all hooks
REQUIRED_FIELDS = {"name", "description", "event"}


def _load_docstring(hook_filename: str) -> str:
    """Import a hook module and return its __doc__ string."""
    path = HOOKS_DIR / hook_filename
    assert path.exists(), f"Hook file not found: {path}"

    importlib.util.spec_from_file_location(hook_filename.removesuffix(".py"), path)
    # Don't actually execute the module (it calls sys.exit) — just compile it
    # and extract the docstring from the AST
    import ast

    tree = ast.parse(path.read_text())
    docstring = ast.get_docstring(tree)
    assert docstring is not None, f"{hook_filename} has no module docstring"
    return docstring


def _parse_front_matter(docstring: str) -> tuple[dict, str]:
    """Extract YAML front matter and body from a docstring.

    Returns (fields_dict, body_text).
    """
    match = re.match(r"^---\n(.*?)\n---\n?(.*)", docstring, re.DOTALL)
    assert match, f"Docstring does not contain --- delimited front matter:\n{docstring[:200]}"

    yaml_block = match.group(1)
    body = match.group(2)

    fields = yaml.safe_load(yaml_block)
    assert isinstance(fields, dict), f"Front matter YAML is not a dict: {yaml_block}"

    return fields, body


@pytest.mark.parametrize("hook_file", HOOK_FILES, ids=lambda f: f.removesuffix(".py"))
class TestHookFrontMatter:
    """Validate front matter structure for all hooks."""

    def test_has_yaml_front_matter(self, hook_file):
        """Hook docstring contains parseable YAML front matter."""
        docstring = _load_docstring(hook_file)
        fields, _ = _parse_front_matter(docstring)
        assert fields, "Front matter should not be empty"

    def test_required_fields_present(self, hook_file):
        """Front matter contains name, description, and event."""
        docstring = _load_docstring(hook_file)
        fields, _ = _parse_front_matter(docstring)

        missing = REQUIRED_FIELDS - set(fields.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_event_is_valid(self, hook_file):
        """Event field is one of the known Claude Code hook events."""
        docstring = _load_docstring(hook_file)
        fields, _ = _parse_front_matter(docstring)

        valid_events = {"PreToolUse", "PostToolUse"}
        assert fields["event"] in valid_events, (
            f"Invalid event '{fields['event']}', expected one of {valid_events}"
        )

    def test_has_ascii_diagram(self, hook_file):
        """Body contains an ASCII flow diagram in a code block."""
        docstring = _load_docstring(hook_file)
        _, body = _parse_front_matter(docstring)

        # Look for a fenced code block containing arrow characters
        assert "```" in body, "Body should contain a fenced code block with a flow diagram"
        # Check for arrow-like characters that indicate a flow diagram
        assert re.search(r"[─►▼▲◄┼┌┐└┘│├┤]", body), (
            "Body should contain ASCII box-drawing characters for the flow diagram"
        )

    def test_yaml_is_valid(self, hook_file):
        """Front matter YAML is valid and round-trips cleanly."""
        docstring = _load_docstring(hook_file)
        fields, _ = _parse_front_matter(docstring)

        # Verify all values are simple types (str, int, list, None)
        for key, value in fields.items():
            assert isinstance(value, (str, int, float, list, type(None))), (
                f"Field '{key}' has unexpected type {type(value).__name__}"
            )


# Individual hook-specific tests


class TestWritesCheckHook:
    def test_safety_layer_is_1(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_writes_check.py"))
        assert fields.get("safety_layer") == 1

    def test_tools_include_channel_write(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_writes_check.py"))
        assert "channel_write" in fields.get("tools", "")


class TestLimitsHook:
    def test_safety_layer_is_3(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_limits.py"))
        assert fields.get("safety_layer") == 3

    def test_tools_is_channel_write(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_limits.py"))
        assert "channel_write" in fields.get("tools", "")


class TestApprovalHook:
    def test_safety_layer_is_2(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_approval.py"))
        assert fields.get("safety_layer") == 2


class TestErrorGuidanceHook:
    def test_no_safety_layer(self):
        """Error guidance is not a safety gate — no safety_layer field."""
        fields, _ = _parse_front_matter(_load_docstring("osprey_error_guidance.py"))
        assert fields.get("safety_layer") is None

    def test_event_is_post_tool_use(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_error_guidance.py"))
        assert fields["event"] == "PostToolUse"


class TestNotebookUpdateHook:
    def test_no_safety_layer(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_notebook_update.py"))
        assert fields.get("safety_layer") is None

    def test_event_is_post_tool_use(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_notebook_update.py"))
        assert fields["event"] == "PostToolUse"


class TestMemoryGuardHook:
    def test_safety_layer_is_0(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_memory_guard.py"))
        assert fields.get("safety_layer") == 0

    def test_event_is_pre_tool_use(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_memory_guard.py"))
        assert fields["event"] == "PreToolUse"

    def test_tools_include_write(self):
        fields, _ = _parse_front_matter(_load_docstring("osprey_memory_guard.py"))
        assert "Write" in fields.get("tools", "")
