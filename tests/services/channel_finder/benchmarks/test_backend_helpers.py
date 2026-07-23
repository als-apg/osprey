"""Unit tests for pure helpers in the benchmark backends.

Covers the provider-free text-extraction helper and the deterministic
early-return branches of the LiteLLM endpoint resolver. The provider-driving
``run_query`` paths are the human-babysat benchmark surface and are not
unit-tested here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from osprey.services.channel_finder.benchmarks.backends.in_context_backend import _extract_text
from osprey.services.channel_finder.benchmarks.backends.react_backend import (
    _resolve_litellm_endpoint,
)


class TestExtractText:
    def test_joins_text_blocks(self):
        tool_result = SimpleNamespace(
            content=[
                SimpleNamespace(text="line one"),
                SimpleNamespace(text="line two"),
            ]
        )
        assert _extract_text(tool_result) == "line one\nline two"

    def test_skips_blocks_without_text(self):
        tool_result = SimpleNamespace(
            content=[
                SimpleNamespace(text="kept"),
                SimpleNamespace(other="ignored"),
            ]
        )
        assert _extract_text(tool_result) == "kept"

    def test_empty_content_falls_back_to_str(self):
        tool_result = SimpleNamespace(content=[])
        assert _extract_text(tool_result) == str(tool_result)

    def test_missing_content_attribute_falls_back_to_str(self):
        assert _extract_text("raw string result") == "raw string result"


class TestResolveLitellmEndpoint:
    def test_ollama_returns_none(self, tmp_path: Path):
        assert _resolve_litellm_endpoint(tmp_path, "ollama") is None

    def test_missing_config_file_returns_none(self, tmp_path: Path):
        # No config.yml in the project dir -> resolver bails out early.
        assert _resolve_litellm_endpoint(tmp_path, "als-apg") is None
