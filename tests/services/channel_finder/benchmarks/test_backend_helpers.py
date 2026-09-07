"""Unit tests for pure helpers in the benchmark backends.

Covers the provider-free text-extraction helper and the deterministic
early-return branches of the LiteLLM endpoint resolver. The provider-driving
``run_query`` paths are the human-babysat benchmark surface and are not
unit-tested here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.services.channel_finder.benchmarks.backends import SdkBackend, create_backend
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

    @staticmethod
    def _gateway_project(tmp_path: Path) -> Path:
        """A project whose provider names its endpoint through a variable.

        The shape the shipped provider catalog writes: the gateway host is the
        deployment's own, so ``base_url`` is a reference, not a literal.
        """
        (tmp_path / "config.yml").write_text(
            "api:\n"
            "  providers:\n"
            "    als-apg:\n"
            "      base_url: ${BENCH_GATEWAY_URL}\n"
            "      api_key: ${ALS_APG_API_KEY}\n"
        )
        return tmp_path

    def test_unset_endpoint_variable_is_refused_by_name(self, tmp_path: Path, monkeypatch):
        """An unexported ${VAR} is not a hostname to hand to litellm.

        The config resolver keeps the reference verbatim when the variable is
        unset, so without this the benchmark dials a host called
        ``${BENCH_GATEWAY_URL}`` and reports the failure as the model's.
        """
        monkeypatch.delenv("BENCH_GATEWAY_URL", raising=False)
        monkeypatch.setenv("ALS_APG_API_KEY", "test-key")

        with pytest.raises(ValueError, match="BENCH_GATEWAY_URL"):
            _resolve_litellm_endpoint(self._gateway_project(tmp_path), "als-apg")

    def test_endpoint_variable_is_expanded(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("BENCH_GATEWAY_URL", "https://gateway.example.com/v1")
        monkeypatch.setenv("ALS_APG_API_KEY", "test-key")

        resolved = _resolve_litellm_endpoint(self._gateway_project(tmp_path), "als-apg")

        assert resolved == {"api_base": "https://gateway.example.com", "api_key": "test-key"}

    def test_endpoint_variable_may_come_from_the_project_env_file(
        self, tmp_path: Path, monkeypatch
    ):
        """A benchmark run reads the deployment's ``.env``, as it does for the key."""
        monkeypatch.delenv("BENCH_GATEWAY_URL", raising=False)
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        project = self._gateway_project(tmp_path)
        (project / ".env").write_text(
            "BENCH_GATEWAY_URL=https://from-dotenv.example.com\nALS_APG_API_KEY=dotenv-key\n"
        )

        resolved = _resolve_litellm_endpoint(project, "als-apg")

        assert resolved == {"api_base": "https://from-dotenv.example.com", "api_key": "dotenv-key"}


def _graph_project(tmp_path: Path) -> Path:
    """A project directory configured for the graph paradigm."""
    (tmp_path / "config.yml").write_text("channel_finder:\n  pipeline_mode: graph\n")
    return tmp_path


class TestGraphIsSdkOnly:
    """The graph paradigm's one backend exemption, pinned.

    Its surface is agentic Cypher, which the SDK tool-use loop drives and the
    manual ReAct loop does not. Refusing at construction keeps the unsupported
    path from quietly producing numbers.
    """

    def test_react_refuses_a_graph_project(self, tmp_path: Path):
        with pytest.raises(ValueError, match="graph"):
            create_backend("react", _graph_project(tmp_path), "anthropic/claude-haiku-4-5")

    def test_auto_sends_a_graph_project_to_the_sdk_backend(self, tmp_path: Path):
        backend = create_backend("auto", _graph_project(tmp_path), "anthropic/claude-haiku-4-5")
        assert isinstance(backend, SdkBackend)
