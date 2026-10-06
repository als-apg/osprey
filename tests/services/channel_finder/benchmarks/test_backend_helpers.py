"""Unit tests for pure helpers in the benchmark backends.

Covers the provider-free text-extraction helper, the deterministic
early-return branches of the LiteLLM endpoint resolver, how the ReAct
backend arms the shared rate limiter at construction, and what the SDK
backend sends and scores around a stubbed query. The provider-driving
``run_query`` paths are the human-babysat benchmark surface and are not
unit-tested here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.services.channel_finder.benchmarks.backends import SdkBackend, create_backend
from osprey.services.channel_finder.benchmarks.backends.in_context_backend import _extract_text
from osprey.services.channel_finder.benchmarks.backends.react_backend import (
    ReactBackend,
    _resolve_litellm_endpoint,
)
from osprey.services.channel_finder.benchmarks.project_env import project_config
from osprey.services.channel_finder.benchmarks.sdk import SDKWorkflowResult, ToolTrace
from osprey.services.channel_finder.core.exceptions import ConfigurationError
from osprey.services.channel_finder.rate_limiter import configure_rate_limiter, get_rate_limiter


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
        assert _resolve_litellm_endpoint(tmp_path, {}, "ollama") is None

    def test_missing_config_returns_none(self, tmp_path: Path):
        # No config.yml in the project dir -> resolver bails out early.
        assert _resolve_litellm_endpoint(tmp_path, None, "als-apg") is None

    @staticmethod
    def _gateway_config(tmp_path: Path) -> dict | None:
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
        return project_config(tmp_path)

    def test_unset_endpoint_variable_is_refused_by_name(self, tmp_path: Path, monkeypatch):
        """An unexported ${VAR} is not a hostname to hand to litellm.

        The config resolver keeps the reference verbatim when the variable is
        unset, so without this the benchmark dials a host called
        ``${BENCH_GATEWAY_URL}`` and reports the failure as the model's.
        """
        monkeypatch.delenv("BENCH_GATEWAY_URL", raising=False)
        monkeypatch.setenv("ALS_APG_API_KEY", "test-key")

        with pytest.raises(ValueError, match="BENCH_GATEWAY_URL"):
            _resolve_litellm_endpoint(tmp_path, self._gateway_config(tmp_path), "als-apg")

    def test_endpoint_variable_is_expanded(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("BENCH_GATEWAY_URL", "https://gateway.example.com/v1")
        monkeypatch.setenv("ALS_APG_API_KEY", "test-key")

        resolved = _resolve_litellm_endpoint(tmp_path, self._gateway_config(tmp_path), "als-apg")

        assert resolved == {"api_base": "https://gateway.example.com", "api_key": "test-key"}

    def test_endpoint_variable_may_come_from_the_project_env_file(
        self, tmp_path: Path, monkeypatch
    ):
        """A benchmark run reads the deployment's ``.env``, as it does for the key."""
        monkeypatch.delenv("BENCH_GATEWAY_URL", raising=False)
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        config = self._gateway_config(tmp_path)
        (tmp_path / ".env").write_text(
            "BENCH_GATEWAY_URL=https://from-dotenv.example.com\nALS_APG_API_KEY=dotenv-key\n"
        )

        resolved = _resolve_litellm_endpoint(tmp_path, config, "als-apg")

        assert resolved == {"api_base": "https://from-dotenv.example.com", "api_key": "dotenv-key"}


#: The smallest valid provider entry; the key reference is exported per test.
_GATEWAY_ENTRY = {
    "api_key": "${GW_API_KEY}",
    "base_url": "https://gateway.example.test/v1",
    "default_model": "m-1",
    "models": ["m-1"],
}


class TestReactBackendPacing:
    """The backend paces its calls to the provider's catalog cap, and only that."""

    @pytest.fixture(autouse=True)
    def _clear_limiter(self, monkeypatch):
        monkeypatch.setenv("GW_API_KEY", "test-key")
        configure_rate_limiter(None)
        yield
        configure_rate_limiter(None)

    @staticmethod
    def _project(tmp_path: Path, providers: dict) -> Path:
        (tmp_path / "config.yml").write_text(yaml.safe_dump({"api": {"providers": providers}}))
        return tmp_path

    def test_arms_the_limiter_from_the_catalog_cap(self, tmp_path: Path):
        project = self._project(tmp_path, {"gw": {**_GATEWAY_ENTRY, "requests_per_minute": 7}})
        ReactBackend(project, "gw/m-1", 3)
        limiter = get_rate_limiter()
        assert limiter is not None
        assert limiter.max_calls == 7

    @pytest.mark.parametrize(
        ("provider", "model"), [("cborg", "cborg/m-1"), ("ollama", "ollama/gpt-oss:20b")]
    )
    def test_a_provider_without_a_cap_clears_an_armed_limiter(
        self, tmp_path: Path, provider: str, model: str
    ):
        project = self._project(tmp_path, {provider: dict(_GATEWAY_ENTRY)})
        configure_rate_limiter(3)
        ReactBackend(project, model, 3)
        assert get_rate_limiter() is None

    def test_a_project_without_a_config_is_not_paced(self, tmp_path: Path):
        ReactBackend(tmp_path, "gw/m-1", 3)
        assert get_rate_limiter() is None

    @pytest.mark.parametrize("model", ["gw/m-1", "ollama/gpt-oss:20b"])
    def test_a_malformed_config_is_refused(self, tmp_path: Path, model: str):
        (tmp_path / "config.yml").write_text("- not\n- a mapping\n")

        with pytest.raises(ConfigurationError, match="config.yml"):
            ReactBackend(tmp_path, model, 3)


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


class TestAutoBackendReadsTheMode:
    """``auto`` reads the project's pipeline mode, and a file it cannot read is no mode."""

    def test_auto_refuses_a_malformed_config(self, tmp_path: Path):
        (tmp_path / "config.yml").write_text("channel_finder: [unclosed\n")

        with pytest.raises(ConfigurationError, match="config.yml"):
            create_backend("auto", tmp_path, "anthropic/claude-haiku-4-5")

    def test_auto_without_a_config_takes_the_default_backend(self, tmp_path: Path):
        backend = create_backend("auto", tmp_path, "anthropic/claude-haiku-4-5")
        assert isinstance(backend, SdkBackend)


_RUN_SDK_QUERY = "osprey.services.channel_finder.benchmarks.backends.sdk_backend.run_sdk_query"


class TestSdkBackend:
    """What the SDK backend sends to the query and what it scores from the result."""

    async def test_scores_the_agent_text_not_the_tool_output(self, tmp_path: Path):
        result = SDKWorkflowResult(
            text_blocks=["Use SR:A"],
            tool_traces=[
                ToolTrace(name="mcp__channel-finder__query", input={}, result="SR:B SR:C"),
            ],
        )
        backend = SdkBackend(tmp_path, "als-apg/claude-haiku-4-5-20251001", 5, 0.2)
        with patch(_RUN_SDK_QUERY, new=AsyncMock(return_value=result)):
            output = await backend.run_query("q", "hierarchical")

        assert output.response_text == "use sr:a"
        assert output.tool_traces[0].result == "SR:B SR:C"

    async def test_sends_the_bare_wire_id_and_its_provider(self, tmp_path: Path):
        backend = SdkBackend(tmp_path, "als-apg/claude-haiku-4-5-20251001", 5, 0.2)
        query = AsyncMock(return_value=SDKWorkflowResult())
        with patch(_RUN_SDK_QUERY, new=query):
            await backend.run_query("q", "hierarchical")

        query.assert_awaited_once_with(
            tmp_path,
            "q",
            model="claude-haiku-4-5-20251001",
            provider="als-apg",
            max_turns=5,
            max_budget_usd=0.2,
        )
