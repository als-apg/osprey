"""E2E Tests for LLM Provider Integration.

Tests the LiteLLM-based provider system with real API calls across a
provider × model matrix. Tests are conditionally run based on available API keys.

Test Matrix:
- Providers: anthropic, openai, google, ollama, cborg, amsc
- Models: Multiple tiers per provider (haiku/sonnet, mini/4o, mistral/gptoss)
- Tasks: completion, structured output, ReAct agent with tools

Run with: pytest tests/e2e/test_llm_providers.py -v
"""

import os
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# MODEL MATRIX CONFIGURATION
# =============================================================================

MODEL_MATRIX: dict[str, list[tuple[str, str]]] = {
    "anthropic": [
        ("claude-haiku-4-5-20251001", "haiku"),
        ("claude-sonnet-4-5-20250929", "sonnet"),
    ],
    "openai": [
        ("gpt-4o-mini", "mini"),
        # gpt-4o removed: 80% flaky on react_agent (adds extra fields to structured output)
    ],
    "google": [
        ("gemini-2.0-flash", "flash"),
    ],
    "cborg": [
        ("anthropic/claude-haiku", "haiku"),
        ("anthropic/claude-sonnet", "sonnet"),
    ],
    "amsc": [
        ("claude-haiku", "haiku"),
        ("claude-sonnet", "sonnet"),
    ],
    "ollama": [
        ("ministral-3:8b", "ministral"),
        ("mistral:7b", "mistral7b"),
        ("gpt-oss:20b", "gptoss20b"),  # Uses direct Ollama API to bypass LiteLLM bug #15463
    ],
    "vllm": [
        # Models depend on what's served by the vLLM instance
        # These are detected dynamically at runtime
    ],
}

# Providers that support structured output
# Ollama uses direct API call to bypass LiteLLM bug #15463
STRUCTURED_OUTPUT_PROVIDERS = ["anthropic", "openai", "google", "cborg", "amsc", "vllm", "ollama"]

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    model_config = ConfigDict(extra="forbid")  # Required for Anthropic structured output

    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation")


class AgentResponse(BaseModel):
    """ReAct agent response with tool planning and final answer."""

    model_config = ConfigDict(extra="forbid")  # Required for Anthropic structured output

    needs_tool: bool = Field(description="Whether a tool call is needed")
    tool_name: str = Field(description="Name of tool to call (calculator or weather)")
    tool_input: str = Field(description="Input for the tool")
    final_answer: str = Field(description="Final answer after tool execution")
    reasoning: str = Field(description="Step-by-step reasoning")


# =============================================================================
# PROVIDER DETECTION
# =============================================================================


def get_available_providers_raw() -> dict[str, dict[str, Any]]:
    """Detect which providers have valid API keys configured."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # python-dotenv is optional; if unavailable, rely on existing environment variables
        pass

    available = {}

    providers_to_check = [
        (
            "anthropic",
            ["ANTHROPIC_API_KEY_o", "ANTHROPIC_API_KEY"],
            None,
            "claude-haiku-4-5-20251001",
        ),
        ("openai", ["OPENAI_API_KEY"], None, "gpt-4o-mini"),
        ("google", ["GOOGLE_API_KEY"], None, "gemini-2.0-flash"),
        ("cborg", ["CBORG_API_KEY"], "https://api.cborg.lbl.gov", "anthropic/claude-haiku"),
        (
            "amsc",
            ["AMSC_I2_API_KEY"],
            "https://api.i2-core.american-science-cloud.org",
            "claude-haiku",
        ),
    ]

    for provider_name, env_vars, default_base_url, default_model in providers_to_check:
        api_key = None
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key and not ("${" in api_key or "YOUR_API_KEY" in api_key.upper()):
                break
            api_key = None

        if api_key:
            available[provider_name] = {
                "api_key": api_key,
                "base_url": default_base_url,
                "default_model": default_model,
            }

    # Check Ollama connectivity
    ollama_base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import httpx

        response = httpx.get(f"{ollama_base_url}/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            preferred = ["ministral-3:8b", "mistral:7b", "llama3.2:1b", "llama3.1:latest"]
            default_model = next(
                (m for m in preferred if m in models), models[0] if models else None
            )
            if default_model:
                available["ollama"] = {
                    "api_key": None,
                    "base_url": ollama_base_url,
                    "default_model": default_model,
                }
    except Exception:
        # Ollama is optional; ignore connectivity/timeout errors during provider detection
        pass

    # Check vLLM connectivity
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    try:
        import httpx

        response = httpx.get(f"{vllm_base_url}/models", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            if models:
                available["vllm"] = {
                    "api_key": "EMPTY",  # vLLM doesn't require API key
                    "base_url": vllm_base_url,
                    "default_model": models[0],
                }
                # Update MODEL_MATRIX with discovered models
                MODEL_MATRIX["vllm"] = [
                    (m, m.split("/")[-1].replace("-", "_")[:10]) for m in models
                ]
    except Exception:
        # vLLM is optional; ignore connectivity/timeout errors during provider detection
        pass

    return available


# Cache at import time
_AVAILABLE_PROVIDERS = get_available_providers_raw()

# Cache Ollama models
_AVAILABLE_OLLAMA_MODELS: list[str] = []
try:
    import httpx

    resp = httpx.get(
        f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags", timeout=2.0
    )
    if resp.status_code == 200:
        _AVAILABLE_OLLAMA_MODELS = [m["name"] for m in resp.json().get("models", [])]
except Exception:
    # Ollama is optional; ignore connectivity errors during model discovery
    pass

# Cache vLLM models
_AVAILABLE_VLLM_MODELS: list[str] = []
try:
    import httpx

    resp = httpx.get(
        f"{os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')}/models", timeout=2.0
    )
    if resp.status_code == 200:
        _AVAILABLE_VLLM_MODELS = [m["id"] for m in resp.json().get("data", [])]
except Exception:
    # vLLM is optional; ignore connectivity errors during model discovery
    pass


def skip_if_provider_unavailable(provider_name: str):
    """Skip test if provider is not available."""
    if provider_name not in _AVAILABLE_PROVIDERS:
        pytest.skip(f"Provider '{provider_name}' not available")
    return _AVAILABLE_PROVIDERS[provider_name]


def skip_if_model_unavailable(provider_name: str, model_id: str):
    """Skip test if specific model is not available."""
    config = skip_if_provider_unavailable(provider_name)
    if provider_name == "ollama" and model_id not in _AVAILABLE_OLLAMA_MODELS:
        pytest.skip(f"Ollama model '{model_id}' not installed")
    if provider_name == "vllm" and model_id not in _AVAILABLE_VLLM_MODELS:
        pytest.skip(f"vLLM model '{model_id}' not served")
    return config


def handle_quota_errors(func):
    """Decorator to convert API quota/rate limit errors to pytest warnings.

    This prevents transient rate limiting from failing the test suite while
    still reporting the issue for visibility.
    """
    import functools
    import warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Check for common quota/rate limit error indicators
            quota_indicators = [
                "quota",
                "rate limit",
                "rate_limit",
                "ratelimit",
                "too many requests",
                "429",
                "resource_exhausted",
                "resourceexhausted",
            ]
            if any(indicator in error_str for indicator in quota_indicators):
                warnings.warn(
                    f"API quota/rate limit hit (test skipped): {e}",
                    UserWarning,
                    stacklevel=2,
                )
                pytest.skip(f"API quota/rate limit: {type(e).__name__}")
            raise

    return wrapper


def get_matrix_params():
    """Generate pytest parameters for full matrix."""
    params = []
    for provider, models in MODEL_MATRIX.items():
        for model_id, tier in models:
            params.append(pytest.param(provider, model_id, id=f"{provider}-{tier}"))
    return params


def get_structured_output_params():
    """Generate pytest parameters for structured output capable providers."""
    params = []
    for provider, models in MODEL_MATRIX.items():
        if provider not in STRUCTURED_OUTPUT_PROVIDERS:
            continue
        for model_id, tier in models:
            params.append(pytest.param(provider, model_id, id=f"{provider}-{tier}"))
    return params


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def available_providers():
    """Fixture providing available provider configurations."""
    return _AVAILABLE_PROVIDERS


@pytest.fixture(autouse=True)
def setup_llm_test_environment(test_config, tmp_path):
    """Set up test environment with provider configurations."""
    config = yaml.safe_load(test_config.read_text())
    config["provider_configs"] = {}

    # Ensure models section references all available providers so config-driven
    # provider filtering (#138) doesn't skip them during registry initialization
    if "models" not in config:
        config["models"] = {}
    for provider_name, provider_config in _AVAILABLE_PROVIDERS.items():
        config["provider_configs"][provider_name] = {
            "api_key": provider_config["api_key"],
            "base_url": provider_config["base_url"],
            "default_model_id": provider_config["default_model"],
        }
        # Add a models entry so the provider isn't filtered out
        if not any(
            m.get("provider") == provider_name
            for m in config["models"].values()
            if isinstance(m, dict)
        ):
            config["models"][f"test_{provider_name}"] = {
                "provider": provider_name,
                "model_id": provider_config["default_model"],
            }

    with open(test_config, "w") as f:
        yaml.dump(config, f)

    original_config_file = os.environ.get("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(test_config)

    # Remove proxy env vars that interfere with direct API calls
    proxy_env_vars = [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ]
    saved_proxy_vars = {}
    for var in proxy_env_vars:
        if var in os.environ:
            saved_proxy_vars[var] = os.environ.pop(var)

    from osprey.registry import initialize_registry, reset_registry
    from osprey.utils import config as config_module

    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()
    initialize_registry(config_path=str(test_config))

    yield

    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    if original_config_file is not None:
        os.environ["CONFIG_FILE"] = original_config_file
    elif "CONFIG_FILE" in os.environ:
        del os.environ["CONFIG_FILE"]

    for var, value in saved_proxy_vars.items():
        os.environ[var] = value


# =============================================================================
# MOCK TOOLS FOR REACT AGENT
# =============================================================================


def calculator_tool(expression: str) -> dict:
    """Calculator tool for testing."""
    try:
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters"}
        return {"result": float(eval(expression)), "expression": expression}
    except Exception as e:
        return {"error": str(e)}


def weather_tool(city: str) -> dict:
    """Mock weather tool for testing."""
    data = {
        "tokyo": {"temperature": 28.0, "conditions": "Humid"},
        "london": {"temperature": 15.0, "conditions": "Cloudy"},
    }
    return {"city": city, **data.get(city.lower(), {"temperature": 20.0, "conditions": "Unknown"})}


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.e2e
class TestProviderAvailability:
    """Meta-tests for provider detection."""

    def test_at_least_one_provider_available(self, available_providers, setup_llm_test_environment):
        """Ensure at least one provider is available."""
        assert len(available_providers) > 0, "No LLM providers available for testing"

    def test_provider_detection(self, available_providers, setup_llm_test_environment):
        """Document which providers are available."""
        print(f"\nAvailable: {list(available_providers.keys())}")


@pytest.mark.e2e
class TestExtendedThinking:
    """Test Anthropic extended thinking (provider-specific feature)."""

    @handle_quota_errors
    def test_anthropic_extended_thinking(self, setup_llm_test_environment):
        """Test extended thinking returns thinking blocks."""
        config = skip_if_provider_unavailable("anthropic")

        from osprey.models import get_chat_completion

        result = get_chat_completion(
            message="Think step by step: What is 17 * 23?",
            provider="anthropic",
            model_id="claude-sonnet-4-5-20250929",
            provider_config=config,
            enable_thinking=True,
            budget_tokens=1024,
            max_tokens=4000,
            temperature=1.0,
        )

        assert result is not None
        if isinstance(result, list):
            block_types = [getattr(b, "type", None) for b in result]
            assert "thinking" in block_types or "text" in block_types
        else:
            assert "391" in str(result)


@pytest.mark.e2e
class TestLLMMatrix:
    """Matrix tests across provider × model combinations."""

    @pytest.mark.parametrize("provider_name,model_id", get_matrix_params())
    @handle_quota_errors
    def test_completion(self, provider_name: str, model_id: str, setup_llm_test_environment):
        """Test basic completion works for each model."""
        config = skip_if_model_unavailable(provider_name, model_id)

        from osprey.models import get_chat_completion

        response = get_chat_completion(
            message="What is 2 + 2? Reply with just the number.",
            provider=provider_name,
            model_id=model_id,
            provider_config=config,
            max_tokens=10,
            temperature=0.0,
        )

        assert response is not None
        assert isinstance(response, str)
        assert "4" in response

    @pytest.mark.parametrize("provider_name,model_id", get_structured_output_params())
    @handle_quota_errors
    def test_structured_output(self, provider_name: str, model_id: str, setup_llm_test_environment):
        """Test structured output with Pydantic model."""
        config = skip_if_model_unavailable(provider_name, model_id)

        from osprey.models import get_chat_completion

        result = get_chat_completion(
            message="Analyze sentiment: 'This is wonderful!'",
            provider=provider_name,
            model_id=model_id,
            provider_config=config,
            output_model=SentimentResult,
            max_tokens=200,
            temperature=0.0,
        )

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize(
        "provider_name,model_id",
        [
            pytest.param("anthropic", "claude-sonnet-4-5-20250929", id="anthropic-sonnet"),
            # gpt-4o removed: 80% flaky on react_agent (adds extra fields to structured output)
            pytest.param("cborg", "anthropic/claude-sonnet", id="cborg-sonnet"),
        ],
    )
    @handle_quota_errors
    def test_react_agent(self, provider_name: str, model_id: str, setup_llm_test_environment):
        """Test ReAct agent: planning, tool use, and final answer in one flow."""
        config = skip_if_model_unavailable(provider_name, model_id)

        from osprey.models import get_chat_completion

        # Single comprehensive prompt that tests planning and tool integration
        prompt = """You are an assistant with calculator and weather tools.

User question: "What is 15 * 7 + 23?"

Analyze this request:
1. Determine if you need a tool (calculator for math)
2. Specify the tool and input
3. The calculator returned: {"result": 128.0, "expression": "15 * 7 + 23"}
4. Provide your final answer

Respond with your analysis."""

        result = get_chat_completion(
            message=prompt,
            provider=provider_name,
            model_id=model_id,
            provider_config=config,
            output_model=AgentResponse,
            max_tokens=400,
            temperature=0.0,
        )

        assert isinstance(result, AgentResponse)
        assert result.needs_tool is True
        assert "calculator" in result.tool_name.lower()
        assert "128" in result.final_answer
        assert len(result.reasoning) > 0
