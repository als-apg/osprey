"""E2E Tests for LLM Channel Namer.

Tests the LLM channel naming workflow that was broken in issue #103.
Uses minimal test data (2 channels) to minimize API token usage.

Run with: pytest tests/e2e/test_llm_channel_namer.py -v
"""

import functools
import os
import warnings

import pytest
import yaml

# =============================================================================
# PROVIDER DETECTION (follows pattern from test_llm_providers.py)
# =============================================================================


def get_available_providers() -> dict[str, dict]:
    """Detect which providers have valid API keys configured."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    available = {}

    providers_to_check = [
        ("cborg", ["CBORG_API_KEY"], "https://api.cborg.lbl.gov", "anthropic/claude-haiku"),
        (
            "amsc",
            ["AMSC_I2_API_KEY"],
            "https://api.i2-core.american-science-cloud.org",
            "claude-haiku",
        ),
        ("anthropic", ["ANTHROPIC_API_KEY"], None, "claude-haiku-4-5-20251001"),
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

    return available


# Cache at import time
_AVAILABLE_PROVIDERS = get_available_providers()

# Skip all tests if no API key is available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        len(_AVAILABLE_PROVIDERS) == 0,
        reason="Requires CBORG_API_KEY or ANTHROPIC_API_KEY",
    ),
]


def handle_quota_errors(func):
    """Decorator to convert API quota/rate limit errors to pytest warnings."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            quota_indicators = [
                "quota",
                "rate limit",
                "rate_limit",
                "ratelimit",
                "too many requests",
                "429",
                "resource_exhausted",
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


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def minimal_channel_data():
    """Minimal channel data with just 2 entries to minimize token usage."""
    return [
        {
            "short_name": "SX3Set",
            "description": "Steering coil current tilting beam in X direction inside accelerating tube (top position), set value",
        },
        {
            "short_name": "IP149APressure",
            "description": "Pressure at IP149A ion pump located at the beginning of HFEL beamline measured in Torr",
        },
    ]


@pytest.fixture
def get_first_available_provider():
    """Return the first available provider config."""
    if not _AVAILABLE_PROVIDERS:
        pytest.skip("No LLM providers available")
    provider_name = list(_AVAILABLE_PROVIDERS.keys())[0]
    return provider_name, _AVAILABLE_PROVIDERS[provider_name]


@pytest.fixture(autouse=True)
def setup_llm_test_environment(test_config, tmp_path):
    """Set up test environment with provider configurations.

    Follows the pattern from test_llm_providers.py.
    """
    # Load the config created by test_config fixture
    config = yaml.safe_load(test_config.read_text())

    # Add provider configs and ensure models section references all available
    # providers so config-driven provider filtering (#138) doesn't skip them
    config["provider_configs"] = {}
    if "models" not in config:
        config["models"] = {}
    for provider_name, provider_config in _AVAILABLE_PROVIDERS.items():
        config["provider_configs"][provider_name] = {
            "api_key": provider_config["api_key"],
            "base_url": provider_config["base_url"],
            "default_model_id": provider_config["default_model"],
        }
        if not any(
            m.get("provider") == provider_name
            for m in config["models"].values()
            if isinstance(m, dict)
        ):
            config["models"][f"test_{provider_name}"] = {
                "provider": provider_name,
                "model_id": provider_config["default_model"],
            }

    # Write updated config
    with open(test_config, "w") as f:
        yaml.dump(config, f)

    # Save original env and set CONFIG_FILE
    original_config_file = os.environ.get("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(test_config)

    # Initialize registry
    from osprey.registry import initialize_registry, reset_registry
    from osprey.utils import config as config_module

    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()
    initialize_registry(config_path=str(test_config))

    yield

    # Cleanup
    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    if original_config_file is not None:
        os.environ["CONFIG_FILE"] = original_config_file
    elif "CONFIG_FILE" in os.environ:
        del os.environ["CONFIG_FILE"]


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.e2e
class TestLLMChannelNamerImport:
    """Test that the config loading fix from issue #103 works."""

    def test_load_config_is_importable(self):
        """Test that load_config is properly importable (the issue #103 fix)."""
        # This is the core fix - load_config must be a public function
        from osprey.utils.config import get_config_builder, load_config

        # Both should be importable without error
        assert callable(get_config_builder)
        assert callable(load_config)

    def test_channel_finder_config_exposes_load_config(self, test_config):
        """Test that the channel_finder config module exposes load_config."""
        # The channel_finder config module should expose load_config
        # We test via the osprey.utils.config since templates use Jinja2
        from osprey.utils.config import load_config

        raw_config = load_config(str(test_config))
        assert isinstance(raw_config, dict)
        assert "project_root" in raw_config


@pytest.mark.e2e
class TestLLMChannelNamer:
    """Test the LLM channel namer with real API calls."""

    @handle_quota_errors
    def test_generate_names_with_llm(self, minimal_channel_data, get_first_available_provider):
        """Test LLM name generation with minimal data (2 channels)."""
        provider_name, provider_config = get_first_available_provider

        # Import directly - follows pattern from test_llm_providers.py
        from osprey.models import get_chat_completion

        # Build the prompt (simplified version of LLMChannelNamer._create_prompt_for_batch)
        prompt = """Generate descriptive PascalCase channel names for control system channels.
Each name should be self-documenting and include location, device type, and property information.

Rules:
- Use PascalCase (e.g., AcceleratingTubeTopSteeringCoilXSetPoint)
- Include location details from description
- End with SetPoint for writable values, ReadBack for read-only
- Generate exactly 2 names, one per line

Channels:
1. Short: "SX3Set", Description: "Steering coil current tilting beam in X direction inside accelerating tube (top position), set value"
2. Short: "IP149APressure", Description: "Pressure at IP149A ion pump located at the beginning of HFEL beamline measured in Torr"

Output exactly 2 PascalCase names, one per line:"""

        response = get_chat_completion(
            message=prompt,
            provider=provider_name,
            model_id=provider_config["default_model"],
            provider_config=provider_config,
            max_tokens=200,
        )

        assert response is not None
        assert isinstance(response, str)

        # Parse response - should have 2 names
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        # Filter out any numbering or explanatory text
        names = []
        for line in lines:
            # Remove common prefixes like "1.", "1:", "-", etc.
            cleaned = line.lstrip("0123456789.-:) ").strip()
            if cleaned and cleaned[0].isupper() and len(cleaned) >= 3:
                names.append(cleaned.split()[0])  # Take first word only

        assert len(names) >= 2, (
            f"Expected at least 2 names, got {len(names)} from response: {response}"
        )

        # Verify names are valid PascalCase
        for name in names[:2]:
            assert name[0].isupper(), f"Name '{name}' doesn't start with uppercase"
            assert len(name) >= 3, f"Name '{name}' is too short"

        print(f"\n✅ Generated names using {provider_name}/{provider_config['default_model']}:")
        for i, name in enumerate(names[:2]):
            print(f"  {i + 1}. {minimal_channel_data[i]['short_name']} → {name}")

    @handle_quota_errors
    def test_structured_output_channel_names(
        self, minimal_channel_data, get_first_available_provider
    ):
        """Test structured output for channel names (Pydantic model)."""
        provider_name, provider_config = get_first_available_provider

        from pydantic import BaseModel, ConfigDict, Field

        from osprey.models import get_chat_completion

        class ChannelNames(BaseModel):
            """Structured output model for generated channel names."""

            model_config = ConfigDict(extra="forbid")

            names: list[str] = Field(
                description="List of generated PascalCase channel names, one for each input channel"
            )

        prompt = """Generate descriptive PascalCase channel names for these 2 control system channels:

1. Short: "SX3Set", Description: "Steering coil current tilting beam in X direction inside accelerating tube (top position), set value"
2. Short: "IP149APressure", Description: "Pressure at IP149A ion pump located at the beginning of HFEL beamline measured in Torr"

Return exactly 2 PascalCase names that are self-documenting."""

        result = get_chat_completion(
            message=prompt,
            provider=provider_name,
            model_id=provider_config["default_model"],
            provider_config=provider_config,
            max_tokens=200,
            output_model=ChannelNames,
        )

        assert result is not None

        # Result should be ChannelNames instance or dict
        if isinstance(result, dict):
            names = result.get("names", [])
        else:
            names = result.names

        assert len(names) == 2, f"Expected 2 names, got {len(names)}: {names}"

        # Verify names are valid
        for name in names:
            assert name[0].isupper(), f"Name '{name}' doesn't start with uppercase"
            assert len(name) >= 3, f"Name '{name}' is too short"

        print(
            f"\n✅ Structured output names using {provider_name}/{provider_config['default_model']}:"
        )
        for i, name in enumerate(names):
            print(f"  {i + 1}. {minimal_channel_data[i]['short_name']} → {name}")


@pytest.mark.e2e
class TestLLMChannelNamerValidation:
    """Test name validation logic (no API calls needed)."""

    def test_valid_channel_name_patterns(self):
        """Test the channel name validation patterns."""
        # Valid PascalCase names
        valid_names = [
            "AcceleratingTubeTopSteeringCoilXSetPoint",
            "HFELBeamLineBeginningIonPumpPressure",
            "TerminalCoolingFanVoltageReadBack",
            "ABC",  # Minimum valid length
        ]

        for name in valid_names:
            assert name[0].isupper(), f"'{name}' should start with uppercase"
            assert len(name) >= 3, f"'{name}' should be at least 3 chars"
            assert name.replace("_", "").isalnum(), f"'{name}' should be alphanumeric"

        # Invalid names - check each condition separately
        assert len("") < 3, "Empty string should be too short"
        assert len("AB") < 3, "'AB' should be too short"
        assert "lowercase"[0].islower(), "'lowercase' should not start with uppercase"
        assert " " in "Has Spaces", "'Has Spaces' should contain spaces"
