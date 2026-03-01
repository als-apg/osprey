"""Pytest fixtures for end-to-end workflow tests.

⚠️ IMPORTANT: Run these tests with 'pytest tests/e2e/' NOT 'pytest -m e2e'
See tests/e2e/README.md for details.
"""

import os
from pathlib import Path

import pytest

from osprey.registry import reset_registry
from tests.e2e.judge import LLMJudge


# Warn if tests are being run the wrong way
def pytest_configure(config):
    """Warn users if e2e tests are being run incorrectly."""
    # Check if we're running with -m e2e marker from outside tests/e2e/
    if config.option.markexpr and "e2e" in config.option.markexpr:
        # Get the invocation directory
        invocation_dir = config.invocation_params.dir
        Path(__file__).parent

        # If not invoked from tests/e2e/ directory, warn
        if not str(invocation_dir).endswith("tests/e2e"):
            import warnings

            warnings.warn(
                "\n" + "=" * 80 + "\n"
                "⚠️  WARNING: You are running e2e tests with '-m e2e' marker!\n"
                "\n"
                "This can cause test collection order issues and registry failures.\n"
                "\n"
                "✅ CORRECT way to run e2e tests:\n"
                "   pytest tests/e2e/ -v\n"
                "\n"
                "❌ AVOID using:\n"
                "   pytest -m e2e\n"
                "\n"
                "See tests/e2e/README.md for details.\n" + "=" * 80,
                UserWarning,
                stacklevel=2,
            )


@pytest.fixture(autouse=True, scope="function")
def reset_registry_between_tests():
    """Auto-reset registry before each e2e test to ensure isolation.

    This is critical for e2e tests as they create full framework instances
    with registries that can leak state between tests.

    IMPORTANT: Also clears config cache AND CONFIG_FILE env var to prevent
    stale configuration from leaking between tests that use different config files.
    """
    # Reset before test
    reset_registry()

    # CRITICAL: Clear config cache to prevent stale config from previous tests
    # The config module has global caches that persist across registry resets
    from osprey.utils import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Clear CONFIG_FILE environment variable to prevent contamination
    if "CONFIG_FILE" in os.environ:
        del os.environ["CONFIG_FILE"]

    yield

    # Reset after test for good measure
    reset_registry()

    # Clear config cache again after test
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Clear CONFIG_FILE env var again
    if "CONFIG_FILE" in os.environ:
        del os.environ["CONFIG_FILE"]


@pytest.fixture
def llm_judge(request):
    """Fixture providing an LLM judge for evaluation.

    Can be configured via pytest command line:
        pytest --judge-provider=cborg --judge-model=anthropic/claude-haiku
    """
    provider = request.config.getoption("--judge-provider", default="cborg")
    model = request.config.getoption("--judge-model", default="anthropic/claude-haiku")
    verbose = request.config.getoption("--judge-verbose", default=False)

    return LLMJudge(provider=provider, model=model, verbose=verbose)


def pytest_addoption(parser):
    """Add custom command-line options for E2E tests."""
    parser.addoption(
        "--judge-provider",
        action="store",
        default="cborg",
        help="AI provider to use for LLM judge evaluation",
    )
    parser.addoption(
        "--judge-model",
        action="store",
        default="anthropic/claude-haiku",
        help="Model to use for LLM judge evaluation",
    )
    parser.addoption(
        "--judge-verbose",
        action="store_true",
        default=False,
        help="Print detailed judge evaluation information",
    )
    parser.addoption(
        "--e2e-verbose",
        action="store_true",
        default=False,
        help="Show real-time progress updates during E2E test execution",
    )


def pytest_configure(config):
    """Register custom markers for E2E tests."""
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests (requires API keys, slow)")
    config.addinivalue_line("markers", "e2e_smoke: Quick smoke tests for critical workflows")
    config.addinivalue_line("markers", "e2e_tutorial: Tutorial workflow validation tests")
    config.addinivalue_line("markers", "e2e_benchmark: Channel finder benchmark validation tests")
