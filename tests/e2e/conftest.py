"""Pytest fixtures for end-to-end workflow tests.

⚠️ IMPORTANT: Run these tests with 'pytest tests/e2e/' NOT 'pytest -m e2e'
See tests/e2e/README.md for details.
"""

import json
import os

import pytest

from osprey.registry import reset_registry


def pytest_runtest_logreport(report):
    """Stream each test outcome to OSPREY_E2E_LIVE (one JSON line per test) so the
    model-matrix dashboard fills in test-by-test AS the run progresses, instead of
    only when the whole 92-test suite finishes (issue #259). No-op unless the env
    var is set, so normal e2e runs are unaffected. Timeouts (pytest-timeout) are
    detected from the failure text and tagged distinctly (never a plain 'failed')."""
    live = os.environ.get("OSPREY_E2E_LIVE")
    if not live:
        return
    # one record per test: skips/setup-errors on 'setup', pass/fail on 'call'
    if report.when == "call" or (
        report.when == "setup" and report.outcome in ("skipped", "failed")
    ):
        outcome = report.outcome
        if outcome == "failed":
            text = str(getattr(report, "longrepr", "")).lower()
            outcome = (
                "timeout"
                if "timeout" in text
                else ("error" if report.when == "setup" else "failed")
            )
        try:
            with open(live, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "name": report.nodeid,
                            "outcome": outcome,
                            "duration_s": round(getattr(report, "duration", 0.0), 2),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass


def pytest_configure(config):
    """Register E2E markers and warn if tests are being run incorrectly."""
    # Register custom markers
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests (requires API keys, slow)")
    config.addinivalue_line("markers", "e2e_smoke: Quick smoke tests for critical workflows")
    config.addinivalue_line("markers", "e2e_tutorial: Tutorial workflow validation tests")
    config.addinivalue_line("markers", "e2e_benchmark: Channel finder benchmark validation tests")

    # Warn if invoked with `-m e2e` from outside tests/e2e/ (causes registry leaks)
    if config.option.markexpr and "e2e" in config.option.markexpr:
        invocation_dir = config.invocation_params.dir
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


@pytest.fixture(autouse=True, scope="session")
def _e2e_translation_proxy():
    """Start the OSPREY translation proxy for the whole session when targeting
    an OpenAI-protocol (open) CBORG model — issue #259 model matrix.

    The SDK e2e path does not start the proxy on its own (only ``osprey claude``
    does), so open models would otherwise route an Anthropic request straight at
    CBORG's OpenAI endpoint and fail. When ``OSPREY_E2E_PROXY_UPSTREAM`` is set
    (by ``scripts/run_e2e_for_model.sh`` for non-``claude-*`` models), we start
    the same in-process proxy the L5 harness uses and publish its loopback URL
    via ``OSPREY_E2E_PROXY_BASE_URL``; ``_apply_e2e_overrides`` swaps that into
    each project's ``ANTHROPIC_BASE_URL``. The proxy thread lives in the pytest
    process and serves every ``claude`` CLI subprocess on 127.0.0.1.

    Inert (pure no-op) when ``OSPREY_E2E_PROXY_UPSTREAM`` is unset, so normal
    e2e runs and the direct-routing ``claude-*`` models are unaffected.
    """
    upstream = os.environ.get("OSPREY_E2E_PROXY_UPSTREAM")
    if not upstream:
        yield
        return
    from osprey.infrastructure.proxy.lifecycle import start_proxy, stop_proxy

    key = os.environ.get("CBORG_API_KEY", "")
    port = start_proxy(upstream, key)
    os.environ["OSPREY_E2E_PROXY_BASE_URL"] = f"http://127.0.0.1:{port}"
    try:
        yield
    finally:
        os.environ.pop("OSPREY_E2E_PROXY_BASE_URL", None)
        stop_proxy()


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


def pytest_addoption(parser):
    """Add custom command-line options for E2E tests."""
    parser.addoption(
        "--e2e-verbose",
        action="store_true",
        default=False,
        help="Show real-time progress updates during E2E test execution",
    )
