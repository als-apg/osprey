"""Tests for channel finder prompt loader.

Validates function signatures and call-site compatibility to prevent
TypeError regressions from signature mismatches.
"""

import inspect

import pytest

from osprey.services.channel_finder.utils.prompt_loader import (
    _try_load_builtin_prompts,
    _try_load_facility_prompts,
    _try_load_prompts_directly,
    load_prompts,
)


class TestFunctionSignatures:
    """Verify that internal loader functions accept the args passed by load_prompts."""

    def test_try_load_facility_prompts_accepts_require_query_splitter(self):
        """_try_load_facility_prompts must accept require_query_splitter param.

        Regression test: the call site in load_prompts() passes 3 positional args
        (facility_path, pipeline_mode, require_query_splitter) but the function
        previously only accepted 2, causing TypeError at runtime.
        """
        sig = inspect.signature(_try_load_facility_prompts)
        params = list(sig.parameters.keys())
        assert "require_query_splitter" in params

    def test_try_load_prompts_directly_accepts_require_query_splitter(self):
        """_try_load_prompts_directly must accept require_query_splitter param."""
        sig = inspect.signature(_try_load_prompts_directly)
        params = list(sig.parameters.keys())
        assert "require_query_splitter" in params

    def test_try_load_builtin_prompts_accepts_require_query_splitter(self):
        """_try_load_builtin_prompts must accept require_query_splitter param."""
        sig = inspect.signature(_try_load_builtin_prompts)
        params = list(sig.parameters.keys())
        assert "require_query_splitter" in params

    def test_all_loaders_have_consistent_signatures(self):
        """All three loader functions should accept (path, pipeline_mode, require_query_splitter)."""
        for func in [_try_load_prompts_directly, _try_load_facility_prompts]:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            assert len(params) >= 3, (
                f"{func.__name__} should accept at least 3 params, got {params}"
            )
            assert params[1] == "pipeline_mode", f"{func.__name__} param[1] should be pipeline_mode"
            assert params[2] == "require_query_splitter", (
                f"{func.__name__} param[2] should be require_query_splitter"
            )


class TestLoadPromptsCallSiteCompatibility:
    """Test that load_prompts can invoke internal loaders without TypeError."""

    @pytest.fixture
    def minimal_config(self):
        return {"channel_finder": {"pipeline_mode": "in_context"}}

    def test_load_prompts_with_facility_config_no_typeerror(self, minimal_config):
        """load_prompts should not raise TypeError when facility prompts are configured.

        Regression: _try_load_facility_prompts previously lacked require_query_splitter
        param, causing TypeError when called from load_prompts.
        """
        minimal_config["facility"] = {
            "prompts": "facility",
            "path": "/nonexistent/facility/path",
        }
        # Should not raise TypeError; falls back to built-in prompts
        result = load_prompts(minimal_config)
        assert result is not None

    def test_load_prompts_with_query_splitting_disabled(self, minimal_config):
        """load_prompts should work with require_query_splitter=False."""
        minimal_config["facility"] = {
            "prompts": "facility",
            "path": "/nonexistent/facility/path",
        }
        result = load_prompts(minimal_config, require_query_splitter=False)
        assert result is not None
