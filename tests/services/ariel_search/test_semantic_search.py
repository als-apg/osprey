"""Tests for the ARIEL semantic search module's settings parser.

The parser is what turns ``search_modules.semantic.settings`` from a bag of
whatever YAML happened to hold into a typed value the query path can trust. A
present-but-malformed threshold used to travel all the way to the repository
(and to the capabilities slider) as-is; these tests pin the refusal instead.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    SearchModuleConfig,
)
from osprey.services.ariel_search.search.semantic import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SemanticSearchSettings,
    get_parameter_descriptors,
    semantic_search,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_config(settings: dict[str, Any] | None = None) -> ARIELConfig:
    """Build an ARIELConfig whose ``semantic`` module carries *settings*."""
    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
    config.search_modules["semantic"] = SearchModuleConfig(enabled=True, settings=settings or {})
    return config


def make_wired_config(settings: dict[str, Any] | None = None) -> ARIELConfig:
    """Build a config complete enough for :func:`semantic_search` to run."""
    module: dict[str, Any] = {"enabled": True, "model": "test-model"}
    if settings is not None:
        module["settings"] = settings
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {"semantic": module},
        }
    )


@pytest.fixture
def mock_repository():
    """A repository that records the arguments the query path hands it."""
    repo = MagicMock()
    repo.semantic_search = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_embedder():
    """An embedding provider that answers with a fixed vector."""
    embedder = MagicMock()
    embedder.default_base_url = "http://localhost:11434"
    embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return embedder


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class TestSettings:
    """``search_modules.semantic.settings`` resolution."""

    def test_defaults_when_unconfigured(self):
        settings = SemanticSearchSettings.from_ariel_config(make_config())
        assert settings.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_defaults_when_module_absent_entirely(self):
        bare = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
        parsed = SemanticSearchSettings.from_ariel_config(bare)
        assert parsed.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_defaults_when_config_is_none(self):
        parsed = SemanticSearchSettings.from_ariel_config(None)
        assert parsed.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_config_sets_threshold(self):
        settings = SemanticSearchSettings.from_ariel_config(
            make_config({"similarity_threshold": 0.85})
        )
        assert settings.similarity_threshold == 0.85

    @pytest.mark.parametrize("value", [0, 1])
    def test_integer_bounds_are_accepted_as_floats(self, value):
        settings = SemanticSearchSettings.from_ariel_config(
            make_config({"similarity_threshold": value})
        )
        assert settings.similarity_threshold == float(value)
        assert isinstance(settings.similarity_threshold, float)

    @pytest.mark.parametrize("bad", ["0.8", None, [], {}])
    def test_malformed_threshold_is_refused(self, bad):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    @pytest.mark.parametrize("bad", [True, False])
    def test_boolean_threshold_is_refused(self, bad):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    @pytest.mark.parametrize("bad", [-0.1, 1.5, -1, 2])
    def test_out_of_range_threshold_is_refused(self, bad):
        with pytest.raises(ValueError, match=r"must be a float in \[0, 1\]"):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    def test_settings_are_frozen(self):
        settings = SemanticSearchSettings.from_ariel_config(make_config())
        with pytest.raises(Exception):
            settings.similarity_threshold = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------
# Wiring into the query path
# --------------------------------------------------------------------------


class TestThresholdResolution:
    """What :func:`semantic_search` hands the repository."""

    @pytest.mark.asyncio
    async def test_explicit_argument_wins_over_config(self, mock_repository, mock_embedder):
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": 0.8}),
            mock_embedder,
            similarity_threshold=0.3,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_config_value_is_used_without_an_argument(self, mock_repository, mock_embedder):
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": 0.9}),
            mock_embedder,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_default_applies_when_settings_are_absent(self, mock_repository, mock_embedder):
        await semantic_search("test", mock_repository, make_wired_config(), mock_embedder)
        assert (
            mock_repository.semantic_search.call_args.kwargs["similarity_threshold"]
            == DEFAULT_SIMILARITY_THRESHOLD
        )

    @pytest.mark.asyncio
    async def test_malformed_settings_raise_instead_of_reaching_the_repository(
        self, mock_repository, mock_embedder
    ):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            await semantic_search(
                "test",
                mock_repository,
                make_wired_config({"similarity_threshold": "high"}),
                mock_embedder,
            )
        mock_repository.semantic_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_argument_short_circuits_the_config_block(
        self, mock_repository, mock_embedder
    ):
        # A caller that named the threshold never reaches the config block, so
        # a malformed one does not fail a query that had no use for it. The
        # deployment-wide answer to a malformed block is startup validation,
        # not a per-query surprise on the one path that already has a value.
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": "high"}),
            mock_embedder,
            similarity_threshold=0.4,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.4

    @pytest.mark.asyncio
    async def test_empty_query_returns_before_parsing(self, mock_repository, mock_embedder):
        # The empty-query short circuit stays ahead of the parser, so a blank
        # box in the panel is not the thing that surfaces a config error.
        result = await semantic_search(
            "   ",
            mock_repository,
            make_wired_config({"similarity_threshold": "high"}),
            mock_embedder,
        )
        assert result == []


# --------------------------------------------------------------------------
# Parameter descriptors
# --------------------------------------------------------------------------


class TestParameterDescriptors:
    """What the capabilities API reports as the panel's starting point.

    Only the semantic module grew a ``config`` parameter alongside hybrid's.
    The keyword module's two descriptors (``include_highlights``,
    ``fuzzy_fallback``) have no counterpart in ``KeywordSearchSettings``, which
    covers ``patterns_enabled`` and ``pattern_timeout_seconds`` instead — there
    is no configured value for them to report, so ``keyword`` stays zero-arg
    deliberately rather than by oversight.
    """

    def test_defaults_to_the_shipped_threshold_with_no_config(self):
        descriptor = get_parameter_descriptors()[0]
        assert descriptor.name == "similarity_threshold"
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    def test_bounds_are_the_slider_the_panel_draws(self):
        descriptor = get_parameter_descriptors()[0]
        assert descriptor.param_type == "float"
        assert descriptor.min_value == 0.0
        assert descriptor.max_value == 1.0
        assert descriptor.step == 0.01
        assert descriptor.section == "Retrieval"

    def test_reports_the_configured_threshold(self):
        """The panel opens on what a query would do, not on what ships."""
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": 0.8}))[0]
        assert descriptor.default == 0.8

    def test_malformed_config_falls_back_instead_of_raising(self):
        """Describing the module survives a key the query path would refuse.

        Without the fallback the capabilities endpoint raises and the slider
        renders with no default at all; startup validation is what names the
        offending key.
        """
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": "high"}))[0]
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.parametrize("bad", [True, -0.1, 1.5, None])
    def test_every_refused_spelling_falls_back(self, bad):
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": bad}))[0]
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    def test_config_of_none_matches_the_zero_argument_call(self):
        assert get_parameter_descriptors(None) == get_parameter_descriptors()
