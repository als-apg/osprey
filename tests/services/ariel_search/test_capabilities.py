"""Tests for ARIEL capabilities assembly and parameter descriptors."""

from __future__ import annotations

import pytest

from osprey.services.ariel_search.capabilities import get_capabilities
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.pipelines import (
    AGENT_PIPELINE,
    RAG_PIPELINE,
    get_pipeline_descriptors,
)
from osprey.services.ariel_search.search.base import ParameterDescriptor
from osprey.services.ariel_search.search.keyword import (
    get_parameter_descriptors as keyword_params,
)
from osprey.services.ariel_search.search.semantic import (
    get_parameter_descriptors as semantic_params,
)


class TestParameterDescriptor:
    """Tests for ParameterDescriptor dataclass."""

    def test_create_float_param(self):
        """Float parameter descriptor has correct fields."""
        param = ParameterDescriptor(
            name="threshold",
            label="Threshold",
            description="A threshold",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            section="Retrieval",
        )
        assert param.name == "threshold"
        assert param.param_type == "float"
        assert param.default == 0.7
        assert param.min_value == 0.0
        assert param.max_value == 1.0

    def test_create_bool_param(self):
        """Bool parameter descriptor has correct fields."""
        param = ParameterDescriptor(
            name="enable_flag",
            label="Enable Flag",
            description="A boolean flag",
            param_type="bool",
            default=True,
            section="Options",
        )
        assert param.param_type == "bool"
        assert param.default is True
        assert param.min_value is None

    def test_to_dict(self):
        """to_dict serializes correctly."""
        param = ParameterDescriptor(
            name="threshold",
            label="Threshold",
            description="A threshold",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            section="Retrieval",
        )
        d = param.to_dict()
        assert d["name"] == "threshold"
        assert d["type"] == "float"
        assert d["default"] == 0.7
        assert d["min"] == 0.0
        assert d["max"] == 1.0
        assert d["step"] == 0.01
        assert d["section"] == "Retrieval"

    def test_to_dict_omits_none_values(self):
        """to_dict omits min/max/step/options when None."""
        param = ParameterDescriptor(
            name="flag",
            label="Flag",
            description="A flag",
            param_type="bool",
            default=False,
        )
        d = param.to_dict()
        assert "min" not in d
        assert "max" not in d
        assert "step" not in d
        assert "options" not in d

    def test_to_dict_includes_placeholder(self):
        """to_dict includes placeholder when set."""
        param = ParameterDescriptor(
            name="author",
            label="Author",
            description="Filter by author",
            param_type="text",
            default=None,
            placeholder="Filter by author...",
        )
        d = param.to_dict()
        assert d["placeholder"] == "Filter by author..."

    def test_to_dict_includes_options_endpoint(self):
        """to_dict includes options_endpoint when set."""
        param = ParameterDescriptor(
            name="source",
            label="Source",
            description="Filter by source",
            param_type="dynamic_select",
            default=None,
            options_endpoint="/api/filter-options/source_systems",
        )
        d = param.to_dict()
        assert d["options_endpoint"] == "/api/filter-options/source_systems"

    def test_to_dict_omits_placeholder_when_none(self):
        """to_dict omits placeholder when None."""
        param = ParameterDescriptor(
            name="x",
            label="X",
            description="x",
            param_type="text",
            default=None,
        )
        d = param.to_dict()
        assert "placeholder" not in d
        assert "options_endpoint" not in d

    def test_create_date_param(self):
        """Date parameter descriptor has correct fields."""
        param = ParameterDescriptor(
            name="start_date",
            label="Start Date",
            description="Filter start date",
            param_type="date",
            default=None,
            section="Filters",
        )
        assert param.param_type == "date"
        d = param.to_dict()
        assert d["type"] == "date"

    def test_create_dynamic_select_param(self):
        """Dynamic select parameter descriptor has correct fields."""
        param = ParameterDescriptor(
            name="source",
            label="Source",
            description="Filter by source",
            param_type="dynamic_select",
            default=None,
            options_endpoint="/api/filter-options/source_systems",
        )
        assert param.param_type == "dynamic_select"
        assert param.options_endpoint == "/api/filter-options/source_systems"

    def test_frozen(self):
        """ParameterDescriptor is frozen (immutable)."""
        param = ParameterDescriptor(
            name="x",
            label="X",
            description="x",
            param_type="int",
            default=1,
        )
        with pytest.raises(AttributeError):
            param.name = "y"  # type: ignore[misc]


class TestKeywordParameterDescriptors:
    """Tests for keyword module parameter descriptors."""

    def test_returns_list(self):
        """keyword get_parameter_descriptors returns a list."""
        params = keyword_params()
        assert isinstance(params, list)
        assert len(params) == 2

    def test_include_highlights_descriptor(self):
        """include_highlights descriptor has correct attributes."""
        params = {p.name: p for p in keyword_params()}
        assert "include_highlights" in params
        p = params["include_highlights"]
        assert p.param_type == "bool"
        assert p.default is True
        assert p.section == "Options"

    def test_fuzzy_fallback_descriptor(self):
        """fuzzy_fallback descriptor has correct attributes."""
        params = {p.name: p for p in keyword_params()}
        assert "fuzzy_fallback" in params
        p = params["fuzzy_fallback"]
        assert p.param_type == "bool"
        assert p.default is True


class TestSemanticParameterDescriptors:
    """Tests for semantic module parameter descriptors."""

    def test_returns_list(self):
        """semantic get_parameter_descriptors returns a list."""
        params = semantic_params()
        assert isinstance(params, list)
        assert len(params) == 1

    def test_similarity_threshold_descriptor(self):
        """similarity_threshold descriptor has correct attributes."""
        params = {p.name: p for p in semantic_params()}
        assert "similarity_threshold" in params
        p = params["similarity_threshold"]
        assert p.param_type == "float"
        assert p.default == 0.7
        assert p.min_value == 0.0
        assert p.max_value == 1.0
        assert p.step == 0.01
        assert p.section == "Retrieval"


class TestPipelineDescriptors:
    """Tests for pipeline descriptors."""

    def test_get_pipeline_descriptors_returns_list(self):
        """get_pipeline_descriptors returns a list."""
        pipelines = get_pipeline_descriptors()
        assert isinstance(pipelines, list)
        assert len(pipelines) >= 2

    def test_rag_pipeline_descriptor(self):
        """RAG pipeline descriptor has correct attributes."""
        assert RAG_PIPELINE.name == "rag"
        assert RAG_PIPELINE.category == "llm"
        assert len(RAG_PIPELINE.parameters) == 4
        param_names = {p.name for p in RAG_PIPELINE.parameters}
        assert "similarity_threshold" in param_names
        assert "max_context_chars" in param_names
        assert "max_chars_per_entry" in param_names
        assert "temperature" in param_names

    def test_agent_pipeline_descriptor(self):
        """Agent pipeline descriptor has correct attributes."""
        assert AGENT_PIPELINE.name == "agent"
        assert AGENT_PIPELINE.category == "llm"
        assert len(AGENT_PIPELINE.parameters) == 1
        assert AGENT_PIPELINE.parameters[0].name == "temperature"

    def test_pipeline_descriptor_frozen(self):
        """PipelineDescriptor is frozen."""
        with pytest.raises(AttributeError):
            RAG_PIPELINE.name = "other"  # type: ignore[misc]


class TestGetCapabilities:
    """Tests for get_capabilities() function."""

    def _make_config(
        self,
        search_modules: dict | None = None,
        pipelines: dict | None = None,
    ) -> ARIELConfig:
        """Create an ARIELConfig for testing."""
        config_dict: dict = {
            "database": {"uri": "postgresql://localhost:5432/test"},
        }
        if search_modules:
            config_dict["search_modules"] = search_modules
        if pipelines:
            config_dict["pipelines"] = pipelines
        return ARIELConfig.from_dict(config_dict)

    def test_returns_correct_structure(self):
        """get_capabilities returns correct top-level structure."""
        config = self._make_config(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test"},
            }
        )
        result = get_capabilities(config)

        assert "categories" in result
        assert "shared_parameters" in result
        assert "llm" in result["categories"]
        assert "direct" in result["categories"]

    def test_includes_enabled_search_modules(self):
        """Enabled search modules appear as direct modes."""
        config = self._make_config(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test"},
            }
        )
        result = get_capabilities(config)
        direct_modes = result["categories"]["direct"]["modes"]
        mode_names = [m["name"] for m in direct_modes]

        assert "keyword" in mode_names
        assert "semantic" in mode_names

    def test_excludes_disabled_search_modules(self):
        """Disabled search modules do not appear."""
        config = self._make_config(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
            }
        )
        result = get_capabilities(config)
        direct_modes = result["categories"]["direct"]["modes"]
        mode_names = [m["name"] for m in direct_modes]

        assert "keyword" in mode_names
        assert "semantic" not in mode_names

    def test_includes_pipelines(self):
        """Pipeline modes appear under llm category."""
        config = self._make_config()
        result = get_capabilities(config)
        llm_modes = result["categories"]["llm"]["modes"]
        mode_names = [m["name"] for m in llm_modes]

        assert "rag" in mode_names
        assert "agent" in mode_names

    def test_modes_have_parameters(self):
        """Each mode includes its parameter descriptors."""
        config = self._make_config(search_modules={"keyword": {"enabled": True}})
        result = get_capabilities(config)

        # Find keyword mode
        direct_modes = result["categories"]["direct"]["modes"]
        keyword_mode = next(m for m in direct_modes if m["name"] == "keyword")

        assert "parameters" in keyword_mode
        assert len(keyword_mode["parameters"]) == 2
        param_names = [p["name"] for p in keyword_mode["parameters"]]
        assert "include_highlights" in param_names
        assert "fuzzy_fallback" in param_names

    def test_shared_parameters_included(self):
        """Shared parameters are included in the response."""
        config = self._make_config()
        result = get_capabilities(config)

        assert len(result["shared_parameters"]) > 0
        param_names = [p["name"] for p in result["shared_parameters"]]
        assert "max_results" in param_names
        assert "start_date" in param_names
        assert "end_date" in param_names
        assert "author" in param_names
        assert "source_system" in param_names

    def test_shared_filter_params_have_correct_types(self):
        """Filter shared params have correct param types."""
        config = self._make_config()
        result = get_capabilities(config)
        params = {p["name"]: p for p in result["shared_parameters"]}

        assert params["start_date"]["type"] == "date"
        assert params["end_date"]["type"] == "date"
        assert params["author"]["type"] == "text"
        assert params["source_system"]["type"] == "dynamic_select"

    def test_author_param_has_placeholder(self):
        """Author parameter includes placeholder."""
        config = self._make_config()
        result = get_capabilities(config)
        params = {p["name"]: p for p in result["shared_parameters"]}

        assert "placeholder" in params["author"]
        assert params["author"]["placeholder"] == "Filter by author..."

    def test_source_system_param_has_options_endpoint(self):
        """Source system parameter includes options_endpoint."""
        config = self._make_config()
        result = get_capabilities(config)
        params = {p["name"]: p for p in result["shared_parameters"]}

        assert "options_endpoint" in params["source_system"]
        assert params["source_system"]["options_endpoint"] == "/api/filter-options/source_systems"

    def test_no_search_modules_returns_empty_direct(self):
        """No search modules produces empty direct modes list."""
        config = self._make_config()
        result = get_capabilities(config)
        direct_modes = result["categories"]["direct"]["modes"]
        assert direct_modes == []

    def test_excludes_disabled_pipelines(self):
        """Disabled pipelines do not appear in llm modes."""
        config = self._make_config(
            pipelines={
                "rag": {"enabled": False},
                "agent": {"enabled": True},
            }
        )
        result = get_capabilities(config)
        llm_modes = result["categories"]["llm"]["modes"]
        mode_names = [m["name"] for m in llm_modes]

        assert "rag" not in mode_names
        assert "agent" in mode_names

    def test_pipeline_respects_config(self):
        """Only enabled pipelines appear in capabilities."""
        config = self._make_config(
            pipelines={
                "rag": {"enabled": True},
                "agent": {"enabled": False},
            }
        )
        result = get_capabilities(config)
        llm_modes = result["categories"]["llm"]["modes"]
        mode_names = [m["name"] for m in llm_modes]

        assert "rag" in mode_names
        assert "agent" not in mode_names
