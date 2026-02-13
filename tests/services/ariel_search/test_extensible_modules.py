"""Tests for ARIEL extensible module system.

Validates:
- Pipeline descriptor named lookup
- Capabilities shared parameters
"""

import pytest

from osprey.services.ariel_search.pipelines import (
    AGENT_PIPELINE,
    RAG_PIPELINE,
    get_pipeline_descriptor,
    get_pipeline_descriptors,
)


class TestPipelineDescriptorLookup:
    """Test get_pipeline_descriptor() named lookup."""

    def test_lookup_rag(self):
        descriptor = get_pipeline_descriptor("rag")
        assert descriptor is RAG_PIPELINE
        assert descriptor.name == "rag"
        assert descriptor.category == "llm"

    def test_lookup_agent(self):
        descriptor = get_pipeline_descriptor("agent")
        assert descriptor is AGENT_PIPELINE
        assert descriptor.name == "agent"

    def test_lookup_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_pipeline_descriptor("nonexistent")

    def test_get_pipeline_descriptors_unchanged(self):
        """Existing function still returns all pipelines."""
        descriptors = get_pipeline_descriptors()
        names = [d.name for d in descriptors]
        assert "rag" in names
        assert "agent" in names


class TestCapabilitiesSharedParameters:
    """Test capabilities.py shared parameters."""

    def test_shared_parameters_intact(self):
        """SHARED_PARAMETERS are still present."""
        from osprey.services.ariel_search.capabilities import SHARED_PARAMETERS

        names = [p.name for p in SHARED_PARAMETERS]
        assert "max_results" in names
        assert "start_date" in names
        assert "end_date" in names
