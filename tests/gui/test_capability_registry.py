"""
Tests for CapabilityRegistry

Tests the capability registry functionality including registration,
lookup, disambiguation, and metadata management.
"""

from unittest.mock import Mock

import pytest

from osprey.interfaces.pyqt.capability_registry import (
    AmbiguousCapabilityError,
    CapabilityRegistry,
)


class TestCapabilityRegistryInitialization:
    """Test suite for CapabilityRegistry initialization."""

    def test_init_creates_empty_registry(self):
        """Test initialization creates empty registry."""
        registry = CapabilityRegistry()

        assert registry._capabilities == {}
        assert registry._metadata == {}

    def test_init_has_logger(self):
        """Test initialization includes logger."""
        registry = CapabilityRegistry()

        assert registry.logger is not None


class TestRegisterProjectCapabilities:
    """Test suite for registering project capabilities."""

    def test_register_single_capability(self):
        """Test registering a single capability."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})

        assert "cap1" in registry._capabilities
        assert "project1" in registry._capabilities["cap1"]
        assert registry._capabilities["cap1"]["project1"] is cap

    def test_register_multiple_capabilities(self):
        """Test registering multiple capabilities."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1, "cap2": cap2})

        assert len(registry._capabilities) == 2
        assert registry._capabilities["cap1"]["project1"] is cap1
        assert registry._capabilities["cap2"]["project1"] is cap2

    def test_register_with_metadata(self):
        """Test registering capabilities with metadata."""
        registry = CapabilityRegistry()
        cap = Mock()
        metadata = Mock(description="Test capability")

        registry.register_project_capabilities("project1", {"cap1": cap}, {"cap1": metadata})

        assert registry._metadata["project1"]["cap1"] is metadata

    def test_register_same_capability_different_projects(self):
        """Test registering same capability name in different projects."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        assert len(registry._capabilities["cap1"]) == 2
        assert registry._capabilities["cap1"]["project1"] is cap1
        assert registry._capabilities["cap1"]["project2"] is cap2

    def test_register_updates_existing_project(self):
        """Test registering updates existing project capabilities."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project1", {"cap2": cap2})

        # Both capabilities should be registered
        assert "cap1" in registry._capabilities
        assert "cap2" in registry._capabilities


class TestGetCapability:
    """Test suite for getting capabilities."""

    def test_get_capability_by_name(self):
        """Test getting capability by name."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        result = registry.get_capability("cap1")

        assert result is cap

    def test_get_capability_with_project(self):
        """Test getting capability with project specification."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        result = registry.get_capability("cap1", "project2")
        assert result is cap2

    def test_get_nonexistent_capability(self):
        """Test getting nonexistent capability returns None."""
        registry = CapabilityRegistry()

        result = registry.get_capability("nonexistent")
        assert result is None

    def test_get_ambiguous_capability_raises_error(self):
        """Test getting ambiguous capability raises error."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        with pytest.raises(AmbiguousCapabilityError):
            registry.get_capability("cap1")

    def test_get_capability_wrong_project(self):
        """Test getting capability from wrong project returns None."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        result = registry.get_capability("cap1", "project2")

        assert result is None


class TestGetCapabilityWithProject:
    """Test suite for getting capability with project name."""

    def test_get_capability_with_project_returns_tuple(self):
        """Test returns tuple of (project, capability)."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        result = registry.get_capability_with_project("cap1")

        assert result == ("project1", cap)

    def test_get_capability_with_project_specified(self):
        """Test getting capability with project specified."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        result = registry.get_capability_with_project("cap1", "project2")
        assert result == ("project2", cap2)

    def test_get_capability_with_project_nonexistent(self):
        """Test getting nonexistent capability returns None."""
        registry = CapabilityRegistry()

        result = registry.get_capability_with_project("nonexistent")
        assert result is None

    def test_get_capability_with_project_ambiguous_raises_error(self):
        """Test ambiguous capability raises error."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        with pytest.raises(AmbiguousCapabilityError):
            registry.get_capability_with_project("cap1")


class TestFindCapabilitiesByTag:
    """Test suite for finding capabilities by tag."""

    def test_find_capabilities_by_tag(self):
        """Test finding capabilities by tag."""
        registry = CapabilityRegistry()
        cap = Mock()
        metadata = Mock(tags=["tag1", "tag2"])

        registry.register_project_capabilities("project1", {"cap1": cap}, {"cap1": metadata})
        results = registry.find_capabilities_by_tag("tag1")

        assert len(results) == 1
        assert results[0] == ("project1", "cap1", cap)

    def test_find_capabilities_by_tag_multiple_results(self):
        """Test finding multiple capabilities with same tag."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()
        metadata1 = Mock(tags=["tag1"])
        metadata2 = Mock(tags=["tag1"])

        registry.register_project_capabilities("project1", {"cap1": cap1}, {"cap1": metadata1})
        registry.register_project_capabilities("project2", {"cap2": cap2}, {"cap2": metadata2})

        results = registry.find_capabilities_by_tag("tag1")
        assert len(results) == 2

    def test_find_capabilities_by_tag_no_results(self):
        """Test finding capabilities with nonexistent tag."""
        registry = CapabilityRegistry()
        cap = Mock()
        metadata = Mock(tags=["tag1"])

        registry.register_project_capabilities("project1", {"cap1": cap}, {"cap1": metadata})
        results = registry.find_capabilities_by_tag("nonexistent")

        assert results == []


class TestGetAllCapabilities:
    """Test suite for getting all capabilities."""

    def test_get_all_capabilities(self):
        """Test getting all capabilities."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1, "cap2": cap2})
        result = registry.get_all_capabilities()

        assert "cap1" in result
        assert "cap2" in result
        assert result["cap1"] == [("project1", cap1)]

    def test_get_all_capabilities_multiple_projects(self):
        """Test getting all capabilities from multiple projects."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        result = registry.get_all_capabilities()
        assert len(result["cap1"]) == 2

    def test_get_all_capabilities_empty(self):
        """Test getting all capabilities when empty."""
        registry = CapabilityRegistry()

        result = registry.get_all_capabilities()
        assert result == {}


class TestGetCapabilitiesByProject:
    """Test suite for getting capabilities by project."""

    def test_get_capabilities_by_project(self):
        """Test getting capabilities for a project."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1, "cap2": cap2})
        result = registry.get_capabilities_by_project("project1")

        assert len(result) == 2
        assert result["cap1"] is cap1
        assert result["cap2"] is cap2

    def test_get_capabilities_by_project_filters_correctly(self):
        """Test getting capabilities filters by project."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap2": cap2})

        result = registry.get_capabilities_by_project("project1")
        assert "cap1" in result
        assert "cap2" not in result

    def test_get_capabilities_by_nonexistent_project(self):
        """Test getting capabilities for nonexistent project."""
        registry = CapabilityRegistry()

        result = registry.get_capabilities_by_project("nonexistent")
        assert result == {}


class TestGetCapabilityDescription:
    """Test suite for getting capability descriptions."""

    def test_get_capability_description_simple(self):
        """Test getting description for simple capability."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        description = registry.get_capability_description("cap1")

        assert "cap1" in description

    def test_get_capability_description_with_metadata(self):
        """Test getting description with metadata."""
        registry = CapabilityRegistry()
        cap = Mock()
        metadata = Mock(description="Test description")

        registry.register_project_capabilities("project1", {"cap1": cap}, {"cap1": metadata})
        description = registry.get_capability_description("cap1")

        assert "Test description" in description

    def test_get_capability_description_with_project(self):
        """Test getting description with project specified."""
        registry = CapabilityRegistry()
        cap = Mock()
        metadata = Mock(description="Test description")

        registry.register_project_capabilities("project1", {"cap1": cap}, {"cap1": metadata})
        description = registry.get_capability_description("cap1", "project1")

        assert "project1" in description
        assert "Test description" in description

    def test_get_capability_description_nonexistent(self):
        """Test getting description for nonexistent capability."""
        registry = CapabilityRegistry()

        description = registry.get_capability_description("nonexistent")
        assert "Unknown capability" in description

    def test_get_capability_description_multiple_projects(self):
        """Test getting description for capability in multiple projects."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        description = registry.get_capability_description("cap1")
        assert "multiple projects" in description


class TestHasCapability:
    """Test suite for checking capability existence."""

    def test_has_capability_exists(self):
        """Test has_capability returns True for existing capability."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        assert registry.has_capability("cap1") is True

    def test_has_capability_not_exists(self):
        """Test has_capability returns False for nonexistent capability."""
        registry = CapabilityRegistry()

        assert registry.has_capability("nonexistent") is False

    def test_has_capability_with_project(self):
        """Test has_capability with project specification."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        assert registry.has_capability("cap1", "project1") is True
        assert registry.has_capability("cap1", "project2") is False


class TestGetCounts:
    """Test suite for getting counts."""

    def test_get_capability_count(self):
        """Test getting capability count."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1, "cap2": cap2})
        assert registry.get_capability_count() == 2

    def test_get_project_count(self):
        """Test getting project count."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        registry.register_project_capabilities("project2", {"cap2": cap})

        assert registry.get_project_count() == 2

    def test_get_capability_projects(self):
        """Test getting projects for a capability."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        projects = registry.get_capability_projects("cap1")
        assert set(projects) == {"project1", "project2"}

    def test_get_capability_projects_nonexistent(self):
        """Test getting projects for nonexistent capability."""
        registry = CapabilityRegistry()

        projects = registry.get_capability_projects("nonexistent")
        assert projects == []


class TestClearAndUnregister:
    """Test suite for clearing and unregistering."""

    def test_clear_removes_all(self):
        """Test clear removes all capabilities."""
        registry = CapabilityRegistry()
        cap = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap})
        registry.clear()

        assert registry._capabilities == {}
        assert registry._metadata == {}

    def test_unregister_project(self):
        """Test unregistering a project."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap2": cap2})

        result = registry.unregister_project("project1")

        assert result is True
        assert not registry.has_capability("cap1")
        assert registry.has_capability("cap2")

    def test_unregister_nonexistent_project(self):
        """Test unregistering nonexistent project."""
        registry = CapabilityRegistry()

        result = registry.unregister_project("nonexistent")
        assert result is False

    def test_unregister_project_with_shared_capability(self):
        """Test unregistering project with shared capability name."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1})
        registry.register_project_capabilities("project2", {"cap1": cap2})

        registry.unregister_project("project1")

        # cap1 should still exist for project2
        assert registry.has_capability("cap1", "project2")
        assert not registry.has_capability("cap1", "project1")


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_multiple_registries_independent(self):
        """Test multiple registry instances are independent."""
        registry1 = CapabilityRegistry()
        registry2 = CapabilityRegistry()

        cap = Mock()
        registry1.register_project_capabilities("project1", {"cap1": cap})

        assert registry1.has_capability("cap1")
        assert not registry2.has_capability("cap1")

    def test_register_empty_capabilities(self):
        """Test registering empty capabilities dict."""
        registry = CapabilityRegistry()

        registry.register_project_capabilities("project1", {})
        assert registry.get_project_count() == 1
        assert registry.get_capability_count() == 0

    def test_get_all_capability_descriptions(self):
        """Test getting all capability descriptions."""
        registry = CapabilityRegistry()
        cap1 = Mock()
        cap2 = Mock()

        registry.register_project_capabilities("project1", {"cap1": cap1, "cap2": cap2})
        descriptions = registry.get_all_capability_descriptions()

        assert "cap1" in descriptions
        assert "cap2" in descriptions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
