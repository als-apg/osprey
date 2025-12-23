"""Tests for classification node - task classification and capability selection."""

import inspect

import pytest

from osprey.infrastructure.classification_node import ClassificationNode


# =============================================================================
# Test ClassificationNode Class
# =============================================================================


class TestClassificationNode:
    """Test ClassificationNode infrastructure node."""

    def test_node_exists_and_is_callable(self):
        """Verify ClassificationNode can be instantiated."""
        node = ClassificationNode()
        assert node is not None
        assert hasattr(node, "execute")

    def test_execute_is_instance_method(self):
        """Test execute() is an instance method, not static."""
        execute_method = inspect.getattr_static(ClassificationNode, "execute")
        assert not isinstance(execute_method, staticmethod), (
            "ClassificationNode.execute() should be instance method"
        )

    def test_has_langgraph_node_attribute(self):
        """Test that ClassificationNode has langgraph_node from decorator."""
        assert hasattr(ClassificationNode, "langgraph_node")
        assert callable(ClassificationNode.langgraph_node)

    def test_classify_error_method_exists(self):
        """Test that classify_error static method exists."""
        assert hasattr(ClassificationNode, "classify_error")
        assert callable(ClassificationNode.classify_error)

    def test_node_name_and_description(self):
        """Test node has correct name and description."""
        assert ClassificationNode.name == "classifier"
        assert ClassificationNode.description is not None
        assert len(ClassificationNode.description) > 0



# =============================================================================
# Test Error Classification
# =============================================================================


class TestClassificationErrorClassification:
    """Test error classification for classification operations."""

    def test_classify_timeout_error(self):
        """Test timeout errors are classified as retriable."""
        exc = TimeoutError("LLM timeout")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "retriable"
        assert "retry" in classification.user_message.lower() or "timeout" in classification.user_message.lower()

    def test_classify_connection_error(self):
        """Test connection errors are classified as retriable."""
        exc = ConnectionError("Network error")
        context = {"operation": "llm_call"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "retriable"
        assert "retry" in classification.user_message.lower() or "network" in classification.user_message.lower()

    def test_classify_value_error(self):
        """Test ValueError is classified as critical."""
        exc = ValueError("Invalid configuration")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "critical"

    def test_classify_type_error(self):
        """Test TypeError is classified as critical."""
        exc = TypeError("Invalid type")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "critical"

    def test_classify_import_error(self):
        """Test ImportError is classified as critical."""
        exc = ImportError("Missing dependency")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "critical"
        assert "dependencies" in classification.user_message.lower()

    def test_classify_module_not_found_error(self):
        """Test ModuleNotFoundError is classified as critical."""
        exc = ModuleNotFoundError("Module not found")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "critical"

    def test_classify_name_error(self):
        """Test NameError is classified as critical."""
        exc = NameError("Name not defined")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        assert classification.severity.value == "critical"

    def test_classify_generic_exception(self):
        """Test generic exceptions have appropriate classification."""
        exc = Exception("Generic error")
        context = {"operation": "classification"}
        
        classification = ClassificationNode.classify_error(exc, context)
        
        # Should have some classification
        assert classification is not None
        assert classification.severity is not None



