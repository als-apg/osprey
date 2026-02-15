"""Tests for classification node - task classification and capability selection."""

import inspect
from unittest.mock import MagicMock, patch

from osprey.infrastructure.classification_node import (
    ClassificationNode,
    _create_classification_result,
    _detect_reclassification_scenario,
    _expand_capability_dependencies,
)

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
        assert (
            "retry" in classification.user_message.lower()
            or "timeout" in classification.user_message.lower()
        )

    def test_classify_connection_error(self):
        """Test connection errors are classified as retriable."""
        exc = ConnectionError("Network error")
        context = {"operation": "llm_call"}

        classification = ClassificationNode.classify_error(exc, context)

        assert classification.severity.value == "retriable"
        assert (
            "retry" in classification.user_message.lower()
            or "network" in classification.user_message.lower()
        )

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

    def test_classify_reclassification_required_error(self):
        """Test ReclassificationRequiredError is handled correctly."""
        from osprey.base.errors import ReclassificationRequiredError

        exc = ReclassificationRequiredError("Need new capabilities")
        context = {"operation": "classification"}

        classification = ClassificationNode.classify_error(exc, context)

        assert classification.severity.value == "reclassification"
        assert "reclassification" in classification.user_message.lower()


# =============================================================================
# Test Retry Policy
# =============================================================================


class TestRetryPolicy:
    """Test custom retry policy for classification."""

    def test_get_retry_policy(self):
        """Test retry policy returns correct values."""
        policy = ClassificationNode.get_retry_policy()

        assert "max_attempts" in policy
        assert "delay_seconds" in policy
        assert "backoff_factor" in policy
        assert policy["max_attempts"] >= 3
        assert policy["delay_seconds"] > 0
        assert policy["backoff_factor"] > 1


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions used by classification node."""

    def test_create_classification_result_normal(self):
        """Test creating classification result in normal mode."""
        active_caps = ["python", "respond"]
        state = {"control_reclassification_count": 0}
        message = "Test message"

        result = _create_classification_result(
            active_capabilities=active_caps,
            state=state,
            message=message,
            is_bypass=False,
            previous_failure=None,
        )

        assert result["planning_active_capabilities"] == active_caps
        assert result["planning_execution_plan"] is None
        assert result["planning_current_step_index"] == 0
        assert result["control_reclassification_count"] == 0
        assert result["control_reclassification_reason"] is None

    def test_create_classification_result_bypass_mode(self):
        """Test creating classification result in bypass mode."""
        active_caps = ["python", "respond", "search"]
        state = {"control_reclassification_count": 0}
        message = "Bypass mode active"

        result = _create_classification_result(
            active_capabilities=active_caps,
            state=state,
            message=message,
            is_bypass=True,
            previous_failure=None,
        )

        assert result["planning_active_capabilities"] == active_caps
        assert len(result["planning_active_capabilities"]) == 3

    def test_create_classification_result_with_reclassification(self):
        """Test creating classification result during reclassification."""
        active_caps = ["python"]
        state = {"control_reclassification_count": 2}
        message = "Reclassification complete"
        previous_failure = "Orchestrator failed"

        result = _create_classification_result(
            active_capabilities=active_caps,
            state=state,
            message=message,
            is_bypass=False,
            previous_failure=previous_failure,
        )

        # Should increment reclassification count
        assert result["control_reclassification_count"] == 3
        # Should clear error state
        assert result["control_has_error"] is False
        assert result["control_error_info"] is None

    def test_detect_reclassification_scenario_no_error(self):
        """Test detecting reclassification when no error exists."""
        state = {"control_has_error": False}

        result = _detect_reclassification_scenario(state)

        assert result is None

    def test_detect_reclassification_scenario_with_reclassification(self):
        """Test detecting reclassification with proper error state."""
        from osprey.base.errors import ErrorClassification, ErrorSeverity

        error_classification = ErrorClassification(
            severity=ErrorSeverity.RECLASSIFICATION,
            user_message="Need reclassification",
            metadata={},
        )

        state = {
            "control_has_error": True,
            "control_error_info": {
                "classification": error_classification,
                "capability_name": "orchestrator",
            },
        }

        result = _detect_reclassification_scenario(state)

        assert result is not None
        assert "orchestrator" in result
        assert "reclassification" in result.lower()

    def test_detect_reclassification_scenario_wrong_severity(self):
        """Test detecting reclassification with non-reclassification error."""
        from osprey.base.errors import ErrorClassification, ErrorSeverity

        error_classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL, user_message="Critical error", metadata={}
        )

        state = {
            "control_has_error": True,
            "control_error_info": {
                "classification": error_classification,
                "capability_name": "python",
            },
        }

        result = _detect_reclassification_scenario(state)

        assert result is None

    def test_detect_reclassification_scenario_missing_classification(self):
        """Test detecting reclassification with missing classification."""
        state = {"control_has_error": True, "control_error_info": {}}

        result = _detect_reclassification_scenario(state)

        assert result is None


# =============================================================================
# Test Dependency Expansion
# =============================================================================


def _make_mock_cap(name, provides=None, requires=None):
    """Create a mock capability with name, provides, and requires."""
    cap = MagicMock()
    cap.name = name
    cap.provides = provides or []
    cap.requires = requires or []
    return cap


def _mock_registry_for_expansion(capabilities):
    """Build a mock registry from a list of mock capabilities."""
    registry = MagicMock()
    cap_map = {cap.name: cap for cap in capabilities}
    registry.get_capability.side_effect = lambda n: cap_map.get(n)
    registry.get_all_capabilities.return_value = capabilities
    return registry


class TestExpandCapabilityDependencies:
    """Test _expand_capability_dependencies transitive dependency resolution."""

    def test_single_dependency_resolved(self):
        """channel_write requires CHANNEL_ADDRESSES → adds channel_finding."""
        caps = [
            _make_mock_cap("channel_write", requires=["CHANNEL_ADDRESSES"]),
            _make_mock_cap("channel_finding", provides=["CHANNEL_ADDRESSES"]),
            _make_mock_cap("respond"),
        ]
        registry = _mock_registry_for_expansion(caps)
        state = {"capability_context_data": {}}
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(["channel_write", "respond"], state, logger)

        assert "channel_finding" in result
        assert "channel_write" in result
        assert "respond" in result

    def test_dependency_satisfied_by_context_data(self):
        """No expansion when required context already exists from a previous turn."""
        caps = [
            _make_mock_cap("channel_write", requires=["CHANNEL_ADDRESSES"]),
            _make_mock_cap("channel_finding", provides=["CHANNEL_ADDRESSES"]),
        ]
        registry = _mock_registry_for_expansion(caps)
        state = {
            "capability_context_data": {"CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}}}
        }
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(["channel_write"], state, logger)

        assert result == ["channel_write"]

    def test_transitive_dependencies(self):
        """C requires B requires A → all three added."""
        caps = [
            _make_mock_cap("cap_c", requires=["TYPE_B"]),
            _make_mock_cap("cap_b", provides=["TYPE_B"], requires=["TYPE_A"]),
            _make_mock_cap("cap_a", provides=["TYPE_A"]),
        ]
        registry = _mock_registry_for_expansion(caps)
        state = {"capability_context_data": {}}
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(["cap_c"], state, logger)

        assert set(result) == {"cap_c", "cap_b", "cap_a"}

    def test_no_requires_no_expansion(self):
        """Capability with no requires → list unchanged."""
        caps = [_make_mock_cap("respond")]
        registry = _mock_registry_for_expansion(caps)
        state = {"capability_context_data": {}}
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(["respond"], state, logger)

        assert result == ["respond"]

    def test_duplicate_prevention(self):
        """Provider already in selected list → no duplicate added."""
        caps = [
            _make_mock_cap("channel_write", requires=["CHANNEL_ADDRESSES"]),
            _make_mock_cap("channel_finding", provides=["CHANNEL_ADDRESSES"]),
        ]
        registry = _mock_registry_for_expansion(caps)
        state = {"capability_context_data": {}}
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(
                ["channel_write", "channel_finding"], state, logger
            )

        assert result == ["channel_write", "channel_finding"]

    def test_unknown_provider_skipped(self):
        """Required context type with no registered provider → no crash."""
        caps = [
            _make_mock_cap("channel_write", requires=["NONEXISTENT_TYPE"]),
        ]
        registry = _mock_registry_for_expansion(caps)
        state = {"capability_context_data": {}}
        logger = MagicMock()

        with patch("osprey.infrastructure.classification_node.get_registry", return_value=registry):
            result = _expand_capability_dependencies(["channel_write"], state, logger)

        assert result == ["channel_write"]
