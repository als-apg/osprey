"""Tests for PythonCapability instance method pattern migration and prompt builder integration."""

import inspect

import pytest

from osprey.capabilities.python import PythonCapability, _create_python_capability_prompts


class TestPythonCapabilityMigration:
    """Test PythonCapability successfully migrated to instance method pattern."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(PythonCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(PythonCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_state_can_be_injected(self, mock_state, mock_step):
        """Verify capability instance can receive _state and _step injection."""
        capability = PythonCapability()
        capability._state = mock_state
        capability._step = mock_step

        assert capability._state == mock_state
        assert capability._step == mock_step

    def test_has_langgraph_node_decorator(self):
        """Verify @capability_node decorator created langgraph_node attribute."""
        assert hasattr(PythonCapability, "langgraph_node")
        assert callable(PythonCapability.langgraph_node)


class TestPythonPromptBuilderIntegration:
    """Test Python capability integration with prompt builder system."""

    def test_capability_prompts_include_builder_instructions(self):
        """Verify _create_python_capability_prompts uses prompt builder system."""
        from osprey.prompts.defaults import DefaultPromptProvider
        from osprey.prompts.loader import register_framework_prompt_provider, _prompt_loader

        # Clear any existing providers
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None

        # Register default provider
        provider = DefaultPromptProvider()
        register_framework_prompt_provider('test', provider)

        # Create prompts
        prompts = _create_python_capability_prompts(
            task_objective="Test task",
            user_query="Test query",
            context_description="Test context"
        )

        # Should have 4 prompts: task, user query, context, AND builder instructions
        assert len(prompts) >= 4, f"Expected at least 4 prompts, got {len(prompts)}"

        # Verify builder instructions are included
        combined = '\n'.join(prompts)
        assert 'CODE GENERATION INSTRUCTIONS' in combined, (
            "Prompt builder instructions not included in capability_prompts!"
        )
        assert 'results' in combined.lower(), (
            "Critical 'results' requirement from prompt builder not included!"
        )

    def test_prompt_builder_instructions_content(self):
        """Verify prompt builder provides proper Python code generation guidance."""
        from osprey.prompts.defaults.python import DefaultPythonPromptBuilder

        builder = DefaultPythonPromptBuilder()
        instructions = builder.get_instructions()

        # Verify critical elements
        assert 'results' in instructions.lower(), "Missing 'results' dictionary requirement"
        assert 'executable' in instructions.lower(), "Missing executable code requirement"
        assert 'import' in instructions.lower(), "Missing imports guidance"

        # Should be substantial (not just a placeholder)
        assert len(instructions) > 200, f"Instructions too short: {len(instructions)} chars"

    def test_fallback_when_prompt_builder_unavailable(self):
        """Verify graceful fallback if prompt builder system fails."""
        from osprey.prompts.loader import _prompt_loader

        # Clear providers to simulate failure
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None

        # Should not crash, should gracefully continue
        prompts = _create_python_capability_prompts(
            task_objective="Test task",
            user_query="Test query"
        )

        # Should still have basic prompts (task, user query)
        assert len(prompts) >= 2, "Should have at least task and user query"

        # Verify task and query are present
        combined = '\n'.join(prompts)
        assert 'Test task' in combined
        assert 'Test query' in combined

    @pytest.mark.skip(reason="Requires full registry initialization - tested in integration tests")
    def test_orchestrator_guide_from_prompt_builder(self):
        """Verify orchestrator guide comes from prompt builder.

        Note: This test requires registry initialization and is better tested
        in integration tests. The key functionality (prompt builder instructions
        injection into capability_prompts) is tested above.
        """
        pass

    def test_classifier_guide_uses_prompt_builder(self):
        """Verify _create_classifier_guide delegates to prompt builder."""
        from osprey.prompts.defaults import DefaultPromptProvider
        from osprey.prompts.loader import register_framework_prompt_provider, _prompt_loader

        # Clear and register
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None
        provider = DefaultPromptProvider()
        register_framework_prompt_provider('test', provider)

        # Create capability instance
        capability = PythonCapability()

        # Get classifier guide
        guide = capability._create_classifier_guide()

        # Should return the guide from prompt builder
        assert guide is not None
        assert len(guide.examples) > 0
        assert "computational" in guide.instructions.lower()
