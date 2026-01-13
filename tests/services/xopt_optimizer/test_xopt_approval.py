"""Unit Tests for XOpt Approval Interrupt Function.

This module tests the create_xopt_approval_interrupt function.
"""

from osprey.approval import create_xopt_approval_interrupt


class TestCreateXOptApprovalInterrupt:
    """Test create_xopt_approval_interrupt function."""

    def test_basic_interrupt_creation(self):
        """Should create interrupt data with required fields."""
        result = create_xopt_approval_interrupt(
            yaml_config="xopt:\n  generator: random",
            strategy="exploration",
            objective="Maximize efficiency",
        )

        assert "user_message" in result
        assert "resume_payload" in result

        # Check user message content
        assert "HUMAN APPROVAL REQUIRED" in result["user_message"]
        assert "Maximize efficiency" in result["user_message"]
        assert "EXPLORATION" in result["user_message"]
        assert "xopt:" in result["user_message"]

        # Check resume payload
        payload = result["resume_payload"]
        assert payload["approval_type"] == "xopt_optimizer"
        assert payload["yaml_config"] == "xopt:\n  generator: random"
        assert payload["strategy"] == "exploration"
        assert payload["objective"] == "Maximize efficiency"

    def test_interrupt_with_machine_state_details(self):
        """Should include machine state details when provided."""
        machine_details = {
            "beam_current": 50.0,
            "status": "ready",
        }

        result = create_xopt_approval_interrupt(
            yaml_config="test: yaml",
            strategy="optimization",
            objective="Test objective",
            machine_state_details=machine_details,
        )

        # Machine state should appear in message
        assert "Machine State Assessment" in result["user_message"]
        assert "beam_current" in result["user_message"]

        # Should be in payload
        assert result["resume_payload"]["machine_state_details"] == machine_details

    def test_interrupt_with_custom_step_objective(self):
        """Should use custom step objective."""
        result = create_xopt_approval_interrupt(
            yaml_config="test: yaml",
            strategy="exploration",
            objective="Test",
            step_objective="Custom optimization task",
        )

        assert "Custom optimization task" in result["user_message"]
        assert result["resume_payload"]["step_objective"] == "Custom optimization task"

    def test_interrupt_contains_approval_instructions(self):
        """Should contain clear approval instructions."""
        result = create_xopt_approval_interrupt(
            yaml_config="test: yaml",
            strategy="exploration",
            objective="Test",
        )

        message = result["user_message"]
        assert "yes" in message.lower()
        assert "no" in message.lower()
        assert "approve" in message.lower()

    def test_interrupt_yaml_displayed_correctly(self):
        """Should display YAML in code block."""
        yaml_config = """xopt:
  generator:
    name: bayesian
  vocs:
    variables:
      x1: [0, 10]
"""
        result = create_xopt_approval_interrupt(
            yaml_config=yaml_config,
            strategy="optimization",
            objective="Test",
        )

        # YAML should be in code block
        assert "```yaml" in result["user_message"]
        assert yaml_config in result["user_message"]
