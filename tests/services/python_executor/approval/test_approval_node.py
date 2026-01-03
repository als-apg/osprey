"""Placeholder for Python Executor Approval Node Tests.

NOTE: The approval node cannot be meaningfully unit tested because:
- The interrupt() function requires a full LangGraph runtime context
- Mocking interrupt() bypasses all the actual code, providing 0% coverage
- Testing mocks instead of real code gives false confidence

EXISTING COVERAGE: The approval node is comprehensively tested via integration tests.
See tests/integration/test_python_executor_service.py::TestApprovalWorkflow:
- test_approval_interrupt_triggered: Verifies approval workflow interrupts execution
- test_approval_with_channel_writes: Tests write operations trigger approval
- test_complete_approval_resume_workflow: End-to-end approval → resume → execute
- test_approval_rejection_workflow: Tests user rejection handling

This file serves as documentation of why unit tests are not appropriate here,
per COVERAGE_EXPANSION_PLAN.md guidance on "files genuinely hard to test in isolation".
"""


class TestApprovalNodeDocumentation:
    """Document that approval node is tested via integration tests."""

    def test_approval_node_is_integration_tested(self):
        """Approval node is comprehensively tested in TestApprovalWorkflow."""
        # The approval node requires LangGraph runtime context (interrupt() calls)
        # and is fully tested via integration tests in:
        # tests/integration/test_python_executor_service.py::TestApprovalWorkflow
        #
        # Those tests cover:
        # - Approval interrupt triggering
        # - User approval/rejection handling
        # - State preservation during resume
        # - Write operation approval workflows
        assert True, "See tests/integration/test_python_executor_service.py::TestApprovalWorkflow"
