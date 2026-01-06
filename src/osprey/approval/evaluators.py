"""Approval Evaluators for Capability-Specific Business Logic.

This module contains evaluator classes that implement capability-specific
business logic for determining approval requirements. Each evaluator encapsulates
the rules and decision-making process for its respective capability, providing
clear separation between configuration and business logic.

The evaluator pattern allows capabilities to define their own approval rules
while maintaining consistent interfaces and behavior. This design supports
complex approval scenarios with multiple factors and conditions.

Key Features:
    - Capability-specific approval decision logic
    - Structured decision results with reasoning
    - Support for multiple approval modes and conditions
    - Extensible architecture for new capability types
    - Comprehensive logging for audit trails

Evaluator Classes:
    - PythonExecutionApprovalEvaluator: Python code execution approval logic
    - MemoryApprovalEvaluator: Memory operation approval logic

Examples:
    Python execution evaluation::

        >>> from .config_models import PythonExecutionApprovalConfig, ApprovalMode
        >>> config = PythonExecutionApprovalConfig(enabled=True, mode=ApprovalMode.EPICS_WRITES)
        >>> evaluator = PythonExecutionApprovalEvaluator(config)
        >>> decision = evaluator.evaluate(has_epics_writes=True, has_epics_reads=False)
        >>> print(f"Needs approval: {decision.needs_approval}")
        >>> print(f"Reasoning: {decision.reasoning}")

    Memory operation evaluation::

        >>> from .config_models import MemoryApprovalConfig
        >>> config = MemoryApprovalConfig(enabled=True)
        >>> evaluator = MemoryApprovalEvaluator(config)
        >>> decision = evaluator.evaluate(operation_type="save")
        >>> print(f"Approval required: {decision.needs_approval}")

.. note::
   Evaluators are stateless and thread-safe. They can be created once and
   reused for multiple evaluations with the same configuration.
"""

import logging
from typing import NamedTuple

from .config_models import ApprovalMode, MemoryApprovalConfig, PythonExecutionApprovalConfig

logger = logging.getLogger(__name__)


class ApprovalDecision(NamedTuple):
    """Structured result of an approval evaluation decision.

    Represents the outcome of an approval evaluation with both the decision
    and the reasoning behind it. This structured approach ensures consistent
    decision reporting across all evaluators and provides clear audit trails
    for approval decisions.

    :param needs_approval: Whether human approval is required for the operation
    :type needs_approval: bool
    :param reasoning: Human-readable explanation of the decision logic
    :type reasoning: str

    Examples:
        Approval required decision::

            >>> decision = ApprovalDecision(
            ...     needs_approval=True,
            ...     reasoning="Code contains EPICS write operations"
            ... )
            >>> print(f"Decision: {decision.needs_approval}")
            >>> print(f"Reason: {decision.reasoning}")

        No approval needed::

            >>> decision = ApprovalDecision(
            ...     needs_approval=False,
            ...     reasoning="Python execution approval is disabled"
            ... )

    .. note::
       The reasoning field is crucial for logging, debugging, and providing
       clear feedback to users about why approval was or wasn't required.

    .. seealso::
       :class:`PythonExecutionApprovalEvaluator` : Evaluator class that returns this decision
       :class:`MemoryApprovalEvaluator` : Evaluator class that returns this decision
       :func:`osprey.approval.create_code_approval_interrupt` : Uses reasoning for user messages
    """

    needs_approval: bool
    reasoning: str


class PythonExecutionApprovalEvaluator:
    """Business logic evaluator for Python code execution approval decisions.

    Implements capability-specific rules for determining when Python code
    execution requires human approval. The evaluator supports multiple approval
    modes ranging from disabled (no approval) to all_code (approval for everything).

    The evaluation logic considers both the configured approval mode and the
    specific characteristics of the code being evaluated, such as EPICS operations.
    This provides granular control over approval requirements based on operational
    risk assessment.

    Supported Approval Modes:
        - DISABLED: No approval required regardless of code content
        - EPICS_WRITES: Approval required only for code with EPICS write operations
        - ALL_CODE: Approval required for all Python code execution

    :param config: Configuration object containing approval settings
    :type config: PythonExecutionApprovalConfig

    Examples:
        Create evaluator with EPICS writes mode::

            >>> config = PythonExecutionApprovalConfig(
            ...     enabled=True,
            ...     mode=ApprovalMode.EPICS_WRITES
            ... )
            >>> evaluator = PythonExecutionApprovalEvaluator(config)

        Evaluate code with EPICS writes::

            >>> decision = evaluator.evaluate(
            ...     has_epics_writes=True,
            ...     has_epics_reads=False
            ... )
            >>> print(f"Approval needed: {decision.needs_approval}")
            >>> print(f"Reason: {decision.reasoning}")

    .. note::
       The evaluator is stateless and can be reused for multiple evaluations
       with the same configuration settings.

    .. seealso::
       :class:`PythonExecutionApprovalConfig` : Configuration model used by this evaluator
       :class:`ApprovalDecision` : Decision model returned by evaluation methods
       :class:`ApprovalManager` : Manager that creates instances of this evaluator
       :meth:`evaluate` : Main evaluation method of this class
    """

    def __init__(self, config: PythonExecutionApprovalConfig):
        """Initialize evaluator with Python execution approval configuration.

        :param config: Configuration object containing approval mode and settings
        :type config: PythonExecutionApprovalConfig
        """
        self.config = config

    def evaluate(self, has_epics_writes: bool, has_epics_reads: bool) -> ApprovalDecision:
        """Evaluate whether Python code execution requires human approval.

        Applies configured approval rules to determine if the given code
        characteristics require human approval before execution. The evaluation
        considers both global settings and code-specific risk factors.

        The evaluation logic follows this hierarchy:
        1. Check if approval is globally disabled
        2. Apply mode-specific rules (disabled, epics_writes, all_code)
        3. Fall back to secure default (approval required) for unknown modes

        :param has_epics_writes: Whether code contains EPICS write operations
        :type has_epics_writes: bool
        :param has_epics_reads: Whether code contains EPICS read operations
        :type has_epics_reads: bool
        :return: Decision object with approval requirement and reasoning
        :rtype: ApprovalDecision

        Examples:
            Evaluate read-only EPICS code::

                >>> decision = evaluator.evaluate(
                ...     has_epics_writes=False,
                ...     has_epics_reads=True
                ... )
                >>> # Result depends on configured mode

            Evaluate code with EPICS writes::

                >>> decision = evaluator.evaluate(
                ...     has_epics_writes=True,
                ...     has_epics_reads=True
                ... )
                >>> # Will require approval in EPICS_WRITES or ALL_CODE modes

            Evaluate pure Python code::

                >>> decision = evaluator.evaluate(
                ...     has_epics_writes=False,
                ...     has_epics_reads=False
                ... )
                >>> # Requires approval only in ALL_CODE mode

        .. note::
           Unknown approval modes default to requiring approval for security.

        .. seealso::
           :class:`ApprovalDecision` : Decision structure returned by this method
           :class:`ApprovalMode` : Enum values processed by this evaluation logic
           :class:`PythonExecutionApprovalConfig` : Configuration that controls evaluation
           :func:`osprey.approval.create_code_approval_interrupt` : Uses evaluation results
        """
        # If approval is disabled globally for Python execution
        if not self.config.enabled:
            return ApprovalDecision(
                needs_approval=False, reasoning="Python execution approval is disabled"
            )

        # Apply mode-specific logic
        if self.config.mode == ApprovalMode.DISABLED:
            return ApprovalDecision(needs_approval=False, reasoning="Approval mode is disabled")

        elif self.config.mode in (ApprovalMode.CONTROL_WRITES, ApprovalMode.EPICS_WRITES):
            # Support both new and deprecated modes
            if has_epics_writes:
                return ApprovalDecision(
                    needs_approval=True, reasoning="Code contains control system write operations"
                )
            else:
                return ApprovalDecision(
                    needs_approval=False,
                    reasoning="Code contains no control system writes (read-only or pure Python)",
                )

        elif self.config.mode == ApprovalMode.ALL_CODE:
            return ApprovalDecision(needs_approval=True, reasoning="All code requires approval")

        else:
            # Fail-safe to approval required for unknown modes to maintain security
            logger.warning(
                f"Unknown approval mode: {self.config.mode}, defaulting to approval required"
            )
            return ApprovalDecision(
                needs_approval=True, reasoning=f"Unknown approval mode: {self.config.mode}"
            )


class MemoryApprovalEvaluator:
    """Business logic evaluator for memory operation approval decisions.

    Implements approval rules for memory operations including creating, updating,
    and deleting stored memories. Currently supports simple enabled/disabled
    logic but is designed for extension with more sophisticated rules based on
    content sensitivity, user permissions, or operation types.

    The evaluator provides a foundation for future enhancements such as:
    - Content-based approval (sensitive data detection)
    - User-specific approval requirements
    - Operation-type granularity (create vs update vs delete)
    - Memory size or complexity thresholds

    :param config: Configuration object containing memory approval settings
    :type config: MemoryApprovalConfig

    Examples:
        Create evaluator with approval enabled::

            >>> config = MemoryApprovalConfig(enabled=True)
            >>> evaluator = MemoryApprovalEvaluator(config)

        Evaluate memory operation::

            >>> decision = evaluator.evaluate(operation_type="create")
            >>> print(f"Approval required: {decision.needs_approval}")
            >>> print(f"Reasoning: {decision.reasoning}")

    .. note::
       This evaluator is designed for future extensibility. Additional
       evaluation parameters can be added without breaking existing usage.
    """

    def __init__(self, config: MemoryApprovalConfig):
        """Initialize evaluator with memory approval configuration.

        :param config: Configuration object containing memory approval settings
        :type config: MemoryApprovalConfig
        """
        self.config = config

    def evaluate(self, operation_type: str = "general") -> ApprovalDecision:
        """Evaluate whether memory operation requires human approval.

        Determines approval requirements for memory operations based on
        configuration settings. Currently implements simple enabled/disabled
        logic but the interface supports future extensions for operation-specific
        or content-based approval rules.

        :param operation_type: Type of memory operation for future rule extensions
        :type operation_type: str
        :return: Decision object with approval requirement and reasoning
        :rtype: ApprovalDecision

        Examples:
            Evaluate general memory operation::

                >>> decision = evaluator.evaluate()
                >>> print(f"Needs approval: {decision.needs_approval}")

            Evaluate specific operation type::

                >>> decision = evaluator.evaluate(operation_type="delete")
                >>> # Currently same logic, but ready for future extensions

        .. note::
           The operation_type parameter is reserved for future functionality
           where different operation types may have different approval rules.
        """
        if self.config.enabled:
            return ApprovalDecision(
                needs_approval=True, reasoning="Memory operations require approval"
            )
        else:
            return ApprovalDecision(
                needs_approval=False, reasoning="Memory operation approval is disabled"
            )
