"""
Channel Write Capability

This capability handles writing values to control system channels with
comprehensive safety controls.

Safety Layers (enforced in order):
1. Global writes_enabled flag (control_system.writes_enabled)
   - Master safety switch - all writes blocked if false
   - Checked by capability before calling connector
2. Human approval workflow (approval.capabilities.python_execution)
   - Requires user approval before executing writes (asked FIRST)
   - Handled by capability through approval system
3. Channel limits validation (control_system.limits_checking)
   - Enforces min/max/step/writable constraints from limits database
   - Handled automatically by connector during write_channel()
4. Write verification (control_system.write_verification)
   - Verifies write success via callback or readback
   - Handled automatically by connector (per-channel or global config)

All safety layers must pass for writes to succeed.

Architecture Note:
The capability focuses on high-level orchestration (parsing, approval) while
the connector handles low-level safety (limits, verification). This clean
separation means the capability doesn't need to know about limits databases
or verification configuration - it just calls connector.write_channel() and
the connector handles everything automatically.

The capability uses LLM-based parsing to extract write operations from task
descriptions and available context data (PYTHON_RESULTS, CHANNEL_VALUES, etc.).
This allows it to handle diverse scenarios:
- Direct values: "Set HCM01 to 5.0"
- Calculated values: "Set magnet to calculated optimal value"
- Statistical values: "Set to yesterday's average"

Configuration:
    # Safety Controls (all layers)
    control_system:
      writes_enabled: false  # Master safety switch - must be true for writes

    control_system:
      limits_checking:
        enabled: true
        database_path: "data/channel_limits.json"
        allow_unlisted_channels: false  # Strict mode - reject unlisted channels

    approval:
      capabilities:
        python_execution:
          enabled: true
          mode: "epics_writes"  # Require approval for all control system writes

    # Write verification (final layer, after approval)
    control_system:
      write_verification:
        default_level: callback  # or readback for full verification

    # Connector configuration (development vs production)
    control_system:
      type: mock  # or epics for production

    # Production EPICS configuration
    control_system:
      connector:
        epics:
          timeout: 5.0
          gateways:
            write_access:
              address: cagw.your-facility.edu
              port: 5084

Note: All three safety layers (writes_enabled, limits, approval) apply to
both Python-based writes (via epics.caput) and direct channel writes.
"""

import asyncio
import json
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.connectors.factory import ConnectorFactory
from osprey.context import CapabilityContext
from osprey.utils.config import get_model_config

# Import model completion for LLM-based parsing
try:
    from osprey.models import get_chat_completion
except ImportError:
    get_chat_completion = None


# ========================================================
# Context Classes
# ========================================================


class WriteVerificationInfo(BaseModel):
    """Verification information for a channel write operation."""

    level: str  # "none", "callback", "readback"
    verified: bool  # Whether verification succeeded
    readback_value: float | None = None  # Actual value read back (for "readback" level)
    tolerance_used: float | None = None  # Tolerance used for comparison
    notes: str | None = None  # Additional verification details


class ChannelWriteResult(BaseModel):
    """
    Individual channel write result data with optional verification.

    This is the high-level result model used in capabilities and context classes.
    Provides detailed information about write success and verification status.
    """

    channel_address: str
    value_written: Any
    success: bool  # Write command succeeded
    verification: WriteVerificationInfo | None = None  # Verification details (if performed)
    error_message: str | None = None


class ChannelWriteResultsContext(CapabilityContext):
    """
    Result from channel write operation and context for downstream capabilities.
    Provides detailed results for each write operation including success/failure status.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_WRITE_RESULTS"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

    results: list[ChannelWriteResult]  # List of write results
    total_writes: int
    successful_count: int
    failed_count: int

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Rich description for LLM consumption."""
        # Get first few channel names for preview
        channels_preview = [r.channel_address for r in self.results[:3]]
        example_channel = channels_preview[0] if channels_preview else "MAG:HCM01:CURRENT:SP"
        example_value = self.results[0].value_written if self.results else "50.0"

        return {
            "total_writes": self.total_writes,
            "successful_writes": self.successful_count,
            "failed_writes": self.failed_count,
            "channels": channels_preview,
            "data_structure": "List[ChannelWriteResult] where each result has .channel_address, .value_written, .success, .verification (optional), .error_message fields",
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.results[index].success",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.results[0].channel_address gives '{example_channel}', .value_written gives '{example_value}', .verification.verified shows if write was verified",
            "available_fields": [
                "channel_address",
                "value_written",
                "success",
                "verification",
                "error_message",
            ],
            "verification_fields": [
                "level",
                "verified",
                "readback_value",
                "tolerance_used",
                "notes",
            ],
        }

    def get_summary(self) -> dict[str, Any]:
        """
        FOR HUMAN DISPLAY: Create readable summary for UI/debugging.
        Always customize for better user experience.
        """
        results_summary = []
        for r in self.results:
            result_dict = {
                "channel": r.channel_address,
                "value": r.value_written,
                "success": r.success,
                "error": r.error_message,
            }

            # Add verification info if available
            if r.verification:
                result_dict["verification"] = {
                    "level": r.verification.level,
                    "verified": r.verification.verified,
                    "readback_value": r.verification.readback_value,
                    "notes": r.verification.notes,
                }

            results_summary.append(result_dict)

        return {
            "type": "Channel Write Results",
            "total_writes": self.total_writes,
            "successful": self.successful_count,
            "failed": self.failed_count,
            "results": results_summary,
        }


# ========================================================
# Internal Models for Write Operation Parsing
# ========================================================


class WriteOperation(BaseModel):
    """Internal model for a single write operation."""

    channel_address: str = Field(description="Channel address to write to")
    value: float = Field(description="Numeric value to write")
    units: str | None = Field(default=None, description="Units if specified")
    notes: str | None = Field(default=None, description="Additional context")


class WriteOperationsOutput(BaseModel):
    """Structured output from LLM-based write operation parser."""

    write_operations: list[WriteOperation] = Field(
        description="List of parsed write operations with channel-value pairs. "
        "Empty list if no valid operations could be identified.",
    )


# ========================================================
# Channel Write-Related Errors
# ========================================================


class ChannelWriteError(Exception):
    """Base class for all channel write-related errors."""

    pass


class ChannelNotWritableError(ChannelWriteError):
    """Raised when attempting to write to a read-only channel."""

    pass


class ChannelWriteTimeoutError(ChannelWriteError):
    """Raised when channel write operations time out."""

    pass


class ChannelWriteAccessError(ChannelWriteError):
    """Raised when there are channel write access errors."""

    pass


class ChannelWriteDependencyError(ChannelWriteError):
    """Raised when required dependencies are missing."""

    pass


class WriteParsingError(ChannelWriteError):
    """Raised when write operation parsing fails."""

    pass


class AmbiguousWriteOperationError(ChannelWriteError):
    """Raised when write operations cannot be clearly identified."""

    pass


# ========================================================
# Helper Functions
# ========================================================


def _log_write_outcome(
    result: ChannelWriteResult, operation: WriteOperation, units_str: str, logger
) -> None:
    """
    Log the outcome of a channel write operation.

    Helper function to keep the main execution flow clean.
    Handles logging for success, verification status, and warnings.

    Args:
        result: The write result from the connector
        operation: The original write operation
        units_str: Units string for display (e.g., " A" or "")
        logger: Logger instance to use
    """
    if not result.success:
        # Failed writes are logged by the exception handler
        return

    if result.verification:
        if result.verification.verified:
            logger.success(
                f"✓ Wrote {operation.channel_address} = {operation.value}{units_str} "
                f"[{result.verification.level} verified]"
            )
        else:
            logger.warning(
                f"⚠️  Wrote {operation.channel_address} = {operation.value}{units_str} "
                f"but verification failed: {result.verification.notes}"
            )
    else:
        logger.success(f"✓ Wrote {operation.channel_address} = {operation.value}{units_str}")


# ========================================================
# LLM-Based Write Parsing Helper
# ========================================================


def _get_write_parsing_system_prompt(
    task_objective: str,
    channel_addresses: list[str],
    channel_queries: dict[str, str],
    step_inputs: list[dict[str, str]],
    state: dict[str, Any],
) -> str:
    """Create context-agnostic system prompt for parsing write operations.

    Builds capability-specific runtime context (context summaries, channel mapping)
    and delegates prompt composition to the prompt builder system.

    Args:
        task_objective: The write task to parse
        channel_addresses: List of available channel addresses
        channel_queries: Mapping of channel addresses to their original queries for semantic linking
        step_inputs: Input contexts from the execution plan
        state: Agent state for accessing other context data (PYTHON_RESULTS, etc.)
    """
    from osprey.prompts.loader import get_framework_prompts
    from osprey.registry import get_registry

    # Build context summaries section - CONTEXT TYPE AGNOSTIC
    context_summaries = []
    if step_inputs:
        for input_spec in step_inputs:
            for context_type, context_key in input_spec.items():
                # Skip CHANNEL_ADDRESSES (handled separately)
                if context_type == "CHANNEL_ADDRESSES":
                    continue

                # Get context data
                context_data = (
                    state.get("capability_context_data", {}).get(context_type, {}).get(context_key)
                )

                if context_data:
                    try:
                        registry = get_registry()
                        context_class = registry.get_context_class(context_type)

                        if context_class:
                            # Get summary for LLM
                            context_obj = context_class(**context_data)
                            summary = context_obj.get_summary()

                            context_summaries.append(f"""
{context_type} (from step '{context_key}'):
{json.dumps(summary, indent=2, default=str)}
                            """)
                    except Exception as e:
                        from osprey.utils.logger import get_logger

                        get_logger("channel_write").warning(
                            f"Could not get summary for {context_type}.{context_key}: {e}"
                        )

    available_data = (
        "\n".join(context_summaries)
        if context_summaries
        else "No additional context data available"
    )

    # Build semantic channel mapping section
    channel_mapping_lines = []
    for address in channel_addresses:
        query = channel_queries.get(address, "")
        if query:
            channel_mapping_lines.append(f'  "{query}" → {address}')
        else:
            channel_mapping_lines.append(f"  {address}")

    channel_mapping = "\n".join(channel_mapping_lines)

    # Delegate prompt composition to the builder
    builder = get_framework_prompts().get_channel_write_prompt_builder()
    return builder.build_prompt(
        task_objective=task_objective,
        channel_mapping=channel_mapping,
        available_data=available_data,
    )


# ========================================================
# Capability Implementation
# ========================================================


@capability_node
class ChannelWriteCapability(BaseCapability):
    """Channel write capability for writing values to control system channels."""

    name = "channel_write"
    description = "Write values to control system channels"
    provides = ["CHANNEL_WRITE_RESULTS"]
    requires = ["CHANNEL_ADDRESSES"]

    async def execute(self) -> dict[str, Any]:
        """
        Execute channel write operation with LLM-based value parsing.

        This method:
        1. Parses write operations from task and available contexts using LLM
        2. Executes the parsed operations using the configured connector
        3. Returns structured results with success/failure status

        The parsing is context-agnostic and handles:
        - Direct values from task: "Set X to 5"
        - Calculated values from PYTHON_RESULTS
        - Statistical values from ARCHIVER_DATA
        - Any other context data the orchestrator provides

        Returns:
            State updates containing the CHANNEL_WRITE_RESULTS context with write confirmations.

        Raises:
            ChannelWriteDependencyError: If required CHANNEL_ADDRESSES context is missing.
            AmbiguousWriteOperationError: If write operations cannot be parsed.
            ChannelWriteAccessError: If channel writing fails.
        """

        # Get unified logger with automatic streaming
        logger = self.get_logger()

        # Check global writes_enabled flag FIRST (Safety Layer 1)
        # NOTE: This is operator-level authorization and ALWAYS raises exception if disabled
        # (unlike limits violations which can be configured to skip)
        from osprey.utils.config import get_config_value

        # Try new location first (control_system.writes_enabled)
        writes_enabled = get_config_value("control_system.writes_enabled", None)

        # Fall back to old location for backward compatibility
        if writes_enabled is None:
            writes_enabled = get_config_value("execution_control.epics.writes_enabled", None)
            if writes_enabled is not None:
                logger.warning(
                    "⚠️  DEPRECATED: 'execution_control.epics.writes_enabled' is deprecated."
                )
                logger.warning(
                    "   Please move this setting to 'control_system.writes_enabled' in your config.yml"
                )
            else:
                writes_enabled = False  # Default to disabled for safety

        if not writes_enabled:
            # ALWAYS raise exception - this is operator authorization, not value validation
            error_msg = (
                "Control system writes are disabled in configuration. "
                "Set control_system.writes_enabled: true to enable hardware writes."
            )
            logger.error(error_msg)
            raise ChannelWriteAccessError(error_msg)

        # Extract channel address contexts
        try:
            (channel_contexts,) = self.get_required_contexts()
        except ValueError as e:
            raise ChannelWriteDependencyError(str(e)) from e

        # Parse write operations using LLM (pass full context objects for semantic linking)
        logger.status("Parsing write operations...")
        write_operations = await self._parse_write_operations(channel_contexts)

        # Log what we're about to write
        logger.info(f"Writing {len(write_operations)} channel(s):")
        for op in write_operations:
            units_str = f" {op.units}" if op.units else ""
            logger.info(f"  • {op.channel_address} = {op.value}{units_str}")

        # Check if approval is needed based on configuration (Safety Layer 2 - Ask user FIRST)
        from osprey.approval.approval_manager import get_python_execution_evaluator
        from osprey.approval.approval_system import (
            create_channel_write_approval_interrupt,
            get_approval_resume_data,
        )

        # Check if we're resuming from an approval interrupt
        has_approval_resume, approved_payload = get_approval_resume_data(
            self._state, "channel_write"
        )

        if has_approval_resume:
            # We're resuming after approval
            if not approved_payload:
                # User rejected the approval
                error_msg = "Write operation cancelled by user"
                logger.warning(error_msg)
                raise ChannelWriteAccessError(error_msg)
            # User approved - continue with execution
            logger.info("Resuming approved write operation")
        else:
            # First time execution - check if approval is needed
            # Treat channel writes as EPICS writes for approval purposes
            approval_evaluator = get_python_execution_evaluator()
            approval_decision = approval_evaluator.evaluate(
                has_epics_writes=True,  # Channel writes are control system writes
                has_epics_reads=False,
            )

            if approval_decision.needs_approval:
                logger.info("Write operation requires human approval")

                # Create approval interrupt
                interrupt_data = create_channel_write_approval_interrupt(
                    operations=write_operations,
                    analysis_details={
                        "operation_count": len(write_operations),
                        "channels": [op.channel_address for op in write_operations],
                        "values": [(op.channel_address, op.value) for op in write_operations],
                        "safety_level": "high",
                    },
                    safety_concerns=[
                        f"Direct hardware write: {op.channel_address} = {op.value}"
                        for op in write_operations
                    ],
                    step_objective=self.get_task_objective(),
                )

                # Pause execution for human approval
                from langgraph.types import interrupt

                interrupt(interrupt_data)

                # If we reach here, something went wrong with the interrupt mechanism
                raise RuntimeError("Interrupt mechanism failed - execution should have paused")

        # Create control system connector from configuration
        # NOTE: Connector now handles limits validation (Safety Layer 3) and
        # verification config (Safety Layer 4) automatically
        connector = await ConnectorFactory.create_control_system_connector()

        try:
            # Execute all write operations
            results = []
            total_writes = len(write_operations)

            for i, operation in enumerate(write_operations, 1):
                units_str = f" {operation.units}" if operation.units else ""
                logger.status(f"Writing {i}/{total_writes}: {operation.channel_address}...")

                try:
                    # Write using connector (handles limits validation and verification automatically)
                    connector_result = await connector.write_channel(
                        operation.channel_address, operation.value
                    )

                    # Convert connector result to context model
                    verification_info = None
                    if connector_result.verification:
                        verification_info = WriteVerificationInfo(
                            level=connector_result.verification.level,
                            verified=connector_result.verification.verified,
                            readback_value=connector_result.verification.readback_value,
                            tolerance_used=connector_result.verification.tolerance_used,
                            notes=connector_result.verification.notes,
                        )

                    # Store result
                    result = ChannelWriteResult(
                        channel_address=connector_result.channel_address,
                        value_written=connector_result.value_written,
                        success=connector_result.success,
                        verification=verification_info,
                        error_message=connector_result.error_message,
                    )
                    results.append(result)

                    # Log outcome (delegated to helper)
                    _log_write_outcome(result, operation, units_str, logger)

                except Exception as e:
                    logger.error(f"Failed to write {operation.channel_address}: {e}")
                    results.append(
                        ChannelWriteResult(
                            channel_address=operation.channel_address,
                            value_written=operation.value,
                            success=False,
                            verification=None,
                            error_message=str(e),
                        )
                    )
                    # Continue with other channels rather than failing completely

        finally:
            # Always disconnect connector
            await connector.disconnect()

        # Calculate counts
        successful_count = sum(1 for r in results if r.success)
        failed_count = total_writes - successful_count
        verified_count = sum(
            1 for r in results if r.success and r.verification and r.verification.verified
        )

        # Single summary message with verification info
        if failed_count == 0:
            if verified_count > 0:
                logger.success(
                    f"Successfully wrote {successful_count} channel(s) ({verified_count} verified)"
                )
            else:
                logger.success(f"Successfully wrote {successful_count} channel(s)")
        else:
            logger.warning(
                f"Wrote {successful_count}/{total_writes} channel(s) successfully "
                f"({failed_count} failed, {verified_count} verified)"
            )

        # Create structured result with explicit counts
        result = ChannelWriteResultsContext(
            results=results,
            total_writes=total_writes,
            successful_count=successful_count,
            failed_count=failed_count,
        )

        # Store result in execution context
        state_updates = self.store_output_context(result)

        # Clear approval state after successful approved write to prevent
        # downstream nodes (e.g., reactive_orchestrator) from misinterpreting
        # stale approval flags as their own approval resume.
        if has_approval_resume:
            from osprey.approval import clear_approval_state

            state_updates.update(clear_approval_state())

        return state_updates

    async def _parse_write_operations(self, channel_contexts: list) -> list[WriteOperation]:
        """
        Parse write operations from task and available contexts using LLM.

        This is an internal helper that uses LLM-based structured output to extract
        channel-value pairs from the task description and any available context data.

        Args:
            channel_contexts: List of ChannelAddressesContext objects with channels and original_query

        Returns:
            List of parsed WriteOperation objects

        Raises:
            WriteParsingError: If LLM parsing fails
            AmbiguousWriteOperationError: If no valid operations can be parsed
        """
        logger = self.get_logger()

        # Get current step information using helper methods
        task_objective = self.get_task_objective()
        step_inputs = self.get_step_inputs()

        # Extract channel addresses and build semantic mapping from context objects
        channel_addresses = []
        channel_queries = {}

        for ctx in channel_contexts:
            for address in ctx.channels:
                channel_addresses.append(address)
                channel_queries[address] = ctx.original_query

        logger.debug(f"Channel semantic mapping: {channel_queries}")

        # Build context-agnostic prompt
        full_prompt = _get_write_parsing_system_prompt(
            task_objective=task_objective,
            channel_addresses=channel_addresses,
            channel_queries=channel_queries,
            step_inputs=step_inputs,
            state=self._state,
        )

        logger.debug(f"Prompt length: {len(full_prompt)} characters")

        # LLM call with structured output
        try:
            model_config = get_model_config("channel_write")

            # Set caller context for API call logging (propagates through asyncio.to_thread)
            from osprey.models import set_api_call_context

            set_api_call_context(
                function="_parse_write_operations",
                module="channel_write",
                class_name="ChannelWriteCapability",
                extra={"capability": "channel_write"},
            )

            response_data = await asyncio.to_thread(
                get_chat_completion,
                model_config=model_config,
                message=full_prompt,
                output_model=WriteOperationsOutput,
            )
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            raise WriteParsingError(f"Failed to parse write operations: {str(e)}") from e

        # Validate response
        if not isinstance(response_data, WriteOperationsOutput):
            logger.error(f"Invalid LLM response type: {type(response_data)}")
            raise WriteParsingError("LLM failed to return structured write operations output")

        logger.debug(f"LLM result: {len(response_data.write_operations)} operation(s)")

        # Check if valid operations were found
        if not response_data.write_operations:
            logger.warning(f"No write operations found in: '{task_objective}'")
            raise AmbiguousWriteOperationError(
                f"Could not parse write operations from task: '{task_objective}'. "
                f"Available channels: {channel_addresses}. "
                f"Task may require calculation or additional context data."
            )

        return response_data.write_operations

    def process_extracted_contexts(self, contexts):
        """
        Normalize CHANNEL_ADDRESSES to always be a list of context objects.

        Keeps the full context objects (with original_query) for semantic linking.
        """
        channels_raw = contexts["CHANNEL_ADDRESSES"]

        # Normalize to list of context objects
        if not isinstance(channels_raw, list):
            contexts["CHANNEL_ADDRESSES"] = [channels_raw]

        return contexts

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Channel write-specific error classification."""

        if isinstance(exc, AmbiguousWriteOperationError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=str(exc),
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, WriteParsingError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Write parsing failed, retrying...",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelWriteTimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Channel write timeout: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelWriteAccessError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Channel write access error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelNotWritableError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Channel is read-only: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelWriteDependencyError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Missing dependency: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelWriteError):
            # Generic channel write error
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Channel write error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        else:
            # Not a channel write-specific error, use default classification
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected error during write: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Define retry policy for channel write operations."""
        return {
            "max_attempts": 2,  # Fewer retries for writes (be conservative)
            "delay_seconds": 2.0,
            "backoff_factor": 2.0,
        }

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Delegate orchestrator guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_write_prompt_builder()
        return builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Delegate classifier guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_write_prompt_builder()
        return builder.get_classifier_guide()
