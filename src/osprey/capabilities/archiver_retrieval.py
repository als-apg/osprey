"""
Archiver Data Retrieval Capability

This capability retrieves historical time-series data from the archiver.
It provides access to archived data for analysis, plotting, and trend monitoring.

Based on ALS Assistant's Get Archiver Data capability pattern.

Configuration:
    The archiver connector is configured in config.yml. By default, the template
    uses the mock archiver connector which works with any channel names.

    Development Mode (config.yml):
        archiver:
            type: mock_archiver
            # Mock uses sensible defaults - no config needed!

    Production Mode (config.yml):
        archiver:
            type: epics_archiver
            epics_archiver:
                url: https://archiver.your-facility.edu:8443
                timeout: 60

    The capability code remains the same - just change the config!
"""

import logging
import textwrap
from datetime import datetime
from typing import Any, ClassVar

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.base.planning import PlannedStep
from osprey.connectors.factory import ConnectorFactory
from osprey.context import CapabilityContext

# ========================================================
# Context Class
# ========================================================


class ArchiverDataContext(CapabilityContext):
    """
    Structured context for archiver data capability results.

    This stores archiver data with datetime objects for full datetime functionality and consistency.
    Based on ALS Assistant's ArchiverDataContext pattern with downsampling support.
    """

    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

    timestamps: list[datetime]  # List of datetime objects for full datetime functionality
    precision_ms: int  # Data precision in milliseconds
    time_series_data: dict[
        str, list[float]
    ]  # Channel name -> time series values (aligned with timestamps)
    available_channels: list[str]  # List of available channel names for intuitive filtering

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Rich description of the archiver data structure."""
        total_points = len(self.timestamps)

        # Get example channel for demo purposes
        example_channel = self.available_channels[0] if self.available_channels else "SR:CURRENT:RB"
        example_value = (
            self.time_series_data[example_channel][0]
            if self.available_channels and self.time_series_data.get(example_channel)
            else 100.5
        )

        start_time = self.timestamps[0]
        end_time = self.timestamps[-1]
        duration = end_time - start_time

        return {
            "total_points": total_points,
            "precision_ms": self.precision_ms,
            "channel_count": len(self.available_channels),
            "available_channels": self.available_channels,
            "time_info": f"Data spans from {start_time} to {end_time} (duration: {duration})",
            "data_structure": "4 attributes: timestamps (list of datetime objects), precision_ms (int), time_series_data (dict of channel_name -> list of float values), available_channels (list of channel names)",
            "CRITICAL_ACCESS_PATTERNS": {
                "get_channel_names": f"channel_names = context.{self.CONTEXT_TYPE}.{key}.available_channels",
                "get_channel_data": f"data = context.{self.CONTEXT_TYPE}.{key}.time_series_data['CHANNEL_NAME']",
                "get_timestamps": f"timestamps = context.{self.CONTEXT_TYPE}.{key}.timestamps",
                "get_single_value": f"value = context.{self.CONTEXT_TYPE}.{key}.time_series_data['CHANNEL_NAME'][index]",
                "get_time_at_index": f"time = context.{self.CONTEXT_TYPE}.{key}.timestamps[index]",
            },
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.time_series_data['{example_channel}'][0] gives {example_value}, context.{self.CONTEXT_TYPE}.{key}.timestamps[0] gives datetime object",
            "datetime_features": "Full datetime functionality: arithmetic, comparison, formatting with .strftime(), timezone operations",
        }

    def get_summary(self) -> dict[str, Any]:
        """
        FOR HUMAN DISPLAY: Format data for response generation.
        Downsamples large datasets to prevent context window overflow.
        """
        max_samples = 10

        try:
            total_points = len(self.timestamps)

            # Create sample indices (start, middle, end)
            if total_points <= max_samples:
                sample_indices = list(range(total_points))
            else:
                # Include start, end, and evenly distributed middle points
                step = max(1, total_points // (max_samples - 2))
                sample_indices = (
                    [0]
                    + list(range(step, total_points - 1, step))[: max_samples - 2]
                    + [total_points - 1]
                )
                sample_indices = sorted(set(sample_indices))  # Remove duplicates and sort

            # Sample timestamps
            sample_timestamps = [self.timestamps[i] for i in sample_indices]

            # Sample channel data
            channel_summary = {}
            for channel_name, values in self.time_series_data.items():
                sample_values = [values[i] for i in sample_indices]

                channel_summary[channel_name] = {
                    "sample_values": sample_values,
                    "sample_timestamps": sample_timestamps,
                    "statistics": {
                        "total_points": len(values),
                        "min_value": min(values),
                        "max_value": max(values),
                        "first_value": values[0],
                        "last_value": values[-1],
                        "mean_value": sum(values) / len(values),
                    },
                }

            return {
                "WARNING": "🚨 THIS IS DOWNSAMPLED ARCHIVER DATA - DO NOT USE FOR FINAL NUMERICAL ANSWERS! 🚨",
                "guidance": "For accurate analysis results, use ANALYSIS_RESULTS context instead of raw archiver data",
                "data_info": {
                    "total_points": total_points,
                    "precision_ms": self.precision_ms,
                    "time_range": {
                        "start": self.timestamps[0] if self.timestamps else None,
                        "end": self.timestamps[-1] if self.timestamps else None,
                    },
                    "downsampling_info": f"Showing {len(sample_indices)} sample points out of {total_points} total points",
                },
                "channel_data": channel_summary,
                "IMPORTANT_NOTE": "Use this only for understanding data structure. For analysis results, request ANALYSIS_RESULTS context.",
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error downsampling archiver data: {e}")
            return {
                "ERROR": f"Failed to downsample archiver data: {str(e)}",
                "WARNING": "Could not process archiver data - use ANALYSIS_RESULTS instead",
            }


# === Archiver-Related Errors ===
class ArchiverError(Exception):
    """Base class for all archiver-related errors."""

    pass


class ArchiverTimeoutError(ArchiverError):
    """Raised when archiver requests time out."""

    pass


class ArchiverConnectionError(ArchiverError):
    """Raised when archiver connectivity issues."""

    pass


class ArchiverDataError(ArchiverError):
    """Raised when archiver returns unexpected data format."""

    pass


class ArchiverDependencyError(ArchiverError):
    """Raised when required dependencies are missing."""

    pass


# ========================================================
# Capability Implementation
# ========================================================


@capability_node
class ArchiverRetrievalCapability(BaseCapability):
    """
    Archiver Data Retrieval Capability with production patterns.

    - Complete archiver service integration with error handling
    - Comprehensive validation and timeout protection
    - Registry-based patterns for context types
    - Rich orchestrator examples and classifier configuration
    - Automatic context management with dependency validation
    """

    name = "archiver_retrieval"
    description = "Retrieve historical channel data from the archiver"
    provides = ["ARCHIVER_DATA"]
    requires = ["CHANNEL_ADDRESSES", ("TIME_RANGE", "single")]

    async def execute(self) -> dict[str, Any]:
        """
        Retrieve historical channel data from the archiver service.

        This method queries the configured archiver (mock or EPICS) for historical
        time-series data across specified channels and time ranges. It automatically
        extracts required inputs, handles data conversion, and provides structured results.

        Returns:
            State updates containing ARCHIVER_DATA context with historical time series.

        Raises:
            ArchiverDependencyError: If required contexts are missing or invalid.
            ArchiverConnectionError: If archiver service is unreachable.
            ArchiverDataError: If data retrieval fails.
        """

        # Get unified logger with automatic streaming
        logger = self.get_logger()

        # Get task description for logging
        task_objective = self.get_task_objective(default="unknown")
        logger.info(f"Starting archiver data retrieval: {task_objective}")
        logger.status("Initializing archiver data retrieval...")

        try:
            # Extract required contexts from execution state
            try:
                channels_to_retrieve, time_range_context = self.get_required_contexts()
                logger.info(
                    "Successfully extracted both required contexts: CHANNEL_ADDRESSES and TIME_RANGE"
                )
            except ValueError as e:
                raise ArchiverDependencyError(str(e)) from e

            # Validate that we have channel addresses to work with
            if not channels_to_retrieve or len(channels_to_retrieve) == 0:
                raise ArchiverDependencyError(
                    "No channel addresses available for archiver data retrieval. The channel finding step may have failed to locate suitable channels."
                )

            logger.status(f"Found {len(channels_to_retrieve)} channels, retrieving data...")

            logger.debug(
                f"Retrieving archiver data for {len(channels_to_retrieve)} channels from {time_range_context.start_date} to {time_range_context.end_date}"
            )

            # Extract optional parameters
            params = self.get_parameters()
            precision_ms = params.get("precision_ms", 1000)

            # Create archiver connector from configuration
            # This will use 'mock_archiver' for development or 'epics_archiver' for production
            # based on the 'archiver' section in config.yml
            connector = await ConnectorFactory.create_archiver_connector()

            try:
                # Retrieve the data from archiver (returns pandas DataFrame)
                archiver_df = await connector.get_data(
                    pv_list=channels_to_retrieve,
                    start_date=time_range_context.start_date,
                    end_date=time_range_context.end_date,
                    precision_ms=precision_ms,
                )

                logger.status("Converting archiver data to structured format...")

                # Extract timestamps from DataFrame index
                timestamps = [ts.to_pydatetime() for ts in archiver_df.index]

                # Extract time series data for each channel
                time_series_data = {
                    channel: archiver_df[channel].tolist()
                    for channel in channels_to_retrieve
                    if channel in archiver_df.columns
                }

                logger.debug(
                    f"Retrieved archiver data with {len(timestamps)} timestamps and {len(time_series_data)} channels"
                )

            finally:
                # Always disconnect connector
                await connector.disconnect()

            logger.status("Creating archiver data context...")

            # Create rich context object
            archiver_context = ArchiverDataContext(
                timestamps=timestamps,
                precision_ms=precision_ms,
                time_series_data=time_series_data,
                available_channels=list(time_series_data.keys()),
            )

            # Log archiver data info with safe timestamp access
            start_time = archiver_context.timestamps[0] if archiver_context.timestamps else "N/A"
            end_time = archiver_context.timestamps[-1] if archiver_context.timestamps else "N/A"
            logger.info(
                f"Retrieved archiver data: {len(archiver_context.timestamps)} points for {len(archiver_context.available_channels)} channels from {start_time} to {end_time}"
            )

            # Store result in execution context
            return self.store_output_context(archiver_context)

        except Exception as e:
            logger.error(f"Archiver data retrieval failed: {e}")
            logger.error(f"Archiver data retrieval failed: {str(e)}")
            raise

    def process_extracted_contexts(self, contexts):
        """
        Flatten CHANNEL_ADDRESSES contexts into single list.

        **HOOK METHOD**: This method is automatically called by get_required_contexts()
        after extracting contexts from state but before returning them to execute().
        Override this method to customize how contexts are processed.

        In this capability, we need to handle the case where multiple CHANNEL_ADDRESSES
        contexts exist (e.g., from multiple channel_finding steps). This hook flattens
        them into a single list of channel strings for archiver retrieval.

        Args:
            contexts: Dict mapping context type names to extracted context objects
                     (may contain lists if multiple contexts of same type exist)

        Returns:
            Processed contexts dict (CHANNEL_ADDRESSES converted to flat list of strings)

        Note:
            Without this override, channels_to_retrieve would be a ChannelAddressesContext
            object (or list of them), but archiver expects a flat list of channel strings.
        """
        channels_raw = contexts["CHANNEL_ADDRESSES"]

        if isinstance(channels_raw, list):
            # Flatten multiple contexts into single channel list
            channels_flat = []
            for ctx in channels_raw:
                channels_flat.extend(ctx.channels)
            self.get_logger().info(
                f"Merged {len(channels_raw)} CHANNEL_ADDRESSES contexts into {len(channels_flat)} total channels"
            )
            contexts["CHANNEL_ADDRESSES"] = channels_flat
        else:
            # Single context - extract channel list
            contexts["CHANNEL_ADDRESSES"] = channels_raw.channels

        return contexts

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """
        Domain-specific error classification with detailed recovery suggestions.
        """

        if isinstance(exc, ArchiverTimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Archiver timeout error: {str(exc)}",
                metadata={
                    "technical_details": "Archiver requests timed out",
                    "suggestions": [
                        "Reduce the time range of your query",
                        "Request fewer channels in a single query",
                        "Increase precision_ms parameter to reduce data points",
                    ],
                },
            )
        elif isinstance(exc, ArchiverConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Archiver connection error: {str(exc)}",
                metadata={
                    "technical_details": "Cannot establish connection to archiver service",
                    "suggestions": [
                        "Verify the archiver service is accessible",
                        "Check network connectivity",
                        "Contact operations if archiver service appears down",
                    ],
                },
            )
        elif isinstance(exc, ArchiverDataError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Archiver data error: {str(exc)}",
                metadata={
                    "technical_details": "Archiver returned unexpected data format",
                    "replanning_reason": f"Archiver data format issue: {exc}",
                    "suggestions": [
                        "Verify that the requested channel names exist and are archived",
                        "Check if the time range contains actual data",
                        "Try a different time range",
                    ],
                },
            )
        elif isinstance(exc, ArchiverDependencyError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Missing dependency: {str(exc)}",
                metadata={
                    "technical_details": "Required input context (CHANNEL_ADDRESSES or TIME_RANGE) not available for archiver data retrieval",
                    "replanning_reason": f"Missing required inputs: {exc}",
                    "suggestions": [
                        "Ensure channel addresses have been found using channel_finding capability",
                        "Verify time range has been parsed using the time_range_parsing capability",
                        "Check that required input contexts are available from previous steps",
                    ],
                },
            )
        elif isinstance(exc, ArchiverError):
            # Generic archiver error
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Archiver error: {str(exc)}",
                metadata={
                    "technical_details": "General archiver service error",
                    "suggestions": [
                        "Retry the request as this may be a temporary service issue",
                        "Simplify the query by reducing time range or number of channels",
                    ],
                },
            )
        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Archiver data retrieval failed: {exc}",
                metadata={"technical_details": str(exc)},
            )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create prompt snippet for archiver data capability."""

        # Define structured examples
        archiver_retrieval_example = OrchestratorExample(
            step=PlannedStep(
                context_key="historical_beam_current_data",
                capability="archiver_retrieval",
                task_objective="Retrieve historical beam current data from archiver for the last 24 hours",
                expected_output="ARCHIVER_DATA",
                success_criteria="Historical data retrieved successfully for specified time range",
                inputs=[
                    {"CHANNEL_ADDRESSES": "beam_current_channels"},
                    {"TIME_RANGE": "last_24_hours_timerange"},
                ],
            ),
            scenario_description="Retrieve historical time-series data from the archiver",
            notes="Requires channel addresses and time range from previous steps. Output stored under ARCHIVER_DATA context type. Optional parameter: precision_ms (default: 1000).",
        )

        # Workflow example: Show what comes AFTER archiver_retrieval for plotting
        plotting_workflow_python_step = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_plot",
                capability="python",
                task_objective="Create a matplotlib time-series plot of the beam current data showing trends over the 24-hour period",
                expected_output="PYTHON_RESULTS",
                success_criteria="Time-series plot created with proper labels, showing beam current trends",
                inputs=[{"ARCHIVER_DATA": "historical_beam_current_data"}],
            ),
            scenario_description="WORKFLOW: Use python capability to plot archiver data from previous step",
            notes="Typical plotting workflow: archiver_retrieval (gets data) → python (creates plot) → respond (delivers to user). The python capability consumes the ARCHIVER_DATA from the archiver_retrieval step.",
        )

        # Workflow example: Show what comes AFTER archiver_retrieval for analysis
        analysis_workflow_python_step = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_statistics",
                capability="python",
                task_objective="Calculate mean, standard deviation, min, and max values of the beam current data over the time period",
                expected_output="PYTHON_RESULTS",
                success_criteria="Statistical metrics calculated and displayed with clear labels",
                inputs=[{"ARCHIVER_DATA": "historical_beam_current_data"}],
            ),
            scenario_description="WORKFLOW: Use python capability to analyze archiver data from previous step",
            notes="Typical analysis workflow: archiver_retrieval (gets data) → python (calculates statistics) → respond (delivers results). The python capability consumes the ARCHIVER_DATA from the archiver_retrieval step.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "archiver_retrieval" steps:**
                - When tasks require historical channel data
                - When retrieving past values from the archiver
                - When time-series data is needed from archived sources

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "historical_data", "trend_data")
                - inputs: Specify required inputs:
                {"CHANNEL_ADDRESSES": "context_key_with_channel_data", "TIME_RANGE": "context_key_with_time_range"}

                **Required Inputs:**
                - CHANNEL_ADDRESSES data: typically from a "channel_finding" step
                - TIME_RANGE data: typically from a "time_range_parsing" step

                **Input flow and sequencing:**
                1. "channel_finding" step must precede this step (if CHANNEL_ADDRESSES data is not present already)
                2. "time_range_parsing" step must precede this step (if TIME_RANGE data is not present already)

                **Output: ARCHIVER_DATA**
                - Contains: Structured historical data from the archiver
                - Available to downstream steps via context system

                **Common downstream workflow patterns:**
                - For plotting requests: archiver_retrieval → python (create plot) → respond
                - For analysis/statistics: archiver_retrieval → python (calculate stats) → respond
                - For complex analysis: archiver_retrieval → data_analysis → respond
                - Combined: archiver_retrieval → data_analysis → python (plot analysis) → respond

                Do NOT plan this for current values; use "channel_read" for real-time data.
                """),
            examples=[
                archiver_retrieval_example,
                plotting_workflow_python_step,
                analysis_workflow_python_step,
            ],
            priority=15,
        )

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier for archiver data capability."""
        return TaskClassifierGuide(
            instructions="Determines if the task requires accessing the archiver. This is relevant for requests involving historical data or trends.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about the AI's capabilities.",
                ),
                ClassifierExample(
                    query="Plot the historical data for vacuum pressure for the last week.",
                    result=True,
                    reason="The query explicitly asks for historical data plotting.",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=False,
                    reason="The query asks for a current value, not historical data.",
                ),
                ClassifierExample(
                    query="Can you plot that over the last 4h?",
                    result=True,
                    reason="The query asks for historical data plotting.",
                ),
                ClassifierExample(
                    query="What was that value yesterday?",
                    result=True,
                    reason="The query asks for historical data.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )
