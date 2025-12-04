"""
Matplotlib plotting capability for archiver data.

Takes ARCHIVER_DATA and produces a saved PNG plot path.
"""

import csv
import logging
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from osprey.base.decorators import capability_node
from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.planning import PlannedStep
from osprey.base.examples import (
    OrchestratorGuide,
    OrchestratorExample,
    TaskClassifierGuide,
    ClassifierExample,
    ClassifierActions,
)
from osprey.state import AgentState, StateManager
from osprey.registry import get_registry
from osprey.utils.streaming import get_streamer
from osprey.utils.logger import get_logger
from osprey.context.context_manager import ContextManager
from osprey.utils.config import _get_config
from zoneinfo import ZoneInfo

from ..context_classes import ArchiverDataContext, PlotImageContext

logger = get_logger("archiver_plotting")
registry = get_registry()


class PlottingError(Exception):
    """Raised when plot generation fails."""


@capability_node
class ArchiverPlottingCapability(BaseCapability):
    """
    Generate matplotlib plots from ARCHIVER_DATA.
    """

    name = "archiver_plotting"
    description = "Plot archiver time-series data to a PNG image"
    provides = ["PLOT_IMAGE"]
    requires = ["ARCHIVER_DATA"]

    @staticmethod
    async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
        step = StateManager.get_current_step(state)
        streamer = get_streamer("archiver_plotting", state)
        streamer.status("Preparing plot...")

        cm = ContextManager(state)
        try:
            contexts = cm.extract_from_step(
                step, state, constraints=[registry.context_types.ARCHIVER_DATA], constraint_mode="hard"
            )
            archiver_data: ArchiverDataContext = contexts[registry.context_types.ARCHIVER_DATA]
        except ValueError as e:
            raise PlottingError(str(e))

        channels: List[str] = archiver_data.available_channels
        if not channels:
            raise PlottingError("No channels available to plot.")

        # Optional channel subset via parameters
        params = step.get("parameters") or {}
        channels_to_plot = params.get("channels") or channels

        # Normalize timestamps to configured local timezone (default America/Chicago)
        config = _get_config()
        tz_name = config.get("archiver.local_timezone", "America/Chicago")
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = timezone.utc

        timestamps = []
        for ts in archiver_data.timestamps:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            timestamps.append(ts.astimezone(local_tz))

        if not timestamps:
            raise PlottingError("No timestamps available to plot.")

        # Debug: Log data statistics
        logger.info(f"Total timestamps: {len(timestamps)}")
        logger.info(f"Channels to plot: {channels_to_plot}")
        for ch in channels_to_plot:
            if ch in archiver_data.time_series_data:
                values = archiver_data.time_series_data[ch]
                logger.info(f"Channel {ch}: {len(values)} values")
                non_null = [v for v in values if v is not None]
                logger.info(f"Channel {ch}: {len(non_null)} non-null values")
                if non_null:
                    try:
                        numeric_vals = [float(v) for v in non_null if not math.isnan(float(v))]
                        if numeric_vals:
                            logger.info(f"Channel {ch}: min={min(numeric_vals):.6f}, max={max(numeric_vals):.6f}")
                            # Additional diagnostics for sparse/pulsed data
                            sorted_vals = sorted(numeric_vals)
                            median_val = sorted_vals[len(sorted_vals)//2]
                            p95_val = sorted_vals[int(len(sorted_vals)*0.95)]
                            values_above_1 = [v for v in numeric_vals if v > 1.0]
                            logger.info(f"Channel {ch}: median={median_val:.6f}, 95th percentile={p95_val:.6f}")
                            logger.info(f"Channel {ch}: {len(values_above_1)} values > 1.0 (spike events)")
                            if len(values_above_1) > 0:
                                logger.info(f"Channel {ch}: spike values: min={min(values_above_1):.2f}, max={max(values_above_1):.2f}, avg={sum(values_above_1)/len(values_above_1):.2f}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Channel {ch}: Error converting to numeric: {e}")

        # Prepare plot
        fig, ax = plt.subplots(figsize=(10, 4))
        plotted_any = False
        
        # Prepare CSV export data
        csv_data = []
        csv_headers = ["timestamp"] + channels_to_plot
        
        for ch in channels_to_plot:
            if ch not in archiver_data.time_series_data:
                logger.warning(f"Channel {ch} missing from data; skipping")
                continue
            values = archiver_data.time_series_data[ch]

            # Align lengths defensively
            series = []
            for t, v in zip(timestamps, values):
                if v is None:
                    continue
                try:
                    vnum = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(vnum):
                    continue
                series.append((t, vnum))

            if not series:
                logger.warning(f"Channel {ch} has no numeric data; skipping")
                continue
            ts, vs = zip(*series)
            ax.plot(ts, vs, label=ch)
            plotted_any = True
            
            # Collect data for CSV export
            for t, v in series:
                csv_data.append({"timestamp": t.isoformat(), ch: v})

        if not plotted_any:
            plt.close(fig)
            raise PlottingError("No plottable data found in archiver series.")

        ax.set_title(params.get("title") or "Archiver data")
        ax.set_xlabel("Time")
        ax.set_ylabel(params.get("y_label") or "Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        
        # Enable auto-scaling for y-axis to fit all data
        ax.autoscale(enable=True, axis='y', tight=False)
        ax.relim()  # Recalculate limits based on current data
        ax.autoscale_view()  # Update view to show all data
        
        if len(channels_to_plot) > 1:
            ax.legend()
        fig.autofmt_xdate()

        # Determine output path
        config = _get_config()
        build_dir = Path(config.get("build_dir", "build"))
        plot_dir = build_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = params.get("filename") or f"archiver_plot_{timestamp_slug}.png"
        plot_path = plot_dir / filename

        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        # Export CSV file with the data
        csv_filename = filename.replace(".png", ".csv")
        csv_path = plot_dir / csv_filename
        
        if csv_data:
            # Merge rows by timestamp
            merged_data = {}
            for row in csv_data:
                ts = row["timestamp"]
                if ts not in merged_data:
                    merged_data[ts] = {"timestamp": ts}
                merged_data[ts].update(row)
            
            # Write CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writeheader()
                for ts in sorted(merged_data.keys()):
                    writer.writerow(merged_data[ts])
            logger.info(f"CSV data exported to {csv_path}")
            streamer.status(f"Data exported to {csv_path}")
        else:
            logger.warning("No data to export to CSV")

        description = params.get("description") or "Matplotlib plot of archiver data"
        result = PlotImageContext(path=str(plot_path), description=description)

        streamer.status(f"Plot generated at {plot_path}")

        return StateManager.store_context(
            state,
            registry.context_types.PLOT_IMAGE,
            step.get("context_key"),
            result,
        )

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        if isinstance(exc, PlottingError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Plotting error: {exc}",
                metadata={"technical_details": str(exc)},
            )
        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Unexpected plotting error: {exc}",
            metadata={"technical_details": str(exc)},
        )

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
        example = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_plot",
                capability="archiver_plotting",
                task_objective="Plot beam current history",
                expected_output=registry.context_types.PLOT_IMAGE,
                success_criteria="Plot image generated from archiver data",
                inputs=[{registry.context_types.ARCHIVER_DATA: "historical_beam_current_data"}],
                parameters={"title": "Beam Current", "y_label": "mA"},
            ),
            scenario_description="Plot archiver data with matplotlib",
            notes="Takes ARCHIVER_DATA and produces a PNG plot stored in build/plots/",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
            Use `archiver_plotting` to render matplotlib plots from ARCHIVER_DATA.
            Typical flow:
              - channel_finding → time_range_parsing → archiver_retrieval → archiver_plotting → respond
            """).strip(),
            examples=[example],
            priority=8,
        )

    def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
        return TaskClassifierGuide(
            instructions="Select plotting when user explicitly asks for a plot/graph of historical data.",
            examples=[
                ClassifierExample(
                    query="Plot the beam current over the last 8 hours.",
                    result=True,
                    reason="Plot request for historical data."
                ),
                ClassifierExample(
                    query="What is the beam current right now?",
                    result=False,
                    reason="This is a current value request; use channel_value_retrieval."
                ),
            ],
            actions_if_true=ClassifierActions(),
        )
