"""Integration guardrail: the archiver data path places daily ``at_time`` events
at the facility wall-clock, independent of the deploy host's ``$TZ``.

Drives the full ``MockArchiverConnector.get_data`` -> archive composite path over
the control-assistant build's ``vacuum-burst`` scenario (SR07 pressure spike at
14:32:08) and asserts the spike materializes at 14:32 — and lands at the *same*
wall-clock whether the box runs UTC or a wildly different zone. No LLM involved;
this locks the regression that broke the benchmark on non-UTC boxes.
"""

import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.va_in_process_connector import simulation_state_dir
from tests.facility.served_tree import in_process_config

SR07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"


def _vacuum_burst_view(built_control_assistant, tmp_path: Path) -> Path:
    """The control-assistant build's simulator view, copied and pinned to ``vacuum-burst``.

    The copy sits at ``build/data/simulator`` under ``tmp_path`` with no
    rendered config beside it, so its scenario state lives under ``tmp_path``.
    """
    view = tmp_path / "build" / "data" / "simulator"
    shutil.copytree(built_control_assistant.build_dir / "data" / "simulator", view)
    state_dir = simulation_state_dir(view)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_scenarios").write_text("nominal\nvacuum-burst\n")
    return view


async def _sr07_window(view: Path):
    """SR07 pressure over a 10-min window straddling 14:32:08 UTC (per-second)."""
    connector = MockArchiverConnector()
    await connector.connect(in_process_config(view, sample_rate_hz=1.0))
    try:
        center = (datetime.now(UTC) - timedelta(days=1)).replace(
            hour=14, minute=32, second=8, microsecond=0
        )
        start = center - timedelta(minutes=5)
        end = center + timedelta(minutes=5)
        df = await connector.get_data([SR07], start, end, precision_ms=1000)
    finally:
        await connector.disconnect()
    return df, center


def _sr07_rows(df):
    """SR07's own rows from the long-format frame, as a channel-only slice."""
    return df.loc[df["channel"] == SR07]


def _assert_spike_at_center(df, center: datetime, box: str) -> None:
    """SR07 bursts well above baseline, and the peak sits at ``center``."""
    sr07 = _sr07_rows(df)
    baseline = float(sr07["value"].median())
    peak = float(sr07["value"].max())
    # SR07 baseline pressure ~5e-8; the burst spikes it ~4x. The seed's keyed
    # noise rides on the series, so assert a margin comfortably below the ~4x
    # ceiling rather than on it.
    assert peak > 2.5 * baseline, (
        f"[{box}] no SR07 burst: peak {peak:.2e} vs baseline {baseline:.2e}"
    )

    # The peak sits at 14:32:08, not some box-local-shifted hour.
    peak_ts = sr07.loc[sr07["value"].idxmax(), "timestamp"].to_pydatetime()
    assert abs((peak_ts - center).total_seconds()) < 30, f"[{box}] peak at {peak_ts}"


@pytest.mark.asyncio
async def test_archiver_spike_is_at_facility_wall_clock_whatever_the_box_tz(
    built_control_assistant, tmp_path
):
    """Same window, same facility zone (UTC) → the burst is there, at 14:32:08,
    whether the host ``$TZ`` is UTC or UTC+14. Before the fix the spike shifted
    with the box zone."""
    view = _vacuum_burst_view(built_control_assistant, tmp_path)

    df_utc, center_utc = await _sr07_window(view)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TZ", "Pacific/Kiritimati")  # UTC+14
        time.tzset()
        try:
            df_far, center_far = await _sr07_window(view)
        finally:
            mp.undo()
            time.tzset()

    _assert_spike_at_center(df_utc, center_utc, "UTC box")
    _assert_spike_at_center(df_far, center_far, "UTC+14 box")
