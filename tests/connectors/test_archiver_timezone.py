"""Integration guardrail: the archiver data path places daily ``at_time`` events
at the facility wall-clock, independent of the deploy host's ``$TZ``.

Drives the full ``MockArchiverConnector.get_data -> SimulationEngine`` path over
the shipped control_assistant ``vacuum-burst`` scenario (SR07 pressure spike at
14:32:08) and asserts the spike materializes at 14:32 — and lands at the *same*
wall-clock whether the box runs UTC or a wildly different zone. No LLM involved;
this locks the regression that broke the benchmark on non-UTC boxes.
"""

import os
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector

TEMPLATE_SIM = (
    Path(__file__).parents[1].parent / "src/osprey/templates/apps/control_assistant/data/simulation"
)
SR07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"


def _vacuum_burst_machine(tmp_path: Path) -> Path:
    """Copy the shipped machine + scenario bundle, pinned to ``vacuum-burst``."""
    machine = tmp_path / "machine.json"
    shutil.copy(TEMPLATE_SIM / "machine.json", machine)
    shutil.copytree(TEMPLATE_SIM / "scenarios", tmp_path / "scenarios")
    (tmp_path / "active_scenarios").write_text("nominal\nvacuum-burst\n")
    return machine


async def _sr07_window(machine: Path):
    """SR07 pressure over a 10-min window straddling 14:32:08 UTC (per-second)."""
    connector = MockArchiverConnector()
    await connector.connect(
        {"simulation_file": str(machine), "sample_rate_hz": 1.0, "noise_level": 0.0}
    )
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


@pytest.mark.asyncio
async def test_archiver_spike_materializes_at_facility_wall_clock(tmp_path):
    machine = _vacuum_burst_machine(tmp_path)
    df, center = await _sr07_window(machine)

    series = df[SR07]
    baseline = float(series.median())
    peak = float(series.max())
    # SR07 baseline pressure ~5e-8; the burst spikes it ~4x. The engine applies
    # unseeded per-channel multiplicative noise to synthesized series (independent
    # of the connector's noise_level, which only gates the generic non-engine
    # path), so assert a margin comfortably below the ~4x ceiling rather than on it.
    assert peak > 2.5 * baseline, f"no SR07 burst: peak {peak:.2e} vs baseline {baseline:.2e}"

    # The peak sits at 14:32:08, not some box-local-shifted hour.
    peak_ts = series.idxmax().to_pydatetime()
    assert abs((peak_ts - center).total_seconds()) < 30


@pytest.mark.asyncio
async def test_archiver_spike_is_box_tz_independent(tmp_path):
    """Same window, same facility zone (UTC) → identical peak position regardless
    of the host ``$TZ``. Before the fix the spike shifted with the box zone."""
    machine = _vacuum_burst_machine(tmp_path)

    df_utc, center = await _sr07_window(machine)
    peak_utc = df_utc[SR07].idxmax().to_pydatetime()

    old_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14
        time.tzset()
        df_far, _ = await _sr07_window(machine)
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        time.tzset()

    peak_far = df_far[SR07].idxmax().to_pydatetime()
    # Both peaks land at 14:32:08 facility-wall-clock, within sampling resolution.
    assert abs((peak_utc - center).total_seconds()) < 30
    assert abs((peak_far - center).total_seconds()) < 30
