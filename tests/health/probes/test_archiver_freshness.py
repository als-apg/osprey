"""Tests for the ``archiver_freshness`` probe.

Pins the archiver-freshness contract: a missing ``channel`` and a missing
``archiver:`` block are config-style errors; an unreachable archiver (or a
raising query) is an error carrying ``str(exc)``; a sample within ``max_age_s``
is ``ok`` with its age reported; a sample older than the threshold — or an empty
query window — is a ``warning``; timezone-naive archiver timestamps are read as
UTC. The archiver connector is acquired lazily via ``ctx.runtime.get_archiver``
with the ``ctx.config`` archiver block taking precedence, and one test drives the
real in-tree :class:`MockArchiverConnector` end to end through a real
:class:`HealthRuntime`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe
from osprey.health.probes.archiver_freshness import run
from osprey.health.runtime import HealthRuntime


class _SpyArchiver:
    """Returns a preset DataFrame from ``get_data``; records the query args."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._frame = frame if frame is not None else pd.DataFrame()
        self._raise_exc = raise_exc
        self.get_data_calls: list[tuple[list[str], Any, Any, Any]] = []
        self.disconnected = False

    async def get_data(
        self,
        pv_list: list[str],
        start_date: Any,
        end_date: Any,
        precision_ms: int = 1000,
        timeout: int | None = None,
    ) -> pd.DataFrame:
        self.get_data_calls.append((pv_list, start_date, end_date, timeout))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._frame

    async def disconnect(self) -> None:
        self.disconnected = True


class _SpyRuntime:
    """Lazy archiver owner: hands out the connector only when asked."""

    def __init__(self, archiver: Any) -> None:
        self._archiver = archiver
        self.get_archiver_calls: list[dict[str, Any]] = []

    async def get_archiver(self, archiver_config: dict[str, Any]) -> Any:
        self.get_archiver_calls.append(archiver_config)
        return self._archiver


def _ctx(
    archiver: Any,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[ProbeContext, _SpyRuntime]:
    if config is None:
        config = {"archiver": {"type": "mock_archiver"}}
    runtime = _SpyRuntime(archiver)
    return ProbeContext(runtime=runtime, config=config), runtime  # type: ignore[arg-type]


def _frame(newest_age_s: float, value: float = 1.5, *, tz: bool = True) -> pd.DataFrame:
    """A single-channel frame whose newest sample is ``newest_age_s`` seconds old.

    With ``tz=False`` the index is timezone-*naive UTC* — how the EPICS Archiver
    Appliance can return timestamps — not naive local time.
    """
    now = datetime.now(UTC)
    if not tz:
        now = now.replace(tzinfo=None)
    stamps = [now - timedelta(seconds=newest_age_s + 30), now - timedelta(seconds=newest_age_s)]
    return pd.DataFrame({"BEAM:CURRENT": [value - 0.1, value]}, index=pd.to_datetime(stamps))


# --- Registry wiring --------------------------------------------------------


def test_probe_is_registered() -> None:
    assert get_probe("archiver_freshness") is run


# --- Config-style errors ----------------------------------------------------


async def test_missing_channel_is_error() -> None:
    ctx, runtime = _ctx(_SpyArchiver())
    result = await run({}, ctx)
    assert result.status is Status.ERROR
    assert "channel" in result.message
    assert runtime.get_archiver_calls == []  # never touched the archiver


async def test_no_archiver_block_is_error() -> None:
    ctx, runtime = _ctx(_SpyArchiver(), config={})
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message
    assert runtime.get_archiver_calls == []  # misconfig detected before construction


async def test_empty_archiver_block_is_error() -> None:
    ctx, _runtime = _ctx(_SpyArchiver(), config={"archiver": {}})
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message


# --- Unreachable archiver ---------------------------------------------------


async def test_unreachable_archiver_is_error_with_details() -> None:
    archiver = _SpyArchiver(raise_exc=ConnectionError("archiver refused connection"))
    ctx, _runtime = _ctx(archiver)
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "archiver unreachable" in result.message
    assert "archiver refused connection" in result.details
    assert result.latency_ms > 0  # latency measured on the failure branch


# --- Freshness grading ------------------------------------------------------


async def test_fresh_sample_is_ok() -> None:
    archiver = _SpyArchiver(_frame(newest_age_s=42, value=401.2))
    ctx, _runtime = _ctx(archiver)
    result = await run({"channel": "BEAM:CURRENT", "max_age_s": 600}, ctx)
    assert result.status is Status.OK
    assert "Fresh" in result.message
    assert "401.2" in result.value
    assert result.latency_ms > 0


async def test_stale_sample_is_warning_with_age() -> None:
    archiver = _SpyArchiver(_frame(newest_age_s=1200))
    ctx, _runtime = _ctx(archiver)
    result = await run({"channel": "BEAM:CURRENT", "max_age_s": 600}, ctx)
    assert result.status is Status.WARNING
    assert "Stale" in result.message
    assert "1200" in result.message  # observed age reported
    assert "600" in result.message  # threshold reported


async def test_empty_window_is_warning() -> None:
    ctx, _runtime = _ctx(_SpyArchiver(pd.DataFrame()))
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.WARNING
    assert "No samples" in result.message


async def test_all_null_column_is_warning() -> None:
    frame = pd.DataFrame(
        {"BEAM:CURRENT": [float("nan"), float("nan")]},
        index=pd.to_datetime([datetime.now(UTC) - timedelta(seconds=s) for s in (60, 30)]),
    )
    ctx, _runtime = _ctx(_SpyArchiver(frame))
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.WARNING
    assert "No samples" in result.message


async def test_default_max_age_is_600() -> None:
    # 500 s old with the default threshold (600) is fresh; 700 s old is stale.
    fresh_ctx, _ = _ctx(_SpyArchiver(_frame(newest_age_s=500)))
    assert (await run({"channel": "BEAM:CURRENT"}, fresh_ctx)).status is Status.OK
    stale_ctx, _ = _ctx(_SpyArchiver(_frame(newest_age_s=700)))
    assert (await run({"channel": "BEAM:CURRENT"}, stale_ctx)).status is Status.WARNING


# --- Timezone handling ------------------------------------------------------


async def test_naive_timestamps_read_as_utc() -> None:
    # A naive index (no tzinfo) must not raise and is read as UTC; a recent
    # naive sample grades fresh rather than crashing on naive/aware subtraction.
    archiver = _SpyArchiver(_frame(newest_age_s=42, tz=False))
    ctx, _runtime = _ctx(archiver)
    result = await run({"channel": "BEAM:CURRENT", "max_age_s": 600}, ctx)
    assert result.status is Status.OK


# --- Lazy acquisition and config precedence ---------------------------------


async def test_archiver_acquired_lazily_with_config_block() -> None:
    archiver = _SpyArchiver(_frame(newest_age_s=10))
    block = {"type": "epics_archiver", "epics_archiver": {"url": "https://arch.example"}}
    ctx, runtime = _ctx(archiver, config={"archiver": block})
    assert runtime.get_archiver_calls == []  # nothing acquired before the run
    await run({"channel": "BEAM:CURRENT"}, ctx)
    assert runtime.get_archiver_calls == [block]  # ctx.config block passed through, once


async def test_timeout_passed_to_get_data() -> None:
    archiver = _SpyArchiver(_frame(newest_age_s=10))
    ctx, _runtime = _ctx(archiver)
    await run({"channel": "BEAM:CURRENT", "timeout_s": 7.0}, ctx)
    pv_list, _start, _end, timeout = archiver.get_data_calls[0]
    assert pv_list == ["BEAM:CURRENT"]
    assert timeout == 7  # float seconds coerced to the connector's int timeout


# --- Real in-tree MockArchiverConnector end to end --------------------------


async def test_real_mock_archiver_is_fresh() -> None:
    runtime = HealthRuntime({})
    try:
        ctx = ProbeContext(runtime=runtime, config={"archiver": {"type": "mock_archiver"}})
        result = await run({"channel": "BEAM:CURRENT", "name": "arch"}, ctx)
        assert result.status is Status.OK
        assert result.name == "arch"
        assert result.value  # a reading was captured
        assert runtime.archiver_ever_constructed
    finally:
        await runtime.shutdown()
