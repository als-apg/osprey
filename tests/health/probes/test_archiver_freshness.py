"""Tests for the ``archiver_freshness`` probe.

Pins the archiver-freshness contract: a missing ``channel`` and a missing
``archiver:`` block are config-style errors; an unreachable archiver (or a
raising query) is an error carrying ``str(exc)``; a sample within ``max_age_s``
is ``ok`` with its age reported; a sample older than the threshold — or an empty
query window — is a ``warning``; a frame that breaks the long-format contract is
*not* laundered into a staleness warning. The archiver connector is acquired
lazily via ``ctx.runtime.get_archiver`` with the ``ctx.config`` archiver block
taking precedence, and one test drives the real in-tree
:class:`MockArchiverConnector` end to end through a real :class:`HealthRuntime`.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

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


def _frame(newest_age_s: float, value: float = 1.5) -> pd.DataFrame:
    """A single-channel long-format frame whose newest sample is ``newest_age_s`` s old."""
    now = datetime.now(UTC)
    stamps = [now - timedelta(seconds=newest_age_s + 30), now - timedelta(seconds=newest_age_s)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps),
            "channel": ["BEAM:CURRENT", "BEAM:CURRENT"],
            "value": [value - 0.1, value],
        }
    )


def _empty_long_frame() -> pd.DataFrame:
    """The contract's empty result: the three-column long frame every connector
    returns for a query that matched no samples."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "channel": pd.Series(dtype=str),
            "value": pd.Series(dtype="float64"),
        }
    )


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
    ctx, _runtime = _ctx(_SpyArchiver(_empty_long_frame()))
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.WARNING
    assert "No samples" in result.message


async def test_channel_absent_from_a_populated_frame_is_warning() -> None:
    """A PV with no rows in a frame that carries other channels' samples."""
    frame = _frame(newest_age_s=10)
    ctx, _runtime = _ctx(_SpyArchiver(frame))
    result = await run({"channel": "OTHER:PV"}, ctx)
    assert result.status is Status.WARNING
    assert "No samples" in result.message


async def test_all_null_column_is_warning() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [datetime.now(UTC) - timedelta(seconds=s) for s in (60, 30)]
            ),
            "channel": ["BEAM:CURRENT", "BEAM:CURRENT"],
            "value": [float("nan"), float("nan")],
        }
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


# --- Frame handling ---------------------------------------------------------


async def test_non_conforming_frame_is_not_laundered_into_a_staleness_warning() -> None:
    """A connector that breaks the long-format contract must not read as "no samples".

    A frame without a ``channel`` column is a broken connector; the probe lets
    it raise so the runner reports an error naming the exception, rather than
    a misleading staleness warning.
    """
    ctx, _runtime = _ctx(_SpyArchiver(pd.DataFrame({"ts": [], "val": []})))
    with pytest.raises(KeyError):
        await run({"channel": "BEAM:CURRENT"}, ctx)


async def test_nanosecond_timestamps_are_not_truncated_with_a_warning() -> None:
    """The EPICS connector carries real nanoseconds into the ``timestamp`` column.

    ``Timestamp.to_pydatetime()`` discards them with a UserWarning; the age
    comparison never needed that conversion.
    """
    now = pd.Timestamp.now(tz=UTC).as_unit("ns")
    stamps = pd.DatetimeIndex(
        [now - pd.Timedelta(seconds=72), now - pd.Timedelta(seconds=42) + pd.Timedelta(123, "ns")]
    )
    assert stamps[-1].nanosecond == 123  # the input really carries sub-µs precision
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "channel": ["BEAM:CURRENT", "BEAM:CURRENT"],
            "value": [1.4, 1.5],
        }
    )
    ctx, _runtime = _ctx(_SpyArchiver(frame))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await run({"channel": "BEAM:CURRENT", "max_age_s": 600}, ctx)

    assert result.status is Status.OK
    assert [str(w.message) for w in caught if "nanosecond" in str(w.message)] == []


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


# --- Global-config fallback (ctx.config is None) -----------------------------


async def test_none_config_falls_back_to_global_archiver_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI/standalone: with no per-run config the block comes from the global
    singleton via ``get_config_value("archiver", ...)`` and the run proceeds."""
    import osprey.utils.config as config_module

    block = {"type": "mock_archiver"}
    keys_asked: list[str] = []

    def fake_get_config_value(key: str, default: Any = None) -> Any:
        keys_asked.append(key)
        return block

    monkeypatch.setattr(config_module, "get_config_value", fake_get_config_value)
    archiver = _SpyArchiver(_frame(newest_age_s=10))
    runtime = _SpyRuntime(archiver)
    ctx = ProbeContext(runtime=runtime, config=None)  # type: ignore[arg-type]

    result = await run({"channel": "BEAM:CURRENT"}, ctx)

    assert result.status is Status.OK
    assert keys_asked == ["archiver"]
    assert runtime.get_archiver_calls == [block]  # the global block reached the runtime


def _raising_get_config_value(key: str, default: Any = None) -> Any:
    raise RuntimeError("config file not found")


@pytest.mark.parametrize(
    "get_config_value",
    [
        pytest.param(_raising_get_config_value, id="config-load-raises"),
        pytest.param(lambda key, default=None: "epics", id="scalar-archiver-value"),
    ],
)
async def test_none_config_with_unusable_global_config_is_error(
    monkeypatch: pytest.MonkeyPatch, get_config_value: Any
) -> None:
    """Config unavailability -- a raising loader or a scalar ``archiver:`` value
    -- degrades to the misconfiguration error, not a crash."""
    import osprey.utils.config as config_module

    monkeypatch.setattr(config_module, "get_config_value", get_config_value)
    runtime = _SpyRuntime(_SpyArchiver())
    ctx = ProbeContext(runtime=runtime, config=None)  # type: ignore[arg-type]

    result = await run({"channel": "BEAM:CURRENT"}, ctx)

    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message
    assert runtime.get_archiver_calls == []


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
