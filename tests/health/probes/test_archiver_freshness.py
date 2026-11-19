"""Tests for the ``archiver_freshness`` probe.

Pins the archiver-freshness contract: a missing ``channel`` and a missing
``archiver:`` block are config-style errors; a block that serves no target, an
archive whose settings are refused, or one that gives no answer is an error
carrying the reason; a sample within ``max_age_s`` is ``ok`` with its age
reported; a sample older than the threshold — or an empty query window — is a
``warning``; a frame that breaks the long-format contract is *not* laundered
into a staleness warning. The archive is acquired lazily via
``ctx.runtime.get_archive`` with the first target's spec of the archive row,
``ctx.config`` taking precedence, and one test drives the real in-tree
:class:`MockArchiverConnector` end to end through a real :class:`HealthRuntime`.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.facility.served_tree import served_tree

from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe
from osprey.health.probes.archiver_freshness import run
from osprey.health.runtime import HealthRuntime
from osprey_connectors.connector_types import TARGET_LIVE, TARGET_VA
from osprey_connectors.service_answers import BackendSpec, Unavailable
from osprey_connectors.testing import config_of


class _SpyArchiver:
    """Answers a preset DataFrame (or ``Unavailable``) from ``history``; records the query args."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        unavailable: Unavailable | None = None,
    ) -> None:
        self._frame = frame if frame is not None else pd.DataFrame()
        self._unavailable = unavailable
        self.history_calls: list[tuple[list[str], Any, Any, Any]] = []
        self.closed = False

    async def history(
        self,
        addresses: list[str],
        *,
        start_at: Any,
        end_at: Any,
        precision_ms: int = 1000,  # noqa: ARG002 - the archive contract fixes this signature
        processing: str = "raw",  # noqa: ARG002 - the archive contract fixes this signature
        timeout_s: float | None = None,
    ) -> pd.DataFrame | Unavailable:
        self.history_calls.append((addresses, start_at, end_at, timeout_s))
        if self._unavailable is not None:
            return self._unavailable
        return self._frame

    async def aclose(self) -> None:
        self.closed = True


class _SpyRuntime:
    """Lazy archive owner: hands out the archive only when asked."""

    def __init__(self, archiver: Any, *, refuse: Exception | None = None) -> None:
        self._archiver = archiver
        self._refuse = refuse
        self.get_archive_calls: list[BackendSpec] = []

    async def get_archive(self, spec: BackendSpec) -> Any:
        self.get_archive_calls.append(spec)
        if self._refuse is not None:
            raise self._refuse
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
            "address": ["BEAM:CURRENT", "BEAM:CURRENT"],
            "value": [value - 0.1, value],
        }
    )


def _empty_long_frame() -> pd.DataFrame:
    """The contract's empty result: the three-column long frame every connector
    returns for a query that matched no samples."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "address": pd.Series(dtype=str),
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
    assert runtime.get_archive_calls == []  # never touched the archiver


async def test_no_archiver_block_is_error() -> None:
    ctx, runtime = _ctx(_SpyArchiver(), config={})
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message
    assert runtime.get_archive_calls == []  # misconfig detected before construction


async def test_empty_archiver_block_is_error() -> None:
    ctx, _runtime = _ctx(_SpyArchiver(), config={"archiver": {}})
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message


# --- Unreachable archiver ---------------------------------------------------


async def test_unreachable_archiver_is_error_with_details() -> None:
    unreachable = Unavailable(cause="unreachable", reason="archiver refused connection")
    archiver = _SpyArchiver(unavailable=unreachable)
    ctx, _runtime = _ctx(archiver)
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "archiver unreachable" in result.message
    assert result.details == "archiver refused connection"
    assert result.latency_ms > 0  # latency measured on the failure branch


async def test_refused_settings_are_error_with_details() -> None:
    """The archive reads its settings when it is built; a refused block is the error."""
    refused = ValueError("archiver URL is required for EPICS archiver")
    runtime = _SpyRuntime(_SpyArchiver(), refuse=refused)
    block = {"type": "epics_archiver", "settings": {"url": "https://arch.example"}}
    ctx = ProbeContext(runtime=runtime, config={"archiver": block})  # type: ignore[arg-type]
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "archiver unreachable" in result.message
    assert result.details == "archiver URL is required for EPICS archiver"


async def test_a_block_serving_no_target_is_error() -> None:
    """A facility archive naming no address holds no target's past: nothing is built."""
    runtime = _SpyRuntime(_SpyArchiver())
    ctx = ProbeContext(
        runtime=runtime,  # type: ignore[arg-type]
        config={"archiver": {"type": "epics_archiver", "settings": {}}},
    )
    result = await run({"channel": "BEAM:CURRENT"}, ctx)
    assert result.status is Status.ERROR
    assert "archiver unreachable" in result.message
    assert result.details == "this deployment's archiver block serves no control target"
    assert runtime.get_archive_calls == []


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
            "address": ["BEAM:CURRENT", "BEAM:CURRENT"],
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
            "address": ["BEAM:CURRENT", "BEAM:CURRENT"],
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
    assert runtime.get_archive_calls == []  # nothing acquired before the run
    await run({"channel": "BEAM:CURRENT"}, ctx)
    # The ctx.config block, as the first target's spec of the archive row, once.
    (spec,) = runtime.get_archive_calls
    assert (spec.kind, spec.type, spec.target) == ("archive", "epics_archiver", TARGET_LIVE)
    assert dict(spec.settings) == {"url": "https://arch.example"}


async def test_the_store_is_probed_on_the_target_it_holds() -> None:
    """The mongodb store holds the virtual accelerator's past, so its spec is ``va``'s."""
    block = {"type": "mongodb_archiver", "settings": {"host": "localhost"}}
    config = {"archiver": block, "services": {"virtual_accelerator": {"port": 5064}}}
    ctx, runtime = _ctx(_SpyArchiver(_frame(newest_age_s=10)), config=config)
    await run({"channel": "BEAM:CURRENT"}, ctx)
    (spec,) = runtime.get_archive_calls
    assert (spec.type, spec.target) == ("mongodb_archiver", TARGET_VA)


async def test_timeout_passed_to_history() -> None:
    archiver = _SpyArchiver(_frame(newest_age_s=10))
    ctx, _runtime = _ctx(archiver)
    await run({"channel": "BEAM:CURRENT", "timeout_s": 7.0}, ctx)
    channels, _start, _end, timeout = archiver.history_calls[0]
    assert channels == ["BEAM:CURRENT"]
    assert timeout == 7


# --- Global-config fallback (ctx.config is None) -----------------------------


async def test_none_config_falls_back_to_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI/standalone: with no per-run config the config comes from the global
    config via ``current_config()`` and the run proceeds."""
    import osprey_connectors.config as config_module

    loaded = config_of({"archiver": {"type": "mock_archiver", "settings": {"sample_rate_hz": 2}}})
    loads: list[None] = []

    def fake_current_config() -> Any:
        loads.append(None)
        return loaded

    monkeypatch.setattr(config_module, "current_config", fake_current_config)
    archiver = _SpyArchiver(_frame(newest_age_s=10))
    runtime = _SpyRuntime(archiver)
    ctx = ProbeContext(runtime=runtime, config=None)  # type: ignore[arg-type]

    result = await run({"channel": "BEAM:CURRENT"}, ctx)

    assert result.status is Status.OK
    assert len(loads) == 1
    # The global block reached the runtime, as the first target's spec.
    (spec,) = runtime.get_archive_calls
    assert (spec.type, dict(spec.settings)) == ("mock_archiver", {"sample_rate_hz": 2})


def _raising_current_config() -> Any:
    raise RuntimeError("config file not found")


@pytest.mark.parametrize(
    "current_config",
    [
        pytest.param(_raising_current_config, id="config-load-raises"),
        pytest.param(lambda: config_of({"archiver": "epics"}), id="scalar-archiver-value"),
    ],
)
async def test_none_config_with_unusable_global_config_is_error(
    monkeypatch: pytest.MonkeyPatch, current_config: Any
) -> None:
    """Config unavailability -- a raising loader or a scalar ``archiver:`` value
    -- degrades to the misconfiguration error, not a crash."""
    import osprey_connectors.config as config_module

    monkeypatch.setattr(config_module, "current_config", current_config)
    runtime = _SpyRuntime(_SpyArchiver())
    ctx = ProbeContext(runtime=runtime, config=None)  # type: ignore[arg-type]

    result = await run({"channel": "BEAM:CURRENT"}, ctx)

    assert result.status is Status.ERROR
    assert "no archiver configured" in result.message
    assert runtime.get_archive_calls == []


# --- Real in-tree MockArchiverConnector end to end --------------------------


async def test_real_mock_archiver_is_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The render of a built tree, its config beside the simulator view the archiver serves."""
    render = served_tree(tmp_path, readings=["BEAM:CURRENT"]).parent.parent
    (render / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    monkeypatch.chdir(render)
    runtime = HealthRuntime({})
    try:
        ctx = ProbeContext(runtime=runtime, config={"archiver": {"type": "mock_archiver"}})
        result = await run({"channel": "BEAM:CURRENT", "name": "arch"}, ctx)
        assert result.status is Status.OK
        assert result.name == "arch"
        assert result.value  # a reading was captured
        assert runtime.archive_ever_constructed
    finally:
        await runtime.shutdown()
