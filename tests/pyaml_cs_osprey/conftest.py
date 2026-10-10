"""Shared fixtures: a journaled guarded run and a dict-backed ``osprey.runtime`` stand-in."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pytest

import osprey.runtime
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelReadFailedError,
    ChannelWriteFailedError,
)
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import journaled_run
from osprey.runtime.journal import Journal, active_journals, journaled_write
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig

RUNTIME_FUNCTIONS = (
    "read_channel",
    "read_channels",
    "write_channel",
    "write_channels",
    "channel_limits",
)


class Clock(Protocol):
    """A fake clock a write advances by ``FakeRuntime.write_s``."""

    def advance(self, seconds: float) -> None: ...


class FakeRuntime:
    """Stands in for the channel functions of ``osprey.runtime`` over a dict of values.

    Every call is recorded: ``calls`` names each function in order, ``reads`` holds the
    single-channel read addresses, ``batch_reads`` and ``batch_timeouts`` the batch reads,
    ``writes`` each single-channel write as ``(address, value, kwargs)`` and
    ``batch_writes`` each batch write as ``(channel_values, kwargs)``.

    Failures are injected by setting ``read_error`` / ``write_error`` (raised by every
    read / write), adding an address to ``fail`` (its writes are unconfirmed) or to
    ``refuse`` (its writes are refused as a ``MAX_STEP`` violation when the predicate
    on the value holds). With a ``clock``, every write advances it by ``write_s``.
    With ``batch_only``, a single-channel read or write fails the test.
    """

    def __init__(
        self,
        values: dict[str, Any],
        *,
        clock: Clock | None = None,
        batch_only: bool = False,
    ) -> None:
        self.values = dict(values)
        self.clock = clock
        self.write_s = 0.0
        self.batch_only = batch_only
        self.limits: dict[str, ChannelLimitsConfig] = {}
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self.refuse: dict[str, Callable[[Any], bool]] = {}
        self.fail: set[str] = set()
        self.calls: list[str] = []
        self.reads: list[str] = []
        self.batch_reads: list[list[str]] = []
        self.batch_timeouts: list[float | None] = []
        self.writes: list[tuple[str, Any, dict[str, Any]]] = []
        self.batch_writes: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def read_channel(self, channel_address: str, **kwargs: Any) -> Any:
        if self.batch_only:
            raise AssertionError("a device list never reads one channel at a time")
        self.calls.append("read_channel")
        self.reads.append(channel_address)
        if self.read_error is not None:
            raise self.read_error
        return self.values.get(channel_address)

    def read_channels(self, addresses: Sequence[str], *, timeout: float | None = None) -> list[Any]:
        self.calls.append("read_channels")
        self.batch_reads.append(list(addresses))
        self.batch_timeouts.append(timeout)
        missing = [a for a in addresses if self.values.get(a) is None]
        if missing:
            raise ChannelReadFailedError(missing)
        return [self.values[a] for a in addresses]

    def write_channel(self, channel_address: str, value: Any, **kwargs: Any) -> None:
        if self.batch_only:
            raise AssertionError("a device list never writes one channel at a time")
        self.calls.append("write_channel")
        if self.clock is not None:
            self.clock.advance(self.write_s)
        self.writes.append((channel_address, value, kwargs))
        if self.write_error is not None:
            raise self.write_error
        refused = self.refuse.get(channel_address)
        if refused is not None and refused(value):
            raise ChannelLimitsViolationError(
                channel_address,
                value,
                "MAX_STEP",
                "step too large",
                current_value=self.values[channel_address],
            )
        if channel_address in self.fail:
            raise ChannelWriteFailedError(channel_address, "UNCONFIRMED", "readback did not follow")
        self.values[channel_address] = value

    def write_channels(self, channel_values: dict[str, Any], **kwargs: Any) -> None:
        self.calls.append("write_channels")
        self.batch_writes.append((dict(channel_values), kwargs))
        if self.write_error is not None:
            raise self.write_error
        self.values.update(channel_values)

    def channel_limits(self, address: str) -> ChannelLimitsConfig | None:
        return self.limits.get(address)

    def set(self, address: str, value: Any) -> None:
        """A guarded device write: journaled, then written."""
        journaled_write([address], lambda: self.write_channel(address, value))

    def install(
        self, monkeypatch: pytest.MonkeyPatch, names: Sequence[str] = RUNTIME_FUNCTIONS
    ) -> FakeRuntime:
        """Replace the named ``osprey.runtime`` functions with this fake's methods."""
        for name in names:
            monkeypatch.setattr(osprey.runtime, name, getattr(self, name))
        return self


class DictConnector(ControlSystemConnector):
    """A connector over a dict of values, armed for writes, served through ``osprey.runtime``.

    ``_state`` holds every channel's value; a read of an address it lacks fails, and
    every write lands in it through :meth:`_put`, which a subclass overrides to fail
    one address.
    """

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = dict(values or {})
        self._limits_validator = None

    @property
    def _writes_enabled(self) -> bool:
        return True

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, channel_address: str, timeout: float | None = None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        if channel_address not in self._state:
            raise ValueError(f"{channel_address} is not served")
        return ChannelValue(
            value=self._state[channel_address],
            timestamp=datetime.now(UTC),
            metadata=ChannelMetadata(),
        )

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        return await self._read_concurrently(list(channel_addresses), timeout)

    def _put(self, channel_address: str, value: Any) -> None:
        self._state[channel_address] = value

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
        confirm: bool | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
    ) -> ChannelWriteResult:
        try:
            self._put(channel_address, value)
        except Exception as exc:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"write failed: {exc}",
            )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.UNREQUESTED,
        )

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, subscription_id):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:
        return channel_address in self._state


@pytest.fixture
def guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Journal]:
    """A journaled guarded run on ``live`` in a throwaway deployment repo, for the test.

    The repo holds only its ``profile.yml``, so the run's lock and durable journal
    live under ``<tmp>/repo/var/guarded_run/live/``.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv(ENV_CONTROL_TARGET, raising=False)
    monkeypatch.delenv(ENV_CONTROL_TARGET_GENERATION, raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    with journaled_run("live") as journal:
        yield journal
    assert active_journals() == ()
