"""Two control systems that can be told apart by what they read back.

The mock-pair integration suite needs a deployment with *two servable targets*
and no EPICS anywhere: a switch that moves nothing observable proves nothing,
so both targets have to answer the same channel with different numbers. That is
the whole job of this module.

``live`` reaches :class:`LiveConnector` by dotted path — ``resolve_target``
returns ``control_system.type`` verbatim, so naming the class there makes it a
real, servable live target — and ``va`` reaches :class:`VaConnector` through the
registry name ``virtual_accelerator``, which a scratch ``sitecustomize`` on the
child's ``PYTHONPATH`` registers before ``register_builtin_connectors()`` runs
(that function never replaces an existing registration). Neither class imports a
control-system client library, so a child running either one touches no gateway.

What each connector serves
--------------------------
Every value is a constant derived from the connector's own :attr:`seed`, never
randomised: an assertion about *which* process answered has to be an assertion
about an exact number.

* :data:`SHARED_CHANNEL` — served by both, answering the seed itself. Reading it
  before and after a switch is the routing claim.
* :data:`BATCH_CHANNELS` — served by both, one distinct value each, for the
  batched-read round-trip count.
* A probe channel per target (:data:`LIVE_PROBE`, :data:`VA_PROBE`), served
  *only* by the connector it belongs to. Reading the other target's probe
  channel therefore fails, which is the routing claim stated negatively.
* :data:`SLOW_CHANNEL` — blocks far longer than any deadline the suite uses, so
  a read can be caught in flight and the child killed underneath it.
* :data:`LIMITS_CHANNEL` — writing it raises a fully populated
  :class:`~osprey_connectors.errors.ChannelLimitsViolationError`. Every field is
  set, including ``max_step`` and ``current_value``, which no naturally
  occurring violation populates without a readback; a field that crossed the
  process boundary as another field's value is then visible as a failure rather
  than hidden behind a default.

Writes reach that raise only on a deployment whose config says
``control_system.writes_enabled: true`` — the base class's guard returns a
blocked *result* otherwise and this module's ``write_channel`` is never entered.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteVerification,
)
from osprey_connectors.errors import ChannelLimitsViolationError

#: What ``control_system.type`` names to make ``live`` resolve to this module's
#: live connector. Spelled here rather than in the test so the config and the
#: class cannot drift apart.
LIVE_TYPE = "tests.integration._mock_pair_connectors.LiveConnector"

#: The one channel both targets serve, with a different value each.
SHARED_CHANNEL = "PAIR:BEAM:CURRENT"

#: The channel each target proves itself with during a switch.
LIVE_PROBE = "PAIR:LIVE:PROBE"
VA_PROBE = "PAIR:VA:PROBE"

#: Read by the batched-read test. Six is enough that "one frame per channel"
#: and "one frame for the batch" cannot be confused for each other.
BATCH_CHANNELS = [f"PAIR:BATCH:{index}" for index in range(6)]

#: A read that never finishes inside any deadline this suite sets.
SLOW_CHANNEL = "PAIR:SLOW"
SLOW_SECONDS = 120.0

#: Writing this raises the limits violation below.
LIMITS_CHANNEL = "PAIR:LIMITED:SP"

#: The value each connector answers :data:`SHARED_CHANNEL` with.
LIVE_SEED = 101.0
VA_SEED = 202.0

#: Every field ``ChannelLimitsViolationError`` carries, each a distinct value.
LIMITS_FIELDS: dict[str, Any] = {
    "channel_address": LIMITS_CHANNEL,
    "attempted_value": 150.5,
    "violation_type": "MAX_EXCEEDED",
    "violation_reason": "Value 150.5 above maximum 100.0",
    "min_value": -25.0,
    "max_value": 100.0,
    "max_step": 5.0,
    "current_value": 12.25,
}


def make_limits_violation() -> ChannelLimitsViolationError:
    """The exception the child raises, built from the same constants.

    The parent-side comparison exception is built by calling this too, so
    "the same envelope on both sides" is a claim about the boundary rather than
    about two hand-copied literals.
    """
    fields = dict(LIMITS_FIELDS)
    return ChannelLimitsViolationError(value=fields.pop("attempted_value"), **fields)


class _PairConnector(ControlSystemConnector):
    """One half of the pair: constant values, offset by :attr:`seed`."""

    #: What :data:`SHARED_CHANNEL` reads back as here, and the base of every
    #: other served value. Overridden per target.
    seed: float = 0.0

    #: The probe channel this target — and only this target — serves.
    probe_channel: str = ""

    async def connect(self, config: dict[str, Any] | None = None) -> None:
        self.connected_config = dict(config or {})

    async def disconnect(self) -> None:
        return None

    def value_for(self, channel_address: str) -> float:
        """This connector's constant for *channel_address*.

        Raises:
            ConnectionError: The channel belongs to the other target, or to no
                target at all. A refusal is how a read proves which process
                answered it when the value alone would not.
        """
        if channel_address in (SHARED_CHANNEL, self.probe_channel, SLOW_CHANNEL):
            return self.seed
        if channel_address in BATCH_CHANNELS:
            return self.seed + BATCH_CHANNELS.index(channel_address) + 1
        raise ConnectionError(f"{type(self).__name__} serves no channel named {channel_address!r}")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        if channel_address == SLOW_CHANNEL:
            await asyncio.sleep(SLOW_SECONDS)
        return ChannelValue(
            value=self.value_for(channel_address),
            timestamp=datetime.now(UTC),
            metadata=ChannelMetadata(units="mA"),
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        readings = await asyncio.gather(
            *(self.read_channel(address, timeout=timeout) for address in channel_addresses),
            return_exceptions=True,
        )
        return {
            address: reading
            for address, reading in zip(channel_addresses, readings, strict=True)
            if isinstance(reading, ChannelValue)
        }

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        if channel_address == LIMITS_CHANNEL:
            raise make_limits_violation()
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            success=True,
            verification=WriteVerification(level="none", verified=True),
        )

    async def subscribe(self, channel_address: str, callback: Any) -> str:
        raise NotImplementedError("the pair connectors serve reads and writes only")

    async def unsubscribe(self, subscription_id: str) -> None:
        raise NotImplementedError("the pair connectors serve reads and writes only")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        return ChannelMetadata(units="mA")

    async def validate_channel(self, channel_address: str) -> bool:
        return True


class LiveConnector(_PairConnector):
    """The deployment's ``live`` target, reached by dotted path."""

    seed = LIVE_SEED
    probe_channel = LIVE_PROBE


class VaConnector(_PairConnector):
    """The deployment's ``va`` target, registered as ``virtual_accelerator``."""

    seed = VA_SEED
    probe_channel = VA_PROBE
