#!/usr/bin/env python3
"""Caproto soft IOC for the OSPREY ↔ Phoebus bridge demo.

Serves a small set of EPICS Channel Access PVs (prefix ``DEMO:``) that both
Phoebus (the ``.bob`` demo panels bind their widgets to these PVs) and the
OSPREY agent (via its ``epics`` control-system connector, ``pyepics``) read and
write. This is the shared control-system backend that makes the demo's
"drive the GUI, observe the PV; or write the PV directly" loop possible — the
in-process OSPREY ``mock`` connector cannot serve Channel Access, so a real
soft IOC is required.

PV map (all under the ``DEMO:`` prefix):

==================  ======  ==========  ======================================
PV                  type    access      demo role / widget kind it drives
==================  ======  ==========  ======================================
DEMO:Setpoint       ao      read/write  Text Entry widget (verb ``type``) AND
                                        the "Set 42" action button (verb
                                        ``click`` → WritePV 42)
DEMO:Readback       ai      read-only   Text Update / meter — mirrors Setpoint
DEMO:Current        ai      read-only   live noisy value — trend plot / meter
DEMO:Enable         bo      read/write  Bool Button (toggle) — OFF/ON
DEMO:Valve          bo      read/write  Bool Button (momentary) — CLOSED/OPEN
DEMO:Status         mbbi    read-only   LED / status — OK/WARN/FAULT (derived)
==================  ======  ==========  ======================================

Writing DEMO:Setpoint mirrors the value to DEMO:Readback and re-derives
DEMO:Status (>=80 → WARN, >=95 → FAULT). DEMO:Current wobbles once per second
around the readback so live widgets visibly update.

Run (binds Channel Access on the loopback interface)::

    EPICS_CAS_INTF_ADDR_LIST=127.0.0.1 \\
    EPICS_CAS_BEACON_ADDR_LIST=127.0.0.1 \\
    python demo_ioc.py --list-pvs

Then point both Phoebus and OSPREY at it with
``EPICS_CA_ADDR_LIST=127.0.0.1`` / ``EPICS_CA_AUTO_ADDR_LIST=NO``.
"""

from __future__ import annotations

import random

from caproto import ChannelType
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run


class DemoIOC(PVGroup):
    """A handful of PVs that mirror a simple setpoint→readback control loop."""

    # Explicit CamelCase ``name=`` so the PVs are DEMO:Setpoint (not the
    # lowercased attribute name caproto would derive) — EPICS names are
    # case-sensitive and the panels/config bind to the CamelCase form.
    setpoint = pvproperty(
        name="Setpoint",
        value=0.0,
        units="mA",
        precision=2,
        doc="Operator setpoint (writable). Drives Readback + Status.",
    )
    readback = pvproperty(
        name="Readback",
        value=0.0,
        units="mA",
        precision=2,
        read_only=True,
        doc="Mirrors the last committed setpoint.",
    )
    current = pvproperty(
        name="Current",
        value=100.0,
        units="mA",
        precision=2,
        read_only=True,
        doc="Live (noisy) current — updates once per second for trend widgets.",
    )
    enable = pvproperty(
        name="Enable",
        value="OFF",
        enum_strings=["OFF", "ON"],
        dtype=ChannelType.ENUM,
        doc="Enable toggle (bool button).",
    )
    valve = pvproperty(
        name="Valve",
        value="CLOSED",
        enum_strings=["CLOSED", "OPEN"],
        dtype=ChannelType.ENUM,
        doc="Valve open/closed (momentary bool button).",
    )
    status = pvproperty(
        name="Status",
        value="OK",
        enum_strings=["OK", "WARN", "FAULT"],
        dtype=ChannelType.ENUM,
        read_only=True,
        doc="Derived health status from the setpoint magnitude.",
    )

    @staticmethod
    def _derive_status(value: float) -> str:
        if value >= 95:
            return "FAULT"
        if value >= 80:
            return "WARN"
        return "OK"

    @setpoint.putter
    async def setpoint(self, instance, value):
        """Commit a setpoint: mirror to readback and re-derive status."""
        await self.readback.write(value)
        await self.status.write(self._derive_status(value))
        return value

    @current.scan(period=1.0)
    async def current(self, instance, async_lib):
        """Wobble around the current readback so live widgets visibly update."""
        base = self.readback.value if self.readback.value else 100.0
        await instance.write(base + random.uniform(-1.5, 1.5))


def main() -> None:
    ioc_options, run_options = ioc_arg_parser(
        default_prefix="DEMO:",
        desc="OSPREY ↔ Phoebus bridge demo soft IOC",
    )
    ioc = DemoIOC(**ioc_options)
    run(ioc.pvdb, **run_options)


if __name__ == "__main__":
    main()
