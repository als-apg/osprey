---
type: device
title: Ion Pumps
description: Sputter-ion pumps and the current-based gauging they provide.
tags: [vacuum, pumps]
---

# Ion Pumps

Sputter-ion pumps hold the ring at its operating vacuum. They have no moving parts
and no exhaust: gas molecules are ionized in a Penning cell, accelerated into a
titanium cathode, and buried there, so the pump is also the place the pumped gas
ends up.

Because the discharge current is proportional to the gas density in the cell, the
supply current is calibrated directly into a residual-gas density and displayed as
the sector's operating value. That makes every pump a gauge as well as a pump, and
it is why there is a per-sector number available everywhere in the control system
without a separate gauge controller.

The calibration is gas-species dependent and degrades once the cathode is heavily
loaded, so a pump whose current has been climbing for months is replaced even if
it still holds the sector. Below roughly 1e-11 mbar the current is dominated by
field emission and the reading is no longer meaningful.

Each supply publishes its current, its high-voltage state, and a trip flag to the
[Machine Protection System](/safety/machine_protection.md).
