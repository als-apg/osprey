---
type: Subsystem
title: Personnel Safety System (PSS)
description: Access control and radiation interlock system governing beam permits, door status, and area searches.
tags: [safety, interlock, access-control, radiation, beam-permit]
---

# Personnel Safety System (PSS)

The Personnel Safety System (PSS) is the primary radiation and access safety
layer for the Example Research Facility.  It controls beam permits, monitors
door status, and enforces area access interlocks.

The example facility file holds no records for the PSS: its states are read
and its resets made at the PSS panels, not through the control tools.

## Key States

| State | Values |
|-------|--------|
| Area interlock | `FAULTED` / `OK` |
| Area beam permit | `GRANTED` / `DENIED` |
| Area door | `OPEN` / `CLOSED` / `LATCHED` |
| Area search | `REQUIRED` / `COMPLETE` |
| Global beam inhibit | asserted = beam off |

The areas are the six experimental stations, the transport lines and the
injector.

## Radiation Monitors

The radiation trips the PSS acts on are measured by the area radiation
monitors, which the facility file does hold, for example the gamma and neutron
dose rates [SR:DIAG:GAMMA:01:DOSE_RATE:INST](facility:SR:DIAG:GAMMA:01:DOSE_RATE:INST)
and [SR:DIAG:NEUTRON:01:DOSE_RATE:INST](facility:SR:DIAG:NEUTRON:01:DOSE_RATE:INST),
and their alarms [SR:DIAG:GAMMA:01:STATUS:ALARM](facility:SR:DIAG:GAMMA:01:STATUS:ALARM)
and [SR:DIAG:NEUTRON:01:STATUS:ALARM](facility:SR:DIAG:NEUTRON:01:STATUS:ALARM).

## Beam Permit Hierarchy

1. **Global inhibit** — overrides all local permits.
2. **Area interlock** — must be `OK` for the area to run.
3. **Local permit** — final gate for beam to that area.

The OSPREY agent **must never attempt to bypass** PSS interlocks.  All beam
permit requests must go through the approved workflow; see
[Safety Rules](/references/safety-rules.md).

## Search and Secure

Before beam can be established in any area, the area must be searched and
secured:

1. Personnel exit the area.
2. Operator confirms search completion (area search `COMPLETE`).
3. PSS latches the door (area door `LATCHED`).
4. Beam permit is granted automatically if all interlocks are clear.

See [PSS Reset Procedure](/procedures/pss-reset.md) for the fault recovery sequence.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — beam-inhibit recipient
- [Experimental Stations](/subsystems/experimental-stations.md) — per-station interlocks
- [PSS Reset Procedure](/procedures/pss-reset.md) — clearing a PSS fault
- [Safety Rules](/references/safety-rules.md) — agent safety constraints
