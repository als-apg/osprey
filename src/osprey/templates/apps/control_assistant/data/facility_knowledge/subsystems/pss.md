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

## Key Channels

```
PSS:{Area}:Interlock     # Interlock state: FAULTED / OK
PSS:{Area}:BeamPermit    # Beam permit for area: GRANTED / DENIED
PSS:{Area}:Door:State    # Door state: OPEN / CLOSED / LATCHED
PSS:{Area}:Search:Status # Area search status: REQUIRED / COMPLETE
PSS:Global:BeamInhibit   # Global beam inhibit (1 = beam off)
```

Where `{Area}` is one of: `ES1`–`ES6`, `TD`, `PS`.

## Beam Permit Hierarchy

1. **Global inhibit** (`PSS:Global:BeamInhibit`) — overrides all local permits.
2. **Area interlock** (`PSS:{Area}:Interlock`) — must be `OK` for the area to run.
3. **Local permit** (`PSS:{Area}:BeamPermit`) — final gate for beam to that area.

The OSPREY agent **must never attempt to bypass** PSS interlocks.  All beam
permit requests must go through the approved workflow; see
[Safety Rules](/references/safety-rules.md).

## Search and Secure

Before beam can be established in any area, the area must be searched and
secured:

1. Personnel exit the area.
2. Operator confirms search completion (`PSS:{Area}:Search:Status = COMPLETE`).
3. PSS latches the door (`PSS:{Area}:Door:State = LATCHED`).
4. Beam permit is granted automatically if all interlocks are clear.

See [PSS Reset Procedure](/procedures/pss-reset.md) for the fault recovery sequence.

## Cross-links

- [Transport and Delivery](/subsystems/transport-delivery.md) — beam-inhibit recipient
- [Experimental Stations](/subsystems/experimental-stations.md) — per-station interlocks
- [PSS Reset Procedure](/procedures/pss-reset.md) — clearing a PSS fault
- [Safety Rules](/references/safety-rules.md) — agent safety constraints
