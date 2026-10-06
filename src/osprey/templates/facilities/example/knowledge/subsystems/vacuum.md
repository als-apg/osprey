---
type: Subsystem
title: Vacuum System (VAC)
description: Ultra-high vacuum across all beamline sections, maintained by ion pumps, turbomolecular pumps, and gate valves.
tags: [vacuum, uhv, ion-pump, gate-valve, pressure]
---

# Vacuum System (VAC)

The Vacuum System maintains ultra-high vacuum (UHV) conditions throughout the
Example Research Facility beamline.  Nominal operating pressure is
`< 1e-8 Torr` in the photon beam path and `< 1e-9 Torr` in the storage ring.

## Pressure Channels

Pressure readings follow the pattern `{SYS}:VAC:{Section}:Pressure`:

```
PS:VAC:INJ:Pressure     # Injection section pressure (Torr)
TD:VAC:T01:Pressure     # Transport line section T01 pressure (Torr)
ES3:VAC:End:Pressure    # Station 3 endstation pressure (Torr)
```

## Gate Valves

Gate valves isolate sections when a pressure excursion is detected:

```
TD:VAC:GV01:State      # Gate valve state: OPEN / CLOSED
TD:VAC:GV01:Cmd        # Command: OPEN / CLOSE (interlocked)
```

Gate valves are **interlock-controlled**: they close automatically when
neighbouring section pressure exceeds `1e-6 Torr`.  Manual open commands
require both adjacent pressures to be `< 1e-7 Torr`.

## Pump Types

| Device | Typical location | Channel prefix |
|--------|-----------------|----------------|
| [Ion Pump](/devices/ion-pump.md) | All sections | `*:VAC:IP{N}:*` |
| Turbomolecular Pump | Injection section | `PS:VAC:TMP01:*` |
| Roughing Pump | Fore-vacuum stages | `PS:VAC:RP01:*` |

## Venting and Pumpdown

Venting any section requires shutting the adjacent gate valves first.  A
complete pumpdown from atmosphere to operating pressure takes 24–48 hours.

See [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) for the
full pumpdown sequence.

## Cross-links

- [Ion Pump](/devices/ion-pump.md) — primary UHV pump device
- [Primary Source](/subsystems/primary-source.md) — injection vacuum requirements
- [Personnel Safety System](/subsystems/pss.md) — vacuum interlock authority
- [Vacuum Recovery Procedure](/procedures/vacuum-recovery.md) — recovery from a vacuum event
