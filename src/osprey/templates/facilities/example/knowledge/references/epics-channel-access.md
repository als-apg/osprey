---
type: Reference
title: EPICS Channel Access Reference
description: Channel naming conventions, access modes, and common PV types used at the Example Research Facility.
tags: [epics, channel-access, pv, naming, convention, reference]
resource: https://epics.anl.gov/EpicsDocumentation/AppDevManuals/ChannelAccess/cadoc_4.htm
---

# EPICS Channel Access Reference

The Example Research Facility uses EPICS (Experimental Physics and Industrial
Control System) Channel Access as its primary control-system protocol.  This
document summarises how ERF channels are named and the common record types.

## Naming Convention

ERF channel names do not follow one grammar: names grew over decades under
several conventions.  A channel is identified by its record in the facility
file, never by a pattern over its address.  Find a channel with the channel
finder and use the address its record states; do not compose an address from
a family or an index.

Each channel record states its access mode as its role:

| Role | Mode | Example |
|------|------|---------|
| setpoint | Write — triggers a hardware action | [SR:MAG:HCM:01:CURRENT:SP](facility:SR:MAG:HCM:01:CURRENT:SP) |
| readback | Read | [SR:MAG:HCM:01:CURRENT:RB](facility:SR:MAG:HCM:01:CURRENT:RB) |

## Common Record Types

| EPICS record | ERF usage | Example |
|-------------|-----------|---------|
| `ai` / `ao` | Analog readback / setpoint | [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB), [SR:RF:CAVITY:01:VOLTAGE:SP](facility:SR:RF:CAVITY:01:VOLTAGE:SP) |
| `bi` / `bo` | Binary readback / output | [SR:VAC:ION-PUMP:01:STATUS:ON](facility:SR:VAC:ION-PUMP:01:STATUS:ON), [GTL:GUN:EGUN:01:HV_ON:SP](facility:GTL:GUN:EGUN:01:HV_ON:SP) |
| `waveform` | Array data | [SR:MAG:SEPTUM:01:CURRENT:WAVEFORM](facility:SR:MAG:SEPTUM:01:CURRENT:WAVEFORM) |

## Access Control

- **Read access** is unrestricted for diagnostic channels.
- **Write access** to setpoints and commands requires the OSPREY approval
  workflow (see [Safety Rules](/references/safety-rules.md)).
- **Interlocked channels** (PSS, vacuum gate valves) have hardware-enforced
  write guards that the software cannot override.

## Units Convention

Each channel record states its unit; the channel tools return it with the
value, and the KNOWLEDGE panel shows it when you hover over a facility link.
Read the unit from the record rather than assuming one: the beam current
readback [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB) and
the vacuum gauge [SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB)
each carry their own.

## Cross-links

- [Safety Rules](/references/safety-rules.md) — write-access policy
- [Vacuum System](/subsystems/vacuum.md) — pressure channels
- [Transport and Delivery](/subsystems/transport-delivery.md) — magnet and BPM channels

# Citations

[1] [EPICS Channel Access Reference Manual](https://epics.anl.gov/EpicsDocumentation/AppDevManuals/ChannelAccess/cadoc_4.htm)
