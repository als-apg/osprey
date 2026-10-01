---
type: Procedure
title: PSS Fault Reset
description: Procedure to clear a Personnel Safety System fault and restore beam permits after an interlock trip.
tags: [pss, safety, interlock, reset, beam-permit, procedure]
---

# PSS Fault Reset

This procedure clears a [Personnel Safety System (PSS)](/subsystems/pss.md)
fault and restores beam permits to the affected area.

> **Safety note:** Never attempt to bypass or defeat a PSS interlock.
> All resets require a physical area search and confirmation by the operator
> on shift.  See [Safety Rules](/references/safety-rules.md).

## When to Use

- `PSS:{Area}:Interlock = FAULTED`
- Beam permit denied after a door opening or radiation interlock trip
- After a controlled maintenance entry requiring area re-commissioning

## Prerequisites

- The root cause of the PSS fault has been identified and corrected.
- No personnel are in the area being reset.
- Shift supervisor has authorized the reset.

## Steps

### 1. Identify the fault

1. Read the faulted area interlock: `PSS:{Area}:Interlock`
2. Check door status: `PSS:{Area}:Door:State`
3. Check global inhibit: `PSS:Global:BeamInhibit`

### 2. Physical area search

1. Walk the area perimeter and visually confirm no personnel inside.
2. Activate the area search button at the local panel (physical action).
3. Confirm: `PSS:{Area}:Search:Status = COMPLETE`

### 3. Latch the door

1. Close and latch the area access door.
2. Confirm: `PSS:{Area}:Door:State = LATCHED`

### 4. Reset the interlock

1. Press the reset button at the local PSS panel (physical action — cannot
   be done remotely).
2. Monitor: `PSS:{Area}:Interlock → OK`
3. Confirm: `PSS:{Area}:BeamPermit → GRANTED`

### 5. Restore beam

1. If global inhibit was asserted: `PSS:Global:BeamInhibit → 0` (shift supervisor).
2. Restore beam via the normal [PS Startup Procedure](/procedures/ps-startup.md)
   or injection restart, as appropriate.

## Common PSS Fault Causes

| Fault indication | Likely cause |
|-----------------|-------------|
| Door open fault | Door opened without area clear |
| Radiation trip | Dose rate exceeded threshold (investigate source) |
| Search incomplete | Search button not pressed / sensor fault |
| HV interlock | High-voltage component fault in area |

## Cross-links

- [Personnel Safety System](/subsystems/pss.md) — interlock hierarchy
- [Primary Source Startup](/procedures/ps-startup.md) — restart after inhibit clears
- [Safety Rules](/references/safety-rules.md) — agent constraints
