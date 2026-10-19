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

The example facility file holds no records for the PSS: the interlock, door,
search and permit states below are read at the local PSS panel, not through
the control tools.

## When to Use

- The area interlock shows faulted
- Beam permit denied after a door opening or radiation interlock trip
- After a controlled maintenance entry requiring area re-commissioning

## Prerequisites

- The root cause of the PSS fault has been identified and corrected.
- No personnel are in the area being reset.
- Shift supervisor has authorized the reset.

## Steps

### 1. Identify the fault

1. Read the faulted area interlock at the PSS panel.
2. Check the area door status.
3. Check whether the global beam inhibit is asserted.
4. For a radiation trip, read the area radiation monitors, for example
   [SR:DIAG:GAMMA:01:STATUS:ALARM](facility:SR:DIAG:GAMMA:01:STATUS:ALARM) and
   [SR:DIAG:NEUTRON:01:STATUS:ALARM](facility:SR:DIAG:NEUTRON:01:STATUS:ALARM).

### 2. Physical area search

1. Walk the area perimeter and visually confirm no personnel inside.
2. Activate the area search button at the local panel (physical action).
3. Confirm the panel shows the search complete.

### 3. Latch the door

1. Close and latch the area access door.
2. Confirm the panel shows the door latched.

### 4. Reset the interlock

1. Press the reset button at the local PSS panel (physical action — cannot
   be done remotely).
2. Monitor the area interlock until it returns to OK.
3. Confirm the area beam permit is granted.

### 5. Restore beam

1. If the global inhibit was asserted, the shift supervisor clears it.
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
