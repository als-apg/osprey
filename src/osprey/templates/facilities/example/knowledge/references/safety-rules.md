---
type: Reference
title: Agent Safety Rules
description: Binding constraints on what the OSPREY agent may and may not do at the Example Research Facility control system.
tags: [safety, policy, agent, constraints, write-protection]
---

# Agent Safety Rules

This document states the binding safety constraints for the OSPREY agent
operating on the Example Research Facility control system.  These rules
supplement the general OSPREY safety rules and take precedence for this
facility.

## Hard Limits (Never Violate)

1. **No PSS bypass.** The agent must never attempt to defeat a PSS interlock,
   beam permit or global beam inhibit without explicit operator
   authorization in the current session.  The example facility file holds no
   PSS records; the PSS is operated at its own panel.

2. **No unsolicited beam-off commands.** The agent must not assert the global
   beam inhibit, switch off the gun high voltage
   ([GTL:GUN:EGUN:01:HV_ON:SP](facility:GTL:GUN:EGUN:01:HV_ON:SP)) or stop
   injection without a direct operator instruction in the current turn.

3. **No RF ramp jumps.** RF setpoints, such as the cavity voltage
   [SR:RF:CAVITY:01:VOLTAGE:SP](facility:SR:RF:CAVITY:01:VOLTAGE:SP), must be
   changed in steps of ≤ 10% of the current value per command; never jump
   to a large value in one step.

4. **No motor moves during beam.** Motor commands at experimental stations
   require the station shutter to be closed unless the operator explicitly
   states the shutter should remain open.

5. **Channel limits are authoritative.** Never attempt to write a value
   outside the bounds of the records in `data/facility/limits.yaml`, which
   `osprey build` renders into `build/data/channel_limits.json`, the file the
   connector's reference monitor and the runtime's limits check enforce on
   every write, even if the operator requests it.  Explain the limit and ask
   for confirmation if the operator insists.

## Soft Limits (Require Confirmation)

- Orbit corrections exceeding 1 mm/step in any plane — confirm with operator.
- Quadrupole current changes `> 5%` of nominal — confirm intent.
- Gate valve commands — confirm the adjacent section is at safe pressure.

## Read-Only Channels

A channel whose record states the role `readback` must never be written,
regardless of operator requests.  Among them:

- Beam current, for example [SR:DIAG:DCCT:01:CURRENT:RB](facility:SR:DIAG:DCCT:01:CURRENT:RB)
- BPM positions and sums, for example [SR:DIAG:BPM:01:POSITION:X](facility:SR:DIAG:BPM:01:POSITION:X)
  and [BTS:DIAG:BPM:01:SUM:RB](facility:BTS:DIAG:BPM:01:SUM:RB)
- Pressure gauges, for example [SR:VAC:GAUGE:SR01:PRESSURE:RB](facility:SR:VAC:GAUGE:SR01:PRESSURE:RB)
- Hardware-driven status, for example [SR:RF:CAVITY:01:STATUS:INTERLOCK](facility:SR:RF:CAVITY:01:STATUS:INTERLOCK)

The channel finder returns each channel's role; check it rather than guessing
from the address.

## Escalation

If an operator request would require violating any hard limit, the agent must:

1. Explain clearly which rule is being triggered.
2. Suggest the closest safe alternative.
3. Offer to page the on-call accelerator physicist if the situation requires
   an exception.

## Cross-links

- [Personnel Safety System](/subsystems/pss.md) — PSS interlock hierarchy
- [EPICS Channel Access Reference](/references/epics-channel-access.md) — channel naming and access modes
