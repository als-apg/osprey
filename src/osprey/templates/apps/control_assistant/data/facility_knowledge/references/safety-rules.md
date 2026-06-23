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

1. **No PSS bypass.** The agent must never attempt to write to PSS interlock
   channels (`PSS:*:Interlock`, `PSS:*:BeamPermit`, `PSS:Global:BeamInhibit`)
   without explicit operator authorization in the current session.

2. **No unsolicited beam-off commands.** The agent must not assert
   `PSS:Global:BeamInhibit = 1` or close injection gates without a direct
   operator instruction in the current turn.

3. **No RF ramp jumps.** RF power setpoints (`PS:RF:*:Power:SP`) must be
   changed in steps of ≤ 10% of the current value per command; never jump
   to a large value in one step.

4. **No motor moves during beam.** Motor commands at experimental stations
   (`ES{N}:MOT:*:Position:SP`) require the station shutter to be closed
   (`ES{N}:SHUTTER:State = CLOSED`) unless the operator explicitly states
   the shutter should remain open.

5. **Channel limits are authoritative.** Never attempt to write a value
   outside the bounds in `channel_limits.json`, even if the operator requests
   it.  Explain the limit and ask for confirmation if the operator insists.

## Soft Limits (Require Confirmation)

- Orbit corrections exceeding 1 mm/step in any plane — confirm with operator.
- Quadrupole current changes `> 5%` of nominal — confirm intent.
- Gate valve commands — confirm the adjacent section is at safe pressure.

## Read-Only Channels

The following channels must never be written, regardless of operator requests:

```
PS:DCCT:Current         # Readback only
TD:BPM:*:X, *:Y, *:Sum # Readback only
*:VAC:*:Pressure        # Readback only (pressure gauges)
*:Status                # Hardware-driven status — never write
```

## Escalation

If an operator request would require violating any hard limit, the agent must:

1. Explain clearly which rule is being triggered.
2. Suggest the closest safe alternative.
3. Offer to page the on-call accelerator physicist if the situation requires
   an exception.

## Cross-links

- [Personnel Safety System](/subsystems/pss.md) — PSS interlock hierarchy
- [EPICS Channel Access Reference](/references/epics-channel-access.md) — channel naming and access modes
