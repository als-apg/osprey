---
type: convention
title: Channel Naming Convention
description: How control-system channels are named across the facility.
tags: [controls, epics, convention]
---

# Channel Naming Convention

Every control-system channel in the facility follows a four-field pattern:
`<system>:<device><instance>:<property>:<qualifier>`. The system field names the
subsystem (`SR` for storage ring, `BR` for booster, `LN` for linac). The device
field names the equipment class, the instance field its sequence number around
the ring, and the property field the quantity being exposed.

The qualifier distinguishes a live readback from a commanded value. A bare
channel, or one ending in `:AM`, is a measurement. A channel ending in `:SP` is a
setpoint and is therefore writable — the write-safety layer keys off exactly this
suffix, so a channel that is meant to be commanded must carry it.

Names are case-sensitive and are never reused after a device is retired; the
retired identifier is left dangling so that archived data stays unambiguous.

Refer to [EPICS Gateways](/controls/epics_gateways.md) for which of these
identifiers are reachable from the office network.
