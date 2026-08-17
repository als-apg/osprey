---
type: subsystem
title: EPICS Gateways
description: Read-only bridges between the machine network and the office network.
tags: [controls, epics, network]
---

# EPICS Gateways

The machine network is not routable from the office network. Two gateway hosts
bridge the two, republishing channels from the machine side so that archivers,
dashboards, and analysis notebooks can subscribe without holding an account on a
control-room console.

Both gateways are configured read-only: they serve monitors and single reads, and
they reject writes outright. An operator who needs to command a device must do so
from a console on the machine network. Attempting a write through a gateway
returns an access-denied error rather than failing silently.

Gateway throughput is finite. A client that subscribes to several thousand
channels at full update rate will be throttled, and the gateway logs the offending
client address. Bulk data pulls belong in the archiver, not in a live monitor.
