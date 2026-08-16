---
type: procedure
title: Top-Off Injection
description: Frequent small injections that hold the beam intensity flat for users.
tags: [operations, injection]
---

# Top-Off Injection

Rather than filling the ring once and letting the intensity decay for eight hours,
the injector delivers a small shot every minute or two. Each shot replaces only
what has been lost since the last one, so the circulating intensity stays inside a
narrow band all shift long and beamline detectors never have to be renormalized.

Because the shutters stay open during the shot, the mode carries extra safety
requirements: the booster energy, the transport-line optics, and the injection
kicker timing are all interlocked, and a deviation in any of them inhibits the
next shot rather than steering an off-energy bunch down a beamline.

The refill decision is taken by a supervisory loop that watches the circulating
intensity readback and requests a shot when it falls below the low limit. If the
injector cannot deliver, the loop stops requesting and raises an alarm instead of
retrying indefinitely.

Bunch-by-bunch feedback runs throughout, since the injected charge arrives into
already-populated buckets and would otherwise drive coupled-bunch motion.
