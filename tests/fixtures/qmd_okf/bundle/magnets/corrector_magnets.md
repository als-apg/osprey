---
type: device
title: Corrector Magnets
description: Small dipoles that trim the closed orbit horizontally and vertically.
tags: [magnets, orbit]
---

# Corrector Magnets

Each girder carries a pair of small air-cooled dipoles, one deflecting the beam in
the horizontal plane and one in the vertical. Their job is to nudge the circulating
beam back onto the reference trajectory after the main lattice has already bent
and focused it — a few tens of microradians of deflection each, no more.

Because they are pure dipoles, they change where the beam goes without changing
the focusing strength of the lattice, so the tune is unaffected by ordinary
trimming. That separation is what makes them safe to drive automatically.

A fast subset shares the same yoke but is wound for low inductance and driven by
a switching supply, so it can respond at kilohertz rates. The slow set is driven
from the same setpoint table used at the start of every fill.

Setpoint channels end in `:SP`; the readback channel reports the delivered current
rather than the requested one, and the two are compared before a correction is
accepted.
