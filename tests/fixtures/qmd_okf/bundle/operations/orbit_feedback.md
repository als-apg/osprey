---
type: procedure
title: Orbit Feedback
description: The slow and fast loops that hold the circulating beam still.
tags: [operations, orbit, feedback]
---

# Orbit Feedback

Two loops keep the trajectory where the beamlines expect it. The slow loop runs at
roughly 1 Hz over every position monitor and every trim dipole in the ring, and
it removes thermal and mechanical wander: girders settling as the tunnel warms,
insertion-device jaws moving, ground motion over a shift.

The fast loop runs at several kilohertz over a reduced set of monitors and
low-inductance trims, and it suppresses the residual motion the slow loop cannot
follow — power-supply ripple, cooling-water pressure oscillations, and the kick
that arrives with each injection shot.

Both loops solve the same inverse problem against a response matrix measured
during commissioning, truncated to the well-conditioned singular values so that a
single bad monitor cannot dominate the solution. If a monitor reads outside its
plausibility window the loop drops it and continues degraded rather than stopping.

The loops are the reason a beamline sees a stable source point for hours at a
time; when they are off, the source point wanders by tens of microns over the
course of a fill as the ring reaches thermal equilibrium.
