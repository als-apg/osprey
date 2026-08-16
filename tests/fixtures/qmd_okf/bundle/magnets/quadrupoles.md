---
type: device
title: Quadrupole Magnets
description: Gradient magnets that focus the beam in one plane and defocus in the other.
tags: [magnets, optics, tune]
---

# Quadrupole Magnets

A quadrupole has four poles and a field that grows linearly with distance from the
axis, so it acts as a lens: converging in one transverse plane and diverging in
the other. Alternating converging and diverging lenses around the ring produces
net confinement in both planes.

Quadrupoles are grouped into families that share a power supply, and the family
currents are what operators actually adjust when they move the betatron tunes onto
the working point. A per-magnet trim winding allows small individual adjustments
for beta-beat compensation.

Family currents are re-optimized after any chromaticity correction, because the
sextupole change shifts the measured tunes slightly and the working point has to
be restored before user operation resumes.

Ramping is rate-limited in the supply firmware; a step request larger than the
limit is stretched over several seconds rather than rejected.
