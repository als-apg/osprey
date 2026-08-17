---
type: device
title: Sextupole Magnets
description: Nonlinear magnets that set the ring's momentum-dependent optics.
tags: [magnets, optics, nonlinear]
---

# Sextupole Magnets

A sextupole has six poles and a field that grows quadratically with distance from
the axis. Its purpose is to cancel the energy dependence of quadrupole focusing:
off-momentum electrons are focused too strongly or too weakly by the lattice, and
the sextupoles restore them to the design working point.

Two families do the bulk of that job, one in each plane, and their strengths are
what an operator changes when the measured tune shifts with beam energy. Additional
harmonic families are set from the lattice model and are rarely touched, since they
control the dynamic aperture rather than the linear optics.

Sextupole strength is quoted as an integrated normalized coefficient rather than a
current, and the conversion table lives with the lattice deck. Setting them too
strong shrinks the momentum acceptance and shortens how long the beam survives;
setting them too weak lets head-tail motion grow until the beam is unstable.
