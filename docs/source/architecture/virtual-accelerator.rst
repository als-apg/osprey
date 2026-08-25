.. _architecture-virtual-accelerator:

===================
Virtual Accelerator
===================

The Virtual Accelerator is a single container that puts a whole facility on
real EPICS. One process serves the facility's entire channel namespace, with
the physics behind the storage-ring channels supplied through the `LUME
<https://www.lume.science/>`_ model interface: a pyAT lattice, wrapped as a
``LUMEModel`` via ``lume-pyat``, sits behind the magnet and BPM channels so
that writing a corrector moves the orbit that the BPMs report. Everything else
on the namespace is composed by the same simulation engine the ``mock``
connector uses, so a client sees one machine rather than a physics island
surrounded by dead addresses.

This page is about how those pieces fit. Running one, and the
``control_system.type`` switch that selects it, are in
:doc:`/how-to/control-systems/use-virtual-accelerator`.

The layer map
=============

The service lives at ``src/osprey/services/virtual_accelerator/``:

.. raw:: html
   :file: ../_diagrams/va-layer-map.html

``serving/`` is typed against ``lume.model.LUMEModel`` and never imports
``ioc/`` or ``model/``; ``entrypoint.py`` joins the halves. That boundary under
``model/`` is the one that matters: everything downstream reaches the ring only
through a ``LUMEModel``'s public ``set()`` and ``get()``, which is what makes
the physics replaceable (see `Bringing your own model`_).

What gets served
================

``manifest/`` derives the served channel set from the facility's channel
databases rather than listing it, and classifies every address into one of
three physics-fidelity partitions:

**pyat-coupled**
   Storage-ring magnet currents and BPM positions. A write re-solves the
   closed orbit and pushes the new BPM readings before its completion is
   signalled, so a readback taken after a completed write is already the new
   orbit.

**sp-echo**
   Booster and transfer-line magnets, RF and vacuum setpoints. The setpoint
   echoes onto its readback immediately, with no physics behind it.

**static-noisy**
   Everything else, driven by the in-image simulation engine from the mounted
   ``machine.json`` — with the ``mock`` connector's synthesis as the fallback,
   so the two backends never disagree about a channel neither has data for.

The authoritative channel count lives in the manifest's
``_metadata.total_channels`` — a few thousand addresses — rather than in prose
that would rot.

Two transports, one write path
==============================

One process serves both protocols. **Channel Access carries the whole
namespace and is the authoritative view**; PVAccess additionally serves the
model's own variables natively. Every setpoint write from either transport
enters the same ``write_path``, passes the same drive-limit clamp and physics
hand-off, and is committed on both views — a write on either transport moves
both, a refused write moves neither. Only the completion differs, forced by
the protocols: CA put-completion carries no status, so a refusal withholds the
echo and raises an alarm; a PVAccess put completes with the model's error
string. Only the Channel Access port (``5064/tcp``) is published from the
container.

Physics is optional
===================

``VA_LATTICE=none`` boots the same service with no lattice: pyAT is never
imported and the served model is the empty ``NullModel``. The Channel Access
namespace is *identical* to a lattice-backed boot; the only difference is that
a pyat-coupled setpoint simply latches its written value. That is what makes
the service usable for a facility that has a channel list but no model behind
it yet.

The LUME stack and its pins
===========================

Three young upstream packages are pinned exactly, because their surfaces are
still settling:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Package
     - Pin
     - What it contributes
   * - ``lume-base``
     - ``0.5.0``
     - The generic model contract: ``LUMEModel``, ``ScalarVariable``.
   * - ``lume-pyat``
     - ``0.1.0``
     - The facility-agnostic pyAT backend --- one persistent lattice, atomic
       multi-variable writes, one solve per batch, rollback on a lost closed
       orbit.
   * - ``lume-pva-apg[ca,pva]``
     - ``0.1.2``
     - The serving stack ``runner.py`` subclasses --- ``pcaspy`` for Channel
       Access, ``p4p`` for PVAccess.

``lume-pva-apg`` and ``pcaspy`` publish wheels for linux-x86_64 only and are
marked accordingly, so ``serving/runner.py`` alone is unimportable off that
platform; it is reached lazily, and the rest of ``serving/`` imports anywhere.
The live Channel Access suites run in the container venue under
``scripts/va/live_ca/``.

What OSPREY ships on top of ``lume-pyat`` is the *facility adapter*, not the
pyAT machinery: ``PyATRingModel`` supplies only the facts upstream cannot know
— which lattice to build, how a commanded current becomes a magnet strength,
which variables exist, and how a boot failure should read.

Bringing your own model
=======================

Because that one boundary is the only way to the ring, replacing the physics
replaces one object: a different backend — a surrogate, Cheetah, Bmad, or
another facility's pyAT ring — is injected through ``model=`` without the
serving layer changing. Model variables are keyed by their full channel
address and resolved before the backend sees them, so a backend parses no
channel names. The seam, its floor (``NullModel``) and its ceiling are written
up under :ref:`extending-lume-model`.

.. seealso::

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      Running the Virtual Accelerator, and the ``control_system.type`` switch.

   :doc:`/contributing/extending-osprey`
      The extension seams, including the LUME model seam.

   :doc:`/how-to/control-systems/use-connectors`
      How the EPICS connector reaches a control system, virtual or real.
