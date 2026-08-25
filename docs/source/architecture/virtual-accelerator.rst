.. _architecture-virtual-accelerator:

===================
Virtual Accelerator
===================

The Virtual Accelerator is a single container that puts a whole facility on
real EPICS. One process serves the facility's entire channel namespace, and a
pyAT lattice sits behind the storage-ring magnet and BPM channels so that
writing a corrector moves the orbit that the BPMs report. Everything else on
the namespace is composed by the same simulation engine the ``mock`` connector
uses, so a client sees one machine rather than a physics island surrounded by
dead addresses.

This page is about how those pieces fit and why the seams fall where they do.
Running one, and the ``control_system.type`` switch that selects it, are in
:doc:`/how-to/control-systems/use-virtual-accelerator`.

The layer map
=============

The service lives at ``src/osprey/services/virtual_accelerator/``. Values flow
down this map; the import edges do not all follow it:

.. code-block:: text

   manifest/            which channels exist, and what each one is
        |               (generated from the facility databases, never hand-listed)
        v
   lattice/             the ring itself: inventory, calibration, orbit solve
        |
        v
   model/               PyATRingModel -- a LUMEModel, via lume-pyat
        |
        v
   ioc/physics_bridge   setpoint -> calibration -> solve -> BPM readings
        |
        |                bound by entrypoint.py, the composition root
        v
   serving/             pvdb + write_path (no server library)
                        + runner (over lume-pva-apg)
        |
        +--> Channel Access (the whole namespace, authoritative)
        +--> PVAccess      (the model's own variables, natively)

``serving/`` imports nothing from ``ioc/`` or ``model/``: it is typed against
``lume.model.LUMEModel`` and reads the manifest directly. What joins the two
halves is ``entrypoint.py``, which builds the physics bridge, binds it to the
serving records so BPM readings reach their channels, and hands
``PhysicsBridge.on_setpoint`` to the runner.

The boundary that matters most is the one under ``model/``: everything
downstream of it --- the physics bridge and the whole serving layer --- reaches
the ring only through the public ``set()`` and ``get()`` of a ``LUMEModel``.
What that buys is the subject of `Bringing your own model`_.

What gets served
================

``manifest/`` derives the served channel set rather than listing it. It expands
the tutorial's channel-finder databases, checks that the interchangeable
paradigm formats agree, unions in the channels the scenario seed
``machine.json`` names, and classifies every address into one of three
physics-fidelity partitions:

**pyat-coupled**
   Storage-ring magnet currents and BPM positions. A write is handed to the
   run loop, which applies it to the lattice, re-solves the closed orbit and
   pushes the new BPM readings --- all before the write's completion is
   signalled back to the client. So a readback taken after a completed write
   is already the new orbit, and never waits on a polling tick.

**sp-echo**
   Booster and transfer-line magnets, RF and vacuum setpoints. Writing the
   setpoint echoes onto its readback immediately, with no physics behind it.

**static-noisy**
   Everything else --- reference values, status flags, temperatures,
   pressures --- driven by the in-image simulation engine
   (``ioc/engine_source.py``) from the mounted ``machine.json``. Channels the
   engine does not define fall back to the same synthesis the ``mock``
   connector uses, so the two backends never disagree about a channel for
   which neither has real data.

The authoritative channel count lives in
``src/osprey/services/virtual_accelerator/manifest/channel_manifest.json``
under ``_metadata.total_channels`` --- a few thousand addresses --- rather than
being repeated in prose that would rot.

Two transports, one write path
==============================

``serving/runner.py`` serves both protocols from one process
(``DEFAULT_PROTOCOLS`` is ``("ca", "pva")``). The split is deliberate:

* **Channel Access carries the whole namespace and is the authoritative view
  of the machine.** It is what the EPICS connector reads and writes, and every
  CA process variable comes from the manifest-derived database --- including
  the addresses that also happen to be model variables, so no channel is ever
  served twice from two different specifications.
* **PVAccess carries the model's own variables, natively**, alongside the
  ``model_info`` structure that describes them.

Keeping the two views of one address in step is the write path's job, never a
model read's. Every setpoint write from either transport enters
``osprey.services.virtual_accelerator.serving.write_path``, passes the same
drive-limit clamp and the same physics hand-off, and every value it publishes
is committed on both views. A write arriving on either transport moves both; a
refused write moves neither.

The one place they differ is how a client learns the outcome, and there the
difference is forced by the protocols. Channel Access put-completion carries no
status --- it can only report success --- so a refusal is expressed by
withholding the echo and raising an alarm on the setpoint. A PVAccess put is
completed with the model's own error string. Neither transport ever records a
value the model did not take.

Only the Channel Access port is published from the container: the
``Containerfile`` exposes ``5064/tcp`` and nothing else, so PVAccess is
reachable inside the container only. The published port and the port the server
binds must be the same number, because a CA search reply carries the server's
own port back to the client.

Two further contracts hold the process together. The model is **not
thread-safe**, so every model access happens on the run loop's thread: a write
arriving on the server thread is enqueued and completed later from the loop.
And the **boot order is a contract** --- the serving database is built and every
boot value pushed into it before the runner exists, because the Channel Access
server copies each specification when it creates the process variable, and a
value written afterwards is never served.

Physics is optional
===================

``VA_LATTICE=none`` boots the same service with no lattice at all. pyAT is
never imported; the served model is the empty ``NullModel`` in
``serving/model_stub.py``, which describes no variables, so no PVAccess
variable channel is created. The Channel Access namespace is *identical* to a
lattice-backed boot --- same addresses, same count, same boot values --- and the
difference is confined to what happens after a write: a pyat-coupled setpoint
with no physics behind it simply latches the value written to it. That path is
what makes the service usable for a facility that has a channel list but no
model to put behind it yet.

The LUME stack and its pins
===========================

Three young upstream packages are pinned exactly, because their surfaces are
still settling and the code here is written against a specific contract:

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
     - The serving stack ``runner.py`` subclasses. ``[ca]`` brings ``pcaspy``
       for Channel Access, ``[pva]`` brings ``p4p`` --- both are required,
       because the value layer is ``p4p``-typed even on the CA side.

``lume-pva-apg`` and ``pcaspy`` carry the environment marker
``sys_platform == 'linux' and platform_machine == 'x86_64'``: those wheels are
published for linux-x86_64 only, and an unmarked requirement would push a
developer's Mac into an EPICS source build. The consequence is that
``serving/runner.py`` alone is unimportable off that platform. It is the one
module of the serving layer that imports a server library, and it is reached
lazily rather than re-exported eagerly, so the rest of ``serving/`` --- the
database, the write path, the null model --- imports anywhere, which is what
keeps the no-lattice boot path cheap. The live Channel Access suites are the
ones that skip off-platform, and they run in the container venue under
``scripts/va/live_ca/`` instead. ``pyproject.toml``'s
``exclude-newer-package`` entries hold ``lume-pyat`` and ``lume-pva-apg`` at
the resolution date the pins were taken.

What OSPREY ships on top of ``lume-pyat`` is the *facility adapter*, not the
pyAT machinery: ``osprey.services.virtual_accelerator.model.pyat.PyATRingModel``
subclasses the upstream model and supplies only the four facts upstream cannot
know --- which lattice to build, how a commanded current becomes a magnet
strength, which variables exist and what each is bound to, and how a boot
failure should read. Unit conversion in particular stays on this side: the
upstream writable variable writes the value it is handed, unconverted, and
amps-to-strength is a facility subclass.

Bringing your own model
=======================

Because that one boundary is the only way to the ring, replacing the physics
replaces one object and nothing else: a different backend --- a surrogate,
Cheetah, Bmad, or another facility's pyAT ring --- is injected through
``model=`` without the serving layer changing. Model variables are keyed by
their full channel address, and the binding layer resolves each address to an
element locator before the backend sees it, so a backend parses no channel
names.
The seam, its floor (``NullModel``) and its ceiling (a shipped ring model
wrapped so setpoint writes carry a calibration) are written up under
:ref:`extending-lume-model`.

.. seealso::

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      Running the Virtual Accelerator, and the ``control_system.type`` switch.

   :doc:`/contributing/extending-osprey`
      The extension seams, including the LUME model seam.

   :doc:`/how-to/control-systems/use-connectors`
      How the EPICS connector reaches a control system, virtual or real.
