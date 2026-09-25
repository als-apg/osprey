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
string. The ``virtual_accelerator`` instance publishes two ports from its
container: Channel Access (``5064/tcp``) and pvAccess (``5075/tcp`` unless
``virtual_accelerator.pva_port`` names another), the second of them for the
model surface described below. A pvAccess client reaches that port by name
server (``EPICS_PVA_NAME_SERVERS=<host>:<port>``),
which is TCP, so TCP is all that is published — there is no UDP search to
answer. A stand-in instance publishes its Channel Access port alone: the model
surface belongs to the ``virtual_accelerator`` instance.

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
     - ``0.1.4``
     - The serving stack ``runner.py`` subclasses --- ``pcaspy`` for Channel
       Access, ``p4p`` for PVAccess. 0.1.3 adds the two hooks the model
       surface is built on: ``_enqueue(..., jobs=)``, which runs a callable on
       the run loop's own thread batched with the writes already queued there,
       and ``_cycle_output_names()``, which lets a subclass narrow the set of
       variables re-read after each cycle. 0.1.4 makes the model info the
       server announces follow the configuration it was given, so a variable
       held back from the channel namespace is not advertised as one.

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

Served and model-only variables
-------------------------------

A model declares more variables than the facility serves as channels. The line
between the two is drawn once, at boot, and by name alone: a variable is
**served** when its name is an address the channel manifest already serves,
and **model-only** when it is not. Served variables are read and written like
any other channel, on either transport. Model-only variables sit on no channel
at all; the model RPC is the only way to reach them.

Where the line falls here is not a judgement call. Every variable the model
builds from the served tree is named for the channel address its binding
claims, so the served half is exactly the machine's own pyat-coupled
addresses. Everything on the model-only half is something the model or the
serving layer added beside them: the simulated imperfections, the optics of
the whole solved ring, and the set of setpoints currently stuck.

That RPC is one pvAccess channel, ``model_rpc``, and it takes six verbs.
``info`` lists the model's variables and says which side of the line each one
is on; ``get`` reads what the model holds for any of them, served side
included; ``diff`` puts the value the control system serves beside the model's
own for each served variable; ``status`` reports on the server itself. ``set``
writes model-only variables, and ``reset`` returns them to the values they
were seeded with at boot.

Those two write verbs ask for an administrative token. The container reads it
from ``VA_MODEL_WRITE_TOKEN`` at startup and compares what a caller presents
against it; a call carrying no token, or a token that does not match, is
refused before any model is touched, and a container started without the
variable set refuses model writes outright. The read verbs are not gated.

The model-only roster is the served tree's own. Every monitor the bindings
publish a reading for carries nine reading-error fields, and every magnet they
drive a calibration factor and offset; a device the tree does not serve
carries neither. Each is named for the element the deck spells it at, with a
dot between element and field (``<element>.offset_x``,
``<element>.cal_factor``), so no fault name parses as a channel address. A
seed names its device either way the facility knows it — by an address the
control system carries, or by the element it sits at — and a name the served
bindings know under neither ends the boot rather than perturbing nothing.
Where one element is driven by two setpoints, one calibration scales both,
which is what a miscalibrated magnet does.

Three read-only arrays sit beside them: the transverse tunes, and beta and the
true orbit at each monitor, one row per monitor in ring order. They are
computed when one of them is read and served from memory until the next solve,
so a setpoint write never pays for them.

The last model-only variable is the serving layer's, not the model's:
``stuck_setpoints``, the addresses whose readbacks are frozen, written as text
and settable at runtime. It is the one fault a deployment with no lattice
behind it can still take.

Writing a fault changes what the model holds, and the served readings derived
from it are recomputed as the write lands — a magnet whose calibration moved is
re-commanded with the value its operator last asked for, so the ring ends where
the new calibration puts it. ``diff`` is where a readout fault becomes visible,
as a served reading that has parted from the model's own value.

.. seealso::

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      Running the Virtual Accelerator, and the ``control_system.type`` switch.

   :doc:`/contributing/extending-osprey`
      The extension seams, including the LUME model seam.

   :doc:`/how-to/control-systems/use-connectors`
      How the EPICS connector reaches a control system, virtual or real.
