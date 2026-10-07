.. _architecture-virtual-accelerator:

===================
Virtual Accelerator
===================

The Virtual Accelerator is a single container that puts a whole facility on
real EPICS. One process serves the facility's entire channel namespace from the
simulator view ``osprey build`` writes. Behind the channels a facility wires to
a deck sits a physics model built through the `LUME
<https://www.lume.science/>`_ model interface: the shipped pyat engine wraps a
pyAT lattice as a ``LUMEModel`` via ``lume-pyat``, so that writing a corrector
moves the orbit that the BPMs report. The texture model serves every channel no
model wires, so a client sees one machine rather than a physics island
surrounded by dead addresses. The ``mock`` connector serves the same view
in-process.

This page is about how those pieces fit. Running one, and the
``control_system.type`` switch that selects it, are in
:doc:`/how-to/control-systems/use-virtual-accelerator`.

The layer map
=============

The service lives at ``src/osprey/services/virtual_accelerator/``:

.. raw:: html
   :file: ../_diagrams/va-layer-map.html

``entrypoint.py`` builds one composite over the simulator view and hands it to
``ModelRunner`` in ``serving/``, which serves it on both transports. Under the
composite, each physics model is an engine plug-in reached only through its
``set()`` and ``get()``; that boundary is what makes the physics replaceable
(see `Bringing your own model`_).

What gets served
================

The served list is the simulator view's ``addresses.json``: its ``channels``,
every channel address the facility declares, then one
``<code>:SIM:<model>:STATUS`` address for each served physics model. Which
models are served is the view's ``served_models.json``, with the texture model
always last. A physics model is one child of the composite, built from the
view's ``variables.json`` and its deck, ``decks/<model>.json``, through the
engine plug-in its record names; the texture serves every channel no model
wires. The container's ready line prints how many channels it serves, so the
count is read off the running service rather than kept in prose that would
rot.

Two transports, one write path
==============================

One process serves both protocols. **Channel Access carries the whole
namespace and is the authoritative view**; PVAccess additionally serves the
model's own variables natively. Every setpoint write from either transport
passes the same drive-limit clamp and physics hand-off, and is committed on both views — a write on either transport moves
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

Which models are served is the view's ``served_models.json``, written from the
profile's ``simulation.models``; the texture model is always served, last. A
render whose served set is ``texture`` alone boots the same service with no
physics model built. It serves the same channels, without the status address a
physics model adds: a setpoint latches its written value and a readback
follows its seed. That is what makes the service usable for a facility that has
a channel list but no model behind it yet.

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
     - ``0.2.0``
     - The facility-agnostic pyAT backend --- one persistent lattice, atomic
       multi-variable writes, one solve per batch, rollback on a lost closed
       orbit.
   * - ``lume-pva-apg[ca,pva]``
     - ``0.1.5``
     - The serving stack ``runner.py`` subclasses --- ``pcaspy`` for Channel
       Access, ``p4p`` for PVAccess. 0.1.3 adds the two hooks the model
       surface is built on: ``_enqueue(..., jobs=)``, which runs a callable on
       the run loop's own thread batched with the writes already queued there,
       and ``_cycle_output_names()``, which lets a subclass narrow the set of
       variables re-read after each cycle. 0.1.4 makes the model info the
       server announces follow the configuration it was given, so a variable
       held back from the channel namespace is not advertised as one. 0.1.5
       takes each variable's display precision and description from the
       runner configuration, raises an alarm on an output the model names as
       failed, serves integer variables as integers on PVAccess, and exposes a
       public tick period whose ticks coalesce.

``pcaspy`` publishes wheels for linux-x86_64 only, so the ``lume-pva-apg``
entry that brings it is marked accordingly, and ``serving/runner.py`` alone is
unimportable off that platform; it is reached lazily, and the rest of ``serving/`` imports anywhere.
The live Channel Access suites run in the container venue under
``scripts/va/live_ca/``.

What OSPREY ships on top of ``lume-pyat`` is the *facility adapter*, not the
pyAT machinery: the pyat engine plug-in (``osprey.simulation.engines.pyat``,
with ``variable_from_wiring`` in ``pyat_variables``) supplies only the facts
upstream cannot know — which deck to build, how a hardware value becomes a
strength through its record's calibration, which variables exist, and how a
failed build reads on the model's ``STATUS`` channel.

Bringing your own model
=======================

The runner serves one composite, and the composite builds one child per
served physics model from the view's ``variables.json`` and the model's deck,
``decks/<model>.json``, through the engine plug-in the model's record names:
its ``engine``, looked up in the ``osprey.simulation.engines`` entry-point
group. Nothing in the entrypoint, a profile or ``.env`` names an engine.

**A pyAT deck is data.** Any facility's pyAT deck is served by the shipped
``pyat`` engine. The facility tree stages it beside the model record that names
it and the wiring that ties channels to its elements, and ``osprey build``
copies it byte for byte into the view. How a tree comes to carry one is in
:doc:`/how-to/control-systems/use-virtual-accelerator`.

**Any other backend is a plug-in.** A surrogate, Cheetah or Bmad is a different
``LUMEModel``, so it arrives as an engine module registered under that group,
whose ``build(model, wiring, deck, settings, active=...)`` returns it. The image
the facility builds installs the plug-in beside OSPREY; the entrypoint stays
the shipped one.

Either way the serving layer does not change. Each variable is named for the
channel address its wiring record claims before the engine sees it, so a
backend parses no channel names. The extension seam is under
:ref:`extending-lume-model`.

Served and model-only variables
-------------------------------

A model declares more variables than the facility serves as channels. The line
between the two is drawn by name alone: a variable is **served** when its name
is an address of ``addresses.json``, and **model-only** when it is not. Served
variables are read and written like any other channel, on either transport.
Model-only variables sit on no channel at all; the model RPC is the only way to
reach them, as ``<model>/<name>``.

Where the line falls here is not a judgement call. The pyat engine names every
variable it builds from a wiring record for the channel address that record
claims, so the served half is exactly the model's wired addresses. Everything
on the model-only half is something the model declares beside them: the faults
every wired monitor and magnet can carry, and the optics of the whole solved
lattice.

That RPC is one pvAccess channel, ``model_rpc``, and it takes six verbs.
``info`` lists every served address with ``surface: served``, then each built
model's own variables as ``<model>/<name>`` with ``surface: model``, each with
its name, unit, value range and whether it is read-only. ``get`` reads served
addresses from the composite and model variables from their model. ``diff``
puts the value the control system serves beside the composite's own for every
channel of ``addresses.json``, the composite's read without motion or readout.
``status`` reports on the server itself: ``instance``, ``endpoint``,
``last_cycle_ms``, ``queue_depth``, ``uptime_s``, ``last_refused_write`` and
``last_failed_pass``, which is ``null`` or the ``error`` and ``uptime_s`` of
the latest publishing pass that failed.
``set`` writes model variables only: a served address, a value that is not
finite and a name the composite refuses are each refused. ``reset`` writes
every drifted writable model variable back to the value it held when its model
was built at the active scenarios, and returns the names it reset.

Those two write verbs ask for an administrative token. The container reads it
from ``VA_MODEL_WRITE_TOKEN`` at startup and compares what a caller presents
against it; a call carrying no token, or a token that does not match, is
refused before any model is touched, and a container started without the
variable set refuses model writes outright. The read verbs are not gated.

The model-only roster is the wiring's own. Each fault is named
``<address>/<field>`` after the wired address it perturbs, so no fault name is
a channel address. A monitor reading carries ``offset``, ``gain``, ``noise``
and ``polarity`` on its own axis, and a monitor one ``roll``, on its x-axis
reading or its only one. A setpoint on ``PolynomB`` or ``KickAngle`` carries
``cal_factor`` and ``cal_offset``; a calibration is the supply's, so where one
element is driven by two setpoints, each is miscalibrated alone.

Four read-only arrays sit beside them: ``tunes``, ``chromaticity``,
``beta_at_monitors`` and ``orbit_at_monitors``, the last two one row per
monitor element in lattice order. A ``single_pass`` model has no tunes, so it
declares neither of the first two. They are computed when one of them is read
and served from memory until the next solve, so a setpoint write never pays
for them.

The imperfections a machine starts from are scenario faults. A scenario under
``data/facility/scenarios/`` names them in its ``faults`` block, keyed by model
and then by address, each a value, ``stuck``, or a map of fault fields to
values; the composite seeds them into its models when the scenario is
active. ``stuck`` is a fault of the composite's write path rather than of a
model: a stuck setpoint accepts a write, reads the value written, and forwards
none of it; its readback shows the model where it was.

A readout fault never moves the orbit; it changes what a monitor reports.
``diff`` is where it becomes visible, as a served reading that has parted from
the composite's own value.

.. seealso::

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      Running the Virtual Accelerator, and the ``control_system.type`` switch.

   :doc:`/contributing/extending-osprey`
      The extension seams, including the LUME model seam.

   :doc:`/how-to/control-systems/use-connectors`
      How the EPICS connector reaches a control system, virtual or real.
