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
surrounded by dead addresses. A second shipped engine, ``injection``, needs no
deck: it steps an injection chain through each shot a client requests, so
software that fills the machine has an injector to ask (`Time-driven models`_).
Served in process
(``control_system.connector.virtual_accelerator.serving: in_process``), the
same simulator reads the same view through the same composite, with no
container and no network.

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
``<code>:SIM:<model>:STATUS`` address for each served physics model, then the
view's ``ioc_health`` list: read-only ``<code>:SIM:<model>:HEARTBEAT``,
``UPTIME`` and ``ENGINEER`` addresses for every served model, texture
included. Which models are served is the view's ``served_models.json``, with
the texture model always last. A physics model is one child of the composite,
built from the view's ``variables.json`` and its deck,
``decks/<model>.<its suffix>``, through the engine plug-in its record names;
the texture serves every channel no model wires. The container's ready line
prints how many channels it serves, so the count is read off the running
service rather than kept in prose that would rot.

The entrypoint, the runner and the model surface read the view through
``osprey_connectors.simulation.view.SimulatorView``, the one reader every
consumer of a view opens it with. Each wiring record of ``variables.json``
states what its engine's ``describe()`` said of it at build time, written
into the facility file:

- its ``engine_role``: ``setpoint``, ``readback``, ``monitor`` or ``output``;
- the transverse ``plane`` it steers or reads: ``x``, ``y`` or none;
- its ``refresh``: ``pass``, a value each write's solve updates;
  ``periodic``, one only the periodic solve publishes; ``fast``, one read at
  the fast period, ``simulation.fast_tick_s`` (default ``0.1`` seconds); or
  ``event``, one that changes with each event the engine raises, such as a
  shot.

Each channel also names the node it is ``on``.

The server publishes channel values in *passes*: one after each write, one at
start-up, the periodic passes, and the timed passes of a view holding a
``fast`` or ``event`` channel:

- **What a write's pass reads.** It reads what the write can change, and
  never a channel whose binding refreshes ``periodic``; the next periodic pass
  publishes those. A physics setting reads itself, its readback and what its
  model computes from its settings. A setting no model binds reads itself,
  the readback it pairs with and the channels derived from it. A write to any
  other channel reads every served channel.
- **What a write's pass posts.** Of what it reads, it posts the written
  channels and those whose value or undefined condition changed. An unchanged
  channel keeps the timestamp of its last post.
- **What the periodic pass does.** The start-up pass posts every channel. A
  periodic pass reads every channel and posts those whose value or undefined
  condition changed since their last post, and the channels of a write
  refused since, which clears the alarm that write raised.
- **What a timed pass does.** A timed pass runs at the earlier of the next
  fast period and the composite's next deadline, such as a shot's next step.
  It advances the composite and posts the ``event`` channels that fired and,
  when the fast period fell due, the ``fast`` channels. Each raises a Channel
  Access event even at a repeated value, unless a ``frozen`` or
  ``disconnected`` fault holds it. A view with no ``fast`` or ``event``
  channel runs no timed pass.

Timed passes take their turn with the others, so one waits while a write's or
a periodic pass's solve runs: the fast period and a shot's step offsets hold
between solves, not through one. A long post holds them back less: a pass
posting more than 2000 channels posts them in slices and runs a due timed
pass between two. The ``event`` channels one advance fired go together in the
first slice, however many, so no timed pass posts a later step between two of
them.

A view an older OSPREY wrote is refused before the ready line, with a message
asking for ``osprey build``. The one exception is a view of the previous
schema, ``osprey.facility.simulator/3``, which loads with no model inputs.

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

Time-driven models
==================

A lattice model changes only when it is solved. An injector does not wait for
a solve: a shot plays out over a fraction of a second, and the software
watching it expects each reading to step on time. A model whose engine states
the time contract (see :ref:`extending-lume-model`) is therefore
*time-driven*: a state machine with no thread or clock of its own, which the
composite advances.

Served, the composite advances each time-driven model at its build, rebuild
and reset and before every read and write, and hands it the client writes
that reach it. Only a client write starts what a write triggers; a restore, a
scenario's start state and a write a ``stuck`` or ``disconnected`` setpoint
holds start nothing. A rebuild, which a scenario change makes, discards what
a model had pending. In process nothing advances the model: it serves its
start values and a write starts nothing.

The shipped ``injection`` engine (``osprey.simulation.engines.injection``) is
time-driven. A model naming it names no deck: its wiring
records and settings are the whole chain. A client write of its request
starts a shot, which steps each reading at its own offset; those readings and
the free-running cycle signal refresh ``event``, the fast copy of the stored
current refreshes ``fast``, and the setpoints and the stored current refresh
``pass``. It is the only source of shots. Declaring a chain is in
:ref:`va-injection-chain`.

A ``pyat`` ring whose settings state a ``turn_by_turn`` block is time-driven
too. It reads a chain's shot counter, charge and stored current as model
inputs; a rise of the counter outside a baseline is a shot, which the next
advance tracks through the deck for the monitors a client write armed. The
arrays and turns tracked are emitted in one advance and ``armed`` drops in a
later one, so a client sees the arrays before ``armed`` reads ``0``. Its
outputs refresh ``event`` and its settings ``pass``. Tracking runs inside the
advance and holds the serving loop for its length, a read answering from the
last post meanwhile. Such a ring without a closed orbit, as with its RF
off, keeps serving with its orbit readings NaN, where any other periodic ring
refuses the write. Wiring a ring is in :ref:`va-turn-by-turn`.

Model inputs
============

A model may read records another model wires: its *inputs*, declared on the
model record (see :ref:`architecture-simulation-models`). An input is a value
handed over, not a shared channel: the producer serves the address, and the
consumer's engine receives the producer's plain value through
``take_inputs``.

- **Order.** The composite orders its models so every producer comes before
  its consumers, the rest in name order. A facility without inputs keeps name
  order and makes no delivery.
- **Delivery.** Served, a consumer gets its inputs immediately before each
  advance of it, so it sees what its producers reached on that same pass. A
  client write to a producer reaches its consumers on the next pass; there is
  no separate delivery after a write.
- **Baseline.** A delivery is a *baseline* when it is the first after the
  consumer or one of its producers was built, rebuilt or reset; a scenario
  change rebuilds every model, so it is one too. A baseline says "start from
  here": the engine compares nothing against what it held, so a step in an
  input across a rebuild triggers nothing.
- **A failed producer.** Its inputs are left out of the delivery; when none
  remain, no call is made. A ``take_inputs`` that raises fails the consumer.

Only the served Virtual Accelerator advances time. In process the composite
does not evolve: it delivers at build, at reset and after every write, each
delivery a baseline, so an input never starts anything there.

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
     - ``0.1.6``
     - The serving stack ``runner.py`` subclasses --- ``pcaspy`` for Channel
       Access, ``p4p`` for PVAccess.

       Releases 0.1.3 to 0.1.6 add the features OSPREY relies on, listed
       below the table.

OSPREY relies on these ``lume-pva-apg`` features, by the release that added
them:

- **0.1.3** adds the two hooks the model surface is built on:
  ``_enqueue(..., jobs=)``, which runs a callable on the run loop's own thread
  batched with the writes already queued there, and ``_cycle_output_names()``,
  which lets a subclass narrow the set of variables re-read after each cycle.
- **0.1.4** makes the model info the server announces follow the
  configuration it was given, so a variable held back from the channel
  namespace is not advertised as one.
- **0.1.5** takes each variable's display precision and description from the
  runner configuration, raises an alarm on an output the model names as
  failed, serves integer variables as integers on PVAccess, and exposes a
  public tick period whose ticks coalesce.
- **0.1.6** adds the forced-event hook, through which a repeated value on a
  ``fast`` or ``event`` channel reaches a Channel Access monitor; lets the
  runner narrow a pass's rollback snapshot; and raises the Channel Access
  events of a post at the cost of the channels posted, not of every channel
  served.

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
``decks/<model>.<its suffix>``, through the engine plug-in the model's record names:
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
``last_cycle_ms``, ``queue_depth``, ``uptime_s``, ``last_refused_write``,
``last_failed_pass``, which is ``null`` or the ``error`` and ``uptime_s`` of
the latest publishing pass that failed, and ``health``, the server's health
record.
``set`` writes model variables only: a served address, a value that is not
finite and a name the composite refuses are each refused. ``reset`` writes
every drifted writable model variable back to the value it held when its model
was built at the active scenarios, and returns the names it reset.

The health record counts every publishing pass. Its ``state`` is ``serving``
when the latest pass succeeded, ``degraded`` while up to
``failed_pass_tolerance`` passes in a row have failed (three by default), and
``failed`` once more than that have; a pass that succeeds returns it to
``serving``. It also carries the latest pass's outcome, ``uptime_s`` and
wall-clock ``at``, the ``passes_ok`` and ``passes_failed`` counters, and the
latest failed pass. The runner rewrites the record atomically to
``/run/osprey-va/health.json`` inside the container after every pass, and the
container's compose healthcheck reads that file: it passes while the record is
``serving`` or ``degraded`` and fails on ``failed``, a missing file or one that
is not JSON. The ready line the container prints at boot reports the first
pass only.

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
element is driven by two setpoints, each is miscalibrated alone. A monitor's
roll turns the beam position in the deck's unit, before the monitor's
conversion to the unit it publishes, and its offset, gain, noise and polarity
then act on the published reading.

Four read-only arrays sit beside them: ``tunes``, ``chromaticity``,
``beta_at_monitors`` and ``orbit_at_monitors``, the last two one row per
monitor element in lattice order. A ``single_pass`` model has no tunes, so it
declares neither of the first two. They are computed when one of them is read
and served from memory until the next solve, so a setpoint write never pays
for them.

The imperfections a machine starts from are scenario faults. A scenario under
``data/facility/scenarios/`` names them in its ``faults`` block, keyed by model
and then by address, each a value or a map of fault fields to values; the
composite seeds them into its models when the scenario is active. Channel
faults, named in the scenario's ``channel_faults`` block, are faults of the
composite's channel layer rather than of a model, and act on wired and
texture channels alike. A ``stuck`` setpoint accepts a write, reads the value
written, and forwards none of it; its readback shows the machine where it
was. A ``frozen`` reading serves the value it served when the scenario became
active. A ``disconnected`` channel holds writes as ``stuck`` does, and its
reading serves not-a-number, or for a non-float the value it served when the
scenario became active, with the ``udf`` condition.

Element errors are the magnets' own imperfections. One series supply address
feeds many magnets, so these are keyed by the served deck element's name
rather than by an address: ``<element>/dx``, ``<element>/dy`` and
``<element>/roll`` on every element a setpoint or a non-monitor readback binds,
and ``<element>/dB<n>`` and ``<element>/dA<n>`` (``n`` from 0 to 9, the index
into ``PolynomB`` or ``PolynomA``) on every element bound on a polynomial. A
scenario writes them under ``faults.<model>`` like any engine-named fault,
either as ``{QF_01_1/dx: 1.0e-4}`` or as ``{QF_01_1: {dx: 1.0e-4, dB2: 0.3}}``.
Each takes an explicit value; ``dx`` and ``dy`` are bounded to 1 cm and
``roll`` to 0.1 rad, and a multipole delta must be finite. The engine draws
nothing at random: a random realisation of errors is the scenario author's to
draw and write down, so the same scenario always boots the same machine.

A magnet no wiring record binds by ``element`` carries none of these on its
own: a bend whose only channel is wired to the deck's ``energy``, say, or a
magnet with no channel at all. The model's ``pyat`` settings declare such an
element under ``error_elements``:

.. code-block:: yaml

   settings:
     pyat:
       solve: periodic
       error_elements:
       - element: BEND_1_1
         multipoles: true
       - element: SEPTUM

A declared element carries ``dx``, ``dy`` and ``roll``, and with ``multipoles:
true`` also ``dB<n>`` and ``dA<n>``, with the bounds and the seeding of a bound
element. ``multipoles`` is false when left out, and may be true only on an
element that has both ``PolynomB`` and ``PolynomA``. Declaring an element the
wiring already binds is allowed and can only add slots: it carries the union of
the two. The build stops, naming the element, when a declared element is not in
the deck or is in it more than once, and when the list names an element twice.
A monitor's element takes readout faults instead, so declaring one is refused,
with the same words by the scenario check of ``osprey build`` and when the
model is built.

The errors are seeded once, when the model is built. A misalignment becomes
the element's entrance and exit transforms. A multipole delta is added to its
coefficient and held there: a setpoint write on the same coefficient moves the
nominal strength and keeps the delta, and the setpoint reads back what was
written. Systematic multipoles are facts of the deck, written into the lattice
file, not scenario faults. A scenario whose errors leave no closed orbit does
not boot quietly wrong: its model fails with the solve error, which names the
elements seeded, and the model's status names the scenario.

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
