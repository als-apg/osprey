.. _how-to-use-virtual-accelerator:

===========================
Use the Virtual Accelerator
===========================

How to run the Control Assistant tutorial against a **Virtual Accelerator** — a
containerized simulator that serves real EPICS Channel Access, with PyAT physics
behind the channels your facility wires to a lattice, so correctors move and
BPMs respond.
How it is put together is :doc:`/architecture/virtual-accelerator`.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the Virtual Accelerator is (and is not)
   - What ``control_system.type`` selects, and which machine each value names
   - Which channels the simulator serves, and what the build will not invent
   - Pointing a project at the Virtual Accelerator the stack already deploys
   - Moving a running deployment between the machines it describes
   - The simulator's two venues, and why plans go browse-only in process
   - How ``osprey sim apply`` scenarios behave in Virtual Accelerator mode
   - Write limits
   - The stored archive the stack deploys, and the one pairing it refuses

   **Prerequisites:** Docker (or Podman) installed; the Control Assistant
   tutorial project (see :doc:`/getting-started/control-assistant`).

Overview
========

The Control Assistant tutorial ships interchangeable control-system backends,
selected by a single ``control_system.type`` value. The value picks the machine
the deployment **starts** on:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``type``
     - Backend
   * - ``virtual_accelerator`` *(default)*
     - The simulator. Served from its container, it speaks real EPICS Channel
       Access: the magnet setpoints your facility wires to a deck drive a live
       pyAT lattice and BPM readbacks respond, and every other channel is
       composed by the same simulation engine. The tutorial's default, and
       deployed as part of its stack. The same simulator can also run in
       process, with no container; see `Two venues`_.
   * - ``epics``
     - Production EPICS, pointed at the facility gateway. Untouched by this
       guide.
   * - ``live_standin``
     - The **live stand-in** — a second soft IOC the deployment runs for
       itself, served by the EPICS connector from its own connector block.
       Available only where the build profile stood one up; see `Rehearsing
       against a live target`_.

The Virtual Accelerator is a **local physics simulator**, not a digital twin —
it is not synced to any real machine. The OSPREY agent reads and writes it
exactly as it does a real machine; only the backend changes.

The physics behind the channels comes from the profile's
``simulation.models``: the container builds each served physics model through
the engine its record names and serves them together with the texture model as
one composite. The shipped engine is pyAT, built on the facility-agnostic
``lume-pyat`` package, over the deck your facility tree stages. A render that
serves ``texture`` alone serves the same channels with no physics behind them:
a setpoint holds the value written to it and a readback follows its seed. A
physics engine other than pyAT is an engine plug-in, covered in
:ref:`va-serving-your-own-model`.

What channels it serves
=======================

Your project's own. Every ``osprey build`` renders the facility definition
under ``data/facility/`` into the simulator view, ``build/data/simulator/``,
which the compose file mounts at ``/data/simulator/``. The container serves the
``channels`` of the view's ``addresses.json`` --- every channel address the
facility declares --- and one ``<code>:SIM:<model>:STATUS`` channel for each
served physics model; its ready line in ``docker logs`` prints the count.
Nothing is invented and no other channel list feeds the set: a data mount
without the view refuses the boot and names the missing file, and a facility
tree the build cannot render stops in ``osprey facility validate``'s own stops
before the view is written.

A channel's role, not its address, says what it is. A channel record with
``role: setpoint`` is written, and its readback is the record's ``pair``; a
setpoint that names no ``pair`` is its own readback. A setpoint's ``tolerance``,
next to its ``pair``, says how close that readback must come before a move
counts as done; the build stops when the readback's seeded motion is wider
than it. A channel with no role is a readback. The address text is the facility's own and is served as it is
written: no token inside it means anything to OSPREY.

A facility whose models wire no channel is served by the texture alone: a
setpoint holds the value written to it, a readback follows its seed, and no
physics runs behind either.

Quickstart
==========

The Control Assistant stack ships pointed at the Virtual Accelerator and
**already deploys** it: the preset's ``virtual_accelerator:`` block renders a
compose service, so ``osprey up`` brings the Virtual Accelerator up alongside the
rest of the stack and the connector is already talking to it. There is nothing
to switch on.

.. code-block:: bash

   osprey up   # brings up the Virtual Accelerator with the rest of the stack
   osprey web         # the agent talks to real Channel Access

``osprey up`` brings up more than the Virtual Accelerator. Because the preset also
declares a ``va_archiver:`` block, the deploy stands up the machine's **archive**
next to it — a MongoDB store and a recorder service — and seeds it before the
rest of the stack starts. See `The archive`_ below.

The very first ``osprey up`` that includes the Virtual Accelerator
builds its container image (installing the physics and EPICS serving stack), so
expect it to take several minutes — it is building, not hanging. Later deploys
reuse the image.

If your deployment came from a preset or profile that selects a different
connector, point it at the Virtual Accelerator explicitly:

.. code-block:: bash

   osprey set connector=virtual_accelerator
   osprey build
   osprey up

.. note::

   All three steps matter. ``osprey set`` writes the setting into
   ``profile.yml``; ``osprey build`` carries it into ``build/``, where each
   service gets its own copy of the rendered config; ``osprey up`` starts what
   was just rendered. Anything already running in a container keeps the old
   setting until you restart it. No image rebuild is involved.

**The archive has to come first.** On a deployment created from the
``control-assistant`` preset the switch just works: the preset declares where the
archive lives, so the deployment already reads a real store. On one with no
archive of its own — still reading the mock archiver, which makes its history
up as it is asked for it — the build **refuses** the profile, and says what to
do instead: point ``archiver.type`` at a store this deployment writes
(``mongodb_archiver`` for the store the preset deploys), or keep the simulator
in process for an honestly storeless deployment. `The honesty rule`_ below
explains why.

Switching a running deployment
==============================

Those three commands set which control system the deployment **starts** on; on a
deployment that describes more than one machine, a running deployment can also
be moved between them — rehearse a script against the simulator, then run it on
the machine — with one approval-gated tool call and no rebuild, no redeploy and
no restart.

There is one control target per deployment, so a switch applies everywhere at
once and is recorded: it outlives the conversation that made it, and a restart
adopts it rather than resetting it. See :doc:`switch-control-target` for the
whole workflow: the two tools, the reachability proof that keeps a failed
switch from stranding the deployment, the posture a move toward the live
machine requires, what the switch refuses, and how Bluesky plans behave while
the deployment is switched.

Rehearsing against a live target
================================

A deployment usually has nothing to rehearse the real-machine procedure on: the
``epics`` connector points at a facility gateway that may not exist yet, or not
from this laptop. Setting ``virtual_accelerator.live_standin: true`` in the build
profile gives it a machine to rehearse on — a **second** simulator container,
deployed as a control target of its own, ``standin``. The ``control-assistant``
preset ships it on; delete the line to run one machine again.

The stand-in is a third machine, not a rewrite of ``live``. It has its own
connector block, and ``control_system.connector.epics`` stays whatever your
facility wrote there — so ``live`` still names your machine while the rehearsal
runs beside it. ``control_target_set standin`` moves the deployment onto the stand-in;
``control_target_set live`` from there walks the real go-live path, gates and
all.

The two simulated machines run one image over one simulator view, and the
stand-in carries no errors of its own: it exists to rehearse the live safety
posture, not different physics. The two machines are told apart by a write to
the sandbox not showing on the stand-in and by the model RPC status naming its
instance (``virtual_accelerator`` or ``live_standin``). The label stays honest:
the banner reads ``LIVE MACHINE (stand-in)`` and the Web Terminal's header chip
reads ``STAND-IN``.
:doc:`switch-control-target` has the ritual itself.

**Scenarios reach both machines.** ``osprey sim apply`` writes one scenario file
and both containers poll it, so a scenario changes the world rather than one
lane. There is no scenario that applies to the simulator but not to the stand-in,
and switching targets does not undo one.

**The archive belongs to the machine.** The recorder records the stand-in when
one is deployed, so the store's past and the stand-in's present describe one
machine, the way a real machine's do. While that store is being
recorded, the ``live`` target is refused — a real machine's readings must not
land in a stand-in's archive. :doc:`switch-control-target` says how to clear
that.

.. _va-two-venues:

Two venues
==========

The simulator is one connector type, ``virtual_accelerator``, reached in one of
two venues. Where it runs is a setting under its connector block:

.. code-block:: yaml

   control_system:
     type: virtual_accelerator
     connector:
       virtual_accelerator:
         serving: in_process   # or: served (the default)

- ``served`` --- the simulator's container, serving the facility's channels over
  Channel Access and PVAccess and answering the model RPC. This is the default
  whenever the type is stated, and what the rest of this page describes.
- ``in_process`` --- the same simulator view and composite, run inside the
  process that asks. It needs no Docker, opens no port and serves no Channel
  Access, and it answers no model RPC. The ``hello-world`` preset runs this way.

A config whose ``control_system:`` section states no ``type`` gets the simulator
in process, and its ``serving`` leaf is not read: a section that names nothing
dials nothing. ``serving: in_process`` is refused on a deployment whose own type
is not ``virtual_accelerator``; on such a deployment the ``va`` target is the
served container.

An environment with no containers to depend on can run the tutorial on the
simulator in process:

.. code-block:: bash

   osprey set config.control_system.connector.virtual_accelerator.serving=in_process
   osprey build
   osprey up

Read one consequence before you do: **plans become browse-only.** The simulator
in process speaks no Channel Access, and the queue worker builds its devices
over Channel Access, so a plan started there could not drive a channel. Rather
than let one start and fail, the stack refuses earlier --- plans can still be
listed, authored, validated and staged into the shared draft, but the queue
will not hold them, and both the panels and the agent report a browse-only
deployment with the exact command that flips it back
(``osprey set config.control_system.connector.virtual_accelerator.serving=served``).
Everything that is not a plan --- channel reads and writes, the archiver, the
Channel Finder --- works as before. The ``epics`` block keeps its production
values throughout.

The archive follows the flip on its own. The recorder records **only** a machine
this deployment owns and serves over the network, so with no stand-in deployed
it stops writing while the simulator runs in process and idles; it re-reads the
project's ``config.yml`` every 30 seconds, so the change takes effect within one
poll and no restart or rebuild is involved. Nothing is deleted --- the history
already in the store stays readable, it simply stops growing, and it ages out
under the retention window as usual. ``osprey health`` will report the archive
as **stale** (a warning, not an error) once the newest sample is older than the
freshness threshold, which is the honest answer to "is this archive still being
written". Flipping back to ``served`` restarts recording within a poll too.

A stand-in changes that answer, because the recorder follows the machine rather
than the ``control_system`` section: the stand-in keeps running and is still
the machine this deployment records, so it keeps being recorded while the
simulator runs in process as well.

``mock`` is no longer a control-system type. A config or profile that states it
is refused, and the refusal names the new spelling:

.. code-block:: text

   `mock` is retired: the simulator in process is `control_system.type: virtual_accelerator` with `control_system.connector.virtual_accelerator.serving: in_process` (`osprey set connector=virtual_accelerator config.control_system.connector.virtual_accelerator.serving=in_process`), then rebuild with `osprey build`.

Connecting to the IOC
=====================

The container serves Channel Access on ``127.0.0.1:5064`` in EPICS name-server
mode — the one host-to-container configuration that works reliably across
container runtimes, since broadcast discovery does not cross the container VM
boundary. The project's ``virtual_accelerator`` connector block is configured to
match and sets ``EPICS_CA_NAME_SERVERS`` itself, so no client-side EPICS
environment setup is needed.

What the IOC serves, and how often
==================================

Two settings reach the container through its compose environment rather
than being fixed in the image, because each is a property of the machine you
are standing in for, or of who may alter it, rather than of OSPREY:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - What it does
   * - ``VA_POLL_INTERVAL_S``
     - Seconds between the served model's passes --- how often the IOC
       republishes the values it reads out of the composite. Set by
       ``simulation.tick_s`` (default ``1.0``), which ``osprey build`` renders
       into the compose file; a project ``.env`` value is not read. Lower it
       for a demo that should look live; raise it on a very large namespace.
   * - ``VA_MODEL_WRITE_TOKEN``
     - The credential a write to the model's own variables must present over
       the model RPC (see :doc:`/architecture/virtual-accelerator`). Unset, the
       container refuses every such write; reads need no token. There is no
       default to fall back on --- set it in the deployment's ``.env``.

The poll interval is refused at boot if it is not a number, or not greater
than zero, rather than being clamped --- so a typo shows up in ``docker logs``
instead of quietly changing what the machine looks like.

.. _va-serving-your-own-model:

Serving your facility's own model
=================================

A pyAT lattice
--------------

Stage the deck in the facility tree as ``data/facility/decks/<model>.json``
and name it in the model's record in ``data/facility/models.yaml``, with the
``wiring`` that ties each channel to an element of the deck:

.. code-block:: yaml

   - name: <model>
     engine: pyat
     deck: decks/<model>.json

``osprey build`` copies the deck into the simulator view as
``decks/<model>.json``, and the container builds the model from that copy when
``simulation.models`` serves it. No code and no image are involved.

Another backend
---------------

A physics engine other than pyAT is an engine plug-in. The container looks up
the ``engine`` each model record names in the ``osprey.simulation.engines``
entry-point group and calls the registered module's ``build``, which returns
the ``LUMEModel`` serving that model's wiring over its deck; ``osprey build``
reaches the same module through the same group when it checks the facility
tree. The shipped ``osprey.simulation.engines.pyat`` is the reference
implementation of the contract.

1. Write the engine module to the contract ``osprey.simulation.engines.pyat``
   implements.
2. Register it under the entry-point group in your package's metadata, and
   install the package where ``osprey build`` runs:

   .. code-block:: toml

      [project.entry-points."osprey.simulation.engines"]
      my_engine = "my_facility.engine"

3. Build an image that carries OSPREY's virtual-accelerator install and your
   package. The usual shape is a Dockerfile ``FROM`` the image OSPREY builds
   for the project, adding your package.
4. Name it as the service's image with ``services.virtual_accelerator.image``,
   or ``OSPREY_VA_IMAGE`` for one shell (:ref:`deployment-image-overrides`).
5. Name the engine in the model's record: ``engine: my_engine``.

Naming the image in ``services.virtual_accelerator.image`` renders the service
without a build, so no deploy rebuilds it from OSPREY's recipe.
``OSPREY_VA_IMAGE`` keeps it out of every build a start makes. Either way the
image has to be on the host or pullable where it is named;
:ref:`deployment-image-builds` says when each start builds.

Scenarios
=========

``osprey sim apply <scenario>`` works the same in both venues. Applying a
scenario writes the project's ``var/agent_data/simulation/active_scenarios``
file; the in-container engine polls it and, within about a second, composed
channel values reflect the new scenario. One behavioral difference between the
venues: served, a scenario switch only refreshes the engine-composed channels —
setpoints you wrote during the session live in the IOC's own records and
**survive** the switch. (In process, written values are reset.)

The served simulator's container mounts two directories: the render's data root
(``build/data``, read-only, holding the simulator view the build writes, rebuilt
on every ``osprey build``) and the scenario state directory
(``var/agent_data/simulation``, written by ``osprey sim apply`` while the system
runs). Both mounts are automatic for the deployed service.

What a scenario may contain, how scenarios compose, and what ``osprey sim
apply`` refuses is in :doc:`/how-to/run-scenarios`.

Write limits
============

Channels with a record in ``data/facility/limits.yaml`` carry a min/max range,
and a write outside that range is rejected before it reaches the IOC; an
in-range write goes through. The mandatory write-approval flow applies
unchanged — the Virtual Accelerator connector inherits the same write-safety
wiring as the EPICS connector.

Arm writes here without arming the machine
------------------------------------------

Write posture is per control target, and this is the page where that matters
most: the Virtual Accelerator can be write-armed while the live machine the
same deployment knows about stays read-only.
``control_system.writes_enabled`` is the posture a connector type inherits when
it says nothing about itself; a block under ``connector:`` answers for that type
instead.

.. code-block:: yaml

   control_system:
     writes_enabled: false          # what every type inherits
     connector:
       epics:
         writes_enabled: false      # the live machine, pinned by name
       virtual_accelerator:
         writes_enabled: true       # ... and the simulator alone is armed

Only a literal ``true`` arms a target. The quoted string ``'true'`` and the
number ``1`` do not, at either level, and a config that uses one of them will
find its writes refused. A type that states its own posture never falls back to
the inherited key, so the ``false`` above holds for the live machine even if a
profile turns the deployment-wide key on.

Switching to the live target (see
:doc:`switch-control-target`) therefore takes its writes away, with no config
edit and no rebuild — the same write tool that moves the simulator is refused
on the machine. The bundled ``control-assistant-readwrite`` and
``control-assistant-admin`` personas ship exactly those three keys: they pin
the live block by name, so no later per-type ``true`` can lift it.

.. note::

   On a deployment whose targets disagree like this, ``settings.json`` denies
   nothing up front — it is rendered once, before any target has been picked
   — so every refusal arrives per call instead, from the safety hook and
   the connector, naming the target that refused it. Tools you list under
   ``control_system.write_tools`` are refused by that same hook, which is how
   they are gated in every deployment.

.. note::

   The limits posture is a separate decision, and it is per connector type in
   the same way. ``control_system.limits_checking`` is the pair a type inherits
   when it says nothing about itself; a ``limits_checking`` block under a type's
   ``connector:`` entry answers for that type instead.

   .. code-block:: yaml

      control_system:
        limits_checking:
          enabled: true          # the pair every type inherits: only channels
          mode: exclusive        # in the limits file can be written
        connector:
          virtual_accelerator:
            limits_checking:
              enabled: true      # ... and on the simulator alone a channel
              mode: optional     # with no record is written with no limits

   That is the shape ``config.yml`` ends up in; write it in the build profile's
   ``config:`` block as flat dotted keys. A per-type block replaces the inherited pair as a *whole*: nothing is
   borrowed from the deployment-wide block, so both settings have to be written
   out. A block stating one of them alone is refused by ``osprey build`` and
   ``osprey validate``, naming the one that is missing. The limits database
   itself stays deployment-wide — the build renders one limits database from
   ``data/facility/limits.yaml`` for every target, and a per-type block does
   not take a path.

   With the pair above, a write to the simulator is still checked against the
   ranges the records in ``data/facility/limits.yaml`` give; what changes
   is that a channel with *no* record is allowed through on the
   simulator and refused on the live machine and the stand-in. That strict
   posture is what a switch to either real-machine target requires, and
   rehearsing it is what the stand-in is for. See `Rehearsing against a live
   target`_.

The archive
===========

A simulated machine still needs somewhere to keep what its channels did, and the
stack deploys one. ``osprey up`` brings up two more containers beside the
Virtual Accelerator:

- **the store** — a MongoDB service (``archiver-mongodb`` on the deployment's
  network, published on host port 27017 by default), holding one collection of
  timestamped samples; every channel the facility declares is recorded under
  its own address, an EPICS field name (``.RBV``) included;
- **the recorder** — a small service that reads the running machine on a fixed
  cadence and writes what answered into that collection.

The project's archiver connector (``archiver.type: mongodb_archiver``) reads
history back out of the same collection, so what the agent plots is what the
deployment recorded.

.. raw:: html
   :file: ../../_diagrams/va-archive-loop.html

What history is there
---------------------

Two halves of one timeline, and the deploy makes both true before you ask
anything of them.

**The seeded past.** The first deploy writes a base series for every channel the
machine serves, covering the whole retention window — at the shipped defaults,
**30 days back**, of which the most recent **48 hours** are sampled every
10 seconds and the rest every 60. The values are generated, but they are
generated the way the live machine generates its own: each channel's history is
built around the same baseline the Virtual Accelerator boots it at, with
excursions scaled to that channel's own noise. Nothing invents an event nobody would find in the
live machine.

Writing it takes a minute or two on a first deploy, and the deploy says so as it
goes ("seeding archive: N documents written across N channels", every 15 seconds
or so), then reports the span and the document count when it finishes. Later
deploys check the archive against the knobs now in force and skip the seed when
it already covers them.

**The recorded present.** From then on the recorder samples the machine every
10 seconds and stores what answered. A setpoint you write is readable out of the
archive within about half a minute. A channel that did not answer contributes
nothing — a gap in the archive is the honest record of a channel that was not
answering, never a value carried forward.

The join between the two is meant to be invisible: seeded samples and recorded
samples land on the same timestamps and around the same baselines, so where the
seed ends and recording begins there is noise, not a step an operator would
rightly chase.

**With a stand-in deployed, this is the stand-in's archive.** The seeded past is
the composite's history over the shared active set, the same samples whichever
machine the archive belongs to, and the recorder then samples the stand-in, so
the history read out of the store — including from the simulator target — is
the stand-in's. See `Rehearsing against a live target`_.

Retention is enforced by the store itself: dense samples expire after the hot
span, the coarse ones after the retention window, so a long-running deployment
stays bounded rather than growing forever. The collection is zstd-compressed;
the project's end-to-end test budgets the seeded store at under 2 GiB on disk at
these defaults.

.. note::

   Every number in this section is a knob in the build profile's ``va_archiver:`` block —
   ``retention_days``, ``hot_span_hours``, the cadences — not a constant in the
   code. Changing one is a profile edit and a rebuild; the next
   ``osprey up`` notices the archive no longer describes what the profile
   asks for and reseeds it. See :doc:`../build-profiles`.

What the archive will not claim
-------------------------------

Ask for a window older than the archive reaches and you get **no points** —
not a plausible-looking series stretched to fill the request. ``get_metadata``
likewise reports the oldest and newest samples the collection really holds,
rather than the window the profile declared. The archive never claims more than
it has.

The honesty rule
================

There is one configuration this stack refuses: a machine the deployment stands
up for itself and serves over the network — the ``virtual_accelerator`` control
system served from its container, or the ``live_standin`` one — paired with the
**mock archiver**, or with no archiver set at all, which resolves to the same
thing.

The reason is what the two do differently. The Virtual Accelerator serves
channels that move for modelled reasons: you step a corrector, the orbit
responds. The mock archiver does not store anything; it synthesizes a
plausible-looking history at read time, for questions nobody recorded the answer
to. Put them together and the agent reports a past that never happened, next to a
present that did — with nothing connecting the two, so the fiction can never be
caught by disagreeing with the machine it claims to describe.

The pairing is refused at every point it can be created:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Where
     - What happens
   * - ``osprey build``
     - The build refuses the profile, and names the profile keys to change: add
       a ``va_archiver:`` block (which is what makes the store exist) and set
       ``config: {archiver.type: mongodb_archiver}``.
   * - ``osprey up`` / ``restart``
     - The deploy aborts before starting anything, and names the ``config.yml``
       edit: set ``type:`` under ``archiver:`` to a connector reading a store
       this stack writes, or serve the simulator in process
       (``control_system.connector.virtual_accelerator.serving: in_process``).
   * - MCP server startup
     - The server refuses to start on such a ``config.yml``, so a file
       hand-edited after the build cannot quietly bring the pairing back.
   * - ``osprey validate``
     - Reports the same refusal without building anything, so the pairing is
       caught the moment it is written into ``profile.yml`` rather than at
       deploy time.

Two pairings that look similar are perfectly legal, because nothing lies in
either: **the simulator in process + mock archiver** is the honestly storeless
deployment (it has no recorder, so a synthesized archive is the only one it can
have, and nothing is claimed to be real), and **EPICS + mock archiver** is a
real machine that simply has no archive attached yet.

.. warning::

   ``config.yml`` is read as **nested sections**. A top-level dotted line like
   ``archiver.type: mongodb_archiver`` added at the top of the file configures
   nothing at all — the archiver is whatever the ``archiver:`` section says.
   The refusal messages call this out when they find such a line, rather than
   reporting the key as merely unset.
