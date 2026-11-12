.. _how-to-use-virtual-accelerator:

===========================
Use the Virtual Accelerator
===========================

How to run the demo tutorial against a **Virtual Accelerator**.
The Virtual Accelerator is a containerized simulator that serves real EPICS
Channel Access. PyAT physics runs behind the channels your facility wires to a
lattice, so correctors move and BPMs respond.
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

   **Prerequisites:** Docker (or Podman) installed; the demo
   tutorial project (see :doc:`/getting-started/demo`).

Overview
========

The demo tutorial ships interchangeable control-system backends,
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

The physics behind the channels comes from the profile's ``simulation.models``.
The container builds each served physics model through the engine its record
names. It serves those models together with the texture model as one composite.
The shipped lattice engine is pyAT, built on the facility-agnostic ``lume-pyat``
package, over the deck your facility tree stages. A render that serves
``texture`` alone serves the same channels with no physics behind them: a
setpoint holds the value written to it and a readback follows its seed. The
second shipped engine models an injection chain, and any other engine is a
plug-in; both are covered in :ref:`va-serving-your-own-model`.

What channels it serves
=======================

Your project's own. Every ``osprey build`` renders the facility definition
under ``data/facility/`` into the simulator view, ``build/data/simulator/``,
which the compose file mounts at ``/data/simulator/``.

The container serves the ``channels`` of the view's ``addresses.json``, which
holds every channel address the facility declares. It also serves one
``<code>:SIM:<model>:STATUS`` channel for each served physics model. Its ready
line in ``docker logs`` prints the count.

Nothing is invented, and no other channel list feeds the set:

- A data mount without the view refuses the boot and names the missing file.
- A facility tree the build cannot render stops in
  ``osprey facility validate``'s own stops before the view is written.

A channel's role, not its address, says what it is. A channel record with
``role: setpoint`` is written, and its readback is the record's ``pair``. A
setpoint that names none is paired by the build with the one readback on the
same device whose signal names the same quantity, else it is its own readback. A
setpoint's ``tolerance``, next to its ``pair``, says how close that readback
must come before a move counts as done. The build stops when the readback's
seeded motion is wider than the tolerance. A channel with no role is a readback.
The address text is the facility's own and is served as it is written: no token
inside it means anything to OSPREY.

A facility whose models wire no channel is served by the texture alone. A
setpoint holds the value written to it, a readback follows its seed, and no
physics runs behind either.

Quickstart
==========

The demo stack ships pointed at the Virtual Accelerator and
**already deploys** it. The preset's ``virtual_accelerator:`` block renders a
compose service, so ``osprey up`` brings the Virtual Accelerator up alongside
the rest of the stack. The connector is already talking to it. There is
nothing to enable.

.. code-block:: bash

   osprey up   # brings up the Virtual Accelerator with the rest of the stack
   osprey web         # the agent talks to real Channel Access

``osprey up`` brings up more than the Virtual Accelerator. The preset also
declares a ``va_archiver:`` block, so the deploy stands up the machine's
**archive** next to it: a MongoDB store and a recorder service. The deploy
seeds the archive before the rest of the stack starts. See `The archive`_ below.

The very first ``osprey up`` that includes the Virtual Accelerator builds its
container image, installing the physics and EPICS serving stack. Expect it to
take several minutes: it is building, not hanging. Later deploys reuse the
image.

If your deployment came from a preset or profile that selects a different
connector, point it at the Virtual Accelerator explicitly:

.. code-block:: bash

   osprey set connector=virtual_accelerator
   osprey build
   osprey up

.. note::

   All three steps matter:

   - ``osprey set`` writes the setting into ``profile.yml``.
   - ``osprey build`` carries it into ``build/``, where each service gets its
     own copy of the rendered config.
   - ``osprey up`` starts what was just rendered.

   Anything already running in a container keeps the old setting until you
   restart it. No image rebuild is involved.

**The archive has to come first.** On a deployment created from the
``demo`` preset the switch just works. The preset declares where
the archive lives, so the deployment already reads a real store.

A deployment with no archive of its own still reads the mock archiver, which
makes its history up as it is asked for it. On such a deployment the build
**refuses** the profile, and says what to do instead, one of:

- Point ``archiver.type`` at a store this deployment writes
  (``mongodb_archiver`` for the store the preset deploys).
- Keep the simulator in process for an honestly storeless deployment.

`The honesty rule`_ below explains why.

Switching a running deployment
==============================

Those three commands set which control system the deployment **starts** on. A
deployment that describes more than one machine can also be moved between them
while it runs. You can rehearse a script against the simulator, then run it on
the machine. The move takes one approval-gated tool call, with no rebuild, no
redeploy and no restart.

There is one control target per deployment, so a switch applies everywhere at
once. The switch is recorded: it outlives the conversation that made it, and a
restart adopts it rather than resetting it. See :doc:`switch-control-target`
for the whole workflow:

- the two tools;
- the reachability proof that keeps a failed switch from stranding the
  deployment;
- the posture a move toward the live machine requires;
- what the switch refuses;
- how Bluesky plans behave while the deployment is switched.

Rehearsing against a live target
================================

A deployment usually has nothing to rehearse the real-machine procedure on: the
``epics`` connector points at a facility gateway that may not exist yet, or not
from this laptop. Setting ``virtual_accelerator.live_standin: true`` in the build
profile gives it a machine to rehearse on — a **second** simulator container,
deployed as a control target of its own, ``standin``. The ``demo``
preset ships it on; delete the line to run one machine again.

The stand-in is a third machine, not a rewrite of ``live``. It has its own
connector block, and ``control_system.connector.epics`` stays whatever your
facility wrote there — so ``live`` still names your machine while the rehearsal
runs beside it. ``control_target_set standin`` moves the deployment onto the stand-in;
``control_target_set live`` from there walks the real go-live path, gates and
all.

The two simulated machines run one image over one simulator view. The
stand-in carries no errors of its own: it exists to rehearse the live safety
posture, not different physics. The two machines are told apart by a write to
the sandbox not showing on the stand-in and by the model RPC status naming its
instance (``virtual_accelerator`` or ``live_standin``). The label stays honest:
the banner reads ``Rehearsal — nothing moves`` and the Web Terminal's header
chip reads *Rehearsal*.
:doc:`switch-control-target` has the ritual itself.

**Scenarios reach both machines.** ``osprey sim apply`` writes one scenario file
and both containers poll it, so a scenario changes the world rather than one
lane. There is no scenario that applies to the simulator but not to the stand-in,
and switching targets does not undo one.

**The archive belongs to the machine.** The store this deployment seeds and
records is the simulator's, with or without a stand-in. The recorder samples
the virtual accelerator, and the seeded past is the simulator's own history.
The stand-in rehearses the facility's real machine and is shown that machine's
past instead --- the ``epics_archiver`` appliance when the deployment names one,
and no archive otherwise. Nothing the stand-in reads is ever written into the
simulator's store, so no target is gated on the archive.
:doc:`switch-control-target` has the whole picture.

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
in process speaks no Channel Access. The queue worker builds its devices over
Channel Access, so a plan started there could not drive a channel. Rather than
let one start and fail, the stack refuses earlier:

- Plans can still be listed, authored, validated and staged into the shared
  draft, but the queue will not hold them.
- Both the panels and the agent report a browse-only deployment, with the
  exact command that flips it back
  (``osprey set config.control_system.connector.virtual_accelerator.serving=served``).

Everything that is not a plan --- channel reads and writes, the archiver, the
Channel Finder --- works as before. The ``epics`` block keeps its production
values throughout.

The archive follows the flip on its own. The store is the simulator's, and the
recorder samples it only while the ``va`` target lands on this deployment's own
served simulator: loopback, on the port ``services.virtual_accelerator.port``
publishes. The simulator in process serves nothing on the network to sample, so
the recorder stops writing and idles. Pointing the ``va`` block's gateways at
another host idles it the same way, and so does taking the simulator out of the
deployment. It re-reads the project's ``config.yml`` every 30 seconds, so a
change takes effect within one poll and no restart or rebuild is involved.
Nothing is deleted --- the history already in the store stays readable, it
simply stops growing, and it ages out under the retention window as usual.
``osprey health`` will report the archive as **stale** once the newest sample is
older than the freshness threshold. Stale is a warning, not an error: it is the
honest answer to "is this archive still being written". Flipping back to
``served`` restarts recording within a poll too.

A stand-in beside the simulator changes none of this: it is never what the
recorder samples.

``mock`` is no longer a control-system type. A config or profile that states it
is refused, and the refusal names the new spelling:

.. code-block:: text

   `mock` is retired: the simulator in process is `control_system.type: virtual_accelerator` with `control_system.connector.virtual_accelerator.serving: in_process` (`osprey set connector=virtual_accelerator config.control_system.connector.virtual_accelerator.serving=in_process`), then rebuild with `osprey build`.

Connecting to the IOC
=====================

The container serves Channel Access on ``127.0.0.1:5064`` in EPICS name-server
mode. That is the one host-to-container configuration that works reliably across
container runtimes, since broadcast discovery does not cross the container VM
boundary. The project's ``virtual_accelerator`` connector block is configured to
match and sets ``EPICS_CA_NAME_SERVERS`` itself, so no client-side EPICS
environment setup is needed.

What the IOC serves, and how often
==================================

Three settings reach the container through its compose environment rather
than being fixed in the image. Each is a property of the machine you are
standing in for, or of who may alter it, rather than of OSPREY:

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
   * - ``VA_FAST_INTERVAL_S``
     - The fast period: seconds between publications of the readings a model
       serves faster than its passes, such as an injection chain's
       fast current (:ref:`va-injection-chain`). Set by
       ``simulation.fast_tick_s`` (default ``0.1``) in the same way. A
       facility with no such reading publishes nothing at this period.
   * - ``VA_MODEL_WRITE_TOKEN``
     - The credential a write to the model's own variables must present over
       the model RPC (see :doc:`/architecture/virtual-accelerator`). Unset, the
       container refuses every such write; reads need no token. There is no
       default to fall back on --- set it in the deployment's ``.env``.

Either interval is refused at boot if it is not a number, or not greater
than zero, rather than being clamped. A typo therefore shows up in
``docker logs`` instead of quietly changing what the machine looks like.

.. _va-serving-your-own-model:

Serving your facility's own model
=================================

To version the model with the other machine facts apart from any one
deployment, keep them in a :doc:`machine repository <machine-repository>`;
``data/facility/`` in this section is then its machine root.

.. toctree::
   :hidden:

   machine-repository

A pyAT lattice
--------------

#. Stage the deck in the facility tree as ``data/facility/decks/<model>.json``.
#. Name it in the model's record in ``data/facility/models.yaml``, with the
   ``wiring`` that ties each channel to an element of the deck:

   .. code-block:: yaml

      - name: <model>
        engine: pyat
        deck: decks/<model>.json

``osprey build`` copies the deck into the simulator view as
``decks/<model>.json``, and the container builds the model from that copy when
``simulation.models`` serves it. No code and no image are involved.

.. _va-injection-chain:

An injection chain
------------------

A lattice answers a corrector write, but nothing in it injects. Software that
fills the machine --- a fill script, or a facility's own bucket-loading IOC
run unmodified beside the simulator --- asks for a shot and then watches
counters, charges and the stored current step. The shipped ``injection``
engine gives it an injector to ask. It models no beam dynamics and needs no
deck: the model record's ``settings`` and ``wiring`` are the whole chain, and
each address is a channel the facility declares. With invented addresses:

.. code-block:: yaml

   - name: INJECTOR
     engine: injection
     settings:
       injection:
         cycle:
           period_s: 2.0   # one cycle every 2 s
           width_s: 0.4    # the cycle signal stays high for 0.4 s
         stages: [transport, storage]
         current_per_charge: 1.5
     wiring:
       - address: INJ:REQ
         engine: {signal: request}
       - address: INJ:SRC:Q:SP
         engine: {signal: parameter, quantity: charge, start: 1.0}
       - address: INJ:TL:EFF:SP
         engine: {signal: parameter, quantity: efficiency, stage: 0, start: 0.9}
       - address: INJ:STORE:EFF:SP
         engine: {signal: parameter, quantity: efficiency, stage: 1, start: 0.7}
       - address: INJ:STORE:I:SP
         engine: {signal: parameter, quantity: current, start: 10.0}
       - address: INJ:SHOTS
         engine: {signal: counter, offset_s: 0.02}
       - address: INJ:TL:Q
         engine: {signal: charge, offset_s: 0.05, stage: 0}
       - address: INJ:STORE:EFF
         engine: {signal: efficiency, offset_s: 0.08, stage: 1}
       - address: INJ:STORE:I
         engine: {signal: current, offset_s: 0.1}
       - address: INJ:STORE:I:FAST
         engine: {signal: current_fast}
       - address: INJ:CYCLE
         engine: {signal: busy}

A client write of ``INJ:REQ`` starts one shot; a request while a shot is in
flight is ignored. Each wiring record's ``signal`` says what the shot does to
its channel:

- ``parameter``: a setpoint of the chain, starting at ``start``. Its
  ``quantity`` is ``charge``, the source charge per shot; ``efficiency``, that
  of the stage at index ``stage`` of ``stages``; or ``current``, the stored
  current.
- ``counter``, ``charge`` and ``efficiency``: readings the shot steps
  ``offset_s`` seconds after its request. A counter adds one. A charge shows
  the charge after the stage at index ``stage``, the source charge times
  every efficiency up to that stage's, and an efficiency shows that stage's.
- ``current``: the stored current, which the shot raises at ``offset_s`` by
  the final charge times ``current_per_charge``. Nothing saturates or decays
  it.
- ``current_fast``: the stored current again, published at the fast period
  (`What the IOC serves, and how often`_). Either current reading may state a
  ``scale``: ``{signal: current_fast, scale: 0.001}`` reads a stored current
  in mA as A.
- ``busy``: the cycle signal, high for ``width_s`` every ``period_s`` whether
  or not a shot runs.

Shots run only in the container: in process the chain serves its start
values and a request starts nothing. What scenarios and channel faults do to
a chain is in :doc:`/how-to/run-scenarios`.

.. _va-turn-by-turn:

Turn-by-turn data from an injection chain
-----------------------------------------

A first-turn steering or injection-tuning tool arms a ring's monitors, waits
for a shot, and reads each monitor's turn-by-turn arrays. A ``pyat`` ring
serves those arrays when its settings state a ``turn_by_turn`` block and it
reads the shot from an injection chain through model inputs. The chain stays
the only source of shots: the ring tracks a bunch through its deck each time
the chain's shot counter rises.

.. raw:: html
   :file: ../../_diagrams/va-turn-by-turn.html

**The chain.** Wire a charge after the last stage and a shot counter that
steps after it, so each shot's delivery carries that shot's charge. To the
chain of :ref:`va-injection-chain`, add:

.. code-block:: yaml

   - address: INJ:STORE:Q
     engine: {signal: charge, offset_s: 0.05, stage: 1}
   - address: INJ:STORE:SHOTS
     engine: {signal: counter, offset_s: 0.06}

A counter whose ``offset_s`` is not larger than the charge's makes each shot
carry the previous shot's charge. To fill from an empty ring, start the
chain's ``current`` parameter at ``0`` and give its channel a limits band that
holds ``0``.

**The ring.** With invented addresses, and one monitor shown:

.. code-block:: yaml

   - name: RING
     engine: pyat
     deck: decks/RING.json
     settings:
       pyat:
         solve: periodic
         turn_by_turn:
           injection_element: INJ_POINT
           turns: 250
           trigger_sample: 10
           particles: 16
           launch: {x: 0.001, y: -0.0005}
           aperture: {x: 0.004, y: 0.004}
     inputs:
       - {address: INJ:STORE:SHOTS, engine: {signal: shot}}
       - {address: INJ:STORE:Q, engine: {signal: charge}}
       - {address: INJ:STORE:I, engine: {signal: stored_current}}
     wiring:
       - address: RING:RF:ON
         engine: {signal: rf_switch}
       - address: RING:TBT:PILOT
         engine: {signal: pilot_tone}
       - address: RING:TBT:ATTEN
         engine: {signal: stored}
       - address: RING:TBT:TURNS
         engine: {signal: turns_tracked}
       - address: RING:BPM1:ARM
         element: BPM1
         engine: {signal: arm}
       - address: RING:BPM1:ARMED
         element: BPM1
         engine: {signal: armed}
       - address: RING:BPM1:TBT:X
         element: BPM1
         engine: {signal: turn_x}
       - address: RING:BPM1:TBT:Y
         element: BPM1
         engine: {signal: turn_y}
       - address: RING:BPM1:TBT:SUM
         element: BPM1
         engine: {signal: turn_sum}

Each input's ``signal`` says what the ring reads from the chain: ``shot``,
the counter whose rise is a shot; ``charge``, the shot's charge; and
``stored_current``, the current already stored. Declare all three: without
``shot`` no shot is taken, without ``charge`` every shot carries zero charge,
and without ``stored_current`` there is no stored beam. Each wiring record's
``signal``:

- ``arm``: a monitor's arm setting. Wire one per monitor, naming its
  monitor ``element``.
- ``armed``: whether that monitor waits for a shot.
- ``turn_x``, ``turn_y``, ``turn_sum``: the monitor's arrays, positions in
  metres and the sum signal.
- ``turns_tracked``: the turns the last shot tracked, the ring's; it names no
  element.
- ``rf_switch``: ``0`` sets every cavity's voltage to zero, anything else
  restores the deck's voltages. It starts at ``1`` and names no element.
- ``pilot_tone``: non-zero turns the pilot tone on. It starts at ``0`` and
  names no element.
- ``stored``: a setting that holds what is written and drives no physics,
  for an attenuation or a mode the software sets; it may name an element.

Declare ``arm``, ``rf_switch``, ``pilot_tone`` and ``stored`` channels
``role: setpoint``, and each array ``value_type: waveform`` with
``shape: [<trigger_sample + turns>]``, here ``[260]``; every other record is a
plain float.

**The settings.** Only ``injection_element`` is required:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - What it sets
   * - ``injection_element``
     - required
     - The deck element a shot starts at; the deck holds it exactly once.
   * - ``turns``
     - ``250``
     - Turns tracked per shot, a whole number of at least 1.
   * - ``trigger_sample``
     - ``10``
     - Samples before injected turn 1, a whole number of at least 0.
   * - ``particles``
     - ``16``
     - Particles tracked per shot.
   * - ``launch``
     - all ``0``
     - The bunch centroid at injection: ``x``, ``px``, ``y``, ``py``,
       ``delta``, ``ct``, each signed.
   * - ``emittance_x``, ``emittance_y``, ``sigma_delta``, ``sigma_z``
     - ``0``
     - The bunch's size, matched to the optics at injection. All zero, every
       particle starts at ``launch``.
   * - ``jitter``
     - ``{x: 5.0e-5, y: 3.0e-5}``
     - The shot-to-shot centroid jitter's sigma, metres.
   * - ``aperture``
     - none
     - ``{x, y, shape}``: positive half-axes in metres, both required, and
       ``elliptical`` (the default) or ``rectangular``; set on every element
       with a length.
   * - ``sum_per_charge``
     - ``1.0``
     - Sum per unit of the shot's charge.
   * - ``sum_per_stored``
     - ``1.0``
     - Sum per unit of stored current.
   * - ``sum_floor``
     - ``0.0``
     - The lowest sum a sample reads.
   * - ``position_noise``
     - ``1.0e-5``
     - Position noise sigma, metres.
   * - ``sum_noise``
     - ``0.0``
     - Sum noise sigma.
   * - ``floor_noise``
     - ``1.0e-3``
     - Position noise sigma where the sum is at the floor, metres.
   * - ``tone_period``
     - ``10``
     - Samples between pilot-tone spikes, a whole number of at least 1.
   * - ``tone_amplitude``
     - ``1.0``
     - Pilot-tone spike size.

Every number but ``launch`` is finite and not negative; an unknown key, also
inside ``launch``, ``jitter`` or ``aperture``, stops the build. The apertures
decide when an RF-off bunch is lost; they are the facility's to calibrate.

**The arrays.** Each array holds ``trigger_sample + turns`` samples. The
samples before ``trigger_sample`` see the stored beam only, and injected
turn 1 sits at index ``trigger_sample``. The tracking starts at the injection
element, so a monitor upstream of it reads the end of each turn. The sum is
the shot's charge times the share of the bunch still alive times
``sum_per_charge``, plus, with the RF on, the stored current held before the
shot times ``sum_per_stored``, plus noise, floored at ``sum_floor``. Positions
are the sum-weighted mix of the stored beam, at zero, and the bunch's
centroid, plus ``position_noise``; where the sum is at the floor they are
``floor_noise``. With the pilot tone on, every sample whose index is a
multiple of ``tone_period`` carries a spike in all three arrays. The noise is
keyed on the record's address, the shot and the sample, so a shot reads the
same in every run.

**Arming.**

- Arming is per monitor. A client write of a non-zero ``arm`` arms it, ``0``
  disarms it, and arming an armed monitor changes nothing. ``arm`` reads back
  what was written; ``armed`` is the state.
- A restore, a scenario's start state and a write a ``stuck`` or
  ``disconnected`` fault holds never arm.
- A shot reaches the armed monitors only. Their arrays and ``turns_tracked``
  are posted, then their ``armed`` drops to ``0``; every other monitor keeps
  its arrays. A shot with no monitor armed changes nothing.
- ``armed`` at ``0`` guarantees the shot's arrays are already served, on
  Channel Access and PVAccess both, so a client may time a shot on ``armed``
  alone. A shot that arrives before the drop finds those monitors unarmed.
- A reset or a scenario change returns every monitor to unarmed, every array
  to zeros and ``turns_tracked`` to ``0``, and is not mistaken for a shot.

**Shots.**

- One pass sees one delivery and tracks at most once. A counter that rose by
  more than one between two passes tracks once, with the latest charge. A
  counter that falls is not a shot.
- A zero-charge shot is still a shot: the arrays read the floor, the noise
  and any stored beam, and ``armed`` drops.
- A tracking that fails leaves the armed monitors' arrays at zeros and
  ``turns_tracked`` at ``0``; ``armed`` still drops and the ring keeps
  serving.
- With the RF off the ring has no closed orbit: its orbit readings read NaN
  and its settings stay writable. The stored beam adds nothing to the sum, so
  the samples before the trigger read the floor, and a bunch with declared
  apertures is lost within some hundred turns.

**Cost.** Tracking runs on the serving loop, about 0.5 s for 16 particles
over 250 turns and 34 s for 1000 particles on a ring of 72 monitors: roughly
0.13 ms per particle and turn, growing with the deck. During it a read
answers at once with the last posted value; a write, a model RPC call and a
response-matrix request wait until it ends and are then served, never
refused. A client timeout shorter than the tracking is the client's.

**Refusals.** ``osprey build`` stops the ring when the block or a record is
malformed: a missing or repeated ``injection_element``, a value out of range,
an unknown key, a turn record on a ring without the block, a monitor record
without a monitor ``element``, a ring record naming one, a setting not wired
for write, an array that is not a waveform of ``trigger_sample + turns``
samples, any other turn record that is not a float, and an input whose
``engine`` is not one ``{signal: ...}`` the ring reads. The model fails to
build, with the text in its status channel, when an ``rf_switch`` is wired
over a deck with no RF cavity. The container refuses to start when an array
is larger than the configuration's ``max_array_bytes`` at 8 bytes a sample,
naming the channel and both sizes.

**The response matrix.** ``model_response_matrix`` takes ``kind``, in the
Python executor tool (:doc:`/reference/contracts/python-executor`) and in
``osprey.apps.sdk``: ``orbit`` (the default) or ``trajectory``. The
trajectory response is the first-turn response of one particle, without
jitter, from ``launch`` through the declared apertures, in the orbit's rows
and units; it is computed from the deck and settings, not from served state,
and only a ring with the block answers it. A corrector whose kicked particle
is lost before a monitor has no column; the tool's ``missing`` and the SDK's
``unsolved`` name it with where the particle was lost. When the particle is
lost with no corrector stepped, every corrector is named and the matrix has
no column. An engine whose ``response_matrix`` takes no ``kind`` answers the
orbit response alone and refuses any other kind.

**What is not modelled.** The ring's capture or loss does not feed back into
the chain's stored current; the stored current neither decays nor
saturates; there is one bunch, no fill pattern; no bunch cleaning, tune scan,
beam-based alignment or RF frequency correction. Shots run only in the
container: in process every delivery is a baseline, so no shot is taken and
the arrays stay zeros.

Another backend
---------------

An engine OSPREY does not ship is an engine plug-in. The container looks up
the ``engine`` each model record names in the ``osprey.simulation.engines``
entry-point group. It calls the registered module's ``build``, which returns
the ``LUMEModel`` serving that model's wiring over its deck. ``osprey build``
reaches the same module through the same group when it checks the facility
tree. The shipped ``osprey.simulation.engines.pyat`` is the reference
implementation of the contract.

1. Write the engine module to the contract
   ``osprey.simulation.engines.EnginePlugin`` states;
   ``osprey.simulation.engines.pyat`` implements it.
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
channel values reflect the new scenario. There is one behavioral difference
between the venues. Served, a scenario switch only refreshes the engine-composed
channels: setpoints you wrote during the session live in the IOC's own records
and **survive** the switch. (In process, written values are reset.) ``osprey sim
apply still`` serves every reading without drift, couplings or noise.

The served simulator's container mounts two directories:

- the render's data root, ``build/data``: read-only, holding the simulator view
  the build writes, rebuilt on every ``osprey build``;
- the scenario state directory, ``var/agent_data/simulation``: written by
  ``osprey sim apply`` while the system runs.

Both mounts are automatic for the deployed service.

What a scenario may contain, how scenarios compose, and what ``osprey sim
apply`` refuses is in :doc:`/how-to/run-scenarios`.

Write limits
============

Channels with a record in ``data/facility/limits.yaml`` carry a min/max range,
and a write outside that range is rejected before it reaches the IOC; an
in-range write goes through. The write switch's positions apply
unchanged — the Virtual Accelerator connector inherits the same write-safety
wiring as the EPICS connector.

Write to the simulator without arming the machine
-------------------------------------------------

Writes are decided per login and per machine, and this is the page where that
matters most. The tier key says how far a login may write:

.. code-block:: yaml

   control_system:
     writes: asked         # never | asked | allowed; silence is never

The tier is target-blind. Which machine a write reaches is the person's own
switch, set per machine from the
:ref:`control-target chip <web-terminal-session-posture>`, and every machine
starts off. Arm the simulator and leave the real machine off. The same write
tool that moves the simulator is then refused on the machine after a switch
(:doc:`switch-control-target`). That takes no config edit and no rebuild. The
rule is on :ref:`write-safety-config`.

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
   ranges the records in ``data/facility/limits.yaml`` give. What changes is
   that a channel with *no* record is allowed through on the simulator, and
   refused on the live machine and the stand-in. That strict
   posture is what a switch to either real-machine target requires, and
   rehearsing it is what the stand-in is for. See `Rehearsing against a live
   target`_.

The archive
===========

A simulated machine still needs somewhere to keep what its channels did, and the
stack deploys one. ``osprey up`` brings up two more containers beside the
Virtual Accelerator:

- **the store** — a MongoDB service (``archiver-mongodb`` on the deployment's
  network, published on host port 10801 at the default port base, set by the
  profile's ``va_archiver.port_host``; see :ref:`reference-ports-layout`). It
  holds one collection of timestamped samples. Every channel the facility
  declares is recorded under its own address, an EPICS field name (``.RBV``)
  included, and waveform channels are kept in a sibling collection (see
  `Waveform channels`_);
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
machine serves, covering the whole retention window. At the shipped defaults
that window is **30 days back**. The most recent **48 hours** are sampled every
10 seconds and the rest every 60.

The values are generated, but they are generated the way the live machine
generates its own. Each channel's history is built around the same baseline the
Virtual Accelerator boots it at, with excursions scaled to that channel's own
noise. Nothing invents an event nobody would find in the
live machine.

Writing it takes a minute or two on a first deploy. The deploy says so as it
goes, every 15 seconds or so ("seeding archive: N documents written across N
channels"). When it finishes, it reports the span and the document count. Later
deploys check the archive against the knobs now in force and skip the seed when
it already covers them.

**The recorded present.** From then on the recorder samples the machine every
10 seconds and stores what answered. A setpoint you write is readable out of the
archive within about half a minute. A channel that did not answer contributes
nothing — a gap in the archive is the honest record of a channel that was not
answering, never a value carried forward.

The join between the two is meant to be invisible. Seeded samples and recorded
samples land on the same timestamps and around the same baselines. Where the
seed ends and recording begins there is noise, not a step an operator would
rightly chase.

**With a stand-in deployed, this is still the simulator's archive.** The seeded
past is the composite's history over the shared active set, and the recorder
samples the virtual accelerator. The stand-in is shown the facility's own
archive instead, and never this store. See `Rehearsing against a live target`_.

Retention is enforced by the store itself. Dense samples expire after the hot
span and the coarse ones after the retention window, so a long-running
deployment stays bounded rather than growing forever. The collection is
zstd-compressed; the project's end-to-end test budgets the seeded store at under
2 GiB on disk at these defaults.

.. note::

   Every number in this section is a knob in the build profile's ``va_archiver:`` block —
   ``retention_days``, ``hot_span_hours``, the cadences — not a constant in the
   code. Changing one is a profile edit and a rebuild; the next
   ``osprey up`` notices the archive no longer describes what the profile
   asks for and reseeds it. See :doc:`../build-profiles`.

Waveform channels
-----------------

A waveform channel — one whose value is a whole trace, such as a septum's
current pulse — is archived beside the scalars, not among them:

- **its own collection.** Traces live in a sibling collection named after the
  scalar one with ``_waveforms`` appended (``<collection>_waveforms``). It
  holds one document per channel per instant, the trace stored whole as a list
  of floats.
- **one trace a minute.** The seed and the recorder both write a waveform at the
  tail cadence: **every 60 seconds** at the shipped defaults
  (``tail_cadence_sec``). Neither writes one at the 10-second scalar cadence, so
  the hot span holds no denser waveform history than the rest of the window. The
  build requires the recorder's waveform cadence (``recorder_tail_cadence_sec``)
  to equal the seed's, so both write one history on one grid.
- **raw reads only.** The archiver connector returns a waveform channel's value
  as ``list[float]``, every stored trace in range, unbinned. Raw is the only
  processing defined for a waveform; asking for an average, a min/max or any
  other aggregation of one is refused with an error naming the channel.
- **a budget per read.** ``archiver_read`` never bins a waveform channel; it
  returns every trace in range at the channel's stored cadence and counts the
  points. One read may return at most **32,000** waveform points summed over
  every trace of every waveform channel requested — an hour of one 500-point
  trace a minute fits. A read over the budget is refused before anything is
  built, with a hint to narrow the window or read fewer waveform channels at
  once. Scalar samples do not count against it.

The EPICS archiver connector does not support arrays: reading a waveform PV
through it is refused with ``Waveform PVs not supported``. Waveform history is
only available from the store this stack deploys.

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
up for itself and serves over the network, paired with the **mock archiver**.
Such a machine is the ``virtual_accelerator`` control system served from its
container, or the ``live_standin`` one. No archiver set at all resolves to the
same thing as the mock archiver, so it is refused the same way.

The reason is what the two do differently. The Virtual Accelerator serves
channels that move for modelled reasons: you step a corrector, the orbit
responds. The mock archiver does not store anything; it synthesizes a
plausible-looking history at read time, for questions nobody recorded the answer
to. Put them together and the agent reports a past that never happened, next to a
present that did. Nothing connects the two, so the fiction can never be caught
by disagreeing with the machine it claims to describe.

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
either:

- **The simulator in process + mock archiver** is the honestly storeless
  deployment. It has no recorder, so a synthesized archive is the only one it
  can have, and nothing is claimed to be real.
- **EPICS + mock archiver** is a real machine that simply has no archive
  attached yet.

.. warning::

   ``config.yml`` is read as **nested sections**. A top-level dotted line like
   ``archiver.type: mongodb_archiver`` added at the top of the file configures
   nothing at all — the archiver is whatever the ``archiver:`` section says.
   The refusal messages call this out when they find such a line, rather than
   reporting the key as merely unset.
