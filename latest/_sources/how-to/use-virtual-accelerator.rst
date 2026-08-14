===========================
Use the Virtual Accelerator
===========================

How to run the Control Assistant tutorial against a **Virtual Accelerator** — a
containerized soft-IOC that serves real EPICS Channel Access, with PyAT physics
behind the storage-ring lattice channels, so correctors move and BPMs respond.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the Virtual Accelerator is (and is not)
   - The three-state ``control_system.type`` switch
   - Pointing a project at the soft-IOC the stack already deploys
   - Switching back to the mock, and why scans go browse-only there
   - How ``osprey sim apply`` scenarios behave in Virtual Accelerator mode
   - Write limits
   - The stored archive the stack deploys, and the one pairing it refuses

   **Prerequisites:** Docker (or Podman) installed; the Control Assistant
   tutorial project (see :doc:`/getting-started/control-assistant`).

Overview
========

The Control Assistant tutorial ships three interchangeable control-system
backends, selected by a single ``control_system.type`` value:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``type``
     - Backend
   * - ``mock``
     - The in-process simulation. No container, no network — every channel
       returns a synthesized value. The fallback for environments with no
       containers to depend on; scans are browse-only there (below).
   * - ``virtual_accelerator`` *(default)*
     - A containerized PyAT soft-IOC serving real EPICS Channel Access. Storage-
       ring magnet setpoints drive a live lattice and BPM readbacks respond;
       every other channel is composed by the same simulation engine the mock
       uses. The tutorial's default, and deployed as part of its stack.
   * - ``epics``
     - Production EPICS, pointed at the facility gateway. Untouched by this
       guide.

The Virtual Accelerator is a **local physics simulator**, not a digital twin —
it is not synced to any real machine. The OSPREY agent reads and writes it
exactly as it does the mock or a real machine; only the backend changes.

Quickstart
==========

The Control Assistant stack ships pointed at the Virtual Accelerator and
**already deploys** it: the preset's ``virtual_accelerator:`` block renders a
compose service, so ``osprey up`` brings the soft-IOC up alongside the
rest of the stack and the connector is already talking to it. There is nothing
to switch on.

.. code-block:: bash

   osprey up   # brings up the soft-IOC with the rest of the stack
   osprey web         # the agent talks to real Channel Access

``osprey up`` brings up more than the soft-IOC. Because the preset also
declares a ``va_archiver:`` block, the deploy stands up the machine's **archive**
next to it — a MongoDB store and a recorder service — and seeds it before the
rest of the stack starts. See `The archive`_ below.

The very first ``osprey up`` that includes the Virtual Accelerator
builds its container image from source (compiling PyAT and the soft-IOC), so
expect it to take several minutes — it is building, not hanging. Later deploys
reuse the image.

If your deployment came from a preset or profile that selects a different
connector, point it at the soft-IOC explicitly:

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
(``mongodb_archiver`` for the store the preset deploys), or stay on ``mock`` for
an honestly storeless deployment. `The honesty rule`_ below explains why.

Switching back to the mock
==========================

An environment with no containers to depend on can run the tutorial on the
in-process simulation instead:

.. code-block:: bash

   osprey set connector=mock
   osprey build
   osprey up

Read one consequence before you do: **scans become browse-only.** The mock
does not settle-wait a corrector's readback against its setpoint, which every
scan plan needs between grid points, so a scan started there would never
complete. Rather than let one start and hang, the stack refuses earlier — plans
can still be listed, authored, validated and staged into the shared draft, but
the queue will not hold them, and both the panels and the agent report a
browse-only deployment with the exact command that flips it back. Everything
that is not a scan — channel reads and writes, the archiver, the Channel
Finder — works as before. The ``epics`` block keeps its production values
throughout.

The archive follows the flip on its own. The recorder records **only** a virtual
accelerator, so on ``mock`` it stops writing and idles; it re-reads the project's
``config.yml`` every 30 seconds, so the change takes effect within one poll and
no restart or rebuild is involved. Nothing is deleted — the history already in
the store stays readable, it simply stops growing, and it ages out under the
retention window as usual. ``osprey health`` will report the archive as **stale**
(a warning, not an error) once the newest sample is older than the freshness
threshold, which is the honest answer to "is this archive still being written".
Flipping back to ``virtual_accelerator`` restarts recording within a poll too.

Connecting to the IOC
=====================

The container serves Channel Access on ``127.0.0.1:5064`` in EPICS name-server
mode — the one host-to-container configuration that works reliably across
container runtimes, since broadcast discovery does not cross the container VM
boundary. The project's ``virtual_accelerator`` connector block is configured to
match and sets ``EPICS_CA_NAME_SERVERS`` itself, so no client-side EPICS
environment setup is needed.

Running from a source checkout
==============================

If you are working from an OSPREY **source checkout** rather than a generated
project — developing the IOC itself, or running it without deploying a stack —
launch the container directly:

.. code-block:: bash

   ./scripts/va/run_va.sh [DATA_DIR]

The image is defined under ``docker/virtual-accelerator/``; see its
``README.md`` for build details. The script builds the image if it is missing
(``OSPREY_VA_REBUILD=1`` forces a rebuild) and runs in the foreground.

.. warning::

   ``DATA_DIR`` is the ``data/simulation`` **directory** (never a single file)
   that the container mounts read-only. It defaults to the *packaged preset's*
   copy, **not** your project — so with no argument, ``osprey sim apply`` in
   your project writes a scenario file the running IOC never sees. Pass your
   project's directory explicitly to use its scenarios (the script then also
   mounts the sibling ``_agent_data/simulation`` state directory, which is what
   makes scenario switches reach the IOC):

   .. code-block:: bash

      ./scripts/va/run_va.sh ~/my-project/data/simulation

Scenarios
=========

``osprey sim apply <scenario>`` works in Virtual Accelerator mode exactly as it
does for the mock. Applying a scenario writes the project's
``_agent_data/simulation/active_scenarios`` file; the in-container engine polls
it and, within about a second, composed channel values reflect the new scenario.
One behavioral difference from the mock: in VA mode a scenario switch only
refreshes the engine-composed channels — setpoints you wrote during the session
live in the IOC's own records and **survive** the switch. (In mock mode, written
values are reset.)

The container mounts two of the project's directories: ``data/simulation`` for
the machine model (rebuilt from your profile on every build) and
``_agent_data/simulation`` for that scenario state (written while the system
runs). Both are automatic for the deployed service; if you launched the
container by hand, see the warning under `Running from a source checkout`_.

Write limits
============

Channels listed in the project's ``channel_limits.json`` carry a min/max range,
and a write outside that range is rejected before it reaches the IOC; an
in-range write goes through. The mandatory write-approval flow and the
``control_system.writes_enabled`` switch apply unchanged — the Virtual
Accelerator connector inherits the same write-safety wiring as the EPICS
connector.

.. note::

   The tutorial runs the limits checker in permissive mode
   (``limits_checking.allow_unlisted_channels: true``), so a channel *absent*
   from ``channel_limits.json`` is not blocked. Range enforcement covers listed
   channels; it is not a closed allowlist here.

The archive
===========

A simulated machine still needs somewhere to keep what its channels did, and the
stack deploys one. ``osprey up`` brings up two more containers beside the
soft-IOC:

- **the store** — a MongoDB service (``archiver-mongodb`` on the deployment's
  network, published on host port 27017 by default), holding one collection of
  timestamped samples;
- **the recorder** — a small service that reads the running machine on a fixed
  cadence and writes what answered into that collection.

The project's archiver connector (``archiver.type: mongodb_archiver``) reads
history back out of the same collection, so what the agent plots is what the
deployment recorded.

What history is there
---------------------

Two halves of one timeline, and the deploy makes both true before you ask
anything of them.

**The seeded past.** The first deploy writes a base series for every channel the
machine serves, covering the whole retention window — at the shipped defaults,
**30 days back**, of which the most recent **48 hours** are sampled every
10 seconds and the rest every 60. The values are generated, but they are
generated the way the live machine generates its own: each channel's history is
built around the same baseline the soft-IOC boots it at, with excursions scaled
to that channel's own noise. Nothing invents an event nobody would find in the
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

Retention is enforced by the store itself: dense samples expire after the hot
span, the coarse ones after the retention window, so a long-running deployment
stays bounded rather than growing forever. The collection is zstd-compressed;
the project's end-to-end test budgets the seeded store at under 2 GiB on disk at
these defaults.

.. note::

   Every number above is a knob in the build profile's ``va_archiver:`` block —
   ``retention_days``, ``hot_span_hours``, the cadences — not a constant in the
   code. Changing one is a profile edit and a rebuild; the next
   ``osprey up`` notices the archive no longer describes what the profile
   asks for and reseeds it. See :doc:`build-profiles`.

What the archive will not claim
-------------------------------

Ask for a window older than the archive reaches and you get **no points** —
not a plausible-looking series stretched to fill the request. ``get_metadata``
likewise reports the oldest and newest samples the collection really holds,
rather than the window the profile declared. The archive never claims more than
it has.

The honesty rule
================

There is one configuration this stack refuses: a ``virtual_accelerator`` control
system paired with the **mock archiver** — or with no archiver set at all, which
resolves to the same thing.

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
       this stack writes, or set ``type:`` under ``control_system:`` back to
       ``mock``.
   * - MCP server startup
     - The server refuses to start on such a ``config.yml``, so a file
       hand-edited after the build cannot quietly bring the pairing back.
   * - ``osprey validate``
     - Reports the same refusal without building anything, so the pairing is
       caught the moment it is written into ``profile.yml`` rather than at
       deploy time.

Two pairings that look similar are perfectly legal, because nothing lies in
either: **mock control system + mock archiver** is the honestly storeless
deployment (nothing is claimed to be real), and **EPICS + mock archiver** is a
real machine that simply has no archive attached yet.

.. warning::

   ``config.yml`` is read as **nested sections**. A top-level dotted line like
   ``archiver.type: mongodb_archiver`` added at the top of the file configures
   nothing at all — the archiver is whatever the ``archiver:`` section says.
   The refusal messages call this out when they find such a line, rather than
   reporting the key as merely unset.
