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
   - Write limits, and the archiver live-vs-history divergence

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
compose service, so ``osprey deploy up`` brings the soft-IOC up alongside the
rest of the stack and the connector is already talking to it. There is nothing
to switch on.

.. code-block:: bash

   osprey deploy up   # brings up the soft-IOC with the rest of the stack
   osprey web         # the agent talks to real Channel Access

The very first ``osprey deploy up`` that includes the Virtual Accelerator
builds its container image from source (compiling PyAT and the soft-IOC), so
expect it to take several minutes — it is building, not hanging. Later deploys
reuse the image.

If your project was built from an older preset (or a profile that sets it),
point it at the soft-IOC explicitly:

.. code-block:: bash

   osprey config set-control-system virtual_accelerator
   osprey deploy up

.. note::

   The second command is what makes a switch take effect for **deployed
   services**. ``set-control-system`` edits the project's ``config.yml``, but
   services do not read that file directly — each gets a copy staged into its
   own directory at deploy time. A purely local ``osprey web`` run picks the
   change up immediately; anything already running in a container does not,
   until you re-deploy. No image rebuild is involved either way.

Switching back to the mock
==========================

An environment with no containers to depend on can run the tutorial on the
in-process simulation instead:

.. code-block:: bash

   osprey config set-control-system mock
   osprey deploy up

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

Archiver: live values vs. history
==================================

The mock archiver synthesizes channel *history* from ``machine.json``. That
history is independent of the Virtual Accelerator's live physics:

- In **mock** mode, a channel's live value and its archived history are both
  synthesized from ``machine.json`` — they agree by construction.
- In **virtual_accelerator** mode, live storage-ring readbacks come from the PyAT
  lattice (a corrector write recomputes the orbit), while the archiver's history
  is still the synthetic ``machine.json`` series.

So in Virtual Accelerator mode a lattice channel's **live value and its archived
history can diverge in meaning**: the live value reflects real simulated physics,
the history does not. This is expected — a VA-backed archiver is a separate,
future addition.
