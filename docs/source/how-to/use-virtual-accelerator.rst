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
   * - ``mock`` *(default)*
     - The in-process simulation. No container, no network — every channel
       returns a synthesized value. This is the tutorial's default and needs no
       setup.
   * - ``virtual_accelerator``
     - A containerized PyAT soft-IOC serving real EPICS Channel Access. Storage-
       ring magnet setpoints drive a live lattice and BPM readbacks respond;
       every other channel is composed by the same simulation engine the mock
       uses. Requires the container (below).
   * - ``epics``
     - Production EPICS, pointed at the facility gateway. Untouched by this
       guide.

The Virtual Accelerator is a **local physics simulator**, not a digital twin —
it is not synced to any real machine. The OSPREY agent reads and writes it
exactly as it does the mock or a real machine; only the backend changes.

Quickstart
==========

The Control Assistant stack **already deploys** the Virtual Accelerator: the
preset's ``virtual_accelerator:`` block renders a compose service, so
``osprey deploy up`` brings the soft-IOC up alongside the rest of the stack.
What ships pointed at ``mock`` is the *connector*, not the container — so
switching backends is a config change, not a new deployment.

.. code-block:: bash

   # 1. Point the project at the soft-IOC it already deploys
   osprey config set-control-system virtual_accelerator

   # 2. Re-deploy so the running services pick up the new connector
   osprey deploy up

   # 3. Run the assistant as usual — the agent now talks to real Channel Access
   osprey web

Switch back to the mock at any time with
``osprey config set-control-system mock``. The ``epics`` block keeps its
production values throughout.

.. note::

   Step 2 is what makes the switch take effect for **deployed services**.
   ``set-control-system`` edits the project's ``config.yml``, but services do
   not read that file directly — each gets a copy staged into its own directory
   at deploy time. A purely local ``osprey web`` run picks the change up
   immediately; anything already running in a container does not, until you
   re-deploy. No image rebuild is involved either way.

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
   project's directory explicitly to use its scenarios:

   .. code-block:: bash

      ./scripts/va/run_va.sh ~/my-project/data/simulation

Scenarios
=========

``osprey sim apply <scenario>`` works in Virtual Accelerator mode exactly as it
does for the mock. Applying a scenario writes the project's ``active_scenarios``
file; the in-container engine polls it and, within about a second, composed
channel values reflect the new scenario. Switching scenarios also resets any
values you wrote during the session, mirroring the mock engine's semantics — the
simulation engine is the single implementation behind both backends, so they
cannot drift apart.

This assumes the IOC is reading the same project's ``data/simulation``. That is
automatic for the deployed service; if you launched the container by hand, see
the warning under `Running from a source checkout`_.

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
