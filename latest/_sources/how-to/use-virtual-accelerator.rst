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
   - Starting the container and pointing the tutorial at it
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

.. code-block:: bash

   # 1. Start the Virtual Accelerator container (serves EPICS CA on localhost)
   ./scripts/va/run_va.sh

   # 2. Point the tutorial project at it
   osprey config set-control-system virtual_accelerator

   # 3. Run the assistant as usual — the agent now talks to real Channel Access
   osprey web

Switch back to the mock at any time with
``osprey config set-control-system mock``. The ``epics`` block keeps its
production values throughout.

Starting the container
======================

The Virtual Accelerator image is defined under ``docker/virtual-accelerator/``;
see its ``README.md`` for build details. Launch it with the provided script:

.. code-block:: bash

   ./scripts/va/run_va.sh

The script publishes Channel Access on ``127.0.0.1:5064`` using EPICS
name-server mode — the one host-to-container configuration that works reliably
across container runtimes (broadcast discovery does not cross the container VM
boundary). The tutorial's ``virtual_accelerator`` connector block is already
configured to match, so no client-side EPICS environment setup is needed.

.. note::

   The container mounts the project's ``data/simulation`` **directory** (never a
   single file) read-only, so ``osprey sim apply`` on the host stays visible to
   the running IOC. This is handled for you by ``run_va.sh``.

Scenarios
=========

``osprey sim apply <scenario>`` works in Virtual Accelerator mode exactly as it
does for the mock. Applying a scenario writes the project's ``active_scenarios``
file; the in-container engine polls it and, within about a second, composed
channel values reflect the new scenario. Switching scenarios also resets any
values you wrote during the session, mirroring the mock engine's semantics — the
simulation engine is the single implementation behind both backends, so they
cannot drift apart.

Write limits
============

Every writable setpoint carries a min/max range in the project's
``channel_limits.json``. A write outside a channel's range is rejected before it
reaches the IOC; an in-range write goes through. The mandatory write-approval
flow and the ``control_system.writes_enabled`` switch apply unchanged — the
Virtual Accelerator connector inherits the same write-safety wiring as the EPICS
connector.

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
