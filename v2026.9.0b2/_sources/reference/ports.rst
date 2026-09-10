.. _reference-ports:

=====
Ports
=====

Every host port a deployment publishes is one number plus a fixed offset. The
number is ``deployment.port_base``, which is ``10000`` unless you set it, and
it reserves the thousand ports from there up -- ``10000``--``10999`` by
default. Move the base and the whole deployment moves with it, so a second
deployment can share the host without being renumbered service by service:

.. code-block:: yaml

   # profile.yml
   config:
     deployment:
       port_base: 20000

With that base the web terminal answers on ``20100`` instead of ``10100``, the
database on ``20800`` instead of ``10800``, and so on for every row of the
table below.

.. _reference-ports-layout:

The layout
==========

Offsets count from ``deployment.port_base``; the **Port** column is what each
offset comes to at the default base of ``10000``. A row with a range is a band
-- one port per user, per worker, or per instance -- and the first port of the
band is the one a build publishes when only one of that thing exists. The
virtual accelerator is the exception, and has its own section below.

.. osprey-ports::

**Service** is the slot name. It is what a deployment template spells as
``osprey_ports.<slot>``, and for the panels it is also the port-family name the
per-user config keys are built from (``modules.web_terminals.<family>_base_port``).

Two rows share a key on purpose. The second Bluesky lane sits directly above
the first and is derived from it, so ``services.bluesky.port`` moves both.

.. _reference-ports-resolution:

Which value wins
================

A service settles on its port in this order, and the first one that is set
wins:

#. **The environment variable**, for the panels that have one. It is meant for
   a single run -- the container the build renders sets it from the config, and
   setting it by hand overrides that container's port for as long as it lives.
#. **The** ``--port`` **flag**, on the commands that take one (``osprey web``,
   ``osprey ariel web``, ``osprey artifacts``, ``osprey channel-finder``). It
   moves that one run and nothing else.
#. **The service's own key** in the rendered ``config.yml`` -- the **Override
   key** column. The value is an absolute port, not an offset, so a service
   pinned this way stays where it is put when ``port_base`` moves. Set it in
   ``profile.yml`` under ``config:`` so the next build keeps it.
#. **The layout** -- ``deployment.port_base`` plus the offset in the table.

Nothing in OSPREY reads a port from anywhere else, so a port you did not expect
is always one of these four.

The web terminal is the one row of the table with an empty environment-variable
cell that still has a variable behind it. The column lists what a panel's own
service reads, and the terminal is not one of those panels; its port comes from
``--port`` or the config. In a multi-user deployment, though, each per-user
container is started with ``OSPREY_TERMINAL_WEB_PORT`` set, and that
declaration outranks both -- it is how the reverse proxy and the container stay
agreed on which port the container is actually listening on, so nothing you
pass on the command line inside that container can desync them.

.. _reference-ports-panels:

The panel bands
===============

Each panel family owns a hundred ports, one per user: user *i* answers on the
family's first port plus *i*, so the user index reads off the port. With the
default base, the second user's terminal is on ``10101`` and their artifact
gallery on ``10201``.

Single-user mode is user index 0, which is the same slot the first user of a
multi-user deployment takes. Two deployments at the same base therefore collide
on those ports whichever mode they are in -- give the second one its own
``port_base`` rather than moving panels individually.

.. _reference-ports-workers:

The worker band
===============

The worker band is used only when event dispatch runs in host-network mode.
Under the default bridge network the workers reach the dispatcher inside the
container network and publish nothing on the host, so the band stays empty.

In host mode, worker 1 takes the first port of the band and each further worker
the next one up. ``dispatch.worker_port_stride`` (default ``1``) widens the gap
between them when a worker needs a range of its own. The band ends at
``+49``: a build whose last worker would land past that is refused, and the
message names the service the overflow would have collided with.

.. _reference-ports-facility:

The facility band
=================

The framework publishes nothing between ``+900`` and ``+999``, which is
``10900``--``10999`` at the default base. That band is yours: a facility's own
services, added to a deployment alongside the ones OSPREY ships, can claim
ports there without ever colliding with the framework. The exemplar profile's
facility MCP server takes the first of them, ``10900``.

.. _reference-ports-channel-access:

The Channel Access exception
============================

Virtual-accelerator instance 1 is the one port ``port_base`` does not move. It
serves EPICS on ``5064``, the Channel Access protocol port, so that clients
configured for a real facility reach it without being reconfigured. A second
deployment on the same host that also runs a virtual accelerator has to move it
by hand, with ``services.virtual_accelerator.port``.

Every further virtual accelerator is inside the block, in the ``va_standin``
band, and moves with the base like everything else.

.. _reference-ports-keys:

The keys
========

All three are set in ``profile.yml``, but not at the same level of it, and a
key written at the wrong level is simply not read.

.. list-table::
   :header-rows: 1
   :widths: 28 10 22 40

   * - Key
     - Default
     - Where it goes
     - What it does
   * - ``deployment.port_base``
     - ``10000``
     - Under ``config:``
     - The first port of the deployment's block. Everything in the table above
       is this number plus an offset.
   * - ``worker_port_stride``
     - ``1``
     - Top-level ``dispatch:``
     - The gap between one dispatch worker's port and the next, in host-network
       mode. A build profile key, not a ``config:`` one -- the build writes it
       through to ``services.dispatch_worker.worker_port_stride`` in the
       rendered config.
   * - Each row's **Override key**
     - --
     - Under ``config:``
     - An absolute port for that one service, ignoring the layout.

.. code-block:: yaml

   # profile.yml
   dispatch:
     network: host
     worker_port_stride: 10

   config:
     deployment:
       port_base: 20000

A base below ``1024`` is refused: those ports are privileged, and the
deployment could not bind its own block without running as root. A base whose
block would run past ``65535`` is refused for the same practical reason. A base
of ``32768`` or above builds, but warns -- part of the block then overlaps the
range the kernel hands out for outgoing connections, where a service can find
its port already taken.
