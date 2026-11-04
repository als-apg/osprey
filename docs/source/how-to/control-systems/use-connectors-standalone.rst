.. _how-to-use-connectors-standalone:

Use the Connectors Without the Framework
========================================

``osprey-connectors`` installs on its own, without the agent, the web terminal
or a built deployment. A script, a notebook or a service of your own can read a
control system and its archiver through the same connectors OSPREY uses, and
gets their error taxonomy and their write safety with them.

.. code-block:: bash

   pip install osprey-connectors

This page covers reading. A connector built in a script refuses every write
before anything is sent; `When a write is refused`_ says why and where writing
is decided.

Read channels
-------------

A connector is built from a configuration dict by its type name. The dict has
the shape of a deployment's ``control_system:`` section:

.. code-block:: python

   import asyncio

   from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

   CONFIG = {
       "type": "epics",
       "connector": {
           "epics": {
               "gateways": {"read_only": {"address": "cagw.facility.edu", "port": 5064}},
               "timeout_s": 5.0,
           }
       },
   }


   async def main() -> None:
       register_builtin_connectors()   # registers the built-in type names; idempotent
       connector = await ConnectorFactory.create_control_system_connector(CONFIG)
       try:
           reading = await connector.read_channel("DEMO:READBACK")
           print(reading.value, reading.metadata.units)

           values = await connector.read_multiple_channels(["DEMO:SETPOINT", "DEMO:MODE"])
           for address, channel in values.items():
               print(address, channel.value, channel.metadata.enum_label)
       finally:
           await connector.disconnect()


   asyncio.run(main())

Against an IOC serving those three records, it prints:

.. code-block:: text

   1.5 A
   DEMO:SETPOINT 1.5 None
   DEMO:MODE 1 STANDBY

``create_control_system_connector`` connects before it returns. A state channel
reads as its integer index, with the state's name in ``metadata.enum_label``.
``read_multiple_channels`` leaves out a channel that failed to read rather than
raising, so check the keys of what it returns.

Every connector answers the same methods. To read a DOOCS or Tango control
system, change ``type`` and its ``connector`` block;
:doc:`use-connectors` has the block for each, and the client library the DOOCS
and Tango connectors need, which this package does not install.

Read history
------------

An archiver connector is built the same way, from the shape of a deployment's
``archiver:`` section, and returns history as a pandas ``DataFrame``:

.. code-block:: python

   import asyncio
   from datetime import datetime, timedelta, timezone

   from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

   ARCHIVER = {
       "type": "epics_archiver",
       "settings": {"url": "https://archiver.facility.edu:8443", "timeout_s": 30},
   }


   async def main() -> None:
       register_builtin_connectors()
       archiver = await ConnectorFactory.create_archiver_connector(ARCHIVER)
       try:
           end = datetime.now(timezone.utc)
           frame = await archiver.get_data(
               channels=["DEMO:READBACK", "DEMO:SETPOINT"],
               start_date=end - timedelta(hours=1),
               end_date=end,
           )
           print(frame)
       finally:
           await archiver.disconnect()


   asyncio.run(main())

The frame is long: one row per sample, with ``timestamp`` (UTC), ``channel``
and ``value`` columns:

.. code-block:: text

                     timestamp        channel  value
   0 2026-11-04 23:33:24+00:00  DEMO:READBACK    1.5
   1 2026-11-04 23:34:24+00:00  DEMO:READBACK    1.6
   2 2026-11-04 23:35:24+00:00  DEMO:READBACK    1.7
   3 2026-11-04 23:33:24+00:00  DEMO:SETPOINT    1.5
   ...

``get_data`` also takes ``precision_ms``, a bin width, and ``processing``, how
each bin is reduced; the archiver section of
:doc:`/reference/contracts/connectors` says what each mode returns.

What needs a deployment
-----------------------

Three things on the connector side are a deployment's, and a script without one
does without them:

The simulator
   ``virtual_accelerator`` and ``mock_archiver`` serve the simulated machine of
   a built facility, and refuse to connect without the simulator view
   ``osprey build`` writes. Every other connector talks to a real system.

``config.yml``
   A few facts are deployment-wide and read from ``config.yml`` rather than
   from the dict a connector is built from: the facility's timezone, the write
   tier and the channel limits. A script that has none reads them at their
   defaults and logs a warning that it fell back to UTC. To set the timezone, name
   a ``config.yml`` in ``CONFIG_FILE``:

   .. code-block:: yaml

      facility_timezone: America/Los_Angeles

Writes
   Every write passes the verdict in :ref:`write-safety-config`: the write tier,
   the machine the connector writes to, the login it writes for, and that
   login's switch for that machine, the first refusal winning. A connector built
   in a script names no machine and has no switch to read, so every write is
   refused before anything is sent.

When a write is refused
-----------------------

The refusal is the package's ordinary one.
:meth:`~osprey_connectors.control_system.ControlSystemConnector.write_channel`
returns a result whose ``outcome`` is ``refused``;
``write_channel_checked`` raises
:class:`~osprey_connectors.errors.ChannelWriteBlockedError`, whose message ends
with the reason:

.. code-block:: python

   from osprey_connectors.errors import ChannelWriteBlockedError

   try:
       await connector.write_channel_checked("DEMO:SETPOINT", 2.0)
   except ChannelWriteBlockedError as refused:
       print(refused.reason)   # WRITES_DISABLED
       print(refused)          # ... [osprey-refusal: tier_never]

``tier_never`` is the answer with no ``config.yml``; with the tier armed in one,
the next term answers ``target_unknown``. Writing is decided by a deployment, so
a program that has to write runs in one and is let write by a person: from the
switch, or, for a program with nobody at a prompt, through a bounded
:ref:`write authorization <how-to-write-from-a-program>`.

.. seealso::

   :doc:`/reference/connectors-api/index`
       Every public name of the package.

   :doc:`use-connectors`
       The configuration block of every connector.

   :doc:`/reference/contracts/connectors`
       What every connector must do: large values, write confirmation, who may
       write, and the archiver contract.
