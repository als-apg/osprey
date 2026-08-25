.. _how-to-use-connectors:

Use Connectors
==============

**What you'll build:** Control system connectors for accessing hardware abstraction layers

Overview
--------

The Control System Integration system provides a **two-layer abstraction** for working with control systems and archivers. This enables development and R&D work using mock connectors (without hardware access) and migration to production by changing ``control_system.type`` and its connector block, not code.

**Capabilities:**

- **Mock Mode**: Work with any channel names without hardware access
- **Production Mode**: EPICS and DOOCS ship in-tree; LabVIEW, Tango, and other stacks via user-registered custom connectors
- **One API**: the same code works with mock and production connectors
- **Custom connectors**: register your own via ``ConnectorFactory``

**Built-in Connectors:**

- **mock** / **mock_archiver**: Development/R&D mode (no hardware access required)
- **epics** / **epics_archiver**: EPICS Channel Access / Archiver Appliance (production)
- **virtual_accelerator**: the containerized Virtual Accelerator, over real EPICS
  Channel Access — behaves like ``epics`` but tracks setpoints through the
  simulator's LUME-backed physics, so plans actually run (the mock connector
  can't do that); see :doc:`use-virtual-accelerator` and
  :doc:`/architecture/virtual-accelerator`
- **mongodb_archiver**: MongoDB time-series archiver
- **doocs** / **doocs_archiver**: DOOCS properties and DOOCS local histories
  (DESY, European XFEL). Both require ``doocs4py``, which is supplied by the
  DOOCS environment rather than installed from PyPI — the import is deferred to
  ``connect()``, so the names register everywhere and only fail where a DOOCS
  environment is genuinely absent.


Quick Start: Using Connectors
-----------------------------

Mock Mode (Development & R&D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory

   # Create mock connector - works with ANY channel names
   connector = await ConnectorFactory.create_control_system_connector({
       'type': 'mock',
       'connector': {
           'mock': {
               'response_delay_ms': 10,
               'noise_level': 0.01
           }
       }
   })

   channel_value = await connector.read_channel('ANY:MADE:UP:NAME')
   print(f"Value: {channel_value.value} {channel_value.metadata.units}")

   # A state channel (EPICS mbbi/bi/bo, PVAccess NTEnum) reads as its integer
   # state index, with the state names alongside it:
   mode = await connector.read_channel('SR:DIAG:MODE')
   print(f"{mode.value} means {mode.metadata.enum_label}")   # e.g. 2 means ACQUIRING
   print(mode.metadata.enum_labels)  # ['OFFLINE', 'STANDBY', 'ACQUIRING', 'FAULT']

   await connector.disconnect()

Production Mode (EPICS)
~~~~~~~~~~~~~~~~~~~~~~~

Switch to real hardware by changing ``type`` and adding the connector's own block:

.. code-block:: yaml

   # Mock (default, for development):
   control_system:
     type: mock
     connector:
       mock: { response_delay_ms: 10, noise_level: 0.01 }

   # Production:
   control_system:
     type: epics
     connector:
       epics:
         gateways:
           # EPICS uses one process-wide CA context, so the connector points at a
           # single gateway. When control_system.writes_enabled is true and a
           # write_access gateway is set, writes route through it; otherwise the
           # connector uses read_only (so a read-only deployment rejects writes at
           # the network layer as well).
           read_only:   { address: cagw.facility.edu, port: 5064 }
           write_access: { address: cagw.facility.edu, port: 5084 }
         timeout: 5.0

The Python API is identical -- only the config changes.

**Archiver configuration** uses a parallel ``archiver:`` block. Switch from the mock
archiver (synthetic data) to the EPICS Archiver Appliance the same way:

.. code-block:: yaml

   # Mock archiver (default, for development):
   archiver:
     type: mock_archiver

   # Production:
   archiver:
     type: epics_archiver
     epics_archiver:
       url: https://archiver.facility.edu:8443   # required
       timeout: 60                                # seconds, default 60

.. note::

   Write operations require explicit opt-in. See :ref:`write-safety-config` in
   :doc:`/reference/contracts/connectors` for the ``writes_enabled`` setting that
   controls write permissions.

PVAccess Channels (PVA)
~~~~~~~~~~~~~~~~~~~~~~~

The ``epics`` connector speaks Channel Access by default. Some data -- camera
frames above all -- is served only over pvAccess, EPICS' second protocol. List
those addresses as glob patterns under ``pva_channels`` and the connector reads
them through pvAccess, while every other address keeps using Channel Access. One
deployment covers both:

.. code-block:: yaml

   control_system:
     type: epics
     connector:
       epics:
         timeout: 5.0
         pva_channels:
           - "*:IMAGE*"
           - "BL:CAM?:ARRAY"
         pva_gateway:
           address: pvagw.facility.edu
           port: 5075               # default
           use_name_server: false
         gateways:
           read_only: { address: cagw.facility.edu, port: 5064 }

Patterns are matched with ``fnmatch.fnmatchcase`` -- case-sensitive, and the
same on every operating system. Addresses stay exactly as you write them: no
``pva://`` prefix is ever added, so the limits database, channel databases and
audit records keep matching the strings they already use. An absent or empty
``pva_channels`` list is a complete no-op -- the connector behaves exactly as it
did before and never even imports the pvAccess client.

``pva_gateway`` is a single flat block, unlike the ``read_only`` /
``write_access`` pair on the Channel Access side, because pvAccess is read-only
here. The block is only consulted when ``pva_channels`` is non-empty, and it becomes
``EPICS_PVA_ADDR_LIST`` in the form ``address:port`` -- or
``EPICS_PVA_NAME_SERVERS``, same form, when ``use_name_server`` is true, which
makes the client connect over TCP instead of searching by UDP broadcast (that is
what an SSH tunnel needs). Whenever the block is present,
``EPICS_PVA_AUTO_ADDR_LIST`` is forced to ``"NO"`` -- the same containment the
Channel Access side applies, so a deployment deliberately pinned to one gateway
cannot also broadcast-discover servers on the local subnet.

**PVA channels are read-only.** A write to an address matching a
``pva_channels`` pattern comes straight back as a blocked
``ChannelWriteResult`` naming the reason, with nothing sent on the network. That
is a deliberate safety decision rather than an unfinished feature: a supervised
pvAccess write path is separate work with its own review, and until it exists a
refusal is the honest answer.

pvAccess **RPC services** are refused too, and for a different reason. Code the
agent runs through the Python executor cannot call ``Context.rpc(...)``: it is
blocked at runtime and no approval can let it through, because an rpc payload
carries an arbitrary request -- there is no way to tell a read from a write, and
nothing for limits checking to validate. Use ``channel_read`` and
``channel_write`` for the operation you actually need.

A camera frame is almost always too large to return to the agent as raw
numbers; see :doc:`/reference/contracts/connectors` for what comes back instead.

.. note::

   Compressed NTNDArray frames are not decoded. When a camera's areaDetector
   pvAccess plugin is configured with a codec (blosc, lz4, jpeg), the read fails
   with an error naming the codec and the remedy -- disable pvAccess compression
   for that channel -- rather than reshaping a compressed blob into a
   plausible-looking image full of meaningless statistics.

.. note::

   The pvAccess client (``p4p``) ships with OSPREY as an ordinary dependency, so
   there is normally nothing to install. The exception is a bare-metal install on
   arm64 Linux, where ``p4p``, ``pvxslibs`` and ``epicscorelibs`` publish no
   wheels and pip builds them from source: that host needs a C toolchain
   (a compiler and ``make``) present. OSPREY's own project images already stage
   one, so containerized deployments are unaffected.

MongoDB Archiver
~~~~~~~~~~~~~~~~

For facilities that store time-series PV data in MongoDB rather than EPICS Archiver
Appliance, configure the archiver block independently of the control-system choice:

.. code-block:: yaml

   archiver:
     type: mongodb_archiver
     mongodb_archiver:
       host: mongodb.facility.edu
       port: 27017
       name: archiver_db
       collection: pv_data
       auth: admin
       username: readonly
       password_env: MONGODB_READONLY_PASSWORD

Documents in the collection are expected to have a ``date`` field (``ISODate``) and
one or more PV names as top-level fields: ``{date: ISODate(...), PV1: value1, PV2:
value2, ...}``. A query matches any document that carries **at least one** of the
requested PVs (an ``$or`` across per-PV ``$exists`` checks) -- documents do not need
to carry every requested PV together, so channels archived at different cadences, or
written into separate documents by different collectors, are still returned
correctly, each on its own timestamp series. The connector's MongoDB client ships
with OSPREY, so there is nothing extra to install.

Production Mode (DOOCS)
~~~~~~~~~~~~~~~~~~~~~~~

DOOCS facilities select both connectors by name. Channel addresses are DOOCS
properties (``FACILITY/DEVICE/LOCATION/PROPERTY``), and the control-system
connector needs no options -- it reads its environment from the DOOCS
installation:

.. code-block:: yaml

   control_system:
     type: doocs

   archiver:
     type: doocs_archiver
     doocs_archiver:
       avg_window: 20    # optional moving average, in samples

The archiver reads DOOCS *local histories*, so it only makes sense alongside
``type: doocs``. Both connectors need ``doocs4py``, which the DOOCS environment
provides rather than PyPI; without it, ``connect()`` fails with a clear
``ImportError`` instead of silently degrading.

.. note::

   DOOCS supports the ``none`` and ``readback`` write-verification levels.
   ``callback`` is accepted but has no DOOCS equivalent, so it performs a
   readback and reports the level as ``readback``.



Contracts and Custom Connectors
-------------------------------

The behaviour every connector has to implement -- how large array values come
back, how a write reports whether it took effect and which safety gates it must
pass first, and what an archiver's historical data must look like -- is
documented in :doc:`/reference/contracts/connectors`.

To write, register and test a connector — control system or archiver — for a
system OSPREY does not ship, see :doc:`/contributing/extending-osprey`.

.. seealso::

   :doc:`/reference/contracts/connectors`
       Connector contracts: reading large values, write verification and its
       safety gates, and the archiver ``get_data`` contract.
