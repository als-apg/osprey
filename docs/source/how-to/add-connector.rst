==========================
Control System Integration
==========================

**What you'll build:** Control system connectors for accessing hardware abstraction layers

Overview
========

The Control System Integration system provides a **two-layer abstraction** for working with control systems and archivers. This enables development and R&D work using mock connectors (without hardware access) and seamless migration to production by changing a single configuration line.

**Key Features:**

- **Mock Mode**: Work with any channel names without hardware access
- **Production Mode**: Real control system connectors (EPICS, LabVIEW, Tango, custom)
- **Unified API**: Same code works with mock and production connectors
- **Pluggable Architecture**: Register custom connectors via ``ConnectorFactory``

**Built-in Connectors:**

- **mock** / **mock_archiver**: Development/R&D mode (no hardware access required)
- **epics** / **epics_archiver**: EPICS Channel Access / Archiver Appliance (production)


Quick Start: Using Connectors
=============================

Mock Mode (Development & R&D)
------------------------------

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
   await connector.disconnect()

Production Mode (EPICS)
-----------------------

Switch to real hardware by changing ``type`` in ``config.yml``:

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
           read_only: { address: cagw.facility.edu, port: 5064 }
           read_write: { address: cagw-rw.facility.edu, port: 5065 }
         timeout: 5.0

The Python API is identical -- only the config changes.


.. admonition:: PLACEHOLDER: Orchestration / Approval
   :class: warning

   **Old content (line 258-315):** Pattern detection section describing how pattern detection feeds into the approval system for generated Python code.
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old approval workflow (pattern detection -> capability_node approval) is deleted. Claude Code has its own tool-approval mechanism, but how pattern detection integrates with it (if at all) is unclear.
   **Action needed:** Does pattern detection still exist? If so, document how it connects to Claude Code's approval flow. If not, delete this placeholder.


Write Verification
==================

All ``write_channel()`` calls return :class:`~osprey.connectors.control_system.base.ChannelWriteResult`:

.. code-block:: python

   connector = await ConnectorFactory.create_control_system_connector()

   result = await connector.write_channel("BEAM:CURRENT", 100.0)

   if result.verification and result.verification.verified:
       print(f"Write confirmed ({result.verification.level})")
   else:
       print(f"Verification failed: {result.verification.notes}")

   # Override verification level
   result = await connector.write_channel(
       "MOTOR:POSITION", 50.0,
       verification_level="readback",
       tolerance=0.1
   )

**Verification levels:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Level
     - Speed
     - Confidence
     - When to Use
   * - ``none``
     - Instant
     - Low
     - Development, non-critical writes
   * - ``callback``
     - Fast (~1-10ms)
     - Medium
     - Most production writes (default)
   * - ``readback``
     - Slow (~50-100ms)
     - High
     - Critical setpoints, safety-critical operations

**Configuration (global default):**

.. code-block:: yaml

   control_system:
     write_verification:
       default_level: "callback"
       default_tolerance_percent: 0.1

**Per-channel configuration** (in limits database):

.. code-block:: json

   {
     "MOTOR:POSITION": {
       "min_value": -100.0,
       "max_value": 100.0,
       "verification": {
         "level": "readback",
         "tolerance_absolute": 0.1
       }
     }
   }

.. seealso::

   :class:`~osprey.connectors.control_system.base.ChannelValue`
       Channel read result data model

   :class:`~osprey.connectors.control_system.base.ChannelWriteResult`
       Complete write operation result

   :class:`~osprey.connectors.control_system.base.WriteVerification`
       Verification result data model


Implementing Custom Connectors
==============================

Subclass :class:`~osprey.connectors.control_system.base.ControlSystemConnector` and implement the abstract methods: ``connect``, ``disconnect``, ``read_channel``, ``write_channel``, ``read_multiple_channels``, ``subscribe``, ``unsubscribe``, ``get_metadata``, ``validate_channel``.

The old doc included a full LabVIEW example connector. The API surface it targeted (``ChannelValue``, ``ChannelMetadata``, ``ChannelWriteResult``, ``WriteVerification``) remains unchanged in ``osprey.connectors.control_system.base``.

.. admonition:: PLACEHOLDER: Connector Registration
   :class: warning

   **Old content (line 781-807):** Registration via ``ConnectorRegistration``, ``extend_framework_registry``, and ``ConnectorFactory.register_control_system()``.
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old registry-based registration (``extend_framework_registry``, ``ConnectorRegistration``) may have changed during the native-capabilities migration. The current registration path for custom connectors is unclear.
   **Action needed:** Verify how custom connectors are registered in the current codebase and document the current mechanism.

Testing Custom Connectors
-------------------------

Test in three phases:

1. **Capability logic** -- use ``type: mock`` connector, no hardware needed.
2. **Interface compliance** -- instantiate your connector against a local simulator.
3. **Integration** -- mark with ``@pytest.mark.integration``; run against real hardware.

Switch connectors via environment variables in ``conftest.py``:

.. code-block:: python

   @pytest.fixture
   def connector_config():
       if os.getenv('USE_REAL_CONNECTOR') == '1':
           return {'type': 'epics', 'connector': {'epics': {}}}
       return {'type': 'mock', 'connector': {'mock': {}}}
