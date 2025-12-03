=========================
Control System Connectors
=========================

Pluggable connector abstraction for control systems and archivers with mock and production implementations. Enables development without hardware access and seamless migration to production by changing configuration.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/06_control-system-integration`.

.. currentmodule:: osprey.connectors

Factory Classes
===============

.. autoclass:: osprey.connectors.factory.ConnectorFactory
   :members:
   :show-inheritance:

The factory provides centralized creation and configuration of connectors with plugin-style registration.

Registry Integration
====================

Connectors can be registered through the Osprey registry system for unified component management.

.. currentmodule:: osprey.registry

.. autoclass:: ConnectorRegistration
   :members:
   :show-inheritance:

Registration dataclass for control system and archiver connectors. Used to register connectors
through the registry system, providing lazy loading and unified management alongside other framework components.

**Usage Example:**

.. code-block:: python

   from osprey.registry import ConnectorRegistration, extend_framework_registry

   def get_registry_config(self):
       return extend_framework_registry(
           connectors=[
               ConnectorRegistration(
                   name="labview",
                   connector_type="control_system",
                   module_path="my_app.connectors.labview_connector",
                   class_name="LabVIEWConnector",
                   description="LabVIEW Web Services connector for NI systems"
               ),
               ConnectorRegistration(
                   name="tango",
                   connector_type="control_system",
                   module_path="my_app.connectors.tango_connector",
                   class_name="TangoConnector",
                   description="Tango control system connector"
               ),
               ConnectorRegistration(
                   name="tango_archiver",
                   connector_type="archiver",
                   module_path="my_app.connectors.tango_archiver",
                   class_name="TangoArchiverConnector",
                   description="Tango archiver connector"
               )
           ],
           capabilities=[...],
           context_classes=[...]
       )

.. currentmodule:: osprey.connectors

Control System Interfaces
=========================

Base Classes
------------

.. autoclass:: osprey.connectors.control_system.base.ControlSystemConnector
   :members:
   :show-inheritance:

Abstract base class defining the contract for all control system connectors (EPICS, LabVIEW, Tango, Mock, etc.).

Data Models
-----------

Read Operation Models
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: osprey.connectors.control_system.base.ChannelValue
   :members:
   :show-inheritance:

Result container for channel reads with value, timestamp, and metadata.

.. autoclass:: osprey.connectors.control_system.base.ChannelMetadata
   :members:
   :show-inheritance:

Metadata about a control system channel (units, precision, alarms, limits, etc.).

**Backward Compatibility:**

.. deprecated:: 0.9.5
   The classes ``PVValue`` and ``PVMetadata`` are deprecated and aliased to ``ChannelValue``
   and ``ChannelMetadata`` respectively. The "PV" terminology is EPICS-specific; "channel"
   is control-system agnostic and supports any control system (EPICS, Tango, LabVIEW, etc.).

Write Operation Models
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: osprey.connectors.control_system.base.ChannelWriteResult
   :members:
   :show-inheritance:

Result container for channel write operations with success status, written value, and optional verification.

**Fields:**

- ``success`` (bool): Whether the write operation succeeded
- ``written_value`` (Any): The value that was written to the channel
- ``verification`` (Optional[WriteVerification]): Verification result if verification was requested

**Example:**

.. code-block:: python

   # Automatic verification (uses per-channel or global config)
   result = await connector.write_channel('BEAM:SETPOINT', 450.0)

   if result.success and result.verification and result.verification.verified:
       print(f"Write verified ({result.verification.level}): {result.written_value}")
   elif result.success:
       print(f"Write succeeded but not verified")
   else:
       print(f"Write failed")

   # Manual override (optional)
   result = await connector.write_channel(
       'BEAM:SETPOINT',
       450.0,
       verification_level='readback',  # Override auto-config
       tolerance=0.01
   )

.. autoclass:: osprey.connectors.control_system.base.WriteVerification
   :members:
   :show-inheritance:

Verification result for channel write operations.

**Fields:**

- ``verification_level`` (str): Level of verification performed ('none', 'callback', 'readback')
- ``verified`` (bool): Whether the write was successfully verified
- ``readback_value`` (Optional[Any]): The value read back from the channel (readback mode only)
- ``tolerance_check`` (Optional[bool]): Whether readback matched within tolerance (readback mode only)

**Verification Levels:**

1. **'none'**: No verification performed (``verified=False``)
2. **'callback'**: Uses Channel Access callback to confirm write (EPICS only)
3. **'readback'**: Reads back the value and compares with tolerance

**Example:**

.. code-block:: python

   # Automatic verification (connector determines level from config)
   result = await connector.write_channel('BEAM:CURRENT', 400.0)

   if result.verification and result.verification.verified:
       print(f"Verification: {result.verification.level}")
       if result.verification.readback_value is not None:
           print(f"Readback: {result.verification.readback_value}")
   else:
       print("Verification failed")

Built-in Implementations
------------------------

.. autoclass:: osprey.connectors.control_system.mock_connector.MockConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Development connector that accepts any PV names and generates realistic simulated data. Ideal for R&D when you don't have control room access.

**Key Features:**

- Accepts any PV name (no real control system required)
- Configurable response delays and noise levels
- Realistic units, timestamps, and metadata
- Optional write operation support

.. autoclass:: osprey.connectors.control_system.epics_connector.EPICSConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Production EPICS Channel Access connector using ``pyepics`` library. Supports gateway configuration for secure access.

**Key Features:**

- EPICS Channel Access protocol
- Read-only and read-write gateway support
- Configurable timeouts and retry logic
- Full metadata support (units, precision, alarms, limits)

**Requirements:**

- ``pyepics`` library: ``pip install pyepics``
- Access to EPICS gateway or IOCs

Archiver Interfaces
===================

Base Classes
------------

.. autoclass:: osprey.connectors.archiver.base.ArchiverConnector
   :members:
   :show-inheritance:

Abstract base class defining the contract for all archiver connectors.

.. autoclass:: osprey.connectors.archiver.base.ArchivedData
   :members:
   :show-inheritance:

Container for historical time series data with timestamps and values.

Built-in Implementations
------------------------

.. autoclass:: osprey.connectors.archiver.mock_archiver_connector.MockArchiverConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Development archiver that generates synthetic historical data. Ideal for R&D when you don't have archiver access.

**Key Features:**

- Generates realistic time series with trends and noise
- Configurable retention period and sample rates
- Works with any PV names
- Consistent with mock control system connector

.. autoclass:: osprey.connectors.archiver.epics_archiver_connector.EPICSArchiverConnector
   :members:
   :show-inheritance:
   :exclude-members: connect, disconnect

Production connector for EPICS Archiver Appliance using ``archivertools`` library.

**Key Features:**

- EPICS Archiver Appliance integration
- Efficient bulk data retrieval
- Configurable precision and time ranges
- Connection pooling for performance

**Requirements:**

- ``archivertools`` library: ``pip install archivertools``
- Access to EPICS Archiver Appliance URL

Pattern Detection
=================

.. currentmodule:: osprey.services.python_executor.pattern_detection

Static code analysis for detecting control system operations in generated code. **Critical security layer** that catches both approved API usage and circumvention attempts.

.. autofunction:: detect_control_system_operations

Analyzes Python code using framework-standard or custom patterns to detect control system operations.
The framework provides comprehensive security-focused patterns by default - no configuration needed.

**Security Purpose:** Detects both approved ``osprey.runtime`` API usage AND direct control system library
calls that would bypass connector safety features (limits checking, verification, approval workflows).

**Example:**

.. code-block:: python

   from osprey.services.python_executor.analysis.pattern_detection import detect_control_system_operations

   # Detects approved API
   code_approved = "write_channel('BEAM:CURRENT', 500)"
   result = detect_control_system_operations(code_approved)
   # result['has_writes'] == True

   # Also detects circumvention attempts
   code_circumvent = "epics.caput('BEAM:CURRENT', 500)"
   result = detect_control_system_operations(code_circumvent)
   # result['has_writes'] == True  # Caught by security layer!

.. autofunction:: get_framework_standard_patterns

Returns framework-standard security-focused patterns. These patterns detect:

- ✅ Approved ``osprey.runtime`` API (with all safety features)
- 🔒 EPICS direct calls (``epics.caput``, ``PV().put`` - bypasses safety)
- 🔒 Tango direct calls (``DeviceProxy().write_attribute`` - bypasses safety)
- 🔒 LabVIEW integration patterns (bypasses safety)
- 🔒 Direct connector access (advanced use)

Configuration Schema
====================

Control System Configuration
----------------------------

Control system connector configuration in ``config.yml``:

.. code-block:: yaml

   control_system:
     type: mock | epics | labview | tango | custom

     # Pattern detection is automatic - framework provides defaults
     # Optional: Override for custom workflows (rarely needed)
     # patterns:
     #   write:
     #     - 'custom_write_function\('
     #   read:
     #     - 'custom_read_function\('

     # Type-specific connector configurations
     connector:
       mock:
         response_delay_ms: 10
         noise_level: 0.01
         enable_writes: true

      epics:
        gateways:
          read_only:
            address: cagw.facility.edu
            port: 5064
            use_name_server: false    # Use EPICS_CA_NAME_SERVERS vs CA_ADDR_LIST (default: false)
          read_write:
            address: cagw-rw.facility.edu
            port: 5065
            use_name_server: false
        timeout: 5.0
        retry_count: 3
        retry_delay: 0.5

Archiver Configuration
----------------------

Archiver connector configuration in ``config.yml``:

.. code-block:: yaml

   archiver:
     type: mock_archiver | epics_archiver | custom_archiver

     # Mock archiver uses sensible defaults
     mock_archiver:
       sample_rate_hz: 1.0
       noise_level: 0.01

     epics_archiver:
       url: https://archiver.facility.edu:8443
       timeout: 60
       max_retries: 3
       verify_ssl: true
       pool_connections: 10
       pool_maxsize: 20

Usage Examples
==============

Basic Usage
-----------

Create and use a connector from global configuration:

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory

   # Create connector from config.yml
   connector = await ConnectorFactory.create_control_system_connector()

   try:
       # Read a channel
       result = await connector.read_channel('BEAM:CURRENT')
       print(f"Current: {result.value} {result.metadata.units}")

       # Read multiple channels
       results = await connector.read_multiple_channels([
           'BEAM:CURRENT',
           'BEAM:LIFETIME',
           'BEAM:ENERGY'
       ])

       # Get metadata
       metadata = await connector.get_metadata('BEAM:CURRENT')
       print(f"Units: {metadata.units}, Range: {metadata.min_value}-{metadata.max_value}")

   finally:
       await connector.disconnect()

Usage in Generated Python Code
------------------------------

When the Python execution service generates code that needs to interact with control systems, it uses the ``osprey.runtime`` module instead of direct connector imports. This provides a simple, synchronous API that works with any configured control system:

.. code-block:: python

   # In generated Python code
   from osprey.runtime import write_channel, read_channel

   # Read from control system (synchronous, like EPICS caget)
   current = read_channel("BEAM:CURRENT")
   print(f"Current: {current} mA")

   # Write to control system (synchronous, like EPICS caput)
   write_channel("MAGNET:SETPOINT", 5.0)

**Key Benefits:**

- **Control-System Agnostic**: Same code works with EPICS, Mock, LabVIEW, or any registered connector
- **Automatic Configuration**: Uses control system settings from execution context (reproducible notebooks)
- **Safety Integration**: All boundary checking, limits validation, and approval workflows happen automatically
- **Simple API**: Synchronous functions (async handled internally)

The runtime module is automatically configured by the execution wrapper and uses the same connector configuration shown above.

.. seealso::
   :doc:`../../../getting-started/control-assistant-part3-production`
      Complete tutorial on how generated code interacts with control systems

Custom Configuration
--------------------

Create connector with inline configuration:

.. code-block:: python

   # Mock connector with custom settings
   config = {
       'type': 'mock',
       'connector': {
           'mock': {
               'response_delay_ms': 5,
               'noise_level': 0.02
           }
       }
   }
   connector = await ConnectorFactory.create_control_system_connector(config)

   # EPICS connector with specific gateway
   config = {
       'type': 'epics',
       'connector': {
           'epics': {
               'gateways': {
                   'read_only': {
                       'address': 'cagw.als.lbl.gov',
                       'port': 5064
                   }
               },
               'timeout': 3.0
           }
       }
   }
   connector = await ConnectorFactory.create_control_system_connector(config)

Archiver Usage
--------------

Retrieve historical data:

.. code-block:: python

   from osprey.connectors.factory import ConnectorFactory
   from datetime import datetime, timedelta

   # Create archiver connector
   connector = await ConnectorFactory.create_archiver_connector()

   try:
       # Define time range
       end_time = datetime.now()
       start_time = end_time - timedelta(hours=24)

       # Retrieve data for multiple PVs
       data = await connector.get_data(
           pv_list=['BEAM:CURRENT', 'BEAM:LIFETIME'],
           start_date=start_time,
           end_date=end_time,
           precision_ms=1000  # 1 second precision
       )

       # Process results
       for pv_name, pv_data in data.items():
           print(f"{pv_name}: {len(pv_data.timestamps)} data points")

   finally:
       await connector.disconnect()

Pattern Detection Usage
-----------------------

Detect control system operations in generated code:

.. code-block:: python

   from osprey.services.python_executor.analysis.pattern_detection import detect_control_system_operations

   code = """
   # Read beam current
   current = epics.caget('BEAM:CURRENT')

   # Adjust setpoint if needed
   if current < 400:
       epics.caput('BEAM:SETPOINT', 420.0)
   """

   # Detect operations
   result = detect_control_system_operations(code)

   if result['has_writes']:
       print("⚠️ Code performs write operations - requires approval")

   if result['has_reads']:
       print("✓ Code performs read operations")

   print(f"Control system: {result['control_system_type']}")
   print(f"Write patterns detected: {result['detected_patterns']['writes']}")
   print(f"Read patterns detected: {result['detected_patterns']['reads']}")

Custom Connector Registration
-----------------------------

Custom connectors are registered through the Osprey registry system:

.. code-block:: python

   # In your application's registry.py
   from osprey.registry import ConnectorRegistration, extend_framework_registry

   class MyAppRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self):
           return extend_framework_registry(
               connectors=[
                   # Control system connectors
                   ConnectorRegistration(
                       name="labview",
                       connector_type="control_system",
                       module_path="my_app.connectors.labview_connector",
                       class_name="LabVIEWConnector",
                       description="LabVIEW Web Services connector"
                   ),
                   ConnectorRegistration(
                       name="tango",
                       connector_type="control_system",
                       module_path="my_app.connectors.tango_connector",
                       class_name="TangoConnector",
                       description="Tango control system connector"
                   ),
                   # Archiver connectors
                   ConnectorRegistration(
                       name="tango_archiver",
                       connector_type="archiver",
                       module_path="my_app.connectors.tango_archiver",
                       class_name="TangoArchiverConnector",
                       description="Tango archiver connector"
                   ),
               ],
               capabilities=[...],
               context_classes=[...]
           )

After registration, connectors are available via configuration:

.. code-block:: yaml

   # config.yml
   control_system:
     type: labview  # or tango, epics, mock
     connector:
       labview:
         base_url: "http://labview-server:8080"
         api_key: "your-api-key"
       tango:
         device_name: "tango://host:10000/sys/dev/1"

   archiver:
     type: tango_archiver
     tango_archiver:
       url: "https://archiver.facility.edu"

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/06_control-system-integration`
       Complete implementation guide with step-by-step examples

   :doc:`../../../getting-started/control-assistant-part1-setup`
       See connectors in action in the Control Assistant tutorial

   :doc:`01_human-approval`
       How pattern detection integrates with approval workflows

   :doc:`03_python-execution`
       Pattern detection in secure Python code execution

