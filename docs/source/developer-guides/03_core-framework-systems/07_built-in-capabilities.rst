=====================
Built-in Capabilities
=====================

.. versionadded:: 0.11

Since v0.11, seven capabilities are provided **natively by the framework** in :mod:`osprey.capabilities`. They are automatically registered when your application starts --- no code generation or template files required. Your application only needs prompt customizations and data.

This page is a concise reference for each capability. For a hands-on walkthrough of how they work together in a real session, see the :doc:`Control Assistant Tutorial </getting-started/control-assistant>`.

.. contents:: On this page
   :local:
   :depth: 1


Control System Capabilities
===========================

These four capabilities form the control-system interaction layer, handling the full lifecycle from channel discovery through live reads, writes, and historical retrieval.


``channel_finding``
-------------------

Resolves natural language descriptions to control system channel addresses using configurable search pipelines.

**Context flow:**

- **Provides:** ``CHANNEL_ADDRESSES`` --- list of matched channel address strings with the original search query
- **Requires:** *(none)* --- extracts the search query from the task objective

**Configuration:**

.. code-block:: yaml

   channel_finder:
     pipeline_mode: hierarchical          # in_context | hierarchical | middle_layer
     pipelines:
       hierarchical:
         database:
           type: hierarchical
           path: src/.../channel_databases/hierarchical.json

**Error handling:** ``ChannelNotFoundError`` triggers replanning; ``ChannelFinderServiceError`` is critical.

To view or customize: ``osprey eject capability channel_finding``

.. dropdown:: Context class reference: ``ChannelAddressesContext``

   .. autoclass:: osprey.capabilities.channel_finding.ChannelAddressesContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/getting-started/control-assistant-part2-channel-finder` --- pipeline modes, database building, benchmarks
   - :doc:`04_prompt-customization` --- facility-specific prompt overrides for channel finder


``channel_read``
----------------

Reads current live values from control system channels using the configured connector (mock, EPICS, Tango, or LabVIEW).

**Context flow:**

- **Provides:** ``CHANNEL_VALUES`` --- dictionary mapping channel addresses to ``ChannelValue`` (value, timestamp, units)
- **Requires:** ``CHANNEL_ADDRESSES`` --- typically from a preceding ``channel_finding`` step

**Configuration:**

.. code-block:: yaml

   control_system:
     type: mock            # mock | epics | tango | labview
     connector:
       epics:
         timeout: 5.0

**Error handling:** Timeout and access errors are retriable (up to 3 attempts with exponential backoff). Missing dependencies trigger replanning.

To view or customize: ``osprey eject capability channel_read``

.. dropdown:: Context class reference: ``ChannelValuesContext``

   .. autoclass:: osprey.capabilities.channel_read.ChannelValuesContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

   .. autoclass:: osprey.capabilities.channel_read.ChannelValue
      :members:
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/getting-started/control-assistant-part3-production` --- live read examples in the tutorial walkthrough
   - :doc:`/developer-guides/05_production-systems/06_control-system-integration` --- connector architecture and custom connectors


``channel_write``
-----------------

Writes values to control system channels with four mandatory safety layers.

**Safety layers (enforced in order):**

1. **Master switch** --- ``control_system.writes_enabled`` must be ``true``
2. **Human approval** --- LangGraph interrupt for operator confirmation
3. **Limits checking** --- min/max/step/writable constraints from limits database
4. **Write verification** --- callback or readback confirmation after write

**Context flow:**

- **Provides:** ``CHANNEL_WRITE_RESULTS`` --- per-channel success/failure with optional verification details
- **Requires:** ``CHANNEL_ADDRESSES`` --- plus optionally ``PYTHON_RESULTS``, ``ARCHIVER_DATA``, or any other context containing the value to write

**Configuration:**

.. code-block:: yaml

   control_system:
     writes_enabled: false               # Master safety switch

   approval:
     capabilities:
       python_execution:
         enabled: true
         mode: control_writes            # Approval for writes

**Error handling:** Access errors and read-only violations are critical. Write parsing failures are retriable. Ambiguous operations trigger replanning.

To view or customize: ``osprey eject capability channel_write``

.. dropdown:: Context class reference: ``ChannelWriteResultsContext``

   .. autoclass:: osprey.capabilities.channel_write.ChannelWriteResultsContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

   .. autoclass:: osprey.capabilities.channel_write.ChannelWriteResult
      :members:
      :show-inheritance:
      :no-index:

   .. autoclass:: osprey.capabilities.channel_write.WriteVerificationInfo
      :members:
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/developer-guides/05_production-systems/01_human-approval-workflows` --- approval interrupt mechanics
   - :doc:`/getting-started/control-assistant-part3-production` --- write workflow walkthrough


``archiver_retrieval``
----------------------

Retrieves historical time-series data from the facility archiver for analysis and visualization.

**Context flow:**

- **Provides:** ``ARCHIVER_DATA`` --- timestamps, time-series values per channel, precision, and available channel list
- **Requires:** ``CHANNEL_ADDRESSES`` + ``TIME_RANGE`` (single) --- from preceding ``channel_finding`` and ``time_range_parsing`` steps

**Configuration:**

.. code-block:: yaml

   archiver:
     type: mock_archiver        # mock_archiver | epics_archiver
     epics_archiver:
       url: https://archiver.facility.edu:8443
       timeout: 60

**Common downstream patterns:**

- ``archiver_retrieval`` → ``python`` (create plot) → respond
- ``archiver_retrieval`` → ``python`` (calculate statistics) → respond

**Error handling:** Timeouts are retriable. Connection errors are critical. Data format issues and missing dependencies trigger replanning.

To view or customize: ``osprey eject capability archiver_retrieval``

.. dropdown:: Context class reference: ``ArchiverDataContext``

   .. autoclass:: osprey.capabilities.archiver_retrieval.ArchiverDataContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/getting-started/control-assistant-part3-production` --- archiver + plotting workflow
   - :doc:`/developer-guides/05_production-systems/06_control-system-integration` --- archiver connector configuration


Analysis & Execution Capabilities
=================================

These three capabilities provide computational analysis, temporal reasoning, and cross-session memory.


``python``
----------

Generates and executes Python code through the Python executor service. Acts as a gateway between the agent graph and the sandboxed execution environment.

**Context flow:**

- **Provides:** ``PYTHON_RESULTS`` --- generated code, stdout/stderr, computed results (``results.json``), execution time, figure paths, and notebook link
- **Requires:** *(none)* --- but commonly receives ``ARCHIVER_DATA``, ``CHANNEL_VALUES``, or other contexts as inputs for analysis

**Configuration:**

.. code-block:: yaml

   services:
     jupyter:
       path: ./services/jupyter
       containers:
         read:
           name: jupyter-read
           port_host: 8088

   approval:
     capabilities:
       python_execution:
         enabled: true
         mode: control_writes       # disabled | all_code | control_writes

**Error handling:** All service errors are retriable (the service handles retries internally with up to 3 attempts).

To view or customize: ``osprey eject capability python``

.. dropdown:: Context class reference: ``PythonResultsContext``

   .. autoclass:: osprey.capabilities.python.PythonResultsContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/developer-guides/05_production-systems/03_python-execution-service/index` --- service architecture, code generation, and security
   - :doc:`04_prompt-customization` --- Python prompt builder for domain-specific instructions


``time_range_parsing``
----------------------

Converts natural language time expressions into precise datetime ranges using LLM-based analysis. Supports relative periods (*"last 24 hours"*), named periods (*"yesterday"*), and absolute date references.

**Context flow:**

- **Provides:** ``TIME_RANGE`` --- ``start_date`` and ``end_date`` as Python ``datetime`` objects with full arithmetic and formatting support
- **Requires:** *(none)* --- extracts temporal expressions from the task objective

**Configuration:**

The capability uses the model configured under the ``time_parsing`` key in ``models:``. No additional configuration required.

**Error handling:** Invalid formats are retriable (LLM may parse correctly on retry). Ambiguous time references trigger replanning to request user clarification.

To view or customize: ``osprey eject capability time_range_parsing``

.. dropdown:: Context class reference: ``TimeRangeContext``

   .. autoclass:: osprey.capabilities.time_range_parsing.TimeRangeContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/getting-started/control-assistant-part3-production` --- time parsing in the archiver retrieval workflow
   - :doc:`04_prompt-customization` --- time range parsing prompt builder


``memory``
----------

Saves and retrieves user information across conversations. Uses LLM-based analysis to extract memory-worthy content from chat history and integrates with the approval system for controlled modifications.

**Context flow:**

- **Provides:** ``MEMORY_CONTEXT`` --- operation type (``save``/``retrieve``), result message, and memory data
- **Requires:** *(none)*

**Supported operations:**

- **Save** --- extracts content from chat history using LLM analysis, optionally requests approval, then stores persistently
- **Retrieve** --- fetches all stored memory entries for the current user

**Configuration:**

.. code-block:: yaml

   approval:
     capabilities:
       memory:
         enabled: true             # Require approval for memory saves

   session:
     user_id: operator_1           # Required for memory operations

**Error handling:** Missing user ID is critical. Content extraction failures trigger replanning. Storage and retrieval errors are retriable.

To view or customize: ``osprey eject capability memory``

.. dropdown:: Context class reference: ``MemoryContext``

   .. autoclass:: osprey.capabilities.memory.MemoryContext
      :members: get_summary, get_access_details
      :show-inheritance:
      :no-index:

.. seealso::

   - :doc:`/developer-guides/05_production-systems/04_memory-storage-service` --- storage backend and file format
   - :doc:`04_prompt-customization` --- memory extraction prompt builder


Customization
=============

All built-in capabilities support customization through two mechanisms:

**Prompt overrides** --- Place prompt builder files in your project's ``framework_prompts/`` directory to customize orchestrator guides, classifier examples, and domain-specific instructions without modifying capability code. See :doc:`04_prompt-customization`.

**Eject for full control** --- Use ``osprey eject <capability>`` to copy a capability's source into your project for complete customization. The ejected copy takes precedence over the framework version. See :ref:`osprey eject <cli-eject>` for details.

.. seealso::

   - :doc:`/getting-started/control-assistant-part4-customization` --- customization tutorial with practical examples
