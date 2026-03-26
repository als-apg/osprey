.. _how-to-python-executor:

========================
Python Execution Service
========================

The Python Execution Service runs user-provided code in an isolated
environment with safety checks, process isolation, and timeout enforcement.
Claude uses it via the ``execute`` MCP tool to perform data analysis,
plotting, and control-system interactions on behalf of the operator.

What It Does
============

The service accepts Python source code, applies layered safety checks, and
runs it in either a **container** (Jupyter kernel over WebSocket) or a
**local subprocess** (``ExecutionWrapper``). Results---stdout, stderr,
figures, and saved artifacts---are returned as structured JSON.

.. code-block:: text

   Claude → execute MCP tool → safety checks → container / subprocess → result JSON

All packages installed in the deployment environment are available to
executed code (numpy, pandas, scipy, matplotlib, plotly, etc.).
A ``save_artifact(obj, title="Untitled", description="", artifact_type=None)``
helper is injected into the subprocess namespace for saving objects to the
artifact gallery. The ``artifact_type`` parameter overrides automatic type
detection (e.g., ``"figure"``, ``"dataframe"``).

.. _executor-mcp-tool:

MCP Tool Interface
==================

The server exposes a single tool, ``execute``, registered on the ``python``
FastMCP server (``osprey.mcp_server.python_executor``).

.. list-table:: ``execute`` parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``code``
     - *(required)*
     - Python source code to run.
   * - ``description``
     - *(required)*
     - Human-readable description of what the code does.
   * - ``execution_mode``
     - ``"readonly"``
     - ``"readonly"`` blocks detected write patterns; ``"readwrite"`` allows
       them.
   * - ``save_output``
     - ``True``
     - Save code and output to a workspace data file and artifact store.

The tool returns a JSON object containing:

- **output** / **error** --- stdout and stderr truncated to 500 characters.
  Full output is available in the saved data file.
- **artifact_ids** --- IDs for any figures or saved objects.
- **notebook_artifact_id** --- ID of an auto-generated notebook capturing the
  run.
- **gallery_url** --- link to the artifact gallery (when available).
- **has_errors**, **status** (``"Success"`` / ``"Failed"``), and
  **detected_patterns** for control-system operation metadata.

Execution Modes
===============

Container Execution
-------------------

The default mode. Code is sent to a Jupyter kernel running inside a Docker
container via WebSocket. Separate containers can be configured for
``readonly`` and ``readwrite`` modes:

.. code-block:: yaml

   # config.yml
   osprey:
     execution:
       execution_method: container
     services:
       jupyter:
         containers:
           read:
             hostname: localhost
             port_host: 8088
           write:
             hostname: localhost
             port_host: 8089

Container mode provides the strongest isolation---the MCP server
process is never exposed to user code.

Local Subprocess Execution
--------------------------

When containers are unavailable, the service falls back to local subprocess
execution. The ``ExecutionWrapper`` wraps user code with safety
monkeypatches (e.g., ``epics.caput()`` validation against the limits
database), writes the wrapped script to an execution folder, and runs it
as a subprocess:

.. code-block:: yaml

   osprey:
     execution:
       execution_method: local
       python_env_path: /path/to/venv   # optional; defaults to sys.executable

The subprocess working directory is set to the project root so that
relative workspace paths (e.g. ``osprey-workspace/data/002_archiver_read.json``)
resolve correctly.

Security Model
==============

Five safety layers are applied in sequence:

1. **Static safety check** (``quick_safety_check``)---blocks dangerous
   patterns such as dynamic code evaluation, dynamic imports, and
   ``subprocess`` calls before execution begins.

2. **Control-system pattern detection**
   (``detect_control_system_operations``)---identifies read and write
   patterns. In ``readonly`` mode, detected writes cause immediate
   rejection.

3. **Limits monkeypatch** (``ExecutionWrapper`` /
   ``LimitsValidator``)---at runtime, ``epics.caput()`` calls are
   intercepted and validated against the channel limits database.
   Out-of-range values are blocked.

4. **Process isolation**---code always runs outside the MCP server
   process, either in a container or a local subprocess.

5. **Execution timeout**---configurable via
   ``python_executor.execution_timeout_seconds`` (default 600 s). The
   process is killed if it exceeds the limit.

.. code-block:: yaml

   osprey:
     python_executor:
       execution_timeout_seconds: 300

.. admonition:: Control system operations in user code

   Python code interacts with control systems using
   ``osprey.runtime`` utilities (``read_channel()``, ``write_channel()``),
   not direct connector imports. The execution wrapper configures these
   automatically from the deployment context, so code works with any
   connector (EPICS, Mock, etc.) and notebooks remain reproducible.

.. note::

   There is no in-framework code-generation pipeline. Claude Code generates
   Python code itself and invokes the ``execute`` MCP tool directly.

.. note::

   Write approval is handled by the ``execution_mode`` parameter. Claude Code
   requests user confirmation before calling ``execute`` with
   ``execution_mode="readwrite"``---there is no separate approval API.

Installation
============

The Python executor is included in the default Osprey installation:

.. code-block:: bash

   uv sync

No additional setup is needed for local subprocess mode. For container
mode, configure the Jupyter container endpoints in ``config.yml`` as shown
above.

See Also
========

- :doc:`MCP Servers </architecture/mcp-servers>` for how the
  ``python`` server fits into the overall system.
- ``src/osprey/mcp_server/python_executor/`` for the full server source.
- ``src/osprey/services/python_executor/`` for execution engine internals,
  safety checks, and pattern detection.
