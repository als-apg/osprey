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
A ``save_artifact(obj, title, description)`` helper is injected into the
subprocess namespace for saving objects to the artifact gallery.

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

The tool returns a JSON object containing a compact summary (truncated
stdout/stderr), artifact IDs for any figures or notebooks produced, and a
data-file path for full results.

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

.. _executor-code-generation-placeholder:

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 14):** "The Python Execution Service orchestrates code generation, security analysis, approval, and execution through a LangGraph-based pipeline."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old architecture had a full code-generation pipeline (generator selection, iterative retry with error_chain, approval interrupts) that was removed. Claude Code now generates code itself and calls the ``execute`` MCP tool directly---there is no in-framework generation layer.
   **Action needed:** Decide whether to document Claude Code's code-generation flow (prompt engineering, tool use patterns) as the conceptual replacement, or omit generation workflow docs entirely.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 89):** "Approval handling via handle_service_with_interrupts"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old LangGraph-based approval system (``handle_service_with_interrupts``) is dead. Human approval for hardware writes now flows through Claude Code's built-in approval prompts and the ``readwrite`` execution mode gate, but there is no single function call to point users at.
   **Action needed:** Document the current approval flow for ``readwrite`` operations (Claude Code prompt-based approval) or link to the relevant how-to guide once it exists.

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
