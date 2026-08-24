.. _how-to-python-executor:

========================
Python Execution Service
========================

The Python Execution Service runs user-provided code in a separate host
subprocess with layered safety checks, a process boundary, and timeout
enforcement. It is a process boundary on the host, not a sandbox or container:
the code runs with the executor's own environment (from which sensitive
credentials are stripped), not inside an isolated machine. Code it runs can
write anywhere the executor process can --- including the project's
``config.yml`` and its hook scripts.
The Osprey agent uses it via the ``execute`` MCP tool to perform data analysis,
plotting, and control-system interactions on behalf of the operator.

What It Does
============

The service accepts Python source code, applies layered safety checks, and
runs it as a **subprocess** on the host (``ExecutionWrapper``).
Results---stdout, stderr, figures, and saved artifacts---are returned as
structured JSON.

.. raw:: html
   :file: ../_diagrams/python-executor.html

Executed code can import whatever is installed in the project's environment;
see :ref:`executor-environment` for how to find out what that is.
A ``save_artifact(obj, title="Untitled", description="", artifact_type=None, category="")``
helper is injected into the subprocess namespace for saving objects to the
artifact gallery. The ``artifact_type`` parameter overrides automatic type
detection (e.g., ``"figure"``, ``"dataframe"``); ``category`` is a free-form
grouping label used by the gallery UI.

.. _executor-mcp-tool:

MCP Tool Interface
==================

The server exposes two tools, ``execute`` and ``execute_file``, registered on
the ``python`` FastMCP server (``osprey.mcp_server.python_executor``).
``execute_file`` runs an existing ``.py`` file on disk through the same safety
pipeline as ``execute``; both share the parameters below (``execute_file``
takes ``file_path`` and optional ``script_args`` in place of ``code``).

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

On success the tool returns a JSON object whose top level describes the saved
run:

- **status** (``"success"``), **artifact_id**, **title**, and **data_file**
  for the saved artifact.
- **summary** --- a nested object with the run details: the truncated
  **output** / **error** (500 characters; full output is in the saved data
  file), a ``"Success"`` / ``"Failed"`` **status**, **has_errors**, and
  **detected_patterns** for control-system operation metadata.
- **artifact_ids** --- IDs for any figures or saved objects, and
  **notebook_artifact_id** for the auto-generated notebook capturing the run.
- **gallery_url** --- link to the artifact gallery (when available).

A failing run does not return this object as a normal result: the tool raises
a structured error (``error_type: "execution_error"``) whose ``details`` field
carries the same information, so the agent receives an explicit error rather
than a status field it might overlook.

Where Code Runs
===============

Agent-authored Python runs as a subprocess on the same host as the Osprey
agent. It is the only execution backend, so there is nothing to choose
between:

.. code-block:: yaml

   # config.yml
   execution:
     execution_method: subprocess

``subprocess`` is the default and the key can be left out entirely. Two other
spellings load: ``local`` is an alias for this same backend and is accepted
silently, and ``container`` is treated as ``subprocess`` and logs a one-time
warning naming the config file it came from. Any other value is a
configuration error.

The ``ExecutionWrapper`` wraps user code with safety monkeypatches (e.g.
``epics.caput()`` validation against the limits database), writes the wrapped
script to an execution folder, and runs it. The subprocess working directory is
set to the project root so that relative workspace paths (e.g.
``_agent_data/data/002_archiver_read.json``) resolve correctly.

.. _executor-environment:

The Execution Environment
=========================

Executed code runs in the project's own virtual environment
(``<project>/.venv``) when the project has one, and otherwise in the
interpreter running Osprey. ``config.yml`` records no interpreter path and
offers no setting to point execution somewhere else --- which environment
exists is decided when the project is built, from the build profile's
``environment:`` block (see :doc:`build-profiles`).

Anything installed in that environment is importable by executed code:

.. code-block:: bash

   cd my-project
   uv pip list             # what executed code can import
   uv pip install lmfit    # importable by the next execution

The description the agent sees for the ``execute`` tool is generated from this
environment rather than from a fixed list, so the agent is told what is really
installed. It is computed once when the MCP server starts: a package installed
mid-session is importable straight away, but the agent will not know about it
until the next session. If the environment cannot be read at startup, the
description names no packages at all instead of guessing.

A container image built from the project installs the same package set as the
project's own environment, so executed code sees the same imports either way.

Security Model
==============

Seven safety layers are applied in sequence:

1. **Static safety check** (``quick_safety_check``)---blocks dangerous
   patterns such as dynamic code evaluation, dynamic imports, and
   ``subprocess`` calls before execution begins.

2. **Readonly import denylist** (``check_readonly_imports``)---a
   ``readonly`` run may not import control-system client libraries
   (``epics``, ``p4p``, ``caproto``, ``pvaccess``, ``tango``) at all.
   Reads go through ``read_channel()``.

3. **Control-system pattern detection**
   (``detect_control_system_operations``)---identifies read and write
   patterns. In ``readonly`` mode, detected writes cause immediate
   rejection.

4. **Readonly runtime guard**---the declared mode is exported into the
   subprocess as ``OSPREY_EXECUTION_MODE``. In a ``readonly`` run every
   entry point listed under :ref:`python-executor-write-surface` is
   replaced with a refusing function, the connectors refuse
   ``write_channel``, and the EPICS connector stays on the ``read_only``
   gateway --- so a write is refused at runtime and at the network layer
   however it is spelled.

5. **Limits monkeypatch** (``ExecutionWrapper`` /
   ``LimitsValidator``)---at runtime, ``epics.caput()`` calls are
   intercepted and validated against the channel limits database.
   Out-of-range values are blocked.

6. **Process isolation**---code always runs in a separate subprocess, never
   inside the MCP server process.

7. **Execution timeout**---configurable via
   ``python_executor.execution_timeout_seconds`` (default 600 s). The
   process is killed if it exceeds the limit.

.. code-block:: yaml

   python_executor:
     execution_timeout_seconds: 300

.. _python-executor-write-surface:

What counts as a control-system write
-------------------------------------

A ``readonly`` run refuses all of the following. The list is generated from
one table in the code (``_READONLY_WRITE_TARGETS`` in
``services/python_executor/execution/wrapper.py``), so adding a library there
is all it takes to enforce it.

**Approved API** --- the path everything should use:

- ``write_channel()`` / ``write_channels()`` from ``osprey.runtime``, and
  ``connector.write_channel()`` beneath them.

**Control-system clients**, refused whether they are imported, aliased, or
resolved dynamically:

- **pyepics** --- ``caput``, ``caput_many``, ``PV.put``, ``ca.put``
- **p4p** --- ``Context.put``/``rpc`` for the thread, asyncio and cothread
  clients; ``SharedPV.post``/``open`` on the server side
- **caproto** --- ``sync.client.write``, ``threading.client.PV.write``,
  ``Batch.write``, ``asyncio.client.PV.write``
- **pvaPy** --- every ``Channel.put*`` method, including the typed setters
  such as ``putDouble`` and ``putScalarArray``
- **Tango** --- ``DeviceProxy.write_attribute`` and its variants,
  ``AttributeProxy.write``, and ``command_inout`` (a Tango command acts on
  the device rather than reading it)

**Routes out of Python.** These reach a control system without importing a
client at all, so they are refused in ``readonly`` runs too:

- Starting a process --- ``subprocess.run``/``Popen``/``call``/
  ``check_output``, ``os.system``, ``os.popen``, and the ``os.exec*``,
  ``os.spawn*`` and ``posix_spawn`` families. This is what stops a
  shelled-out ``caput``.
- Loading a shared library --- ``ctypes.CDLL`` and friends, which could
  otherwise open ``libca`` directly.

``os.fork`` is deliberately left working: forking alone cannot start a new
program, and the ``exec`` half of every fork-and-exec is already refused.

.. note::

   The consequence of the second group is that a ``readonly`` script cannot
   shell out or load a shared library *at all*, even for something unrelated
   to the control system. Resubmit such work with
   ``execution_mode="readwrite"``, which requires human approval.

When a write is refused
-----------------------

Three things happen, at whichever layer catches it:

1. The agent gets an error naming the mode and what to do instead.
2. The operator is alerted --- the Web Terminal reports that a write was
   attempted and blocked, not merely that a script failed.
3. The attempt is recorded in ``var/audit/readonly-refusals.jsonl`` with a
   timestamp, the layer that refused, and the offending source. That
   directory is durable: builds never touch it, and ``osprey reset`` keeps
   it unless you pass ``--purge-audit``.

.. admonition:: Control system operations in user code

   Python code interacts with control systems using
   ``osprey.runtime`` utilities (``read_channel()``, ``write_channel()``),
   not direct connector imports. The execution wrapper configures these
   automatically from the deployment context, so code works with any
   connector (EPICS, Mock, etc.) and notebooks remain reproducible.

.. note::

   There is no in-framework code-generation pipeline. The Osprey agent generates
   Python code itself and invokes the ``execute`` MCP tool directly.

.. note::

   Write approval is handled by the ``execution_mode`` parameter. The Osprey agent
   requests user confirmation before calling ``execute`` with
   ``execution_mode="readwrite"``---there is no separate approval API.

Installation
============

The Python executor is included in the default Osprey installation:

.. code-block:: bash

   uv sync

No additional setup is needed.

See Also
========

- :doc:`MCP Servers </architecture/mcp-servers>` for how the
  ``python`` server fits into the overall system.
- ``src/osprey/mcp_server/python_executor/`` for the full server source.
- ``src/osprey/services/python_executor/`` for execution engine internals,
  safety checks, and pattern detection.
