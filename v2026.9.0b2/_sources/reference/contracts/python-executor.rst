.. _reference-python-executor:

========================
Python Executor Contract
========================

The two MCP tools the Osprey agent calls to run Python --- their parameters,
what a successful call returns, and how a failing one reports itself. For how
the service works and what it is allowed to touch, see
:doc:`/architecture/python-executor`.

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

Installation
============

The Python executor is included in the default Osprey installation:

.. code-block:: bash

   uv sync

No additional setup is needed.
