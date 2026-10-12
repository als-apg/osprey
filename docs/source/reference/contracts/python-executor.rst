.. _reference-python-executor:

========================
Python Executor Contract
========================

The MCP tools the OSPREY agent calls to run Python --- their parameters,
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
     - ``"readonly"`` blocks detected write patterns. ``"readwrite"`` may write,
       through ``osprey.runtime.write_channel()`` only; a raw client put is
       refused (see
       :ref:`python-executor-armed-block`).
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

.. _executor-measurement-tools:

Measurement Tools
=================

A render that carries a pyAML view (``data/pyaml/<model>/configuration.yaml``,
written by the build for each served model with a measurement file) adds two
tools to the ``python`` server. A render without one registers neither.

``pyaml_measure``
-----------------

Steps setpoints on the control target through pyAML and writes them back. Every
call asks for approval, and with writes off it is denied like every write tool.
The prompt lists every setpoint the call steps; a call stepping more than 500
is refused without asking.

.. list-table:: ``pyaml_measure`` parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``kind``
     - *(required)*
     - ``orm``, ``dispersion``, ``trm``, ``crm`` or ``chromaticity_monitor``.
       A kind the model's measurement file does not allow is refused, naming
       the group or instrument it needs.
   * - ``model``
     - the one model
     - The model measured; may be omitted only when exactly one model has a
       pyAML view.
   * - ``correctors``
     - every corrector
     - Corrector setpoint addresses, for ``orm`` only.
   * - ``modulation``
     - ``"bipolar"``; ``"unipolar"`` with ``by`` ``"group"``
     - ``"bipolar"`` steps each setting to both sides of its present value,
       ``"unipolar"`` to one side.
   * - ``by``
     - ``"magnet"``
     - ``trm`` and ``crm`` only. ``"magnet"`` steps one magnet per column;
       ``"group"`` steps every magnet of the group once, from its present
       setting, by the same integrated strength. Group stepping is unipolar:
       an omitted modulation is ``"unipolar"`` and ``"bipolar"`` is refused.

Each step comes from the measurement file (``corrector_delta``,
``frequency_delta``, ``quad_delta``, ``sextu_delta``) and is shrunk until no
setpoint's largest single write exceeds half its ``max_step`` in
``data/facility/limits.yaml``; ``delta_limited_by`` and ``rf_limited_by`` name
the setpoint that bounded it. A strength, ``max_step`` or write that is not a
finite number, or a ``max_step`` not above zero, refuses the measurement before
any setpoint moves. A group step is first sized from the view's
``trm.json`` or ``crm.json`` so the tune or chromaticity moves no more than
under the strongest single-magnet step, and no magnet moves by more than one
magnet's step; ``delta_basis`` is ``tune excursion``, ``chromaticity
excursion``, ``magnet step`` or ``max_step``. A group column is per unit
integrated strength: the sum of its members' magnet-by-magnet columns. The
build writes ``trm.json`` or ``crm.json`` whenever every member of the group has
a modelled column; a group with an unmodelled member has none, and its group
step is refused.

The run starts only on the view the build recorded and the approval prompt
showed (``approved_view_sha256``), and every write runs inside one guarded run:
a run that stops part-way is written back before the call returns. The result
names every row, column and step by address and device id and carries the
view's ``sha256``, the ``shape``, ``data_file`` and ``artifact_id``. The file
saved is the one the run left in its own execution folder under the script's
fixed name; a run that names any other file saves nothing, answers
``data_file`` as ``null`` and says why in ``data_file_refused``. A stopped
run carries its ``remainder`` and the ``restore`` report, taken from the run's
execution record and never from a line the script printed.

``model_response_matrix``
-------------------------

Solves a served model's orbit response on the model's own deck, in a sandbox
child stopped at the executor's timeout. It writes nothing and asks for no
approval.

.. list-table:: ``model_response_matrix`` parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``model``
     - the one model
     - A served deck model; may be omitted only when the render serves one.
   * - ``correctors``
     - every corrector
     - Corrector setpoint addresses the model wires; omitted, every one,
       horizontal then vertical.

Each corrector steps by the measurement file's ``corrector_delta`` (radians),
converted to the corrector's hardware unit through its wiring calibration; a
model whose measurement file states none is refused. Each entry is a readback
change in its hardware unit per setpoint change in its own. The result carries
``rows`` and ``columns``, each named by address and device id, each column's
``step`` and the ``matrix``.

Installation
============

The Python executor is included in the default Osprey installation:

.. code-block:: bash

   uv sync

No additional setup is needed.
