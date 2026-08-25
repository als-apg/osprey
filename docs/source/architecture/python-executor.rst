.. _architecture-python-executor:

===============
Python Executor
===============

The Python Execution Service runs user-provided code in a separate host
subprocess with layered safety checks, a process boundary, and timeout
enforcement. It is a process boundary on the host, not a sandbox or container:
the code runs with the executor's own environment (from which sensitive
credentials are stripped), not inside an isolated machine. Code it runs can
write anywhere the executor process can, apart from the deployment's own
sources and render --- see :ref:`python-executor-protected-paths`, which also
says plainly how far that protection goes.
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

The ``execute`` and ``execute_file`` tool parameters and the JSON they return
are documented in :doc:`/reference/contracts/python-executor`.

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
``environment:`` block (see :doc:`/how-to/build-profiles`).

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

Nine safety layers are applied in sequence:

1. **Static safety check** (``quick_safety_check``)---blocks dangerous
   patterns such as dynamic code evaluation, dynamic imports, and
   ``subprocess`` calls before execution begins.

2. **Protected-path policy** (``path_policy_issues``)---reads the submitted
   source and refuses code that aims a write at the deployment's own
   sources or render. Applied in **every** execution mode; see
   :ref:`python-executor-protected-paths`.

3. **Readonly import denylist** (``check_readonly_imports``)---a
   ``readonly`` run may not import control-system client libraries
   (``epics``, ``p4p``, ``caproto``, ``pvaccess``, ``tango``) at all.
   Reads go through ``read_channel()``.

4. **Control-system pattern detection**
   (``detect_control_system_operations``)---identifies read and write
   patterns. In ``readonly`` mode, detected writes cause immediate
   rejection.

5. **Readonly runtime guard**---the declared mode is exported into the
   subprocess as ``OSPREY_EXECUTION_MODE``. In a ``readonly`` run every
   entry point listed under :ref:`python-executor-write-surface` is
   replaced with a refusing function, the connectors refuse
   ``write_channel``, and the EPICS connector stays on the ``read_only``
   gateway --- so a write is refused at runtime and at the network layer
   however it is spelled.

6. **Filesystem guard**---the runtime half of layer 2, patched into the
   subprocess so it also sees paths the source did not spell out. Emitted
   in **every** execution mode.

7. **Limits monkeypatch** (``ExecutionWrapper`` /
   ``LimitsValidator``)---at runtime, ``epics.caput()`` calls are
   intercepted and validated against the channel limits database.
   Out-of-range values are blocked.

8. **Process isolation**---code always runs in a separate subprocess, never
   inside the MCP server process.

9. **Execution timeout**---configurable via
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

.. _python-executor-protected-paths:

What executed code may not change
---------------------------------

Refusing control-system writes is one boundary; this is a different one. Whatever
mode a run is submitted in, it may not rewrite the deployment that runs it.
Three locations are protected:

- **The render zone** (``build/``) --- rebuilt wholesale by every
  ``osprey build``, and the rendered ``config.yml`` inside it is where the next
  run reads its own permissions from. A write here is either lost at the next
  build or, worse, survives as a rendered setting nobody wrote.
- **The profile sources** --- what the build reads to produce that render:
  ``profile.yml`` (with ``triggers.yml``, ``ci-extra.yml`` and
  ``osprey.service`` beside it), the ``personas/``,
  ``profiles/``, ``data/`` and ``scripts/`` directories, the convention
  directories (``rules/``, ``skills/``, ``agents/``, ``commands/``,
  ``output-styles/``, ``hooks/``, ``web-terminal-context/``, ``mcp_servers/``,
  ``services/``) and the ``project/`` mirror.
- **The audit ledger** (``var/audit/``) --- the record of what was refused. A
  run that could rewrite it could erase the evidence of its own refusal.

A protected entry that does not exist yet is protected all the same: in a repo
with no ``personas/`` directory, creating one is exactly the write to refuse.

This boundary is about *zones* — whole trees, wherever inside them a path lands.
A companion boundary, :doc:`the protected set </how-to/control-systems/protected-set>`, closes named
files and config keys to every agent-side writer (the settings panels, the
galleries, the ``setup_patch`` tool) rather than to executed code.

The agent's own data zone (``var/agent_data/``) and the run's execution folder
are carved back out, so analysis output, figures and saved artifacts keep
working normally --- as does everything else on disk the executor process can
reach.

**Both modes, same answer.** ``readwrite`` buys a run human approval to move a
magnet; it buys no approval to rewrite ``profile.yml``. Only the wording of the
refusal changes --- a ``readonly`` run is told the mode, a ``readwrite`` run is
told the path --- and re-running as ``readwrite`` never lifts it. To change any
of these, edit the profile source yourself and run ``osprey build``.

``.env`` is deliberately **not** in this set. What matters about the secrets
zone is that executed code should not *read* it, and this is a write-side
guard; listing the path here would advertise it while protecting nothing that
matters. Read that omission as "a different problem, not yet solved", not as a
verdict that ``.env`` is safe for executed code to touch.

Two layers enforce the set:

1. **Before the run.** The submitted source is walked for write calls ---
   ``open()`` in a write mode, ``Path.write_text``/``Path.open``, the
   ``shutil`` copy/move/remove family, ``os.remove``/``rename``/``mkdir`` and
   friends --- and any whose path is written out as a literal landing inside a
   protected location is refused before anything is spawned, so the agent gets
   a readable message instead of a traceback. A path the code computes at
   runtime is deliberately *not* guessed at here; it is left to layer 2.
   The same pass refuses code that names the runtime guard's own internals,
   which is the one thing that guard cannot police for itself.

2. **During the run.** A guard is emitted into the subprocess, ahead of the
   submitted code, that intercepts the filesystem entry points (``open``,
   ``io.open``, ``os.open``, ``os.truncate``, ``os.remove``, ``os.unlink``,
   ``os.rmdir``, ``os.removedirs``, ``os.rename``, ``os.replace``,
   ``os.makedirs``, ``os.mkdir``, ``os.symlink``, ``os.link``, and
   ``shutil.copy``, ``copy2``, ``copyfile``, ``move`` and ``rmtree``) and
   raises ``PermissionError`` for a write landing in a protected location,
   however the path was assembled. The protected locations are resolved by
   the parent process and baked into the guard as fixed strings; the
   subprocess never works them out for itself.

.. warning::

   The runtime guard is **defense in depth, not a security boundary.** It runs
   inside the same process as the code it is guarding, so code written to
   defeat it can simply switch it off --- nothing placed *inside* a process can
   be hidden from that process. What the guard reliably stops is the honest
   miss: the computed or concatenated path the pre-run pass cannot see, and the
   everyday accident of writing output one directory too high.

   What contains code that is deliberately attacking this boundary is the
   operating system. In a container deployment the render zone and the profile
   sources are owned by a different user than the one the agent runs as, so the
   refusal is the kernel's rather than Python's. On a bare host, treat the
   guard as a safety net and not as a wall.

.. note::

   The workspace server's visualization sandbox is built from the same guard
   with the verdict inverted --- nothing is reachable unless a directory says
   so. It intercepts only ``open`` and ``io.open``, and ``io.open`` for write
   modes alone: CPython routes ``pathlib`` through it, so guarding reads there
   would change what plotting code is allowed to *read* today.

.. _python-executor-session-posture:

Session posture
---------------

An operator can step a whole Web Terminal session into a **sandbox posture**
from its session card (see :ref:`web-terminal-session-posture`). Every process
that session launches, this server included, then runs with
``OSPREY_EXECUTION_MODE=readonly``, and a ``readwrite`` ``execute`` is refused
by the executor's own posture gate, whatever the deployment itself permits:

   This terminal session is in the sandbox posture, which refuses
   control-system writes regardless of what the run asks for.

The agent is told to re-run as ``readonly`` --- reads are untouched by the
posture --- or to ask the operator to switch the session back. The deployment's
``config.yml`` is not the gate here and the message does not point at it.

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

One case is deliberately not written to that ledger: a protected-path write
refused at runtime inside a ``readwrite`` run. The ledger is keyed on runs that
were readonly, and filing a readwrite refusal there would put "readonly
execution mode" in front of an operator whose run was approved for writes. The
refusal still holds and still reaches the agent and the operator; only the
ledger entry is missing.

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

.. seealso::

   - :doc:`/reference/contracts/python-executor` --- the ``execute`` and
     ``execute_file`` tool parameters, return shape, and installation note.
   - :doc:`/how-to/control-systems/index` --- the control-system tasks that
     executed code is usually written to perform.
