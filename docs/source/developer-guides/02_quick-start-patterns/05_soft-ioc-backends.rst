======================
Soft IOC Custom Backends
======================

**What you'll learn:** How to implement custom simulation backends for soft IOCs, including the SimulationBackend Protocol, chained backend composition, and when to use each approach.

.. dropdown:: Prerequisites
   :color: info
   :icon: list-unordered

   **Required:**

   - Basic understanding of EPICS and PV concepts
   - Familiarity with the ``osprey generate soft-ioc`` command (see :doc:`00_cli-reference`)
   - Python development experience

   **Recommended:**

   - Having successfully generated and run a basic soft IOC before attempting custom backends

Overview
========

Custom backends let you build simulation environments that mirror your real control system's
behavior—setpoints that respond with realistic dynamics, readbacks that drift, faults that
trigger at the right moments. This enables testing agent workflows, validating recovery
strategies, and iterating on control logic without requiring hardware access.

This feature extends soft IOCs generated with the :ref:`cli-generate-soft-ioc` command.
Before implementing custom backends, ensure you can generate and run a basic soft IOC first.

The soft IOC generator supports composable simulation backends:

1. **Built-in backends** (``mock_style``, ``passthrough``) - Ready-to-use simulation behaviors
2. **Custom backends** - User-implemented physics simulation
3. **Chained backends** - Multiple backends composed together

**Architecture:**

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                         Generated IOC                           │
   ├─────────────────────────────────────────────────────────────────┤
   │  ┌─────────────────────────────────────────────────────────────┐│
   │  │                    ChainedBackend                           ││
   │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             ││
   │  │  │   Base     │→ │  Override  │→ │  Override  │   ...       ││
   │  │  │(mock_style)│  │ (physics)  │  │ (faults)   │             ││
   │  │  └────────────┘  └────────────┘  └────────────┘             ││
   │  └─────────────────────────────────────────────────────────────┘│
   │                              │                                  │
   │                              ▼                                  │
   │  ┌─────────────────────────────────────────────────────────────┐│
   │  │                    caproto PVGroup                          ││
   │  │           (PVs served over EPICS Channel Access)            ││
   │  └─────────────────────────────────────────────────────────────┘│
   └─────────────────────────────────────────────────────────────────┘

**When to use each approach:**

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Approach
     - Use Case
     - Example
   * - ``mock_style`` backend
     - General testing with realistic PV behaviors
     - Development, integration testing
   * - ``passthrough`` backend
     - Manual testing or debugging
     - Step-by-step verification
   * - **Custom backend**
     - Physics-accurate simulation
     - Lattice modeling with pyAT
   * - **Chained backends**
     - Fault injection, mixed behaviors
     - Broken feedback on specific PVs

SimulationBackend Protocol
==========================

All backends (built-in and custom) implement the unified ``SimulationBackend`` protocol.
This uses duck typing - no inheritance required.

.. tab-set::

   .. tab-item:: Protocol Definition

      The protocol defines three methods:

      .. code-block:: python

         from typing import Any, Protocol, runtime_checkable

         @runtime_checkable
         class SimulationBackend(Protocol):
             """Protocol for simulation backends."""

             def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
                 """Called once at IOC startup.

                 Args:
                     pv_definitions: List of PV dicts with 'name' and 'type' keys

                 Returns:
                     Dict mapping PV names to initial values.
                 """
                 ...

             def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
                 """Called when a client writes to a PV.

                 Returns:
                     Dict of updates, or None to delegate to next backend.
                 """
                 ...

             def step(self, dt: float) -> dict[str, Any]:
                 """Called periodically at update_rate.

                 Returns:
                     Dict mapping PV names to new values.
                 """
                 ...

      **Key insight:** ``on_write()`` returns ``dict | None``:

      - Return a ``dict`` (even empty ``{}``) to **handle** the write
      - Return ``None`` to **delegate** to the next backend in the chain

   .. tab-item:: Physics Example

      Minimal first-order dynamics - readback approaches setpoint exponentially:

      .. code-block:: python

         import math

         class FirstOrderBackend:
             """RB approaches SP with exponential dynamics."""

             def __init__(self, tau: float = 1.0):
                 """tau: time constant in seconds"""
                 self.tau = tau
                 self._setpoints: dict[str, float] = {}
                 self._readbacks: dict[str, float] = {}

             def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
                 return {}  # Let base backend set initial values

             def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
                 if not pv_name.endswith(':SP'):
                     return None  # Delegate non-setpoints

                 rb_name = pv_name.replace(':SP', ':RB')
                 self._setpoints[pv_name] = float(value)
                 if rb_name not in self._readbacks:
                     self._readbacks[rb_name] = float(value)
                 return {pv_name: value}

             def step(self, dt: float) -> dict[str, Any]:
                 updates = {}
                 for sp_name, sp_val in self._setpoints.items():
                     rb_name = sp_name.replace(':SP', ':RB')
                     rb = self._readbacks.get(rb_name, sp_val)
                     # Exponential approach: RB += (SP - RB) * (1 - e^(-dt/tau))
                     rb += (sp_val - rb) * (1 - math.exp(-dt / self.tau))
                     self._readbacks[rb_name] = rb
                     updates[rb_name] = rb
                 return updates

      **What this shows:**

      - ``on_write``: Capture setpoint changes, delegate non-SP writes
      - ``step``: Evolve physics each timestep
      - State tracking between calls

   .. tab-item:: Fault Example

      Simple drift - readback drifts away from setpoint over time:

      .. code-block:: python

         class DriftBackend:
             """RB drifts independently of SP (broken feedback)."""

             def __init__(self, target_pv: str, drift_rate: float = 0.1):
                 """
                 Args:
                     target_pv: Base PV name (without :SP/:RB suffix)
                     drift_rate: Drift in units/second
                 """
                 self.target_rb = f"{target_pv}:RB"
                 self.target_sp = f"{target_pv}:SP"
                 self.drift_rate = drift_rate
                 self._rb_value = 0.0

             def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
                 return {}  # Let base set initial

             def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
                 if pv_name == self.target_sp:
                     return {}  # Block normal SP->RB update
                 return None  # Delegate everything else

             def step(self, dt: float) -> dict[str, Any]:
                 self._rb_value += self.drift_rate * dt
                 return {self.target_rb: self._rb_value}

      **What this shows:**

      - ``on_write``: Return ``{}`` to handle (but block) SP writes
      - ``step``: Drive RB independently
      - Targeted override: only affects one PV pair

Configuration
-------------

Configure backends using the ``base`` + ``overlays`` structure in ``config.yml``:

.. code-block:: yaml

   simulation:
     channel_database: "data/channels.json"
     ioc:
       name: "my_sim"
       port: 5064
       output_dir: "generated_iocs/"
     base:
       type: "mock_style"                      # Base: defaults for all PVs
       noise_level: 0.01
       update_rate: 10.0
     overlays:
       - file_path: "my_backends/physics.py"   # Override: custom physics
         class_name: "FirstOrderBackend"
         params:
           tau: 2.0

**Benefits of base + overlays:**

- ``base`` is a single dict (no dash) - clear that it's the foundation
- ``overlays`` is a list (with dashes) - clear that multiple can stack
- ``base`` is optional (defaults to ``mock_style``)
- ``overlays`` is optional (defaults to empty list)

**Configuration fields:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Field
     - Description
   * - ``type``
     - Built-in type: ``"mock_style"`` or ``"passthrough"``
   * - ``file_path``
     - Path to Python file (relative to config.yml)
   * - ``module_path``
     - Python import path (alternative to ``file_path``, requires PYTHONPATH setup)
   * - ``class_name``
     - Class name to instantiate (required for custom backends)
   * - ``params``
     - Dict of kwargs passed to ``__init__``

Backend Chaining
================

Multiple backends can be composed together using ``base`` + ``overlays``.
Order matters: **base runs first, overlays override in order**.

Chain Semantics
---------------

**For ``on_write()``:**

1. Backends are checked from **last to first** (later overlays get priority)
2. First backend to return a ``dict`` (not ``None``) handles the write
3. If all return ``None``, empty dict ``{}`` is used

**For ``step()``:**

1. All backends run in order (base, then overlays)
2. Results are merged with **last wins** on conflicts

**For ``initialize()``:**

1. All backends run in order (base, then overlays)
2. Results are merged with **last wins** on conflicts

Configuration Examples
----------------------

Using the ``DriftBackend`` from the Fault Example tab:

.. code-block:: yaml

   simulation:
     base:
       type: "mock_style"                        # Base
     overlays:
       - file_path: "my_backends/physics.py"     # Override QUAD:Q1
         class_name: "DriftBackend"
         params:
           target_pv: "QUAD:Q1:CURRENT"
           drift_rate: 0.5

Multiple overlays (chaining three backends):

.. code-block:: yaml

   simulation:
     base:
       type: "mock_style"                        # Base for all PVs
     overlays:
       - file_path: "my_backends/physics.py"     # Physics for setpoints
         class_name: "FirstOrderBackend"
         params:
           tau: 2.0
       - file_path: "my_backends/physics.py"     # Break one specific PV
         class_name: "DriftBackend"
         params:
           target_pv: "QUAD:Q1:CURRENT"
           drift_rate: 0.1

.. note::

   **Later overlays completely override earlier ones for conflicting PVs.**

   In the example above, ``DriftBackend`` takes full control of ``QUAD:Q1:CURRENT``:

   - ``on_write``: DriftBackend returns ``{}`` for the SP, blocking ``FirstOrderBackend`` from seeing it
   - ``step``: Both backends run, but DriftBackend's RB value overwrites FirstOrderBackend's

   This is intentional for fault injection - the fault backend needs complete control.
   For PVs not targeted by ``DriftBackend``, ``FirstOrderBackend`` operates normally.

   The ``target_pv`` must reference a PV that exists in your ``channel_database``.

Quick Start
===========

**1. Create your backend** (copy from the Protocol tabs above):

.. code-block:: bash

   mkdir -p my_backends
   # Add FirstOrderBackend or DriftBackend to my_backends/physics.py

No ``__init__.py`` needed - the ``file_path`` approach loads Python files directly.

**2. Add to config.yml:**

.. code-block:: yaml

   simulation:
     channel_database: "data/channels.json"
     ioc:
       name: "my_sim"
       output_dir: "generated_iocs/"
     base:
       type: "mock_style"
     overlays:
       - file_path: "my_backends/physics.py"
         class_name: "FirstOrderBackend"
         params:
           tau: 1.0

The ``file_path`` is resolved relative to ``config.yml``.

**3. Generate and run:**

.. code-block:: bash

   osprey generate soft-ioc
   python generated_iocs/my_sim_ioc.py

**4. Test with caget/caput:**

.. code-block:: bash

   caput QUAD:Q1:CURRENT:SP 150   # Write setpoint
   caget QUAD:Q1:CURRENT:RB       # Watch RB approach SP

.. seealso::

   :doc:`00_cli-reference`
       Complete ``osprey generate soft-ioc`` command reference

   :doc:`../05_production-systems/06_control-system-integration`
       Control system connector architecture

   `caproto documentation <https://caproto.github.io/caproto/>`_
       Python EPICS server library used by generated IOCs
