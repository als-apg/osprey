.. _architecture-safety-chain:

Safety Chain
============

Every tool call the agent makes through the harness passes through a configurable chain of
**PreToolUse hooks** before reaching the MCP server. OSPREY's MCP servers are stdio child
processes the harness starts --- they expose no network port of their own --- so every tool call
that reaches one has already passed the hook chain. That guarantee belongs to the harness: a
process on the host that runs the server command itself is outside it, which is why host access
is part of the trust boundary. The chain for ``channel_write`` — the most safety-critical tool — has three stages:

.. raw:: html
   :file: ../_diagrams/safety-chain.html

1. **osprey_writes_check** — Kill switch. Blocks a write when the session's control
   target is not armed for writes — ``control_system.connector.<type>.writes_enabled``
   in ``config.yml``, or the ``control_system.writes_enabled`` a type with no block of
   its own inherits. Applies to both ``channel_write`` and ``execute``.

2. **osprey_limits** — Validates the setpoint against the channel limits database
   (min, max, step size, writable flag). Only applies to ``channel_write``.

3. **osprey_approval** — Human approval gate. Per-tool policy dispatch: ``always`` (require
   approval every time), ``selective`` (ask the Osprey agent to decide), or ``skip``.
