================
How OSPREY works
================

This page explains what OSPREY is, how it works, and why it is designed this
way. You do not need it for the hands-on tutorials. It helps you understand
what happens under the hood.


What is OSPREY?
===============

Running a large scientific facility, such as a particle accelerator, means
juggling dozens of tools: archivers, logbooks, channel finders, OPI screens.
You also hold the operational context in your head. Every task requires
knowing which tool to open, which PV names to use, and what the safe operating
ranges are.

OSPREY puts an AI agent in front of those tools. You ask in natural language:
"read the beam current", "bump the horizontal corrector by 0.5 A", "what
happened to the vacuum last Tuesday". The agent works out which tools to call,
which channels to query, and how to present the results.

The catch: the agent cannot write to hardware on its own. Safety enforcement
lives in the runtime, outside the agent's reasoning. It runs in two places:

- **The hook chain.** A hook is a small program that Claude Code runs before a
  tool call (its PreToolUse event). OSPREY installs a chain of these hooks. They
  check every write and can refuse it.
- **The MCP server** that carries out the call.

The agent has no credential to reach in and disable either one. The agent
proposes actions; the safety system and the operator decide whether they
happen.


How the Agent Works
===================

OSPREY gives you two parts that work together: **the agent** (Claude Code) and
**MCP tools**.

**The OSPREY agent** is an AI assistant that runs in your terminal (or behind
a web interface). It reads natural language, reasons about what to do,
and calls tools to carry out actions. It does not connect to your
control system directly. It has no knowledge of PV names, channel
addresses, or hardware protocols until you give it tools.

**MCP tools** are the interface between the agent and your infrastructure.
OSPREY ships tools for reading and writing control system channels, querying
archivers, searching logbooks, and finding channel addresses. When the agent
starts, it discovers which tools are available and what each one can do. When
you ask a question, the agent decides which tools to call, calls them, and uses
the results to form a response.

This separation matters. The agent is the reasoning layer, and the tools are
the execution layer. You control what the agent can do by choosing which tools
to enable. If you enable only ``channel_read``, the agent can read channels but
has no way to write to them, whatever you ask.


The Safety Model
================

The agent can propose writes but cannot execute them alone. When the agent
proposes a write, you see the target channel and value. You must approve it
before anything reaches hardware, unless you set that machine's write switch to
*allowed*.

A channel write passes three guards, one hook each, in this order:

#. **Who may write.** The hook asks for the **write verdict**: the runtime's
   yes or no on this write to this machine. The verdict refuses the write, with
   no prompt, unless two things hold. First, the login's tier allows writes.
   The **tier** is the most a login may do, set by ``control_system.writes``:
   ``never``, ``asked`` or ``allowed``. Second, that login has armed the
   machine's write switch from the web terminal's chip. The switch is per
   machine, so you can write to the simulator while the machine that has beam
   stays off (:ref:`write-safety-config`).
#. **Limits.** The limits system validates each value against per-channel safe
   ranges and read-only flags.
#. **Approval.** A write that passes both checks still waits for your yes or
   no, unless the switch is at *allowed*.

The runtime enforces these checks in a layer that the agent's tool calls pass
through. They run whether or not the agent cooperates. The agent holds no
credential for the operator interfaces that could change safety settings.


The Connector Abstraction
=========================

OSPREY does not talk to your control system directly. It uses a
**connector**: an adapter that translates tool calls into the right protocol
(EPICS Channel Access, the built-in simulator, or a custom backend).

You select the connector in ``profile.yml`` by setting
``control_system.type`` in its ``config:`` block. Change
``virtual_accelerator`` (the simulator, served in process or from its
container) to ``epics``, and the same agent, tools, and safety hooks work
against real hardware. You change configuration, not code. A project that also
runs an archiver changes the archiver settings in the same step:
:doc:`demo` shows the three ``osprey set`` lines that move its
control system and archiver together.


Next Steps
==========

Put these concepts to work, or go deeper:

**Start here:** :doc:`hello-world-tutorial`
  Build your first agent with one MCP server and the simulator served in process.
  Three commands: ``osprey init``, ``osprey build``, ``osprey web``.

**Then scale up:** :doc:`demo`
  Add channel finding, logbook search, an archive of its own, and
  logbook and channel-finder panels in the web terminal.

**Go deeper:** :doc:`../architecture/index`
  Detailed diagrams of the safety chain, data flow, and the full MCP
  tool catalog.
