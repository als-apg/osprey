===================
Conceptual Tutorial
===================

This page explains what OSPREY is, how it works, and why it is
designed the way it is. You do not need to read this before starting
the hands-on tutorials, but it will help you understand what is
happening under the hood.


What is OSPREY?
===============

Operating a particle accelerator, beamline, or fusion experiment means
juggling dozens of tools --- archivers, logbooks, channel finders, OPI
screens --- while holding operational context in your head. Every task
requires knowing which tool to open, which PV names to use, and what
the safe operating ranges are.

OSPREY puts an AI agent in front of those tools. You interact in
natural language --- "read the beam current", "bump the horizontal
corrector by 0.5 A", "what happened to the vacuum last Tuesday" ---
and the agent figures out which tools to call, which channels to query,
and how to present the results.

The catch: the agent can never write to hardware on its own. Safety
enforcement lives in code that the agent cannot modify or bypass, not
in the agent's own judgment. The agent proposes actions; the safety
system and the operator decide whether they happen.


How the Agent Works
===================

The agent is built on two components: **Claude Code** and **MCP tools**.

**Claude Code** is an AI assistant that runs in your terminal (or behind
a web interface). It reads natural language, reasons about what to do,
and calls tools to carry out actions. It does not connect to your
control system directly --- it has no knowledge of PV names, channel
addresses, or hardware protocols until you give it tools.

**MCP tools** are the interface between the agent and your
infrastructure. OSPREY ships tools for reading and writing control
system channels, querying archivers, searching logbooks, and finding
channel addresses. When the agent starts, it discovers which tools are
available and what each tool can do. When you ask a question, Claude
decides which tools to call, calls them, and uses the results to form
a response.

This separation matters: Claude is the reasoning layer, and the tools
are the execution layer. You control what the agent can do by choosing
which tools to enable. If you only enable ``channel_read``, the agent
can read channels but has no way to write to them --- regardless of
what the user asks.


The Safety Model
================

The agent can propose writes but cannot execute them alone. When a
write is proposed, the operator sees the target channel and value and
must explicitly approve before anything reaches hardware.

Two automated checks run behind the scenes. The **kill switch**
(``writes_enabled`` in ``config.yml``) blocks all writes globally when
off --- no prompt appears. The **limits system** validates each value
against per-channel safe ranges and read-only flags. If either check
fails, the write is blocked before the operator is ever asked.

These checks are enforced by the runtime. The agent cannot modify,
skip, or influence them.


The Connector Abstraction
=========================

OSPREY does not talk to your control system directly. It uses a
**connector** --- an adapter that translates tool calls into the
appropriate protocol (EPICS Channel Access, mock data, or a custom
backend).

You select the connector in ``config.yml`` by setting
``control_system.type``. Change ``mock`` to ``epics`` and the same
agent, tools, and safety hooks work against real hardware. No code
changes, no reconfiguration --- one setting.


Next Steps
==========

Now that you understand the core concepts, you're ready to build:

**Start here:** :doc:`hello-world-tutorial`
  Build your first agent with one MCP server and a mock control system.
  One command to create, one command to run.

**Then scale up:** :doc:`control-assistant`
  Add channel finding, logbook search, archiver access, and a web
  terminal for production deployment.

**Go deeper:** :doc:`../architecture/index`
  Detailed diagrams of the safety chain, data flow, and the full MCP
  tool catalog.
