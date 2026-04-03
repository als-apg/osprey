===================
Conceptual Tutorial
===================

Before building useful agentic AI applications using Osprey, it's important to
understand the framework's core architecture and operational model.

This conceptual tutorial introduces the fundamental concepts and design patterns
that will prepare you for the hands-on coding journey ahead.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 12):** "Building on Proven Foundations -- Osprey is built on LangGraph,
   a production-ready orchestration framework developed by LangChain..."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old foundation section described LangGraph as the orchestration
   layer. Osprey now uses Claude Code with MCP servers for orchestration, but describing
   how Claude Code orchestrates things is out of scope for mechanical porting.
   **Action needed:** Write a new "Building on Proven Foundations" section explaining
   Claude Code + MCP as the orchestration foundation.

How Osprey Works
================

An agentic AI application can be treated as a chatbot with tools. Currently there are two major
types of agentic AI applications: ReAct agents and Planning agents.

.. tab-set::

   .. tab-item:: ReAct Agents

      ReAct agents work in a way that is similar to how LLMs handle chat history.
      When a user query comes in, the agent processes the entire conversation history,
      along with the previous tool usage records, to decide the next action.

      The advantage of ReAct agents is that they can leverage the full power of LLMs to
      dynamically decide what to do next based on the entire context. However, this
      also means that ReAct agents can be less efficient and less predictable, as
      they may revisit previous steps or make decisions that are hard to foresee.
      Additionally, ReAct agents may get lost in complicated setups with many tools
      and complex state management.

   .. tab-item:: Planning Agents

      Planning agents, on the other hand, separate the "thinking" and "acting" phases.
      For every user query, they first create a comprehensive plan, breaking down
      the task into manageable steps. Once the plan is formulated, the agent
      executes each step sequentially, utilizing tools as necessary to accomplish
      each subtask.

      The advantage of Planning agents is that the execution path is more structured and predictable,
      as the plan is created upfront. This can lead to more efficient use of tools and resources.
      Additionally, planning agents have less dependency on the LLM's ability to generate stable outputs
      since they decompose the task into smaller, easier steps. Each step can be
      handled with more focused prompts and potentially smaller models.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 62):** "Osprey supports both orchestration modes and defaults to
   the Planning approach... the building blocks are the same: Capabilities -- modular
   components that encapsulate domain-specific business logic and tool integrations."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** Capabilities are now MCP tools, not Python classes with
   requires/provides declarations. The old text described capability chaining and
   context-based data flow, which no longer applies.
   **Action needed:** Describe how Osprey now exposes domain logic as MCP tools and how
   Claude Code selects and chains them.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 70):** "Osprey uses Contexts -- strictly typed Pydantic data classes
   that provide a structured layer for storing and communicating data between capabilities."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The Context system (CapabilityContext, Pydantic data classes for
   inter-capability data flow) has been removed. Data now flows through MCP tool
   call/response cycles, but describing that mechanism requires new content.
   **Action needed:** Explain how data flows between MCP tools in the current architecture.

Mindflow to Build a Weather Assistant in Osprey
================================================

Assume we want to build a weather assistant that can provide weather information based on user queries.

What would users ask
--------------------

First step is to think about what queries we want to support, or we imagine users would ask. Based on our
experience in real life, for the weather assistant, users would typically ask questions like:

- "What's the weather like in San Francisco today?"
- "Will it rain tomorrow in New York?"
- "Give me a 5-day weather forecast for Los Angeles."
- "What about the day after tomorrow?" -- referring to previous query

.. admonition:: PLACEHOLDER: MISSING-EXAMPLE
   :class: warning

   **Old content (line 103):** "What capabilities are needed -- FetchWeatherCapability,
   ExtractLocationCapability, ExtractDateCapability with context classes LocationContext,
   DateContext, WeatherContext, and RespondCapability."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old design walkthrough was framed around BaseCapability
   subclasses with requires/provides context declarations. The new architecture uses
   MCP tools, so the design process and building blocks are different.
   **Action needed:** Rewrite the weather assistant design walkthrough using MCP tools
   instead of Capability classes and Context classes.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 193):** "The Osprey Design Pattern -- 1. Identify required capabilities,
   2. Define necessary contexts, 3. Check for missing data, 4. Repeat."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The iterative design pattern was framed around capabilities and
   contexts. The new design process likely centers on MCP tool design, but the exact
   recommended workflow needs human authoring.
   **Action needed:** Write a new "Osprey Design Pattern" admonition for MCP-based design.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 209):** "How Osprey Chains Capabilities Together -- The orchestrator
   automatically chains them... Plan-First Mode (default) vs Reactive Mode (ReAct)...
   orchestration_mode: react in config.yml."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old orchestration section described LangGraph-based plan-first
   and ReAct modes with capability chaining via contexts. Claude Code handles orchestration
   differently, and the old config knob no longer applies.
   **Action needed:** Describe how Claude Code orchestrates MCP tool calls, including
   any relevant configuration options.

Next Steps
==========

Now that you understand the core concepts, you're ready to build:

**Start here:** :doc:`hello-world-tutorial`
  Implements a weather assistant using an even simpler architecture (single MCP tool)
  to help you get hands-on quickly. Learn the framework basics before tackling complexity.

**Then scale up:** :doc:`control-assistant`
  Demonstrates the full modular architecture from this tutorial applied to a real
  industrial control system with multiple MCP tools and production deployment.
