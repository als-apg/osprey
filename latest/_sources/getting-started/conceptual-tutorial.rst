===================
Conceptual Tutorial
===================

Before building useful agentic AI applications using Osprey, it's important to
understand the framework's core architecture and operational model.

This conceptual tutorial introduces the fundamental concepts and design patterns
that will prepare you for the hands-on coding journey ahead.

Building on Proven Foundations
===============================

Osprey doesn't reinvent the wheel. At its core, the framework is built on
`LangGraph <https://github.com/langchain-ai/langgraph>`_, a production-ready orchestration
framework developed by LangChain for building stateful, long-running AI agents. LangGraph
provides the foundational capabilities that make reliable agent systems possible: durable
execution, checkpointing, human-in-the-loop workflows, and comprehensive state management.

Rather than creating yet another generic agent framework from scratch, Osprey extends LangGraph
with specialized patterns and infrastructure specifically designed for scientific and
high-stakes operational environments. This approach allows us to focus on solving domain-specific
challenges—such as managing large sets of specialized tools, handling complex data flows, and
ensuring transparent execution—while leveraging a battle-tested foundation for the core
orchestration layer.

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

Osprey supports
:doc:`both orchestration modes <../developer-guides/01_understanding-the-framework/04_orchestration-architecture>`
and defaults to the Planning approach (switchable to ReAct in your project configuration).
Regardless of which mode you choose, the building blocks are the same: **Capabilities** --
modular components that encapsulate domain-specific business logic and tool integrations.
Given a user query, the framework determines which capabilities to use, in what order, and
with what inputs, effectively chaining them together to accomplish complex tasks.

A critical architectural distinction of Osprey is how data flows between capabilities.
Unlike standard ReAct agents where tool outputs are returned directly to the LLM's context
(which works for short strings but fails when tools produce large datasets that would overflow
the context window), Osprey uses **Contexts** - strictly typed Pydantic data classes that
provide a structured layer for storing and communicating data between capabilities. This
approach enables efficient handling of large outputs, maintains type safety, and allows data
to persist across conversation turns without consuming valuable context window space.

Capabilities and contexts are the central building blocks of Osprey applications. So when designing your
Osprey application, always think in terms of capabilities and contexts:

- What capabilities do I need to accomplish the task?
- What contexts would the capability need to work?
- What contexts should the capability produce as output?

Now let's look at a simple example to better understand those concepts.

Mindflow to Build a Weather Assistant in Osprey
===============================================

Assume we want to build a weather assistant that can provide weather information based on user queries.

What would users ask
--------------------

First step is to think about what queries we want to support, or we imagine users would ask. Based on our
experience in real life, for the weather assistant, users would typically ask questions like:

- "What's the weather like in San Francisco today?"
- "Will it rain tomorrow in New York?"
- "Give me a 5-day weather forecast for Los Angeles."
- "What about the day after tomorrow?" -- referring to previous query

What capabilities are needed
----------------------------

To deal with the queries above, obviously we need a capability that can fetch weather data,
given the location and date. Let's call it `FetchWeatherCapability`.
This capability would require the location and date as inputs, and return the weather information.
Therefore we'll need the following contexts:

- `LocationContext` -- to represent the location information
- `DateContext` -- to represent the date information
- `WeatherContext` -- to represent the weather information returned by the capability

Beyond fetching weather data, we need a capability to present results to users. Once we have a
`WeatherContext` with weather data, how do we communicate it back in natural language? Osprey provides
:ref:`RespondCapability <respond-capability>` - a built-in capability that generates natural language
responses from execution results and available contexts.

Are those capabilities sufficient? Maybe not. Thinking more carefully about what we have so far: how can we get the
`LocationContext` and `DateContext` from user queries? It could be easy if the user query is straightforward, but
what if the user query looks like:

- "What's the weather like in NYC?"
- "Will it rain tonight around Stanford university?"

So we probably need another capability to extract location and date information from user queries:

- `ExtractLocationCapability` -- to extract location information from user queries
- `ExtractDateCapability` -- to extract date information from user queries

The Final Design
----------------

Following this iterative thinking process, here's the complete weather assistant architecture:

**Capabilities**

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🌍 ExtractLocationCapability
      :class-header: bg-primary text-white

      Parse location information from user queries.

      **Requires:** None

      **Provides:** LocationContext

   .. grid-item-card:: 📅 ExtractDateCapability
      :class-header: bg-primary text-white

      Parse date information from user queries.

      **Requires:** None

      **Provides:** DateContext

   .. grid-item-card:: ☀️ FetchWeatherCapability
      :class-header: bg-info text-white

      Call weather API to fetch weather information based on location and date.

      **Requires:** LocationContext, DateContext

      **Provides:** WeatherContext

   .. grid-item-card:: 💬 RespondCapability
      :class-header: bg-secondary text-white

      Generate natural language responses from execution results.

      **Requires:** None

      **Provides:** None

**Context Classes**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Context
     - Purpose
   * - ``LocationContext``
     - Stores parsed location information (city, region)
   * - ``DateContext``
     - Stores parsed date/time information
   * - ``WeatherContext``
     - Stores weather data (temperature, conditions, etc.)

.. admonition:: The Osprey Design Pattern
   :class: tip

   When designing any Osprey application, follow this iterative process:

   1. **Identify required capabilities** - What domain-specific tasks need to be performed?
   2. **Define necessary contexts** - What data does each capability need and produce?
   3. **Check for missing data** - How will capabilities get their required contexts?

      * If straightforward to extract → no additional capabilities needed
      * If complex → create extraction/transformation capabilities

   4. **Repeat** until all data dependencies are resolved

   This bottom-up thinking ensures your agent has all the pieces needed to accomplish user goals.

How Osprey Chains Capabilities Together
=======================================

Once you've designed your capabilities and contexts, Osprey's orchestrator automatically
chains them together. The framework supports two orchestration modes that differ in *when*
decisions are made, but both use the same capability and context infrastructure.

.. tab-set::

   .. tab-item:: Plan-First Mode (default)

      The orchestrator creates a **complete plan upfront** before any execution begins.

      **Query: "What's the weather in San Francisco today?"**

      The orchestrator creates this plan:

      1. **ExtractLocationCapability** → produces ``LocationContext(location="San Francisco")``
      2. **ExtractDateCapability** → produces ``DateContext(date="today")``
      3. **FetchWeatherCapability** → uses ``LocationContext`` + ``DateContext`` → produces ``WeatherContext``
      4. **RespondCapability** → generates natural language response from ``WeatherContext``

      **Key Observations:**

      - Each plan is created **upfront** before execution begins
      - Capabilities are **chained together** - the output of one becomes the input to another
      - Predictable and efficient: a single LLM call creates the entire plan

   .. tab-item:: Reactive Mode (ReAct)

      The orchestrator decides **one step at a time**, observing results between steps.

      **Query: "What's the weather in San Francisco today?"**

      The reactive orchestrator loop:

      1. **Decide** → ExtractLocationCapability → **Observe** result
      2. **Decide** → ExtractDateCapability → **Observe** result
      3. **Decide** → FetchWeatherCapability → **Observe** result
      4. **Decide** → Respond to user

      **Key Observations:**

      - Each step is decided **after** observing previous results
      - Adapts dynamically to intermediate outcomes and errors
      - Better suited for exploratory or error-prone tasks

      Enable with: ``execution_control.agent_control.orchestration_mode: react`` in ``config.yml``

**In both modes:**

- The orchestrator **selects capabilities** based on what's needed for each query
- Capabilities are **chained together** - the output of one becomes the input to another
- **RespondCapability** accesses available contexts to generate responses
- Your capabilities work identically regardless of orchestration mode

Next Steps
==========

Now that you understand the core concepts, you're ready to build:

**Start here:** :doc:`hello-world-tutorial`
  Implements a weather assistant using an even simpler architecture (single capability)
  to help you get hands-on quickly. Learn the framework basics before tackling complexity.

**Then scale up:** :doc:`control-assistant`
  Demonstrates the full modular architecture from this tutorial applied to a real
  industrial control system with 8+ capabilities and production deployment.
