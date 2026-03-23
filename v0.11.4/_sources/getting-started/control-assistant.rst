===================================
Production Control Systems Tutorial
===================================

This tutorial demonstrates **production-grade patterns for control system integration** through an accelerator physics application based on the deployed :doc:`ALS Accelerator Assistant <../example-applications/als-assistant>`. Unlike the Hello World tutorial's simple mock API and workflow, you'll work with scalable patterns for scientific facility integration. The tutorial provides **two alternative semantic channel finding implementations**—in-context search and hierarchical navigation—allowing you to choose the approach that best fits your facility's needs. You'll implement a service layer architecture with supporting tools including a standalone CLI, database utilities, and benchmark runners.

What You'll Build
=================

An accelerator control system assistant that finds control system addresses (like EPICS PVs) using natural language queries. This demonstrates the core challenge in any large control system: **translating human intent to specific hardware addresses**.

.. tab-set::

   .. tab-item:: Example Queries

      .. code-block:: text

         "What is the current beam current?"
         → Retrieves live value from the control system

         "Plot the beam current over the last 24 hours"
         → Demonstrates archiver access and data visualization

         "Calculate the mean and standard deviation of the beam current in the last 24 hours"
         → Performs statistical analysis on historical data

   .. tab-item:: The Challenge

      Control systems at scientific facilities have naming challenges:

      - **Scale**: few hundred to hundreds of thousands of addressable channels
      - **Inconsistent Naming**: Legacy systems, evolving conventions
      - **Domain Knowledge**: Requires subsystem expertise
      - **Critical Impact**: Wrong channel = equipment damage
      - **Safe Execution**: Requires secure Python execution for data analysis

      Natural language channel finding solves this by letting operators describe what they want instead of memorizing cryptic addresses.

   .. tab-item:: Real-World Context

      At large accelerator facilities like the Advanced Light Source:

      - **Hundreds of thousands of channels** across all subsystems
      - **Distributed expertise** across physics, RF, magnets, vacuum, controls
      - **Complex operations** requiring rapid diagnosis and response
      - **Safety-critical** decisions based on correct channel identification

      This tutorial's patterns scale from small control systems :doc:`to facilities of this scale and complexity <../example-applications/als-assistant>`.

Tutorial Structure
==================

This tutorial is divided into four parts that build progressively on each other:

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 📦 Part 1: Getting Started
      :link: control-assistant-part1-setup
      :link-type: doc

      Create your project and configure the framework

      - Create your control assistant project
      - Configure AI models, providers, and safety controls
      - Set up environment variables and services

   .. grid-item-card:: 🔍 Part 2: Channel Finder
      :link: control-assistant-part2-channel-finder
      :link-type: doc

      Build and test semantic channel finding

      - Understand in-context vs hierarchical pipelines
      - Build channel databases from your control system
      - Test with CLI and validate with benchmarks
      - Service layer integration patterns

   .. grid-item-card:: ⚛︎ Part 3: Running Your Agent
      :link: control-assistant-part3-production
      :link-type: doc

      Execute your agent and see the framework in action

      - Run queries through the complete pipeline
      - Observe multi-step orchestration
      - Use mock services for hardware-free development
      - Deploy to production control systems

   .. grid-item-card:: 🎨 Part 4: Customization & Extension
      :link: control-assistant-part4-customization
      :link-type: doc

      Customize and extend your assistant

      - Add facility-specific domain knowledge
      - Configure models for optimal performance
      - Use advanced debugging features
      - Build custom capabilities

.. toctree::
   :hidden:
   :maxdepth: 1

   control-assistant-part1-setup
   control-assistant-part2-channel-finder
   control-assistant-part3-production
   control-assistant-part4-customization
