Osprey Framework Documentation
================================

.. admonition:: 🚧 Early Access Documentation
   :class: warning

   **Current Release**: |release| Early Access

   This documentation is part of an early access release and is **under active development**.
   Many sections are still being written, edited, or reorganized.
   Expect **inconsistencies**, missing content, outdated references, and broken cross-links.

   We welcome feedback! If you find issues or have suggestions, please open an issue on our GitHub page.


What is Osprey Framework?
--------------------------

The **Osprey Framework** is a production-ready architecture for deploying agentic AI in large-scale, safety-critical control system environments. Built on :doc:`LangGraph's StateGraph foundation <developer-guides/01_understanding-the-framework/03_langgraph-integration>`, it transforms natural language inputs into transparent, auditable execution plans designed for operational safety and reliability.

Developed for scientific facilities managing complex technical infrastructure such as particle accelerators, fusion experiments, beamlines, and large telescopes, Osprey addresses control-specific challenges: :doc:`semantic addressing across large channel namespaces <getting-started/control-assistant-part2-channel-finder>`, :doc:`plan-first orchestration <developer-guides/01_understanding-the-framework/04_orchestrator-first-philosophy>` with hardware-write detection, :doc:`protocol-agnostic integration with control stacks <developer-guides/05_production-systems/06_control-system-integration>` (EPICS, LabVIEW, Tango), and :doc:`mandatory human oversight for safety-critical operations <api_reference/03_production_systems/01_human-approval>`.

Core Architecture
------------------

Osprey transforms control room operations through a control-aware architecture designed for safety-critical environments. The framework converts natural-language requests into transparent, executable plans through a four-stage pipeline—Task Extraction, Classification, Orchestration, and Execution—with comprehensive safety enforcement, checkpointing, and artifact tracking:

1. :doc:`Task Extraction <developer-guides/04_infrastructure-components/02_task-extraction-system>` → Convert conversational context into structured, actionable objectives

2. :doc:`Classification <developer-guides/04_infrastructure-components/03_classification-and-routing>` → Dynamically select relevant capabilities from your domain-specific toolkit

3. :doc:`Orchestration <developer-guides/04_infrastructure-components/04_orchestrator-planning>` → Generate complete execution plans with explicit dependencies and human oversight

4. **Execution** → Execute capabilities with checkpointing, artifact management, and safety controls


.. figure:: _static/resources/workflow_overview.pdf
   :alt: Osprey Framework Architecture
   :align: center
   :width: 100%

   **Production Deployment Example**: This diagram illustrates the framework architecture using capabilities from the :doc:`ALS Accelerator Assistant <example-applications/als-assistant>` - our production deployment at Lawrence Berkeley National Laboratory's Advanced Light Source particle accelerator.

The framework provides:

* **Plan-First Orchestration**: :doc:`Complete execution planning <developer-guides/04_infrastructure-components/04_orchestrator-planning>` with explicit dependencies before any hardware interaction, enabling operator review of all proposed control system operations
* **Control-System Awareness**: Pattern detection and static analysis identify hardware writes; :doc:`PV boundary checking <developer-guides/05_production-systems/06_control-system-integration>` validates setpoints against facility-defined limits before execution
* **Protocol-Agnostic Integration**: :doc:`Pluggable connectors <developer-guides/05_production-systems/06_control-system-integration>` for EPICS, LabVIEW, Tango, and mock environments enable development without hardware and seamless production migration through configuration
* **Scalable Capability Management**: :doc:`Dynamic classification <developer-guides/04_infrastructure-components/03_classification-and-routing>` selects relevant capabilities from large inventories, preventing prompt explosion as facilities expand toolsets
* **Secure Code Execution**: :doc:`Containerized Python generation and execution <developer-guides/05_production-systems/03_python-execution-service/index>` with read-only and write-enabled environments, static analysis, and mandatory approval for hardware-interacting scripts
* **Facility Data Integration**: :doc:`Automatic retrieval from archiver appliances, channel databases, and knowledge bases <developer-guides/05_production-systems/02_data-source-integration>` with intelligent downsampling for large time-series datasets
* **LangGraph Foundation**: Native StateGraph workflows with :doc:`checkpoints, interrupts <developer-guides/01_understanding-the-framework/03_langgraph-integration>`, and :doc:`persistent state management <developer-guides/03_core-framework-systems/01_state-management-architecture>`
* **Safety-First Design**: :ref:`Transparent execution plans <planning-mode-example>` with :doc:`human approval workflows <developer-guides/05_production-systems/01_human-approval-workflows>` and network-level isolation for control room deployment
* **Proven in Production**: Deployed at :doc:`Lawrence Berkeley National Laboratory's Advanced Light Source <example-applications/als-assistant>` managing hundreds of thousands of control channels across accelerator operations


Documentation Structure
-----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: getting-started/index
      :link-type: doc
      :class-header: bg-primary text-white

      Complete implementation guide from environment setup to production deployment, including tutorial applications.

   .. grid-item-card:: 🧠 Developer Guides
      :link: developer-guides/index
      :link-type: doc
      :class-header: bg-info text-white

      Architectural concepts and implementation patterns for deploying agentic AI in control system environments.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 📚 API Reference
      :link: api_reference/index
      :link-type: doc
      :class-header: bg-secondary text-white

      Complete technical reference for all framework components and interfaces.

   .. grid-item-card:: 💡 Applications
      :link: example-applications/index
      :link-type: doc
      :class-header: bg-success text-white

      Reference implementations demonstrating framework usage across different domains.

   .. grid-item-card:: 🤝 Contributing
      :link: contributing/index
      :link-type: doc
      :class-header: bg-warning text-white

      Framework internals, development guidelines, and contribution workflows.
.. dropdown:: Citation
   :color: primary
   :icon: quote

   If you use the Osprey Framework in your research or projects, please cite our `paper <https://arxiv.org/abs/2508.15066>`_:

   .. code-block:: bibtex

      @misc{hellert2025osprey,
            title={Osprey: A Scalable Framework for the Orchestration of Agentic Systems},
            author={Thorsten Hellert and João Montenegro and Antonin Sulc},
            year={2025},
            eprint={2508.15066},
            archivePrefix={arXiv},
            primaryClass={cs.MA},
            url={https://arxiv.org/abs/2508.15066},
      }

.. toctree::
   :hidden:

   getting-started/index
   developer-guides/index
   api_reference/index
   example-applications/index
   contributing/index

