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

Developed for scientific facilities managing complex technical infrastructure such as particle accelerators, fusion experiments, beamlines, and large telescopes, Osprey addresses control-specific challenges: :doc:`semantic addressing across large channel namespaces <getting-started/control-assistant-part2-channel-finder>`, :doc:`plan-first orchestration <developer-guides/01_understanding-the-framework/04_orchestrator-first-philosophy>` with hardware-write detection, :doc:`protocol-agnostic integration with control stacks <developer-guides/05_production-systems/06_control-system-integration>` (EPICS, LabVIEW, Tango), :doc:`intelligent logbook search <developer-guides/05_production-systems/07_logbook-search-service/index>` across facility electronic logbooks, and :doc:`mandatory human oversight for safety-critical operations <developer-guides/05_production-systems/01_human-approval-workflows>`.

.. figure:: _static/resources/architecture_overview.pdf
   :alt: Osprey Framework Architecture
   :align: center
   :width: 100%

   Osprey provides agentic orchestration with human-in-the-loop safety review, translating natural language requests into approved, isolated execution on facility control systems. For a detailed view of the pipeline workflow and component interactions, see :doc:`Understanding Osprey <developer-guides/01_understanding-the-framework/index>`.

Key Features
------------

* **Plan-First Orchestration**: :doc:`Complete execution planning <developer-guides/04_infrastructure-components/04_orchestrator-planning>` with explicit dependencies before any hardware interaction, enabling operator review of all proposed control system operations
* **Control-System Awareness**: Pattern detection and static analysis identify hardware writes; :doc:`PV boundary checking <developer-guides/05_production-systems/06_control-system-integration>` validates setpoints against facility-defined limits before execution
* **Protocol-Agnostic Integration**: :doc:`Pluggable connectors <developer-guides/05_production-systems/06_control-system-integration>` for EPICS, LabVIEW, Tango, and mock environments enable development without hardware and seamless production migration through configuration
* **Scalable Capability Management**: :doc:`Dynamic classification <developer-guides/04_infrastructure-components/03_classification-and-routing>` selects relevant capabilities from large inventories, preventing prompt explosion as facilities expand toolsets
* **Secure Code Execution**: :doc:`Containerized Python generation and execution <developer-guides/05_production-systems/03_python-execution-service/index>` with read-only and write-enabled environments, static analysis, and mandatory approval for hardware-interacting scripts
* **Facility Data Integration**: :doc:`Automatic retrieval from archiver appliances, channel databases, and knowledge bases <developer-guides/05_production-systems/02_data-source-integration>` with intelligent downsampling for large time-series datasets
* **Logbook Search (ARIEL)**: :doc:`Intelligent search over facility electronic logbooks <developer-guides/05_production-systems/07_logbook-search-service/index>` with keyword, semantic, RAG, and agentic retrieval modes, pluggable ingestion adapters for any facility, and a built-in web interface
* **LangGraph Foundation**: Native StateGraph workflows with :doc:`checkpoints, interrupts <developer-guides/01_understanding-the-framework/03_langgraph-integration>`, and :doc:`persistent state management <developer-guides/03_core-framework-systems/01_state-management-architecture>`
* **Safety-First Design**: :ref:`Transparent execution plans <planning-mode-example>` with :doc:`human approval workflows <developer-guides/05_production-systems/01_human-approval-workflows>` and network-level isolation for control room deployment
* **Proven in Production**: Deployed at :doc:`Lawrence Berkeley National Laboratory's Advanced Light Source <example-applications/als-assistant>` managing tens of thousands of control channels across accelerator operations


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

   If you use the Osprey Framework in your research or projects, please cite our `paper <https://doi.org/10.1063/5.0306302>`_:

   .. code-block:: bibtex

      @article{10.1063/5.0306302,
            author = {Hellert, Thorsten and Montenegro, João and Sulc, Antonin},
            title = {Osprey: Production-ready agentic AI for safety-critical control systems},
            journal = {APL Machine Learning},
            volume = {4},
            number = {1},
            pages = {016103},
            year = {2026},
            month = {02},
            doi = {10.1063/5.0306302},
            url = {https://doi.org/10.1063/5.0306302},
      }

.. toctree::
   :hidden:

   getting-started/index
   developer-guides/index
   api_reference/index
   example-applications/index
   contributing/index

