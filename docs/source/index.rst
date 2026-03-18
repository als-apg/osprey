Osprey Framework Documentation
================================

What is Osprey Framework?
--------------------------

The **Osprey Framework** is a production-ready architecture for deploying agentic AI in large-scale, safety-critical control system environments. It uses **Claude Code with MCP servers** for orchestration, transforming natural language inputs into transparent, auditable multi-step workflows designed for operational safety and reliability.

Developed for scientific facilities managing complex technical infrastructure such as particle accelerators, fusion experiments, beamlines, and large telescopes, Osprey addresses control-specific challenges: :doc:`semantic addressing across large channel namespaces <getting-started/control-assistant-part2-channel-finder>`, flexible orchestration with hardware-write detection, :doc:`protocol-agnostic integration with control stacks <how-to/add-connector>` (EPICS, LabVIEW, Tango), :doc:`intelligent logbook search <how-to/ariel/index>` across facility electronic logbooks, and mandatory human oversight for safety-critical operations.

.. figure:: _static/resources/architecture_overview.pdf
   :alt: Osprey Framework Architecture
   :align: center
   :width: 100%

   Osprey provides agentic orchestration with human-in-the-loop safety review, translating natural language requests into approved, isolated execution on facility control systems. For a detailed view of the architecture, see :doc:`Architecture <architecture/index>`.

Key Features
------------

* **Claude Code Orchestration**: MCP servers expose domain-specific tools that Claude Code discovers and calls, with full audit trails and operator review
* **Control-System Awareness**: Pattern detection and static analysis identify hardware writes; PV boundary checking validates setpoints against facility-defined limits before execution
* **Protocol-Agnostic Integration**: :doc:`Pluggable connectors <how-to/add-connector>` for EPICS, LabVIEW, Tango, and mock environments enable development without hardware and seamless production migration through configuration
* **Secure Code Execution**: :doc:`Containerized Python execution <how-to/use-python-executor>` with read-only and write-enabled environments, static analysis, and mandatory approval for hardware-interacting scripts
* **Logbook Search (ARIEL)**: :doc:`Intelligent search over facility electronic logbooks <how-to/ariel/index>` with keyword, semantic, RAG, and agentic retrieval modes, pluggable ingestion adapters for any facility, and a built-in web interface
* **Safety-First Design**: Transparent execution plans with human approval workflows and network-level isolation for control room deployment
* **Proven in Production**: Deployed at Lawrence Berkeley National Laboratory's Advanced Light Source managing tens of thousands of control channels across accelerator operations


Documentation Structure
-----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting-started/index
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white

      Install Osprey, create your first project, and deploy a control assistant
      with Claude Code and MCP servers.

   .. grid-item-card:: Architecture
      :link: architecture/index
      :link-type: doc
      :class-header: sd-bg-info sd-text-white

      Core concepts: Claude Code orchestration, MCP servers, connectors,
      human-in-the-loop safety, and the runtime API.

   .. grid-item-card:: How-To Guides
      :link: how-to/index
      :link-type: doc
      :class-header: sd-bg-success sd-text-white

      Task-oriented recipes for adding connectors, configuring providers,
      deploying projects, and customising MCP servers.

   .. grid-item-card:: Contributing
      :link: contributing/index
      :link-type: doc
      :class-header: sd-bg-light

      Development setup, coding standards, testing guidelines, and the
      contribution workflow.

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
   architecture/index
   how-to/index
   contributing/index
