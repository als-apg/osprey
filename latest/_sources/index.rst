Osprey Framework Documentation
================================

**An agentic interface for safety-critical control systems.**

The **Osprey Framework** is an agentic interface and harness for scientific facilities managing complex technical infrastructure — particle accelerators, fusion experiments, beamlines, and large telescopes. It wraps a coding agent in an operator-facing safety policy, a hook-based approval chain, and an MCP-server multiplexer, so the agent layer, the underlying LLM, and the compute backend are each replaceable without changing what the operator sees. The current reference implementation is a browser-based operator workstation; other surfaces (control-room consoles, chat clients, headless services) are possible.

Osprey addresses control-specific challenges: semantic addressing across large channel namespaces, :doc:`protocol-agnostic integration with control stacks <how-to/add-connector>` (EPICS, LabVIEW, Tango), :doc:`intelligent logbook search <how-to/ariel/index>` across facility electronic logbooks, and mandatory human oversight for safety-critical operations.

.. figure:: _static/resources/architecture.png
   :alt: Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line.
   :align: center
   :width: 100%

   Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line. For a detailed view, see :doc:`Architecture <architecture/index>`.

Documentation Structure
-----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting-started/index
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white

      Install Osprey, create your first project, and deploy a control assistant
      with a coding agent and MCP servers.

   .. grid-item-card:: Architecture
      :link: architecture/index
      :link-type: doc
      :class-header: sd-bg-info sd-text-white

      Core concepts: agentic orchestration, MCP servers, connectors,
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


Proven in Production
--------------------

Osprey is deployed at Lawrence Berkeley National Laboratory's Advanced Light Source, managing tens of thousands of control channels across accelerator operations.


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
