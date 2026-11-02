Osprey Framework Documentation
================================

**An agentic interface to scientific control systems.**

The **Osprey Framework** is an agentic interface and harness for scientific facilities managing complex technical infrastructure, such as particle accelerators. It runs Claude Code as its agent and wraps it in an operator-facing safety policy, a hook-based approval chain, and an MCP-server multiplexer. The model behind the agent is a configuration choice, and the code specific to Claude Code sits in :ref:`one adapter package <extending-agent-harness>`. The current reference implementation is a browser-based operator workstation; other surfaces (control-room consoles, chat clients, headless services) are possible.

Osprey addresses control-specific challenges: semantic addressing across large channel namespaces, :doc:`protocol-agnostic integration with control stacks </how-to/control-systems/use-connectors>` (EPICS, DOOCS, TANGO and the built-in simulator ship in-tree; LabVIEW and other stacks are supported via custom connectors), :doc:`logbook search <how-to/ariel/index>` across facility electronic logbooks, and mandatory human oversight for safety-critical operations.

A short demo of the operator workstation:

.. raw:: html

   <div class="osprey-demo">
     <video id="osprey-demo" preload="metadata" muted loop playsinline controls
            poster="_static/demo/osprey-demo-light-poster.jpg"
            aria-label="Demo video of the Osprey operator workstation"
            aria-describedby="osprey-demo-note">
       <source src="_static/demo/osprey-demo-light.mp4" type="video/mp4">
       <img src="_static/demo/osprey-demo-light-poster.jpg"
            alt="The Osprey operator workstation with a 3D plot the agent made of three BPMs.">
     </video>
     <p class="osprey-demo-source">
       Recorded on the control-assistant demo, which runs a simulated accelerator
       rather than a real facility. To reproduce it, follow the
       <a href="getting-started/control-assistant.html">control-assistant tutorial</a>.
     </p>
     <p class="osprey-demo-note" id="osprey-demo-note">
       One unedited session of several minutes. The agent's working time is sped up
       (marked ⏩&nbsp;with its factor); typing, clicks and the approval play in real time.
       The clock in the corner shows the real elapsed time.
     </p>
     <div class="osprey-demo-speed" role="group" aria-label="Playback speed" hidden>
       <span>Speed</span>
       <button type="button" data-rate="1">1×</button>
       <button type="button" data-rate="1.5">1.5×</button>
       <button type="button" data-rate="2">2×</button>
       <button type="button" data-rate="3">3×</button>
     </div>
   </div>
   <script src="_static/js/doc-video.js"></script>

For the system design, see :doc:`Architecture <architecture/index>`.

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

   .. grid-item-card:: How-To Guides
      :link: how-to/index
      :link-type: doc
      :class-header: sd-bg-success sd-text-white

      Task-oriented recipes for adding connectors, configuring providers,
      deploying projects, and customising MCP servers.

   .. grid-item-card:: Architecture
      :link: architecture/index
      :link-type: doc
      :class-header: sd-bg-info sd-text-white

      Core concepts: agentic orchestration, MCP servers, connectors,
      human-in-the-loop safety, and the runtime API.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc
      :class-header: sd-bg-secondary sd-text-white

      The exact keys, shapes, and commands: the CLI, every configuration
      file, and the contracts services exchange.

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
   how-to/index
   architecture/index
   reference/index
   contributing/index
