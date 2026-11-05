Getting Started
===============

These pages take you from a fresh machine to a working agent on a simulated
accelerator, and then to a production-style deployment. Follow steps 1 to 3
in order; each one ends with something you can run. Step 4 is optional reading.

**Before you start:** you need a terminal and a text editor.
:doc:`installation` covers everything else, in order: Node.js, Claude Code,
``uv`` (which manages Python for you), an API key for your AI provider, and the
``osprey`` CLI.

.. tip::

   To set up OSPREY for your own facility right away, use the **guided setup**:
   :doc:`osprey-install`. It is a conversation with your coding agent. The
   agent takes stock of what you have and builds the project with you. The
   step-by-step tutorials cover the same ground and teach the concepts.

Your path
---------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 1. Install OSPREY
      :link: installation
      :link-type: doc
      :class-header: bg-info text-white

      Install the ``osprey`` CLI and configure a provider. A container runtime (Docker or Podman) is the optional last step, needed for ``osprey up``.

   .. grid-item-card:: 2. Hello World
      :link: hello-world-tutorial
      :link-type: doc
      :class-header: bg-success text-white

      Build your first agent: one MCP server and the simulator, served in
      process. You read channels, enable writes, and see the three safety
      guards at work.

   .. grid-item-card:: 3. Control Systems
      :link: control-assistant
      :link-type: doc
      :class-header: bg-success text-white

      Build the ``control-assistant`` preset. It adds natural-language channel
      finding, a seeded logbook and an archive, on the same safety model. The
      last step points it at real hardware.

   .. grid-item-card:: 4. How OSPREY works
      :link: conceptual-tutorial
      :link-type: doc
      :class-header: bg-info text-white

      Learn what the agent, its tools, the safety guards and the connectors
      do, and why. You can read this page at any point; no tutorial needs it.

Next
----

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Guided facility setup
      :link: osprey-install
      :link-type: doc
      :class-header: bg-primary text-white

      Set OSPREY up for your own detector, beamline or accelerator subsystem
      through a guided conversation.

   .. grid-item-card:: How-to guides
      :link: ../how-to/index
      :link-type: doc
      :class-header: bg-primary text-white

      Task-oriented guides for connectors, MCP servers, providers and more.

   .. grid-item-card:: Architecture
      :link: ../architecture/index
      :link-type: doc
      :class-header: bg-warning text-white

      The OSPREY agent and MCP architecture, data flow and key concepts.


.. toctree::
   :hidden:

   installation
   hello-world-tutorial
   control-assistant
   conceptual-tutorial
   osprey-install
