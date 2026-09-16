===============
Control Systems
===============

A connector is OSPREY's single interface to a control system. The agent, the
plans and the safety layers are all written against that one interface, which
is what makes the machine underneath it interchangeable: a mock for
development, a simulator for rehearsal, the real hardware for production.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Use Connectors
      :link: use-connectors
      :link-type: doc
      :shadow: md

      The two-layer connector abstraction, the connectors that ship in-tree,
      and where to register your own -- and mock to production is a change of
      ``control_system.type``, not of code.

   .. grid-item-card:: Use the Virtual Accelerator
      :link: use-virtual-accelerator
      :link-type: doc
      :shadow: md

      Run against a containerized simulator serving real EPICS Channel Access,
      with PyAT physics behind the storage-ring lattice channels, so correctors
      move and BPMs respond.

   .. grid-item-card:: Switch the Control Target at Run Time
      :link: switch-control-target
      :link-type: doc
      :shadow: md

      Rehearse a piece of work on the simulator, then run the same work on the
      live machine without rebuilding the project or restarting anything.

.. toctree::
   :hidden:

   use-connectors
   use-virtual-accelerator
   switch-control-target
