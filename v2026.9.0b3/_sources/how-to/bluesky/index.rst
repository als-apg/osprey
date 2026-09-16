=============
Bluesky Plans
=============

The Osprey agent can run real measurement plans — sweep magnets, read beam
monitors, and collect the data — through the same chat and panels you already
use. The plans and the queue are `Bluesky <https://blueskyproject.io/>`_'s;
Osprey deploys those services, wires them to your control system, and puts the
agent and a human approval step in front.

.. raw:: html
   :file: ../../_diagrams/bluesky-overview.html

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Run Your First Plan
      :link: run-first-plan
      :link-type: doc
      :shadow: md

      Ten minutes on the Virtual Accelerator — ask for a plan in plain words,
      press two buttons, watch the points land. Nothing to configure.

   .. grid-item-card:: Plans and the Queue
      :link: queue
      :link-type: doc
      :shadow: md

      How running actually works: the queue, what needs arming, and the two
      ways to stop.

   .. grid-item-card:: Write Your Own Plans
      :link: write-plans
      :link-type: doc
      :shadow: md

      Let the agent draft a new plan for you, or install your facility's own
      plan library.

.. seealso::

   :doc:`/how-to/web-terminal/panels`
      The PLAN and BLUESKY tabs these guides use.

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      The simulated machine the tutorial runs against.

.. toctree::
   :hidden:

   run-first-plan
   queue
   write-plans
