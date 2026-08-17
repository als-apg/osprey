=============
Bluesky Plans
=============

The Osprey agent can run real measurement scans — sweep magnets, read beam
monitors, and collect the data — through the same chat and panels you already
use. Three promises shape everything on these pages: a scan waits in a
**queue** until a human deliberately starts it, the queue **survives
restarts**, and **stopping is never locked** — not by any switch, on any
surface.

.. mermaid::

   flowchart LR
       A["You + the agent<br/>compose a plan"] -->|add| B["Queue<br/>waits until started"]
       B -->|start| C["Machine<br/>points land live"]
       C --> D["Data<br/>kept for good"]

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Run Your First Scan
      :link: run-first-scan
      :link-type: doc
      :shadow: md

      Ten minutes on the Virtual Accelerator — ask for a scan in plain words,
      press two buttons, watch the points land. Nothing to configure.

   .. grid-item-card:: Scans and the Queue
      :link: queue
      :link-type: doc
      :shadow: md

      How running actually works: the queue, what needs arming, and the two
      ways to stop.

   .. grid-item-card:: Write Your Own Scan Plans
      :link: write-plans
      :link-type: doc
      :shadow: md

      Let the agent draft a new plan for you, or install your facility's own
      plan library.

.. seealso::

   :doc:`/how-to/web-terminal/panels`
      The PLAN and BLUESKY tabs these guides use.

   :doc:`/how-to/use-virtual-accelerator`
      The simulated machine the tutorial runs against.

.. toctree::
   :hidden:

   run-first-scan
   queue
   write-plans
