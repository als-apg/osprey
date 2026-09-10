=========
Reference
=========

What is the exact key, the exact shape, the exact command. These pages are
lookup material; the how-to guides link here for every value they mention.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Every ``osprey`` command, its options, and its exit codes.

   .. grid-item-card:: Configuration
      :link: configuration/index
      :link-type: doc

      Paged by the file you have open: ``profile.yml``, ``config.yml``, and
      the environment variables.

   .. grid-item-card:: Contracts
      :link: contracts/index
      :link-type: doc

      The JSON and tool shapes services exchange: health reports, connectors,
      the Python executor, ARIEL, Channel Finder, and the facility graph.

   .. grid-item-card:: Ports
      :link: ports
      :link-type: doc

      Every host port a deployment publishes, as an offset from
      ``deployment.port_base`` -- and which setting wins when two disagree.

.. toctree::
   :hidden:

   cli
   configuration/index
   contracts/index
   ports
