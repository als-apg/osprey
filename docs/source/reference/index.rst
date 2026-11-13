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

      Paged by the file you have open: ``profile.yml``, ``config.yml``, the
      environment variables, and a scenario file.

   .. grid-item-card:: Contracts
      :link: contracts/index
      :link-type: doc

      The JSON and tool shapes services exchange: health reports, connectors,
      the Python executor, ARIEL, Channel Finder, the facility graph, and six
      more.

   .. grid-item-card:: osprey-connectors API
      :link: connectors-api/index
      :link-type: doc

      The Python API of the connectors package: the connector contracts, the
      factory, the errors, and the write-safety records.

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
   connectors-api/index
   ports
