=============
Configuration
=============

Two files and the environment decide what a deployment does, and these pages
are split by the one you have open. ``profile.yml`` is the file you edit: the
build profile a facility writes, and the input ``osprey build`` renders from.
``config.yml`` is what a running project reads -- rendered from the profile, so
it is where you look a setting up; a change that should survive the next build
belongs in the profile. Anything that must not sit in a committed file, such as
API keys and per-host values, arrives from the environment instead.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Build Profile — profile.yml
      :link: profile
      :link-type: doc
      :shadow: md

      Every key a build profile accepts, from the core field table through
      ``config:`` overrides, MCP servers, tool permissions, services and
      dependencies -- and what the build does with each.

   .. grid-item-card:: Runtime Configuration — config.yml
      :link: config
      :link-type: doc
      :shadow: md

      The rendered runtime file: the ``health:`` diagnostic suite, the ``web:``
      block behind the browser UI, and the deployment keys that choose each
      service's container image.

   .. grid-item-card:: Environment Variables
      :link: environment-variables
      :link-type: doc
      :shadow: md

      The environment variables the OSPREY CLI reads and where each belongs --
      provider keys live in the deployment repository's ``.env``.

.. toctree::
   :hidden:

   profile
   config
   environment-variables
