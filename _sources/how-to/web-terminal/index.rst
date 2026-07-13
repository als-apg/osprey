Web Terminal
============

The OSPREY Web Terminal is a browser-based cockpit for the Osprey agent: a real
terminal on the left, your live workspace on the right, and themed side panels
for your control-system tools. Launch it from any project directory:

.. code-block:: bash

   osprey web

It opens in your browser at ``http://127.0.0.1:8087``. From there you chat with
the agent, watch files and plots appear the moment they are created, and switch
between companion tools without ever leaving the page.

.. figure:: /_static/screenshots/web_terminal_hero_light.png
   :alt: The OSPREY Web Terminal — a live agent session on the left, the workspace with a beam-current plot on the right
   :align: center
   :width: 100%

   A live agent session on the left; the workspace and its artifacts on the right.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Run it
      :link: operate
      :link-type: doc
      :shadow: md

      Launch it, run it in the background, and change the handful of settings
      that matter.

   .. grid-item-card:: Theming
      :link: theming
      :link-type: doc
      :shadow: md

      Pick a light or dark theme for every OSPREY interface at once — or design
      your own.

   .. grid-item-card:: Panels
      :link: panels
      :link-type: doc
      :shadow: md

      Add your own tools as themed side panels that sit beside the chat.

.. toctree::
   :hidden:

   operate
   theming
   panels
