Panels
======

A **panel** is a self-contained, themed mini-app the Web Terminal shows as a tab
beside the chat. OSPREY's own tools — Channel Finder, ARIEL, the lattice
dashboard, the artifact gallery — are all panels, and you can add your own.
Because panels use the shared design tokens (:doc:`theming`), they match your
theme automatically, in light and dark, with no extra work.

Enable OSPREY's built-in panels in ``config.yml``:

.. code-block:: yaml

   web:
     panels:
       ariel: true
       channel-finder: true
       lattice: true

Adding your own panel
---------------------

The easy path is the guided skill — a coding agent follows it to produce a
panel that already meets every rule:

.. code-block:: bash

   osprey skills install creating-an-osprey-panel

Once the panel is written and validated, drop its folder under your project's
``panels/`` directory and turn on discovery:

.. code-block:: yaml

   web:
     allow_runtime_panels: true    # off by default

On the next start, every valid panel under ``panels/`` shows up as a tab. Invalid
ones are skipped and logged, so one bad panel never breaks the others.

.. warning::

   The Web Terminal has **no application-level login**. Turning on
   ``allow_runtime_panels`` serves whatever panels are on disk to anyone who can
   reach the port — right for the intended single-operator, local setup, but a
   facility that exposes the terminal more widely should put its own
   authentication in front of it.

Panel layouts ("presets")
-------------------------

A **layout** is a named set of panels an operator applies in one click — "the
machine-setup view is these four panels." Define them under ``web.presets``, where
each key is the menu label and each value is the exact set of panels to show:

.. code-block:: yaml

   web:
     presets:
       "Machine setup": [channel-finder, lattice, artifacts, okf]
       "Logbook review": [ariel, artifacts]

Each layout appears under a **Layouts** section at the top of the panel ``+``
menu. Picking one is *exclusive*: its panels open and every other panel closes.
Members must be enabled built-in panels or custom panels you have declared; an
unknown id is dropped with a warning, and a layout with no valid members is
skipped. When no ``web.presets`` are configured — the default — the ``+`` menu is
unchanged, so layouts never add clutter to a deployment that has not opted in.

Layouts are just a shortcut over the ordinary show/hide of panels, so the OSPREY
agent can achieve the same result with its ``show_panel`` / ``hide_panel`` tools;
``list_panels`` reports the configured layouts so the agent can honor a request
like "set up for machine setup."

.. dropdown:: Going deeper — how panels work
   :icon: package

   You only need this to write a panel by hand or wire a new one into the hub;
   the ``creating-an-osprey-panel`` skill and the source (which the panel
   validator and its browser test suite back) are the real reference. The rough
   idea:

   .. tab-set::

      .. tab-item:: Authoring rules

         A panel is one HTML entry point plus a ``manifest.json`` (its id, label,
         and entry file). Two rules make it theme itself for free, and a
         validator enforces both: it must **boot the shared theme before the page
         paints** (so it never flashes the wrong colors), and it must use **only
         design tokens** for color — never a raw hex value. Copy OSPREY's
         reference panel and you start compliant.

      .. tab-item:: Serving & discovery

         Discovered panels are served straight from disk, from the same origin as
         the terminal — no proxy. Discovery is **fail-closed**: only a fully valid
         panel is ever served. The ``allow_runtime_panels`` switch is deliberately
         off by default because it is also what lets the agent register panels at
         runtime — turning it on is your explicit decision to trust the panels the
         terminal makes available.

      .. tab-item:: Embedding contract

         The hub and its panels coordinate over a small shared contract — enough
         for a panel to pick up the current theme and to hide its own logo when
         embedded, so it sits flush inside the hub instead of showing two titles.
         The browser test suite is the exact, up-to-date spec; this is only the
         shape of it.

.. seealso::

   :doc:`theming`
      The design tokens panels style against.

   :doc:`operate`
      Running the terminal that hosts them.
