Lattice Dashboard
=================

The **LATTICE** tab draws the optics of the physics models your build's
simulator serves, and lets you — or the agent — try a magnet change on a model
and see what it does before anything is written to the machine. It reads the
models and their decks from the build, so there is nothing to load by hand.

Turn it on
----------

List the panel in ``config.yml``:

.. code-block:: yaml

   web:
     panels:
       lattice: true

That line is the whole gesture: the tab launches with ``osprey web`` and needs
no section of its own. A ``lattice_dashboard:`` section (``host``, ``port``,
``auto_launch``) only moves where it listens. The ``control-assistant`` preset
ships the panel in its ``web_panels:`` list.

Pick a model
------------

The selector in the header lists every model of the build that has a deck,
served models first. A model the simulator does not serve is marked
``not served``; that is a label only, and its figures are drawn like any
other's. A build with no simulator view, or one that serves no model with a
deck, says so in a banner.

When you open the dashboard, or after ``osprey build``, it shows the model you
picked; with no pick it shows the model your what-if was on, if the build
still lists it, otherwise the first served model; and a picked model the build
dropped leaves the dashboard with no model until you pick one.

Your **what-if** is the set of inputs you have changed on the selected model:
the magnet overrides and the baseline the figures are compared against. It
belongs to one model and deck, so picking another model starts from that
model's unmodified deck. The dashboard's settings are kept.

What a model draws
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Solve
     - Figures
   * - ``periodic``
     - ``optics``, ``resonance``, ``chromaticity`` and ``footprint``, the fast
       figures, and ``da`` and ``lma``, the two long ones.
   * - ``single_pass``
     - ``optics`` only. A single-pass model has no tune, so every other panel
       reads ``not available for a single-pass model``.

When a figure is current
------------------------

A figure is shown only for the inputs on screen: the deck, the overrides, the
baseline and the settings. Each figure carries one of five statuses:

- ``ready`` — computed for the inputs on screen, and shown.
- ``computing`` — being computed for the inputs on screen.
- ``failed`` — the last computation for the inputs on screen failed; the panel
  shows why.
- ``stale`` — computed for other inputs of the same deck. The first override
  on a deck turns its figures ``stale`` until they are recomputed.
- ``not computed`` — nothing computed for this deck yet. Switching to another
  deck answers ``not computed`` until its figures are recomputed, and the
  earlier deck's figure comes back when you switch back.

**Refresh** recomputes the fast figures and **Verify** the two long ones.

From the agent
--------------

The agent drives the same dashboard through the workspace server's lattice
tools, listed in :doc:`/architecture/mcp-servers`. It is told a figure's status
the same way: a figure that is not current for the inputs on screen is
refused, naming the status, and never answered with an older one.
