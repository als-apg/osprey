.. _reference-dock-engine:

====================
Dock Engine Contract
====================

The web terminal's dock runs CANOPY's dock engine, which lives in the CANOPY
screen runtime (``packages/dock-engine``): the layout model and its
validation, geometry, drop planning, open/close/reopen and hit-testing, as pure
functions with no DOM and no runtime dependencies. OSPREY does not edit it.

What OSPREY holds
=================

OSPREY holds three things from the runtime. Each is pinned to one runtime
commit and checked by a test under ``tests/dock_engine/``:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - What
     - Pinned by
     - Checked by
   * - **The built engine**: ``dock-engine.mjs`` and its types
       ``dock-engine.d.mts``, in
       ``src/osprey/interfaces/web_terminal/static/vendor/``
     - The runtime commit, path and git blob of each, in
       ``tests/dock_engine/cases-pin.json``; the sha256 and source commit of
       each, with no URL, in ``src/osprey/interfaces/vendor_manifest.json``
     - ``test_cases_pin.py``: each copy matches its blob, and the manifest's
       sha256 names the vendored bytes
   * - **The engine's cases**: the runtime's ``validate`` and ``normalize``
       cases, in ``tests/dock_engine/cases/``
     - The runtime commit, path and git blob of each copy, in
       ``cases-pin.json``
     - ``test_cases_pin.py``: no copy changed, missing or extra
   * - **The server's layout rules**:
       ``osprey.interfaces.web_terminal.dock_layout`` --- ``validate_layout``,
       ``layout_refusal``, ``normalize_layout`` and the reason vocabulary
       ``LAYOUT_REASONS``
     - The engine's cases
     - ``test_case_replay.py`` replays every case through ``dock_layout``;
       ``test_cases_pin.py`` checks that the bundle's reasons are
       ``LAYOUT_REASONS``, in order

The engine files are byte copies of the runtime's ``dist/``. Their manifest
entries have no URL, so ``osprey vendor fetch`` never fetches them: a missing
or changed copy is an error that names the commit to re-copy from. The terminal
serves the engine at ``/static/vendor/dock-engine.mjs`` beside dockview.

The server's rules use the standard library only, so a preset, a layout report
or an agent operation is refused for the reason the browser would give.

OSPREY's policy
===============

The engine's layout grammar lets a tile hold several panels (a tile's ``tabs``
field is a list). OSPREY holds one panel per tile, as a policy of its own
above the engine. It lives in three places: the browser's ``dock-policy.js``
and ``dock-open.js``, the layout holder ``dock-apply.js``, and the server's
``presets.crowded_tile``, which the agent operation routes, the layout report
and the profile build all use. Each refuses a layout with a crowded tile for
the one reason ``one_panel_per_tile``. The vendored engine and its pinned
cases are unchanged: a tile's ``tabs`` is always one entry.

Changing the engine
===================

A rule changes in the runtime first, with its cases. Then, from one runtime
commit, re-copy the bundle and the two case directories, re-pin them in
``cases-pin.json``, update the two manifest entries, and change
``dock_layout`` until the cases pass.
