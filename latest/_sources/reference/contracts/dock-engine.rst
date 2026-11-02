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

- **The built engine.** ``web_terminal/static/vendor/dock-engine.mjs`` and its
  types ``dock-engine.d.mts`` are byte copies of the runtime's ``dist/`` at one
  commit. ``vendor_manifest.json`` lists both by that source and no URL, so
  ``osprey vendor fetch`` never fetches them. The terminal serves the engine at
  ``/static/vendor/dock-engine.mjs`` beside dockview.
- **The server's layout rules.** ``osprey.interfaces.web_terminal.dock_layout``
  implements ``validate_layout``, ``layout_refusal``, ``normalize_layout`` and
  the reason vocabulary ``LAYOUT_REASONS`` in the standard library, so a preset,
  a layout report or an agent operation is refused for the reason the browser
  would give.
- **The engine's cases.** The runtime's ``validate`` and ``normalize`` cases are
  copied into ``tests/dock_engine/cases/``. ``tests/dock_engine/cases-pin.json``
  records the runtime commit, path and git blob of each copy and of the bundle.

``tests/dock_engine/`` replays the cases through ``dock_layout`` and fails when
a copy no longer matches its blob, when a copy is missing or extra, or when the
manifest's sha256 names other bytes than the vendored engine.

Changing the engine
===================

A rule changes in the runtime, with its cases. OSPREY then re-copies the bundle
and the two case directories from one runtime commit, re-pins them, updates the
two manifest entries, and changes ``dock_layout`` until the cases pass.
