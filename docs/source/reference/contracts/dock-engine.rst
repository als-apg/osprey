.. _reference-dock-engine:

====================
Dock Engine Contract
====================

``packages/dock-engine`` is the dock layout engine OSPREY and CANOPY share: the
layout model and its validation, geometry, drop planning, open/close/reopen
and hit-testing, as pure functions with no DOM and no runtime dependencies.
This page says what the package holds, how each consumer obtains it, and
where it is edited.

What the package is
===================

- **One case set.** ``cases/<kind>/<name>.json``, one directory per kind
  (``close``, ``drop``, ``hit``, ``normalize``, ``open``, ``open-set``,
  ``reopen``, ``replace-key``, ``validate``). Each case names the rule it pins.
  The cases are the specification; both implementations replay them.
- **One JS engine.** ``@als-apg/dock-engine``: ``dist/dock-engine.mjs``, one
  dependency-free ES module, types in ``dist/dock-engine.d.mts``.
  ``@als-apg/dock-engine/cases`` is the case runner (``loadCases``,
  ``runCase``, ``checkCase``, ``CASES_DIR``) that replays the cases against any
  implementation of the engine.
- **One Python rule implementation.** The module ``dock_engine`` (Python
  3.11+, standard library only) implements the layout rules --
  ``validate_layout``, ``layout_refusal``, ``normalize_layout`` and the reason
  vocabulary ``LAYOUT_REASONS``. The framework wheel ships it from
  ``packages/dock-engine``, with the case files and ``dist/manifest.json`` as
  package data; an import-linter contract keeps it from importing the
  framework back.

The Python tests replay the ``validate`` and ``normalize`` cases through
``dock_engine`` and check that its reason vocabulary and ``rules_digest()``
match the JS bundle's manifest, so the two implementations cannot drift
apart unnoticed.

How each consumer obtains it
============================

Each consumer is wired in a later stream; none of them copies the source.

- **OSPREY** -- ``import dock_engine``; the framework wheel carries it. The
  web terminal takes the JS engine as a generated vendor copy of
  ``dist/dock-engine.mjs``.
- **CANOPY screen runtime** -- a git dependency on ``@als-apg/dock-engine``.
- **CANOPY backend** -- the same directory installed from git, without the
  framework: ``osprey-dock-engine @ git+https://github.com/als-apg/osprey@<ref>#subdirectory=packages/dock-engine``
  (the directory's own ``pyproject.toml`` exists for this install alone). Never
  beside ``osprey-framework`` in one environment: the two distributions own the
  same ``dock_engine/`` files, so one overwrites the other and uninstalling
  either removes the module; ``dock_engine`` warns at import when it finds both.

Nothing is published to a registry: no PyPI project, no npm package. The
engine is too small for a distribution of its own, and the three consumers
are paired by ``rules_digest()`` -- the hash over the reason vocabulary, the
``normalize``/``validate`` cases and ``layout.ts`` -- rather than by a version
floor. Versions follow OSPREY's ``v*`` CalVer tags wherever metadata exists
(the framework wheel, or the git install); ``0.0.0-dev`` in ``package.json``
marks an untagged checkout.

The cutover
===========

The engine was cut over on 2026-10-24 from the CANOPY screen runtime
(``als-apg/canopy-osprey-mirror-screen-runtime`` at ``386c66cb``,
``apps/viewer/src/workspace/dock/model/``). At the cutover every refusing
``validate`` case gained a ``where`` field. ``CUTOVER.json`` records the source
digest, the blob of each imported file, the 42 exported names and the case
counts. From the cutover on, the package is the only place the engine changes.

Source of truth
===============

``ts/`` and ``cases/`` are edited in the package and rebuilt with
``node packages/dock-engine/scripts/build.mjs``; ``dist/``, vendor copies and
consumer installs are generated, never edited.

``build.mjs --check`` exits non-zero when ``dist/`` is stale; the local gate
and CI run it, together with ``npm run test:dock-engine`` and
``pytest tests/dock_engine/``.

.. seealso::

   :doc:`/contributing/workflow`
      How workspace packages are versioned and released.
