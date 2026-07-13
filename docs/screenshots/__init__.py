"""Declarative documentation-screenshot capture framework.

Every committed screenshot under ``docs/source/_static/screenshots/`` is
regenerated from a single declarative recipe (:class:`~docs.screenshots.recipes.DocShot`)
so images cannot drift from what OSPREY actually ships. The registry in
:mod:`docs.screenshots.recipes` is the authoritative list of every doc image and
how it is produced; :mod:`docs.screenshots.capture` runs the recipes; the
``python -m docs.screenshots`` CLI selects and dispatches them.

The package is capture-only: it is never a CI gate (the tutorial stack needs a
container runtime and the agentic hero needs a live Claude session). Interface
render drift stays guarded by ``tests/interfaces/design_system/test_visual.py``.
"""
