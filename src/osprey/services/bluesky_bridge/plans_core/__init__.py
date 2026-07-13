"""In-image core plan directory: the ``shipped``-tier layer of the layered
directory catalog (see ``plan_loader.py``). Every ``.py`` file directly under
this package is scanned for the ``PLAN_METADATA``/``build_plan``/``PARAMS``
contract; this ``__init__.py`` only makes the directory a proper package (the
loader itself skips dunder-named files during its scan).
"""

from __future__ import annotations
