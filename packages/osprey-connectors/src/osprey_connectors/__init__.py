"""Lean control-system and archiver connectors for OSPREY.

Installable without the OSPREY agent platform: no LLM, web, or agent
dependencies. Each public module of the package lists its API in its own
``__all__``, and the ``control_system``, ``archiver``, ``simulation`` and ``ipc``
subpackages export theirs from the subpackage itself; a module whose name starts
with an underscore is internal. Main osprey depends on this package, and some
modules still answer under a historical ``osprey.*`` path for code outside the
framework.
"""

__all__ = ["__version__"]

# The version rides the framework's calendar stream (see pyproject.toml): a
# built distribution answers from its own metadata, and a bare source tree —
# no dist installed, no build hook run — answers honestly that it is unbuilt.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("osprey-connectors")
except PackageNotFoundError:  # pragma: no cover — source tree without an install
    __version__ = "0.0.0+unbuilt"
