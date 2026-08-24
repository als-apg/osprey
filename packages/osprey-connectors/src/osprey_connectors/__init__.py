"""Lean control-system and archiver connectors for OSPREY.

Installable without the OSPREY agent platform: no LLM, web, or agent
dependencies. Main osprey depends on this package and re-exports every
module under its historical ``osprey.*`` path.
"""

# The version rides the framework's calendar stream (see pyproject.toml): a
# built distribution answers from its own metadata, and a bare source tree —
# no dist installed, no build hook run — answers honestly that it is unbuilt.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("osprey-connectors")
except PackageNotFoundError:  # pragma: no cover — source tree without an install
    __version__ = "0.0.0+unbuilt"
