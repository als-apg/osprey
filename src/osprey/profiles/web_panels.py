"""Built-in web panel registry — single source of truth across the build chain.

The web terminal (``interfaces/web_terminal/app.py``), the template manifest
validator (``cli/templates/manifest.py``), and the preset profile validator
(``cli/build_profile.py``) all gate on this set. Drift between them is what
let unknown panel IDs slip past for two registries simultaneously.
"""

from __future__ import annotations

UNIVERSAL_PANELS: set[str] = {"artifacts"}

BUILTIN_PANELS: set[str] = {
    "artifacts",
    "ariel",
    "tuning",
    "channel-finder",
    "lattice",
}

# Frontend fallback when a profile/config doesn't pin a default tab.
# The web terminal opens this tab first on cold load.
DEFAULT_PANEL_FALLBACK: str = "artifacts"
