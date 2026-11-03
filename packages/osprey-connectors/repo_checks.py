"""The connectors package's checks: the CI lanes that install it on its own.

``scripts/check.py`` collects this module after the framework's. The package
releases on its own tags and is installed without the framework, so the lanes
that prove that are declared here, beside the package, not in the framework's
module. A check of the package's own joins ``CHECKS`` here.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

from check_kit import Check, Lane

CHECKS: tuple[Check, ...] = ()

LANES: tuple[Lane, ...] = (
    Lane(
        "connectors-standalone",
        "installs the connectors alone",
        ("packages/osprey-connectors/**", "pyproject.toml", "uv.lock"),
    ),
    Lane(
        "connectors-compat-plan",
        "reads the release tags",
        ("scripts/ci/connectors_compat_matrix.py", "pyproject.toml"),
    ),
    Lane(
        "connectors-compat",
        "installs released frameworks",
        (
            "packages/osprey-connectors/**",
            "scripts/ci/connectors_compat_lane.py",
            "scripts/ci/connectors_compat_matrix.py",
        ),
    ),
)
