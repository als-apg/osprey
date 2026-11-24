"""Framework-free helpers of the ``osprey-panels`` suite.

The checkout paths are re-exported for short imports. ``_guards`` holds the
autouse guards and the session hook, and is imported only by the conftest.

Nothing in this package imports ``osprey`` or the framework's ``tests``
package.
"""

from osprey_panels_tests._harness._paths import (
    FIXTURE_DIR,
    PACKAGE_ROOT,
    PACKAGE_SRC,
    REPO_ROOT,
    TESTS_ROOT,
)

__all__ = [
    "FIXTURE_DIR",
    "PACKAGE_ROOT",
    "PACKAGE_SRC",
    "REPO_ROOT",
    "TESTS_ROOT",
]
