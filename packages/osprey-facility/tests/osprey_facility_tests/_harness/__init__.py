"""Framework-free helpers of the ``osprey-facility`` suite.

The build helper and the entry-point reset are re-exported for short imports.
``entry_points_read_afresh`` is a fixture: a test module requests it by name
(``tests/conftest.py`` makes it available to the whole suite). ``_guards``
holds the autouse guards and the session hook, and is imported only by the
conftest.

Nothing in this package imports ``osprey`` or the framework's ``tests``
package; ``test_package_never_imports_osprey.py`` holds it to that.
"""

from osprey_facility_tests._harness._builds import NO_EXTRA_CHECKS, build
from osprey_facility_tests._harness._entry_points import read_entry_points_afresh
from osprey_facility_tests._harness._paths import (
    PACKAGE_ROOT,
    PACKAGE_SRC,
    REPO_ROOT,
    TESTS_ROOT,
)

__all__ = [
    "NO_EXTRA_CHECKS",
    "PACKAGE_ROOT",
    "PACKAGE_SRC",
    "REPO_ROOT",
    "TESTS_ROOT",
    "build",
    "read_entry_points_afresh",
]
