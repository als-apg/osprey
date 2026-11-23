"""Where this suite sits in the checkout.

Spelled once, so a test that reads the panel fixture, the font-metrics
generator under ``tools/`` or a file at the repository root does not count
directory levels itself.
"""

from pathlib import Path

#: ``packages/osprey-panels``.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]

#: ``packages/osprey-panels/tests``: the directory a child process needs on
#: ``PYTHONPATH`` to import ``osprey_panels_tests`` and its ``_harness``.
TESTS_ROOT = PACKAGE_ROOT / "tests"

#: The source root a child process needs on ``PYTHONPATH`` to import
#: ``osprey_panels``.
PACKAGE_SRC = PACKAGE_ROOT / "src"

#: The repository checkout the package lives in.
REPO_ROOT = PACKAGE_ROOT.parents[1]

#: The panels fixture: the checked-in ``facility.json`` and its ``panels/``
#: rule tables.
FIXTURE_DIR = TESTS_ROOT / "osprey_panels_tests" / "fixture"
