"""Where this suite sits in the checkout.

Spelled once, so a test that hands a subprocess the package's source tree or
reads a file at the repository root does not count directory levels itself.
"""

from pathlib import Path

#: ``packages/osprey-connectors``.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

#: The source root a child process needs on ``PYTHONPATH`` to import
#: ``osprey_connectors``. The framework's ``src`` is deliberately not beside it:
#: a child that needed the framework would then pass here and fail on a host
#: that has only this distribution.
PACKAGE_SRC = PACKAGE_ROOT / "src"

#: The repository checkout the package lives in.
REPO_ROOT = PACKAGE_ROOT.parents[1]
