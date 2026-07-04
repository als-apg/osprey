"""pytest options local to the design-system test package.

``--regen-baselines`` belongs here (rather than the root ``tests/conftest.py``)
because it is scoped to ``test_visual.py``'s Playwright screenshot baselines —
``pytest_addoption`` must live in a ``conftest.py`` to be seen during
command-line parsing, a plain test module is imported too late.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--regen-baselines",
        action="store_true",
        default=False,
        help=(
            "Overwrite tests/interfaces/design_system/baselines/*.png with freshly "
            "captured screenshots instead of comparing against them. The committed "
            "baselines are Linux-rendered (CI runs this flag on ubuntu-latest and "
            "commits the result) — running this locally on non-Linux produces "
            "baselines that are NOT authoritative for CI."
        ),
    )
