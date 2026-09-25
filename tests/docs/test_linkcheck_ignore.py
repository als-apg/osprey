"""Gate: ``conf.py``'s ``linkcheck_ignore`` covers what no checker can reach, and
nothing else.

A broken link fails ``scripts/ci_check.sh``, which makes this list load-bearing in
two directions. Too narrow and a local run goes red over an address that answers
only on the reader's own machine; too wide and the check cannot fail at all, which
is the state a blocking check exists to replace. Both halves are asserted.

The list is read out of ``conf.py`` with :mod:`runpy`, patching the version source
the way ``test_redirects.py`` does: ``conf.py`` binds
``get_running_version``/``is_release`` at exec time, so the *module attributes* on
:mod:`osprey.version` are what it picks up.
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

import osprey.version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_PY = _REPO_ROOT / "docs" / "source" / "conf.py"

#: Addresses a checker cannot reach from where the docs are built: the reader's own
#: deployment, and pages that refuse an automated request.
UNREACHABLE = [
    "http://127.0.0.1:10100",
    "http://localhost:10100",
    "http://localhost:10050",
    "http://localhost:10010/dashboard",
    "http://localhost:10100/?token=abc123",
    "http://localhost",
    "https://claude.ai/code",
    "https://api.cborg.lbl.gov",
    "https://doi.org/10.1063/5.0306302",
]

#: Links a reader follows to a third party. Every one of these must stay checked,
#: or the step cannot report the thing it is there to report.
CHECKED = [
    "https://github.com/als-apg/osprey/issues",
    "https://als-apg.github.io/osprey/",
    "https://docs.astral.sh/uv/",
    "https://localhost.example.org/",
]


def _run_conf(monkeypatch) -> dict[str, Any]:
    """Execute ``conf.py`` with a pinned version source and return its namespace."""
    monkeypatch.setattr(osprey.version, "is_release", lambda: True)
    monkeypatch.setattr(osprey.version, "get_running_version", lambda: "9.9.9")
    monkeypatch.chdir(_CONF_PY.parent)

    saved_path = list(sys.path)
    try:
        return runpy.run_path(str(_CONF_PY))
    finally:
        sys.path[:] = saved_path


@pytest.fixture
def patterns(monkeypatch) -> list[re.Pattern[str]]:
    """``linkcheck_ignore`` as sphinx compiles and applies it."""
    value = _run_conf(monkeypatch)["linkcheck_ignore"]
    assert value, "conf.py must define `linkcheck_ignore` as a non-empty list of patterns"
    return [re.compile(pattern) for pattern in value]


@pytest.mark.parametrize("uri", UNREACHABLE)
def test_an_unreachable_address_is_not_link_checked(uri: str, patterns) -> None:
    """These have no answer to give a checker, so a red over one of them would be a
    red no contributor can clear."""
    assert any(pattern.match(uri) for pattern in patterns), (
        f"{uri} is not in linkcheck_ignore; a local CI run would go red over it"
    )


@pytest.mark.parametrize("uri", CHECKED)
def test_a_published_link_is_still_link_checked(uri: str, patterns) -> None:
    """The other direction, and the one that matters more: a pattern wide enough to
    cover a real link turns the step into a check that cannot fail."""
    assert not any(pattern.match(uri) for pattern in patterns), (
        f"{uri} is silenced by linkcheck_ignore; the link check can no longer report it"
    )
