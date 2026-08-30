"""Every root-absolute API path the lattice dashboard's JS names must be a quoted literal.

The web terminal serves the dashboard through its panel proxy at
``/u/<user>/panel/lattice/``, and the proxy rewrites ``/api/...`` only when a
quote or backtick immediately precedes it (``routes/proxy.py::_rewrite_content``).
A path assembled with a template literal (``\\`${API_BASE}/api/events\\```) puts
a ``}`` there instead, escapes the rewrite, and resolves at the origin root —
which is how the SSE stream 404ed forever behind the front door (#784).
This pins the contract at the source so the next such URL fails here, not in
a deployment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC_JS = Path(__file__).resolve().parents[3] / (
    "src/osprey/interfaces/lattice_dashboard/static/js"
)

#: A ``/api`` path segment whose preceding character is neither a quote nor a
#: backtick — the one shape the proxy cannot rewrite. Comment lines are
#: skipped by the scan below, not by this pattern.
_UNQUOTED_API = re.compile(r"""(?<!["'`])/api(?=[/"'`])""")


def _code_lines(text: str):
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(("//", "*", "/*")):
            continue
        yield number, line


@pytest.mark.parametrize("js_file", sorted(_STATIC_JS.glob("*.js")), ids=lambda p: p.name)
def test_api_paths_are_quoted_literals(js_file: Path) -> None:
    offenders = [
        f"{js_file.name}:{number}: {line.strip()}"
        for number, line in _code_lines(js_file.read_text(encoding="utf-8"))
        if _UNQUOTED_API.search(line)
    ]
    assert not offenders, (
        "root-absolute /api path not spelled as a quoted literal — the panel proxy "
        "cannot rewrite it under /u/<user>/panel/lattice/:\n  " + "\n  ".join(offenders)
    )
