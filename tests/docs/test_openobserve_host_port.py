"""No doc page may send a reader to the telemetry store on its container port.

OpenObserve listens on one fixed port inside its container. The port a reader
reaches it on from the host is a different number: the ``openobserve`` slot of
the port layout, ``deployment.port_base`` + 50, movable with
``services.openobserve.port``. The two coincided in no shipped configuration
since the layout landed, yet the docs kept telling readers to open
``localhost:<container port>``, because that literal was copied into prose
before the layout existed and nothing ever re-checked it.

The layout table is the one producer of host ports. A page that names the
container port as a host URL is a copy that has already drifted once, so this
sweep fails on any ``localhost`` or ``127.0.0.1`` URL carrying that port in
``docs/source``, including the inline SVG diagrams the pages embed. The
container port itself is still allowed to appear where it is correct: the
compose publish line, the bridge-networked ``OSPREY_OTEL_OPENOBSERVE_PORT``
declaration, and prose that says "the container's 5080" without a host name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.build.claude_code_telemetry import OPENOBSERVE_LISTEN_PORT

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = _REPO_ROOT / "docs" / "source"
_SUFFIXES = (".rst", ".md", ".html")

#: A host URL on the container port: ``localhost:5080`` or ``127.0.0.1:5080``,
#: optionally preceded by a scheme, not followed by another digit.
_HOST_URL_ON_CONTAINER_PORT = re.compile(
    rf"(?:https?://)?(?:localhost|127\.0\.0\.1):{OPENOBSERVE_LISTEN_PORT}(?!\d)"
)


def _doc_files() -> list[Path]:
    return sorted(
        path
        for path in _DOCS_ROOT.rglob("*")
        if path.is_file() and path.suffix in _SUFFIXES and "_build" not in path.parts
    )


@pytest.mark.parametrize("path", _doc_files(), ids=lambda p: str(p.relative_to(_DOCS_ROOT)))
def test_no_host_url_on_the_container_port(path: Path) -> None:
    offending = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _HOST_URL_ON_CONTAINER_PORT.search(line)
    ]
    assert not offending, (
        "a doc page names the telemetry store's container port as a host URL; the host "
        "port is the layout's openobserve slot (deployment.port_base + 50):\n"
        + "\n".join(offending)
    )
