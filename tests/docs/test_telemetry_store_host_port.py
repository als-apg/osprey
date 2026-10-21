"""No doc page may send a reader to the telemetry store on its container port.

The store answers HTTP on one fixed port inside its container. The port a
reader reaches it on from the host is a different number: the
``telemetry_store`` slot of the port layout, ``deployment.port_base`` + 804,
movable with ``services.telemetry_store.port``. The two do not coincide at the
default port base, so a ``localhost:<container port>`` copied into prose sends
the reader to a port nothing on the host listens on.

The layout table is the one producer of host ports, so this sweep fails on any
``localhost`` or ``127.0.0.1`` URL carrying the container port in
``docs/source``, including the inline SVG diagrams the pages embed. The
container port itself may still appear where it is correct: the compose
network address ``telemetry_store:<port>``, and prose that names the
container's port without a host name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.deployment.telemetry_config import STORE_HTTP_CONTAINER_PORT
from osprey.port_layout import default_port

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = _REPO_ROOT / "docs" / "source"
_SUFFIXES = (".rst", ".md", ".html")

#: A host URL on the container port: ``localhost:8123`` or ``127.0.0.1:8123``,
#: optionally preceded by a scheme, not followed by another digit.
_HOST_URL_ON_CONTAINER_PORT = re.compile(
    rf"(?:https?://)?(?:localhost|127\.0\.0\.1):{STORE_HTTP_CONTAINER_PORT}(?!\d)"
)


def _doc_files() -> list[Path]:
    return sorted(
        path
        for path in _DOCS_ROOT.rglob("*")
        if path.is_file() and path.suffix in _SUFFIXES and "_build" not in path.parts
    )


def test_the_host_port_is_not_the_container_port() -> None:
    """The premise of the sweep: the layout never hands out the container port."""
    assert default_port("telemetry_store") != STORE_HTTP_CONTAINER_PORT


def test_the_pattern_finds_a_host_url_on_the_container_port() -> None:
    port = STORE_HTTP_CONTAINER_PORT
    for line in (f"open http://localhost:{port}/play", f"curl 127.0.0.1:{port}"):
        assert _HOST_URL_ON_CONTAINER_PORT.search(line), line
    for line in (f"telemetry_store:{port}", f"localhost:{port}0", "localhost:10804"):
        assert not _HOST_URL_ON_CONTAINER_PORT.search(line), line


@pytest.mark.parametrize("path", _doc_files(), ids=lambda p: str(p.relative_to(_DOCS_ROOT)))
def test_no_host_url_on_the_container_port(path: Path) -> None:
    offending = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _HOST_URL_ON_CONTAINER_PORT.search(line)
    ]
    assert not offending, (
        "a doc page names the telemetry store's container port as a host URL; the host "
        "port is the layout's telemetry_store slot (deployment.port_base + 804):\n"
        + "\n".join(offending)
    )
