"""Ratchet: no test picks a port by binding port 0 and releasing it.

A port taken from ``bind(0)`` and closed again lies in the kernel's ephemeral
range and is free for any ``connect()`` or ``bind(0)`` on the host until the
server the test starts binds it. A test takes its port from
``osprey_connectors_tests._harness._ports`` instead: a server it controls binds port 0 itself and reports the number, a
free port that is held is a ``reserved_port()`` / ``reserved_ports()``, and an
occupied port is a ``bound_socket()``.

A site is a ``.bind((<host>, 0))`` call, whatever expression the host is. The
walk covers both suites, ``tests/`` and the connectors suite. ``REMAINING_SITES``
lists the files that still hold one, relative to the repository root. A file outside it fails the ratchet, and
so does a listed file that no longer holds one, so the list only shrinks.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The trees walked, relative to the repository root: the framework's suite and
#: the connectors suite.
SUITES = ("tests", "packages/osprey-connectors/tests")

#: Files the ratchet never reads, relative to the repository root: the
#: reservation's own probes bind explicit ports.
EXEMPT = frozenset({"packages/osprey-connectors/tests/osprey_connectors_tests/_harness/_ports.py"})

#: Files that still pick a released port, relative to the repository root, sorted.
REMAINING_SITES: tuple[str, ...] = (
    "tests/e2e/_ca_client_name_sniffer.py",
    "tests/e2e/_orm_stack.py",
    "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py",
    "tests/e2e/fixtures/gchat_pubsub_emulator.py",
    "tests/e2e/test_dispatch_tutorial.py",
    "tests/e2e/test_gchat_bridge_e2e.py",
    "tests/e2e/test_nextcloud_talk_bridge_e2e.py",
    "tests/e2e/test_teams_bridge_e2e.py",
    "tests/e2e/test_va_substrate_equivalence.py",
    "tests/e2e/web_terminals/test_loopback_chokepoint.py",
    "tests/e2e/web_terminals/test_terminal_auth_e2e.py",
    "tests/va/e2e/conftest.py",
    "tests/va/e2e/test_bluesky_lanes.py",
    "tests/va/e2e/test_live_standin.py",
    "tests/va/e2e/test_model_status_va.py",
    "tests/va/e2e/test_per_target_posture.py",
    "tests/va/e2e/test_planted_journal.py",
    "tests/va/e2e/test_substrate_parity.py",
    "tests/va/e2e/test_target_switch.py",
    "tests/va/e2e/test_va_put_surface.py",
)

_TOKEN = re.compile(r"\.bind\b")


def _binds_port_zero(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "bind"):
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Tuple):
        return False
    elts = node.args[0].elts
    if len(elts) != 2:
        return False
    port = elts[1]
    return isinstance(port, ast.Constant) and port.value == 0


def has_released_port(source: str) -> bool:
    """Whether a module picks a port by binding port 0 and releasing it."""
    if not _TOKEN.search(source):
        return False
    return any(
        isinstance(node, ast.Call) and _binds_port_zero(node)
        for node in ast.walk(ast.parse(source))
    )


def released_port_sites(anchor: Path) -> set[str]:
    """Files of the suites under *anchor*, relative to it and POSIX, that pick a released port."""
    return {
        path.relative_to(anchor).as_posix()
        for suite in SUITES
        for path in (anchor / suite).rglob("*.py")
        if "__pycache__" not in path.parts
        and path.relative_to(anchor).as_posix() not in EXEMPT
        and has_released_port(path.read_text(encoding="utf-8"))
    }


def ratchet_failures(found: set[str], remaining: tuple[str, ...]) -> list[str]:
    """What the found sites break of the ratchet: new files and stale entries."""
    listed = set(remaining)
    failures = [
        f"{path}: picks a port by binding 0 and releasing it; take the port from "
        "osprey_connectors_tests._harness._ports, which names each remedy"
        for path in sorted(found - listed)
    ]
    failures += [
        f"{path}: no released port left; delete it from REMAINING_SITES"
        for path in sorted(listed - found)
    ]
    if list(remaining) != sorted(listed) or len(remaining) != len(listed):
        failures.append("REMAINING_SITES is not a sorted list of distinct paths")
    return failures


def test_no_new_released_port_sites():
    assert ratchet_failures(released_port_sites(REPO_ROOT), REMAINING_SITES) == []


@pytest.mark.parametrize(
    "source",
    [
        'import socket\nwith socket.socket() as s:\n    s.bind(("127.0.0.1", 0))\n',
        "def pick(sock, host):\n    sock.bind((host, 0))\n",
        "def pick(self):\n    self.sock.bind((self.host, 0))\n",
        "def pick(sock, hosts):\n    sock.bind((hosts[0], 0))\n",
        "def pick(sock, host):\n    sock.bind((host(), 0))\n",
    ],
    ids=[
        "str-host",
        "name-host",
        "attribute-host",
        "subscript-host",
        "call-host",
    ],
)
def test_every_site_shape_is_found(source):
    assert has_released_port(source)


@pytest.mark.parametrize(
    "source",
    [
        'def serve(sock, port):\n    sock.bind(("127.0.0.1", port))\n',
        'def serve(sock):\n    sock.bind(("127.0.0.1", 8080))\n',
        "SOURCE = 's.bind((\"127.0.0.1\", 0))'\n",
    ],
    ids=["port-variable", "explicit-port", "string"],
)
def test_a_held_or_explicit_port_is_no_site(source):
    assert not has_released_port(source)


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for suite in SUITES:
        (root / suite).mkdir(parents=True)
    for relative in (*REMAINING_SITES, *EXEMPT):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return root


def test_a_planted_site_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    planted = root / "packages/osprey-connectors/tests/planted/test_planted.py"
    planted.parent.mkdir()
    planted.write_text('import socket\nsocket.socket().bind(("127.0.0.1", 0))\n')
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert len(failures) == 1
    assert failures[0].startswith("packages/osprey-connectors/tests/planted/test_planted.py:")


def test_a_listed_file_without_its_site_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    converted = REMAINING_SITES[0]
    (root / converted).write_text('"""Converted."""\n')
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert failures == [f"{converted}: no released port left; delete it from REMAINING_SITES"]


def test_the_exempt_reservation_module_is_never_a_site(tmp_path):
    for suite in SUITES:
        (tmp_path / suite).mkdir(parents=True)
    (exempt,) = EXEMPT
    (tmp_path / exempt).parent.mkdir(parents=True)
    (tmp_path / exempt).write_text('import socket\nsocket.socket().bind(("127.0.0.1", 0))\n')
    assert released_port_sites(tmp_path) == set()
