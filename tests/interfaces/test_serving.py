"""Contract test for :mod:`osprey.interfaces._serving`.

Pins ``listening_socket`` as the one place an OSPREY server binds and listens,
and ``authorize_browser_context`` to mint an *ephemeral* (unpersisted)
session: the module's whole reason for existing is authorizing a browser
against a server this same process just spun up, so a session that leaked
into a persisted store would go on outliving the throwaway server it was
minted for.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

import osprey
from osprey.interfaces import _serving
from osprey.interfaces._serving import authorize_browser_context, listening_socket
from osprey.interfaces.common_middleware import session_cookie_name
from osprey.interfaces.web_auth import _digest, get_web_credentials


class _FakeContext:
    """A stand-in exposing only the ``add_cookies`` method the helper calls."""

    def __init__(self) -> None:
        self.cookies: list[dict[str, str]] = []

    def add_cookies(self, cookies: list[dict[str, str]]) -> None:
        self.cookies.extend(cookies)


def test_authorize_browser_context_mints_an_ephemeral_session():
    context = _FakeContext()

    session_id = authorize_browser_context(context)

    assert _digest(session_id) in get_web_credentials()._ephemeral
    assert context.cookies == [
        {
            "name": session_cookie_name(),
            "value": session_id,
            "domain": "127.0.0.1",
            "path": "/",
        }
    ]
    assert get_web_credentials().verify_session(session_id) is True


def test_port_zero_listens_on_the_port_the_socket_reports():
    with listening_socket("127.0.0.1", 0) as sock:
        port = sock.getsockname()[1]
        assert port != 0
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass


def test_a_held_port_is_refused():
    with listening_socket("127.0.0.1", 0) as held:
        with pytest.raises(OSError):
            listening_socket("127.0.0.1", held.getsockname()[1])


class _ListenCallers(ast.NodeVisitor):
    """The innermost function around each ``.listen(...)`` call in one module."""

    def __init__(self) -> None:
        self.scope = ["<module>"]
        self.found: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "listen":
            self.found.add(self.scope[-1])
        self.generic_visit(node)


def _listen_callers(root: Path) -> set[str]:
    """``<module path>:<function>`` for every ``.listen(...)`` call under *root*."""
    callers: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".listen(" not in source:
            continue
        visitor = _ListenCallers()
        visitor.visit(ast.parse(source))
        callers |= {f"{path.relative_to(root).as_posix()}:{name}" for name in visitor.found}
    return callers


def test_only_the_helper_listens():
    root = Path(osprey.__file__).parent
    helper = Path(_serving.__file__).relative_to(root).as_posix()
    assert _listen_callers(root) == {f"{helper}:listening_socket"}
