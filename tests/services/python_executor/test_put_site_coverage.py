"""Every raw Channel Access put in the pinned client sources is refused when armed.

The armed partition (:data:`_ARMED_BLOCKED`) names Python entry points, while a
channel write finally leaves the process through one of three C calls:
``ca_array_put``, ``ca_array_put_callback`` and ``ca_sg_array_put``. This test
reads the installed, lock-pinned pyepics and aioca sources with :mod:`ast`,
lists every call of those three symbols, and asserts that the public function
enclosing each call is refused by an ``_ARMED_BLOCKED`` row -- either directly,
or because every caller of a private helper holding the call is itself covered.

A site that nothing covers fails the test with its file and line, so a client
upgrade that adds a new put path cannot pass unnoticed.

The ``libca`` handle pyepics loads carries the same three symbols as attributes.
A run that reaches ``epics.ca.libca`` directly calls C without going through
any Python entry point; those attributes are listed below as documented
cooperative bypasses rather than rows of the partition.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey.runtime.raw_put_block import _HANDLE_PUT_SYMBOLS
from osprey.services.python_executor.write_surface import _ARMED_BLOCKED

#: The C entry points a Channel Access write leaves the process through.
PUT_SYMBOLS: frozenset[str] = frozenset(
    {"ca_array_put", "ca_array_put_callback", "ca_sg_array_put"}
)

#: Installed sources scanned: (top-level package, path inside the package).
SCANNED_SOURCES: tuple[tuple[str, str], ...] = (
    ("epics", "ca.py"),
    ("epics", "pv.py"),
    ("epics", "__init__.py"),
    ("aioca", "_catools.py"),
)

#: Receivers a put site may call through. ``libca`` is pyepics' module-global
#: handle to the loaded shared library; ``cadef`` is the ctypes binding aioca
#: imports from ``epicscorelibs.ca``.
_LIBCA_HANDLE = "epics.ca.libca"
_CADEF_MODULE = "epicscorelibs.ca.cadef"

#: The attributes of the pyepics ``libca`` handle that write a channel. Code
#: that calls them directly bypasses every Python entry point; the armed
#: partition does not list them because the connector-routed path never
#: touches them, and a caller reaching for the raw handle is outside the
#: cooperative contract the partition enforces.
COOPERATIVE_BYPASSES: dict[tuple[str, str], str] = {
    (_LIBCA_HANDLE, "ca_array_put"): (
        "raw handle attribute; pyepics reaches it only from ca.put, which is refused"
    ),
    (_LIBCA_HANDLE, "ca_array_put_callback"): (
        "raw handle attribute; pyepics reaches it only from ca.put, which is refused"
    ),
    (_LIBCA_HANDLE, "ca_sg_array_put"): (
        "raw handle attribute; pyepics reaches it only from ca.sg_put, which is refused"
    ),
}


@dataclass(frozen=True)
class PutSite:
    """One call of a put symbol in a scanned source."""

    file: str
    line: int
    symbol: str
    receiver: str | None
    #: (dotted owner, name) of the outermost function holding the call, or
    #: ``None`` when the call runs at module or class-body level.
    enclosing: tuple[str, str] | None

    def where(self) -> str:
        owner = ".".join(self.enclosing) if self.enclosing else "<module level>"
        return f"{self.file}:{self.line}: {self.symbol} in {owner}"


def _receiver_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Scanner(ast.NodeVisitor):
    """Collects put sites and a module-local call graph keyed by function."""

    def __init__(self, module: str, file: str) -> None:
        self.module = module
        self.file = file
        self.sites: list[PutSite] = []
        #: function key -> names it calls anywhere in its body (nested defs included)
        self.calls: dict[tuple[str, str], set[str]] = {}
        self._class_stack: list[str] = []
        self._function: tuple[str, str] | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._function is not None:
            self.generic_visit(node)
            return
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._function is not None:
            # A nested def runs only when its outer function runs (or hands it
            # out), so its calls are attributed to the outermost function.
            self.generic_visit(node)
            return
        owner = ".".join([self.module, *self._class_stack])
        self._function = (owner, node.name)
        self.calls.setdefault(self._function, set())
        self.generic_visit(node)
        self._function = None

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        name = _called_name(node.func)
        if name is not None and self._function is not None:
            self.calls[self._function].add(name)
        if name in PUT_SYMBOLS:
            self.sites.append(
                PutSite(
                    file=self.file,
                    line=node.lineno,
                    symbol=name,
                    receiver=_receiver_name(node.func),
                    enclosing=self._function,
                )
            )
        self.generic_visit(node)


def scan_source(source: str, module: str, file: str) -> _Scanner:
    scanner = _Scanner(module, file)
    scanner.visit(ast.parse(source, filename=file))
    return scanner


def is_covered(
    key: tuple[str, str] | None,
    calls: dict[tuple[str, str], set[str]],
    blocked: dict[tuple[str, str], str] | set[tuple[str, str]],
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    """True when every public route to ``key`` passes an armed-blocked row.

    A function is covered when its ``(owner, name)`` is blocked. A private
    function (leading underscore) is also covered when it has at least one
    caller in the same module and every such caller is covered: the put it
    holds then runs only behind a blocked choke point. A public function that
    is not blocked is reachable by name and so is never covered.
    """
    if key is None:
        return False
    if key in blocked:
        return True
    owner, name = key
    if not name.startswith("_") or key in _seen:
        return False
    callers = [caller for caller, names in calls.items() if name in names and caller != key]
    if not callers:
        return False
    seen = _seen | {key}
    return all(is_covered(caller, calls, blocked, seen) for caller in callers)


def _package_dir(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations, (
        f"{package} is a lock-pinned dependency and must be installed for this scan"
    )
    return Path(next(iter(spec.submodule_search_locations)))


def _module_name(package: str, relpath: str) -> str:
    stem = relpath.removesuffix(".py").replace("/", ".")
    return package if stem == "__init__" else f"{package}.{stem}"


@pytest.fixture(scope="module")
def scans() -> list[_Scanner]:
    out = []
    for package, relpath in SCANNED_SOURCES:
        path = _package_dir(package) / relpath
        assert path.is_file(), f"pinned source missing: {path}"
        out.append(
            scan_source(path.read_text(encoding="utf-8"), _module_name(package, relpath), str(path))
        )
    return out


@pytest.fixture(scope="module")
def sites(scans: list[_Scanner]) -> list[PutSite]:
    return [site for scan in scans for site in scan.sites]


# --- the pinned sources ----------------------------------------------------


def test_scan_finds_every_put_symbol(sites: list[PutSite]) -> None:
    """The scan is not vacuous: each of the three C puts has a call site."""
    found = {site.symbol for site in sites}
    assert found == PUT_SYMBOLS, f"put symbols with no call site: {sorted(PUT_SYMBOLS - found)}"


def test_every_put_site_is_behind_an_armed_blocked_row(scans: list[_Scanner]) -> None:
    uncovered = [
        site.where()
        for scan in scans
        for site in scan.sites
        if not is_covered(site.enclosing, scan.calls, _ARMED_BLOCKED)
    ]
    assert not uncovered, (
        "raw Channel Access put sites no _ARMED_BLOCKED row covers "
        "(add a row for the enclosing public function):\n" + "\n".join(uncovered)
    )


def test_every_put_site_calls_a_known_receiver(sites: list[PutSite]) -> None:
    """Each site calls through the libca handle or the blocked cadef binding."""
    unexpected = []
    for site in sites:
        if site.receiver == "libca":
            if (_LIBCA_HANDLE, site.symbol) not in COOPERATIVE_BYPASSES:
                unexpected.append(site.where())
        elif site.receiver == "cadef":
            if (_CADEF_MODULE, site.symbol) not in _ARMED_BLOCKED:
                unexpected.append(site.where())
        else:
            unexpected.append(f"{site.where()} (receiver {site.receiver!r})")
    assert not unexpected, "put sites through an unlisted receiver:\n" + "\n".join(unexpected)


def test_cooperative_bypasses_match_the_handle_symbols() -> None:
    assert {attr for _handle, attr in COOPERATIVE_BYPASSES} == set(_HANDLE_PUT_SYMBOLS)
    assert set(_HANDLE_PUT_SYMBOLS) == PUT_SYMBOLS
    for row, reason in COOPERATIVE_BYPASSES.items():
        assert reason.strip(), f"{row} has no reason"
        assert row not in _ARMED_BLOCKED, f"{row} is a bypass, not a partition row"


# --- the coverage rule, on synthetic sources --------------------------------

_SYNTHETIC = """
def put(chid, v):
    libca.ca_array_put(1, 1, chid, v)

def _helper(chid, v):
    libca.ca_array_put_callback(1, 1, chid, v, None, None)

def sg_put(gid, chid, v):
    _helper(chid, v)

def _orphan(chid, v):
    libca.ca_sg_array_put(0, 1, 1, chid, v)

def open_door(chid, v):
    libca.ca_array_put(1, 1, chid, v)

def _leaky(chid, v):
    libca.ca_array_put(1, 1, chid, v)

def uses_leaky(chid, v):
    _leaky(chid, v)

def blocked_leaky(chid, v):
    _leaky(chid, v)

def _loop_a():
    _loop_b()
    libca.ca_array_put(1, 1, 0, 0)

def _loop_b():
    _loop_a()

class Channel:
    def write(self, v):
        def inner():
            libca.ca_array_put(1, 1, self, v)
        inner()

libca.ca_array_put(1, 1, 0, 0)
"""

_SYNTHETIC_BLOCKED = {
    ("fake", "put"),
    ("fake", "sg_put"),
    ("fake", "blocked_leaky"),
    ("fake.Channel", "write"),
}


@pytest.fixture(scope="module")
def synthetic() -> _Scanner:
    return scan_source(_SYNTHETIC, "fake", "fake.py")


def _coverage(scan: _Scanner) -> dict[int, bool]:
    return {s.line: is_covered(s.enclosing, scan.calls, _SYNTHETIC_BLOCKED) for s in scan.sites}


def test_synthetic_scan_attributes_sites_to_outermost_function(synthetic: _Scanner) -> None:
    by_line = {s.line: s.enclosing for s in synthetic.sites}
    assert by_line[3] == ("fake", "put")
    assert by_line[6] == ("fake", "_helper")
    assert by_line[36] == ("fake.Channel", "write")
    assert by_line[39] is None


@pytest.mark.parametrize(
    ("line", "covered", "why"),
    [
        (3, True, "blocked directly"),
        (6, True, "private helper whose only caller is blocked"),
        (12, False, "private helper with no caller"),
        (15, False, "public function with no row"),
        (18, False, "private helper with one unblocked caller"),
        (28, False, "private cycle with no blocked entry"),
        (36, True, "nested def inside a blocked method"),
        (39, False, "module-level call"),
    ],
)
def test_synthetic_coverage_rule(synthetic: _Scanner, line: int, covered: bool, why: str) -> None:
    assert _coverage(synthetic)[line] is covered, why


def test_uncovered_site_reports_file_and_line(synthetic: _Scanner) -> None:
    site = next(s for s in synthetic.sites if s.line == 15)
    assert site.where() == "fake.py:15: ca_array_put in fake.open_door"
