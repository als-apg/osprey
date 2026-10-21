"""Every name a framework tree takes from ``osprey_connectors`` resolves.

A framework release pins ``osprey-connectors`` with a floor and no ceiling, so
a rebuild of it resolves the newest connectors. A name the framework imports
that the newer connectors no longer carry breaks that framework on the code
path that reaches it, and only there: an import inside a function, an
``except ImportError`` branch or a module path handed to ``importlib`` passes
every test that merely imports the module beside it.

This script reads the framework's source instead of running it. A **site** is
one of:

* ``import osprey_connectors.a.b`` -- the module must import;
* ``from osprey_connectors.a import X`` -- the module must import and ``X``
  must resolve on it: as an attribute (which a lazy module ``__getattr__``
  answers) or as the submodule ``osprey_connectors.a.X``; ``*`` asks only for
  the module;
* a string literal that is exactly a dotted ``osprey_connectors`` path, with
  an optional ``-m`` prefix (a ``python -m`` argument) or ``:attr`` suffix (an
  entry point): it resolves as a module, else as an attribute of its parent
  module, and a ``-m`` path must be a module;
* a call passing ``module_path="osprey_connectors..."`` together with
  ``class_name="X"`` (a connector registration): ``X`` must resolve on the
  module;
* an attribute read through a name an import bound to a connectors module --
  ``import osprey_connectors.settle as step`` then ``step.settle``, ``from
  osprey_connectors import types as t`` then ``t.MOCK``, or ``import
  osprey_connectors.a`` then ``osprey_connectors.a.X``: each step of the chain
  must resolve, through submodules, up to the first name that is not a module.
  What is read off a name ``from`` bound to something other than a module (a
  class, a constant) is not checked; the binding is file-wide, which over-reads only a file
  that rebinds the name to something else.

Each is found wherever it sits -- module level, inside a function, under
``if TYPE_CHECKING:`` or in a ``try`` body. Relative imports stay inside the
framework and are not sites. Resolution imports the connectors module for
real, from the active environment or from ``--connectors-path``, so a module
that exists but fails to import is reported with its error.

Usage::

    python -I scan_framework_imports.py <framework package dir>...
    python -I scan_framework_imports.py --package osprey [--package pyaml_cs_osprey]
    python -I scan_framework_imports.py src/osprey --connectors-path packages/osprey-connectors/src

Prints one line per unresolved site, ``<file>:<line>: <module>[:<name>]`` and
the reason, then a summary per tree on stderr. Exits 1 when any site is unresolved, 2
when the scan itself cannot run. Standard library only: CI runs it with
``python -I`` in a venv where this checkout is not importable.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: The distribution's import name.
CONNECTORS = "osprey_connectors"

#: A string literal that names a connectors module path, whole: ``-m`` prefix
#: and ``:attr`` suffix optional, one dotted component at least.
_STRING_PATH_RE = re.compile(
    rf"^(?P<flag>-m\s+)?(?P<path>{CONNECTORS}(?:\.[A-Za-z_]\w*)+)(?::(?P<attr>[A-Za-z_]\w*))?$"
)

#: The keyword pair a connector registration names its class with.
_REGISTRATION_KEYWORDS = ("module_path", "class_name")


@dataclass(frozen=True)
class Site:
    """One place the framework reaches into the connectors.

    Attributes:
        file: The source file, as given to :func:`collect_sites` joined with its
            path below the scanned root.
        line: The 1-based line of the statement or literal.
        module: The connectors module the site names.
        name: The name it takes from ``module``, or ``None`` for the module itself.
        kind: ``import``, ``from``, ``string``, ``string-m``, ``registration``
            or ``attr`` (``name`` is then the dotted chain read off ``module``).
    """

    file: str
    line: int
    module: str
    name: str | None
    kind: str

    def label(self) -> str:
        return self.module if self.name is None else f"{self.module}:{self.name}"


@dataclass(frozen=True)
class Unresolved:
    """A site the installed connectors cannot answer, and why."""

    site: Site
    reason: str

    def __str__(self) -> str:
        return f"{self.site.file}:{self.site.line}: {self.site.label()}  ({self.reason})"


def _is_connectors(module: str | None) -> bool:
    return module is not None and (module == CONNECTORS or module.startswith(CONNECTORS + "."))


def _string_value(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_bindings(tree: ast.AST) -> dict[str, str]:
    """Each name an import in ``tree`` binds to what may be a connectors module.

    ``import osprey_connectors.a`` binds ``osprey_connectors`` itself; ``import
    osprey_connectors.a as x`` binds ``x`` to ``osprey_connectors.a``; ``from
    osprey_connectors import a [as x]`` binds ``a`` (or ``x``) to
    ``osprey_connectors.a``, which the resolver keeps only if it is a module.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_connectors(alias.name):
                    continue
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[CONNECTORS] = CONNECTORS
        elif isinstance(node, ast.ImportFrom) and not node.level and _is_connectors(node.module):
            for alias in node.names:
                if alias.name != "*":
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _attribute_sites(tree: ast.AST, file: str) -> Iterator[Site]:
    """Every attribute chain read off a name :func:`_module_bindings` found."""
    bound = _module_bindings(tree)
    if not bound:
        return
    inner = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        chain: list[str] = []
        base: ast.expr = node
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if not isinstance(base, ast.Name) or base.id not in bound:
            continue
        chain.reverse()
        if not isinstance(node.ctx, ast.Load):
            chain.pop()  # an assignment or ``del`` needs no attribute to exist
        if chain:
            yield Site(file, node.lineno, bound[base.id], ".".join(chain), "attr")


def _sites_in(tree: ast.AST, file: str) -> Iterator[Site]:
    yield from _attribute_sites(tree, file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_connectors(alias.name):
                    yield Site(file, node.lineno, alias.name, None, "import")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _is_connectors(node.module):
                continue
            assert node.module is not None
            for alias in node.names:
                name = None if alias.name == "*" else alias.name
                yield Site(file, node.lineno, node.module, name, "from")
        elif isinstance(node, ast.Call):
            keywords = {kw.arg: _string_value(kw.value) for kw in node.keywords if kw.arg}
            module, name = (keywords.get(key) for key in _REGISTRATION_KEYWORDS)
            if _is_connectors(module) and name:
                assert module is not None
                yield Site(file, node.lineno, module, name, "registration")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = _STRING_PATH_RE.match(node.value.strip())
            if match is None:
                continue
            kind = "string-m" if match["flag"] else "string"
            yield Site(file, node.lineno, match["path"], match["attr"], kind)


def source_files(root: Path) -> list[Path]:
    """Every ``.py`` file below ``root`` (or ``root`` itself), sorted."""
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def collect_sites(root: Path) -> tuple[list[Site], list[str]]:
    """Every site under ``root`` in source order, and every file that would not parse.

    File labels are relative to ``root``'s parent, so a scan of an installed
    ``.../site-packages/osprey`` reads ``osprey/registry/builtins.py``.
    """
    base = root.parent
    sites: list[Site] = []
    unparsable: list[str] = []
    for path in source_files(root):
        label = path.relative_to(base).as_posix()
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError) as error:
            unparsable.append(f"{label}: {error.__class__.__name__}: {error}")
            continue
        sites.extend(_sites_in(tree, label))
    # A registration's module_path is also a bare string literal: the
    # registration site is the stronger claim and stands for both.
    named = {(site.file, site.line, site.module) for site in sites if site.kind == "registration"}
    sites = [
        site
        for site in sites
        if not (
            site.kind == "string"
            and site.name is None
            and (site.file, site.line, site.module) in named
        )
    ]
    sites.sort(key=lambda site: (site.file, site.line, site.module, site.name or ""))
    return sites, unparsable


class Resolver:
    """Answers sites against the importable ``osprey_connectors``, once per question."""

    def __init__(self) -> None:
        self._modules: dict[str, str | None] = {}
        self._names: dict[tuple[str, str], str | None] = {}

    def module_error(self, module: str) -> str | None:
        """``None`` when ``module`` imports, else why it does not."""
        if module not in self._modules:
            try:
                importlib.import_module(module)
            except Exception as error:  # an import that raises anything is a break
                self._modules[module] = f"{error.__class__.__name__}: {error}"
            else:
                self._modules[module] = None
        return self._modules[module]

    def name_error(self, module: str, name: str) -> str | None:
        """``None`` when ``name`` resolves on ``module``, else why it does not."""
        key = (module, name)
        if key not in self._names:
            self._names[key] = self._resolve_name(module, name)
        return self._names[key]

    def _resolve_name(self, module: str, name: str) -> str | None:
        error = self.module_error(module)
        if error is not None:
            return error
        target = sys.modules[module]
        try:
            getattr(target, name)
        except AttributeError:
            pass
        except Exception as error:  # a lazy __getattr__ that raises is a break
            return f"{error.__class__.__name__} resolving {name}: {error}"
        else:
            return None
        if self.module_error(f"{module}.{name}") is None:
            return None
        return f"{module} has no attribute or submodule {name!r}"

    def chain_error(self, base: str, chain: str) -> str | None:
        """``None`` when each step of ``chain`` resolves off the module ``base``.

        A ``base`` that is not an importable module is no site: either the
        binding took a class or constant, or the module is gone and the import
        that bound it reports that itself. The walk stops, resolved, at the
        first step that is not a module.
        """
        if self.module_error(base) is not None:
            return None
        module = base
        for part in chain.split("."):
            error = self.name_error(module, part)
            if error is not None:
                return error
            value = getattr(sys.modules[module], part, None)
            if isinstance(value, ModuleType):
                module = value.__name__
            elif value is None and f"{module}.{part}" in sys.modules:
                module = f"{module}.{part}"
            else:
                return None
        return None

    def site_error(self, site: Site) -> str | None:
        """``None`` when ``site`` resolves, else why it does not."""
        if site.kind == "attr":
            assert site.name is not None
            return self.chain_error(site.module, site.name)
        if site.kind == "string" and site.name is None:
            # A dotted string is a module, or an attribute of its parent.
            if self.module_error(site.module) is None:
                return None
            parent, _, attr = site.module.rpartition(".")
            if self.name_error(parent, attr) is None:
                return None
            return self.module_error(site.module)
        if site.name is None:
            return self.module_error(site.module)
        return self.name_error(site.module, site.name)


def scan(root: Path) -> tuple[list[Site], list[Unresolved], list[str]]:
    """Scan ``root`` against the importable connectors.

    Returns:
        Every site, the unresolved ones, and every file that would not parse.
    """
    sites, unparsable = collect_sites(root)
    resolver = Resolver()
    unresolved = [
        Unresolved(site, error)
        for site in sites
        if (error := resolver.site_error(site)) is not None
    ]
    return sites, unresolved, unparsable


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"error: no importable package {name!r}")
    return Path(next(iter(spec.submodule_search_locations)))


def _use_connectors_path(path: Path) -> None:
    """Import ``osprey_connectors`` from ``path`` and nowhere else."""
    sys.path.insert(0, str(path))
    module = importlib.import_module(CONNECTORS)
    origin = Path(module.__file__ or "").resolve()
    if path.resolve() not in origin.parents:
        raise SystemExit(f"error: {CONNECTORS} imported from {origin}, not from {path}")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report every osprey_connectors name a framework tree imports that does not resolve."
    )
    parser.add_argument(
        "root", nargs="*", type=Path, help="framework package directory (or file) to scan"
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="scan this importable package's directory instead, e.g. osprey (repeatable)",
    )
    parser.add_argument(
        "--connectors-path",
        type=Path,
        help="directory holding the osprey_connectors package to resolve against "
        "(default: the active environment's)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if bool(args.root) == bool(args.package):
        parser.error("name the trees to scan, or --package, not both")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.connectors_path is not None:
        _use_connectors_path(args.connectors_path)
    try:
        connectors = importlib.import_module(CONNECTORS)
    except Exception as error:
        print(f"error: {CONNECTORS} does not import: {error}", file=sys.stderr)
        return 2
    roots = [_package_dir(name) for name in args.package] if args.package else args.root
    missing = [root for root in roots if not root.exists()]
    if missing:
        print(f"error: {', '.join(map(str, missing))} does not exist", file=sys.stderr)
        return 2

    version = getattr(connectors, "__version__", "unknown")
    status = 0
    for root in roots:
        sites, unresolved, unparsable = scan(root)
        for problem in unresolved:
            print(problem)
        for problem in unparsable:
            print(f"{problem}  (unparsable)")
        print(
            f"scanned {root} against {CONNECTORS} {version} "
            f"({Path(connectors.__file__ or '').parent}): {len(sites)} sites, "
            f"{len(unresolved)} unresolved, {len(unparsable)} unparsable",
            file=sys.stderr,
        )
        if not sites and not unparsable:
            print(
                f"error: no site found in {root} -- the scan read the wrong tree", file=sys.stderr
            )
            return 2
        if unresolved or unparsable:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
