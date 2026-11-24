"""No code under ``src/osprey`` opens the ARIEL store without stating a deadline.

The entry point owns the clock: a command, a service start-up or a tool call
states how long it gives the store to answer and hands that number down, and
nothing below it holds one. The layers below take the number as a keyword that
defaults to ``None`` ("the caller states none"), so a new caller that forgets it
would still run, and wait on a silent store for ever. This guard is what stops
that.

The rule, read off the tree: every function in ``src/osprey`` that takes
``store_deadline_s`` opens the store, as does ``create_connection_pool`` with its
``deadline_s``; every call to one of them under ``src/osprey`` passes that
keyword. Calls are resolved through each module's imports, so another module's
function of the same name is not mistaken for one. A literal ``None`` there is
no deadline either, so it is refused the same way. Tests and fixtures are free
to pass none.
"""

from __future__ import annotations

import ast
import unicodedata
from collections import defaultdict
from pathlib import Path

from tests._tree_scan import REPO_ROOT, product_sources

#: The open itself, and the keyword it takes.
_POOL = "osprey.services.ariel_search.database.connection.create_connection_pool"
_POOL_KEYWORD = "deadline_s"

#: The keyword every function above the open takes and passes on.
_KEYWORD = "store_deadline_s"


def _module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts[1:]  # drop "src"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _absolute(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")
    base = base if is_package else base[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


class _Module:
    """One module's top-level functions, every name its imports bind, and its calls."""

    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self.name = _module_name(path)
        self.tree = ast.parse(text, filename=path)
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        is_package = path.endswith("__init__.py")
        # A name may be bound in several function scopes; keep every target.
        self.bound: dict[str, set[str]] = defaultdict(set)
        for name in self.functions:
            self.bound[name].add(f"{self.name}.{name}")
        #: ``(name imported, name bound)`` for every import that binds a name.
        self.renames: set[tuple[str, str]] = set()
        self.calls: list[ast.Call] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.calls.append(node)
            elif isinstance(node, ast.ImportFrom):
                source = _absolute(self.name, is_package, node)
                for alias in node.names:
                    self.bound[alias.asname or alias.name].add(f"{source}.{alias.name}")
                    self.renames.add((alias.name, alias.asname or alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.bound[alias.asname].add(alias.name)
                        self.renames.add((alias.name.rpartition(".")[2], alias.asname))


def _identifier_text(text: str) -> str:
    """*text* as Python reads its identifiers: NFKC-normalized."""
    if text.isascii() or unicodedata.is_normalized("NFKC", text):
        return text
    return unicodedata.normalize("NFKC", text)


def _modules() -> dict[str, _Module]:
    """The modules under ``src/osprey`` that can define, re-export or call an opener.

    A call reaches an opener only through names that are an opener's own or an
    import's rename of one of those, each spelled in every module on the way:
    the caller, and each module whose import the call is resolved through. So
    a module that spells none of them can neither hold such a call nor lie on
    its way, and nor can one without ``store_deadline_s`` define an opener.
    The names grow as the modules that spell them are read, until no unread
    module spells a new one; parsing only those finds every call the whole
    tree would, and parsing is what the guard spends its time on. A module left
    unread is never parsed, so one that does not parse no longer fails this
    guard; ruff's syntax check covers that.
    """
    sources = [
        (path, text) for path, text in product_sources(REPO_ROOT) if path.startswith("src/osprey/")
    ]
    texts = dict(sources)
    unread = {path: _identifier_text(text) for path, text in sources}
    modules: dict[str, _Module] = {}
    names: set[str] = set()
    new = {_KEYWORD, _POOL.rpartition(".")[2]}
    while new:
        names |= new
        # An unread module spells none of the older names: only the new ones are looked for.
        for path in [path for path, text in unread.items() if any(name in text for name in new)]:
            del unread[path]
            module = _Module(path, texts[path])
            modules[module.name] = module
        grown = names | {name.rpartition(".")[2] for name in _openers(modules)}
        while more := {
            bound
            for module in modules.values()
            for imported, bound in module.renames
            if imported in grown and bound not in grown
        }:
            grown |= more
        new = grown - names
    # In the tree's order, as a guard that read every module would hold them.
    read = {module.path: module for module in modules.values()}
    return {read[path].name: read[path] for path, _text in sources if path in read}


def _openers(modules: dict[str, _Module]) -> dict[str, str]:
    """Every store-opening function's qualified name, with the keyword it takes."""
    found = {_POOL: _POOL_KEYWORD}
    for module in modules.values():
        for name, node in module.functions.items():
            params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if any(param.arg == _KEYWORD for param in params):
                found[f"{module.name}.{name}"] = _KEYWORD
    return found


def _resolve(qualified: str, modules: dict[str, _Module], openers: dict[str, str]) -> str | None:
    """Follow re-exports until *qualified* names an opener, or give up."""
    for _ in range(8):
        if qualified in openers:
            return qualified
        owner, _, name = qualified.rpartition(".")
        module = modules.get(owner)
        if module is None or name in module.functions:
            return None
        targets = module.bound.get(name, set()) - {qualified}
        if len(targets) != 1:
            return None
        (qualified,) = targets
    return None


def _clockless_calls() -> tuple[dict[str, str], list[str]]:
    modules = _modules()
    openers = _openers(modules)
    missing: list[str] = []
    for module in modules.values():
        for node in module.calls:
            func = node.func
            if isinstance(func, ast.Name):
                candidates = module.bound.get(func.id, set())
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                candidates = {
                    f"{base}.{func.attr}" for base in module.bound.get(func.value.id, set())
                }
            else:
                continue
            for candidate in candidates:
                opener = _resolve(candidate, modules, openers)
                if opener is None:
                    continue
                stated = [kw for kw in node.keywords if kw.arg == openers[opener]]
                if not stated:
                    missing.append(
                        f"{module.path}:{node.lineno} calls {opener.rpartition('.')[2]} "
                        f"without {openers[opener]}="
                    )
                elif any(
                    isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in stated
                ):
                    missing.append(
                        f"{module.path}:{node.lineno} calls {opener.rpartition('.')[2]} "
                        f"with {openers[opener]}=None"
                    )
                break
    return openers, missing


def test_every_store_open_under_src_states_its_deadline() -> None:
    openers, missing = _clockless_calls()

    # The guard reads the tree it means to: the open, the service and the
    # entry points' operations are all in the set it checks calls against.
    names = {name.rpartition(".")[2] for name in openers}
    assert {"create_connection_pool", "create_ariel_service", "run_migrate"} <= names
    assert missing == [], "\n".join(missing)
