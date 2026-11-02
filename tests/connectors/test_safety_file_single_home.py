"""OSPREY replaces a file whole one way: ``osprey_connectors.safety_file.atomic_replace``.

An AST scan of every module under ``src/`` and ``packages/`` refuses a rename
call outside the allow-list below: ``os.replace``, ``os.rename``,
``shutil.move``, or a one-argument ``.replace(...)``/``.rename(...)`` whose
argument is not a string literal (``Path.replace``/``Path.rename``). A
hand-rolled "write a temporary, rename it over the target" therefore fails here
with the place to go instead. The allow-list names the renames that move or set
aside existing entries rather than replace a file with new bytes, each with its
reason; an entry with no matching call fails too, so the list cannot rot.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src", "packages")

MESSAGE = (
    "replace a file whole through osprey_connectors.safety_file.atomic_replace "
    "(hooks: osprey_safety_file)"
)

_HOOKS = "src/osprey/templates/claude_code/claude/hooks"

#: (repo-relative path, enclosing function qualname) -> why the rename stays.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "packages/osprey-connectors/src/osprey_connectors/safety_file.py",
        "atomic_replace",
    ): "the one way a file is replaced whole",
    (f"{_HOOKS}/osprey_safety_file.py", "atomic_replace"): (
        "the hooks' byte-identical copy of the one way"
    ),
    (f"{_HOOKS}/osprey_hook_log.py", "_spool_write_sealed"): (
        "seals a fresh spool segment under a new name, as the record's writer does; "
        "never over an existing file"
    ),
    ("src/osprey/agent_record/ingest/spool.py", "_quarantine"): (
        "moves a hostile spool entry aside as evidence"
    ),
    ("src/osprey/agent_record/writer.py", "_seal"): (
        "renames an open spool segment to its sealed name, never over an existing file"
    ),
    ("src/osprey/agent_record/writer.py", "write_batch"): (
        "seals a fresh spool segment under a new name, never over an existing file"
    ),
    ("src/osprey/cli/build_cmd.py", "_clear_leftover"): "sets an existing build entry aside",
    ("src/osprey/cli/build_cmd.py", "_repair_interrupted_swap"): (
        "completes an interrupted directory swap"
    ),
    ("src/osprey/cli/build_cmd.py", "_swap_in_render"): "swaps whole build trees",
    ("src/osprey/cli/build_cmd.py", "_render_container_project"): (
        "renames a rendered tree inside the staged image root"
    ),
    ("src/osprey/cli/init_cmd.py", "_reinstate_held_source_zone"): (
        "moves a held entry back to its place"
    ),
    ("src/osprey/cli/init_cmd.py", "_replacing_source_zone"): (
        "moves existing entries to a stash and back"
    ),
    ("src/osprey/cli/init_revalidate.py", "_git_move"): (
        "moves an existing file where git does not track it"
    ),
    ("src/osprey/cli/init_revalidate.py", "_move_retired_layer_dir"): (
        "moves an existing layer directory or its files where git does not track them"
    ),
    ("src/osprey/cli/scaffold_cmd.py", "claim_into_profile"): "moves an existing tree",
    ("src/osprey/deployment/container_lifecycle.py", "_migrate_legacy_curve_dir"): (
        "relocates a directory"
    ),
    ("src/osprey/deployment/container_lifecycle.py", "_generate_bluesky_curve_certs"): (
        "moves files a library minted in a scratch directory"
    ),
    ("src/osprey/deployment/web_terminals/env_production.py", "migrate_users_env"): (
        "renames an existing secrets file"
    ),
    ("src/osprey/interfaces/web_terminal/ownership.py", "OwnershipStore._quarantine"): (
        "sets a corrupt index aside"
    ),
    ("src/osprey/interfaces/web_terminal/routes/chat_attachments.py", "upload_chat_attachment"): (
        "moves a streamed upload into place; atomic_replace takes bytes"
    ),
    ("src/osprey/services/archive/run.py", "ArchiveTree.place"): (
        "places a staged copy, never over an existing file"
    ),
    ("src/osprey/services/channel_finder/graph_index/builder.py", "build_from_rows"): (
        "the database library writes the file itself; atomic_replace takes bytes"
    ),
}


def _is_rename_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    receiver = func.value
    receiver_name = receiver.id if isinstance(receiver, ast.Name) else None
    if receiver_name == "os" and func.attr in {"replace", "rename"}:
        return True
    if receiver_name == "shutil" and func.attr == "move":
        return True
    if func.attr not in {"replace", "rename"} or receiver_name == "dataclasses":
        return False
    if len(call.args) != 1 or call.keywords:
        return False
    argument = call.args[0]
    return not (isinstance(argument, ast.Constant) and isinstance(argument.value, str))


class _RenameFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scope: list[str] = []
        self.found: list[tuple[str, int]] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_rename_call(node):
            self.found.append((".".join(self._scope) or "<module>", node.lineno))
        self.generic_visit(node)


#: The attribute names a rename call reaches through. A call spells its
#: attribute in the source, and an identifier has no escape sequence, so a module
#: that spells none of them holds no rename call and is passed over unparsed.
_RENAME_ATTRIBUTES = ("replace", "rename", "move")


@functools.cache
def _rename_calls() -> tuple[tuple[str, str, int], ...]:
    """Every rename call in the scanned trees, as (path, qualname, line).

    Both tests read the same scan, and the tree does not change while the suite
    runs, so its parse of every module is done once.
    """
    found: list[tuple[str, str, int]] = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if not any(attribute in source for attribute in _RENAME_ATTRIBUTES):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            finder = _RenameFinder()
            finder.visit(ast.parse(source, filename=relative))
            found += ((relative, qualname, line) for qualname, line in finder.found)
    return tuple(found)


def test_no_file_is_replaced_by_a_hand_rolled_rename() -> None:
    offenders = [
        f"{path}:{line} ({qualname})"
        for path, qualname, line in _rename_calls()
        if (path, qualname) not in ALLOWED
    ]
    assert not offenders, f"{MESSAGE}: " + ", ".join(offenders)


def test_every_allowed_rename_still_exists() -> None:
    seen = {(path, qualname) for path, qualname, _line in _rename_calls()}
    stale = sorted(
        f"{path} ({qualname})" for path, qualname in ALLOWED if (path, qualname) not in seen
    )
    assert not stale, f"allow-list entries with no rename call left: {stale}"
