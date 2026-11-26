"""What a deployment repo still holds at retired root paths, and how it moves.

The retired paths are the moved-from paths of :mod:`osprey.repo_layout`
(:func:`~osprey.repo_layout.retired_moves`), each looked for at
``<repo root>/<old path>``: an old path is where the readers of its time read
it. The one exception is the retired ``data:`` profile key: the rows whose
old readers followed it (the rendered sources) are looked for under the one
directory it names. Nothing here names a path of its own.

:func:`plan_retired` is the one plan, shared by the refusal and by
``osprey init --revalidate``. It looks at the files only, neither runs git
nor prints, and reports:

- the moves and clashes of each retired path that still holds a source, as
  :func:`osprey_facility.retired.place_moves` places them, dot-files included;
- the facility tree's own moves, found in the tree as the root moves leave
  it and re-rooted where it lies then;
- the line edits that follow the moves: values under the profile keys that
  hold repo-relative paths (:data:`PATH_KEYS`) in ``profile.yml``, the tier
  documents and the profile variants, the root pipeline file and the root
  ``.dockerignore``;
- the ``data:`` profile lines, deleted or listed to check by hand;
- what is left under a dissolved directory that no move takes (the
  leftovers), and what is pruned with it;
- the stale canopy staging at the repo root and the data README osprey
  shipped, removed with the moves;
- the steps to take on the host once the files moved;
- what a person must check by hand, because the paths inside it are not
  rewritten.

:func:`source_of` and :func:`after_moves_view` read a path, or a tree, as it
is once moves are done. :func:`refuse_retired_layout` stops a command on such
a repo with one block that moves the root, then the facility tree, and ends
with :func:`osprey_facility.errors._revalidate_command`;
:func:`deployment_root` finds a command's repo and refuses it so.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Literal, Protocol

import click
import yaml

from osprey import repo_layout
from osprey.repo_layout import RootMove
from osprey_facility.errors import (
    MOVE_HEADER,
    PASTE_HEADER,
    refusal_ending,
)
from osprey_facility.retired import (
    Clash,
    LineDelete,
    LineEdit,
    Move,
    RetiredLayout,
    deck_line_edits,
    find_retired,
    place_moves,
    refuse_retired,
    render_script,
)
from osprey_facility.roots import FacilityRoots

__all__ = [
    "PATH_KEYS",
    "RetiredPlan",
    "RetiredRoot",
    "RetiredRootRefused",
    "after_moves_view",
    "deployment_root",
    "find_retired_root",
    "plan_retired",
    "refuse_retired_layout",
    "source_of",
    "successor_of",
]

#: The profile keys whose values are filesystem paths relative to the repo
#: root, as dotted paths through a profile document. ``*`` stands for any one
#: mapping key or sequence item. A key spelled with dots in the document
#: (``config: {code_review.prompt: ...}``) reads as the same dotted path as its
#: nested spelling. Only values under these keys are rewritten. A service's
#: ``template`` is one when it names a facility-owned service directory; a
#: bundled template name never begins with a moved path. A service's ``path``
#: is not one: it names where the build writes the service, inside the build
#: tree, which no root move touches; nor is a key read relative to the render.
PATH_KEYS: tuple[str, ...] = (
    "services.*.template",
    "config.code_review.prompt",
    "config.facility_knowledge.bundle_path",
    "config.modules.web_terminals.landing.notices.*",
    "config.services.optimizer.environments_dir",
    "config.services.bluesky.plan_dir",
    "config.services.qmd.models_dir",
    "config.web.panels.canopy.theme_file",
    "dispatch.triggers",
    "optimizer.environments_dir",
    "lifecycle.*.*.cwd",
)

#: Profile keys whose values are shell commands: a retired path inside one is
#: listed to check by hand, never rewritten.
COMMAND_KEYS: tuple[str, ...] = ("lifecycle.*.*.run",)

#: The retired profile key that named the directory the rendered sources were
#: read from.
_DATA_KEY = "data"

#: What the root pipeline file includes or builds from, and the ignore file of
#: every image build, both at the repo root.
_PIPELINE_KEY = ".gitlab-ci.yml"
_DOCKERIGNORE_KEY = ".dockerignore"

#: How a value anchored at the project root through a placeholder begins.
_PROJECT_ROOT_PLACEHOLDER = "{project_root}/"

#: The keys of the rows whose moved files hold paths no line edit rewrites,
#: each with what its files are.
_INNER_PATHS_UNCHANGED: Mapping[str, str] = {
    "ci-extra.yml": "the CI extra",
    "services": "the service Dockerfiles",
    "scripts": "the deployment scripts",
}

#: The README osprey shipped in a dissolved directory, by name.
_SHIPPED_README = "README.md"

#: The sha256 of every copy of that README osprey shipped: each version at a
#: release tag, and the last copy before it was deleted. Any other content is
#: the deployment's own, so it is a leftover.
_SHIPPED_DATA_README_DIGESTS: frozenset[str] = frozenset(
    {
        "4359e5deb4a2ff1fb3831a7fb0174d8b0efda4f2bbda8b53f5f802ed293e5e55",
        "23b51928f24bebc3fddd23df1c9e7a4cc62cd10bcb772ef5d4b18b4f23d1ec1c",
        "e617bd1a41b3ab5ad360e469151bc71088b7870f8de9683922b112d688404510",
        "efc60ba4e8e16949f84458517e9196ff1bb7d1d12f322285aafed5d0e2ca7f02",
        "6c3966eca0951444f1af328235a1a62976a5af92c8cd46a9f8d16cb2823b81a9",
        "1bd84dfdfbb8be5ff74b93d56580bb2882bf5a25cde5304597642f1156286e50",
    }
)

#: A file macOS writes into any directory it shows; never a leftover.
_FINDER_FILE = ".DS_Store"

_STR_TAG = "tag:yaml.org,2002:str"


class _Moved(Protocol):
    """A move as :func:`source_of` and :func:`successor_of` read it."""

    @property
    def old(self) -> str: ...

    @property
    def new(self) -> str: ...


@dataclass(frozen=True)
class RetiredRoot:
    """Everything a deployment repo still holds at its retired root paths.

    Attributes:
        root: The repo root, resolved.
        moves: The sources to move, in the order they run.
        clashes: What blocks a move.
        line_edits: The edits that follow the moves, each naming its file at
            the path it has once the moves are done.
        line_deletes: The ``data:`` profile lines to delete, each naming its
            file at the path it has once the moves are done.
        deletes: What is removed with the moves: stale generated copies at
            the repo root, the shipped data README, and what is pruned from
            a dissolved directory; POSIX, relative to *root*.
        leftovers: What lies under a dissolved directory, or under the
            directory ``data:`` names, that no move takes; POSIX, relative
            to *root*, a directory ending in ``/``.
        data_keys: Each ``data:`` line of a profile document, as
            ``<document>:<line>: data: <value>``.
        host_steps: The commands to run on the host once the files moved, in
            order.
        check_by_hand: What a person must look at, one line each, because the
            paths inside it are not rewritten or cannot be decided.
        shipped: The data README osprey shipped unchanged, deleted with the
            moves; POSIX, relative to *root*.
        data_root: The one directory every ``data:`` names, other than the
            dissolved one, where the rendered sources are looked for; POSIX,
            relative to *root*.
    """

    root: Path
    moves: tuple[Move, ...] = ()
    clashes: tuple[Clash, ...] = ()
    line_edits: tuple[LineEdit, ...] = ()
    line_deletes: tuple[LineDelete, ...] = ()
    deletes: tuple[str, ...] = ()
    leftovers: tuple[str, ...] = ()
    data_keys: tuple[str, ...] = ()
    host_steps: tuple[str, ...] = ()
    check_by_hand: tuple[str, ...] = ()
    shipped: tuple[str, ...] = ()
    data_root: str | None = None

    @property
    def empty(self) -> bool:
        """True when nothing at a retired path, or naming one, is left.

        A ``data:`` line, a leftover and a shipped README each count. The
        other deletes, the host steps and the check-by-hand lines ride along;
        on their own they are no retired layout.
        """
        return not (
            self.moves
            or self.clashes
            or self.line_edits
            or self.line_deletes
            or self.leftovers
            or self.data_keys
            or self.shipped
        )

    @property
    def finishes(self) -> bool:
        """True when, once its block ran, ``osprey init --revalidate`` can finish the migration."""
        staying = len(self.data_keys) - len(self.line_deletes)
        return not self.clashes and staying == 0


@dataclass(frozen=True)
class RetiredPlan:
    """The one plan of a deployment repo's migration off its retired paths.

    Attributes:
        layout: The root half.
        facility_roots: The facility roots as they lie once the root moves
            are done, or ``None`` when the profile names none that can be read.
        facility: What the facility tree holds at its own retired paths, read
            in the tree as the root moves leave it and re-rooted at
            *facility_roots*; ``None`` without facility roots.
    """

    layout: RetiredRoot
    facility_roots: FacilityRoots | None = None
    facility: RetiredLayout | None = None

    @property
    def empty(self) -> bool:
        """True when neither the root nor the facility tree holds anything retired."""
        return self.layout.empty and (self.facility is None or self.facility.empty)

    @property
    def moves(self) -> tuple[Move, ...]:
        """Every move in the order it runs: the root moves, then the facility moves."""
        facility = () if self.facility is None else self.facility.moves
        return (*self.layout.moves, *facility)

    @property
    def clashes(self) -> tuple[Clash, ...]:
        """Every clash: the root's, then the facility tree's."""
        facility = () if self.facility is None else self.facility.clashes
        return (*self.layout.clashes, *facility)

    def script(self) -> list[str]:
        """The shell lines that carry out every move, edit and delete, in stages."""
        root = self.layout.root
        edits: list[tuple[Path, LineEdit]] = [(root, edit) for edit in self.layout.line_edits]
        if self.facility is not None:
            edits.extend(deck_line_edits(self.facility))
        deletes = [(root, path) for path in (*self.layout.shipped, *self.layout.deletes)]
        return render_script(
            self.moves,
            edits,
            deletes,
            [(root, delete) for delete in self.layout.line_deletes],
        )


class RetiredRootRefused(click.ClickException):
    """A deployment repo still holds sources at retired root paths; nothing was read.

    The message is ``# `` comment lines, then :attr:`script`, the plan's
    shell lines (they run from any working directory), then one comment per
    clash, then the host
    steps, the check-by-hand list and the leftovers as comment blocks, then
    the closing line: :func:`~osprey_facility.errors._revalidate_command` for
    the repo when the block can finish the migration, else a line asking to
    place what is listed. The whole message runs under ``sh``. It prints
    alone, with no ``Error: `` prefix, and the command line exits 1.

    Args:
        plan: The migration: the root half, and the facility tree whose
            retired paths the script moves too.
    """

    exit_code = 1

    def __init__(self, plan: RetiredPlan) -> None:
        layout = plan.layout
        facility_roots = plan.facility_roots
        facility = plan.facility
        self.layout = layout
        self.facility_roots = facility_roots
        self.facility = facility
        self.script = tuple(plan.script())
        lines = [
            f"# The deployment repo at {layout.root} still holds sources at retired paths;"
            " nothing was read."
        ]
        clashes = plan.clashes
        if facility is not None and facility_roots is not None and not facility.empty:
            where = f"# So does the facility tree at {facility_roots.machine.resolve()}"
            if facility_roots.mounted:
                where += f" (content at {facility_roots.content.resolve()})"
            lines.append(f"{where}.")
        finishes = layout.finishes and not clashes
        if self.script:
            lines.append(PASTE_HEADER if finishes else MOVE_HEADER)
            lines.extend(self.script)
        lines.extend(
            f"# {clash.old.resolve()} and {clash.new.resolve()} both exist:"
            " keep one, delete the other."
            for clash in clashes
        )
        if layout.host_steps:
            lines.append("# Host steps, once the files moved:")
            lines.extend(f"#   {step}" for step in layout.host_steps)
        if layout.check_by_hand:
            lines.append("# Check by hand; the paths inside these are not rewritten:")
            lines.extend(f"#   {item}" for item in layout.check_by_hand)
        lines.extend(_leftover_lines(layout.leftovers, layout.data_root))
        lines.extend(refusal_ending(self.script, layout.root, finishes=finishes))
        super().__init__("\n".join(lines))

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the message alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        (file or sys.stderr).write(self.format_message() + "\n")


def plan_retired(repo_root: Path, facility_roots: FacilityRoots | None = None) -> RetiredPlan:
    """The migration of the deployment repo at *repo_root* off every retired path.

    The root half is what :func:`find_retired_root` reports. The facility
    half is read in the tree as the root moves leave it (through
    :func:`after_moves_view`), at *facility_roots* when given, else at the
    roots the profile names (``<repo>/facility``, and ``machine:``), read
    raw; its moves, edits and clashes are re-rooted where the tree lies once
    the root moves are done, and a clash names each side where it lies now.

    Args:
        repo_root: The deployment repo root.
        facility_roots: The facility roots the caller resolved, as they lie
            once the root moves are done, or ``None``.

    Returns:
        The plan; :attr:`RetiredPlan.empty` when nothing is left to do.
    """
    layout = find_retired_root(repo_root)
    root = layout.root
    roots = facility_roots if facility_roots is not None else _profile_facility_roots(root)
    if roots is None:
        return RetiredPlan(layout=layout)
    return RetiredPlan(layout=layout, facility_roots=roots, facility=_facility(root, layout, roots))


def find_retired_root(repo_root: Path) -> RetiredRoot:
    """What the deployment repo at *repo_root* still holds at retired root paths.

    Each retired path is looked for at ``<repo_root>/<old path>``, except the
    rendered sources while a profile document's ``data:`` names one other
    directory inside the repo: those are looked for there. A directory
    merged into an existing successor moves file by file, dot-files
    included; a move into another move's successor runs after it.

    Args:
        repo_root: The deployment repo root.

    Returns:
        The result; :attr:`RetiredRoot.empty` when nothing is left.
    """
    root = repo_root.resolve()
    if not repo_layout.retired_moves():
        return RetiredRoot(root=root)
    keys = _data_keys(root)
    anchor, by_hand = _data_anchor(root, keys)
    table = _effective_moves(anchor)
    moves, clashes = place_moves(root, [(move.old, move.new) for move in table], hidden=True)
    containers = [*repo_layout.dissolved_names()]
    if isinstance(anchor, str) and anchor not in containers:
        containers.append(anchor)
    leftovers, prunes, shipped = _leftovers(root, containers, [move.old for move in table])
    edits: list[LineEdit] = []
    for document in _profile_documents(root):
        _document_edits(root, document, table, edits, by_hand)
    edits.extend(_pipeline_edits(root, table))
    edits.extend(_dockerignore_edits(root, table))
    anchored_left = isinstance(anchor, str) and any(
        item.startswith(f"{anchor}/") for item in leftovers
    )
    deleted = [] if anchor is False or anchored_left else _line_deletes(root, keys, table, by_hand)
    data_keys = tuple(key.shown for key in keys)
    if anchored_left:
        by_hand.extend(
            f"{key.shown}: delete this line once {anchor}/ holds nothing" for key in keys
        )
    if not (moves or clashes or edits or deleted or leftovers or data_keys or shipped):
        return RetiredRoot(root=root)
    acting = bool(moves or edits or deleted or shipped)
    by_hand.extend(_moved_inner_paths(root, table))
    return RetiredRoot(
        root=root,
        moves=tuple(moves),
        clashes=tuple(clashes),
        line_edits=tuple(edits),
        line_deletes=tuple(deleted),
        deletes=(*_canopy_deletes(root), *prunes) if acting else (),
        leftovers=tuple(leftovers),
        data_keys=data_keys,
        host_steps=_host_steps(root) if moves else (),
        check_by_hand=tuple(dict.fromkeys(by_hand)),
        shipped=tuple(shipped),
        data_root=anchor if isinstance(anchor, str) else None,
    )


def refuse_retired_layout(repo_root: Path, facility_roots: FacilityRoots | None = None) -> None:
    """Stop when the repo, or the facility tree it names, holds anything at a retired path.

    When the root holds a retired path, one block moves the root and then
    the facility tree, as :func:`plan_retired` plans them: at the facility
    roots given, else at those the profile names. When only the facility
    tree at *facility_roots* is retired, the facility refusal is raised, its
    closing command naming this repo.

    Args:
        repo_root: The deployment repo root.
        facility_roots: The facility roots the caller resolved, or ``None``.

    Raises:
        RetiredRootRefused: The root holds something at a retired path.
        osprey_facility.errors.RetiredLayoutRefused: Only the facility tree
            at *facility_roots* does.
    """
    plan = plan_retired(repo_root, facility_roots)
    layout = plan.layout
    if layout.empty:
        if facility_roots is not None:
            refuse_retired(facility_roots.machine, facility_roots.content, repo=layout.root)
        return
    raise RetiredRootRefused(plan)


def deployment_root(repo: Path | None) -> Path:
    """The deployment repo *repo* selects, refused when it holds anything at a retired path.

    The repo is found as :func:`osprey.cli.repo_resolver.find_repo_root` finds
    it, then checked by :func:`refuse_retired_layout` before anything in it is
    read or written.

    Args:
        repo: Value of ``--repo``, or ``None`` to walk up from the working
            directory.

    Raises:
        RepoNotFoundError: If no deployment repo encloses the search path.
        RetiredRootRefused: If the repo holds something at a retired path.
    """
    from osprey.cli.repo_resolver import find_repo_root

    repo_root = find_repo_root(repo)
    refuse_retired_layout(repo_root)
    return repo_root


# --- the facility half ---------------------------------------------------------


def _profile_facility_roots(root: Path) -> FacilityRoots | None:
    """The facility roots ``profile.yml`` names, read raw, as they lie after the root moves.

    The content root is the ``facility`` row; the machine root is the same
    tree unless ``machine:`` names another directory. ``None`` when the
    profile cannot be read as a mapping, or when the two roots overlap, which
    the build itself reports.
    """
    document = _load(root / repo_layout.entry("profile.yml").path)
    if not isinstance(document, dict):
        return None
    content = root.joinpath(*repo_layout.entry("facility").path.split("/"))
    machine = document.get("machine")
    if not isinstance(machine, str) or not machine.strip():
        return FacilityRoots.single(content)
    machine_dir = Path(os.path.normpath(root / machine))
    if machine_dir.is_relative_to(content) or content.is_relative_to(machine_dir):
        return None
    return FacilityRoots(machine=machine_dir, content=content)


def _facility(root: Path, layout: RetiredRoot, roots: FacilityRoots) -> RetiredLayout:
    """The facility tree's own retired layout, read as the root moves leave it."""
    inside = [path for path in (roots.machine, roots.content) if path.is_relative_to(root)]
    if not layout.moves or not inside:
        return find_retired(roots.machine, roots.content if roots.mounted else None)
    rels = [path.relative_to(root).as_posix() for path in inside]
    with after_moves_view(root, layout.moves, rels) as view:

        def seen(path: Path) -> Path:
            return view / path.relative_to(root) if path.is_relative_to(root) else path

        found = find_retired(seen(roots.machine), seen(roots.content) if roots.mounted else None)

        def after(path: Path) -> Path:
            return root / path.relative_to(view) if path.is_relative_to(view) else path

        def now(path: Path) -> Path:
            if not path.is_relative_to(view):
                return path
            return root / source_of(path.relative_to(view).as_posix(), layout.moves)

        return RetiredLayout(
            moves=tuple(replace(move, root=after(move.root)) for move in found.moves),
            deck_edits=tuple(replace(edit, file=after(edit.file)) for edit in found.deck_edits),
            clashes=tuple(Clash(old=now(clash.old), new=now(clash.new)) for clash in found.clashes),
        )


# --- the data: key -----------------------------------------------------------------


@dataclass(frozen=True)
class _DataKey:
    """One ``data:`` line of a profile document.

    ``value`` is the string the key holds, or ``None`` when it holds anything
    else. It is read from the document's nodes, never constructed, so a tag
    no loader knows elsewhere in the document does not hide the key.
    """

    document: Path
    line: int
    value: str | None
    text: str
    one_line: bool
    shown: str


def _data_keys(root: Path) -> list[_DataKey]:
    """Every top-level ``data:`` key of the profile documents, in document order."""
    found: list[_DataKey] = []
    for document in _profile_documents(root):
        try:
            text = document.read_text(encoding="utf-8")
            tree = yaml.compose(text, Loader=yaml.SafeLoader)
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(tree, yaml.MappingNode):
            continue
        lines = text.splitlines()
        where = document.relative_to(root).as_posix()
        for key, value in tree.value:
            if not (isinstance(key, yaml.ScalarNode) and key.value == _DATA_KEY):
                continue
            line = key.start_mark.line
            is_string = isinstance(value, yaml.ScalarNode) and value.tag == _STR_TAG
            raw = value.value if is_string else None
            shown_value = value.value if isinstance(value, yaml.ScalarNode) else "..."
            found.append(
                _DataKey(
                    document=document,
                    line=line + 1,
                    value=raw,
                    text=lines[line],
                    one_line=(
                        key.start_mark.column == 0
                        and value.end_mark.line == line
                        and lines[line].startswith(f"{_DATA_KEY}:")
                    ),
                    shown=f"{where}:{line + 1}: {_DATA_KEY}: {shown_value}",
                )
            )
    return found


def _normalised(root: Path, value: Any) -> str | None:
    """*value* as a repo-relative POSIX path (``""`` for the root), or ``None`` outside it."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.rstrip("/")
    if os.path.isabs(text):
        try:
            text = Path(os.path.normpath(text)).relative_to(root).as_posix()
        except ValueError:
            return None
    relative = os.path.normpath(text) if text else "."
    if relative == ".":
        return ""
    if relative == ".." or relative.startswith("../"):
        return None
    return Path(relative).as_posix()


def _data_anchor(
    root: Path, keys: Sequence[_DataKey]
) -> tuple[str | Literal[False] | None, list[str]]:
    """Where the rendered sources' moves are anchored, and the ``data:`` values to check by hand.

    Returns:
        ``None`` when every ``data:`` names the dissolved directory (or there
        is none): the moves stay as declared. The one other directory all of
        them name, when it is a valid anchor. ``False`` otherwise: the
        rendered sources do not move, and every value is checked by hand.
    """
    if not keys:
        return None, []
    values = {_normalised(root, key.value) for key in keys}
    if values <= set(repo_layout.dissolved_names()):
        return (None, []) if len(values) == 1 else (False, _bad(keys))
    if len(values) == 1:
        [value] = values
        if value and _valid_anchor(value):
            return value, []
    return False, _bad(keys)


def _bad(keys: Sequence[_DataKey]) -> list[str]:
    return [
        f"{key.shown}: names no single directory inside the repo; move what it held"
        " by hand, then delete this line"
        for key in keys
    ]


def _valid_anchor(value: str) -> bool:
    """True when *value* neither is, holds, nor lies inside a live row or a dissolved name."""
    taken = [row.path for row in repo_layout.ROOT_ENTRIES] + list(repo_layout.dissolved_names())
    return not any(
        value == path or value.startswith(f"{path}/") or path.startswith(f"{value}/")
        for path in taken
    )


def _effective_moves(anchor: str | Literal[False] | None) -> tuple[RootMove, ...]:
    """The declared moves, the rendered sources' anchored at *anchor* (see :func:`_data_anchor`)."""
    table = repo_layout.retired_moves()
    if anchor is None:
        return table
    rendered = {source.source_rel for source in repo_layout.rendered_sources()}
    dissolved = repo_layout.dissolved_names()
    moves: list[RootMove] = []
    for move in table:
        head, _, rest = move.old.partition("/")
        if move.new not in rendered or head not in dissolved or not rest:
            moves.append(move)
        elif anchor is not False:
            moves.append(RootMove(f"{anchor}/{rest}", move.new, move.kind))
    return tuple(moves)


def _line_deletes(
    root: Path, keys: Sequence[_DataKey], table: Sequence[RootMove], by_hand: list[str]
) -> list[LineDelete]:
    """Each ``data:`` line as a delete; one spanning several lines is checked by hand."""
    deletes: list[LineDelete] = []
    for key in keys:
        if key.one_line:
            target = _after_moves(root, key.document, table)
            deletes.append(LineDelete(file=target, line=key.line, text=key.text))
        else:
            by_hand.append(f"{key.shown}: delete this key by hand")
    return deletes


# --- leftovers -----------------------------------------------------------------------


def _leftovers(
    root: Path, containers: Sequence[str], olds: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """What lies under *containers* that no move in *olds* takes.

    Returns:
        The leftovers, then what is pruned (Finder files, directories holding
        nothing else), then the shipped README; POSIX, relative to *root*.
    """
    leftovers: list[str] = []
    prunes: list[str] = []
    shipped: list[str] = []
    dissolved = set(repo_layout.dissolved_names())
    for container in containers:
        directory = root / container
        if not directory.is_dir() or directory.is_symlink():
            continue
        for child in sorted(directory.iterdir()):
            rel = f"{container}/{child.name}"
            if container in dissolved and child.name == _SHIPPED_README and _shipped_readme(child):
                shipped.append(rel)
                continue
            _sort_entry(root, rel, olds, leftovers, prunes)
    return leftovers, prunes, shipped


def _sort_entry(
    root: Path, rel: str, olds: Sequence[str], leftovers: list[str], prunes: list[str]
) -> None:
    if any(rel == old or rel.startswith(f"{old}/") for old in olds):
        return
    path = root / rel
    if path.is_dir() and not path.is_symlink():
        if any(old.startswith(f"{rel}/") for old in olds):
            for child in sorted(path.iterdir()):
                _sort_entry(root, f"{rel}/{child.name}", olds, leftovers, prunes)
        elif _holds_content(path):
            leftovers.append(f"{rel}/")
        else:
            prunes.append(rel)
    elif path.name == _FINDER_FILE:
        prunes.append(rel)
    else:
        leftovers.append(rel)


def _holds_content(directory: Path) -> bool:
    """True when *directory* holds anything but Finder files and empty directories."""
    return any(name != _FINDER_FILE for _, _, names in os.walk(directory) for name in names)


def _shipped_readme(file: Path) -> bool:
    try:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError:
        return False
    return digest in _SHIPPED_DATA_README_DIGESTS


def _leftover_lines(leftovers: Sequence[str], data_root: str | None) -> list[str]:
    """The leftovers as comment lines, grouped by the directory they are still in."""
    if not leftovers:
        return []
    lines: list[str] = []
    groups: dict[str, list[str]] = {}
    for item in leftovers:
        container = item.split("/", 1)[0]
        if data_root is not None and item.startswith(f"{data_root}/"):
            container = data_root
        groups.setdefault(container, []).append(item)
    for container, items in groups.items():
        lines.append(f"# Still in {container}/, place by hand:")
        lines.extend(f"#   {item}" for item in items)
    facility = repo_layout.entry("facility").path
    lines.append(f"# Content the render must see goes under {facility}/.")
    return lines


# --- paths ---------------------------------------------------------------------


def successor_of(path: str, moves: Iterable[_Moved]) -> str | None:
    """Where the repo-relative *path* lives once *moves* are done, if one moves it.

    The longest old path equal to *path*, or holding it, wins.

    Args:
        path: A POSIX path relative to the repo root, as it is now.
        moves: The moves, each with ``old`` and ``new`` relative to the root.
    """
    best: _Moved | None = None
    for move in moves:
        if (path == move.old or path.startswith(f"{move.old}/")) and (
            best is None or len(move.old) > len(best.old)
        ):
            best = move
    if best is None:
        return None
    return best.new + path[len(best.old) :]


def source_of(path: str, moves: Iterable[_Moved]) -> str:
    """Where the repo-relative *path*, as it is once *moves* are done, lies before them.

    The longest successor equal to *path*, or holding it, wins: successors
    nest (a lane's curve directory inside the runtime directory), and the
    innermost move is the one that brings *path*.

    Args:
        path: A POSIX path relative to the repo root, once the moves are done.
        moves: The moves, each with ``old`` and ``new`` relative to the root.

    Returns:
        The path before the moves; *path* itself when no move brings it.
    """
    best: _Moved | None = None
    for move in moves:
        if (path == move.new or path.startswith(f"{move.new}/")) and (
            best is None or len(move.new) > len(best.new)
        ):
            best = move
    if best is None:
        return path
    return best.old + path[len(best.new) :]


@contextmanager
def after_moves_view(root: Path, moves: Sequence[_Moved], rels: Iterable[str]) -> Iterator[Path]:
    """A directory holding each of *rels* as it reads once *moves* are done.

    Each file is a link to where it is now: at its old path when a move
    brings it to *rels*, else in place; a file a move takes away is left
    out. Nothing in the repo is touched.

    Args:
        root: The repo root.
        moves: The moves, each with ``old`` and ``new`` relative to *root*.
        rels: Repo-relative paths, as they are once the moves are done.

    Yields:
        The view's root; each of *rels* lies below it at its own path.
    """
    leaving = frozenset((root / move.old).resolve() for move in moves)
    innermost_first = sorted(moves, key=lambda move: -len(move.new))
    with tempfile.TemporaryDirectory(prefix="osprey-retired-") as tmp:
        view = Path(tmp)
        for rel in rels:
            for move in innermost_first:
                if move.new == rel or move.new.startswith(f"{rel}/"):
                    _link_tree(root / move.old, view / move.new, frozenset())
                elif rel.startswith(f"{move.new}/"):
                    _link_tree(root / source_of(rel, [move]), view / rel, frozenset())
            _link_tree(root / rel, view / rel, leaving)
        yield view


def _link_tree(source: Path, target: Path, leaving: frozenset[Path]) -> None:
    """Mirror *source* at *target*: directories made, files linked, *leaving* left out."""
    if not source.exists() or source.resolve() in leaving:
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _link_tree(child, target / child.name, leaving)
    elif not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())


def _after_moves(root: Path, file: Path, table: Iterable[_Moved]) -> Path:
    """*file* at the path it has once the moves are done."""
    relative = file.relative_to(root).as_posix()
    moved = successor_of(relative, table)
    return file if moved is None else root / moved


def _load(file: Path) -> Any:
    try:
        return yaml.safe_load(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None


# --- profile documents -----------------------------------------------------------


def _profile_documents(root: Path) -> list[Path]:
    """``profile.yml``, then each tier document, then each profile variant."""
    documents = [root / repo_layout.entry("profile.yml").path]
    for key in ("tiers", "profiles"):
        directory = root / repo_layout.entry(key).path
        if directory.is_dir():
            documents.extend(sorted(directory.glob("*.yml")))
    return [document for document in documents if document.is_file()]


def _key_matches(path: Sequence[str], patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        parts = pattern.split(".")
        if len(parts) == len(path) and all(
            want in ("*", got) for want, got in zip(parts, path, strict=True)
        ):
            return True
    return False


def _scalars(node: yaml.Node, path: tuple[str, ...]) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Every string scalar under *node* with its dotted key path, ``*`` for an item."""
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode):
                yield from _scalars(value, (*path, *key.value.split(".")))
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _scalars(item, (*path, "*"))
    elif isinstance(node, yaml.ScalarNode) and node.tag == _STR_TAG:
        yield path, node


def _document_edits(
    root: Path,
    document: Path,
    table: Sequence[RootMove],
    edits: list[LineEdit],
    by_hand: list[str],
) -> None:
    """Add the line edits and check-by-hand lines one profile document asks for."""
    try:
        text = document.read_text(encoding="utf-8")
        tree = yaml.compose(text, Loader=yaml.SafeLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    if tree is None:
        return
    lines = text.splitlines()
    where = document.relative_to(root).as_posix()
    spans: dict[int, list[tuple[int, int, str]]] = {}
    for path, node in _scalars(tree, ()):
        value: str = node.value
        dotted = ".".join(path)
        if _key_matches(path, COMMAND_KEYS):
            if _names_retired(value, table):
                by_hand.append(f"{where}:{node.start_mark.line + 1}: {dotted}: {value}")
            continue
        if not _key_matches(path, PATH_KEYS):
            continue
        if _anchored_retired(root, value, table):
            by_hand.append(f"{where}:{node.start_mark.line + 1}: {dotted}: {value}")
            continue
        prefix = "./" if value.startswith("./") else ""
        moved = successor_of(value[len(prefix) :], table)
        if moved is None:
            continue
        start, end = node.start_mark, node.end_mark
        raw = text[start.index : end.index]
        if start.line != end.line or value not in raw:
            by_hand.append(f"{where}:{start.line + 1}: {dotted}: {value}")
            continue
        spans.setdefault(start.line, []).append(
            (start.column, end.column, raw.replace(value, prefix + moved, 1))
        )
    target = _after_moves(root, document, table)
    for line, replacements in sorted(spans.items()):
        old = lines[line]
        new = old
        for begin, finish, replacement in sorted(replacements, reverse=True):
            new = new[:begin] + replacement + new[finish:]
        edits.append(LineEdit(file=target, line=line + 1, old=old, new=new))


def _anchored_retired(root: Path, value: str, table: Sequence[RootMove]) -> bool:
    """True when *value* is absolute or project-root-anchored and names a retired path."""
    if value.startswith(_PROJECT_ROOT_PLACEHOLDER):
        return successor_of(value[len(_PROJECT_ROOT_PLACEHOLDER) :], table) is not None
    if not os.path.isabs(value):
        return False
    for base in (root, Path(os.path.abspath(root))):
        try:
            relative = Path(value).relative_to(base).as_posix()
        except ValueError:
            continue
        if successor_of(relative, table) is not None:
            return True
    return False


# --- root text files ---------------------------------------------------------------


def _token_pattern(table: Sequence[RootMove]) -> re.Pattern[str] | None:
    """A path token naming a retired path in free text, longest first.

    A bare directory name counts only before a ``/``; a name that is plainly a
    path (it holds a ``/`` or a ``.``) counts on its own too.
    """
    alternatives = []
    for move in sorted(table, key=lambda move: len(move.old), reverse=True):
        old = re.escape(move.old)
        if "/" in move.old or "." in move.old:
            alternatives.append(rf"{old}(?=/|[\"'\s,\]}})]|$)")
        else:
            alternatives.append(rf"{old}(?=/)")
    if not alternatives:
        return None
    return re.compile(rf"(?<![\w.$/-])(?P<dot>\./)?(?P<old>{'|'.join(alternatives)})")


def _names_retired(text: str, table: Sequence[RootMove]) -> bool:
    pattern = _token_pattern(table)
    return pattern is not None and pattern.search(text) is not None


def _pipeline_edits(root: Path, table: Sequence[RootMove]) -> list[LineEdit]:
    """The root pipeline file's lines that name a retired path, rewritten."""
    file = root / repo_layout.entry(_PIPELINE_KEY).path
    pattern = _token_pattern(table)
    if pattern is None or not file.is_file():
        return []
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    target = _after_moves(root, file, table)
    edits = []
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        new = pattern.sub(
            lambda match: (
                (match["dot"] or "") + (successor_of(match["old"], table) or match["old"])
            ),
            line,
        )
        if new != line:
            edits.append(LineEdit(file=target, line=number, old=line, new=new))
    return edits


def _dockerignore_edits(root: Path, table: Sequence[RootMove]) -> list[LineEdit]:
    """The root ``.dockerignore`` patterns anchored on a retired path, rewritten."""
    file = root / repo_layout.entry(_DOCKERIGNORE_KEY).path
    if not file.is_file():
        return []
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    target = _after_moves(root, file, table)
    edits = []
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(
            r"(?P<lead>\s*!?(?:\./|/)?)(?P<body>[^\s#][^\s]*?)(?P<tail>/?\s*)", line
        )
        if match is None:
            continue
        moved = successor_of(match["body"], table)
        if moved is None:
            continue
        new = match["lead"] + moved + match["tail"]
        edits.append(LineEdit(file=target, line=number, old=line, new=new))
    return edits


# --- deletes, host steps, check by hand ----------------------------------------------


def _canopy_deletes(root: Path) -> tuple[str, ...]:
    """The canopy payload and the two canopy wheels left at the repo root."""
    from osprey.integrations.canopy import PAYLOAD_DIRNAME
    from osprey.integrations.canopy.stage import WHEEL_PREFIXES

    found = []
    if (root / PAYLOAD_DIRNAME).exists():
        found.append(PAYLOAD_DIRNAME)
    found.extend(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_file()
            and item.name.endswith(".whl")
            and item.name.startswith(WHEEL_PREFIXES)
        )
    )
    return tuple(found)


def _scaffolded(root: Path, new: str, marker: str) -> bool:
    """True when a retired location of *new* holds a file the scaffolder wrote as *marker*."""
    from osprey.cli.deploy_scaffold import _marker_of

    for old in repo_layout.former_locations(new):
        file = root / old
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _marker_of(text) == marker:
            return True
    return False


def _host_steps(root: Path) -> tuple[str, ...]:
    """``osprey build``, then re-installing a moved generated unit or boot hook."""
    steps = ["osprey build"]
    unit = repo_layout.entry("osprey.service").path
    hook = repo_layout.child_display_path("scripts", "boot-hook")
    unit_moved = bool(repo_layout.former_locations(unit))
    hook_moved = bool(repo_layout.former_locations(hook))
    if not (unit_moved or hook_moved):
        return tuple(steps)
    from osprey.cli.deploy_scaffold_templates import (
        BOOT_HOOK_MARKER,
        SYSTEMD_MARKER,
        boot_hook_crontab_lines,
    )

    if unit_moved and _scaffolded(root, unit, SYSTEMD_MARKER):
        steps.extend(
            (
                f"cp {shlex.quote(str(root / unit))} ~/.config/systemd/user/",
                "systemctl --user daemon-reload",
                f"systemctl --user enable --now {Path(unit).name}",
            )
        )
    if hook_moved and _scaffolded(root, hook, BOOT_HOOK_MARKER):
        steps.append("crontab -e, replacing the osprey-boot-hook lines with:")
        steps.extend(boot_hook_crontab_lines(str(root / hook)))
    return tuple(steps)


def _moved_inner_paths(root: Path, table: Sequence[RootMove]) -> list[str]:
    """The moved files whose inner paths no edit rewrites: CI extra, Dockerfiles, scripts."""
    found: list[str] = []
    for key in _INNER_PATHS_UNCHANGED:
        new = repo_layout.entry(key).path
        for move in table:
            if move.new != new or not (root / move.old).exists():
                continue
            if key == "services":
                found.extend(
                    f"{new}/{dockerfile.parent.name}/Dockerfile"
                    for dockerfile in sorted((root / move.old).glob("*/Dockerfile"))
                )
            elif key == "scripts":
                found.append(f"{new}/ (operator scripts)")
            else:
                found.append(new)
    return found
