"""What a facility tree still holds at its retired paths, and how it moves.

:func:`find_retired` looks at the filesystem only: it reports each retired path
that still holds a source (:class:`Move`), each same-named file at a retired
path and its successor (:class:`Clash`), and each authored ``deck:`` value that
still names the retired decks directory (:class:`DeckEdit`). The retired paths
come from :func:`osprey_facility.source_tree.retired_moves`; nothing here names
a path of its own. It neither runs git nor prints, so a refusal and a migration
can both act on one result.

:func:`move_script` turns a result into POSIX shell lines that carry out the
moves from any working directory, asking git which repository holds each root
(:func:`repository_root`) and whether each source is tracked; without git, or
outside a work tree, a plain ``mv`` does it. :func:`refuse_retired` stops a
read of a tree that still holds anything at a retired path.

A retired path counts only when it holds at least one file whose name, and
every directory name below the retired path, does not start with ``.``; an
empty leftover directory is not a retired layout.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from osprey_facility import source_tree
from osprey_facility.errors import RetiredLayoutRefused

__all__ = [
    "Clash",
    "DeckEdit",
    "Move",
    "RetiredLayout",
    "find_retired",
    "move_script",
    "refuse_retired",
    "repository_root",
]

#: The tag PyYAML resolves a plain or quoted string scalar to.
_STR_TAG = "tag:yaml.org,2002:str"


@dataclass(frozen=True)
class Move:
    """One source to move from a retired path to its successor.

    Attributes:
        root: The facility root both paths are relative to.
        old: The retired path, POSIX, relative to *root*.
        new: Where it goes, POSIX, relative to *root*.
        is_dir: True when a whole directory moves.
        merge: True when the successor already exists, so the move is one of
            several per-file moves into it.
    """

    root: Path
    old: str
    new: str
    is_dir: bool
    merge: bool

    def vacated(self) -> list[Path]:
        """Each directory above the source and below the root, deepest first, resolved.

        The move may leave any of them empty.
        """
        root = self.root.resolve()
        return [directory for directory in (root / self.old).parents if root in directory.parents]


@dataclass(frozen=True)
class DeckEdit:
    """One authored ``deck:`` value that names the retired decks directory.

    Attributes:
        file: The authored models file at its successor path, where the edit
            applies once the moves are done.
        line: The value's line in that file, 1-based.
        old: The value as written.
        new: The same deck under the successor decks directory.
    """

    file: Path
    line: int
    old: str
    new: str


@dataclass(frozen=True)
class Clash:
    """A file present both at a retired path and at its successor.

    Attributes:
        old: The file at the retired path.
        new: The file at the successor path.
    """

    old: Path
    new: Path


@dataclass(frozen=True)
class RetiredLayout:
    """Everything a facility tree still holds in its retired layout."""

    moves: tuple[Move, ...]
    deck_edits: tuple[DeckEdit, ...]
    clashes: tuple[Clash, ...]

    @property
    def empty(self) -> bool:
        """True when the tree holds nothing at a retired path."""
        return not (self.moves or self.deck_edits or self.clashes)


def find_retired(machine_dir: Path, content_dir: Path | None = None) -> RetiredLayout:
    """What the facility tree at *machine_dir* still holds at retired paths.

    Retired paths of machine facts are looked for under *machine_dir*; those
    of deployment content under *content_dir*, or under *machine_dir* when no
    content root is given. Moves follow the declaration order, per-file moves
    of one directory in path order.
    """
    moves: list[Move] = []
    clashes: list[Clash] = []
    for old, new, content in source_tree.retired_moves():
        root = content_dir if content and content_dir is not None else machine_dir
        _place(root, old, new, moves, clashes)
    return RetiredLayout(
        moves=tuple(moves),
        deck_edits=_deck_edits(machine_dir),
        clashes=tuple(clashes),
    )


def _place(root: Path, old: str, new: str, moves: list[Move], clashes: list[Clash]) -> None:
    """Record how retired path *old* under *root* reaches *new*, if it holds a source."""
    source = root / old
    files = sorted(_sources(source))
    if not files:
        return
    successor = root / new
    is_dir = source.is_dir()
    if not successor.exists():
        moves.append(Move(root=root, old=old, new=new, is_dir=is_dir, merge=False))
        return
    if not is_dir:
        clashes.append(Clash(old=source, new=successor))
        return
    for relative in files:
        if (successor / relative).exists():
            clashes.append(Clash(old=source / relative, new=successor / relative))
        else:
            moves.append(
                Move(
                    root=root,
                    old=f"{old}/{relative}",
                    new=f"{new}/{relative}",
                    is_dir=False,
                    merge=True,
                )
            )


def _sources(source: Path) -> Iterator[str]:
    """Every file at *source* that counts as a source, relative to it.

    A file yields ``""`` when its name does not start with ``.``.
    """
    if source.is_file():
        if not source.name.startswith("."):
            yield ""
        return
    if not source.is_dir():
        return
    for directory, subdirs, names in os.walk(source):
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        relative = Path(directory).relative_to(source)
        for name in names:
            if not name.startswith(".") and (Path(directory) / name).is_file():
                yield (relative / name).as_posix()


def _deck_edits(machine_dir: Path) -> tuple[DeckEdit, ...]:
    """Each top-level model's ``deck:`` value that still names the retired decks directory.

    The authored models file is read at its successor path when that exists,
    else at its retired path; the edits name the successor path either way.
    A file that is not a YAML sequence yields nothing here; reading it reports
    what is wrong.
    """
    new_file = machine_dir / source_tree.path("models")
    candidates = [new_file] + [machine_dir / old for old in source_tree.slot("models").retired]
    authored = next((candidate for candidate in candidates if candidate.is_file()), None)
    if authored is None:
        return ()
    try:
        document = yaml.compose(authored.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    except (yaml.YAMLError, UnicodeDecodeError):
        return ()
    if not isinstance(document, yaml.SequenceNode):
        return ()
    old_prefix, new_prefix = source_tree.retired_deck_prefix()
    edits: list[DeckEdit] = []
    for entry in document.value:
        if not isinstance(entry, yaml.MappingNode):
            continue
        for key, value in entry.value:
            if not (isinstance(key, yaml.ScalarNode) and key.value == "deck"):
                continue
            if not (isinstance(value, yaml.ScalarNode) and value.tag == _STR_TAG):
                continue
            if value.value.startswith(old_prefix):
                edits.append(
                    DeckEdit(
                        file=new_file,
                        line=value.start_mark.line + 1,
                        old=value.value,
                        new=new_prefix + value.value[len(old_prefix) :],
                    )
                )
    return tuple(edits)


def refuse_retired(machine_dir: Path, content_dir: Path | None = None) -> None:
    """Stop when the facility tree still holds anything at a retired path.

    Args:
        machine_dir: The machine root.
        content_dir: The content root when mounted; ``None`` means
            *machine_dir*.

    Raises:
        RetiredLayoutRefused: :func:`find_retired` reports a move, a deck
            edit or a clash; its message carries :func:`move_script`.
    """
    layout = find_retired(machine_dir, content_dir)
    if layout.empty:
        return
    raise RetiredLayoutRefused(
        machine_dir.resolve(),
        None if content_dir is None else content_dir.resolve(),
        layout,
        move_script(layout),
    )


def repository_root(directory: Path) -> Path | None:
    """The top of the git work tree holding *directory*, or ``None``.

    ``None`` also when git is not installed or *directory* does not exist.
    """
    try:
        result = _git(directory, "rev-parse", "--show-toplevel")
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def move_script(layout: RetiredLayout) -> list[str]:
    """POSIX shell lines that carry out *layout*'s moves and deck edits.

    One block per repository: ``cd`` to its absolute top (the root itself when
    no git work tree holds it), ``mkdir -p`` for every missing successor
    parent, ``git mv`` for each tracked source and ``mv`` for the rest, then
    the deck edits on the moved models file, then ``rmdir`` of every retired
    directory the moves leave empty. Every path is shell-quoted and relative
    to the block's ``cd``, so the lines run from any working directory.
    Clashes get no command.
    """
    blocks: dict[Path, _Block] = {}

    def block_for(root: Path) -> _Block:
        resolved = root.resolve()
        top = repository_root(resolved)
        key = top if top is not None else resolved
        if key not in blocks:
            blocks[key] = _Block(top=key, in_git=top is not None)
        return blocks[key]

    for move in layout.moves:
        block_for(move.root).add_move(move)
    depth = len(Path(source_tree.path("models")).parts)
    for edit in layout.deck_edits:
        block_for(edit.file.parents[depth - 1]).add_deck_edit(edit)
    lines: list[str] = []
    for block in blocks.values():
        lines.extend(block.render())
    return lines


@dataclass
class _Block:
    """The commands for one repository, gathered before they are rendered."""

    top: Path
    in_git: bool
    mkdirs: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    rmdirs: list[str] = field(default_factory=list)

    def add_move(self, move: Move) -> None:
        root = move.root.resolve()
        old = _relative(root / move.old, self.top)
        new = _relative(root / move.new, self.top)
        parent = (root / move.new).parent
        if not parent.exists():
            _append_once(self.mkdirs, _relative(parent, self.top))
        verb = "git mv" if self.in_git and self._tracked(old) else "mv"
        self.moves.append(f"{verb} {shlex.quote(old)} {shlex.quote(new)}")
        for directory in move.vacated():
            _append_once(self.rmdirs, _relative(directory, self.top))

    def add_deck_edit(self, edit: DeckEdit) -> None:
        old_prefix, new_prefix = source_tree.retired_deck_prefix()
        expression = f"s{{{old_prefix}}}{{{new_prefix}}} if $. == {edit.line}"
        target = _relative(edit.file.resolve(), self.top)
        self.edits.append(f"perl -pi -e {shlex.quote(expression)} {shlex.quote(target)}")

    def render(self) -> list[str]:
        lines = [f"cd {shlex.quote(str(self.top))}"]
        if self.mkdirs:
            lines.append("mkdir -p " + " ".join(shlex.quote(path) for path in self.mkdirs))
        lines.extend(self.moves)
        lines.extend(self.edits)
        deepest_first = sorted(self.rmdirs, key=lambda path: path.count("/"), reverse=True)
        lines.extend(f"rmdir {shlex.quote(path)} 2>/dev/null || true" for path in deepest_first)
        return lines

    def _tracked(self, relative: str) -> bool:
        try:
            result = _git(self.top, "ls-files", "--error-unmatch", "--", relative)
        except OSError:
            return False
        return result.returncode == 0


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git against *directory*, capturing its output."""
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _relative(path: Path, top: Path) -> str:
    """*path* relative to *top*, POSIX; ``.`` for *top* itself."""
    return path.relative_to(top).as_posix()


def _append_once(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)
