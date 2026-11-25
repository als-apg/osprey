"""What a facility tree still holds at its retired paths, and how it moves.

:func:`find_retired` looks at the filesystem only: it reports each retired path
that still holds a source (:class:`Move`), each same-named file at a retired
path and its successor (:class:`Clash`), and each authored ``deck:`` value that
still names the retired decks directory (:class:`LineEdit`). The retired paths
come from :func:`osprey_facility.source_tree.retired_moves`; nothing here names
a path of its own. It neither runs git nor prints, so a refusal and a migration
can both act on one result.

:func:`move_script` turns a result into POSIX shell lines that carry out the
moves from any working directory, asking git which repository holds each root
(:func:`repository_root`) and whether each source is tracked; without git, or
outside a work tree, a plain ``mv`` does it. :func:`refuse_retired` stops a
read of a tree that still holds anything at a retired path.

A retired path counts only when it is a file, or a directory holding at least
one file whose name, and every directory name between it and the retired
path, does not start with ``.``; an empty leftover directory is not a retired
layout.

The placement and the script assume nothing about the facility tree, so any
tree with paths that moved uses them: :func:`place_moves` finds what a list of
``(old, new)`` paths under one root asks to move or clashes, and
:func:`render_script` renders moves, line edits and deletions as one script.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from osprey_facility import source_tree
from osprey_facility.errors import RetiredLayoutRefused

__all__ = [
    "Clash",
    "LineEdit",
    "Move",
    "RetiredLayout",
    "deck_line_edits",
    "find_retired",
    "move_script",
    "place_moves",
    "refuse_retired",
    "render_script",
    "repository_root",
]

#: The tag PyYAML resolves a plain or quoted string scalar to.
_STR_TAG = "tag:yaml.org,2002:str"


@dataclass(frozen=True)
class Move:
    """One source to move from a retired path to its successor.

    Attributes:
        root: The root both paths are relative to.
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
class LineEdit:
    """One text replacement on one line of a file.

    Attributes:
        file: The file, at the path where the edit applies once the moves are
            done.
        line: The line in that file, 1-based.
        old: The text as written; the edit replaces its first occurrence on
            the line.
        new: The text that replaces it.
    """

    file: Path
    line: int
    old: str
    new: str


@dataclass(frozen=True)
class Clash:
    """A path a move cannot reach because something is already there.

    Either the same file exists at the retired path and at its successor, or a
    file stands where the move needs a directory.

    Attributes:
        old: The file, or directory, at the retired path.
        new: What already stands at the successor side.
    """

    old: Path
    new: Path


@dataclass(frozen=True)
class RetiredLayout:
    """Everything a facility tree still holds in its retired layout.

    Attributes:
        moves: The sources to move.
        deck_edits: Each authored ``deck:`` value under the retired decks
            directory: *old* is the value as written, *new* the same deck
            under the successor decks directory, and *file* the authored
            models file at its successor path.
        clashes: What blocks a move.
    """

    moves: tuple[Move, ...]
    deck_edits: tuple[LineEdit, ...]
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
        placed, blocked = place_moves(root, [(old, new)])
        moves.extend(placed)
        clashes.extend(blocked)
    return RetiredLayout(
        moves=tuple(moves),
        deck_edits=_deck_edits(machine_dir),
        clashes=tuple(clashes),
    )


def place_moves(
    root: Path, moves: Iterable[tuple[str, str]], *, hidden: bool = False
) -> tuple[list[Move], list[Clash]]:
    """How each retired path under *root* reaches its successor, and what blocks it.

    A retired path counts only when it holds a source: it is a file, or a
    directory holding at least one file. Unless *hidden* is true, a file or
    directory below the retired path whose name starts with ``.`` is not a
    source; the retired path itself counts whatever its name.

    When nothing stands at the successor, the whole path moves. When the
    successor is a directory and the retired path is one too, each source
    moves on its own into it, and one already present there is a clash. A
    file onto an existing path is a clash, and so is a file where the move
    needs a directory: a successor, or a directory above it, that is a file.

    Args:
        root: The root every path is relative to.
        moves: ``(old, new)`` pairs, POSIX paths relative to *root*, in the
            order their moves are reported.
        hidden: Count names starting with ``.`` below each retired path.

    Returns:
        The moves, in the order of *moves* and per-file moves of one
        directory in path order, and the clashes.
    """
    placed: list[Move] = []
    clashes: list[Clash] = []
    for old, new in moves:
        _place(root, old, new, hidden, placed, clashes)
    return placed, clashes


def _place(
    root: Path, old: str, new: str, hidden: bool, moves: list[Move], clashes: list[Clash]
) -> None:
    """Record how retired path *old* under *root* reaches *new*, if it holds a source."""
    source = root / old
    files = sorted(_sources(source, hidden))
    if not files:
        return
    successor = root / new
    is_dir = source.is_dir()
    blocker = _file_above(root, new)
    if blocker is not None:
        clashes.append(Clash(old=source, new=blocker))
        return
    if not successor.exists():
        moves.append(Move(root=root, old=old, new=new, is_dir=is_dir, merge=False))
        return
    if not (is_dir and successor.is_dir()):
        clashes.append(Clash(old=source, new=successor))
        return
    blocked: set[str] = set()
    for relative in files:
        if any(relative.startswith(f"{prefix}/") for prefix in blocked):
            continue
        blocker = _file_above(successor, relative)
        if blocker is not None:
            prefix = blocker.relative_to(successor).as_posix()
            blocked.add(prefix)
            clashes.append(Clash(old=source / prefix, new=blocker))
        elif (successor / relative).exists():
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


def _file_above(base: Path, relative: str) -> Path | None:
    """The first path strictly above *base* / *relative*, below *base*, that is not a directory."""
    current = base
    for part in Path(relative).parts[:-1]:
        current = current / part
        if not current.exists():
            return None
        if not current.is_dir():
            return current
    return None


def _sources(source: Path, hidden: bool) -> Iterator[str]:
    """Every file at *source* that counts as a source, relative to it.

    A file at *source* itself yields ``""``. Below *source*, names starting
    with ``.`` are skipped unless *hidden* is true.
    """
    if source.is_file():
        yield ""
        return
    if not source.is_dir():
        return
    for directory, subdirs, names in os.walk(source):
        if not hidden:
            subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        relative = Path(directory).relative_to(source)
        for name in names:
            if (hidden or not name.startswith(".")) and (Path(directory) / name).is_file():
                yield (relative / name).as_posix()


def _deck_edits(machine_dir: Path) -> tuple[LineEdit, ...]:
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
    edits: list[LineEdit] = []
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
                    LineEdit(
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

    One block per repository, as :func:`render_script` lays it out, with the
    deck edits as :func:`deck_line_edits` gives them. Clashes get no command.
    """
    return render_script(layout.moves, deck_line_edits(layout), ())


def deck_line_edits(layout: RetiredLayout) -> list[tuple[Path, LineEdit]]:
    """*layout*'s deck edits as :func:`render_script` takes them.

    Each is paired with its facility root and rewrites the retired decks
    prefix of its value on its line of the moved models file.
    """
    if not layout.deck_edits:
        return []
    depth = len(Path(source_tree.path("models")).parts)
    old_prefix, new_prefix = source_tree.retired_deck_prefix()
    return [
        (
            edit.file.parents[depth - 1],
            LineEdit(file=edit.file, line=edit.line, old=old_prefix, new=new_prefix),
        )
        for edit in layout.deck_edits
    ]


def render_script(
    moves: Iterable[Move],
    line_edits: Iterable[tuple[Path, LineEdit]],
    deletes: Iterable[tuple[Path, str]],
) -> list[str]:
    """POSIX shell lines that carry out *moves*, then *line_edits*, then *deletes*.

    Every command runs in the block of the repository holding its root: a
    move's own root, the root paired with each line edit, and the root paired
    with each deleted path, which is POSIX and relative to that root. A block
    opens with ``cd`` to the repository's absolute top (the root itself when
    no git work tree holds it), then ``mkdir -p`` for every missing successor
    parent, ``git mv`` for each tracked source and ``mv`` for the rest, one
    ``perl -pi`` per line edit, ``git rm -r`` for each tracked deleted path and
    ``rm -r`` for the rest, and last ``rmdir`` of every directory the moves and
    deletions may leave empty, deepest first. Blocks follow the order their
    roots first appear. Every path is shell-quoted and relative to the
    block's ``cd``, so the lines run from any working directory.

    Args:
        moves: The moves.
        line_edits: ``(root, edit)`` pairs; the edit replaces the first
            occurrence of *old* on its line with *new*, both taken literally.
        deletes: ``(root, path)`` pairs naming what to remove.

    Returns:
        The script lines, empty when there is nothing to do.
    """
    blocks: dict[Path, _Block] = {}

    def block_for(root: Path) -> _Block:
        resolved = root.resolve()
        top = repository_root(resolved)
        key = top if top is not None else resolved
        if key not in blocks:
            blocks[key] = _Block(top=key, in_git=top is not None)
        return blocks[key]

    for move in moves:
        block_for(move.root).add_move(move)
    for root, edit in line_edits:
        block_for(root).add_line_edit(edit)
    for root, relative in deletes:
        block_for(root).add_delete(root, relative)
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
    deletes: list[str] = field(default_factory=list)
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

    def add_line_edit(self, edit: LineEdit) -> None:
        expression = f"s{{{_perl_pattern(edit.old)}}}{{{_perl_replacement(edit.new)}}}"
        expression += f" if $. == {edit.line}"
        target = _relative(edit.file.resolve(), self.top)
        self.edits.append(f"perl -pi -e {shlex.quote(expression)} {shlex.quote(target)}")

    def add_delete(self, root: Path, relative: str) -> None:
        resolved = root.resolve()
        target = _relative(resolved / relative, self.top)
        verb = "git rm -r -q" if self.in_git and self._tracked(target) else "rm -r"
        self.deletes.append(f"{verb} -- {shlex.quote(target)}")
        for directory in (resolved / relative).parents:
            if resolved in directory.parents:
                _append_once(self.rmdirs, _relative(directory, self.top))

    def render(self) -> list[str]:
        lines = [f"cd {shlex.quote(str(self.top))}"]
        if self.mkdirs:
            lines.append("mkdir -p " + " ".join(shlex.quote(path) for path in self.mkdirs))
        lines.extend(self.moves)
        lines.extend(self.edits)
        lines.extend(self.deletes)
        deepest_first = sorted(self.rmdirs, key=lambda path: path.count("/"), reverse=True)
        lines.extend(f"rmdir {shlex.quote(path)} 2>/dev/null || true" for path in deepest_first)
        return lines

    def _tracked(self, relative: str) -> bool:
        try:
            result = _git(self.top, "ls-files", "--error-unmatch", "--", relative)
        except OSError:
            return False
        return result.returncode == 0


#: Characters a Perl pattern inside ``s{...}{...}`` reads as other than themselves.
_PERL_PATTERN_SPECIAL = frozenset("\\^$.|?*+()[]{}@")

#: Characters a Perl replacement inside ``s{...}{...}`` reads as other than themselves.
_PERL_REPLACEMENT_SPECIAL = frozenset("\\$@{}")


def _perl_pattern(text: str) -> str:
    """*text* as a Perl pattern that matches it literally."""
    return "".join(f"\\{char}" if char in _PERL_PATTERN_SPECIAL else char for char in text)


def _perl_replacement(text: str) -> str:
    """*text* as a Perl replacement that inserts it literally."""
    return "".join(f"\\{char}" if char in _PERL_REPLACEMENT_SPECIAL else char for char in text)


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
