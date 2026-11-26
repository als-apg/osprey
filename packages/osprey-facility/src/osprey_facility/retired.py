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
Moves are placed and rendered in stages: each one against the tree as the
moves before it leave it, so a move into a directory an earlier move brings
merges file by file, and its ``git mv``-or-``mv`` decision is made where its
source lies before any move.
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
    "LineDelete",
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
class LineDelete:
    """One whole line to remove from a file.

    Attributes:
        file: The file, at the path where the delete applies once the moves
            are done.
        line: The line in that file, 1-based, counted before any delete.
        text: The line as written, without its line ending; a line that no
            longer reads so is kept.
    """

    file: Path
    line: int
    text: str


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

    Each move is placed against the tree as the moves placed before it leave
    it. A move whose successor lies inside another move's successor is placed
    after that one, so it merges into what that move brings. Two moves to the
    same successor that both hold a source clash with each other.

    When nothing stands at the successor, the whole path moves. When the
    successor is a directory and the retired path is one too, each source
    moves on its own into it, and one already present there is a clash. A
    file onto an existing path is a clash, and so is a file where the move
    needs a directory: a successor, or a directory above it, that is a file.
    A clash names each side where it lies before any move.

    Args:
        root: The root every path is relative to.
        moves: ``(old, new)`` pairs, POSIX paths relative to *root*, in the
            order their moves are reported.
        hidden: Count names starting with ``.`` below each retired path.

    Returns:
        The moves in the order they run, per-file moves of one directory in
        path order, and the clashes.
    """
    pairs = list(moves)
    ordered = sorted(pairs, key=lambda pair: sum(_inside(pair[1], other[1]) for other in pairs))
    placed: list[Move] = []
    clashes: list[Clash] = []
    overlay = _Overlay(root, placed)
    claimed: dict[str, str] = {}
    for old, new in ordered:
        if next(_sources(root / old, hidden), None) is None:
            continue
        if new in claimed:
            clashes.append(Clash(old=root / old, new=root / claimed[new]))
            continue
        claimed[new] = old
        _place(root, old, new, hidden, overlay, placed, clashes)
    return placed, clashes


def _inside(path: str, directory: str) -> bool:
    """True when the POSIX *path* lies strictly below *directory*."""
    return path.startswith(f"{directory}/")


def _join(directory: str, relative: str) -> str:
    """*relative* below *directory*, POSIX; either may be ``""``."""
    return "/".join(part for part in (directory, relative) if part)


@dataclass
class _Overlay:
    """The tree under *root* as the moves in *placed* leave it, read without touching it."""

    root: Path
    placed: list[Move]

    def vacated(self, rel: str) -> bool:
        return any(rel == move.old or _inside(rel, move.old) for move in self.placed)

    def now(self, rel: str) -> Path:
        """Where what the tree holds at *rel* after the moves lies before them."""
        best: Move | None = None
        for move in self.placed:
            if (rel == move.new or _inside(rel, move.new)) and (
                best is None or len(move.new) > len(best.new)
            ):
                best = move
        if best is None:
            return self.root / rel
        return self.root / (best.old + rel[len(best.new) :])

    def is_file(self, rel: str) -> bool:
        if self.vacated(rel):
            return False
        return self.now(rel).is_file() or (self.root / rel).is_file()

    def exists(self, rel: str) -> bool:
        if (self.root / rel).exists() and not self.vacated(rel):
            return True
        return any(
            (self.root / move.old).exists()
            if move.new == rel or _inside(move.new, rel)
            else _inside(rel, move.new) and self.now(rel).exists()
            for move in self.placed
        )

    def file_above(self, rel: str) -> str | None:
        """The first directory *rel* needs, below the root, that is a file."""
        current = ""
        for part in Path(rel).parts[:-1]:
            current = _join(current, part)
            if not self.exists(current):
                return None
            if self.is_file(current):
                return current
        return None

    def files(self, rel: str) -> set[str]:
        """Every file at *rel*, relative to it; ``""`` when *rel* is a file."""
        found = {
            item for item in _sources(self.root / rel, True) if not self.vacated(_join(rel, item))
        }
        for move in self.placed:
            if move.new == rel or _inside(move.new, rel):
                sub = move.new[len(rel) + 1 :] if move.new != rel else ""
                found.update(_join(sub, item) for item in _sources(self.root / move.old, True))
            elif _inside(rel, move.new):
                inner = rel[len(move.new) + 1 :]
                found.update(_sources(self.root / move.old / inner, True))
        return found


def _place(
    root: Path,
    old: str,
    new: str,
    hidden: bool,
    overlay: _Overlay,
    moves: list[Move],
    clashes: list[Clash],
) -> None:
    """Record how retired path *old* under *root* reaches *new*, if it holds a source."""
    source = root / old
    files = sorted(_sources(source, hidden))
    if not files:
        return
    is_dir = source.is_dir()
    blocker = overlay.file_above(new)
    if blocker is not None:
        clashes.append(Clash(old=source, new=overlay.now(blocker)))
        return
    if not overlay.exists(new):
        moves.append(Move(root=root, old=old, new=new, is_dir=is_dir, merge=False))
        return
    if not (is_dir and not overlay.is_file(new)):
        clashes.append(Clash(old=source, new=overlay.now(new)))
        return
    present = overlay.files(new)
    directories = {
        "/".join(item.split("/")[:depth])
        for item in present
        for depth in range(1, item.count("/") + 1)
    }
    blocked: set[str] = set()
    for relative in files:
        if any(_inside(relative, prefix) for prefix in blocked):
            continue
        parts = relative.split("/")
        above = next(
            (
                "/".join(parts[:depth])
                for depth in range(1, len(parts))
                if "/".join(parts[:depth]) in present
            ),
            None,
        )
        if above is not None:
            blocked.add(above)
            clashes.append(Clash(old=source / above, new=overlay.now(_join(new, above))))
        elif relative in present or relative in directories:
            clashes.append(Clash(old=source / relative, new=overlay.now(_join(new, relative))))
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


def refuse_retired(
    machine_dir: Path, content_dir: Path | None = None, *, repo: Path | None = None
) -> None:
    """Stop when the facility tree still holds anything at a retired path.

    Args:
        machine_dir: The machine root.
        content_dir: The content root when mounted; ``None`` means
            *machine_dir*.
        repo: The deployment repo the refusal's closing command migrates;
            ``None`` means the one :func:`_deployment_of` finds.

    Raises:
        RetiredLayoutRefused: :func:`find_retired` reports a move, a deck
            edit or a clash; its message carries :func:`move_script`.
    """
    layout = find_retired(machine_dir, content_dir)
    if layout.empty:
        return
    content = None if content_dir is None else content_dir.resolve()
    raise RetiredLayoutRefused(
        machine_dir.resolve(),
        content,
        layout,
        move_script(layout),
        (repo if repo is not None else _deployment_of(content or machine_dir)).resolve(),
    )


def _deployment_of(content_dir: Path) -> Path:
    """The deployment repo a facility content root belongs to.

    The top of the git work tree holding it; without one, the directory
    holding it, since a deployment keeps its facility content in one
    directory at its root.
    """
    top = repository_root(content_dir)
    return top if top is not None else content_dir.resolve().parent


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
    line_deletes: Iterable[tuple[Path, LineDelete]] = (),
) -> list[str]:
    """POSIX shell lines that carry out *moves*, then the line edits and line deletes, then *deletes*.

    Every command runs in the block of the repository holding its root: a
    move's own root, the root paired with each line edit or line delete, and
    the root paired with each deleted path, which is POSIX and relative to
    that root. A root that does not exist yet (an earlier move brings it)
    belongs to the repository of its nearest existing directory. A block
    opens with ``cd`` to the repository's absolute top (the root itself when
    no git work tree holds it).

    The moves run in the order given, in stages: a move may read the tree
    the moves before it leave. Each gets its own ``mkdir -p`` of its
    successor's parent immediately before it, unless that directory is
    already there; it is ``git mv`` when its source, followed back through
    the earlier moves of its block to where it lies before any move, is
    tracked, else ``mv``. Then one ``perl -pi`` per line edit, one
    ``perl -ni`` per line delete (the last lines of a file first, so the line
    numbers hold), ``git rm -r`` for each tracked deleted path and ``rm -r``
    for the rest, and last ``rmdir`` of every directory the moves and
    deletions may leave empty, deepest first. Blocks follow the order their
    roots first appear. Every path is shell-quoted and relative to the
    block's ``cd``, so the lines run from any working directory.

    Args:
        moves: The moves, in the order they run.
        line_edits: ``(root, edit)`` pairs; the edit replaces the first
            occurrence of *old* on its line with *new*, both taken literally.
        deletes: ``(root, path)`` pairs naming what to remove.
        line_deletes: ``(root, delete)`` pairs naming whole lines to remove.

    Returns:
        The script lines, empty when there is nothing to do.
    """
    blocks: dict[Path, _Block] = {}

    def block_for(root: Path) -> _Block:
        resolved = root.resolve()
        existing = next(
            (directory for directory in (resolved, *resolved.parents) if directory.exists()),
            resolved,
        )
        top = repository_root(existing)
        key = top if top is not None else resolved
        if key not in blocks:
            blocks[key] = _Block(top=key, in_git=top is not None)
        return blocks[key]

    for move in moves:
        block_for(move.root).add_move(move)
    for root, edit in line_edits:
        block_for(root).add_line_edit(edit)
    for root, line_delete in sorted(
        line_deletes, key=lambda pair: (str(pair[1].file), -pair[1].line)
    ):
        block_for(root).add_line_delete(line_delete)
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
    moves: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    rmdirs: list[str] = field(default_factory=list)
    made: list[Path] = field(default_factory=list)
    moved: list[tuple[Path, Path]] = field(default_factory=list)

    def add_move(self, move: Move) -> None:
        root = move.root.resolve()
        source = root / move.old
        target = root / move.new
        parent = target.parent
        if parent != self.top and parent not in self.made and not self._standing(parent):
            self.made.append(parent)
            self.moves.append(f"mkdir -p {shlex.quote(_relative(parent, self.top))}")
        before = self._before(source)
        tracked = self.in_git and self._tracked(_relative(before, self.top))
        verb = "git mv" if tracked else "mv"
        old = _relative(source, self.top)
        new = _relative(target, self.top)
        self.moves.append(f"{verb} {shlex.quote(old)} {shlex.quote(new)}")
        self.moved.append((source, target))
        for directory in move.vacated():
            _append_once(self.rmdirs, _relative(directory, self.top))

    def add_line_edit(self, edit: LineEdit) -> None:
        expression = f"s{{{_perl_pattern(edit.old)}}}{{{_perl_replacement(edit.new)}}}"
        expression += f" if $. == {edit.line}"
        target = _relative(edit.file.resolve(), self.top)
        self.edits.append(f"perl -pi -e {shlex.quote(expression)} {shlex.quote(target)}")

    def add_line_delete(self, delete: LineDelete) -> None:
        expression = f"print unless $. == {delete.line} && m{{^{_perl_pattern(delete.text)}$}}"
        target = _relative(delete.file.resolve(), self.top)
        self.edits.append(f"perl -ni -e {shlex.quote(expression)} {shlex.quote(target)}")

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
        lines.extend(self.moves)
        lines.extend(self.edits)
        lines.extend(self.deletes)
        deepest_first = sorted(self.rmdirs, key=lambda path: path.count("/"), reverse=True)
        lines.extend(f"rmdir {shlex.quote(path)} 2>/dev/null || true" for path in deepest_first)
        return lines

    def _standing(self, directory: Path) -> bool:
        """True when *directory* is there now and no earlier move takes it away."""
        if not directory.is_dir():
            return False
        return not any(
            directory == source or source in directory.parents for source, _ in self.moved
        )

    def _before(self, path: Path) -> Path:
        """*path*, followed back through this block's earlier moves to where it lies before them."""
        for source, target in reversed(self.moved):
            if path == target or target in path.parents:
                path = source / path.relative_to(target)
        return path

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
