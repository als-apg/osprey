"""What a deployment repo still holds at retired root paths, and how it moves.

The retired paths are the moved-from paths of :mod:`osprey.repo_layout`
(:func:`~osprey.repo_layout.retired_moves`), each looked for at
``<repo root>/<old path>``: an old path is where the readers of its time read
it, never under a relocatable profile key. Nothing here names a path of its
own.

:func:`find_retired_root` looks at the files only, neither runs git nor
prints, and reports:

- the moves and clashes of each retired path that still holds a source, as
  :func:`osprey_facility.retired.place_moves` places them, dot-files included;
- the line edits that follow the moves: values under the profile keys that
  hold repo-relative paths (:data:`PATH_KEYS`) in ``profile.yml``, the tier
  documents and the profile variants, the root pipeline file and the root
  ``.dockerignore``;
- the stale canopy staging at the repo root, removed with the moves;
- the steps to take on the host once the files moved;
- what a person must check by hand, because the paths inside it are not
  rewritten.

:func:`refuse_retired_layout` stops a command on such a repo with one script
that moves the root and, when the profile names one, the facility tree;
:func:`deployment_root` finds a command's repo and refuses it so.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import click
import yaml

from osprey import repo_layout
from osprey.repo_layout import RootMove
from osprey_facility.retired import (
    Clash,
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
    "RetiredRoot",
    "RetiredRootRefused",
    "deployment_root",
    "find_retired_root",
    "refuse_retired_layout",
]

#: The profile keys whose values are filesystem paths relative to the repo
#: root, as dotted paths through a profile document. ``*`` stands for any one
#: mapping key or sequence item. A key spelled with dots in the document
#: (``config: {code_review.prompt: ...}``) reads as the same dotted path as its
#: nested spelling. Only values under these keys are rewritten. A service's
#: ``template`` is one when it names a facility-owned service directory; a
#: bundled template name never begins with a moved path. A service's ``path``
#: is not one: it names where the build writes the service, inside the build
#: tree, which no root move touches.
PATH_KEYS: tuple[str, ...] = (
    "services.*.template",
    "config.code_review.prompt",
    "config.services.optimizer.environments_dir",
    "config.services.graphdb.ttl_path",
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

_STR_TAG = "tag:yaml.org,2002:str"


@dataclass(frozen=True)
class RetiredRoot:
    """Everything a deployment repo still holds at its retired root paths.

    Attributes:
        root: The repo root, resolved.
        moves: The sources to move, in the order of
            :func:`~osprey.repo_layout.retired_moves`.
        clashes: What blocks a move.
        line_edits: The edits that follow the moves, each naming its file at
            the path it has once the moves are done.
        deletes: Stale generated copies at the repo root, POSIX, relative to
            *root*.
        host_steps: The commands to run on the host once the files moved, in
            order.
        check_by_hand: What a person must look at, one line each, because the
            paths inside it are not rewritten.
    """

    root: Path
    moves: tuple[Move, ...] = ()
    clashes: tuple[Clash, ...] = ()
    line_edits: tuple[LineEdit, ...] = ()
    deletes: tuple[str, ...] = ()
    host_steps: tuple[str, ...] = ()
    check_by_hand: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """True when nothing at a retired path, or naming one, is left to move.

        The deletes, host steps and check-by-hand lines ride along with a
        move; on their own they are no retired layout.
        """
        return not (self.moves or self.clashes or self.line_edits)


class RetiredRootRefused(click.ClickException):
    """A deployment repo still holds sources at retired root paths; nothing was read.

    The message is ``# `` comment lines, then *script* (shell lines that run
    from any working directory), then one comment per clash, then the host
    steps and the check-by-hand list as comment blocks, then the alternative
    command. The whole message runs under ``sh``. It prints alone, with no
    ``Error: `` prefix, and the command line exits 1.

    Args:
        layout: What the repo still holds at retired root paths.
        facility_roots: The facility roots whose retired paths the script
            moves too, or ``None``.
        facility: What the facility tree still holds at retired paths.
        script: The shell lines that carry out every move and edit.
    """

    exit_code = 1

    def __init__(
        self,
        layout: RetiredRoot,
        facility_roots: FacilityRoots | None,
        facility: RetiredLayout | None,
        script: Sequence[str],
    ) -> None:
        self.layout = layout
        self.facility_roots = facility_roots
        self.facility = facility
        self.script = tuple(script)
        lines = [
            f"# The deployment repo at {layout.root} still holds sources at retired paths;"
            " nothing was read."
        ]
        clashes = list(layout.clashes)
        if facility is not None and facility_roots is not None and not facility.empty:
            where = f"# So does the facility tree at {facility_roots.machine.resolve()}"
            if facility_roots.mounted:
                where += f" (content at {facility_roots.content.resolve()})"
            lines.append(f"{where}.")
            clashes.extend(facility.clashes)
        lines.append("# Move them to their current paths, then run the command again:")
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
        lines.append("# or run: osprey init --revalidate")
        super().__init__("\n".join(lines))

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the message alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        (file or sys.stderr).write(self.format_message() + "\n")


def find_retired_root(repo_root: Path) -> RetiredRoot:
    """What the deployment repo at *repo_root* still holds at retired root paths.

    Each retired path is looked for at ``<repo_root>/<old path>``. A directory
    merged into an existing successor moves file by file, dot-files included.

    Args:
        repo_root: The deployment repo root.

    Returns:
        The result; :attr:`RetiredRoot.empty` when nothing is left to move.
    """
    root = repo_root.resolve()
    table = repo_layout.retired_moves()
    if not table:
        return RetiredRoot(root=root)
    moves, clashes = place_moves(root, [(move.old, move.new) for move in table], hidden=True)
    edits: list[LineEdit] = []
    by_hand: list[str] = []
    for document in _profile_documents(root):
        _document_edits(root, document, table, edits, by_hand)
    edits.extend(_pipeline_edits(root, table))
    edits.extend(_dockerignore_edits(root, table))
    if not (moves or clashes or edits):
        return RetiredRoot(root=root)
    by_hand.extend(_moved_inner_paths(root, table))
    return RetiredRoot(
        root=root,
        moves=tuple(moves),
        clashes=tuple(clashes),
        line_edits=tuple(edits),
        deletes=_canopy_deletes(root),
        host_steps=_host_steps(root),
        check_by_hand=tuple(dict.fromkeys(by_hand)),
    )


def refuse_retired_layout(repo_root: Path, facility_roots: FacilityRoots | None = None) -> None:
    """Stop when the repo, or the facility tree it names, holds anything at a retired path.

    When the root holds a retired path, one script moves the root and the
    facility tree together: the facility roots given, else those the
    profile's ``data:`` and ``machine:`` name, read without validation (a
    profile that cannot be read that way adds no facility moves). When only
    the facility tree at *facility_roots* is retired, the facility refusal is
    raised unchanged.

    Args:
        repo_root: The deployment repo root.
        facility_roots: The facility roots the caller resolved, or ``None``.

    Raises:
        RetiredRootRefused: The root holds something at a retired path.
        osprey_facility.errors.RetiredLayoutRefused: Only the facility tree
            at *facility_roots* does.
    """
    layout = find_retired_root(repo_root)
    if layout.empty:
        if facility_roots is not None:
            refuse_retired(facility_roots.machine, facility_roots.content)
        return
    roots = facility_roots if facility_roots is not None else _profile_facility_roots(layout.root)
    facility = None if roots is None else find_retired(roots.machine, roots.content)
    moves: list[Move] = list(layout.moves)
    edits: list[tuple[Path, LineEdit]] = [(layout.root, edit) for edit in layout.line_edits]
    if facility is not None:
        moves.extend(facility.moves)
        edits.extend(deck_line_edits(facility))
    script = render_script(moves, edits, [(layout.root, path) for path in layout.deletes])
    raise RetiredRootRefused(layout, roots, facility, script)


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
    """The facility roots ``profile.yml`` names, read raw; ``None`` when it names none."""
    document = _load(root / repo_layout.entry("profile.yml").path)
    if not isinstance(document, dict):
        return None
    from osprey.cli.build_profile_model import BuildProfile
    from osprey_facility.errors import FacilityError

    try:
        return BuildProfile(
            name="", data=document.get("data"), machine=document.get("machine")
        ).resolved_facility_roots(root)
    except (FacilityError, OSError, TypeError, ValueError):
        return None


# --- paths ---------------------------------------------------------------------


def _successor(path: str, table: Iterable[RootMove]) -> str | None:
    """Where the repo-relative *path* lives once the moves are done, if it moved.

    The longest retired path equal to *path*, or holding it, wins.
    """
    best: RootMove | None = None
    for move in table:
        if (path == move.old or path.startswith(f"{move.old}/")) and (
            best is None or len(move.old) > len(best.old)
        ):
            best = move
    if best is None:
        return None
    return best.new + path[len(best.old) :]


def _after_moves(root: Path, file: Path, table: Iterable[RootMove]) -> Path:
    """*file* at the path it has once the moves are done."""
    relative = file.relative_to(root).as_posix()
    moved = _successor(relative, table)
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
        moved = _successor(value[len(prefix) :], table)
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
        return _successor(value[len(_PROJECT_ROOT_PLACEHOLDER) :], table) is not None
    if not os.path.isabs(value):
        return False
    for base in (root, Path(os.path.abspath(root))):
        try:
            relative = Path(value).relative_to(base).as_posix()
        except ValueError:
            continue
        if _successor(relative, table) is not None:
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
            lambda match: (match["dot"] or "") + (_successor(match["old"], table) or match["old"]),
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
        moved = _successor(match["body"], table)
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
