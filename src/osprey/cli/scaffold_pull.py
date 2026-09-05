"""Core of ``osprey scaffold pull`` — copying packaged app-template content out.

A deployment starts from a packaged app template, and everything in that
template is a starting point rather than a fixture: a facility replaces the
example knowledge base with its own, keeps the channel-database examples as a
shape reference, and adds to the web-terminal context. ``scaffold pull`` is how
that content leaves the installation and lands in a deployment repo where it can
be edited and version-controlled.

The catalog below answers the first question an operator has — *what is in
there?* — and later sections turn a chosen path into a planned, applied copy.
The listing is deliberately the same view the copy acts on, so what an operator
sees named is exactly what they can pull.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Catalog: what a template offers
# ---------------------------------------------------------------------------

#: Direct children of an app template that are build machinery rather than
#: content: the Jinja sources the build renders, and the package marker that
#: makes the template importable. Nothing below the root is machinery — a
#: bundled MCP server ships its own ``__init__.py`` and is content — so these
#: names are dropped at the root only.
_ROOT_ONLY_EXCLUDED_SUFFIXES: tuple[str, ...] = (".j2",)
_ROOT_ONLY_EXCLUDED_NAMES: frozenset[str] = frozenset({"__init__.py"})


def list_pullable_paths(app_root: Path, subtree: str | None = None) -> list[str]:
    """Every path in an app template an operator can pull, directories first.

    The listing excludes what a pull could never usefully produce: the
    root-level build machinery (:data:`_ROOT_ONLY_EXCLUDED_NAMES` and the
    ``.j2`` sources), and — inside ``data/`` — whatever the packaged-data copy
    already drops, read from
    :func:`~.profile_cmd._data_copy_ignore` rather than restated here so the
    two can never disagree about what a wheel actually ships.

    Directories come first because they are what an operator usually wants to
    name: pulling ``data/facility_knowledge/`` is the common case, and pulling a
    single file the exception. Both groups are sorted, so the order is stable
    across releases and machines.

    Args:
        app_root: The app template to list, as returned by
            :func:`~.profile_cmd._app_template_root`.
        subtree: Optional template-relative path to restrict the listing to. A
            directory yields itself and everything below it; a file yields just
            that file.

    Returns:
        Template-relative POSIX paths — directories first, each with a trailing
        ``/``, then files; each group sorted.

    Raises:
        ValueError: If ``subtree`` names nothing pullable in this template. The
            message lists the template's top-level entries, which is what a
            caller needs to correct the path.
    """
    from .profile_cmd import _data_copy_ignore

    data_root = app_root / "data"
    data_ignore = _data_copy_ignore(data_root)

    directories: list[str] = []
    files: list[str] = []

    def visit(directory: Path) -> None:
        names = sorted(entry.name for entry in directory.iterdir())
        at_root = directory == app_root
        ignored: set[str] = set()
        if directory == data_root or data_root in directory.parents:
            ignored = data_ignore(str(directory), names)
        for name in names:
            if name in ignored:
                continue
            path = directory / name
            if at_root and not path.is_dir() and _is_root_machinery(name):
                continue
            relative = path.relative_to(app_root).as_posix()
            if path.is_dir():
                directories.append(f"{relative}/")
                visit(path)
            else:
                files.append(relative)

    visit(app_root)
    directories.sort()
    files.sort()

    if subtree is None:
        return directories + files

    wanted = subtree.strip("/")
    prefix = f"{wanted}/"
    selected_dirs = [entry for entry in directories if entry == prefix or entry.startswith(prefix)]
    selected_files = [entry for entry in files if entry == wanted or entry.startswith(prefix)]
    if not selected_dirs and not selected_files:
        top_level = sorted(
            [entry for entry in directories if entry.count("/") == 1]
            + [entry for entry in files if "/" not in entry]
        )
        raise ValueError(
            f"{subtree!r} is not in this template. Top-level entries: {', '.join(top_level)}"
        )

    return selected_dirs + selected_files


def _is_root_machinery(name: str) -> bool:
    """Whether a direct child of an app template is build machinery, not content."""
    return name in _ROOT_ONLY_EXCLUDED_NAMES or name.endswith(_ROOT_ONLY_EXCLUDED_SUFFIXES)


# ---------------------------------------------------------------------------
# Plan: what a pull would do
# ---------------------------------------------------------------------------

#: What a planned pull can decide about one file. Only ``written`` and
#: ``updated`` touch the disk; ``unchanged`` and ``skipped`` are quiet no-ops,
#: and ``refused`` is the one outcome that stops the whole pull.
PullActionKind = Literal["written", "updated", "unchanged", "refused", "skipped"]

#: The knowledge base is the one subtree a pull thins out by default. Its
#: documents are demo content a facility replaces, while the ``index.md`` files
#: are the structure it keeps, so a plain pull produces a skeleton and
#: ``--with-content`` produces the worked example.
_KNOWLEDGE_ROOT = "data/facility_knowledge"
_KNOWLEDGE_INDEX_NAME = "index.md"


@dataclass(frozen=True)
class PullAction:
    """What a pull would do to one file, and why.

    Attributes:
        source: The file in the app template the copy would read.
        target: Where in the deployment repo it would land.
        action: One of :data:`PullActionKind`.
        reason: A short phrase naming the flag or condition behind ``action``,
            written to be printed beside the path.
    """

    source: Path
    target: Path
    action: PullActionKind
    reason: str


def plan_pull(
    app_root: Path,
    repo_root: Path,
    rel_path: str | None,
    *,
    force: bool,
    with_content: bool,
) -> list[PullAction]:
    """Decide, file by file, what pulling ``rel_path`` would do to ``repo_root``.

    The plan is the whole decision: the apply step copies what this returns and
    decides nothing of its own. Candidates are exactly what
    :func:`list_pullable_paths` lists, so what an operator sees named is what a
    pull acts on, and there is one action per *file* — the directories a copy
    creates on the way are implied by their targets.

    The rules, keyed on each file's template-relative path:

    1. Under ``data/facility_knowledge/``, anything not named ``index.md`` is
       ``skipped`` unless ``with_content``, leaving the structure without the
       demo documents. A request that resolves to exactly one such file is
       ``refused`` instead, because skipping it would do nothing at all.
    2. A destination file that already exists is ``refused`` unless ``force``;
       with ``force`` it is ``updated``, or ``unchanged`` when the bytes already
       match. That comparison is the only time this function reads content.
    3. A symlinked target, a symlinked directory between ``repo_root`` and the
       target, or a kind mismatch (a directory where the file goes, a file where
       a directory has to be) is ``refused`` with or without ``force``. A pull
       resolves nothing on a facility's behalf here.
    4. Everything else is ``written``.

    Callers must treat a single ``refused`` as fatal for the *whole* pull: the
    apply step writes nothing when any action is ``refused``, so an operator
    never has to reason about a half-applied copy.

    Args:
        app_root: The app template to pull from, as returned by
            :func:`~.profile_cmd._app_template_root`.
        repo_root: The deployment repo the copy would land in. Targets mirror
            the template-relative path under it.
        rel_path: Template-relative path to pull, or ``None`` for the whole
            template.
        force: Whether an existing destination file may be overwritten.
        with_content: Whether the knowledge base comes across in full.

    Returns:
        One :class:`PullAction` per candidate file, in the order
        :func:`list_pullable_paths` returns them.

    Raises:
        ValueError: If ``rel_path`` names nothing pullable in this template,
            propagated from :func:`list_pullable_paths` with its list of
            top-level entries.
    """
    candidates = [
        entry for entry in list_pullable_paths(app_root, rel_path) if not entry.endswith("/")
    ]
    # A single filtered file is the one case where "skipped" would be a silent
    # no-op for the entire command, so it is reported as a refusal instead.
    only_one = len(candidates) == 1

    actions: list[PullAction] = []
    for relative in candidates:
        source = app_root / relative
        target = repo_root / relative

        if not with_content and _is_knowledge_content(relative):
            if only_one:
                actions.append(
                    PullAction(
                        source,
                        target,
                        "refused",
                        "only index.md comes from the knowledge base; "
                        "--with-content pulls this file",
                    )
                )
            else:
                actions.append(
                    PullAction(
                        source,
                        target,
                        "skipped",
                        "knowledge content rather than an index; --with-content pulls it",
                    )
                )
            continue

        refusal = _target_refusal(repo_root, target)
        if refusal is not None:
            actions.append(PullAction(source, target, "refused", refusal))
            continue

        if target.exists():
            if not force:
                actions.append(
                    PullAction(
                        source,
                        target,
                        "refused",
                        "already exists in this repo; --force overwrites it",
                    )
                )
            elif source.read_bytes() == target.read_bytes():
                actions.append(
                    PullAction(source, target, "unchanged", "already identical to the template")
                )
            else:
                actions.append(PullAction(source, target, "updated", "replaced under --force"))
            continue

        actions.append(PullAction(source, target, "written", "not in this repo yet"))

    return actions


def _is_knowledge_content(relative: str) -> bool:
    """Whether a template-relative path is a knowledge document, not an index."""
    if not relative.startswith(f"{_KNOWLEDGE_ROOT}/"):
        return False
    return relative.rsplit("/", 1)[-1] != _KNOWLEDGE_INDEX_NAME


def _target_refusal(repo_root: Path, target: Path) -> str | None:
    """Why ``target`` cannot be written, or ``None`` when it can.

    Checked without following a single link: a symlink anywhere on the way to
    the target means the copy would land somewhere the operator did not name,
    and a kind mismatch means it would have to remove something to proceed.
    Neither is something ``--force`` is allowed to decide.
    """
    parts = target.relative_to(repo_root).parts
    walked = repo_root
    for part in parts[:-1]:
        walked = walked / part
        shown = walked.relative_to(repo_root).as_posix()
        if walked.is_symlink():
            return f"{shown} is a symlink; refused with or without --force"
        if walked.exists() and not walked.is_dir():
            return f"{shown} is a file, but this pull needs it to be a directory"

    if target.is_symlink():
        return "the destination is a symlink; refused with or without --force"
    if target.is_dir():
        return "a directory already stands where this file would go"
    return None


# ---------------------------------------------------------------------------
# Apply: do what the plan says
# ---------------------------------------------------------------------------


def apply_pull(
    actions: list[PullAction],
    *,
    repo_root: Path,
    with_content: bool,
) -> list[PullAction]:
    """Copy what ``actions`` decided, and leave the knowledge indexes truthful.

    This step decides nothing. It reads the plan
    :func:`plan_pull` produced and performs it, which keeps every rule about
    *what* a pull does in one place and every rule about *how* it lands here.

    Two invariants hold whatever the plan says:

    * **A refusal stops everything.** If any action is ``refused``, not one byte
      is written and the actions come back unchanged, so an operator never has
      to reason about a half-applied pull. The command layer prints the
      refusals and exits non-zero.
    * **Nothing is ever removed.** A file the template lacks stays where it is,
      including inside a directory the pull writes into.

    ``written`` and ``updated`` are copied with their metadata, creating parent
    directories on the way; ``unchanged`` and ``skipped`` are no-ops.

    A skeleton pull — ``with_content=False``, the same flag that told
    :func:`plan_pull` to leave the demo documents behind — then rebuilds the
    knowledge base's ``index.md`` files from what is actually on disk, because
    the packaged indexes name documents this pull deliberately did not bring.
    Each copied index is emptied first and
    :func:`~osprey.services.facility_knowledge.okf.index.regenerate_indexes`
    writes it again from the directory's real contents, which is deterministic
    and makes no network or model call. A directory that ends up holding no
    documents keeps an empty index rather than a list of things that are not
    there; once the facility adds its own documents, the same regenerator fills
    it in.

    Args:
        actions: The plan to perform, as returned by :func:`plan_pull`.
        repo_root: The deployment repo the plan targets — the same one it was
            planned against. Needed to locate the knowledge base for the
            rebuild, which cannot be inferred from the targets alone.
        with_content: Whether the knowledge base came across in full, passed
            through from the same flag :func:`plan_pull` was given. ``False``
            triggers the index rebuild described above.

    Returns:
        The actions that were applied — the ``written`` and ``updated`` ones, as
        the very objects passed in, ready for the command layer to report. When
        the plan was refused, the whole unchanged ``actions`` list comes back
        instead and nothing was written.
    """
    import shutil

    if any(action.action == "refused" for action in actions):
        return actions

    applied = [action for action in actions if action.action in ("written", "updated")]
    for action in applied:
        action.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.source, action.target)

    if rebuilds_knowledge_indexes(applied, repo_root=repo_root, with_content=with_content):
        from ..services.facility_knowledge.okf.index import regenerate_indexes

        for action in applied:
            if action.target.name == _KNOWLEDGE_INDEX_NAME and _is_knowledge_target(
                repo_root, action.target
            ):
                action.target.write_text("", encoding="utf-8")
        regenerate_indexes(repo_root / _KNOWLEDGE_ROOT)

    return applied


def rebuilds_knowledge_indexes(
    applied: list[PullAction], *, repo_root: Path, with_content: bool
) -> bool:
    """Whether applying ``applied`` rebuilds the deployment's knowledge indexes.

    The one definition of the condition :func:`apply_pull` rebuilds on, exposed
    so the command layer can report the rebuild from the same test rather than
    from a second spelling of it: a skeleton pull (``with_content=False``) that
    landed at least one file inside the knowledge base.

    Args:
        applied: The actions :func:`apply_pull` performed, as it returned them.
        repo_root: The deployment repo the plan targeted.
        with_content: The flag the plan was made and applied with.
    """
    return not with_content and any(
        _is_knowledge_target(repo_root, action.target) for action in applied
    )


def _is_knowledge_target(repo_root: Path, target: Path) -> bool:
    """Whether a planned target lands inside the deployment's knowledge base."""
    try:
        relative = target.relative_to(repo_root).as_posix()
    except ValueError:
        return False
    return relative.startswith(f"{_KNOWLEDGE_ROOT}/")
