"""The one error a facility build stops with.

Every stop is a single line on stderr::

    facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>

A schema failure names its location as record kind ``path`` and the dotted path
as the id. A tree still laid out at retired paths stops before anything is read,
with :class:`RetiredLayoutRefused`: comment lines, then the shell lines that move
it, ending with :func:`_revalidate_command`. The exit code is 1; click keeps 2 for usage errors. A warning shares
the line shape, prints after a clean build, and leaves the exit code at 0.

Every error the description raises derives from :class:`FacilityError`, a plain
exception that knows how to print itself. None of them depends on click; the
command line turns them into click errors at its root group.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from osprey_connectors.simulation.engines import FacilityDefectError

if TYPE_CHECKING:
    from osprey_facility.retired import RetiredLayout

__all__ = [
    "KINDS",
    "WARNING_KINDS",
    "FacilityBuildError",
    "FacilityBuildWarning",
    "FacilityError",
    "FacilityFileError",
    "FacilityInputError",
    "FacilityUsageError",
    "ImportStop",
    "LayerSourceInvalid",
    "MOVE_HEADER",
    "MappingProblems",
    "PASTE_HEADER",
    "RetiredLayoutRefused",
    "exact_number",
    "quoted_slots",
    "refusal_ending",
]

#: Every error kind, each ``<thing>-<problem>``.
KINDS: tuple[str, ...] = (
    "source-invalid",
    "layer-conflict",
    "layer-duplicate",
    "fix-missing",
    "fix-stale",
    "fix-duplicate",
    "fix-computed",
    "fix-authored",
    "fix-referenced",
    "reference-missing",
    "relation-invalid",
    "class-unknown",
    "pair-invalid",
    "command-invalid",
    "value-invalid",
    "seed-invalid",
    "seed-missing",
    "limit-invalid",
    "place-conflict",
    "span-invalid",
    "wiring-conflict",
    "engine-missing",
    "engine-invalid",
    "model-conflict",
    "view-unsupported",
    "profile-invalid",
    "machine-missing",
    "machine-restated",
)

#: Every warning kind, each ``<thing>-<problem>``.
WARNING_KINDS: tuple[str, ...] = (
    "place-wrapped",
    "link-dangling",
    "relation-contested",
    "relation-coarse",
    "alias-shared",
    "identity-ambiguous",
    "part-cycle",
    "relation-stub",
    "place-unplaced",
    "members-disagree",
    "kind-deferred",
)


def exact_number(number: float) -> str:
    """*number* as the shortest text that reads back as the same float.

    A whole number prints bare (``5``, not ``5.0``).
    """
    return repr(float(number)).removesuffix(".0")


def quoted_slots(slots: Iterable[str]) -> str:
    """Name slots in an error line: each in backticks, comma-separated.

    Args:
        slots: The slot names, in the order they are named.

    Returns:
        The names, such as ```noise`, `drift``` for ``noise`` and
        ``drift``.
    """
    return ", ".join(f"`{slot}`" for slot in slots)


class FacilityFileError(ValueError):
    """A facility file is present but cannot be used.

    It is unreadable, not JSON, not an object, declares no header it must
    carry, or (for the identity read) names no identity code. Only an absent
    file is ever an absence; every other defect of the file stops here.
    """


class FacilityError(Exception):
    """An error the facility description stops with.

    It prints itself: :meth:`show` writes its message, and nothing else, as
    the lines a person reads. The command line exits 1 on it.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def format_message(self) -> str:
        """The message the error prints."""
        return self.message

    def __str__(self) -> str:
        return self.message

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the message alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        (file or sys.stderr).write(self.format_message() + "\n")


class FacilityBuildError(FacilityError, FacilityDefectError):
    """A facility build stop: one kind, one record, one remedy.

    It is a :class:`~osprey_connectors.simulation.engines.FacilityDefectError`,
    so an engine that raises it on a request refuses the request as a
    facility-file defect.

    Attributes:
        kind: The error kind, one of ``KINDS``.
        record_id: The id of the offending record, or the dotted path of a
            schema failure.
        sources: The source files the record came from, in source order.
        remedy: What the author changes to clear the stop.
        record_kind: The kind of record the id names (``device``, ``channel``,
            ...), or ``path`` for a schema failure.
        detail: What is wrong with the record.
    """

    def __init__(
        self,
        kind: str,
        record_id: str,
        sources: Sequence[str | Path],
        remedy: str,
        *,
        record_kind: str,
        detail: str,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown facility error kind {kind!r}")
        self.kind = kind
        self.record_id = record_id
        self.sources = tuple(str(source) for source in sources)
        self.remedy = remedy
        self.record_kind = record_kind
        self.detail = detail
        super().__init__(f"facility: {kind}: {record_kind} {record_id} — {detail}; fix: {remedy}")


class LayerSourceInvalid(FacilityError):
    """The source holds records the layer cannot write; nothing was written.

    Attributes:
        errors: One ``source-invalid`` error per problem, in the order the
            layer reports them.
        warnings: The warning lines the layer found before it refused,
            shown ahead of the errors.
    """

    def __init__(self, errors: list[FacilityBuildError], *, warnings: tuple[str, ...] = ()) -> None:
        self.errors = errors
        self.warnings = warnings
        super().__init__("\n".join(error.format_message() for error in errors))

    def show(self, file: IO[Any] | None = None) -> None:
        """Write each warning's line, then each error's, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        stream = file if file is not None else sys.stderr
        for warning in self.warnings:
            stream.write(warning + "\n")
        for error in self.errors:
            error.show(file)


class ImportStop(FacilityError):
    """An importer stop: ``import mml: <problem>: <what>``, one line per finding.

    Args:
        problem: The stop's word: ``mapping-draft``, ``mapping-undecided``,
            ``mapping-invalid``, ``export-invalid`` or ``reference-missing``.
        lines: What stopped the import, one entry per printed line.
    """

    def __init__(self, problem: str, lines: list[str]) -> None:
        self.problem = problem
        self.lines = tuple(lines)
        super().__init__("\n".join(f"import mml: {problem}: {line}" for line in lines))


class MappingProblems(FacilityError):
    """The mapping fails its check: one line per problem, then how many there are.

    Args:
        path: The mapping file.
        problems: What the check refused, in its order; each prints as its
            ``str``.
    """

    def __init__(self, path: Path, problems: Sequence[object]) -> None:
        self.path = path
        self.problems = tuple(problems)
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        lines = [str(problem) for problem in self.problems]
        lines.append(f"{count} {noun} in {path}; fix each and check again.")
        super().__init__("\n".join(lines))


#: The header line of a refusal whose block migrates the tree when pasted whole.
PASTE_HEADER = (
    "# Paste this whole block to move them to their current paths and finish the"
    " migration, then run the command again:"
)

#: The header line of a refusal whose block moves files but cannot finish the
#: migration, because something listed below it needs a person first.
MOVE_HEADER = "# Move them to their current paths, then run the command again:"

#: The closing line of a refusal that ends with a list instead of a command.
_PLACE_THESE = "# Place these, then run the command again."


def _revalidate_command(repo: Path) -> str:
    """The command a refusal's block ends with: ``osprey init --revalidate <repo>``.

    It finishes the migration the block's file commands start: the run finds
    nothing left to move and rewrites what a shell line cannot.

    Args:
        repo: The deployment repo, absolute; it is shell-quoted.
    """
    return f"osprey init --revalidate {shlex.quote(str(repo))}"


def refusal_ending(script: Sequence[str], repo: Path, *, finishes: bool) -> list[str]:
    """The lines a refusal ends with, after its block and its lists.

    Args:
        script: The block's file commands; empty when nothing can be moved.
        repo: The deployment repo the closing command migrates.
        finishes: False when something listed needs a person before the
            migration can finish.

    Returns:
        :func:`_revalidate_command` when *script* is non-empty and *finishes*,
        else :data:`_PLACE_THESE`.
    """
    if script and finishes:
        return [_revalidate_command(repo)]
    return [_PLACE_THESE]


class RetiredLayoutRefused(FacilityError):
    """A facility tree still holds sources at retired paths, so nothing is read.

    The message is ``# `` comment lines, then *script* (shell lines that run
    from any working directory), then one comment per clash, then
    :func:`_revalidate_command` for *repo*. When there is a clash, or no
    script, it ends with :data:`_PLACE_THESE` instead: the command would stop
    on the clash.

    Args:
        machine_dir: The machine root, resolved.
        content_dir: The content root when mounted, resolved; ``None`` when
            it is the machine root.
        layout: What the tree still holds at retired paths.
        script: The shell lines that carry out the moves and deck edits.
        repo: The deployment repo the closing command migrates.
    """

    def __init__(
        self,
        machine_dir: Path,
        content_dir: Path | None,
        layout: RetiredLayout,
        script: Sequence[str],
        repo: Path,
    ) -> None:
        self.machine_dir = machine_dir
        self.content_dir = content_dir
        self.layout = layout
        self.script = tuple(script)
        self.repo = repo
        where = f"# The facility tree at {machine_dir}"
        if content_dir is not None and content_dir != machine_dir:
            where += f" (content at {content_dir})"
        finishes = not layout.clashes
        lines = [f"{where} still holds sources at retired paths; nothing was read."]
        if self.script:
            lines.append(PASTE_HEADER if finishes else MOVE_HEADER)
            lines.extend(self.script)
        lines.extend(
            f"# {clash.old.resolve()} and {clash.new.resolve()} both exist:"
            " keep one, delete the other."
            for clash in layout.clashes
        )
        lines.extend(refusal_ending(self.script, repo, finishes=finishes))
        super().__init__("\n".join(lines))


class FacilityInputError(FacilityError):
    """An outside export a loader cannot read.

    It prints as a command-line error does: ``Error: <message>``.
    """

    def show(self, file: IO[Any] | None = None) -> None:
        """Write ``Error: <message>``.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        (file or sys.stderr).write(f"Error: {self.format_message()}\n")


class FacilityUsageError(FacilityError):
    """The invocation names something the description cannot act on.

    The command that raises it into the command line reports it as a usage
    error of that command.
    """


@dataclass(frozen=True)
class FacilityBuildWarning:
    """A fact of a clean facility build its author should read.

    A warning never stops the build and never changes the exit code; its line
    has the shape of a stop's line.

    Attributes:
        kind: The warning kind, one of ``WARNING_KINDS``.
        record_kind: The kind of record the id names (``device``, ...).
        record_id: The id of the record the warning is about.
        detail: What the build did with the record.
        remedy: What the author changes to clear the warning.
    """

    kind: str
    record_kind: str
    record_id: str
    detail: str
    remedy: str

    def __post_init__(self) -> None:
        if self.kind not in WARNING_KINDS:
            raise ValueError(f"unknown facility warning kind {self.kind!r}")

    @property
    def summary(self) -> str:
        """The line without its remedy."""
        return f"facility: {self.kind}: {self.record_kind} {self.record_id} — {self.detail}"

    @property
    def line(self) -> str:
        """The whole line, remedy included."""
        return f"{self.summary}; fix: {self.remedy}"
