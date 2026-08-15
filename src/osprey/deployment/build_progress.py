"""Live per-service state for an image build, parsed from BuildKit's own stream.

A cold ``compose build`` runs for a quarter of an hour near-silent: the phase
reporter prints one line when it starts and one when it ends, and everything in
between goes to a spool file. BuildKit's plain-progress stream already carries
what an operator needs to tell that apart from a hang — which service is
working, which step it is on, and what that step is doing right now — so this
module keeps that state instead of discarding it. :mod:`osprey.cli.phase_reporter`
renders the result; nothing here touches a terminal.

:class:`BuildModel` is deliberately a pure model. It owns no thread, no clock
and no I/O: every line arrives through :meth:`BuildModel.feed` with the
timestamp already measured by the caller, which is what makes a fifteen-minute
build replayable from a fixture in milliseconds.

The stream is messier than it looks, and three of its shapes will silently
corrupt a naive parser:

* **Vertex numbers are not unique.** ``compose build`` numbers each service's
  sub-graph independently, so ``#7`` is ``[event-dispatcher 1/8]`` early and
  ``[bluesky-bridge 1/8]`` later. Attribution therefore follows the most recent
  header for a number, not the first one.
* **Headers repeat and steps arrive out of order.** A vertex re-emits its full
  header every time it resumes, and ``[ 2/13]`` can follow ``[ 5/13]``. The step
  index is read from the header; it is never inferred from arrival order.
* **Captions carry embedded carriage returns.** ``dpkg`` and ``pip`` redraw a
  progress line in place, so one physical line holds a dozen overwrites. Only
  the last one was ever visible, and only that one is kept.

Anything unrecognized is dropped rather than guessed at. A stream with no
BuildKit headers at all — a podman provider with its own progress format —
leaves the model empty, which the renderer reads as "nothing to show" and falls
back to its plain step lines.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

#: A plain-progress line: a vertex number, then either a bracketed header or a
#: bare continuation caption.
_VERTEX_LINE = re.compile(r"^#(?P<node>\d+)(?: (?P<rest>.*))?$")

#: The bracketed header at the head of a vertex's line, and the caption after it.
_HEADER = re.compile(r"^\[(?P<bracket>[^\]]*)\](?: (?P<caption>.*))?$")

#: A Dockerfile step index, as it appears inside a header: ``1/13``.
_STEP_INDEX = re.compile(r"^\d+/\d+$")

#: The line that marks one image finished. Kept in step with the equivalent in
#: :mod:`osprey.deployment.container_lifecycle`: the bare ``naming to`` line is
#: emitted while the export is still running, so only the ``done`` form counts.
_IMAGE_NAMED = re.compile(r"naming to (?P<ref>\S+?)(?: \d+(?:\.\d+)?s)? done\s*$")

#: Bracket words that name build infrastructure rather than a service. On the
#: single-image path the header is ``[internal] load build context`` — the word
#: sits exactly where a service name sits on the compose path, and taking it at
#: face value would mint a phantom "internal" service beside the real one. The
#: two-word forms (``[va internal]``, ``[panels auth]``) are unaffected: those
#: name a real service and are attributed to it.
_INFRASTRUCTURE_BRACKETS = frozenset({"auth", "internal"})

#: The pseudo-step for a service's export vertex. The export has no Dockerfile
#: step index of its own, but it is the longest single stretch of a cold build
#: and needs a name in the table.
EXPORT_STEP = "exporting"

#: The caption that opens an export vertex. Detected from the caption rather
#: than the bracket shape, because the export is bracketed on the compose path
#: (``#23 [virtual-accelerator] exporting to image``) and bare on the
#: single-image path (``#22 exporting to image``).
_EXPORT_CAPTION = "exporting to image"

#: Stripped from a finished image's ref as registry noise. A real registry in
#: the tag is the operator's own naming and stays.
_IMPLICIT_REGISTRY = "docker.io/library/"


def with_plain_build_progress(cmd: list[str]) -> list[str]:
    """Pin BuildKit's plain renderer on a ``<runtime> build`` argv.

    Everything in this module reads the plain-progress stream. ``auto`` does
    degrade to plain under a capture pipe, but only as undocumented behaviour,
    and a build that rendered anything else would parse into nothing at all --
    silently, because an unrecognized line is dropped rather than reported. So
    the flag is pinned rather than trusted, and it lives here, beside the parser
    whose input it guarantees.

    Docker only, the same caveat
    :func:`osprey.deployment.runtime_helper.with_plain_progress` carries for
    compose: ``podman build`` has no ``--progress``, and an unknown flag there
    would turn a cosmetic improvement into a failed deploy.

    Args:
        cmd: A build argv whose first element is the runtime, e.g.
            ``["docker", "build", "-t", ...]``. Extended in place and returned.
            Call this BEFORE the build context is appended: ``--progress`` is an
            argument of ``build``, and anything after the context is not read as
            one.

    Returns:
        The same list, with the flag appended when the runtime is docker.
    """
    if cmd and cmd[0] == "docker":
        cmd.extend(("--progress", "plain"))
    return cmd


@dataclass(frozen=True)
class BuildRow:
    """One service's current build state — one row of the live table.

    Frozen so that :meth:`BuildModel.snapshot`'s shallow copy is a real
    snapshot: a renderer holding a row cannot see it change underneath as the
    build advances.
    """

    #: Service name, or the model's ``label`` for a single-image build.
    service: str
    #: Current step: a Dockerfile index (``6/8``), a header word (``internal``),
    #: :data:`EXPORT_STEP`, or ``None`` before the first step is known.
    step: str | None
    #: What that step is doing right now, as last shown by BuildKit.
    caption: str
    #: When this service's first line arrived.
    started_at: float
    #: When its most recent line arrived — how a stalled service is spotted.
    last_line_at: float
    #: The finished image ref, once BuildKit has named it; ``None`` until then.
    finished: str | None


class BuildModel:
    """Per-service build state, fed one output line at a time.

    ``label`` names the service for a single-image build, whose headers carry
    no service name of their own: ``[ 1/13]``, ``[internal]``, and every
    bare continuation line bind to it. On the compose path there is no label —
    every header names its service — and a line that cannot be attributed is
    dropped rather than invented.

    ``on_finished_image`` is called once per image, with the ref, the moment
    BuildKit reports it named. It fires for every finished image, including one
    on a vertex too anonymous to attribute to a row, because "an image is done"
    is a fact about the build and not about any one service's row.

    Feeding and snapshotting are both guarded by one lock: ``run_captured`` is
    synchronous, so :meth:`feed` runs on the main thread while the renderer's
    monitor thread calls :meth:`snapshot`. The callback is invoked *after* the
    lock is released, so a callback that reads the model cannot deadlock.
    """

    def __init__(
        self,
        label: str | None = None,
        *,
        on_finished_image: Callable[[str], None] | None = None,
    ) -> None:
        self._label = label
        self._on_finished_image = on_finished_image
        self._lock = threading.Lock()
        self._rows: dict[str, BuildRow] = {}
        self._nodes: dict[int, str] = {}
        self._named: set[str] = set()

    def feed(self, line: str, now: float) -> None:
        """Absorb one line of build output, timestamped by the caller.

        ``now`` is injected rather than read here so the model stays pure: the
        caller measures time once, and a test can replay a whole build against
        a synthetic clock.
        """
        with self._lock:
            ref = self._absorb(line, now)
        if ref is not None and self._on_finished_image is not None:
            self._on_finished_image(ref)

    def snapshot(self) -> list[BuildRow]:
        """The current rows, in the order their services first appeared.

        A fresh list of frozen rows, so the caller can read it at leisure while
        the build continues. Empty until a BuildKit header has been recognized.
        """
        with self._lock:
            return list(self._rows.values())

    # -- parsing ------------------------------------------------------------

    def _absorb(self, line: str, now: float) -> str | None:
        """Apply one line to the model; return a newly finished ref, if any.

        Runs under the lock. The return value is the callback's argument rather
        than the callback itself, so the caller can fire it lock-free.
        """
        vertex = _VERTEX_LINE.match(line.rstrip("\n"))
        if vertex is None:
            return None
        node = int(vertex["node"])
        rest = vertex["rest"] or ""

        header = _HEADER.match(rest)
        if header is None:
            # A continuation line: it belongs to whichever service last claimed
            # this vertex number, or to the single-image label.
            service = self._nodes.get(node, self._label)
            step = None
            caption = _visible(rest)
        else:
            name, step = _split_bracket(header["bracket"])
            if name is None and step is None:
                return None  # An unrecognized bracket: ignored, never guessed.
            service = name if name is not None else self._label
            caption = _visible(header["caption"] or "")
            if service is not None:
                self._nodes[node] = service

        if caption.startswith(_EXPORT_CAPTION):
            step = EXPORT_STEP

        ref = self._new_ref(caption)
        if service is not None:
            self._touch(service, step, caption, now, ref)
        return ref

    def _new_ref(self, caption: str) -> str | None:
        """The image ref this caption finishes, unless it was already reported.

        BuildKit repeats the ``naming to`` line — bare, then with a duration —
        so an image would otherwise appear to build twice.
        """
        named = _IMAGE_NAMED.search(caption)
        if named is None:
            return None
        ref = named["ref"].removeprefix(_IMPLICIT_REGISTRY)
        if ref in self._named:
            return None
        self._named.add(ref)
        return ref

    def _touch(
        self,
        service: str,
        step: str | None,
        caption: str,
        now: float,
        ref: str | None,
    ) -> None:
        """Fold one line into a service's row, creating the row if it is new."""
        row = self._rows.get(service)
        if row is None:
            self._rows[service] = BuildRow(
                service=service,
                step=step,
                caption=caption,
                started_at=now,
                last_line_at=now,
                finished=ref,
            )
            return
        self._rows[service] = replace(
            row,
            step=step if step is not None else row.step,
            caption=caption or row.caption,
            last_line_at=now,
            finished=ref if ref is not None else row.finished,
        )


def _split_bracket(bracket: str) -> tuple[str | None, str | None]:
    """Read a header's brackets as ``(service, step)``.

    ``[virtual-accelerator 6/8]`` and ``[va internal]`` name their service.
    ``[ 1/13]`` and ``[internal]`` do not — they belong to the single-image
    build's one service, and are returned with no name so the caller can bind
    them to its label. A bare ``[virtual-accelerator]`` names a service but no
    step, and anything else returns ``(None, None)`` to be dropped.
    """
    parts = bracket.split()
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        token = parts[0]
        if _STEP_INDEX.match(token) or token in _INFRASTRUCTURE_BRACKETS:
            return None, token
        return token, None
    return None, None


def _visible(text: str) -> str:
    """What a terminal would have left on screen after the carriage returns.

    ``dpkg`` and ``pip`` redraw one progress line in place, so a single line of
    captured output can hold a whole animation. Only the last frame was ever
    visible; a trailing carriage return leaves the frame before it showing.
    """
    for frame in reversed(text.split("\r")):
        stripped = frame.strip()
        if stripped:
            return stripped
    return ""
