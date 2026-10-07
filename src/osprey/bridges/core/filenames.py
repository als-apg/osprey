"""The storage names a bridge gives a run's files.

A bridge that stores a run's files in one directory — a Nextcloud DAV folder, a
SharePoint library folder — needs one naming pass over the whole delivery, and one
rule holds for it: two artifacts of one delivery never reach one storage name. The
worker names an artifact by basename, so a run whose steps each wrote ``plot.png``
arrives as several artifacts claiming one name; storing them under it would overwrite
each earlier file with the next and report every one as delivered.

The pass is split in two because the halves are settled at different times:

* :func:`unique_stems` assigns every artifact of the delivery its stem at once, before
  any fetch — the stem does not depend on the served ``Content-Type``;
* :func:`upload_name` gives one stem its extension once the bytes are in hand, since
  the extension does.

Every name is one safe path segment built from an allowlist of characters, so
worker-supplied data can never address a file outside the run's directory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import KNOWN_EXTENSIONS, safe_label

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
"""Everything replaced with ``_`` in an upload filename. An allowlist rather than a
denylist: the stem comes from worker-supplied data, and a ``/`` or ``..`` reaching a
storage path would address a file outside the run's directory."""


def _segment(value: Any) -> str:
    """Reduce a worker-supplied name to one safe filename stem.

    Args:
        value: Candidate name; anything that is not a string is treated as absent.

    Returns:
        The sanitized stem, or ``""`` when nothing usable survives (a name made only of
        dots and separators, for instance).
    """
    text = safe_label(value if isinstance(value, str) else None, "")
    return _UNSAFE_NAME_CHARS.sub("_", text).strip("._")[:120]


def upload_stem(label: Any, fallback: Any) -> str:
    """Build the filename stem an artifact is stored under, before its extension.

    Args:
        label: Preferred name (the worker's ``filename``, when it sent one).
        fallback: Name to use when ``label`` yields nothing usable — the artifact id, for
            a descriptor carrying no ``filename`` and for a bare id string from an older
            worker, which has none to carry.

    Returns:
        One safe path segment, never empty.
    """
    return _segment(label) or _segment(fallback) or "artifact"


def _disambiguate(stem: str, suffix: str) -> str:
    """Insert ``suffix`` into ``stem``, ahead of any trailing dot-suffix.

    Purely lexical, so no mime is needed: ``plot.png`` becomes ``plot-<suffix>.png``
    rather than ``plot.png-<suffix>``, which both reads as a filename in the
    conversation and leaves :func:`upload_name`'s already-suffixed check able to
    recognize the extension.

    ``_segment`` strips the leading dot run, so the head is never empty. The TAIL can
    be: ``_segment`` strips before it truncates at 120 characters, so a long name cut
    exactly at a dot ends in one, and the partition then yields ``""``. That costs a
    trailing dot in the filename and nothing else — the result is still one non-empty
    segment that is neither ``.`` nor ``..``, which is all a storage path requires.
    """
    head, dot, tail = stem.rpartition(".")
    return f"{head}-{suffix}{dot}{tail}" if dot else f"{stem}-{suffix}"


DELIVERED_EXTENSIONS = KNOWN_EXTENSIONS | {".png"}
"""Every extension a bridge's delivery can append. ``.png`` is unioned in because it
is the fixed extension of the image path and is deliberately absent from the core mime
allowlist (:data:`~osprey.bridges.core.KNOWN_EXTENSIONS`).

With this set a name can need SEVERAL strips: ``plot.png.pdf`` loses ``.pdf`` and then
``.png`` before it can be compared against ``plot``."""

_EXTENSIONS_BY_LENGTH = tuple(sorted(DELIVERED_EXTENSIONS, key=len, reverse=True))
""":data:`DELIVERED_EXTENSIONS`, longest first — the order :func:`_name_key` strips in.

Longest-match is made STRUCTURAL here rather than left to chance. As the set stands no
member is a suffix of another, so at most one can ever match and the order is
immaterial; but that is a property of today's contents, not of the algorithm. Add one
compound suffix (``.tar.gz``, which ends with a hypothetical ``.gz``) and a set-iterating
loop would strip whichever it happened to reach first — a key that differs between
runs, and a collision check that silently stops being reproducible. Sorting by length
means the longest match always wins, so the invariant is enforced by construction and a
future addition cannot quietly break it."""


def _name_key(stem: str) -> str:
    """The key on which two artifacts are judged to collide.

    Deliberately NOT the stem itself. :func:`upload_name` does not double an
    extension the stem already carries, so a stem of ``plot.png`` and a stem of
    ``plot`` both store as ``plot.png`` once the bytes are served as PNG — two
    different stems, one filename.

    Every trailing KNOWN extension is stripped, in a loop, after case-folding.
    Each part of that matters:

    * *known* rather than "any trailing dot-suffix", or ``report.v2`` and
      ``report.v2.pdf`` key differently (``report`` vs ``report.v2``) and still
      collide on ``report.v2.pdf``. It also stops ``report.draft`` and
      ``report.final`` from being disambiguated for no reason;
    * *in a loop*, because one strip is not enough: ``plot.png.pdf`` has to lose
      both before it can be compared with ``plot``. The loop walks the extensions
      longest first, so the longest match wins by construction rather than by luck
      of set iteration order;
    * *case-folded*, because ``upload_name``'s own check is case-insensitive, so
      ``a.PDF`` and ``a`` collide once ``.pdf`` is appended.

    Two artifacts whose extensions would in the end have differed still key alike and
    one is disambiguated needlessly. That is the deliberate direction: the extension
    is not known until after the fetch, and over-separating costs a longer filename
    while under-separating costs an overwritten file.
    """
    key = stem.lower()
    while True:
        for extension in _EXTENSIONS_BY_LENGTH:
            if key.endswith(extension) and len(key) > len(extension):
                key = key[: -len(extension)]
                break
        else:
            return key


def unique_stems(descriptors: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each artifact id in one delivery to the stem it is stored under.

    The worker names an artifact by BASENAME, so a run whose steps each wrote their own
    ``plot.png`` produces several descriptors carrying the SAME ``filename``. A bridge
    that stores a run's files in one directory would overwrite each earlier file under
    that one name — a silently lost plot, reported as delivered. Every artifact after
    the first to claim a name therefore gets a slice of its id inserted, and an ordinal
    after that if even the slice collides, so no two artifacts can reach one path.

    Collision is judged on :func:`_name_key`, not on the stem: two stems that differ
    can still yield ONE filename, so comparing stems would let exactly the overwrite
    this function exists to prevent back through.

    Settled over the whole delivery at once and BEFORE any fetch, which is possible
    because the stem — unlike the extension — does not depend on the served
    ``Content-Type``. That keeps a STEM independent of fetch order and of which fetches
    happened to fail, so a redelivery of the same run reproduces exactly the same stems,
    which is what an upload's idempotency rests on. Two conditions on that, both real:

    * only the STEM is reproduced — the extension is settled per artifact from the
      served ``Content-Type``, so a delivery whose conversion falls back differently
      still writes a different filename than the one before it;
    * assignment is ORDER-DEPENDENT by design (first to claim a key keeps the bare
      stem), so reproducibility holds only while the descriptor list arrives in the
      same order. It does: the list is rebuilt by ``artifact_descriptors`` from the
      run's own ``artifacts``, preserving the worker's order. A caller that sorted or
      de-duplicated the descriptors between deliveries would break this, which is why
      the list is passed through untouched.

    Args:
        descriptors: Every descriptor this delivery will attempt, in worker order.
            An entry naming no artifact is skipped, as is a repeat of an id already seen.

    Returns:
        ``{artifact_id: stem}``, with no two ids sharing a stem.
    """
    stems: dict[str, str] = {}
    taken: set[str] = set()
    for descriptor in descriptors:
        artifact_id = descriptor.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in stems:
            continue
        stem = upload_stem(descriptor.get("filename"), artifact_id)
        if _name_key(stem) in taken:
            stem = _disambiguate(stem, _segment(artifact_id)[:8] or "artifact")
            base, ordinal = stem, 2
            while _name_key(stem) in taken:
                stem = _disambiguate(base, str(ordinal))
                ordinal += 1
        taken.add(_name_key(stem))
        stems[artifact_id] = stem
    return stems


def upload_name(stem: str, extension: str) -> str:
    """Give a stem its extension.

    Args:
        stem: The stem this artifact was assigned by :func:`unique_stems`.
        extension: Extension to ensure, including the dot. Derived from the mime rather
            than from any worker-supplied name, so a worker cannot choose it.

    Returns:
        A single safe path segment carrying ``extension``, CASE-INSENSITIVELY: a stem
        that already ends in it is left alone rather than doubled, so ``("a.PDF",
        ".pdf")`` returns ``a.PDF`` — which carries the extension without literally
        ending in the string passed. :func:`_name_key` folds case for exactly this
        reason, so two stems differing only in an extension's case are still seen as
        one name.
    """
    if extension and stem.lower().endswith(extension.lower()):
        return stem
    # The worker's filename is predicted alongside delivered_mime, so a name carrying a
    # DIFFERENT extension is a prediction the delivery contradicted ("data.png" for a
    # render that never happened). Replace it rather than stack it — the extension still
    # comes only from the mime.
    #
    # Only a KNOWN extension is replaced, and this must stay in lockstep with what
    # _name_key strips. If this stripped more than _name_key does, two stems that
    # unique_stems judged distinct could still converge on one filename here — the
    # overwrite unique_stems exists to prevent, reintroduced one step later. A generic
    # "short alnum suffix" rule does exactly that: report.draft and report.final key
    # apart, then both upload as report.pdf.
    root, dot, suffix = stem.rpartition(".")
    if dot and root and f".{suffix}".lower() in DELIVERED_EXTENSIONS:
        stem = root
    return f"{stem}{extension}"
