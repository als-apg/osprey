"""Warn-only bundle validation for the OKF knowledge panel.

Sweeps an :class:`~osprey.services.facility_knowledge.okf.bundle.OKFBundle` and
reports authoring-level problems without ever raising: missing frontmatter and
dangling links in document bodies (OKF §6.1 has consumers accept broken links,
but the panel surfaces them as warnings so authors can fix them).

The OKF imports resolve to core osprey rather than a vendored ``okf/`` copy.

The OKF API this builds on is intentionally narrow:

* ``bundle.list_concepts()`` -> ``list[ConceptEntry]`` (each has ``concept_id``).
* ``bundle.read_concept(concept_id)`` -> ``OKFDocument``; raises
  :class:`OKFBundleError` if the concept file is missing and
  :class:`OKFDocumentError` on malformed frontmatter.
* ``doc.validate("authoring")`` -> ``None``; *raises* :class:`OKFDocumentError`
  listing the missing required keys.
* ``resolve_in_bundle(target, source_dir=...)`` -> the bundle-relative path a
  link names, or ``None`` for an external one (OKF §6).

Every risky call is wrapped: :func:`validate_bundle` returns a list and never
raises.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from osprey.services.facility_knowledge.okf.bundle import OKFBundle, OKFBundleError
from osprey.services.facility_knowledge.okf.document import OKFDocumentError
from osprey.services.facility_knowledge.okf.paths import resolve_in_bundle

# Markdown inline links: [text](target) or [text](<target>), the angle-bracket
# destination holding spaces or parentheses.
_LINK_RE = re.compile(r"\[([^\]]*)\]\((<[^>\n]*>|[^)]+)\)")


@dataclass(frozen=True)
class ValidationWarning:
    """A single non-fatal problem found while validating a bundle.

    Args:
        concept_id: The concept whose document the problem was found in.
        kind: Problem category — ``"frontmatter"`` or ``"broken-link"``.
        detail: Human-readable description of the specific problem.
    """

    concept_id: str
    kind: str
    detail: str


def validate_bundle(bundle: OKFBundle) -> list[ValidationWarning]:
    """Scan every concept in *bundle* and collect authoring warnings.

    For each concept this checks (a) authoring-level frontmatter completeness
    and (b) that every in-bundle body link names a file the bundle holds: a
    concept, an asset or a directory. External links are skipped: any URI
    scheme (``https:``, ``mailto:``, and ``facility:``, whose records the
    facility build checks), in-page ``#`` anchors, and targets outside the
    bundle root.

    This function never raises: any unexpected error while reading a concept is
    itself recorded as a ``"frontmatter"`` warning so the sweep can continue.

    Args:
        bundle: The OKF bundle to validate.

    Returns:
        A list of :class:`ValidationWarning` (possibly empty for a clean
        bundle).
    """
    warnings: list[ValidationWarning] = []

    try:
        entries = bundle.list_concepts()
    except Exception as exc:  # warn-only sweep, never propagate.
        warnings.append(
            ValidationWarning(
                concept_id="<bundle>",
                kind="frontmatter",
                detail=f"could not list concepts: {exc}",
            )
        )
        return warnings

    for entry in entries:
        concept_id = entry.concept_id

        # 1. Read the concept; tolerate unreadable / malformed files.
        try:
            doc = bundle.read_concept(concept_id)
        except (OKFBundleError, OKFDocumentError) as exc:
            warnings.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="frontmatter",
                    detail=f"unreadable: {exc}",
                )
            )
            continue
        except Exception as exc:  # never let one bad doc abort.
            warnings.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="frontmatter",
                    detail=f"unreadable: {exc}",
                )
            )
            continue

        # 2. Authoring-level frontmatter validation (raises on missing keys).
        try:
            doc.validate("authoring")
        except OKFDocumentError as exc:
            warnings.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="frontmatter",
                    detail=str(exc),
                )
            )
        except Exception as exc:
            warnings.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="frontmatter",
                    detail=f"validation error: {exc}",
                )
            )

        # 3. Cross-link scan over the body.
        warnings.extend(_scan_links(bundle, concept_id, doc.body))

    return warnings


def _scan_links(bundle: OKFBundle, concept_id: str, body: str) -> list[ValidationWarning]:
    """Return link warnings for one document body. Never raises."""
    found: list[ValidationWarning] = []
    source_dir = concept_id.rpartition("/")[0]

    try:
        matches = list(_LINK_RE.finditer(body or ""))
    except Exception:  # defensive; regex over str should not fail.
        return found

    for match in matches:
        target = match.group(2).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        if not target:
            continue
        # OKF §6: "/x.md" resolves from the bundle root, "x.md" from this
        # page's directory. A target with a scheme, or one that resolves
        # outside the root, is external and is never read.
        resolved = resolve_in_bundle(target, source_dir=source_dir)
        if resolved is None:
            continue

        try:
            # A page linked without its ".md" is still found, as it always was.
            exists = (bundle.root / resolved).exists() or (
                not resolved.endswith("/") and (bundle.root / f"{resolved}.md").is_file()
            )
        except Exception:
            exists = False
        if not exists:
            found.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="broken-link",
                    detail=f"{target!r} -> {resolved} (no such file)",
                )
            )

    return found


def log_validation_summary(
    warnings: list[ValidationWarning],
    log: Callable[[str], None] | object = print,
) -> None:
    """Emit a concise summary of *warnings*. Never raises.

    Args:
        warnings: The warnings produced by :func:`validate_bundle`.
        log: Either a logger-like object exposing ``.warning``/``.info``, or a
            plain callable such as :func:`print`. A ``logging.Logger`` works.
    """

    # UNGUARDED: copy passed to this wrapper is not seen by the house-style guard
    # in `tests/cli/test_printed_copy_style.py`, which matches printer names
    # exactly. The name cannot join its set, because `cli/deploy_scaffold.py` has
    # an `_emit` that writes a FILE, and registering the bare name would judge
    # that one's arguments as prose. The summary lines below are read by whoever
    # ran the sweep, so keep them to the same house style by hand.
    def _emit(message: str) -> None:
        try:
            # Prefer a logger's .warning, then .info, else treat as callable.
            warn = getattr(log, "warning", None)
            if callable(warn):
                warn(message)
                return
            info = getattr(log, "info", None)
            if callable(info):
                info(message)
                return
            if callable(log):
                log(message)
        except Exception:  # logging must never break the sweep.
            pass

    try:
        if not warnings:
            _emit("OKF bundle validation: 0 warnings (clean).")
            return

        counts: dict[str, int] = {}
        for w in warnings:
            counts[w.kind] = counts.get(w.kind, 0) + 1
        breakdown = ", ".join(f"{kind}={n}" for kind, n in sorted(counts.items()))
        _emit(f"OKF bundle validation: {len(warnings)} warning(s) ({breakdown}).")
        for w in warnings:
            _emit(f"  [{w.kind}] {w.concept_id}: {w.detail}")
    except Exception:
        pass


def bundle_health(warnings: list[ValidationWarning]) -> dict:
    """Shape a :func:`validate_bundle` result into the ``/api/bundle_health`` payload.

    Args:
        warnings: The warnings produced by :func:`validate_bundle`.

    Returns:
        A JSON-serialisable dict::

            {"ok": <bool>,
             "total": <int>,
             "counts": {<kind>: <n>, ...},
             "warnings": [{"concept_id", "kind", "detail"}, ...]}

        ``ok`` is ``True`` iff there are no warnings.
    """
    counts: dict[str, int] = {}
    for w in warnings:
        counts[w.kind] = counts.get(w.kind, 0) + 1
    return {
        "ok": not warnings,
        "total": len(warnings),
        "counts": counts,
        "warnings": [
            {"concept_id": w.concept_id, "kind": w.kind, "detail": w.detail} for w in warnings
        ],
    }
