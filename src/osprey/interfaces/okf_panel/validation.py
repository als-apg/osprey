"""Warn-only bundle validation for the OKF knowledge panel.

Sweeps an :class:`~osprey.services.facility_knowledge.okf.bundle.OKFBundle` and
reports authoring-level problems without ever raising: missing frontmatter (per
OKF §9 authoring level) and dangling/malformed cross-links in document bodies
(OKF §9 tolerates broken links, but the panel surfaces them as warnings so
authors can fix them).

Ported from the ALS ``mcp_servers/okf_panel`` service; the only change from the
ALS original is that the OKF imports now resolve to core osprey rather than a
vendored ``okf/`` copy.

The OKF API this builds on is intentionally narrow:

* ``bundle.list_concepts()`` -> ``list[ConceptEntry]`` (each has ``concept_id``).
* ``bundle.read_concept(concept_id)`` -> ``OKFDocument``; raises
  :class:`OKFBundleError` if the concept file is missing and
  :class:`OKFDocumentError` on malformed frontmatter.
* ``doc.validate("authoring")`` -> ``None``; *raises* :class:`OKFDocumentError`
  listing the missing required keys.
* ``bundle.resolve_concept_path(concept_id)`` -> ``Path`` (need not exist);
  *raises* :class:`OKFBundleError` if the id escapes the bundle root.

Every risky call is wrapped: :func:`validate_bundle` returns a list and never
raises.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from osprey.services.facility_knowledge.okf.bundle import OKFBundle, OKFBundleError
from osprey.services.facility_knowledge.okf.document import OKFDocumentError

# Markdown inline links: [text](target)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Link targets we never try to resolve as concepts.
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


@dataclass(frozen=True)
class ValidationWarning:
    """A single non-fatal problem found while validating a bundle.

    Args:
        concept_id: The concept whose document the problem was found in.
        kind: Problem category — ``"frontmatter"``, ``"broken-link"``, or
            ``"malformed-link"``.
        detail: Human-readable description of the specific problem.
    """

    concept_id: str
    kind: str
    detail: str


def validate_bundle(bundle: OKFBundle) -> list[ValidationWarning]:
    """Scan every concept in *bundle* and collect authoring warnings.

    For each concept this checks (a) authoring-level frontmatter completeness
    and (b) that every internal body link resolves to an existing concept file.
    External links (``http``, ``https``, ``mailto``, in-page ``#`` anchors) are
    skipped.

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
    except Exception as exc:  # noqa: BLE001 — warn-only sweep, never propagate.
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
        except Exception as exc:  # noqa: BLE001 — never let one bad doc abort.
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
        except Exception as exc:  # noqa: BLE001
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

    try:
        matches = list(_LINK_RE.finditer(body or ""))
    except Exception:  # noqa: BLE001 — defensive; regex over str should not fail.
        return found

    for match in matches:
        target = match.group(2).strip()
        if not target:
            continue
        if target.lower().startswith(_EXTERNAL_PREFIXES):
            continue

        # Map a link target like "/devices/bpm.md" or "devices/bpm.md" to an
        # OKF §2 concept id: strip a leading slash and a trailing ".md".
        link_id = target.lstrip("/")
        if link_id.endswith(".md"):
            link_id = link_id[:-3]
        if not link_id:
            continue

        try:
            path = bundle.resolve_concept_path(link_id)
        except OKFBundleError as exc:
            found.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="malformed-link",
                    detail=f"{target!r}: {exc}",
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            found.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="malformed-link",
                    detail=f"{target!r}: {exc}",
                )
            )
            continue

        try:
            exists = path.exists()
        except Exception:  # noqa: BLE001
            exists = False
        if not exists:
            found.append(
                ValidationWarning(
                    concept_id=concept_id,
                    kind="broken-link",
                    detail=f"{target!r} -> {link_id} (no such concept)",
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
        except Exception:  # noqa: BLE001 — logging must never break the sweep.
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
    except Exception:  # noqa: BLE001
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
