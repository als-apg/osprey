"""Unit tests for okf_panel.validation over real (tmp_path) OKF bundles.

Exercises the warn-only sweep: a clean bundle produces no warnings; missing
frontmatter and broken cross-links each surface as a categorised warning; and
:func:`bundle_health` shapes the result for the ``/api/bundle_health`` endpoint.
"""

from __future__ import annotations

from pathlib import Path

from osprey.interfaces.okf_panel.validation import (
    ValidationWarning,
    bundle_health,
    validate_bundle,
)
from osprey.services.facility_knowledge.okf.bundle import OKFBundle


def _write_concept(root: Path, concept_id: str, *, frontmatter: dict, body: str = "") -> None:
    """Write ``<root>/<concept_id>.md`` with a YAML frontmatter block."""
    path = root / f"{concept_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    path.write_text(f"---\n{fm_lines}\n---\n\n{body}\n", encoding="utf-8")


def _clean_fm(title: str) -> dict:
    return {"type": "concept", "title": title, "description": f"About {title}."}


def test_clean_bundle_has_no_warnings(tmp_path):
    _write_concept(tmp_path, "devices/bpm", frontmatter=_clean_fm("BPM"))
    _write_concept(
        tmp_path,
        "devices/rf",
        frontmatter=_clean_fm("RF"),
        body="See [BPM](/devices/bpm.md) for the monitor.",
    )
    warnings = validate_bundle(OKFBundle(tmp_path))
    assert warnings == []

    health = bundle_health(warnings)
    assert health == {"ok": True, "total": 0, "counts": {}, "warnings": []}


def test_missing_frontmatter_flagged(tmp_path):
    # No 'description' -> authoring validation fails.
    _write_concept(tmp_path, "devices/bpm", frontmatter={"type": "concept", "title": "BPM"})
    warnings = validate_bundle(OKFBundle(tmp_path))

    kinds = {w.kind for w in warnings}
    assert "frontmatter" in kinds
    fm = next(w for w in warnings if w.kind == "frontmatter")
    assert fm.concept_id == "devices/bpm"
    assert "description" in fm.detail


def test_broken_cross_link_flagged(tmp_path):
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="Related: [ghost](/devices/does-not-exist.md).",
    )
    warnings = validate_bundle(OKFBundle(tmp_path))

    broken = [w for w in warnings if w.kind == "broken-link"]
    assert len(broken) == 1
    assert broken[0].concept_id == "devices/bpm"
    assert "does-not-exist" in broken[0].detail


def test_external_links_not_flagged(tmp_path):
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="See [docs](https://example.org) and [anchor](#section).",
    )
    warnings = validate_bundle(OKFBundle(tmp_path))
    assert [w for w in warnings if w.kind in ("broken-link", "malformed-link")] == []


def test_slash_id_concept_validates(tmp_path):
    _write_concept(
        tmp_path,
        "control-system/channel-finding",
        frontmatter=_clean_fm("Channel Finding"),
    )
    warnings = validate_bundle(OKFBundle(tmp_path))
    assert warnings == []


def test_bundle_health_shape_with_mixed_warnings(tmp_path):
    _write_concept(
        tmp_path, "a/bad", frontmatter={"type": "concept", "title": "Bad"}
    )  # missing description
    _write_concept(
        tmp_path,
        "a/good",
        frontmatter=_clean_fm("Good"),
        body="[x](/a/missing.md)",
    )  # broken link
    warnings = validate_bundle(OKFBundle(tmp_path))
    health = bundle_health(warnings)

    assert health["ok"] is False
    assert health["total"] == len(warnings)
    assert health["counts"].get("frontmatter", 0) >= 1
    assert health["counts"].get("broken-link", 0) >= 1
    # Every warning is a flat JSON-serialisable dict.
    for w in health["warnings"]:
        assert set(w) == {"concept_id", "kind", "detail"}


def test_validate_bundle_never_raises_on_missing_root(tmp_path):
    # OKFBundle refuses a missing root at construction; validate_bundle itself
    # tolerates a bundle whose listing fails by recording a warning.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert validate_bundle(OKFBundle(empty)) == []


def test_validation_warning_is_frozen():
    w = ValidationWarning(concept_id="x", kind="frontmatter", detail="d")
    assert (w.concept_id, w.kind, w.detail) == ("x", "frontmatter", "d")
