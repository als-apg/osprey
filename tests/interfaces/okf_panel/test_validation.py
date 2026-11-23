"""Unit tests for okf_panel.validation over real (tmp_path) OKF bundles.

Exercises the warn-only sweep: a clean bundle produces no warnings; missing
frontmatter and broken cross-links each surface as a categorised warning; and
:func:`bundle_health` shapes the result for the ``/api/bundle_health`` endpoint.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from osprey.interfaces.okf_panel.validation import (
    ValidationWarning,
    bundle_health,
    validate_bundle,
)
from osprey.services.facility_knowledge.okf.assets import MAX_IMAGE_BYTES
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


def test_relative_links_resolve_from_the_page(tmp_path):
    """OKF §6: a link without a leading slash resolves from its own directory."""
    _write_concept(tmp_path, "devices/corrector", frontmatter=_clean_fm("Corrector"))
    _write_concept(tmp_path, "physics/orbit", frontmatter=_clean_fm("Orbit"))
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body=(
            "See the [corrector](corrector.md), the [same one](./corrector.md#gain) "
            "and the [orbit](../physics/orbit.md)."
        ),
    )

    assert validate_bundle(OKFBundle(tmp_path)) == []


def test_a_relative_link_is_not_read_from_the_bundle_root(tmp_path):
    """``corrector.md`` in ``devices/`` is not the root's ``corrector.md``."""
    _write_concept(tmp_path, "corrector", frontmatter=_clean_fm("Corrector"))
    _write_concept(
        tmp_path, "devices/bpm", frontmatter=_clean_fm("BPM"), body="[corrector](corrector.md)"
    )

    broken = [w for w in validate_bundle(OKFBundle(tmp_path)) if w.kind == "broken-link"]

    assert [w.concept_id for w in broken] == ["devices/bpm"]
    assert "devices/corrector.md" in broken[0].detail


def test_links_to_assets_and_directories_are_checked_as_files(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "ring.png").write_bytes(b"\x89PNG")
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="![ring](/assets/ring.png) in [devices](/devices/), not [this](/assets/gone.png).",
    )

    broken = [w for w in validate_bundle(OKFBundle(tmp_path)) if w.kind == "broken-link"]

    assert len(broken) == 1
    assert "assets/gone.png" in broken[0].detail


def test_a_link_out_of_the_bundle_is_external(tmp_path):
    """OKF §6: a target outside the root is external, and is never read."""
    (tmp_path / "secret.md").write_text("outside", encoding="utf-8")
    root = tmp_path / "bundle"
    _write_concept(root, "bpm", frontmatter=_clean_fm("BPM"), body="[out](../secret.md)")

    assert validate_bundle(OKFBundle(root)) == []


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


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "Read [X](facility:SR:DIAG:BPM:01:POSITION:X), [gauge](<facility:SR GAUGE 1>) "
            "and [pump](facility:SR/ION-PUMP01).",
            id="facility_links_are_not_broken_concept_links",
        ),
        pytest.param("![Logo](https://example.org/logo.png)", id="external_images_are_not_flagged"),
    ],
)
def test_a_link_outside_the_bundle_is_not_flagged(tmp_path, body):
    _write_concept(tmp_path, "devices/bpm", frontmatter=_clean_fm("BPM"), body=body)

    assert validate_bundle(OKFBundle(tmp_path)) == []


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _image_kinds(warnings: list[ValidationWarning]) -> list[str]:
    return sorted(w.kind for w in warnings if w.kind.startswith("image-"))


def test_each_image_problem_is_reported_once(tmp_path):
    figures = tmp_path / "devices" / "figures"
    figures.mkdir(parents=True)
    (figures / "ok.png").write_bytes(_png())
    (figures / "notes.png").write_text("not a picture", encoding="utf-8")
    (figures / "huge.png").write_bytes(_png() + b"\0" * MAX_IMAGE_BYTES)
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body=(
            "![Good](figures/ok.png)\n\n"
            "![Gone](figures/gone.png)\n\n"
            "![Notes](figures/notes.png)\n\n"
            "![Huge](figures/huge.png)\n\n"
            "![](figures/ok.png)\n"
        ),
    )

    warnings = validate_bundle(OKFBundle(tmp_path))

    assert _image_kinds(warnings) == [
        "image-alt-empty",
        "image-missing",
        "image-too-large",
        "image-type",
    ]
    assert [w for w in warnings if not w.kind.startswith("image-")] == []
    by_kind = {w.kind: w for w in warnings}
    assert by_kind["image-missing"].concept_id == "devices/bpm"
    assert "figures/gone.png" in by_kind["image-missing"].detail
    assert "figures/notes.png" in by_kind["image-type"].detail
    assert "figures/huge.png" in by_kind["image-too-large"].detail
    assert bundle_health(warnings)["counts"] == {
        "image-alt-empty": 1,
        "image-missing": 1,
        "image-too-large": 1,
        "image-type": 1,
    }


def test_a_missing_image_is_not_also_a_broken_link(tmp_path):
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="![Gone](/assets/gone.png) and ![Also gone](missing.png)",
    )

    warnings = validate_bundle(OKFBundle(tmp_path))

    assert [w.kind for w in warnings] == ["image-missing", "image-missing"]


def test_a_broken_concept_link_beside_an_image_is_still_reported(tmp_path):
    (tmp_path / "devices").mkdir()
    (tmp_path / "devices" / "ok.png").write_bytes(_png())
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="![Ring](ok.png) next to [ghost](/devices/does-not-exist.md).",
    )

    warnings = validate_bundle(OKFBundle(tmp_path))

    assert [w.kind for w in warnings] == ["broken-link"]
    assert (
        warnings[0].detail
        == "'/devices/does-not-exist.md' -> devices/does-not-exist.md (no such file)"
    )


def test_a_link_wrapping_a_missing_image_reports_both(tmp_path):
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="[![alt](a.png)](page.md)",
    )

    warnings = validate_bundle(OKFBundle(tmp_path))

    assert sorted(w.kind for w in warnings) == ["broken-link", "image-missing"]
    broken = next(w for w in warnings if w.kind == "broken-link")
    assert "devices/page.md" in broken.detail
    missing = next(w for w in warnings if w.kind == "image-missing")
    assert "a.png" in missing.detail


def test_a_link_wrapping_a_present_image_checks_only_the_link(tmp_path):
    (tmp_path / "devices").mkdir()
    (tmp_path / "devices" / "a.png").write_bytes(_png())
    _write_concept(tmp_path, "devices/page", frontmatter=_clean_fm("Page"))
    _write_concept(
        tmp_path,
        "devices/bpm",
        frontmatter=_clean_fm("BPM"),
        body="[![alt](a.png)](page.md)",
    )

    assert validate_bundle(OKFBundle(tmp_path)) == []
