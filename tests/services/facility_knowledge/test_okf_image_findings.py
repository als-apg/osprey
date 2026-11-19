"""Tests for the image findings: what is wrong with the images a bundle embeds.

Every picture here is built at test time; only leading bytes matter, because the
lookup decides a type from them and never decodes.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from osprey.services.facility_knowledge.okf.assets import (
    IMAGE_FINDING_KINDS,
    MAX_IMAGE_BYTES,
    AssetError,
    ImageFinding,
    bundle_image_findings,
    concept_image_findings,
)

_FRONTMATTER = "---\ntype: Device\ntitle: Page\n---\n\n"


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _concept(root: Path, concept_id: str, body: str) -> Path:
    return _write(root / f"{concept_id}.md", _FRONTMATTER + body)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


_SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'


def _findings(root: Path, concept_id: str, body: str) -> list[ImageFinding]:
    _concept(root, concept_id, body)
    return concept_image_findings(root, concept_id, body)


class TestKinds:
    def test_the_four_kinds(self) -> None:
        assert IMAGE_FINDING_KINDS == frozenset(
            {"image-missing", "image-type", "image-too-large", "image-alt-empty"}
        )


class TestBundleProducesEachKindOnce:
    def test_each_kind_exactly_once(self, tmp_path: Path) -> None:
        _write(tmp_path / "devices" / "ok.png", _png())
        _write(tmp_path / "devices" / "noalt.png", _png())
        _write(tmp_path / "devices" / "notes.png", "plain text, not a picture\n")
        png = _png()
        _write(tmp_path / "devices" / "huge.png", png + b"\x00" * (MAX_IMAGE_BYTES - len(png) + 1))
        _concept(
            tmp_path,
            "devices/bpm",
            "![Fine](ok.png)\n\n![](noalt.png)\n\n![Gone](gone.png)\n",
        )
        _concept(tmp_path, "devices/magnet", "![Notes](notes.png)\n\n![Huge](huge.png)\n")
        _write(tmp_path / "index.md", "# Index\n\n![](index-only.png)\n")

        findings = bundle_image_findings(tmp_path)

        assert sorted(f.kind for f in findings) == sorted(IMAGE_FINDING_KINDS)
        by_kind = {f.kind: f for f in findings}
        assert by_kind["image-alt-empty"] == ImageFinding(
            concept_id="devices/bpm",
            target="noalt.png",
            kind="image-alt-empty",
            detail=by_kind["image-alt-empty"].detail,
        )
        assert (by_kind["image-missing"].concept_id, by_kind["image-missing"].target) == (
            "devices/bpm",
            "gone.png",
        )
        assert "not_found" in by_kind["image-missing"].detail
        assert by_kind["image-type"].concept_id == "devices/magnet"
        assert by_kind["image-too-large"].target == "huge.png"
        assert "too_large" in by_kind["image-too-large"].detail

    def test_findings_follow_bundle_then_source_order(self, tmp_path: Path) -> None:
        _concept(tmp_path, "b", "![B1](b1.png) ![B2](b2.png)\n")
        _concept(tmp_path, "a", "![A](a.png)\n")

        targets = [f.target for f in bundle_image_findings(tmp_path)]

        assert targets == ["a.png", "b1.png", "b2.png"]


class TestMissing:
    def test_dot_path_embed(self, tmp_path: Path) -> None:
        _write(tmp_path / ".hidden" / "fig.png", _png())

        [finding] = _findings(tmp_path, "devices/bpm", "![Fig](/.hidden/fig.png)\n")

        assert finding.kind == "image-missing"
        assert "dot_path" in finding.detail

    def test_symlink_out_of_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        outside = _write(tmp_path / "elsewhere" / "fig.png", _png())
        (root / "devices").mkdir(parents=True)
        (root / "devices" / "fig.png").symlink_to(outside)

        [finding] = _findings(root, "devices/bpm", "![Fig](fig.png)\n")

        assert finding.kind == "image-missing"
        assert "symlink_escape" in finding.detail

    def test_path_leaving_the_root(self, tmp_path: Path) -> None:
        [finding] = _findings(tmp_path, "devices/bpm", "![Fig](../../fig.png)\n")

        assert finding.kind == "image-missing"
        assert finding.target == "../../fig.png"
        assert "outside_bundle" in finding.detail

    def test_not_found(self, tmp_path: Path) -> None:
        [finding] = _findings(tmp_path, "devices/bpm", "![Fig](fig.png)\n")

        assert finding == ImageFinding(
            concept_id="devices/bpm", target="fig.png", kind="image-missing", detail=finding.detail
        )
        assert "not_found" in finding.detail


class TestType:
    def test_text_file_named_png(self, tmp_path: Path) -> None:
        _write(tmp_path / "devices" / "fig.png", "just text\n")

        [finding] = _findings(tmp_path, "devices/bpm", "![Fig](fig.png)\n")

        assert finding.kind == "image-type"
        assert "not_an_image" in finding.detail


class TestNoFinding:
    def test_external_image(self, tmp_path: Path) -> None:
        body = "![](https://example.org/x.png)\n\n![x](//cdn.example.org/y.png)\n"

        assert _findings(tmp_path, "devices/bpm", body) == []

    def test_svg_is_a_panel_type(self, tmp_path: Path) -> None:
        _write(tmp_path / "devices" / "diagram.svg", _SVG)

        assert _findings(tmp_path, "devices/bpm", "![Diagram](diagram.svg)\n") == []

    def test_good_raster(self, tmp_path: Path) -> None:
        _write(tmp_path / "devices" / "fig.png", _png())

        assert _findings(tmp_path, "devices/bpm", "![Fig](fig.png)\n") == []

    def test_bundle_without_images(self, tmp_path: Path) -> None:
        _concept(tmp_path, "devices/bpm", "No pictures, [a link](other.md).\n")
        _write(tmp_path / "index.md", "# Index\n")

        assert bundle_image_findings(tmp_path) == []

    def test_empty_bundle(self, tmp_path: Path) -> None:
        assert bundle_image_findings(tmp_path) == []


class TestAltEmpty:
    def test_beside_another_finding(self, tmp_path: Path) -> None:
        findings = _findings(tmp_path, "devices/bpm", "![  ](fig.png)\n")

        assert sorted(f.kind for f in findings) == ["image-alt-empty", "image-missing"]
        assert {f.target for f in findings} == {"fig.png"}

    def test_external_with_empty_alt_gives_nothing(self, tmp_path: Path) -> None:
        assert _findings(tmp_path, "devices/bpm", "![](https://example.org/x.png)\n") == []


class TestResolution:
    def test_symlinked_concept_resolves_from_its_id(self, tmp_path: Path) -> None:
        real = _concept(tmp_path, "library/pump", "![Pump](pump.png)\n")
        (tmp_path / "devices").mkdir()
        (tmp_path / "devices" / "pump.md").symlink_to(real)
        _write(tmp_path / "devices" / "pump.png", _png())

        findings = bundle_image_findings(tmp_path)

        # The real file resolves from library/ (missing); the link from devices/ (found).
        assert [(f.concept_id, f.kind) for f in findings] == [("library/pump", "image-missing")]

    def test_leading_slash_id(self, tmp_path: Path) -> None:
        _concept(tmp_path, "devices/bpm", "![Fig](fig.png)\n")
        _write(tmp_path / "devices" / "fig.png", _png())

        assert concept_image_findings(tmp_path, "/devices/bpm", "![Fig](fig.png)\n") == []

    def test_unknown_concept_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AssetError) as info:
            concept_image_findings(tmp_path, "devices/none", "![Fig](fig.png)\n")
        assert info.value.code == "outside_bundle"


class TestBundleWalk:
    def test_unparseable_concept_is_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "broken.md", "---\ntype: [unclosed\n---\n\n![](gone.png)\n")
        _concept(tmp_path, "good", "![Fig](fig.png)\n")

        findings = bundle_image_findings(tmp_path)

        assert [(f.concept_id, f.kind) for f in findings] == [("good", "image-missing")]

    def test_non_concepts_are_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "README.md", "![](gone.png)\n")
        _write(tmp_path / "log.md", "---\ntype: Log\n---\n\n![](gone.png)\n")
        _write(tmp_path / ".drafts" / "x.md", _FRONTMATTER + "![](gone.png)\n")

        assert bundle_image_findings(tmp_path) == []

    def test_no_decoder_is_called(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "fig.png", _png())
        _concept(tmp_path, "page", "![Fig](fig.png)\n")

        def _refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("an image decoder ran")

        monkeypatch.setattr(Image, "open", _refuse)

        assert bundle_image_findings(tmp_path) == []
