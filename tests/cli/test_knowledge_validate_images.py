"""``osprey knowledge validate`` reports image problems as warnings.

An image a concept embeds that is missing, not a picture, too large, or has no
alt text is named in one warning block, one line per finding.  Warnings never
change the exit code.
"""

from __future__ import annotations

import io
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from osprey.cli.knowledge_cmd import knowledge
from osprey.services.facility_knowledge.okf.assets import IMAGE_FINDING_KINDS, MAX_IMAGE_BYTES

_FRONTMATTER = "---\ntype: Device\ntitle: {title}\ndescription: A device page.\n---\n\n"


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _image_bundle(root: Path) -> Path:
    """A bundle whose concepts embed one image of each problem kind, plus a good one."""
    png = _png()
    _write(root / "devices" / "ok.png", png)
    _write(root / "devices" / "noalt.png", png)
    _write(root / "devices" / "notes.png", "plain text, not a picture\n")
    _write(root / "devices" / "huge.png", png + b"\x00" * (MAX_IMAGE_BYTES - len(png) + 1))
    _write(
        root / "devices" / "bpm.md",
        _FRONTMATTER.format(title="BPM")
        + "![Fine](ok.png)\n\n![](noalt.png)\n\n![Gone](gone.png)\n",
    )
    _write(
        root / "devices" / "magnet.md",
        _FRONTMATTER.format(title="Magnet") + "![Notes](notes.png)\n\n![Huge](huge.png)\n",
    )
    return root


def _regen(bundle: Path) -> None:
    result = CliRunner().invoke(knowledge, ["regen-index", str(bundle)])
    assert result.exit_code == 0, result.output


def _finding_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if any(k in line for k in IMAGE_FINDING_KINDS)]


def test_four_image_problems_print_four_lines_and_exit_zero(tmp_path: Path) -> None:
    bundle = _image_bundle(tmp_path / "bundle")
    _regen(bundle)

    result = CliRunner().invoke(knowledge, ["validate", str(bundle)])

    assert result.exit_code == 0, result.output
    assert "valid" in result.output
    lines = _finding_lines(result.output)
    assert len(lines) == 4, result.output
    assert "4 image problem(s)" in result.output

    by_kind = {kind: [line for line in lines if kind in line] for kind in IMAGE_FINDING_KINDS}
    assert all(len(found) == 1 for found in by_kind.values()), result.output
    assert "devices/bpm" in by_kind["image-alt-empty"][0]
    assert "noalt.png" in by_kind["image-alt-empty"][0]
    assert "devices/bpm" in by_kind["image-missing"][0]
    assert "gone.png" in by_kind["image-missing"][0]
    assert "devices/magnet" in by_kind["image-type"][0]
    assert "notes.png" in by_kind["image-type"][0]
    assert "devices/magnet" in by_kind["image-too-large"][0]
    assert "huge.png" in by_kind["image-too-large"][0]
    assert "ok.png" not in result.output


def test_the_shipped_bundle_prints_no_warnings() -> None:
    import osprey

    bundle = Path(osprey.__file__).parent / "templates/facilities/example/knowledge"

    result = CliRunner().invoke(knowledge, ["validate", str(bundle)])

    assert result.exit_code == 0, result.output
    assert "image problem" not in result.output
    assert _finding_lines(result.output) == []
    assert "warning" not in result.output.lower(), result.output


def test_a_failing_bundle_still_exits_one_and_lists_images(tmp_path: Path) -> None:
    bundle = _image_bundle(tmp_path / "bundle")
    _regen(bundle)
    # A concept without a description fails the authoring-level check.
    _write(bundle / "devices" / "bad.md", "---\ntype: Device\ntitle: Bad\n---\n\nBody.\n")
    _regen(bundle)

    result = CliRunner().invoke(knowledge, ["validate", str(bundle)])

    assert result.exit_code == 1, result.output
    assert "failed validation" in result.output
    assert len(_finding_lines(result.output)) == 4, result.output
