"""Tests for the facility knowledge ``view_asset`` tool.

Every picture is generated at test time into a temporary bundle. The tool runs
in process against that bundle; the pictures pass the real isolated renderer,
so these tests also prove the byte budget reaches it.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from PIL import Image

from tests.mcp_server._facility_bundle import install_bundle, write_minimal_bundle
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

#: The full tool name an agent sees (C6).
_FULL_NAME = "mcp__osprey_facility_knowledge__view_asset"

_CAP = 512 * 1024


def _encode(im: Image.Image, fmt: str, **params) -> bytes:
    out = io.BytesIO()
    im.save(out, fmt, **params)
    return out.getvalue()


def _noise(mode: str, size: tuple[int, int]) -> Image.Image:
    return Image.frombytes(mode, size, os.urandom(len(mode) * size[0] * size[1]))


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch) -> Path:
    root = write_minimal_bundle(tmp_path / "bundle")
    install_bundle(monkeypatch, root)
    return root


def _put(root: Path, rel: str, data: bytes) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


async def _view(path: str) -> ToolResult:
    import osprey.mcp_server.facility_knowledge.server as srv

    return await get_tool_fn(srv.view_asset)(path=path)


def _parts(result: ToolResult) -> tuple[dict, ImageContent]:
    assert isinstance(result, ToolResult)
    text, image = result.content
    assert isinstance(text, TextContent)
    assert isinstance(image, ImageContent)
    return json.loads(text.text), image


def _sent(image: ImageContent) -> bytes:
    return base64.b64decode(image.data)


def _sent_size(image: ImageContent) -> tuple[int, int]:
    with Image.open(io.BytesIO(_sent(image))) as im:
        return im.size


async def _refusal(path: str, error_type: str) -> dict:
    with assert_raises_error(error_type=error_type) as ctx:
        await _view(path)
    return ctx["envelope"]


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fmt", "type_"),
    [
        ("a.png", "PNG", "png"),
        ("a.jpg", "JPEG", "jpeg"),
        ("a.gif", "GIF", "gif"),
        ("a.webp", "WEBP", "webp"),
    ],
)
async def test_each_agent_type_returns_an_image_block(bundle: Path, name, fmt, type_):
    data = _encode(Image.new("RGB", (40, 30), (200, 30, 30)), fmt)
    rel = _put(bundle, f"figures/{name}", data)

    payload, image = _parts(await _view(rel))

    assert payload["path"] == rel
    assert payload["type"] == type_
    assert payload["bytes"] == len(data)
    assert set(payload) == {"path", "type", "bytes", "width", "height", "scaled"}
    assert image.mimeType in {"image/png", "image/jpeg", "image/gif", "image/webp"}
    assert _sent(image)
    assert _sent_size(image) == (payload["width"], payload["height"])


async def test_small_png_is_sent_as_stored(bundle: Path):
    data = _encode(Image.new("RGB", (64, 48), (10, 120, 200)), "PNG")
    rel = _put(bundle, "figures/small.png", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is False
    assert (payload["width"], payload["height"]) == (64, 48)
    assert image.mimeType == "image/png"
    assert _sent(image) == data


async def test_leading_slash_is_accepted(bundle: Path):
    data = _encode(Image.new("RGB", (64, 48), (10, 120, 200)), "PNG")
    _put(bundle, "figures/small.png", data)

    payload, image = _parts(await _view("/figures/small.png"))

    assert payload["path"] == "figures/small.png"
    assert payload["scaled"] is False
    assert _sent(image) == data


async def test_large_photographic_png_is_scaled_within_the_cap(bundle: Path):
    data = _encode(_noise("RGB", (1500, 1400)), "PNG")
    assert len(data) > 6 * 1000 * 1000
    rel = _put(bundle, "figures/photo.png", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is True
    assert payload["bytes"] == len(data)
    assert len(_sent(image)) <= _CAP
    assert _sent_size(image) == (payload["width"], payload["height"])


async def test_png_with_alpha_is_scaled_within_the_cap(bundle: Path):
    data = _encode(_noise("RGBA", (700, 700)), "PNG")
    assert len(data) > _CAP
    rel = _put(bundle, "figures/alpha.png", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is True
    assert len(_sent(image)) <= _CAP
    assert _sent_size(image) == (payload["width"], payload["height"])


async def test_small_bytes_png_wider_than_the_maximum_side_is_scaled(bundle: Path):
    from osprey.imaging.formats import RENDITION_MAX_SIDE

    data = _encode(Image.new("RGB", (RENDITION_MAX_SIDE * 3, 30), (0, 0, 0)), "PNG")
    assert len(data) < _CAP
    rel = _put(bundle, "figures/wide.png", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is True
    assert payload["width"] == RENDITION_MAX_SIDE
    assert _sent_size(image) == (payload["width"], payload["height"])
    assert _sent(image) != data


async def test_jpeg_with_rotating_orientation_is_scaled(bundle: Path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    data = _encode(Image.new("RGB", (64, 32), (90, 90, 90)), "JPEG", exif=exif.tobytes())
    rel = _put(bundle, "figures/rotated.jpg", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is True
    assert (payload["width"], payload["height"]) == (32, 64)
    assert _sent_size(image) == (32, 64)


async def test_small_cmyk_jpeg_is_scaled(bundle: Path):
    data = _encode(Image.new("CMYK", (16, 16), (10, 20, 30, 40)), "JPEG")
    rel = _put(bundle, "figures/cmyk.jpg", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is True
    assert (payload["width"], payload["height"]) == (16, 16)
    assert _sent_size(image) == (16, 16)
    assert _sent(image) != data


async def test_file_within_cap_whose_reencoding_exceeds_it_is_sent_as_stored(bundle: Path):
    data = _encode(_noise("RGB", (1000, 1000)), "JPEG", quality=30)
    assert len(data) <= _CAP
    assert len(_encode(Image.open(io.BytesIO(data)), "JPEG", quality=85)) > _CAP
    rel = _put(bundle, "figures/dense.jpg", data)

    payload, image = _parts(await _view(rel))

    assert payload["scaled"] is False
    assert (payload["width"], payload["height"]) == (1000, 1000)
    assert image.mimeType == "image/jpeg"
    assert _sent(image) == data


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_percent_encoded_dot_dot_is_not_decoded(bundle: Path):
    outside = bundle.parent / "secret.png"
    outside.write_bytes(_encode(Image.new("RGB", (8, 8)), "PNG"))

    envelope = await _refusal("%2e%2e/secret.png", "not_found")

    assert envelope["details"]["code"] == "not_found"


@pytest.mark.usefixtures("bundle")
async def test_missing_file_is_not_found():
    envelope = await _refusal("figures/nothing.png", "not_found")
    assert envelope["details"]["code"] == "not_found"


async def test_over_limit_file_is_too_large(bundle: Path):
    png = _encode(Image.new("RGB", (4, 3)), "PNG")
    rel = _put(bundle, "figures/huge.png", png + b"\0" * (11 * 1024 * 1024))

    envelope = await _refusal(rel, "validation_error")

    assert envelope["details"]["code"] == "too_large"


async def test_svg_is_refused_with_the_panel_named(bundle: Path):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'
    rel = _put(bundle, "figures/plot.svg", svg)

    envelope = await _refusal(rel, "validation_error")

    assert envelope["details"]["code"] == "svg"
    assert "panel" in envelope["error_message"].lower()


async def test_corrupt_png_cannot_render(bundle: Path):
    png = _encode(Image.new("RGB", (40, 40), (1, 2, 3)), "PNG")
    rel = _put(bundle, "figures/corrupt.png", png[:40] + b"\x00garbage" * 20)

    envelope = await _refusal(rel, "validation_error")

    assert envelope["details"]["code"] == "cannot_render"
    assert envelope["details"].get("reason")


async def test_path_outside_the_bundle(bundle: Path):
    outside = bundle.parent / "secret.png"
    outside.write_bytes(_encode(Image.new("RGB", (8, 8)), "PNG"))

    envelope = await _refusal("../secret.png", "validation_error")

    assert envelope["details"]["code"] == "outside_bundle"


async def test_dot_path(bundle: Path):
    rel = _put(bundle, ".hidden/a.png", _encode(Image.new("RGB", (8, 8)), "PNG"))

    envelope = await _refusal(rel, "validation_error")

    assert envelope["details"]["code"] == "dot_path"


async def test_symlink_out_of_the_root(bundle: Path):
    outside = bundle.parent / "secret.png"
    outside.write_bytes(_encode(Image.new("RGB", (8, 8)), "PNG"))
    (bundle / "figures").mkdir(exist_ok=True)
    (bundle / "figures" / "link.png").symlink_to(outside)

    envelope = await _refusal("figures/link.png", "validation_error")

    assert envelope["details"]["code"] == "symlink_escape"


async def test_non_image(bundle: Path):
    rel = _put(bundle, "figures/notes.png", b"just some text, not a picture\n" * 4)

    envelope = await _refusal(rel, "validation_error")

    assert envelope["details"]["code"] == "not_an_image"


async def test_unavailable_renderer_is_not_cannot_render(bundle: Path, monkeypatch):
    import osprey.imaging.render as render

    async def _unavailable(*args, **kwargs):
        raise render.RenderUnavailable("render worker exited before its ready line")

    monkeypatch.setattr(render, "render_isolated", _unavailable)
    rel = _put(bundle, "figures/small.png", _encode(Image.new("RGB", (8, 8)), "PNG"))

    envelope = await _refusal(rel, "internal_error")

    assert (envelope.get("details") or {}).get("code") != "cannot_render"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered_tool_name_is_the_contract_name():
    import osprey.mcp_server.facility_knowledge.server as srv
    from osprey.registry.mcp import FRAMEWORK_SERVERS
    from tests.mcp_server.conftest import registered_tool_names

    assert "view_asset" in registered_tool_names(srv.mcp)
    definition = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
    assert f"mcp__{definition.name}__view_asset" == _FULL_NAME
    assert "view_asset" in definition.permissions_allow
    assert "view_asset" not in definition.permissions_ask
