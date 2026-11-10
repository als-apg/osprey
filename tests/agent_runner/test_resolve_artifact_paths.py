"""Tests for ``resolve_artifact_paths``: artifact IDs to attachment-ready file paths."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def store(tmp_path) -> Iterator[ArtifactStore]:
    store = ArtifactStore(workspace_root=tmp_path)
    with patch("osprey.stores.artifact_store.get_artifact_store", return_value=store):
        yield store


def _save(store: ArtifactStore, filename: str, mime: str, body: bytes) -> str:
    entry = store.save_file(
        file_content=body,
        filename=filename,
        artifact_type="file",
        title=filename,
        mime_type=mime,
        tool_source="test",
    )
    return entry.id


async def test_converts_rendered_content_and_passes_images_through_in_order(store, monkeypatch):
    import osprey.stores.artifact_converters as conv
    from osprey.agent_runner.artifact_resolve import resolve_artifact_paths

    async def _fake_html_to_png(source: Path, output_dir: Path) -> Path:
        out = output_dir / f"{source.stem}.png"
        out.write_bytes(PNG_BYTES)
        return out

    monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _fake_html_to_png)

    html_id = _save(store, "plot.html", "text/html", b"<html><body>plot</body></html>")
    png_id = _save(store, "shot.png", "image/png", PNG_BYTES)

    paths = await resolve_artifact_paths([html_id, png_id])

    assert len(paths) == 2
    converted = Path(paths[0])
    assert converted.suffix == ".png"
    assert converted.parent.name.startswith("ariel_convert_")
    assert converted.read_bytes() == PNG_BYTES
    assert Path(paths[1]) == store.get_file_path(png_id)


@pytest.mark.usefixtures("store")
async def test_unknown_artifact_id_raises_value_error():
    from osprey.agent_runner.artifact_resolve import resolve_artifact_paths

    with pytest.raises(ValueError, match="not found"):
        await resolve_artifact_paths(["no-such-artifact"])
