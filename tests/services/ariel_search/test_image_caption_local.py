"""A real caption from a local vision model for the probe orbit plot.

Nothing is faked: the orbit plot is drawn at test time by the probe-plot
script, and the caption module sends it through the real Ollama adapter to a
local Ollama serving ``qwen3-vl:4b``. The test skips when that model is not
pulled, so it never asks for a key and never fails on a machine without it.

The device name drawn on the plot must come back in the stored caption record,
the description together with its visible-text list, which is what the
logbook search indexes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from osprey.models.providers import _local_server
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.image_caption import module as caption_mod
from osprey.services.ariel_search.enhancement.image_caption.module import (
    ImageCaptionModule,
    _reply_text,
    parse_caption_reply,
)
from tests.conftest import ollama_has_model

MODEL = "qwen3-vl:4b"
DEVICE = "SR:C07 BPM"
_PLOT_SCRIPT = (
    Path(__file__).resolve().parents[2] / "fixtures" / "llama_server" / "make_probe_plots.py"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not ollama_has_model(MODEL), reason=f"local Ollama without {MODEL}"),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(caption_mod, "_provider_config", lambda provider: {})
    yield
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()


def _orbit_plot(out_dir: Path) -> bytes:
    spec = importlib.util.spec_from_file_location("make_probe_plots", _PLOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    plots.main(["make_probe_plots.py", str(out_dir)])
    return (out_dir / "orbit_kick.png").read_bytes()


def test_orbit_plot_caption_names_the_bpm(tmp_path):
    module = ImageCaptionModule()
    config: dict[str, Any] = {"enabled": True, "provider": "ollama", "model": {"model_id": MODEL}}
    module.configure(config)
    entry = {"raw_text": "Orbit kick after the fill, see plot."}
    rendition = {"rendition_bytes": _orbit_plot(tmp_path), "rendition_mime": "image/png"}

    reply = module._call(entry, rendition)
    caption, visible_text = parse_caption_reply(_reply_text(reply))

    assert caption
    assert DEVICE in f"{caption}\n{visible_text}", (caption, visible_text)
