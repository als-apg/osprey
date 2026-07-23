"""Tests for channel-finder pipeline configuration detection.

`detect_pipeline_config` selects which pipeline (and its database config) a
project is wired for. Two behaviours matter: an explicit ``pipeline_mode``
wins when its database path is set, and otherwise detection auto-probes the
configured pipelines in a fixed precedence (middle_layer > hierarchical >
in_context).
"""

from __future__ import annotations

from osprey.services.channel_finder.utils.detection import detect_pipeline_config


def _cfg(**pipelines) -> dict:
    """Build a config dict with the given pipeline sub-configs."""
    return {"channel_finder": {"pipelines": pipelines}}


def _db(path: str | None) -> dict:
    return {"database": {"path": path}} if path is not None else {"database": {}}


class TestUnconfigured:
    def test_empty_config_returns_none(self):
        assert detect_pipeline_config({}) == (None, None)

    def test_no_paths_returns_none(self):
        cfg = _cfg(hierarchical=_db(None), in_context=_db(None), middle_layer=_db(None))
        assert detect_pipeline_config(cfg) == (None, None)


class TestExplicitPipelineMode:
    def test_explicit_in_context_selected(self):
        cfg = _cfg(in_context=_db("/data/ctx.json"))
        cfg["channel_finder"]["pipeline_mode"] = "in_context"
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "in_context"
        assert db == {"path": "/data/ctx.json"}

    def test_explicit_middle_layer_selected(self):
        cfg = _cfg(middle_layer=_db("/data/ml.json"))
        cfg["channel_finder"]["pipeline_mode"] = "middle_layer"
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "middle_layer"
        assert db["path"] == "/data/ml.json"

    def test_explicit_mode_wins_over_higher_precedence_autodetect(self):
        # middle_layer would win auto-detection, but explicit mode picks hierarchical.
        cfg = _cfg(
            hierarchical=_db("/data/hier.json"),
            middle_layer=_db("/data/ml.json"),
        )
        cfg["channel_finder"]["pipeline_mode"] = "hierarchical"
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "hierarchical"
        assert db["path"] == "/data/hier.json"

    def test_explicit_mode_without_path_falls_through_to_autodetect(self):
        # pipeline_mode names middle_layer, but it has no path -> auto-detect
        # picks the next configured pipeline instead of returning nothing.
        cfg = _cfg(middle_layer=_db(None), hierarchical=_db("/data/hier.json"))
        cfg["channel_finder"]["pipeline_mode"] = "middle_layer"
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "hierarchical"
        assert db["path"] == "/data/hier.json"


class TestAutoDetectPrecedence:
    def test_middle_layer_wins(self):
        cfg = _cfg(
            in_context=_db("/data/ctx.json"),
            hierarchical=_db("/data/hier.json"),
            middle_layer=_db("/data/ml.json"),
        )
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "middle_layer"
        assert db["path"] == "/data/ml.json"

    def test_hierarchical_beats_in_context(self):
        cfg = _cfg(in_context=_db("/data/ctx.json"), hierarchical=_db("/data/hier.json"))
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "hierarchical"

    def test_in_context_last_resort(self):
        cfg = _cfg(in_context=_db("/data/ctx.json"))
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "in_context"
        assert db["path"] == "/data/ctx.json"
