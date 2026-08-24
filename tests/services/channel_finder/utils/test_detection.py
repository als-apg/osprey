"""Tests for channel-finder pipeline configuration detection.

`detect_pipeline_config` selects which pipeline (and its database config) a
project is wired for. Four behaviours matter: ``pipeline_mode: graph`` selects
the graph paradigm on the mode alone, since its store is a service rather than
a database file; an explicit file-paradigm ``pipeline_mode`` wins when its
database path is set; a ``pipeline_mode`` naming no known paradigm is rejected
with ``PipelineModeError``; and otherwise detection auto-probes the configured
file paradigms in a fixed precedence
(middle_layer > hierarchical > in_context).
"""

from __future__ import annotations

import pytest

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.services.channel_finder.core.exceptions import PipelineModeError
from osprey.services.channel_finder.utils.detection import detect_pipeline_config

#: The paradigms whose store is a database file, so the ones auto-detection can
#: probe and the ones an explicit mode resolves to a ``database`` config block.
#: ``graph`` is in the paradigm registry but is not one of them --- it answers on
#: the mode alone. ``tests/build/test_mode_registry_single_source.py`` pins the
#: same exclusion for the benchmark harness's paradigm map.
FILE_PARADIGMS = tuple(m for m in VALID_CHANNEL_FINDER_MODES if m != "graph")


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


class TestGraphPipelineMode:
    """``pipeline_mode: graph`` selects the graph paradigm on the mode alone.

    The graph store is a service reached over the network, not a database file
    on disk, so there is no ``database.path`` to probe: detection answers with
    ``("graph", None)`` and callers get their connection details from the
    ``graph`` config block instead.
    """

    def test_graph_mode_returns_graph_and_no_db_config(self):
        cfg = _cfg()
        cfg["channel_finder"]["pipeline_mode"] = "graph"
        assert detect_pipeline_config(cfg) == ("graph", None)

    def test_graph_mode_wins_over_configured_file_pipelines(self):
        # Every file paradigm is configured and middle_layer would win
        # auto-detection, but the explicit graph mode takes all of it.
        cfg = _cfg(
            in_context=_db("/data/ctx.json"),
            hierarchical=_db("/data/hier.json"),
            middle_layer=_db("/data/ml.json"),
        )
        cfg["channel_finder"]["pipeline_mode"] = "graph"
        assert detect_pipeline_config(cfg) == ("graph", None)

    def test_graph_mode_ignores_a_graph_pipeline_block(self):
        # Even if a project spells out a graph pipeline block, detection does
        # not go looking for a path in it.
        cfg = _cfg(graph={"database": {"path": "/data/graph.json"}})
        cfg["channel_finder"]["pipeline_mode"] = "graph"
        assert detect_pipeline_config(cfg) == ("graph", None)


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

    def test_graph_is_never_auto_detected(self):
        # Graph has no database file to find, so it is only ever reached by
        # naming it explicitly --- a project with only a graph block reads as
        # unconfigured to auto-detection.
        cfg = _cfg(graph={"url": "bolt://localhost:7687"})
        assert detect_pipeline_config(cfg) == (None, None)

    def test_graph_block_does_not_displace_a_file_paradigm(self):
        cfg = _cfg(graph={"url": "bolt://localhost:7687"}, in_context=_db("/data/ctx.json"))
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == "in_context"
        assert db["path"] == "/data/ctx.json"


class TestUnknownPipelineMode:
    """An explicit ``pipeline_mode`` that names no known paradigm is an error.

    Falling through to auto-detect would silently hand back whichever other
    pipeline happens to have a database path, so a typo'd or not-yet-supported
    mode would look like it worked.
    """

    def test_unknown_mode_raises(self):
        cfg = _cfg(hierarchical=_db("/data/hier.json"))
        cfg["channel_finder"]["pipeline_mode"] = "nonsense"
        with pytest.raises(PipelineModeError) as excinfo:
            detect_pipeline_config(cfg)
        message = str(excinfo.value)
        assert "nonsense" in message
        for mode in VALID_CHANNEL_FINDER_MODES:
            assert mode in message

    def test_unknown_mode_raises_even_with_no_pipelines_configured(self):
        cfg = _cfg()
        cfg["channel_finder"]["pipeline_mode"] = "hierarchial"  # typo
        with pytest.raises(PipelineModeError):
            detect_pipeline_config(cfg)

    def test_none_mode_still_auto_detects(self):
        cfg = _cfg(in_context=_db("/data/ctx.json"))
        cfg["channel_finder"]["pipeline_mode"] = None
        assert detect_pipeline_config(cfg) == ("in_context", {"path": "/data/ctx.json"})

    def test_absent_mode_still_auto_detects(self):
        cfg = _cfg(in_context=_db("/data/ctx.json"))
        assert "pipeline_mode" not in cfg["channel_finder"]
        assert detect_pipeline_config(cfg) == ("in_context", {"path": "/data/ctx.json"})

    @pytest.mark.parametrize("mode", FILE_PARADIGMS)
    def test_every_file_paradigm_mode_is_accepted(self, mode):
        cfg = _cfg(**{mode: _db(f"/data/{mode}.json")})
        cfg["channel_finder"]["pipeline_mode"] = mode
        ptype, db = detect_pipeline_config(cfg)
        assert ptype == mode
        assert db == {"path": f"/data/{mode}.json"}

    @pytest.mark.parametrize("mode", FILE_PARADIGMS)
    def test_known_mode_without_its_own_path_still_falls_through(self, mode):
        # A known mode whose database path is unset keeps the auto-detect
        # fallback -- only unknown names are rejected.
        other = next(m for m in FILE_PARADIGMS if m != mode)
        cfg = _cfg(**{mode: _db(None), other: _db(f"/data/{other}.json")})
        cfg["channel_finder"]["pipeline_mode"] = mode
        ptype, _db_cfg = detect_pipeline_config(cfg)
        assert ptype == other
