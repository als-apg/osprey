"""Tests for the fusion calibration script's pure functions.

The live part (query vectors from llama-server, nearest pictures from the
database) is a thin wrapper over the production SQL and adapter; the logic
worth protecting is the set validation, the precision of admitted image-only
hits under the shipped admission rule, and the choice of (margin, floor).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from osprey.services.ariel_search.search import fusion

# import-time required because scripts/ is not a package: fusion_calibration.py is
# loaded by path and registered in sys.modules before exec so @dataclass can resolve
# cls.__module__.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "fusion_calibration.py"
)
_spec = importlib.util.spec_from_file_location("fusion_calibration", _MODULE_PATH)
fc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fc
_spec.loader.exec_module(fc)


def _query(hits, relevant, text=()):
    return fc.LabelledQuery(
        query="q", relevant=frozenset(relevant), text_hits=tuple(text), image_hits=dict(hits)
    )


class TestLoadSet:
    def test_reads_queries_relevant_and_text_hits(self, tmp_path):
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps(
                {
                    "export": "store-2026",
                    "queries": [
                        {"query": "a", "relevant": ["e1", "e2"], "text_hits": ["e3"]},
                        {"query": "b", "relevant": ["e4"]},
                    ],
                }
            )
        )
        labelled = fc.load_set(path)
        assert labelled.export == "store-2026"
        assert [q.query for q in labelled.queries] == ["a", "b"]
        assert labelled.queries[0].relevant == {"e1", "e2"}
        assert labelled.queries[0].text_hits == ("e3",)
        assert labelled.queries[1].text_hits == ()

    def test_query_without_relevant_entries_is_rejected(self, tmp_path):
        path = tmp_path / "set.json"
        path.write_text(json.dumps({"export": "x", "queries": [{"query": "a", "relevant": []}]}))
        with pytest.raises(ValueError, match="relevant"):
            fc.load_set(path)


class TestSetProblems:
    def test_large_enough_set_has_no_problems(self):
        assert fc.set_problems(pictures=100, labelled_queries=30) == []

    def test_too_few_pictures_and_queries_are_both_named(self):
        problems = fc.set_problems(pictures=99, labelled_queries=29)
        assert len(problems) == 2
        assert any("99" in p and "100" in p for p in problems)
        assert any("29" in p and "30" in p for p in problems)


class TestAdmittedImageOnly:
    def test_uses_the_shipped_rule_floor_and_margin(self):
        hits = {"a": 0.60, "b": 0.53, "c": 0.51, "d": 0.40}
        # threshold = max(0.60 - 0.08, 0.45) = 0.52
        assert fc.admitted_image_only((), hits, margin=0.08, floor=0.45, cap=10) == ["a", "b"]

    def test_text_hits_are_never_image_only(self):
        hits = {"a": 0.60, "b": 0.58}
        assert fc.admitted_image_only(("a",), hits, margin=0.08, floor=0.45, cap=10) == ["b"]

    def test_cap_limits_admissions_closest_first(self):
        hits = {"a": 0.60, "b": 0.59, "c": 0.58}
        assert fc.admitted_image_only((), hits, margin=0.08, floor=0.45, cap=2) == ["a", "b"]

    def test_matches_fuse_lanes_directly(self):
        hits = {"a": 0.70, "b": 0.66, "c": 0.47, "t": 0.71}
        image_hits = {k: fusion.ImageHit(f"att-{k}", v) for k, v in hits.items()}
        fused = fusion.fuse_lanes([("t", 1.0)], image_hits, cap=10)
        expected = sorted(h.entry_id for h in fused if h.matched_via == ["image"])
        got = fc.admitted_image_only(
            ("t",), hits, margin=fusion.RELATIVE_MARGIN, floor=fusion.MIN_SIMILARITY, cap=10
        )
        assert sorted(got) == expected


class TestPrecision:
    def test_precision_over_all_queries(self):
        queries = [
            _query({"a": 0.60, "b": 0.55}, relevant={"a"}),
            _query({"c": 0.70}, relevant={"c"}),
        ]
        row = fc.evaluate(queries, margin=0.08, floor=0.45, cap=10)
        assert row["admitted"] == 3
        assert row["relevant_admitted"] == 2
        assert row["precision"] == pytest.approx(2 / 3)

    def test_nothing_admitted_has_no_precision(self):
        row = fc.evaluate([_query({"a": 0.30}, relevant={"a"})], margin=0.08, floor=0.45, cap=10)
        assert row["admitted"] == 0
        assert row["precision"] is None


class TestGrid:
    def test_parses_inclusive_range(self):
        assert fc.parse_grid("0.40:0.50:0.05") == pytest.approx([0.40, 0.45, 0.50])

    def test_single_value(self):
        assert fc.parse_grid("0.08") == [0.08]

    @pytest.mark.parametrize("spec", ["0.5:0.4:0.05", "0.4:0.5:0", "a:b:c"])
    def test_bad_specs_are_rejected(self, spec):
        with pytest.raises(ValueError):
            fc.parse_grid(spec)


class TestChoose:
    def test_picks_the_passing_point_with_most_relevant_admissions(self):
        queries = [
            _query({"a": 0.60, "b": 0.50, "c": 0.48}, relevant={"a", "b"}),
        ]
        result = fc.sweep(queries, margins=[0.08, 0.12], floors=[0.45, 0.49], cap=10, threshold=0.6)
        # (0.12, 0.45) admits a, b, c (2/3) and (0.12, 0.49) admits a, b (2/2):
        # both admit two relevant hits, the higher precision wins.
        assert result["chosen"]["relative_margin"] == pytest.approx(0.12)
        assert result["chosen"]["min_similarity"] == pytest.approx(0.49)
        assert result["chosen"]["precision"] == pytest.approx(1.0)
        assert result["chosen"]["passes"] is True
        assert len(result["grid"]) == 4

    def test_ties_prefer_the_shipped_defaults(self):
        queries = [_query({"a": 0.90}, relevant={"a"})]
        result = fc.sweep(
            queries,
            margins=[0.05, fusion.RELATIVE_MARGIN],
            floors=[fusion.MIN_SIMILARITY, 0.50],
            cap=10,
            threshold=0.5,
        )
        assert result["chosen"]["relative_margin"] == pytest.approx(fusion.RELATIVE_MARGIN)
        assert result["chosen"]["min_similarity"] == pytest.approx(fusion.MIN_SIMILARITY)

    def test_no_passing_point_keeps_the_defaults(self):
        queries = [_query({"a": 0.90, "b": 0.88}, relevant={"x"})]
        result = fc.sweep(queries, margins=[0.08], floors=[0.45], cap=10, threshold=0.9)
        assert result["chosen"]["passes"] is False
        assert result["chosen"]["relative_margin"] == pytest.approx(fusion.RELATIVE_MARGIN)
        assert result["chosen"]["min_similarity"] == pytest.approx(fusion.MIN_SIMILARITY)


class TestMain:
    def test_too_small_set_exits_1_without_measuring(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps({"export": "tiny", "queries": [{"query": "a", "relevant": ["e1"]}]})
        )
        monkeypatch.setattr(fc, "count_pictures", lambda dsn, table: 5)

        def _no_measure(*args, **kwargs):
            raise AssertionError("must not measure a set that is too small")

        monkeypatch.setattr(fc, "measure", _no_measure)
        code = fc.main(["--set", str(path), "--dsn", "postgresql://x", "--threshold", "0.8"])
        assert code == 1
        out = json.loads(capsys.readouterr().out)
        assert out["export"] == "tiny"
        assert len(out["problems"]) == 2

    def test_threshold_is_required(self, tmp_path):
        with pytest.raises(SystemExit):
            fc.main(["--set", str(tmp_path / "s.json"), "--dsn", "postgresql://x"])
