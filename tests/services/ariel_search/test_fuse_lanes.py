"""Tests for the pure fusion of the hybrid text lane with the picture lane."""

from __future__ import annotations

import inspect
import math

import pytest

from osprey.services.ariel_search.search import fusion
from osprey.services.ariel_search.search.fusion import (
    MIN_SIMILARITY,
    RELATIVE_MARGIN,
    FusedHit,
    ImageHit,
    fuse_lanes,
)


def _img(entry_id: str, similarity: float) -> ImageHit:
    return ImageHit(f"pic-{entry_id}", similarity)


def _order(result: list[FusedHit]) -> list[str]:
    return [hit.entry_id for hit in result]


def _by_id(result: list[FusedHit]) -> dict[str, FusedHit]:
    return {hit.entry_id: hit for hit in result}


class TestDefaults:
    def test_module_constants(self):
        assert MIN_SIMILARITY == 0.45
        assert RELATIVE_MARGIN == 0.08

    def test_constants_are_the_keyword_defaults(self):
        params = inspect.signature(fuse_lanes).parameters
        assert params["min_similarity"].default is fusion.MIN_SIMILARITY
        assert params["relative_margin"].default is fusion.RELATIVE_MARGIN
        assert params["k"].default == 60
        assert params["cap"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_image_hit_carries_cosine_similarity(self):
        hit = ImageHit("att-1", 1 - 0.30)
        assert hit.attachment_id == "att-1"
        assert hit.similarity == pytest.approx(0.70)
        assert "similarity" in (ImageHit.__doc__ or "").lower()


class TestPassThrough:
    def test_no_image_hits_is_identity(self):
        text = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
        result = fuse_lanes(text, {}, cap=4)
        assert [(h.entry_id, h.score) for h in result] == text
        assert all(h.matched_via == ["text"] for h in result)
        assert all(h.attachment_id is None for h in result)

    def test_all_image_hits_below_floor_pass_qmd_through(self):
        text = [("A", 0.93), ("B", 0.71), ("C", 0.2)]
        result = fuse_lanes(text, {"A": _img("A", 0.40), "C": _img("C", 0.41)}, cap=4)
        assert [(h.entry_id, h.score) for h in result] == text
        assert all(h.attachment_id is None for h in result)
        assert all(h.matched_via == ["text"] for h in result)

    def test_image_only_hits_below_floor_pass_qmd_through(self):
        text = [("A", 0.8), ("B", 0.4)]
        result = fuse_lanes(text, {"X": _img("X", 0.44)}, cap=4)
        assert [(h.entry_id, h.score) for h in result] == text

    def test_empty_inputs(self):
        assert fuse_lanes([], {}, cap=4) == []


class TestFloor:
    def test_fused_hit_below_floor_gets_no_image(self):
        text = [("A", 0.9), ("B", 0.8)]
        result = _by_id(fuse_lanes(text, {"A": _img("A", 0.44), "B": _img("B", 0.60)}, cap=4))
        assert result["A"].matched_via == ["text"]
        assert result["A"].attachment_id is None
        assert result["B"].matched_via == ["image", "text"]
        assert result["B"].attachment_id == "pic-B"

    def test_floor_is_inclusive(self):
        result = _by_id(fuse_lanes([("A", 0.9)], {"A": _img("A", 0.45)}, cap=4))
        assert result["A"].matched_via == ["image", "text"]

    def test_similarity_from_distance(self):
        # distance 0.30 -> similarity 0.70 admitted; distance 0.70 -> 0.30 dropped.
        result = _by_id(
            fuse_lanes(
                [("T1", 0.9)],
                {"N": ImageHit("near", 1 - 0.30), "F": ImageHit("far", 1 - 0.70)},
                cap=4,
            )
        )
        assert "N" in result and result["N"].attachment_id == "near"
        assert "F" not in result


class TestFusedScores:
    def test_text_only_entries_score_by_reciprocal_rank_not_qmd(self):
        text = [("A", 0.99), ("B", 0.98), ("C", 0.01)]
        result = _by_id(fuse_lanes(text, {"B": _img("B", 0.9)}, cap=4))
        top = 1 / 62 + 1 / 61
        assert result["B"].score == pytest.approx(1.0)
        assert result["A"].score == pytest.approx((1 / 61) / top)
        assert result["C"].score == pytest.approx((1 / 63) / top)

    def test_scores_normalised_by_own_maximum(self):
        result = fuse_lanes([("A", 0.3), ("B", 0.2)], {"A": _img("A", 0.6)}, cap=4)
        assert result[0].score == 1.0
        assert all(0 < h.score <= 1.0 for h in result)

    def test_k_is_the_rrf_constant(self):
        result = _by_id(fuse_lanes([("A", 0.9), ("B", 0.8)], {"A": _img("A", 0.6)}, k=10, cap=4))
        assert result["B"].score == pytest.approx((1 / 12) / (1 / 11 + 1 / 11))

    def test_image_rank_reorders_text(self):
        text = [("A", 0.9), ("B", 0.8)]
        result = fuse_lanes(text, {"B": _img("B", 0.7)}, cap=4)
        assert _order(result) == ["B", "A"]


class TestAdmission:
    def test_image_only_needs_best_minus_margin(self):
        text = [("T1", 0.9)]
        hits = {"X": _img("X", 0.80), "Y": _img("Y", 0.73), "Z": _img("Z", 0.71)}
        result = _by_id(fuse_lanes(text, hits, cap=4))
        assert "X" in result and "Y" in result
        assert "Z" not in result
        assert result["X"].matched_via == ["image"]
        assert result["X"].attachment_id == "pic-X"

    def test_floor_bounds_the_margin_threshold(self):
        # best 0.50 - 0.08 = 0.42 < floor 0.45: the floor decides.
        hits = {"X": _img("X", 0.50), "Y": _img("Y", 0.46), "Z": _img("Z", 0.44)}
        result = _by_id(fuse_lanes([("T1", 0.9)], hits, cap=4))
        assert set(result) == {"T1", "X", "Y"}

    def test_best_includes_fused_hits(self):
        # A fused hit at 0.90 sets best; image-only X at 0.80 falls outside the margin.
        result = _by_id(
            fuse_lanes([("A", 0.9)], {"A": _img("A", 0.90), "X": _img("X", 0.80)}, cap=4)
        )
        assert "X" not in result

    def test_cap_keeps_the_closest(self):
        hits = {f"I{i}": _img(f"I{i}", 0.80 - i * 0.01) for i in range(6)}
        result = fuse_lanes([("T1", 0.9)], hits, cap=math.ceil(10 / 3))
        image_only = [h.entry_id for h in result if h.matched_via == ["image"]]
        assert sorted(image_only) == ["I0", "I1", "I2", "I3"]

    def test_cap_zero_admits_no_image_only(self):
        result = fuse_lanes([("T1", 0.9)], {"X": _img("X", 0.9)}, cap=0)
        assert _order(result) == ["T1"]

    def test_rejected_image_only_takes_no_rank_slot(self):
        # A (text+image) 0.50, X (image-only) 0.70, Y (image-only) 0.55, margin 0.08.
        # Admission threshold 0.62: Y is out, so image ranks are X=1, A=2.
        text = [("A", 0.9)]
        hits = {"A": _img("A", 0.50), "X": _img("X", 0.70), "Y": _img("Y", 0.55)}
        result = fuse_lanes(text, hits, relative_margin=0.08, cap=4)
        by_id = _by_id(result)
        assert "Y" not in by_id
        a_raw = 1 / 61 + 1 / 62
        x_raw = 1 / 61
        assert by_id["A"].score == pytest.approx(1.0)
        assert by_id["X"].score == pytest.approx(x_raw / a_raw)

    def test_capped_out_image_only_takes_no_rank_slot(self):
        text = [("A", 0.9)]
        hits = {"X": _img("X", 0.80), "Y": _img("Y", 0.79), "A": _img("A", 0.78)}
        by_id = _by_id(fuse_lanes(text, hits, cap=1))
        assert "Y" not in by_id
        # Image ranks X=1, A=2 (Y did not contribute).
        assert by_id["X"].score == pytest.approx((1 / 61) / (1 / 61 + 1 / 62))


class TestTies:
    def test_image_only_ranks_after_text_of_equal_score(self):
        # T2 text rank 2 scores 1/62; I image rank 2 scores 1/62.
        text = [("T1", 0.9), ("T2", 0.8)]
        hits = {"T1": _img("T1", 0.80), "I": _img("I", 0.79)}
        result = fuse_lanes(text, hits, cap=4)
        assert _order(result) == ["T1", "T2", "I"]


class TestRequirementFourFixture:
    def test_explicit_fixture_order_and_lanes(self):
        text = [("T1", 0.9), ("ORB", 0.8), ("T3", 0.7)]
        hits = {"ORB": _img("ORB", 0.70), "I2": _img("I2", 0.68), "I3": _img("I3", 0.66)}
        result = fuse_lanes(
            text,
            hits,
            min_similarity=0.45,
            relative_margin=0.08,
            cap=math.ceil(10 / 3),
        )
        assert _order(result) == ["ORB", "T1", "I2", "T3", "I3"]
        by_id = _by_id(result)
        assert by_id["ORB"].matched_via == ["image", "text"]
        assert by_id["I2"].matched_via == ["image"]
        assert by_id["I3"].matched_via == ["image"]
        assert by_id["T1"].matched_via == ["text"]
        assert by_id["T3"].matched_via == ["text"]
        assert {e: by_id[e].attachment_id for e in ("ORB", "I2", "I3")} == {
            "ORB": "pic-ORB",
            "I2": "pic-I2",
            "I3": "pic-I3",
        }
        assert by_id["T1"].attachment_id is None
        top = 1 / 62 + 1 / 61
        assert by_id["ORB"].score == 1.0
        assert by_id["T3"].score == by_id["I3"].score == pytest.approx((1 / 63) / top)
