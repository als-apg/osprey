"""Fusion of the hybrid text lane with the picture lane.

``fuse_lanes`` is pure: it takes the filter-accepted text hits in qmd's order
and the picture lane's nearest picture per entry, and returns one ranked list
of entry ids with fused scores, the lanes each entry matched through, and the
picture that matched it. Hydration, filtering and windowing stay with the
caller.

Rules:

* Every image hit whose cosine similarity is below ``min_similarity`` is
  dropped before anything else, whether its entry is also a text hit or not.
* When no image hit survives the floor, qmd's order and scores pass through
  unchanged and every entry matched through ``text`` only.
* Otherwise an image-only hit is admitted when its similarity is at least
  ``max(best - relative_margin, min_similarity)``, where ``best`` is the
  highest similarity among all floor-surviving image hits (text-matched or
  image-only); at most ``cap`` image-only hits are admitted, closest first.
* Image ranks count only contributing hits (surviving text-matched hits plus
  admitted image-only hits), ordered by similarity; text ranks are positions in
  the text hit list. Every entry scores the sum of ``1 / (k + rank)`` over the
  lanes it survived in, so a text-only entry scores ``1 / (k + text rank)``
  and qmd's 0-1 scores never mix with reciprocal ranks.
* Entries sort by fused score; at equal score a text-lane entry ranks before
  an image-only one. Scores are then divided by the result's own maximum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Cosine similarity below which a picture match is never trusted.
MIN_SIMILARITY = 0.45

#: How far below the best picture match an image-only hit may fall and still
#: be admitted.
RELATIVE_MARGIN = 0.08

#: Reciprocal-rank-fusion constant.
RRF_K = 60


@dataclass(frozen=True)
class ImageHit:
    """An entry's nearest picture for the query.

    Attributes:
        attachment_id: Id of the entry's picture closest to the query.
        similarity: Cosine similarity between query and picture, ``1 - d``
            where ``d`` is the pgvector cosine distance; higher is closer.
    """

    attachment_id: str
    similarity: float


@dataclass(frozen=True)
class FusedHit:
    """One entry of the fused result.

    Attributes:
        entry_id: The logbook entry id.
        score: qmd's score when no picture survived the floor, else the fused
            score normalised by the result's maximum.
        matched_via: Sorted subset of ``{"image", "text"}``.
        attachment_id: The matching picture's id when the entry's image hit
            survived, else ``None``.
    """

    entry_id: str
    score: float
    matched_via: list[str]
    attachment_id: str | None = None


def fuse_lanes(
    text_hits: Sequence[tuple[str, float]],
    image_hits: Mapping[str, ImageHit],
    *,
    k: int = RRF_K,
    relative_margin: float = RELATIVE_MARGIN,
    min_similarity: float = MIN_SIMILARITY,
    cap: int,
) -> list[FusedHit]:
    """Fuse the text lane with the picture lane.

    Args:
        text_hits: ``(entry_id, qmd score)`` pairs for the filter-accepted text
            hits, in qmd's order, each id once.
        image_hits: The picture lane's nearest picture per entry id.
        k: Reciprocal-rank-fusion constant.
        relative_margin: Admission margin below the best picture match.
        min_similarity: Floor every image hit must reach.
        cap: Most image-only entries admitted (``ceil(max_results / 3)``).

    Returns:
        The fused entries, best first.
    """
    surviving = {
        entry_id: hit for entry_id, hit in image_hits.items() if hit.similarity >= min_similarity
    }
    if not surviving:
        return [FusedHit(entry_id, score, ["text"]) for entry_id, score in text_hits]

    text_rank = {entry_id: rank for rank, (entry_id, _) in enumerate(text_hits, start=1)}
    best = max(hit.similarity for hit in surviving.values())
    threshold = max(best - relative_margin, min_similarity)

    def closest_first(entry_id: str) -> tuple[float, str]:
        return (-surviving[entry_id].similarity, entry_id)

    candidates = sorted(
        (
            entry_id
            for entry_id, hit in surviving.items()
            if entry_id not in text_rank and hit.similarity >= threshold
        ),
        key=closest_first,
    )
    admitted = candidates[: max(cap, 0)]
    contributing = sorted(
        [entry_id for entry_id in surviving if entry_id in text_rank] + admitted,
        key=closest_first,
    )
    image_rank = {entry_id: rank for rank, entry_id in enumerate(contributing, start=1)}

    # (score, lane order, position) — text-lane entries (lane 0) win ties.
    ranked: list[tuple[float, int, int, FusedHit]] = []
    for entry_id, _ in text_hits:
        score = 1.0 / (k + text_rank[entry_id])
        hit = surviving.get(entry_id)
        if hit is not None:
            score += 1.0 / (k + image_rank[entry_id])
            fused = FusedHit(entry_id, score, ["image", "text"], hit.attachment_id)
        else:
            fused = FusedHit(entry_id, score, ["text"])
        ranked.append((score, 0, text_rank[entry_id], fused))
    for entry_id in admitted:
        score = 1.0 / (k + image_rank[entry_id])
        fused = FusedHit(entry_id, score, ["image"], surviving[entry_id].attachment_id)
        ranked.append((score, 1, image_rank[entry_id], fused))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    top = ranked[0][0] if ranked else 1.0
    return [
        FusedHit(hit.entry_id, score / top, hit.matched_via, hit.attachment_id)
        for score, _, _, hit in ranked
    ]
