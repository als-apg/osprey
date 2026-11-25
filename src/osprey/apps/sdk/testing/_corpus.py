"""A qmd corpus for an application's tests: a directory and canned hits, no sidecar."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from osprey.apps.sdk._corpus import Corpus, CorpusHit, stand_in

__all__ = ["corpus_stand_in"]


def corpus_stand_in(
    name: str,
    path: str | Path,
    hits: Sequence[CorpusHit] = (),
    *,
    unavailable: bool = False,
) -> AbstractContextManager[Corpus]:
    """Stand a corpus in for the code under test, for as long as the block runs.

    Inside the block :func:`osprey.apps.sdk.corpus` answers *name* with a corpus
    whose ``path`` is *path* and whose ``search`` returns *hits*, cut to the
    asked ``limit``, whatever the query. Nothing is dialled and no environment
    variable is read::

        with corpus_stand_in("procedures", tmp_path, [CorpusHit("a.md", "A", 1.0, 3, "3: ...")]):
            assert client.get("/api/search?q=bakeout").json()["hits"][0]["path"] == "a.md"

    Args:
        name: The corpus name the application asks for.
        path: The directory to serve as the corpus's source.
        hits: What every search returns, best first.
        unavailable: Make every search raise
            :class:`~osprey.apps.sdk.CorpusUnavailable`, as a sidecar that is down does.

    Returns:
        A context manager yielding the stand-in corpus.
    """
    return stand_in(name, Path(path), hits, unavailable=unavailable)
