"""The qmd corpora an application container declared, as its code reads and searches them.

A ``runtime: osprey`` container that lists ``corpora: [<name>, ...]`` in
``app.yml`` is handed each corpus twice: its source directory, bound read-only,
and the URL of the sidecar that indexes it. The render names both in the
container's environment (``OSPREY_APP_CORPORA`` and one
``OSPREY_QMD_<NAME>_URL`` per corpus), and :func:`corpus` is the one reader.

A tool server runs beside the agent, not in the container, and is handed
neither: it asks the application's backend.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from osprey.deployment.qmd_service import QMDServiceConfig, corpus_url_env

#: The variable the runtime template names the bound corpora in:
#: ``<name>=<directory>`` entries joined by ``:``.
CORPORA_ENV_VAR = "OSPREY_APP_CORPORA"


class CorpusUnavailable(RuntimeError):
    """A corpus's search sidecar gave no answer.

    The sidecar is down, is still building its first index (it opens its port
    only once the index exists), or answered with an error. The corpus's files
    are unaffected: :attr:`Corpus.path` still reads.

    Stable: part of the application SDK's public surface.
    """


@dataclass(frozen=True)
class CorpusHit:
    """One ranked hit of :meth:`Corpus.search`.

    Stable: part of the application SDK's public surface.

    Attributes:
        path: The document, relative to :attr:`Corpus.path`. The sidecar reports
            a normalised spelling of the name it indexed (``_`` comes back as
            ``-``); this is the file on disk whenever exactly one file matches
            it, and the sidecar's spelling otherwise.
        title: The document's first heading, as indexed.
        score: Relevance in 0-1, an ordering signal within one result list. It
            is not calibrated: do not threshold it or compare it across queries.
        line: The 1-indexed best-matching line.
        snippet: A line-numbered excerpt around the match.
    """

    path: str
    title: str
    score: float
    line: int
    snippet: str


@dataclass(frozen=True)
class Corpus:
    """One declared corpus: its files and its search.

    Stable: part of the application SDK's public surface.

    Attributes:
        name: The corpus's name, as ``services.qmd.corpora`` declares it.
        path: The corpus's source directory, read-only in a container.
    """

    name: str
    path: Path
    _search: Callable[[str, int], list[CorpusHit]] = field(repr=False, compare=False)

    def search(self, query: str, *, limit: int = 10) -> list[CorpusHit]:
        """Search the corpus and return its ranked hits.

        Args:
            query: The query text.
            limit: The most hits to return.

        Returns:
            The hits, best first; empty when nothing matched.

        Raises:
            CorpusUnavailable: The sidecar gave no answer.
        """
        return self._search(query, limit)


_corpora: dict[str, Corpus] = {}
_stand_ins: dict[str, Corpus] = {}


def corpus(name: str) -> Corpus:
    """The corpus *name*, which this container declares in ``app.yml``.

    Stable: part of the application SDK's public surface.

    Args:
        name: A name in the container's ``corpora`` list.

    Raises:
        LookupError: This process was handed no corpus of that name: the
            container's ``containers.<key>.corpora`` does not list it, or the
            process is not a ``runtime: osprey`` container.
    """
    stand_in = _stand_ins.get(name)
    if stand_in is not None:
        return stand_in
    directory = _declared().get(name)
    if directory is None:
        raise LookupError(
            f"no corpus named {name!r} is bound here; list it under "
            "containers.<key>.corpora in app.yml (a runtime: osprey container "
            "is handed the corpora it lists, and nothing else is)"
        )
    known = _corpora.get(name)
    if known is None or known.path != directory:
        known = _corpora[name] = Corpus(name, directory, _sidecar_search(name, directory))
    return known


def _declared() -> dict[str, Path]:
    """The bound corpora, from :data:`CORPORA_ENV_VAR`."""
    declared: dict[str, Path] = {}
    for entry in os.environ.get(CORPORA_ENV_VAR, "").split(":"):
        name, separator, directory = entry.partition("=")
        if separator and name and directory:
            declared[name] = Path(directory)
    return declared


def _sidecar_search(name: str, directory: Path) -> Callable[[str, int], list[CorpusHit]]:
    """The search of *name*'s sidecar, dialled at the URL the render handed over."""
    from osprey.services.qmd.client import QMDClient, QMDClientError

    env = corpus_url_env(name)
    url = os.environ.get(env, "").strip().rstrip("/")
    client = QMDClient(QMDServiceConfig(dial_url=url), client_name="osprey-app") if url else None

    def search(query: str, limit: int) -> list[CorpusHit]:
        if client is None:
            raise CorpusUnavailable(f"corpus {name!r} has no sidecar address ({env} is unset)")
        try:
            results = client.query(name, query, limit=limit)
        except QMDClientError as exc:
            raise CorpusUnavailable(f"corpus {name!r} cannot be searched: {exc}") from exc
        on_disk = _files_by_reported_name(directory)
        return [
            CorpusHit(
                path=_path_on_disk(result.file, directory, on_disk),
                title=result.title,
                score=result.score,
                line=result.line,
                snippet=result.snippet,
            )
            for result in results
        ]

    return search


_NOT_KEPT = re.compile(r"[^a-z0-9]+")


def _reported_name(relative: str) -> str:
    """What survives the sidecar's path normalisation, component by component."""
    return "/".join(
        _NOT_KEPT.sub("-", part.casefold()).strip("-") for part in PurePosixPath(relative).parts
    )


def _files_by_reported_name(directory: Path) -> Callable[[], dict[str, list[str]]]:
    """The corpus's files grouped by their normalised name, swept at most once."""
    found: dict[str, list[str]] = {}
    swept = False

    def index() -> dict[str, list[str]]:
        nonlocal swept
        if not swept:
            swept = True
            for path in directory.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(directory).as_posix()
                    found.setdefault(_reported_name(relative), []).append(relative)
        return found

    return index


def _path_on_disk(reported: str, directory: Path, index: Callable[[], dict[str, list[str]]]) -> str:
    """The file a reported hit names, or the reported spelling when none is found."""
    relative = unquote(reported).lstrip("/")
    if (directory / relative).is_file():
        return relative
    candidates = index().get(_reported_name(relative), [])
    return candidates[0] if len(candidates) == 1 else relative


@contextmanager
def stand_in(
    name: str, path: Path, hits: Sequence[CorpusHit] = (), *, unavailable: bool = False
) -> Iterator[Corpus]:
    """Answer :func:`corpus` for *name* from *path* and *hits*, with no sidecar.

    The seam :func:`osprey.apps.sdk.testing.corpus_stand_in` documents.
    """

    def search(_query: str, limit: int) -> list[CorpusHit]:
        if unavailable:
            raise CorpusUnavailable(f"corpus {name!r} cannot be searched: stand-in is unavailable")
        return list(hits[:limit])

    previous = _stand_ins.get(name)
    _stand_ins[name] = Corpus(name, Path(path), search)
    try:
        yield _stand_ins[name]
    finally:
        if previous is None:
            del _stand_ins[name]
        else:
            _stand_ins[name] = previous
