"""Build the channel search index that a staged Turtle corpus implies.

The graph paradigm's channel roster reads a DuckDB search index, not the Turtle
corpus it was derived from. A build writes that index; a test that stages a
corpus by hand and then expects channels has to write one too, or the roster
answers with the absence that says the index is not there.

``osprey build`` is covered already -- ``tests/conftest.py`` stubs the build's
own derivation step so the suite's many renders share one parse per corpus.
This module is the other half: the tests that stage a corpus WITHOUT running
that hook, and so have to ask for the index explicitly.

Two rules keep the helper honest about where the index goes:

* with a project config, the path comes from
  :func:`~osprey.deployment.graphdb_service.resolve_graph_index_path` -- the one
  resolver the build, the roster, the app and the health row all use, so a test
  cannot accidentally pin a path rule of its own;
* without one, it is the default ``index_path`` under the render holding the
  corpus, which is the same answer that resolver gives a config with no
  ``index_path`` key.

Parsing a corpus is the expensive half of a build (about a second and a half
for the corpus the control-assistant preset ships), so each distinct corpus is
built once per process and copied into place afterwards, keyed on its bytes and
its filename -- the two things the index records about the corpus it came from.

Import-light on purpose: nothing of osprey is imported at module import, so a
test module importing this one pays for it only when it builds.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: ``services.graphdb.index_path``'s default, relative to the render that holds
#: ``config.yml`` -- split so a test can join it to a render root rather than
#: spell the string a second time.
DEFAULT_INDEX_RELPATH = Path("data") / "channel_databases" / "graph.duckdb"

#: Corpus key -> the index built from it, inside :func:`_cache_dir`.
_BUILT: dict[tuple[str, str], Path] = {}

#: Where those live; created on the first build this process performs.
_CACHE_DIR: Path | None = None


def default_index_path(render: Path | str) -> Path:
    """Where a build writes the index of the project rendered into *render*."""
    return Path(render) / DEFAULT_INDEX_RELPATH


def build_index_from_ttl(
    ttl_path: Path | str,
    config: Mapping[str, Any] | None = None,
    *,
    config_dir: Path | str | None = None,
    index_path: Path | str | None = None,
) -> Path:
    """Build the search index for the corpus at *ttl_path* and return its path.

    Args:
        ttl_path: The staged Turtle corpus to derive the index from.
        config: The project config the roster will be asked about, when there is
            one. Its ``services.graphdb.index_path`` decides where the index
            goes, resolved against the config's own ``config_dir``.
        config_dir: The directory holding ``config.yml``, for a config that does
            not record one. Ignored when *config* is omitted.
        index_path: An explicit destination, for a test that names the index
            itself rather than deriving it from a project.

    Returns:
        The absolute path of the index that now exists on disk.
    """
    corpus = Path(ttl_path)
    target = _target_path(corpus, config, config_dir, index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_cached_index(corpus), target)
    return target


@contextmanager
def demo_corpus_path() -> Iterator[Path]:
    """Yield the path of the Turtle corpus OSPREY ships with its demo preset.

    A packaged resource, so it is reached through
    :func:`importlib.resources.as_file` and is only guaranteed to exist for the
    duration of the ``with`` block.
    """
    from importlib.resources import as_file, files

    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


def build_demo_index(index_path: Path | str) -> Path:
    """Build the shipped demo corpus's index at *index_path* and return it.

    The oracle behind every "what the corpus OSPREY ships holds" assertion:
    2908 channels, 396 of them settable.
    """
    with demo_corpus_path() as corpus:
        return build_index_from_ttl(corpus, index_path=index_path)


def _target_path(
    corpus: Path,
    config: Mapping[str, Any] | None,
    config_dir: Path | str | None,
    index_path: Path | str | None,
) -> Path:
    """Where the index for *corpus* belongs, by the first rule that applies."""
    if index_path is not None:
        return Path(index_path)
    if config is not None:
        from osprey.deployment.graphdb_service import resolve_graph_index_path

        anchor = config_dir if config_dir is not None else config.get("config_dir")
        return resolve_graph_index_path(config, Path(anchor) if anchor else None)
    return default_index_path(_render_root(corpus))


def _render_root(corpus: Path) -> Path:
    """The render a corpus staged at *corpus* belongs to.

    A project's corpus is normally under its ``data/`` directory, which is also
    where the index goes -- so ``data`` is stepped over rather than nested into.
    A corpus staged at the render root anchors there.
    """
    parent = corpus.parent
    return parent.parent if parent.name == "data" else parent


def _cached_index(corpus: Path) -> Path:
    """The index for *corpus*, building it if this process has not yet.

    Keyed on the corpus BYTES and its filename, because those are what the
    index records about where it came from: two tests staging identical text
    under one name share a build, and one that renames the corpus does not
    inherit an index that names the other file.
    """
    payload = corpus.read_bytes()
    key = (hashlib.sha256(payload).hexdigest(), corpus.name)

    cached = _BUILT.get(key)
    if cached is None or not cached.exists():
        from osprey.services.channel_finder.graph_index import build_graph_index

        cached = _cache_dir() / f"{key[0]}-{corpus.name}.duckdb"
        build_graph_index(corpus, cached)
        _BUILT[key] = cached
    return cached


def _cache_dir() -> Path:
    """This process's directory of built indexes, created on first use.

    Outside pytest's ``tmp_path`` tree because it outlives every test that
    fills it, so it is removed when the process ends instead.
    """
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="osprey-graph-index-"))
        atexit.register(shutil.rmtree, _CACHE_DIR, True)
    return _CACHE_DIR
