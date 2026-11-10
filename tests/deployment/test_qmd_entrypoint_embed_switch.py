"""The qmd entrypoint's per-corpus embed switch, run against a stub ``qmd``.

``OSPREY_QMD_EMBED=0`` marks a corpus whose clients query BM25 alone. Its
sidecar builds the keyword index and never runs ``qmd embed``, at startup or on
an update, and a full build of it serves once the index holds documents: a
keyword-only index has no vectors by design. Unset, the sidecar embeds as
before.

The tests source the shipped entrypoint (minus its final ``main`` call) under
``sh`` with a stub ``qmd`` first on PATH that logs each subcommand it is asked
to run.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from tests.deployment._qmd_entrypoint import run_entrypoint, run_entrypoint_for_leftovers

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the image has")

EMBEDDER = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf@sha256:b5ce"

#: A stub qmd: logs each subcommand to $STUB_CALLS; ``status`` reports
#: $STUB_DOCS documents, $STUB_VECTORS vectors and $STUB_PENDING pending, and
#: ``collection list`` the one collection holding $STUB_DOCS files.
_STUB = """#!/bin/sh
echo "$1" >> "$STUB_CALLS"
case "$1" in
    collection)
        echo "notes (qmd://notes/)"
        echo "  Files:    $STUB_DOCS"
        ;;
    status)
        echo "Documents"
        echo "  Total:    $STUB_DOCS files indexed"
        echo "  Vectors:  $STUB_VECTORS embedded"
        if [ "$STUB_PENDING" -gt 0 ]; then
            echo "  Pending:  $STUB_PENDING need embedding (run 'qmd embed')"
        fi
        ;;
esac
exit 0
"""


def _run(
    tmp_path: Path,
    script: str,
    *,
    embed: str | None,
    docs: int = 5,
    vectors: int = 0,
    pending: int = 5,
    stub: str = _STUB,
):
    state = tmp_path / "state"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    calls = tmp_path / "calls"
    calls.write_text("")
    env = {
        "OSPREY_QMD_STATE_DIR": str(state),
        "OSPREY_QMD_CORPUS": "notes",
        "OSPREY_QMD_CORPUS_PATH": str(corpus),
        "OSPREY_QMD_EMBED_MODEL_ID": EMBEDDER,
        "STUB_CALLS": str(calls),
        "STUB_DOCS": str(docs),
        "STUB_VECTORS": str(vectors),
        "STUB_PENDING": str(pending),
    }
    if embed is not None:
        env["OSPREY_QMD_EMBED"] = embed
    result = run_entrypoint(tmp_path, stub=stub, script=script, env=env)
    return result, calls.read_text().split()


def test_embedding_off_never_calls_qmd_embed_at_startup(tmp_path):
    result, calls = _run(tmp_path, "prepare_state; run_startup_pass; echo PASSED", embed="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout
    assert "update" in calls
    assert "embed" not in calls
    assert "embedding off (OSPREY_QMD_EMBED=0): keyword index only" in result.stdout
    assert "every embedding recomputed" not in result.stdout


def test_embedding_off_never_calls_qmd_embed_on_update(tmp_path):
    result, calls = _run(tmp_path, 'run_update test; echo "PENDING=$EMBED_PENDING"', embed="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "update" in calls
    assert "embed" not in calls
    assert "PENDING=5" in result.stdout


@pytest.mark.parametrize("embed", [None, "1"], ids=["unset", "on"])
def test_embedding_on_by_default_still_embeds(tmp_path, embed):
    result, calls = _run(
        tmp_path, "prepare_state; run_startup_pass; run_update test", embed=embed, vectors=5
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.count("embed") >= 2
    assert "every embedding recomputed" in result.stdout


_GATE = "prepare_state; BUILD_MODE=full; assert_index_populated; echo SERVED"


def test_embedding_off_full_build_serves_a_keyword_only_index(tmp_path):
    result, _ = _run(tmp_path, _GATE, embed="0", docs=5, vectors=0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SERVED" in result.stdout


def test_embedding_off_empty_index_is_still_refused(tmp_path):
    result, _ = _run(tmp_path, _GATE, embed="0", docs=0, vectors=0, pending=0)

    assert result.returncode != 0
    assert "index is empty" in result.stdout + result.stderr
    assert "SERVED" not in result.stdout


def test_embedding_on_full_build_with_no_vectors_is_still_refused(tmp_path):
    result, _ = _run(tmp_path, _GATE, embed=None, docs=5, vectors=0)

    assert result.returncode != 0
    assert "0 embeddings" in result.stdout + result.stderr
    assert "SERVED" not in result.stdout


def test_stop_heartbeat_leaves_no_process_behind(tmp_path):
    # The stop comes once the heartbeat's sleep is running. A sleeper left
    # behind holds the captured pipes open past the run's timeout, so a run
    # that returns at all proves the heartbeat took its sleep along.
    script = (
        "start_heartbeat demo; "
        'until pgrep -x -P "$HEARTBEAT_PID" sleep >/dev/null; do :; done; '
        "stop_heartbeat; echo STOPPED"
    )
    result, _ = _run(tmp_path, script, embed=None)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "STOPPED" in result.stdout


@pytest.mark.parametrize("shell", ["dash"])
def test_a_stop_right_after_the_start_leaves_no_process_behind(tmp_path, shell):
    # A stage that finishes at once stops the heartbeat a moment after its
    # start, while its subshell may still be forking or between forking the
    # sleep and saving its pid. Two hundred such stops in a row land the signal
    # in those windows many times over. Every process the run starts stays in
    # its process group, so whatever is still alive in it once the shell has
    # exited is a process the stops left behind. dash is the image's sh, and
    # the one shell the entrypoint's signal handling is written against.
    if shutil.which(shell) is None:
        pytest.skip(f"needs {shell}")
    script = (
        "i=0; while [ $i -lt 200 ]; do start_heartbeat demo; stop_heartbeat; "
        "i=$((i + 1)); done; echo STOPPED"
    )
    returncode, output, leftovers = run_entrypoint_for_leftovers(
        tmp_path, shell=shell, stub=_STUB, script=script, env={}
    )

    assert returncode == 0, f"the stops did not return: {output} {leftovers}"
    assert "STOPPED" in output
    assert leftovers == []


#: A stub qmd whose vectors live in the index file, as qmd's do: ``update``
#: creates ``index.sqlite`` holding 0 vectors when it is absent, ``embed``
#: writes $STUB_DOCS into it, and ``status`` reports what it holds. Deleting the
#: index file drops the vectors with it.
_DB_STUB = """#!/bin/sh
echo "$1" >> "$STUB_CALLS"
db="$OSPREY_QMD_STATE_DIR/.qmd/index.sqlite"
case "$1" in
    update)
        mkdir -p "$(dirname "$db")"
        [ -f "$db" ] || echo 0 > "$db"
        ;;
    embed)
        echo "$STUB_DOCS" > "$db"
        ;;
    collection)
        echo "notes (qmd://notes/)"
        echo "  Files:    $STUB_DOCS"
        ;;
    status)
        vectors=$(cat "$db" 2>/dev/null || echo 0)
        echo "Documents"
        echo "  Total:    $STUB_DOCS files indexed"
        echo "  Vectors:  $vectors embedded"
        pending=$((STUB_DOCS - vectors))
        if [ "$pending" -gt 0 ]; then
            echo "  Pending:  $pending need embedding (run 'qmd embed')"
        fi
        ;;
esac
exit 0
"""


def _switch(tmp_path: Path, *, first: str, second: str):
    """Run a startup pass with embed *first*, then one with embed *second*.

    Returns the run, the qmd calls of the second pass, and the vectors the
    index holds at the end.
    """
    script = (
        f"prepare_state; EMBED={first}; run_startup_pass; "
        'echo PASS2 >> "$STUB_CALLS"; '
        f"EMBED={second}; run_startup_pass; qmd status"
    )
    result, calls = _run(tmp_path, script, embed=first, stub=_DB_STUB)
    second_calls = calls[calls.index("PASS2") + 1 :]
    vectors = re.search(r"Vectors:\s+(\d+)", result.stdout.rsplit("Documents", 1)[-1])
    assert vectors is not None, result.stdout + result.stderr
    return result, second_calls, int(vectors.group(1))


def test_embedding_off_then_on_embeds_the_pending_set_without_a_rebuild(tmp_path):
    result, second_calls, vectors = _switch(tmp_path, first="0", second="1")

    assert result.returncode == 0, result.stdout + result.stderr
    second_pass = result.stdout.split("startup pass finished", 1)[1]
    assert "incremental startup pass" in second_pass
    assert "FULL REINDEX" not in second_pass
    assert "embed" in second_calls
    assert vectors == 5


def test_embedding_on_then_off_rebuilds_the_keyword_index_without_vectors(tmp_path):
    result, second_calls, vectors = _switch(tmp_path, first="1", second="0")

    assert result.returncode == 0, result.stdout + result.stderr
    second_pass = result.stdout.split("startup pass finished", 1)[1]
    assert "FULL REINDEX required: embedding turned off" in second_pass
    assert "embed" not in second_calls
    assert vectors == 0


def _prose(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_a_vector_query_after_embedding_is_turned_off_returns_nothing(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    assert "a vector query on that corpus returns nothing" in _prose(
        repo / "src/osprey/deployment/qmd_service.py"
    )
    assert "a vector search on that corpus returns nothing" in _prose(
        repo / "docs/source/how-to/ariel/search-sidecar.rst"
    )

    result, _, vectors = _switch(tmp_path, first="1", second="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert vectors == 0


def test_a_stamp_without_an_embed_mode_counts_as_embedded(tmp_path):
    script = (
        "prepare_state; "
        'mkdir -p "$(dirname "$DB_FILE")"; : > "$DB_FILE"; '
        f'echo "{EMBEDDER}" > "$IDENTITY_STAMP"; '
        'decide_build_mode; echo "MODE=$BUILD_MODE"'
    )
    result, _ = _run(tmp_path, script, embed="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODE=full" in result.stdout
