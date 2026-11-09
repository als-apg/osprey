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

import shutil
from pathlib import Path

import pytest

from tests.deployment._qmd_entrypoint import run_entrypoint

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


#: The heartbeat is a background sleeper that would hold the captured output
#: open long after the pass returns; these tests are about which qmd commands
#: run, so it is stubbed out.
_NO_HEARTBEAT = "start_heartbeat() { :; }; stop_heartbeat() { :; }; "


def _run(
    tmp_path: Path,
    script: str,
    *,
    embed: str | None,
    docs: int = 5,
    vectors: int = 0,
    pending: int = 5,
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
    result = run_entrypoint(tmp_path, stub=_STUB, script=_NO_HEARTBEAT + script, env=env)
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
