"""The qmd entrypoint's checks on a prebuilt index, run against a stub ``qmd``.

A prebuilt index was built elsewhere and is served as it arrived, so the
entrypoint makes on it the checks a build would have made: the vectors must
come from the embedder this image queries with, the index must hold the
collection clients ask for, and it must not be empty.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the image has")

EMBEDDER = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf@sha256:b5ce"

_STUB = """#!/bin/sh
case "$1" in
    collection)
        echo "$STUB_COLLECTION (qmd://$STUB_COLLECTION/)"
        echo "  Files:    $STUB_DOCS"
        ;;
    status)
        echo "  Total:    $STUB_DOCS files indexed"
        echo "  Vectors:  $STUB_DOCS embedded"
        ;;
esac
exit 0
"""


def _check(tmp_path: Path, *, stamp: str | None, collection: str = "papers", docs: int = 5):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "qmd").write_text(_STUB)
    (bin_dir / "qmd").chmod(0o755)
    state = tmp_path / "state"
    (state / ".qmd").mkdir(parents=True)
    (state / ".qmd" / "index.sqlite").write_bytes(b"")
    (state / ".qmd" / "index.yml").write_text("collections: {}\n")
    if stamp is not None:
        (state / ".qmd" / "osprey-embed-identity").write_text(stamp + "\n")

    lines = ENTRYPOINT.read_text().splitlines()
    assert lines[-1] == 'main "$@"', "the entrypoint no longer ends with its main call"
    library = tmp_path / "entrypoint-lib.sh"
    library.write_text("\n".join(lines[:-1]) + "\n")

    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "OSPREY_QMD_PORT": "1",
        "OSPREY_QMD_STATE_DIR": str(state),
        "OSPREY_QMD_INDEX_MODE": "prebuilt",
        "OSPREY_QMD_CORPUS": "papers",
        "OSPREY_QMD_EMBED_MODEL_ID": EMBEDDER,
        "STUB_COLLECTION": collection,
        "STUB_DOCS": str(docs),
    }
    script = f'. "{library}"; prepare_state; check_prebuilt_index; echo CHECKED'
    return subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def test_a_prebuilt_index_from_this_embedder_is_served(tmp_path):
    result = _check(tmp_path, stamp=EMBEDDER)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKED" in result.stdout


def test_a_prebuilt_index_keeps_its_own_config(tmp_path):
    result = _check(tmp_path, stamp=EMBEDDER)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "state" / ".qmd" / "index.yml").read_text() == "collections: {}\n"


def test_an_index_from_another_embedder_is_refused(tmp_path):
    result = _check(tmp_path, stamp="hf:other/embedder@sha256:0000")
    assert result.returncode != 0
    assert "built with embedder 'hf:other/embedder@sha256:0000'" in result.stdout


def test_an_index_without_the_corpus_collection_is_refused(tmp_path):
    result = _check(tmp_path, stamp=EMBEDDER, collection="library")
    assert result.returncode != 0
    assert "has no collection 'papers'" in result.stdout
    assert "library" in result.stdout


def test_an_empty_prebuilt_index_is_refused(tmp_path):
    result = _check(tmp_path, stamp=EMBEDDER, docs=0)
    assert result.returncode != 0
    assert "index is empty" in result.stdout


def test_an_index_without_an_identity_stamp_is_served_with_a_warning(tmp_path):
    result = _check(tmp_path, stamp=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "records no embedder identity" in result.stdout
